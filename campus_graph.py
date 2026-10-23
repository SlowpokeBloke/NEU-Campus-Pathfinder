"""
pull_campus_graph.py — Download and clean Northeastern's walking network.

Owner: Ronit (data ingestion: OSMnx pull + cleaning)
Requires: OSMnx 2.x  (pip install "osmnx>=2.0")

Run once (needs internet — OSMnx queries the free Overpass API, no key):
    python pull_campus_graph.py                      # point + radius (default)
    python pull_campus_graph.py --mode place         # campus polygon from OSM
    python pull_campus_graph.py --dist 900           # bigger area

Outputs (in data/processed/):
    campus_walk.graphml      cleaned graph — what the search algorithms load
    edges.geojson            edges for viewing in QGIS / geojson.io
    nodes.geojson            nodes for viewing
    segments_to_annotate.csv one row per physical segment, for Kunal's annotation
    graph_report.txt         validation summary (paste numbers into the report)
    campus_walk.png          quick plot

After the first run everything else should load the saved .graphml with
load_campus_graph(); never re-download during experiments, so results stay
reproducible even if OpenStreetMap changes mid-semester.
"""

import argparse
import math
import os
from collections import Counter

import networkx as nx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Approximate centre of the Boston campus (near Snell Library / Centennial
# Common). Check on a map and adjust so your core area is fully covered.
CAMPUS_CENTER = (42.3388, -71.0890)  # (lat, lon)
DEFAULT_DIST_M = 700
PLACE_QUERY = "Northeastern University, Boston, Massachusetts, USA"

OUT_DIR = os.path.join("data", "processed")

# OSM way tags we need. OSMnx drops tags not listed here, so this must be set
# BEFORE downloading.
EXTRA_WAY_TAGS = [
    "footway", "crossing", "crossing:signals", "tactile_paving", "wheelchair",
    "incline", "ramp", "handrail", "step_count", "surface", "smoothness",
    "covered", "indoor", "level", "tunnel", "bridge", "lit", "width",
    "opening_hours", "access", "foot",
]

# Simplification normally merges a footway and a staircase into one edge.
# Listing tags here keeps a node wherever these values change, so stairs,
# indoor corridors and crossings stay as their own edges.
ATTRS_THAT_SPLIT_EDGES = ["highway", "footway", "crossing", "indoor",
                          "covered", "tunnel", "level", "wheelchair"]

# Columns the team fills in by hand during fieldwork. "unknown" is the
# default so nothing is silently treated as accessible.
ANNOTATION_FIELDS = ["has_ramp", "has_elevator", "step_free", "open_area",
                     "wall_or_edge_guide", "tactile_paving_checked",
                     "open_hours", "annotated_by", "notes"]


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_graph(mode="point", dist=DEFAULT_DIST_M):
    """Download the raw, unsimplified walking graph from OpenStreetMap."""
    import osmnx as ox

    ox.settings.use_cache = True        # repeated runs hit the local cache
    ox.settings.log_console = True
    ox.settings.useful_tags_way = list(
        dict.fromkeys(ox.settings.useful_tags_way + EXTRA_WAY_TAGS))

    kwargs = dict(network_type="walk", simplify=False, retain_all=True)
    if mode == "place":
        try:
            return ox.graph_from_place(PLACE_QUERY, **kwargs)
        except Exception as err:  # geocoder may not return a polygon
            print(f"Place query failed ({err}); falling back to point mode.")
    return ox.graph_from_point(CAMPUS_CENTER, dist=dist, dist_type="bbox",
                               **kwargs)


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def clean_graph(G_raw):
    """Simplify, keep the largest connected piece, drop self-loops."""
    import osmnx as ox

    G = ox.simplify_graph(G_raw, edge_attrs_differ=ATTRS_THAT_SPLIT_EDGES)
    # Walk graphs are two-way, so the largest strongly connected component
    # guarantees every node can reach every other node.
    G = ox.truncate.largest_component(G, strongly=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G


# ---------------------------------------------------------------------------
# 3. Standardise attributes (pure networkx — unit-testable without OSMnx)
# ---------------------------------------------------------------------------

def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _first_or(value, default="unknown"):
    values = [v for v in _as_list(value) if v not in (None, "")]
    return str(values[0]) if values else default


def _any_equals(value, *targets):
    return any(str(v) in targets for v in _as_list(value))


def compass_bearing(lon1, lat1, lon2, lat2):
    """Initial compass bearing in degrees [0, 360) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. OSMnx edge lengths are great-circle
    distances along the edge geometry, so this is never larger than the true
    path length: an admissible A* heuristic for the distance cost."""
    r = 6_371_009  # mean Earth radius, same constant OSMnx uses
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _edge_coords(G, u, v, data):
    """(lon, lat) points along the edge, in travel direction u -> v."""
    geom = data.get("geometry")
    if geom is not None:
        coords = list(geom.coords)
        start = (G.nodes[u]["x"], G.nodes[u]["y"])
        # Make sure the geometry runs from u to v.
        if math.dist(coords[0], start) > math.dist(coords[-1], start):
            coords.reverse()
        return coords
    return [(G.nodes[u]["x"], G.nodes[u]["y"]),
            (G.nodes[v]["x"], G.nodes[v]["y"])]


def _crossing_type(data):
    if not (_any_equals(data.get("footway"), "crossing")
            or data.get("crossing") is not None):
        return "none"
    crossing = _first_or(data.get("crossing"))
    if crossing in ("traffic_signals",) or _any_equals(data.get("crossing:signals"), "yes"):
        return "signalized"
    if crossing in ("marked", "zebra", "uncontrolled"):
        return "marked_unsignalized"
    if crossing in ("unmarked", "no"):
        return "unmarked"
    return "unknown"


def standardise_attributes(G):
    """Add the derived, consistently typed fields every algorithm relies on."""
    for u, v, k, d in G.edges(keys=True, data=True):
        coords = _edge_coords(G, u, v, d)
        d["edge_id"] = f"{u}_{v}_{k}"
        # Same id for both directions of one physical segment, so it is only
        # annotated once.
        a, b = sorted((u, v))
        d["segment_id"] = f"{a}_{b}_{k}"
        d["length_m"] = float(d.get("length", 0.0))
        d["is_steps"] = _any_equals(d.get("highway"), "steps")
        d["is_indoor"] = (_any_equals(d.get("indoor"), "yes", "corridor", "room")
                          or _any_equals(d.get("highway"), "corridor"))
        d["is_covered"] = d["is_indoor"] or _any_equals(
            d.get("covered"), "yes", "arcade", "colonnade") or _any_equals(
            d.get("tunnel"), "yes", "building_passage")
        d["crossing_type"] = _crossing_type(d)
        d["tactile_paving"] = _first_or(d.get("tactile_paving"))
        d["wheelchair"] = _first_or(d.get("wheelchair"))
        d["surface"] = _first_or(d.get("surface"))
        d["incline"] = _first_or(d.get("incline"))
        d["name"] = _first_or(d.get("name"), default="")
        d["highway"] = _first_or(d.get("highway"))
        # Start and end headings, for counting turns in the low-vision profile.
        (x0, y0), (x1, y1) = coords[0], coords[1]
        (xa, ya), (xb, yb) = coords[-2], coords[-1]
        d["bearing_start"] = round(compass_bearing(x0, y0, x1, y1), 1)
        d["bearing_end"] = round(compass_bearing(xa, ya, xb, yb), 1)
        for field in ANNOTATION_FIELDS:
            d.setdefault(field, "unknown" if field not in ("annotated_by", "notes") else "")
    return G


def add_projected_coords(G):
    """Store metric x_m / y_m (UTM) on nodes for plotting and quick maths."""
    import osmnx as ox

    G_proj = ox.project_graph(G)
    for n, d in G_proj.nodes(data=True):
        G.nodes[n]["x_m"] = round(d["x"], 2)
        G.nodes[n]["y_m"] = round(d["y"], 2)
    return G


# ---------------------------------------------------------------------------
# 4. Validate
# ---------------------------------------------------------------------------

def validation_report(G):
    lengths = [d["length_m"] for _, _, d in G.edges(data=True)]
    und = G.to_undirected()
    lines = [
        "Campus walking graph — validation report",
        "=" * 42,
        f"Nodes: {G.number_of_nodes()}",
        f"Directed edges: {G.number_of_edges()}",
        f"Physical segments: {len({d['segment_id'] for _, _, d in G.edges(data=True)})}",
        f"Strongly connected: {nx.is_strongly_connected(G)}",
        f"Total length (km, one direction): {sum(lengths) / 2000:.2f}",
        f"Edge length min/median/max (m): {min(lengths):.1f} / "
        f"{sorted(lengths)[len(lengths) // 2]:.1f} / {max(lengths):.1f}",
        f"Edges shorter than 1 m: {sum(1 for x in lengths if x < 1)}",
        f"Dead-end nodes (degree 1): {sum(1 for n in und if und.degree(n) == 1)}",
        "",
        "Edges by highway type:",
    ]
    for hw, c in Counter(d["highway"] for _, _, d in G.edges(data=True)).most_common():
        lines.append(f"  {hw:<15} {c}")
    lines.append("")
    lines.append("Tag coverage (share of directed edges with a known value):")
    for field in ["tactile_paving", "wheelchair", "surface", "incline"]:
        known = sum(1 for _, _, d in G.edges(data=True) if d[field] != "unknown")
        lines.append(f"  {field:<15} {known / G.number_of_edges():.1%}")
    for flag in ["is_steps", "is_indoor", "is_covered"]:
        c = sum(1 for _, _, d in G.edges(data=True) if d[flag])
        lines.append(f"  {flag:<15} {c} edges")
    lines.append("")
    lines.append("Crossings by type:")
    for ct, c in Counter(d["crossing_type"] for _, _, d in G.edges(data=True)).most_common():
        lines.append(f"  {ct:<20} {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Save / load
# ---------------------------------------------------------------------------

def _stringify_lists(gdf):
    for col in gdf.columns:
        if col != "geometry" and gdf[col].map(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].map(lambda x: ";".join(map(str, x)) if isinstance(x, list) else x)
    return gdf


def save_outputs(G, report):
    import osmnx as ox

    os.makedirs(OUT_DIR, exist_ok=True)
    ox.save_graphml(G, os.path.join(OUT_DIR, "campus_walk.graphml"))

    nodes, edges = ox.graph_to_gdfs(G)
    _stringify_lists(nodes).to_file(os.path.join(OUT_DIR, "nodes.geojson"), driver="GeoJSON")
    edges = _stringify_lists(edges.reset_index())
    edges.to_file(os.path.join(OUT_DIR, "edges.geojson"), driver="GeoJSON")

    cols = ["segment_id", "name", "highway", "length_m", "is_steps", "is_indoor",
            "is_covered", "crossing_type", "tactile_paving", "wheelchair"] + ANNOTATION_FIELDS
    seg = edges.drop_duplicates("segment_id").copy()
    seg["mid_lat"] = seg.geometry.interpolate(0.5, normalized=True).y
    seg["mid_lon"] = seg.geometry.interpolate(0.5, normalized=True).x
    seg[cols + ["mid_lat", "mid_lon"]].to_csv(
        os.path.join(OUT_DIR, "segments_to_annotate.csv"), index=False)

    with open(os.path.join(OUT_DIR, "graph_report.txt"), "w") as f:
        f.write(report)

    ox.plot_graph(G, node_size=4, edge_linewidth=0.8, bgcolor="white",
                  node_color="#1F4E5A", edge_color="#999999", show=False,
                  close=True, save=True,
                  filepath=os.path.join(OUT_DIR, "campus_walk.png"), dpi=200)


def _to_bool(s):
    return str(s).lower() == "true"


def load_campus_graph(path=os.path.join(OUT_DIR, "campus_walk.graphml")):
    """Load the cleaned graph with correct types. Use this in every algorithm."""
    import osmnx as ox

    edge_dtypes = {"length_m": float, "bearing_start": float, "bearing_end": float,
                   "is_steps": _to_bool, "is_indoor": _to_bool, "is_covered": _to_bool}
    node_dtypes = {"x_m": float, "y_m": float}
    return ox.load_graphml(path, node_dtypes=node_dtypes, edge_dtypes=edge_dtypes)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["point", "place"], default="point")
    parser.add_argument("--dist", type=int, default=DEFAULT_DIST_M,
                        help="half-width of the bounding box in metres (point mode)")
    args = parser.parse_args()

    G_raw = download_graph(args.mode, args.dist)
    print(f"Raw graph: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    G = clean_graph(G_raw)
    G = standardise_attributes(G)
    G = add_projected_coords(G)
    report = validation_report(G)
    print(report)
    save_outputs(G, report)
    print(f"\nSaved everything to {OUT_DIR}/")


if __name__ == "__main__":
    main()