"""
pull_campus_graph.py

Downloads the Northeastern walking network from OpenStreetMap, cleans it,
and saves it to data/processed/.

Usage:
    python pull_campus_graph.py
    python pull_campus_graph.py --dist 900
"""

import argparse
import math
import os
from collections import Counter

import networkx as nx

# Configuration

# Center of the area to download (lat, lon)
CAMPUS_CENTER = (42.3388, -71.0890)
DEFAULT_DIST_M = 700
PLACE_QUERY = "Northeastern University, Boston, Massachusetts, USA"

OUT_DIR = os.path.join("data", "processed")

# Extra OSM tags to keep when downloading
EXTRA_WAY_TAGS = [
    "footway", "crossing", "crossing:signals", "tactile_paving", "wheelchair",
    "incline", "ramp", "handrail", "step_count", "surface", "smoothness",
    "covered", "indoor", "level", "tunnel", "bridge", "lit", "width",
    "opening_hours", "access", "foot", "layer",
]

# Keep edges separate where these tags change (so stairs stay their own edge)
ATTRS_THAT_SPLIT_EDGES = ["highway", "footway", "crossing", "indoor",
                          "covered", "tunnel", "level", "layer", "wheelchair"]

# Queries for tunnels and corridors (the walk network leaves some of these out)
TUNNEL_FILTERS = [
    '["highway"]["tunnel"~"yes|building_passage"]',
    '["highway"="corridor"]',
    '["highway"]["layer"~"^-"]',
    '["highway"]["level"~"^-"]',
]

# Columns to fill in during fieldwork
ANNOTATION_FIELDS = ["has_ramp", "step_free", "open_area",
                     "wall_or_edge_guide", "tactile_paving_checked",
                     "open_hours", "annotated_by", "notes"]


# Download

def download_graph(mode="point", dist=DEFAULT_DIST_M):
    """Download the walking network."""
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.useful_tags_way = list(
        dict.fromkeys(ox.settings.useful_tags_way + EXTRA_WAY_TAGS))

    kwargs = dict(network_type="walk", simplify=False, retain_all=True)
    if mode == "place":
        try:
            return ox.graph_from_place(PLACE_QUERY, **kwargs)
        except Exception as err:
            print(f"Place query failed ({err}); falling back to point mode.")
    return ox.graph_from_point(CAMPUS_CENTER, dist=dist, dist_type="bbox",
                               **kwargs)


def download_tunnels(mode="point", dist=DEFAULT_DIST_M):
    """Download tunnels and corridors. Returns None if there are none."""
    import osmnx as ox

    G_tun = None
    for osm_filter in TUNNEL_FILTERS:
        kwargs = dict(custom_filter=osm_filter, simplify=False, retain_all=True)
        try:
            if mode == "place":
                G_part = ox.graph_from_place(PLACE_QUERY, **kwargs)
            else:
                G_part = ox.graph_from_point(CAMPUS_CENTER, dist=dist,
                                             dist_type="bbox", **kwargs)
        except Exception as err:
            print(f"No ways for {osm_filter} ({type(err).__name__})")
            continue
        print(f"{osm_filter}: {G_part.number_of_edges()} edges")
        G_tun = G_part if G_tun is None else nx.compose(G_tun, G_part)
    return G_tun


def merge_tunnels(G_walk, G_tun):
    """Add the tunnels to the walk graph."""
    if G_tun is None:
        return G_walk
    G = nx.compose(G_walk, G_tun)
    # Make sure tunnels can be walked both ways
    for u, v, k, d in list(G_tun.edges(keys=True, data=True)):
        if not G.has_edge(v, u):
            G.add_edge(v, u, **d)
    return G


def is_tunnel_edge(d):
    return (_any_equals(d.get("tunnel"), "yes", "building_passage")
            or any(str(x).startswith("-") for x in _as_list(d.get("layer")))
            or any(str(x).startswith("-") for x in _as_list(d.get("level"))))


# Clean

def clean_graph(G_raw):
    """Simplify the graph and keep only the connected part."""
    import osmnx as ox

    G = ox.simplify_graph(G_raw, edge_attrs_differ=ATTRS_THAT_SPLIT_EDGES)
    # Save tunnels that are about to be removed
    G.graph["orphan_tunnels"] = find_orphan_tunnels(G)
    G = ox.truncate.largest_component(G, strongly=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G


def find_orphan_tunnels(G):
    """Tunnels that are not connected to the main graph."""
    largest = max(nx.strongly_connected_components(G), key=len)
    orphans = []
    for u, v, d in G.edges(data=True):
        if u not in largest and is_tunnel_edge(d) and u < v:
            orphans.append({"u": u, "v": v, "name": _first_or(d.get("name"), ""),
                            "lat": G.nodes[u]["y"], "lon": G.nodes[u]["x"]})
    return orphans


# Edge attributes

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
    """Compass direction from point 1 to point 2, in degrees."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two lat/lon points."""
    r = 6_371_009
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _edge_coords(G, u, v, data):
    """Points along an edge from u to v."""
    geom = data.get("geometry")
    if geom is not None:
        coords = list(geom.coords)
        start = (G.nodes[u]["x"], G.nodes[u]["y"])
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
    """Add clean fields to every edge."""
    for u, v, k, d in G.edges(keys=True, data=True):
        coords = _edge_coords(G, u, v, d)
        d["edge_id"] = f"{u}_{v}_{k}"
        # Same id for both directions
        a, b = sorted((u, v))
        d["segment_id"] = f"{a}_{b}_{k}"
        d["length_m"] = float(d.get("length", 0.0))
        d["is_steps"] = _any_equals(d.get("highway"), "steps")
        d["is_indoor"] = (_any_equals(d.get("indoor"), "yes", "corridor", "room")
                          or _any_equals(d.get("highway"), "corridor"))
        d["is_tunnel"] = is_tunnel_edge(d)
        d["is_covered"] = d["is_indoor"] or d["is_tunnel"] or _any_equals(
            d.get("covered"), "yes", "arcade", "colonnade") or _any_equals(
            d.get("tunnel"), "yes", "building_passage")
        d["crossing_type"] = _crossing_type(d)
        d["tactile_paving"] = _first_or(d.get("tactile_paving"))
        d["wheelchair"] = _first_or(d.get("wheelchair"))
        d["surface"] = _first_or(d.get("surface"))
        d["incline"] = _first_or(d.get("incline"))
        d["name"] = _first_or(d.get("name"), default="")
        d["highway"] = _first_or(d.get("highway"))
        # Direction at the start and end of the edge
        (x0, y0), (x1, y1) = coords[0], coords[1]
        (xa, ya), (xb, yb) = coords[-2], coords[-1]
        d["bearing_start"] = round(compass_bearing(x0, y0, x1, y1), 1)
        d["bearing_end"] = round(compass_bearing(xa, ya, xb, yb), 1)
        for field in ANNOTATION_FIELDS:
            d.setdefault(field, "unknown" if field not in ("annotated_by", "notes") else "")
    return G


def add_projected_coords(G):
    """Add x/y positions in metres to each node."""
    import osmnx as ox

    G_proj = ox.project_graph(G)
    for n, d in G_proj.nodes(data=True):
        G.nodes[n]["x_m"] = round(d["x"], 2)
        G.nodes[n]["y_m"] = round(d["y"], 2)
    return G


# Validate

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
    for flag in ["is_steps", "is_indoor", "is_tunnel", "is_covered"]:
        c = sum(1 for _, _, d in G.edges(data=True) if d[flag])
        lines.append(f"  {flag:<15} {c} edges")
    lines.append("")
    lines.append("Crossings by type:")
    for ct, c in Counter(d["crossing_type"] for _, _, d in G.edges(data=True)).most_common():
        lines.append(f"  {ct:<20} {c}")
    orphans = G.graph.get("orphan_tunnels", [])
    lines.append("")
    lines.append(f"Tunnel segments dropped (not connected to surface): {len(orphans)}")
    for o in orphans:
        lines.append(f"  {o['u']} -> {o['v']}  {o['name'] or '(unnamed)'}  "
                     f"near ({o['lat']:.5f}, {o['lon']:.5f})")
    if orphans:
        lines.append("  Connect these to their surface entrances by hand (see "
                     "add_portal_edge) and re-run.")
    return "\n".join(lines)


# Save / load

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


def add_portal_edge(G, surface_node, tunnel_node, length_m, is_steps,
                    card_hours="", public_hours=""):
    """Connect a tunnel to the surface. Call before clean_graph()."""
    for u, v in [(surface_node, tunnel_node), (tunnel_node, surface_node)]:
        G.add_edge(u, v, length=length_m, highway="steps" if is_steps else "footway",
                   tunnel="yes", card_hours=card_hours,
                   public_hours=public_hours, portal=True)


def _to_bool(s):
    return str(s).lower() == "true"


def load_campus_graph(path=os.path.join(OUT_DIR, "campus_walk.graphml")):
    """Load the saved graph."""
    import osmnx as ox

    edge_dtypes = {"length_m": float, "bearing_start": float, "bearing_end": float,
                   "is_steps": _to_bool, "is_indoor": _to_bool,
                   "is_tunnel": _to_bool, "is_covered": _to_bool}
    node_dtypes = {"x_m": float, "y_m": float}
    return ox.load_graphml(path, node_dtypes=node_dtypes, edge_dtypes=edge_dtypes)



def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["point", "place"], default="point")
    parser.add_argument("--dist", type=int, default=DEFAULT_DIST_M,
                        help="half-width of the bounding box in metres (point mode)")
    args = parser.parse_args()

    G_raw = download_graph(args.mode, args.dist)
    print(f"Raw walk graph: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    G_raw = merge_tunnels(G_raw, download_tunnels(args.mode, args.dist))
    print(f"With tunnels: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    # Connect tunnel entrances here, for example:
    # add_portal_edge(G_raw, surface_node=123, tunnel_node=456, length_m=8, is_steps=True)
    G = clean_graph(G_raw)
    G = standardise_attributes(G)
    G = add_projected_coords(G)
    report = validation_report(G)
    print(report)
    save_outputs(G, report)
    print(f"\nSaved everything to {OUT_DIR}/")


if __name__ == "__main__":
    main()