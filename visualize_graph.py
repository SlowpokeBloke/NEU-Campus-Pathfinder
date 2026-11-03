"""
visualize_graph.py

Makes a PNG of the campus graph and an interactive HTML map.

Usage:
    python visualize_graph.py
    python visualize_graph.py --graph path/to.graphml
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = os.path.join("data", "processed")

# Edge types and their colors (an edge gets the first type that matches)
CATEGORIES = [
    ("Closed", "#000000", lambda d: _true(d.get("is_closed"))),
    ("Stairs", "#D62728", lambda d: _true(d.get("is_steps"))),
    ("Tunnel", "#7B3FA0", lambda d: _true(d.get("is_tunnel"))),
    ("Indoor", "#9467BD", lambda d: _true(d.get("is_indoor"))),
    ("Signalized crossing", "#2CA02C", lambda d: d.get("crossing_type") == "signalized"),
    ("Unsignalized crossing", "#E5B000", lambda d: d.get("crossing_type") in ("marked_unsignalized", "unmarked", "unknown")),
    ("Covered", "#1F77B4", lambda d: _true(d.get("is_covered"))),
    ("Path", "#A0A0A0", lambda d: True),
]
COLOR = {name: color for name, color, _ in CATEGORIES}
ROUTE_COLOR = "#E0218A"


def _true(value):
    return value is True or str(value).lower() == "true"


def edge_category(d):
    for name, _, test in CATEGORIES:
        if test(d):
            return name
    return "Path"


def edge_lonlat(G, u, v, d):
    """Points along an edge."""
    geom = d.get("geometry")
    if geom is not None:
        return list(geom.coords)
    return [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]


def physical_segments(G):
    """Each path once, skipping the reverse direction."""
    seen = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        sid = d.get("segment_id", f"{min(u, v)}_{max(u, v)}_{k}")
        if sid in seen:
            continue
        seen.add(sid)
        yield u, v, k, d


def route_edges(G, route):
    """Edges used by a route."""
    out = []
    for u, v in zip(route[:-1], route[1:]):
        options = G.get_edge_data(u, v)
        if not options:
            raise ValueError(f"Route uses a missing edge {u} -> {v}")
        out.append((u, v, min(options.values(), key=lambda d: float(d.get("length_m", d.get("length", 0))))))
    return out


def _setup_axes(ax, G):
    # Keep the map from looking stretched
    lat = sum(d["y"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    ax.set_aspect(1 / math.cos(math.radians(lat)))
    ax.set_axis_off()


def _draw_network(ax, G, color_by_category=True, base_color="#D0D0D0", width=1.0):
    for u, v, k, d in physical_segments(G):
        xs, ys = zip(*edge_lonlat(G, u, v, d))
        if color_by_category:
            cat = edge_category(d)
            ax.plot(xs, ys, color=COLOR[cat], linewidth=width * (2 if cat != "Path" else 1),
                    linestyle="--" if cat == "Tunnel" else "-", zorder=2 if cat != "Path" else 1)
        else:
            ax.plot(xs, ys, color=base_color, linewidth=width, zorder=1)


# Static network plot

def plot_static(G, out_path=os.path.join(OUT_DIR, "network_by_type.png"), title=None):
    """Save a PNG of the graph with each edge colored by type."""
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_network(ax, G)
    _setup_axes(ax, G)
    present = {edge_category(d) for _, _, _, d in physical_segments(G)}
    handles = [Line2D([0], [0], color=COLOR[n], lw=3, linestyle="--" if n == "Tunnel" else "-", label=n)
               for n, _, _ in CATEGORIES if n in present]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_title(title or "Campus walking network by edge type", fontsize=13)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Route plots

def plot_search_comparison(G, results, out_path=os.path.join(OUT_DIR, "search_comparison.png"),
                           title=None, highlight=("Stairs",)):
    """Plot one or more search results side by side.

    results maps a name to a result from astar(), for example
    {"A* step-free": res}. Each result can also have "edges" and "blocked".
    """
    n = len(results)
    cols = n if n <= 3 else (2 if n == 4 else 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    for ax, (name, res) in zip(axes.flat, results.items()):
        _draw_network(ax, G, color_by_category=False)
        for u, v, k, d in physical_segments(G):
            cat = edge_category(d)
            if cat in highlight:
                hx, hy = zip(*edge_lonlat(G, u, v, d))
                ax.plot(hx, hy, color=COLOR[cat], linewidth=4, zorder=2)
        for u, v in res.get("blocked", ()):
            bx, by = zip(*edge_lonlat(G, u, v, {}))
            ax.plot(bx, by, color="black", linewidth=5, zorder=6)
            ax.scatter(sum(bx) / 2, sum(by) / 2, marker="X", s=220, color="black",
                       edgecolor="white", linewidth=1.5, zorder=7)
        explored = res.get("explored", [])
        if explored:
            xs = [G.nodes[x]["x"] for x in explored]
            ys = [G.nodes[x]["y"] for x in explored]
            ax.scatter(xs, ys, c=range(len(explored)), cmap="viridis", s=8, zorder=3)
        path = res.get("path") or []
        if len(path) > 1:
            used = ([(u, v, G.edges[u, v, k]) for u, v, k in res["edges"]]
                    if res.get("edges") else route_edges(G, path))
            for u, v, d in used:
                rx, ry = zip(*edge_lonlat(G, u, v, d))
                ax.plot(rx, ry, color=ROUTE_COLOR, linewidth=3.5, zorder=4)
            for node, marker, label in [(path[0], "o", "start"), (path[-1], "*", "goal")]:
                ax.scatter(G.nodes[node]["x"], G.nodes[node]["y"], marker=marker, s=160,
                           color="white", edgecolor="black", linewidth=1.5, zorder=5, label=label)
        subtitle = f"{len(explored)} nodes expanded"
        if res.get("cost") is not None:
            subtitle += f" · cost {res['cost']:.0f}"
        if not path:
            subtitle += " · NO ROUTE"
        ax.set_title(f"{name}\n{subtitle}", fontsize=11)
        _setup_axes(ax, G)
    present = {edge_category(d) for _, _, _, d in physical_segments(G)}
    handles = [Line2D([0], [0], color=COLOR[c], lw=4, label=c) for c in highlight if c in present]
    handles.append(Line2D([0], [0], color=ROUTE_COLOR, lw=3, label="Route"))
    if any(r.get("blocked") for r in results.values()):
        handles.append(Line2D([0], [0], color="black", marker="X", lw=0, markersize=10, label="Closed"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Interactive map

TOOLTIP_FIELDS = ["segment_id", "name", "highway", "length_m", "crossing_type",
                  "tactile_paving", "wheelchair", "is_steps", "is_tunnel", "is_covered",
                  "card_hours", "public_hours"]


def interactive_map(G, out_path=os.path.join(OUT_DIR, "campus_map.html"), routes=None):
    """Save an HTML map of the graph. routes is optional: {"name": path}."""
    import folium

    lat = sum(d["y"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    lon = sum(d["x"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles="OpenStreetMap",
                   max_zoom=20, control_scale=True)

    layers = {name: folium.FeatureGroup(name=name, show=True) for name, _, _ in CATEGORIES}
    for u, v, k, d in physical_segments(G):
        cat = edge_category(d)
        coords = [(lat_, lon_) for lon_, lat_ in edge_lonlat(G, u, v, d)]
        rows = "".join(f"<tr><td><b>{f}</b></td><td>{d.get(f)}</td></tr>"
                       for f in TOOLTIP_FIELDS if d.get(f) not in (None, "", "unknown"))
        folium.PolyLine(coords, color=COLOR[cat], weight=6 if cat != "Path" else 3,
                        opacity=0.85, dash_array="8 6" if cat == "Tunnel" else None,
                        tooltip=f"{cat}: {d.get('name') or d.get('segment_id', '')}",
                        popup=folium.Popup(f"<table>{rows}</table>", max_width=320),
                        ).add_to(layers[cat])
    for name, layer in layers.items():
        if layer._children:
            layer.add_to(m)

    route_colors = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4"]
    for i, (name, path) in enumerate((routes or {}).items()):
        group = folium.FeatureGroup(name=f"Route: {name}", show=True)
        pts = []
        for u, v, d in route_edges(G, path):
            seg = [(la, lo) for lo, la in edge_lonlat(G, u, v, d)]
            pts.extend(seg if not pts else seg[1:])
        folium.PolyLine(pts, color=route_colors[i % len(route_colors)], weight=7,
                        opacity=0.9, tooltip=f"Route: {name}").add_to(group)
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m.save(out_path)
    return out_path



def main():
    parser = argparse.ArgumentParser(description="Visualize the campus graph")
    parser.add_argument("--graph", default=os.path.join(OUT_DIR, "campus_walk.graphml"))
    args = parser.parse_args()

    from pull_campus_graph import load_campus_graph
    G = load_campus_graph(args.graph)
    print("Saved", plot_static(G))
    print("Saved", interactive_map(G))


if __name__ == "__main__":
    main()
