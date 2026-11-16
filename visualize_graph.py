"""
visualize_graph.py

Makes a PNG of the campus graph and an interactive HTML map, and has a
route finder that compares search algorithms and their heuristics.

Usage:
    python visualize_graph.py                      # PNG + HTML map of the campus
    python visualize_graph.py --route              # route finder, asks for start and goal
    python visualize_graph.py --route --map husky --start Dorm --goal Library --card 009876543 --hour 23
    python visualize_graph.py --route --map campus --start 200 --goal 219 --profile step_free
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = os.path.join("data", "processed")

# Edge types and their colors (an edge gets the first type that matches)
CATEGORIES = [
    ("Closed",                 "#000000", lambda d: _true(d.get("is_closed"))),
    ("Stairs",                 "#D62728", lambda d: _true(d.get("is_steps"))),
    ("Tunnel",                 "#7B3FA0", lambda d: _true(d.get("is_tunnel"))),
    ("Indoor",                 "#9467BD", lambda d: _true(d.get("is_indoor"))),
    ("Signalized crossing",    "#2CA02C", lambda d: d.get("crossing_type") == "signalized"),
    ("Unsignalized crossing",  "#E5B000", lambda d: d.get("crossing_type") in ("marked_unsignalized", "unmarked", "unknown")),
    ("Covered",                "#1F77B4", lambda d: _true(d.get("is_covered"))),
    ("Path",                   "#A0A0A0", lambda d: True),
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
    if G.graph.get("metric"):
        ax.set_aspect("equal")          # x/y already in metres
    else:
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
            ax.scatter(xs, ys, c=range(len(explored)), cmap="viridis",
                       s=60 if G.number_of_nodes() <= 30 else 8, zorder=5)
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
        if G.number_of_nodes() <= 30:   # small map: write the place names
            for node, d in G.nodes(data=True):
                ax.annotate(str(node), (d["x"], d["y"]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points", zorder=8)
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



# Route finder

def build_husky_graph():
    """The Husky Card demo map as a networkx graph, so it can be drawn."""
    import networkx as nx
    import husky_card_demo as hc

    G = nx.MultiDiGraph(metric=True)
    for name, (x, y) in hc.NODES.items():
        G.add_node(name, x=x, y=y)
    for u, v, length, door in hc.EDGES:
        building = door[0] if door else None
        seg = f"{min(u, v)}|{max(u, v)}"
        for a, b in [(u, v), (v, u)]:
            G.add_edge(a, b, length=length, length_m=length, door=building,
                       is_indoor=building is not None, segment_id=seg)
    return G


def setup_husky(args):
    """Graph, search inputs and a description for the Husky Card map."""
    import husky_card_demo as hc
    from algorithms import euclidean

    G = build_husky_graph()
    places = list(hc.NODES)
    start = pick_place("From", args.start, places)
    goal = pick_place("To", args.goal, places)
    card = args.card if args.card is not None else input(
        "Husky Card number (press Enter for none): ").strip() or None
    hour = args.hour if args.hour is not None else int(
        input("Hour (0-23): ").strip() or 12)

    locked = hc.locked_buildings(card, hour)
    blocked = {(u, v) for u, v, d in G.edges(data=True) if d["door"] in locked}
    info = (f"Husky map, {'card ' + card if card else 'no card'} at {hour}:00, "
            f"locked: {', '.join(sorted(locked)) or 'nothing'}")
    return dict(G=G, start=start, goal=goal, neighbors=hc.make_neighbors(locked),
                heuristic=euclidean(hc.NODES, goal), blocked=blocked,
                cost_fn=None, info=info)


def setup_campus(args):
    """Graph, search inputs and a description for the campus graph."""
    from algorithms import haversine, networkx_neighbors
    from profiles import make_cost_fn

    graph_file = os.path.join(OUT_DIR, "campus_walk.graphml")
    if os.path.exists(graph_file):
        from pull_campus_graph import load_campus_graph
        G = load_campus_graph(graph_file)
        default_start = default_goal = None
    else:
        from campus_search import test_grid
        G, default_start, default_goal = test_grid()
        print(f"{graph_file} not found, using the test grid (nodes 0-399, "
              f"try {default_start} to {default_goal})")

    start = pick_node("From", args.start, G, default_start)
    goal = pick_node("To", args.goal, G, default_goal)
    profile = args.profile or (input("Profile, fastest or step_free [step_free]: ")
                               .strip() or "step_free")
    cost_fn = make_cost_fn(profile)
    info = f"Campus graph, profile: {profile}"
    return dict(G=G, start=start, goal=goal, neighbors=networkx_neighbors(G, cost_fn),
                heuristic=haversine(G, goal), blocked=set(), cost_fn=cost_fn, info=info)


def pick_place(label, given, places):
    """Match a typed place name (case-insensitive, start of the name is enough)."""
    text = given
    while True:
        if text is None:
            print("Places: " + ", ".join(places))
            text = input(f"{label}: ").strip()
        matches = [p for p in places if p.lower() == text.lower()] or \
                  [p for p in places if p.lower().startswith(text.lower())]
        if len(matches) == 1:
            return matches[0]
        print(f"'{text}' matches {matches or 'nothing'}, try again.")
        text = None


def pick_node(label, given, G, default):
    """Read a node ID and check it exists in the graph."""
    text = given
    while True:
        if text is None:
            hint = f" [{default}]" if default is not None else ""
            text = input(f"{label} node ID{hint}: ").strip() or (
                str(default) if default is not None else "")
        try:
            node = int(text)
        except ValueError:
            node = text
        if node in G:
            return node
        print(f"Node {text} is not in the graph, try again (IDs are on campus_map.html).")
        text = None


def run_algorithms(start, goal, neighbors, heuristic):
    """Run every algorithm on the same start, goal and costs."""
    from algorithms import astar, bfs, weighted_astar

    return {
        "A*": astar(start, goal, neighbors, heuristic),
        "Weighted A* (w=1.5)": weighted_astar(start, goal, neighbors, heuristic, w=1.5),
        "Dijkstra (no heuristic)": astar(start, goal, neighbors, lambda n: 0.0),
        "BFS (fewest steps)": bfs(start, goal, neighbors),
    }


def trace_route(path, neighbors, heuristic):
    """Walk along a route and record g, h and the true remaining cost at each step."""
    g, rows = 0.0, []
    total = sum(dict(neighbors(u))[v] for u, v in zip(path[:-1], path[1:]))
    for i, node in enumerate(path):
        if i > 0:
            g += dict(neighbors(path[i - 1]))[node]
        rows.append({"node": node, "g": g, "h": heuristic(node), "remaining": total - g})
    return rows


def print_trace(rows, w=1.5):
    print(f"\n{'Step':<5}{'Place':<28}{'g walked':>10}{'h estimate':>12}"
          f"{'f = g+h':>10}{'true left':>11}  h <= true left?")
    for i, r in enumerate(rows):
        ok = "yes" if r["h"] <= r["remaining"] + 1e-6 else "NO"
        print(f"{i:<5}{str(r['node'])[:27]:<28}{r['g']:>10.0f}{r['h']:>12.0f}"
              f"{r['g'] + r['h']:>10.0f}{r['remaining']:>11.0f}  {ok}")


def recommend(results, rows):
    """Pick the algorithm to use and explain why, focusing on the heuristic."""
    found = {n: r for n, r in results.items() if r["found"]}
    if not found:
        return None, ["No algorithm found a route: every way to the goal is blocked."]
    best = min(r["cost"] for r in found.values())
    optimal = {n: r for n, r in found.items() if r["cost"] <= best + 0.5}
    pick = min(optimal, key=lambda n: (len(optimal[n]["explored"]), n != "A*"))

    lines = [f"Use {pick}: it found the shortest route ({best:.0f} m) "
             f"while checking the fewest places ({len(optimal[pick]['explored'])})."]
    a, d = results["A*"], results["Dijkstra (no heuristic)"]
    if a["found"] and d["explored"]:
        saved = 1 - len(a["explored"]) / len(d["explored"])
        if saved > 0:
            lines.append(f"The heuristic saved {saved:.0%} of the work: A* checked "
                         f"{len(a['explored'])} places, Dijkstra checked {len(d['explored'])}.")
        else:
            lines.append("Here the heuristic saved no work: A* and Dijkstra checked the "
                         "same number of places, because the blocked doors forced a detour.")
    wa = results["Weighted A* (w=1.5)"]
    if wa["found"] and wa["cost"] > best + 0.5:
        lines.append(f"Weighted A* checked {len(wa['explored'])} places but its route is "
                     f"{wa['cost'] - best:.0f} m longer, so the speed-up isn't worth it here.")
    bf = results["BFS (fewest steps)"]
    if bf["found"] and bf["cost"] > best + 0.5:
        lines.append(f"BFS ignores distance: its route is {bf['cost'] - best:.0f} m longer.")

    h0, true0 = rows[0]["h"], rows[0]["remaining"]
    quality = h0 / true0 if true0 else 1.0
    if quality >= 0.8:
        verdict = "close to the real distance, so A* heads almost straight for the goal."
    elif quality >= 0.5:
        verdict = "a fair guess: obstacles add some detour, so A* explores a bit more."
    else:
        verdict = ("weak: obstacles force a big detour, so A* behaves more like Dijkstra. "
                   "This is where Weighted A* helps most.")
    lines.append(f"At the start the straight-line estimate is {h0:.0f} m and the real "
                 f"distance is {true0:.0f} m ({quality:.0%}). The heuristic is {verdict}")
    if all(r["h"] <= r["remaining"] + 1e-6 for r in rows):
        lines.append("The heuristic never overestimated along the route (admissible), "
                     "so A* is guaranteed to find the shortest route.")
    return pick, lines


def plot_heuristic_trace(rows, out_path, w=1.5, title=None):
    """Line chart of h(n), w*h(n) and the true remaining cost along the route."""
    steps = range(len(rows))
    true_left = [r["remaining"] for r in rows]
    h = [r["h"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, true_left, "-o", color="black", label="True distance left")
    ax.plot(steps, h, "-o", color="#1F77B4", label="A* heuristic h(n)")
    ax.plot(steps, [w * x for x in h], "--", color="#E5B000", label=f"Weighted A* ({w} × h)")
    ax.fill_between(steps, h, true_left, color="#1F77B4", alpha=0.12,
                    label="Gap = how much the heuristic underestimates")
    ax.set_xticks(list(steps))
    ax.set_xticklabels([str(r["node"])[:12] for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("metres to goal")
    ax.set_title(title or "Heuristic vs real distance along the route")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def route_finder(args):
    """Ask for a start and goal, run every algorithm, trace the route and compare."""
    map_name = args.map or (input("Map, husky or campus [husky]: ").strip() or "husky")
    s = setup_husky(args) if map_name == "husky" else setup_campus(args)
    G, start, goal = s["G"], s["start"], s["goal"]

    print(f"\n{s['info']}\nRoute from {start} to {goal}\n")
    results = run_algorithms(start, goal, s["neighbors"], s["heuristic"])
    print(f"{'Algorithm':<26}{'cost (m)':>10}{'steps':>7}{'checked':>9}{'ms':>8}")
    for name, r in results.items():
        cost = f"{r['cost']:.0f}" if r["found"] else "no path"
        steps = len(r["path"]) - 1 if r["found"] else "-"
        print(f"{name:<26}{cost:>10}{steps:>7}{len(r['explored']):>9}"
              f"{r['runtime_s'] * 1000:>8.2f}")
        r["blocked"] = s["blocked"]
        if r["found"] and s["cost_fn"]:
            from algorithms import path_edges
            r["edges"] = path_edges(G, r["path"], s["cost_fn"])

    if not results["A*"]["found"]:
        print("\nNo route: every way to the goal is blocked.")
        return

    rows = trace_route(results["A*"]["path"], s["neighbors"], s["heuristic"])
    print("\nTracing the A* route:")
    print_trace(rows)
    pick, lines = recommend(results, rows)
    print("\nWhich algorithm to use:")
    for line in lines:
        print("  - " + line)

    tag = f"{map_name}_{start}_{goal}"
    if map_name == "husky":
        tag += f"_{args.card or 'nocard'}_{args.hour if args.hour is not None else ''}"
    tag = tag.replace(" ", "_")
    print("\nSaved", plot_search_comparison(
        G, results, os.path.join(OUT_DIR, f"route_{tag}.png"),
        title=f"{start} to {goal}  ({s['info']})"))
    print("Saved", plot_heuristic_trace(
        rows, os.path.join(OUT_DIR, f"heuristic_{tag}.png"),
        title=f"Heuristic along the A* route, {start} to {goal}"))
    if map_name == "campus" and not G.graph.get("metric"):
        try:
            print("Saved", interactive_map(
                G, os.path.join(OUT_DIR, f"route_{tag}.html"),
                routes={n: r["path"] for n, r in results.items() if r["found"]}))
        except ImportError:
            print("(Install folium to also get an HTML map of the routes.)")


def main():
    parser = argparse.ArgumentParser(description="Visualize the campus graph")
    parser.add_argument("--graph", default=os.path.join(OUT_DIR, "campus_walk.graphml"))
    parser.add_argument("--route", action="store_true", help="run the route finder")
    parser.add_argument("--map", choices=["husky", "campus"])
    parser.add_argument("--start")
    parser.add_argument("--goal")
    parser.add_argument("--profile", choices=["fastest", "step_free"])
    parser.add_argument("--card")
    parser.add_argument("--hour", type=int)
    args = parser.parse_args()

    if args.route:
        route_finder(args)
        return

    from pull_campus_graph import load_campus_graph
    G = load_campus_graph(args.graph)
    print("Saved", plot_static(G))
    print("Saved", interactive_map(G))


if __name__ == "__main__":
    main()