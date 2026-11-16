"""
make_web_app.py

Builds route_finder.html: a web page where you pick a start and goal,
and A*, Weighted A*, Dijkstra and BFS run in the browser (no Python server).

Usage:
    python make_web_app.py             # builds the page and opens it in your browser
    python make_web_app.py --no-open   # builds the page only

Run it again after pull_campus_graph.py so the page uses the real campus.
"""

import argparse
import json
import math
import os
import webbrowser
from pathlib import Path

import husky_card_demo as hc

OUT_DIR = os.path.join("data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "route_finder.html")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_finder_template.html")


def husky_data():
    """The Husky Card demo map, door hours and hashed card table."""
    hours = {}
    edges = []
    for u, v, length, door in hc.EDGES:
        if door:
            building, open_h, close_h = door
            hours[building] = [open_h, close_h]
        edges.append({"u": u, "v": v, "len": length,
                      "door": door[0] if door else None,
                      "seg": f"{min(u, v)}|{max(u, v)}"})
    nodes = [{"id": n, "x": x, "y": y, "label": n, "building": hc.BUILDING.get(n)}
             for n, (x, y) in hc.NODES.items()]
    access = {key: sorted(b) for key, b in hc.ACCESS.items()}
    return {"kind": "husky", "nodes": nodes, "edges": edges, "hours": hours,
            "access": access, "salt": hc.SALT, "start": "Dorm", "goal": "Library"}


def campus_data():
    """The real campus graph if it has been downloaded, otherwise the test grid."""
    graph_file = os.path.join(OUT_DIR, "campus_walk.graphml")
    if os.path.exists(graph_file):
        from pull_campus_graph import load_campus_graph
        G = load_campus_graph(graph_file)
        start = goal = None
        source = "real campus graph"
    else:
        from campus_search import test_grid
        G, start, goal = test_grid()
        source = "test grid (run pull_campus_graph.py for the real campus)"

    # Flat x/y in metres around the centre, only used for drawing
    lat0 = sum(d["y"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    lon0 = sum(d["x"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    kx = 111_320 * math.cos(math.radians(lat0))
    # Label each place with the names of the paths it touches, e.g. "123 (Forsyth St)"
    names = {}
    for u, v, d in G.edges(data=True):
        name = d.get("name")
        if name and str(name) not in ("", "unknown", "nan"):
            for n in (u, v):
                names.setdefault(n, [])
                if str(name) not in names[n]:
                    names[n].append(str(name))
    nodes = [{"id": str(n), "lat": d["y"], "lon": d["x"],
              "x": round((d["x"] - lon0) * kx, 2), "y": round((d["y"] - lat0) * 110_540, 2),
              "label": f"{n} ({' / '.join(names[n][:2])})" if n in names else str(n)}
             for n, d in G.nodes(data=True)]

    # One entry per physical path (both directions share a segment_id)
    seen, edges = set(), []
    for u, v, k, d in G.edges(keys=True, data=True):
        seg = d.get("segment_id", f"{min(u, v)}_{max(u, v)}_{k}")
        if seg in seen:
            continue
        seen.add(seg)
        steps = d.get("is_steps")
        edges.append({"u": str(u), "v": str(v),
                      "len": round(float(d.get("length_m", d.get("length", 0))), 2),
                      "steps": steps is True or str(steps).lower() == "true",
                      "seg": seg, "name": d.get("name") or ""})
    return {"kind": "campus", "nodes": nodes, "edges": edges, "source": source,
            "start": None if start is None else str(start),
            "goal": None if goal is None else str(goal)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true", help="don't open the page in a browser")
    args = ap.parse_args()

    data = {"husky": husky_data(), "campus": campus_data()}
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    page = page.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Campus map: {data['campus']['source']}, "
          f"{len(data['campus']['nodes'])} nodes, {len(data['campus']['edges'])} paths")
    print(f"Saved {OUT_FILE}")

    if not args.no_open:
        url = Path(OUT_FILE).resolve().as_uri()
        if webbrowser.open(url):
            print("Opened it in your browser.")
        else:
            print(f"Couldn't open a browser. Open this file yourself: {url}")


if __name__ == "__main__":
    main()