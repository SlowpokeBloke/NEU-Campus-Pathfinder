"""
campus_search.py

Runs A* on the campus graph for the fastest and step-free profiles,
then closes one path and runs it again.

Usage:
    python campus_search.py
    python campus_search.py --start 123 --goal 456
"""

import argparse
import os
import random

import networkx as nx

from algorithms import astar, haversine, networkx_neighbors
from profiles import make_cost_fn
from pull_campus_graph import OUT_DIR, haversine_m, standardise_attributes

GRAPH_FILE = os.path.join(OUT_DIR, "campus_walk.graphml")


def test_grid(n=20):
    """Small grid used when the real campus graph hasn't been downloaded yet."""
    random.seed(3)
    G = nx.MultiDiGraph(crs="epsg:4326")
    for r in range(n):
        for c in range(n):
            G.add_node(r * n + c, x=-71.0940 + c * 0.00045, y=42.3350 + r * 0.00032)
    for r in range(n):
        for c in range(n):
            for dr, dc in [(0, 1), (1, 0)]:
                if r + dr < n and c + dc < n:
                    u, v = r * n + c, (r + dr) * n + (c + dc)
                    length = haversine_m(G.nodes[u]["y"], G.nodes[u]["x"],
                                         G.nodes[v]["y"], G.nodes[v]["x"])
                    tags = {"highway": "steps"} if random.random() < 0.15 else {"highway": "footway"}
                    for a, b in [(u, v), (v, u)]:
                        G.add_edge(a, b, length=length, **tags)
    standardise_attributes(G)
    mid = n // 2
    return G, mid * n, mid * n + n - 1


def report(label, res):
    status = f"{res['cost']:>9.1f}" if res["found"] else "  no path"
    print(f"{label:<32}{status}{len(res['explored']):>10}{res['runtime_s'] * 1000:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int)
    ap.add_argument("--goal", type=int)
    args = ap.parse_args()

    if os.path.exists(GRAPH_FILE):
        from pull_campus_graph import load_campus_graph
        G = load_campus_graph(GRAPH_FILE)
        nodes = list(G.nodes)
        start = args.start if args.start is not None else nodes[0]
        goal = args.goal if args.goal is not None else nodes[-1]
        print(f"Loaded {GRAPH_FILE}: {G.number_of_nodes()} nodes")
    else:
        G, start, goal = test_grid()
        print(f"{GRAPH_FILE} not found, using a {G.number_of_nodes()}-node test grid")

    h = haversine(G, goal)
    print(f"\nA* from {start} to {goal}\n")
    print(f"{'Scenario':<32}{'cost (m)':>9}{'expanded':>10}{'ms':>8}")

    # Route for each profile
    results = {}
    for profile in ["fastest", "step_free"]:
        res = astar(start, goal, networkx_neighbors(G, make_cost_fn(profile)), h)
        results[profile] = res
        report(profile, res)

    # Close a path on the step-free route and search again
    route = results["step_free"]["path"]
    if len(route) > 2:
        u, v = route[len(route) // 2], route[len(route) // 2 + 1]
        seg = G[u][v][min(G[u][v])]["segment_id"]
        closure = [{"target": seg, "type": "construction", "affects": ["all"]}]
        res = astar(start, goal, networkx_neighbors(G, make_cost_fn("step_free", closure)), h)
        report("step_free, path closed", res)
        print(f"\nClosed segment {seg}: the detour adds "
              f"{res['cost'] - results['step_free']['cost']:.1f} m")


if __name__ == "__main__":
    main()