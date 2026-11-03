"""
algorithms.py

A* search and helper functions used by husky_card_demo.py and campus_search.py.
"""

import heapq
import itertools
import math
import time


# A* search

def astar(start, goal, neighbors, heuristic):
    """Find the cheapest path from start to goal using A*."""
    t0 = time.perf_counter()
    g = {start: 0.0}       # cost from start to each node
    parent = {}            # used to rebuild the path
    closed = set()         # nodes we are done with
    explored = []          # nodes in the order we expanded them
    tie = itertools.count()
    frontier = [(heuristic(start), next(tie), start)]

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        explored.append(node)

        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return {"found": True, "path": path[::-1], "cost": g[goal],
                    "explored": explored, "runtime_s": time.perf_counter() - t0}

        for nxt, step in neighbors(node):
            if step == math.inf or nxt in closed:
                continue
            new_g = g[node] + step
            if new_g < g.get(nxt, math.inf):
                g[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(frontier, (new_g + heuristic(nxt), next(tie), nxt))

    return {"found": False, "path": [], "cost": math.inf, "explored": explored,
            "runtime_s": time.perf_counter() - t0}


# Heuristics

EARTH_RADIUS_M = 6_371_009


def euclidean(coords, goal):
    """Straight-line distance to the goal. coords maps node -> (x, y)."""
    gx, gy = coords[goal]
    return lambda n: math.hypot(gx - coords[n][0], gy - coords[n][1])


def haversine(G, goal):
    """Distance to the goal on the Earth's surface, for OSMnx graphs."""
    lat2, lon2 = math.radians(G.nodes[goal]["y"]), G.nodes[goal]["x"]
    cache = {}

    def h(n):
        if n not in cache:
            lat1 = math.radians(G.nodes[n]["y"])
            dlat, dlon = lat2 - lat1, math.radians(lon2 - G.nodes[n]["x"])
            a = (math.sin(dlat / 2) ** 2
                 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
            cache[n] = 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
        return cache[n]

    return h


# Helpers for networkx graphs

def networkx_neighbors(G, cost_fn):
    """Return a neighbors function for a networkx graph."""
    def neighbors(n):
        best = {}
        for _, v, k, d in G.out_edges(n, keys=True, data=True):
            c = cost_fn(n, v, k, d)
            if c < best.get(v, math.inf):
                best[v] = c
        return best.items()
    return neighbors


def path_edges(G, path, cost_fn):
    """Return the edges (u, v, key) used by a path."""
    return [(u, v, min(G[u][v], key=lambda k: cost_fn(u, v, k, G[u][v][k])))
            for u, v in zip(path[:-1], path[1:])]