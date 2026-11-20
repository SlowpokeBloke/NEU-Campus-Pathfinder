"""
algorithms.py

A* search and helper functions used by husky_card_demo.py and campus_search.py.
"""

from collections import deque
import heapq
import itertools
import math
import time


# A* search
#implemented by Ronit and Kunal
def astar(start, goal, neighbors, heuristic, w=1.0):
    """Find the cheapest path from start to goal using A*."""
    t0 = time.perf_counter()
    g = {start: 0.0}       # cost from start to each node
    parent = {}            # used to rebuild the path
    closed = set()         # nodes we are done with
    explored = []          # nodes in the order we expanded them
    tie = itertools.count()
    frontier = [(w * heuristic(start), next(tie), start)]
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
                heapq.heappush(frontier, (new_g + w * heuristic(nxt), next(tie), nxt))

    return {"found": False, "path": [], "cost": math.inf, "explored": explored,
            "runtime_s": time.perf_counter() - t0}

def weighted_astar(start, goal, neighbors, heuristic, w=1.5):
    """Weighted A*: same as astar() but with a default weight of 1.5."""
    return astar(start, goal, neighbors, heuristic, w=w)

#BFS.
#implemented by Ronit and Kunal
def bfs(start, goal, neighbors):
    #finding path with the fewest steps, ignoring edge costs
    t0 = time.perf_counter()
    parent = {start: None}  #Tracks parent nodes and visited nodes
    step_cost = {start: 0.0}
    explored = []
    queue = deque([start])
 
    while queue:
        node = queue.popleft()
        explored.append(node)
 
        if node == goal:
            path, cost = [], 0.0
            while node is not None:
                path.append(node)
                cost += step_cost[node]
                node = parent[node]
            return {"found": True, "path": path[::-1], "cost": cost,
                    "explored": explored,
                    "runtime_s": time.perf_counter() - t0}
 
        for nxt, step in neighbors(node):
            if step == math.inf or nxt in parent:
                continue
            parent[nxt] = node
            step_cost[nxt] = step
            queue.append(nxt)
 
    return {"found": False, "path": [], "cost": math.inf, "explored": explored,
            "runtime_s": time.perf_counter() - t0}

# Heuristics

EARTH_RADIUS_M = 6_371_009

#implemented by Ronit and Kunal
def euclidean(coords, goal):
    """Straight-line distance to the goal. coords maps node -> (x, y)."""
    gx, gy = coords[goal]
    return lambda n: math.hypot(gx - coords[n][0], gy - coords[n][1])

#implemented by Ronit and Kunal
def haversine(G, goal):
    """Distance to the goal, for OSMnx graphs."""
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

# D* Lite search algorithm (Koenig & Likhachev, 2002)
#implemented by mCoca
#
# D* Lite searches backward from the goal, so g(s) is the cost-to-goal of s and the path is read forward from the start by always stepping to the neighbor with the smallest c(s, s') + g(s').
#  When edge costs change (a path closes, a door locks) only the affected vertices are repaired instead of searching again.
#
# Difference from astar(): the heuristic takes two nodes, heuristic(a, b) = estimate of the distance between a and b (evaluated from the current start to each vertex). Use euclidean_pair() / haversine_pair() below.

class DStarLite:
    def __init__(self, start, goal, neighbors, heuristic, predecessors=None):
        """neighbors(n) -> (next, cost) pairs, and must read the CURRENT costs.
        predecessors(n) -> (prev, cost of prev->n) pairs; if omitted the graph is
        assumed symmetric (true for the campus and Husky maps)."""
        self.start, self.goal, self.last = start, goal, start
        self.succ = neighbors
        self.pred = predecessors or neighbors
        self.h = heuristic
        self.km = 0.0
        self.g, self.rhs = {}, {}            # missing entries mean infinity
        self.U, self.key_of = [], {}         # priority queue with lazy deletion
        self.tie = itertools.count()
        self.explored, self.seen = [], set()
        self.expansions = 0
        self.rhs[goal] = 0.0                 # init
        self._insert(goal)

    def _g(self, s):
        return self.g.get(s, math.inf)

    def _rhs(self, s):
        return self.rhs.get(s, math.inf)

    def calc_key(self, s):
        m = min(self._g(s), self._rhs(s))
        return (m + self.h(self.start, s) + self.km, m)

    def _insert(self, u):
        k = self.calc_key(u)
        self.key_of[u] = k
        heapq.heappush(self.U, (k, next(self.tie), u))

    def _top(self):
        """Smallest valid (key, node) in U, or ((inf, inf), None) if U is empty."""
        while self.U:
            k, _, u = self.U[0]
            if self.key_of.get(u) == k:
                return k, u
            heapq.heappop(self.U)            # stale entry
        return (math.inf, math.inf), None

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min((c + self._g(v) for v, c in self.succ(u)
                               if c != math.inf), default=math.inf)
        self.key_of.pop(u, None)             # remove u from U if it is there
        if self._g(u) != self._rhs(u):
            self._insert(u)

    def compute_shortest_path(self):
        while True:
            k_top, u = self._top()
            if u is None or not (k_top < self.calc_key(self.start)
                                 or self._rhs(self.start) != self._g(self.start)):
                return
            k_new = self.calc_key(u)
            if k_top < k_new:                # if key is out of date, reinsert
                self._insert(u)
                continue
            self.key_of.pop(u, None)
            self.expansions += 1
            if u not in self.seen:
                self.seen.add(u)
                self.explored.append(u)
            if self._g(u) > self._rhs(u):    # if g is overconsistent: lower g
                self.g[u] = self._rhs(u)
                for s, _ in self.pred(u):
                    self.update_vertex(s)
            else:                            # else if g is underconsistent: raise to infinity
                self.g[u] = math.inf
                for s, _ in self.pred(u):
                    self.update_vertex(s)
                self.update_vertex(u)

    def next_step(self, s):
        """Best neighbor of s: argmin c(s, s') + g(s')."""
        best, best_cost = None, math.inf
        for v, c in self.succ(s):
            if c != math.inf and c + self._g(v) < best_cost:
                best, best_cost = v, c + self._g(v)
        return best

    def path(self):
        """Route from the current start to the goal, or [] if none exists."""
        if self._g(self.start) == math.inf:
            return []
        path, node, seen = [self.start], self.start, {self.start}
        while node != self.goal:
            node = self.next_step(node)
            if node is None or node in seen:
                return []
            seen.add(node)
            path.append(node)
        return path

    def path_cost(self, path):
        return sum(dict(self.succ(u))[v] for u, v in zip(path[:-1], path[1:]))

    def move_start(self, new_start):
        """The agent moved: shift the heuristic origin (km) before the next replan."""
        self.km += self.h(self.last, new_start)
        self.last = self.start = new_start

    def replan(self, changed_nodes, new_start=None):
        """Call after edge costs changed. changed_nodes are the endpoints (u and v)
        of every edge whose cost changed. Returns a result dict like astar()."""
        t0 = time.perf_counter()
        before = self.expansions
        if new_start is not None and new_start != self.start:
            self.move_start(new_start)
        for u in changed_nodes:
            self.update_vertex(u)
        self.compute_shortest_path()
        return self.result(t0, self.expansions - before)

    def result(self, t0, expansions=None):
        path = self.path()
        return {"found": bool(path), "path": path,
                "cost": self.path_cost(path) if path else math.inf,
                "explored": list(self.explored), "expansions": expansions,
                "runtime_s": time.perf_counter() - t0}


def dstar_lite(start, goal, neighbors, heuristic, predecessors=None):
    """One-shot D* Lite with the same result dict as astar(). For changing costs,
    keep the DStarLite object and call .replan() instead."""
    t0 = time.perf_counter()
    d = DStarLite(start, goal, neighbors, heuristic, predecessors)
    d.compute_shortest_path()
    return d.result(t0, d.expansions)


def euclidean_pair(coords):
    """heuristic(a, b) for D* Lite. coords maps node -> (x, y)."""
    return lambda a, b: math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])


def haversine_pair(G):
    """heuristic(a, b) for D* Lite on an OSMnx graph."""
    def h(a, b):
        lat1, lat2 = math.radians(G.nodes[a]["y"]), math.radians(G.nodes[b]["y"])
        dlon = math.radians(G.nodes[b]["x"] - G.nodes[a]["x"])
        x = (math.sin((lat2 - lat1) / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(x))
    return h