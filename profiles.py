"""
profiles.py — User profiles as cost functions, plus closures (obstacles).

A closure blocks one path segment:
    {"target": "211_212_0",          # a segment_id from pull_campus_graph.py
     "type": "construction",
     "affects": ["all"]}             # profile names, or ["all"]

Rule for keeping A* admissible: every cost is length + penalties >= length.
Never subtract from the length (no "bonuses").
"""

import math


def _true(value):
    return value is True or str(value).lower() == "true"


def _length(d):
    return float(d.get("length_m", d.get("length", 0.0)))


def fastest(d):
    return _length(d)


def step_free(d):
    if _true(d.get("is_steps")):
        return math.inf                      
    return _length(d)


PROFILES = {"fastest": fastest, "step_free": step_free}


def blocked_segments(closures, profile):
    """segment_ids closed for this profile (a hash set, so lookups are O(1))."""
    return {c["target"] for c in closures
            if profile in c.get("affects", ["all"]) or "all" in c.get("affects", ["all"])}


def make_cost_fn(profile, closures=()):
    """Cost function for algorithms.astar(): the profile's cost, with closures
    that affect this profile turned into blocks."""
    base = PROFILES[profile]
    blocked = blocked_segments(closures, profile)

    def cost(u, v, key, d):
        if d.get("segment_id") in blocked:
            return math.inf
        return base(d)

    return cost


def blocked_edges(G, closures, profile):
    """(u, v) pairs blocked for this profile, e.g. to draw them on a map."""
    blocked = blocked_segments(closures, profile)
    return {(u, v) for u, v, d in G.edges(data=True) if d.get("segment_id") in blocked}