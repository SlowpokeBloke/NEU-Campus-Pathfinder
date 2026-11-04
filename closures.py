"""
closures.py — Let people report temporary closures and feed them to routing.

Owner: (shared) — user-reported disruptions layered on top of the OSM graph.
Requires: the cleaned graph from pull_campus_graph.py (data/processed/campus_walk.graphml)

A closure is a user report that one physical segment is unusable, or harder to
use than normal, for a stretch of time: construction hoarding, an unploughed
path, a locked door, a broken elevator, an event blocking a plaza. Reports live
in a small JSON file, NOT in the graph, so the downloaded network stays exactly
as OpenStreetMap produced it and results stay reproducible. Routing applies the
active reports at query time with apply_closures().

Report one (interactive — it asks for everything it needs):
    python closures.py report

Or in one line, by map location or by segment id:
    python closures.py report --lat 42.3398 --lon -71.0892 --reason construction
    python closures.py report --segment-id 61342573_7725901136_0 --reason snow_ice

See and withdraw reports:
    python closures.py list                  # in force now or coming up
    python closures.py list --all            # including expired and cancelled
    python closures.py cancel c3f9a1

Check what routing would do with them:
    python closures.py apply

Draw them on the map — the PNG and the HTML map from visualize_graph.py, with
the reported segments picked out in the "Closed" layer:
    python closures.py map

In the search algorithms (A* in algorithms.py treats math.inf as impassable,
which is exactly what a blocked segment costs):
    from pull_campus_graph import load_campus_graph
    from algorithms import astar, haversine, networkx_neighbors
    from closures import apply_closures, closure_cost_fn
    G = load_campus_graph()
    apply_closures(G)        # sets is_closed / closure_multiplier on every edge
    res = astar(start, goal, networkx_neighbors(G, closure_cost_fn()),
                haversine(G, goal))

To mark the closures on a search plot, pass blocked_edges(G) through as the
"blocked" key of the result that visualize_graph.plot_search_comparison reads.
"""

import argparse
import json
import math
import os
import uuid
from datetime import datetime, timedelta

from pull_campus_graph import load_campus_graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLOSURES_PATH = os.path.join("data", "closures.json")

# Where the graph lives and where the closure maps are written — the same
# folder visualize_graph.py uses.
OUT_DIR = os.path.join("data", "processed")
GRAPH_PATH = os.path.join(OUT_DIR, "campus_walk.graphml")

# "blocked" means do not route through it at all. "partial" means passable but
# worse than the tags say — a narrowed sidewalk, slush, a propped-open side
# door — so the cost goes up instead of the edge disappearing.
SEVERITIES = ["blocked", "partial"]

# Cost multiplier for a partial closure. Three times the normal length is
# roughly "take it only if the detour is long".
PARTIAL_MULTIPLIER = 3.0

# Free text is allowed, but these are the cases we expect on campus and they
# keep the reports groupable in the write-up.
COMMON_REASONS = ["construction", "snow_ice", "event", "elevator_out",
                  "door_locked", "flooding", "obstruction", "maintenance",
                  "other"]

# A report with no end date is treated as lasting this long, so one forgotten
# report cannot reroute the campus for the rest of the semester.
DEFAULT_DURATION_DAYS = 7

# Reject a point further than this from any segment — almost certainly a typo
# in the coordinates rather than a real spot on campus.
MAX_SNAP_DISTANCE_M = 150.0


# ---------------------------------------------------------------------------
# 1. Store
# ---------------------------------------------------------------------------

def load_closures(path=CLOSURES_PATH):
    """Every report ever filed, newest last. Missing file means none yet."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_closures(closures, path=CLOSURES_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(closures, f, indent=2)
        f.write("\n")


def _parse_time(value, end_of_day=False):
    """Accept '2026-11-20' or '2026-11-20T14:30'. Returns a naive datetime."""
    if value in (None, ""):
        return None
    try:
        when = datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Could not read the date/time {value!r}. "
                         "Use YYYY-MM-DD or YYYY-MM-DDTHH:MM.")
    if end_of_day and len(str(value)) == 10:
        when = when.replace(hour=23, minute=59, second=59)
    return when


def is_active(closure, when=None):
    """Is this report in force at `when` (default: now) and not cancelled?"""
    when = when or datetime.now()
    if closure.get("cancelled_at"):
        return False
    start = _parse_time(closure.get("start"))
    end = _parse_time(closure.get("end"))
    if start and when < start:
        return False
    if end and when > end:
        return False
    return True


def closure_state(closure, when=None):
    """cancelled / scheduled / active / expired, for display."""
    when = when or datetime.now()
    if closure.get("cancelled_at"):
        return "cancelled"
    start = _parse_time(closure.get("start"))
    end = _parse_time(closure.get("end"))
    if start and when < start:
        return "scheduled"
    if end and when > end:
        return "expired"
    return "active"


def active_closures(closures=None, when=None, path=CLOSURES_PATH):
    closures = load_closures(path) if closures is None else closures
    return [c for c in closures if is_active(c, when)]


# ---------------------------------------------------------------------------
# 2. Locate the segment being reported
# ---------------------------------------------------------------------------

def _local_xy(lat, lon, lat0):
    """Flat metres near lat0. Good to centimetres over a campus-sized area."""
    r = 6_371_009  # same mean Earth radius campus_graph uses
    return (math.radians(lon) * math.cos(math.radians(lat0)) * r,
            math.radians(lat) * r)


def _point_to_polyline_m(lat, lon, coords):
    """Shortest distance in metres from a point to a (lon, lat) polyline."""
    px, py = _local_xy(lat, lon, lat)
    best = float("inf")
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        ax, ay = _local_xy(lat1, lon1, lat)
        bx, by = _local_xy(lat2, lon2, lat)
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        # t is how far along the segment the closest point falls, clamped to
        # the segment itself so an endpoint wins when the point is off the end.
        t = 0.0 if span == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _edge_coords(G, u, v, data):
    """(lon, lat) points along the edge."""
    geom = data.get("geometry")
    if geom is not None:
        return list(geom.coords)
    return [(G.nodes[u]["x"], G.nodes[u]["y"]),
            (G.nodes[v]["x"], G.nodes[v]["y"])]


def nearest_segment(G, lat, lon):
    """The closest physical segment to a point: (segment_id, name, metres)."""
    best = (None, "", float("inf"))
    for u, v, d in G.edges(data=True):
        dist = _point_to_polyline_m(lat, lon, _edge_coords(G, u, v, d))
        if dist < best[2]:
            best = (d["segment_id"], d.get("name", ""), dist)
    return best


def segment_index(G):
    """segment_id -> name, for checking a reported id really exists."""
    return {d["segment_id"]: d.get("name", "") for _, _, d in G.edges(data=True)}


# ---------------------------------------------------------------------------
# 3. File a report
# ---------------------------------------------------------------------------

def _expire_after(iso_start, days=DEFAULT_DURATION_DAYS):
    return (datetime.fromisoformat(iso_start)
            + timedelta(days=days)).isoformat(timespec="seconds")


def report_closure(segment_id=None, lat=None, lon=None, reason="other",
                   severity="blocked", reported_by="", notes="",
                   start=None, end=None, G=None, path=CLOSURES_PATH):
    """Record one closure and return it. Give either a segment_id or a point.

    Raises ValueError on anything that would make the report unusable later:
    an unknown segment, a point nowhere near campus, a bad severity or date.
    """
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    if not reason:
        raise ValueError("A reason is required — it is what other users read.")

    start_dt = _parse_time(start)
    end_dt = _parse_time(end, end_of_day=True)
    if start_dt and end_dt and end_dt <= start_dt:
        raise ValueError("The closure ends before it starts.")

    G = load_campus_graph() if G is None else G
    snapped_m = None
    if segment_id:
        name = segment_index(G).get(segment_id)
        if name is None:
            raise ValueError(f"No segment {segment_id!r} in the graph. "
                             "Check data/processed/segments_to_annotate.csv.")
    elif lat is not None and lon is not None:
        segment_id, name, snapped_m = nearest_segment(G, float(lat), float(lon))
        if snapped_m > MAX_SNAP_DISTANCE_M:
            raise ValueError(f"Nearest path is {snapped_m:.0f} m away, further than "
                             f"{MAX_SNAP_DISTANCE_M:.0f} m. Check the coordinates.")
    else:
        raise ValueError("Give either a segment id or both a latitude and a longitude.")

    closure = {
        "closure_id": uuid.uuid4().hex[:6],
        "segment_id": segment_id,
        "segment_name": name,
        "severity": severity,
        "reason": reason,
        "notes": notes,
        "reported_by": reported_by,
        "reported_at": datetime.now().isoformat(timespec="seconds"),
        "start": start_dt.isoformat(timespec="seconds") if start_dt else None,
        "end": end_dt.isoformat(timespec="seconds") if end_dt else None,
        "reported_lat": round(float(lat), 6) if lat is not None else None,
        "reported_lon": round(float(lon), 6) if lon is not None else None,
        "snapped_distance_m": round(snapped_m, 1) if snapped_m is not None else None,
        "cancelled_at": None,
    }
    if closure["end"] is None:
        closure["end"] = _expire_after(closure["start"] or closure["reported_at"])

    closures = load_closures(path)
    closures.append(closure)
    save_closures(closures, path)
    return closure


def cancel_closure(closure_id, path=CLOSURES_PATH):
    """Withdraw a report. Kept in the file with a timestamp rather than
    deleted, so the history stays auditable for the report."""
    closures = load_closures(path)
    for c in closures:
        if c["closure_id"] == closure_id:
            if c["cancelled_at"]:
                raise ValueError(f"{closure_id} was already cancelled.")
            c["cancelled_at"] = datetime.now().isoformat(timespec="seconds")
            save_closures(closures, path)
            return c
    raise ValueError(f"No closure with id {closure_id!r}.")


# ---------------------------------------------------------------------------
# 4. Apply to the graph
# ---------------------------------------------------------------------------

def apply_closures(G, closures=None, when=None, remove_blocked=False,
                   path=CLOSURES_PATH):
    """Mark the edges affected by the active reports; return how many directed
    edges were affected.

    Every edge gets is_closed, closure_multiplier, closure_reason and
    closure_id, so a cost function can read them without checking whether the
    key exists. Both directions of a reported physical segment are affected.
    length_m is left alone — it stays the true walking distance; multiply by
    closure_multiplier (or call closure_cost) when building the cost.
    """
    reports = active_closures(closures, when, path)
    by_segment = {}
    for c in reports:
        # Worst report wins: blocked overrides partial on the same segment.
        current = by_segment.get(c["segment_id"])
        if current is None or (current["severity"] == "partial"
                               and c["severity"] == "blocked"):
            by_segment[c["segment_id"]] = c

    affected = 0
    to_remove = []
    for u, v, k, d in G.edges(keys=True, data=True):
        c = by_segment.get(d["segment_id"])
        d["is_closed"] = bool(c and c["severity"] == "blocked")
        d["closure_multiplier"] = 1.0
        d["closure_reason"] = ""
        d["closure_id"] = ""
        if c:
            d["closure_multiplier"] = (math.inf if d["is_closed"]
                                       else PARTIAL_MULTIPLIER)
            d["closure_reason"] = c["reason"]
            d["closure_id"] = c["closure_id"]
            affected += 1
            if d["is_closed"]:
                to_remove.append((u, v, k))

    if remove_blocked:
        G.remove_edges_from(to_remove)
    return affected


def closure_cost(data, base="length_m"):
    """Cost of one edge with the closures applied; infinite if it is blocked."""
    return data[base] * data.get("closure_multiplier", 1.0)


def closure_weight(base="length_m"):
    """Weight callable for networkx, to use once apply_closures() has run:

        nx.astar_path(G, s, t, heuristic=h, weight=closure_weight())

    Returning None is how networkx is told an edge cannot be used, so a
    blocked segment is skipped outright instead of being taken at a huge cost.
    On a multigraph networkx hands over all parallel edges at once, so take the
    cheapest usable one.
    """
    def weight(u, v, data):
        edges = data.values() if "length_m" not in data else [data]
        costs = [closure_cost(d, base) for d in edges if not d.get("is_closed")]
        return min(costs) if costs else None
    return weight


def closure_cost_fn(base="length_m", inner=None):
    """Cost function in the shape algorithms.py wants — cost(u, v, k, d) — for
    networkx_neighbors() and path_edges(). A blocked segment costs math.inf,
    which astar() skips outright.

    Pass `inner` to layer closures on top of an existing profile cost, e.g.
    closure_cost_fn(inner=step_free_cost): the profile decides the base cost,
    closures then scale it.
    """
    def cost(u, v, k, d):
        base_cost = inner(u, v, k, d) if inner else d[base]
        return base_cost * d.get("closure_multiplier", 1.0)
    return cost


def blocked_edges(G):
    """The (u, v) pairs currently blocked, for the "blocked" key of a result
    handed to visualize_graph.plot_search_comparison, which draws an X on each.
    One entry per physical segment, since both directions overlap on the map.
    """
    seen, out = set(), []
    for u, v, d in G.edges(data=True):
        if d.get("is_closed") and d.get("segment_id") not in seen:
            seen.add(d.get("segment_id"))
            out.append((u, v))
    return out


# ---------------------------------------------------------------------------
# 5. Command line
# ---------------------------------------------------------------------------

def _ask(prompt, default="", choices=None):
    while True:
        answer = input(f"{prompt} [{default}]: " if default else f"{prompt}: ")
        # lstrip the BOM too: piping answers in from PowerShell prefixes one.
        answer = answer.strip().lstrip("﻿").strip() or default
        if choices and answer not in choices:
            print(f"  Pick one of: {', '.join(choices)}")
            continue
        return answer


def _ask_float(prompt):
    while True:
        try:
            return float(_ask(prompt))
        except ValueError:
            print("  That is not a number.")


def _report_interactively():
    print("Report a closure. Press Enter to accept the value in brackets.\n")
    print("Where is it? Give a segment id from segments_to_annotate.csv, or")
    print("leave it blank to point at a spot on the map instead.")
    segment_id = _ask("Segment id", "")
    lat = lon = None
    if not segment_id:
        lat = _ask_float("Latitude (e.g. 42.3398)")
        lon = _ask_float("Longitude (e.g. -71.0892)")
    reason = _ask(f"Reason ({'/'.join(COMMON_REASONS)})", "construction")
    severity = _ask("Severity (blocked = impassable, partial = harder)",
                    "blocked", choices=SEVERITIES)
    reported_by = _ask("Your name or NUID", "")
    start = _ask("Starts (YYYY-MM-DD, blank = now)", "")
    end = _ask(f"Ends (YYYY-MM-DD, blank = +{DEFAULT_DURATION_DAYS} days)", "")
    notes = _ask("Notes (what exactly is blocked?)", "")
    return dict(segment_id=segment_id or None, lat=lat, lon=lon, reason=reason,
                severity=severity, reported_by=reported_by, notes=notes,
                start=start or None, end=end or None)


def _print_closure(c, prefix=""):
    where = c["segment_name"] or c["segment_id"]
    print(f"{prefix}{c['closure_id']}  {c['severity']:<8} "
          f"{closure_state(c):<9} {where}")
    print(f"{prefix}        {c['reason']}" + (f" - {c['notes']}" if c["notes"] else ""))
    print(f"{prefix}        segment {c['segment_id']}"
          + (f", reported by {c['reported_by']}" if c["reported_by"] else ""))
    print(f"{prefix}        {c['start'] or 'from now'} to {c['end'] or 'no end'}")


def cmd_report(args):
    if args.segment_id is None and args.lat is None and args.lon is None:
        fields = _report_interactively()
    else:
        fields = dict(segment_id=args.segment_id, lat=args.lat, lon=args.lon,
                      reason=args.reason, severity=args.severity,
                      reported_by=args.by, notes=args.notes,
                      start=args.since, end=args.until)
    print("\nLooking up the segment...")
    closure = report_closure(**fields)
    print("\nFiled:")
    _print_closure(closure, prefix="  ")
    if closure["snapped_distance_m"] is not None:
        print(f"  Snapped {closure['snapped_distance_m']} m to the nearest path.")
    print(f"\nSaved to {CLOSURES_PATH}. Cancel it with: "
          f"python closures.py cancel {closure['closure_id']}")


def cmd_list(args):
    closures = load_closures()
    if not args.all:
        # Standing reports: in force now, or filed for a date still to come.
        closures = [c for c in closures
                    if closure_state(c) in ("active", "scheduled")]
    if not closures:
        print("No closures reported yet." if args.all else "No standing closures.")
        return
    print(f"{len(closures)} closure(s):\n")
    for c in closures:
        _print_closure(c, prefix="  ")
        print()


def cmd_cancel(args):
    c = cancel_closure(args.closure_id)
    print(f"Cancelled {c['closure_id']} on {c['segment_name'] or c['segment_id']}.")


# visualize_graph already has a "Closed" category keyed on is_closed, so a
# blocked segment draws itself once apply_closures() has run. A partial closure
# has no category there, and would otherwise be drawn as a plain path, so add
# one — and put the closure fields in the popups while we are here.
PARTIAL_COLOR = "#FF7F0E"


def _add_closure_layers(viz):
    """Teach visualize_graph about partial closures. Idempotent, so repeated
    calls in one session do not stack duplicate categories."""
    if "Partial closure" in viz.COLOR:
        return
    test = lambda d: (not viz._true(d.get("is_closed"))
                      and float(d.get("closure_multiplier", 1.0) or 1.0) > 1.0)
    closed_at = next(i for i, (name, _, _) in enumerate(viz.CATEGORIES)
                     if name == "Closed")
    viz.CATEGORIES.insert(closed_at + 1, ("Partial closure", PARTIAL_COLOR, test))
    viz.COLOR["Partial closure"] = PARTIAL_COLOR
    for field in ("closure_reason", "closure_id"):
        if field not in viz.TOOLTIP_FIELDS:
            viz.TOOLTIP_FIELDS.append(field)


def cmd_map(args):
    import visualize_graph as viz

    when = _parse_time(args.when)
    G = load_campus_graph(args.graph)
    affected = apply_closures(G, when=when)
    reports = active_closures(when=when)
    _add_closure_layers(viz)

    stamp = when.strftime("%Y-%m-%d %H:%M") if when else "now"
    print(f"{len(reports)} report(s) in force at {stamp}, "
          f"{affected} directed edge(s) affected.")
    for c in reports:
        print(f"  {c['closure_id']}  {c['severity']:<8} {c['reason']:<14} "
              f"{c['segment_name'] or c['segment_id']}")

    png = viz.plot_static(G, out_path=os.path.join(OUT_DIR, "closures_map.png"),
                          title=f"Campus network with reported closures ({stamp})")
    html = viz.interactive_map(G, out_path=os.path.join(OUT_DIR, "closures_map.html"))
    print(f"\nSaved {png}\nSaved {html}")
    if not reports:
        print("No closures to draw — the map is the plain network.")


def cmd_apply(args):
    G = load_campus_graph()
    before = G.number_of_edges()
    # Count first, then remove — otherwise the blocked edges are gone before
    # the per-report tally can see them.
    affected = apply_closures(G)
    reports = active_closures()
    print(f"{len(reports)} active report(s) affecting {affected} directed edge(s).")
    for c in reports:
        hit = sum(1 for _, _, d in G.edges(data=True)
                  if d.get("closure_id") == c["closure_id"])
        print(f"  {c['closure_id']}  {c['severity']:<8} {c['reason']:<14} "
              f"{c['segment_name'] or c['segment_id']}  ({hit} edges)")
    if args.remove_blocked:
        import networkx as nx
        G.remove_edges_from([(u, v, k) for u, v, k, d
                             in G.edges(keys=True, data=True) if d["is_closed"]])
        print(f"\nRemoved {before - G.number_of_edges()} edge(s). "
              f"Still strongly connected: {nx.is_strongly_connected(G)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="report a closure (interactive if no flags)")
    p.add_argument("--segment-id", help="id from segments_to_annotate.csv")
    p.add_argument("--lat", type=float, help="latitude of the closure")
    p.add_argument("--lon", type=float, help="longitude of the closure")
    p.add_argument("--reason", default="other",
                   help=f"suggested: {', '.join(COMMON_REASONS)}")
    p.add_argument("--severity", choices=SEVERITIES, default="blocked")
    p.add_argument("--by", default="", help="who is reporting it")
    p.add_argument("--notes", default="")
    p.add_argument("--since", help="start date/time (default: now)")
    p.add_argument("--until", help=f"end date/time (default: +{DEFAULT_DURATION_DAYS} days)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("list", help="show reported closures")
    p.add_argument("--all", action="store_true",
                   help="include expired and cancelled reports")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("map", help="draw the closures on the campus map")
    p.add_argument("--graph", default=GRAPH_PATH)
    p.add_argument("--when", help="draw the closures in force at this date/time "
                                  "(default: now) — use it to preview a scheduled one")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("cancel", help="withdraw a report")
    p.add_argument("closure_id")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("apply", help="show what routing would do with the active reports")
    p.add_argument("--remove-blocked", action="store_true",
                   help="also delete the blocked edges and re-check connectivity")
    p.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    try:
        args.func(args)
    except ValueError as err:
        raise SystemExit(f"Error: {err}")


if __name__ == "__main__":
    main()
