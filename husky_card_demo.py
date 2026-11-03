"""
husky_card_demo.py — Locked doors as obstacles, with Husky Card access.

Where hashing is used:
  1. Card IDs are stored as SHA-256 hashes, never as raw numbers.
  2. ACCESS maps card hash -> set of buildings   (dict: O(1) lookup)
  3. Before searching, the locked doors for this user go into a hash set,
     so the check inside A*'s inner loop is O(1).
  4. A* itself uses hash tables: g-scores (dict), parents (dict),
     closed set (set).

Run:  python husky_card_demo.py
"""

import hashlib
import heapq
import math

SALT = "cs5100-demo"   


def card_hash(card_id):
    return hashlib.sha256((SALT + card_id).encode()).hexdigest()


# Campus: node -> (x, y) in metres

NODES = {
    "Dorm":          (0, 0),
    "Path 1":        (150, 0),
    "Path 2":        (300, 0),
    "Path 3":        (450, 0),
    "Khoury first floor room": (150, 100),
    "Gym room":      (225, 150),
    "Game room":     (300, 200),
    "Library":       (450, 250),
    "Path 4":        (450, 125),
}

# Which building each room is in. Locks belong to buildings, not rooms.
BUILDING = {
    "Khoury first floor room": "Khoury building",
    "Gym room":                "Marino Center",
    "Game room":               "Curry Center",
}

# (u, v, length_m, door) — door = (building, public_open_hour, public_close_hour)
# Hours are demo values, not the real opening hours.
EDGES = [
    ("Dorm", "Path 1", 150, None),
    ("Path 1", "Path 2", 150, None),
    ("Path 2", "Path 3", 150, None),
    ("Path 3", "Path 4", 125, None),
    ("Path 4", "Library", 125, None),
    ("Dorm", "Khoury first floor room", 181, None),
    ("Khoury first floor room", "Gym room", 91, ("Marino Center", 7, 22)),   # door into Marino
    ("Gym room", "Path 4", 230, ("Marino Center", 7, 22)),                # Marino side door
    ("Gym room", "Game room", 95, ("Curry Center", 8, 20)),                  # door into Curry
    ("Game room", "Library", 160, None),
]

GRAPH = {n: [] for n in NODES}
for u, v, length, door in EDGES:
    GRAPH[u].append((v, length, door))
    GRAPH[v].append((u, length, door))

# Card hash -> buildings that card opens.
ACCESS = {
    card_hash("001234567"): {"Marino Center", "Curry Center"},  # engineering student
    card_hash("009876543"): {"Marino Center"},                  # biology student
}


# ---------------------------------------------------------------------------
# Locked doors for one user at one hour  ->  a hash set
# ---------------------------------------------------------------------------

def locked_buildings(card_id, hour):
    allowed = ACCESS.get(card_hash(card_id), set()) if card_id else set()
    locked = set()
    for _, _, _, door in EDGES:
        if door is None:
            continue
        building, open_h, close_h = door
        if not (open_h <= hour < close_h) and building not in allowed:
            locked.add(building)
    return locked


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

def h(n, goal):
    (x1, y1), (x2, y2) = NODES[n], NODES[goal]
    return math.hypot(x2 - x1, y2 - y1)   # admissible: straight line


def astar(start, goal, locked):
    g = {start: 0}              # hash table: node -> best cost so far
    parent = {}                 # hash table: node -> previous node
    closed = set()              # hash set: finished nodes
    frontier = [(h(start, goal), start)]
    expanded = 0

    while frontier:
        _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1], g[goal], expanded
        for nxt, length, door in GRAPH[node]:
            if door and door[0] in locked:        # O(1) hash lookup
                continue
            new_g = g[node] + length
            if new_g < g.get(nxt, math.inf):
                g[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(frontier, (new_g + h(nxt, goal), nxt))
    return None, math.inf, expanded


# ---------------------------------------------------------------------------

def main():
    users = [
        ("Visitor, 2 pm",                   None,        14),
        ("Visitor, 11 pm",                  None,        23),
        ("Biology student, 11 pm",          "009876543", 23),
        ("Engineering student, 11 pm",      "001234567", 23),
        ("Unknown card, 11 pm",             "000000000", 23),
    ]
    print("Stored card hashes (no raw card numbers):")
    for key, buildings in ACCESS.items():
        print(f"  {key[:16]}...  ->  {sorted(buildings)}")
    print()
    print(f"{'User':<30}{'locked':<38}{'cost':>6}{'exp':>5}  route")
    for label, card, hour in users:
        locked = locked_buildings(card, hour)
        path, cost, expanded = astar("Dorm", "Library", locked)
        route = " -> ".join(f"{n} ({BUILDING[n]})" if n in BUILDING else n
                            for n in path) if path else "no route"
        print(f"{label:<30}{str(sorted(locked) or '-'):<38}{cost:>6.0f}{expanded:>5}  {route}")


if __name__ == "__main__":
    main()