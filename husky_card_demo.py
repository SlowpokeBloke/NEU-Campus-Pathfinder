"""
husky_card_demo.py

Finds a route from the Dorm to the Library when some buildings are locked
and need a Husky Card. Card numbers are stored as hashes.

Usage:
    python husky_card_demo.py
"""

import hashlib

from algorithms import astar, euclidean

SALT = "cs5100-demo"


def card_hash(card_id):
    return hashlib.sha256((SALT + card_id).encode()).hexdigest()


# Places on the map with (x, y) positions in metres
NODES = {
    "Dorm":                    (0, 0),
    "Path 1":                  (150, 0),
    "Path 2":                  (300, 0),
    "Path 3":                  (450, 0),
    "Khoury first floor room": (150, 100),
    "Gym room":                (225, 150),
    "Game room":               (300, 200),
    "Library":                 (450, 250),
    "Path 4":                  (450, 125),
}

# Building that each room is in
BUILDING = {
    "Khoury first floor room": "Khoury building",
    "Gym room":                "Marino Center",
    "Game room":               "Curry Center",
}

# Paths: (from, to, length, door)
# door is (building, opening hour, closing hour), or None if there is no door
EDGES = [
    ("Dorm", "Path 1", 150, None),
    ("Path 1", "Path 2", 150, None),
    ("Path 2", "Path 3", 150, None),
    ("Path 3", "Path 4", 125, None),
    ("Path 4", "Library", 125, None),
    ("Dorm", "Khoury first floor room", 181, None),
    ("Khoury first floor room", "Gym room", 91, ("Marino Center", 7, 22)),
    ("Gym room", "Path 4", 230, ("Marino Center", 7, 22)),
    ("Gym room", "Game room", 95, ("Curry Center", 8, 20)),
    ("Game room", "Library", 160, None),
]

# Neighbors of each place
GRAPH = {n: [] for n in NODES}
for u, v, length, door in EDGES:
    GRAPH[u].append((v, length, door))
    GRAPH[v].append((u, length, door))

# Buildings each card can open
ACCESS = {
    card_hash("001234567"): {"Marino Center", "Curry Center"},
    card_hash("009876543"): {"Marino Center"},
}


def locked_buildings(card_id, hour):
    """Buildings this person can't enter at this hour."""
    allowed = ACCESS.get(card_hash(card_id), set()) if card_id else set()
    locked = set()
    for _, _, _, door in EDGES:
        if door is None:
            continue
        building, open_h, close_h = door
        if not (open_h <= hour < close_h) and building not in allowed:
            locked.add(building)
    return locked


def make_neighbors(locked):
    """Neighbors of a place, skipping doors into locked buildings."""
    def neighbors(node):
        for nxt, length, door in GRAPH[node]:
            if door and door[0] in locked:
                continue
            yield nxt, length
    return neighbors


h_to_library = euclidean(NODES, "Library")


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
        res = astar("Dorm", "Library", make_neighbors(locked), h_to_library)
        path, cost, expanded = res["path"], res["cost"], len(res["explored"])
        route = " -> ".join(f"{n} ({BUILDING[n]})" if n in BUILDING else n
                            for n in path) if path else "no route"
        print(f"{label:<30}{str(sorted(locked) or '-'):<38}{cost:>6.0f}{expanded:>5}  {route}")


if __name__ == "__main__":
    main()