# NEU Campus Pathfinder

Accessible walking routes across Northeastern's Boston campus. The project
builds a walking network from OpenStreetMap, labels each path with features
that decide whether a route is usable — stairs, tunnels, crossings, indoor
corridors, locked doors — and runs A\* over it with a different cost function
per user profile.

## Setup

```
pip install "osmnx>=2.0" networkx matplotlib folium
```

Python 3.12. Run the scripts from this folder, or they won't find their data.

## Running it

**Download the map first.** Requires internet. The result is not stored in git.

```
python pull_campus_graph.py
```

Writes the network to `data/processed/`, along with QGIS-readable copies, a
spreadsheet of every path for fieldwork notes, and a validation summary. Every
other script reads that saved copy.

**Find routes.**

```
python campus_search.py
```

Routes one trip under both profiles, then closes a path and routes again.
Reports distance, nodes expanded, and runtime.

**Draw the map.**

```
python visualize_graph.py
```

Writes a PNG with each path type in its own colour, and an interactive HTML
map.

**Husky Card demo.**

```
python husky_card_demo.py
```

Separate example on a hand-made map: one trip, five people, different times of
day. Card numbers are stored as hashes.

## Files

- `pull_campus_graph.py` — downloads campus and labels every path.
- `algorithms.py` — the search itself: A\*, weighted A\*, BFS.
- `profiles.py` — what each kind of traveller cares about, and closed paths.
- `campus_search.py` — runs the searches and reports the results.
- `visualize_graph.py` — draws the maps.
- `husky_card_demo.py` — the card access example.

A **profile** sets what a path costs. `fastest` counts metres. `step_free`
gives stairs infinite cost, so the search avoids them. A new profile is one
function.

A **closure** marks a path as shut, for all profiles or some, and the search
routes around it.

Costs must be length plus penalties, never less than length. A\* only
guarantees the best route while that holds.

## Reference papers

In `Research Paper/` — the original A\* paper by Hart, Nilsson and Raphael,
plus *D\* Lite* and *Lifelong Planning A\**. The latter two cover repairing a
route when the map changes instead of starting over, the next step for closure
handling.
