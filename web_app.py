#web_app.py

#Builds route_finder.html: a web page where you pick a start and goal building on the
#Northeastern campus, and A*, Weighted A*, Dijkstra and BFS run in the browser 
#Run it again after pull_campus_graph.py so the page uses the real campus.

import json
import math
import os
import webbrowser
from pathlib import Path
import networkx as nx
import husky_card_demo as hc
import neu_buildings as nb
import weather_demo as wd

OUT_DIR = os.path.join("data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "route_finder.html")
BUILDINGS_FILE = os.path.join(OUT_DIR, "neu_buildings.json")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_finder_template.html")

CAMPUS_MARGIN_M = 90      # streets this far outside the outermost buildings are kept, the rest dropped
SNAP_MAX_M = 150          # a building farther than this from any path is left out
CARD_RADIUS_M = 50        # covered paths this close to a card building count as inside it


def card_hours():
    #buildings with husky card requirement.
    hours = {}
    for _u, _v, _length, door in hc.EDGES:
        if door:
            building, open_h, close_h = door
            hours[building] = [open_h, close_h]
    return hours


def crop_to_campus(G, buildings):
    #To keep the map limited to campus locations
    if not buildings:
        return G
    lat0 = sum(b["lat"] for b in buildings) / len(buildings)
    dlat = CAMPUS_MARGIN_M / 110_540
    dlon = CAMPUS_MARGIN_M / (111_320 * math.cos(math.radians(lat0)))
    lat_lo, lat_hi = min(b["lat"] for b in buildings) - dlat, max(b["lat"] for b in buildings) + dlat
    lon_lo, lon_hi = min(b["lon"] for b in buildings) - dlon, max(b["lon"] for b in buildings) + dlon
    keep = [n for n, d in G.nodes(data=True) if lat_lo <= d["y"] <= lat_hi and lon_lo <= d["x"] <= lon_hi]
    H = G.subgraph(keep).copy()
    biggest = max(nx.connected_components(nx.Graph(H)), key=len)
    return H.subgraph(biggest).copy()


def snap_to_graph(G, buildings):
    #Attach each building to the nearest path junction.
    pts = [(n, d["y"], d["x"]) for n, d in G.nodes(data=True)]
    places = []
    for b in buildings:
        node, lat, lon = min(pts, key=lambda p: nb.distance_m(b["lat"], b["lon"], p[1], p[2]))
        snap = nb.distance_m(b["lat"], b["lon"], lat, lon)
        if snap > SNAP_MAX_M:
            print(f"  Left out {b['name']}: {snap:.0f} m from the nearest path")
            continue
        places.append({"name": b["name"], "lat": b["lat"], "lon": b["lon"], "node": str(node), "snap_m": round(snap)})
    return places


def campus_data():
    #The real campus graph if it has been downloaded, otherwise the test grid.
    graph_file = os.path.join(OUT_DIR, "campus_walk.graphml")
    places = []
    if os.path.exists(graph_file):
        from pull_campus_graph import load_campus_graph
        G = load_campus_graph(graph_file)
        source = "real campus graph"
        buildings = nb.load_buildings(BUILDINGS_FILE)
        G = crop_to_campus(G, buildings)
        places = snap_to_graph(G, buildings)
    else:
        from campus_search import test_grid
        G, start, goal = test_grid()
        source = "test grid (run pull_campus_graph.py for the real campus)"
        for name, node in (("Start (test grid)", start), ("Goal (test grid)", goal)):
            d = G.nodes[node]
            places.append({"name": name, "lat": d["y"], "lon": d["x"], "node": str(node), "snap_m": 0})

    # Flat x/y in metres around the centre, only used for drawing
    lat0 = sum(d["y"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    lon0 = sum(d["x"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    kx = 111_320 * math.cos(math.radians(lat0))

    def flat(lat, lon):
        return round((lon - lon0) * kx, 2), round((lat - lat0) * 110_540, 2)

    # Street names are kept only to label junctions along a route, never as places to pick
    names = {}
    for u, v, d in G.edges(data=True):
        name = d.get("name")
        if name and str(name) not in ("", "unknown", "nan"):
            for n in (u, v):
                names.setdefault(n, [])
                if str(name) not in names[n]:
                    names[n].append(str(name))
    nodes = []
    for n, d in G.nodes(data=True):
        x, y = flat(d["y"], d["x"])
        nodes.append({"id": str(n), "lat": d["y"], "lon": d["x"], "x": x, "y": y,
                      "label": " / ".join(names[n][:2]) if n in names else ""})

    # Buildings that need a Husky Card
    hours_by_key = card_hours()
    place_names = [p["name"] for p in places]
    key_to_name = {}
    for key in hours_by_key:
        full = nb.find_by_key(key, place_names)
        if full:
            key_to_name[key] = full
        else:
            print(f"  Card building '{key}' has no matching campus building, so it has no card doors")
    card = {"hours": {key_to_name[k]: h for k, h in hours_by_key.items() if k in key_to_name}}
    card_points = {p["name"]: (p["lat"], p["lon"]) for p in places if p["name"] in card["hours"]}

    for p in places:
        p["x"], p["y"] = flat(p["lat"], p["lon"])
        p["card"] = p["name"] in card["hours"]

    # One entry per physical path (both directions share a segment_id)
    seen, edges, door_counts = set(), [], {}
    for u, v, k, d in G.edges(keys=True, data=True):
        seg = d.get("segment_id", f"{min(u, v)}_{max(u, v)}_{k}")
        if seg in seen:
            continue
        seen.add(seg)
        steps, covered = d.get("is_steps"), d.get("is_covered")
        steps = steps is True or str(steps).lower() == "true"
        covered = covered is True or str(covered).lower() == "true"
        door = None
        if covered and card_points:   # tunnel with card entrance
            mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
            mid_lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
            near = min(card_points, key=lambda b: nb.distance_m(mid_lat, mid_lon, *card_points[b]))
            if nb.distance_m(mid_lat, mid_lon, *card_points[near]) <= CARD_RADIUS_M:
                door = near
                door_counts[near] = door_counts.get(near, 0) + 1
        edges.append({"u": str(u), "v": str(v),
                  "len": round(float(d.get("length_m", d.get("length", 0))), 2),
                  "steps": steps, "covered": covered, "door": door,  
                  "seg": seg})

    for name in card["hours"]:
        if not door_counts.get(name):
            print(f"  Note: no covered paths found inside {name}, so it adds no card shortcuts yet")
    return {"nodes": nodes, "edges": edges, "places": places, "card": card,
            "source": source}

def main():
    weather = wd.get_weather()
    if weather:
        print("Weather:", weather["condition"])
        print("Temperature:", weather["temperature"], "°C")
        print("Precipitation:", weather["precipitation"], "mm")

    campus = campus_data()
    page_places = [{k: v for k, v in p.items() if k not in ("lat", "lon", "snap_m")}
                   for p in campus["places"]]
    page_campus = {k: v for k, v in campus.items() if k != "source"}
    page_campus["places"] = page_places
    data = {"campus": page_campus, "weather": weather}
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    if "/*DATA*/null" not in page:
        raise SystemExit("route_finder_template.html is missing its /*DATA*/null placeholder")
    page = page.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")), 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Campus map: {campus['source']}, {len(campus['places'])} buildings, "
          f"{len(campus['nodes'])} junctions, {len(campus['edges'])} paths")
    for p in campus["places"]:
        if p["snap_m"] > 80:
            print(f"  Check {p['name']}: {p['snap_m']} m from the nearest path")
    print(f"Saved {OUT_FILE}")
    
    url = Path(OUT_FILE).resolve().as_uri()
    if webbrowser.open(url):
        print("Opened it in your browser.")
    else:
        print(f"Couldn't open a browser. Open this file yourself: {url}")


if __name__ == "__main__":
    main()
