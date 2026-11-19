
#neu_buildings.py

# The list of Northeastern campus buildings used by the route finder.

# Fetching building names from OSM and updating to their original names.

# The POSITIONS are looked up once in OpenStreetMap (needs internet + osmnx, which
# pull_campus_graph.py already uses) and saved to data/processed/neu_buildings.json.

#  After that the page builds offline. You can open that JSON file and fix or add a
#  building by hand: each entry is {"name": ..., "latitude": ..., "longitute": ...}.

#Run `python neu_buildings.py` to refresh the file and see which buildings were found.

import json
import math
import os
import re

CAMPUS_CENTER = (42.3398, -71.0892)   # latitude, longitude: middle of the NEU campus
SEARCH_RADIUS_M = 800                # how far around the centre to look in OpenStreetMap
MAX_FROM_CENTER_M = 700              # a match farther than this is probably a different building

BUILDINGS = [
    ("Badger & Rosen SquashBusters Center", ["squashbusters"]),
    ("Behrakis Health Sciences Center", ["behrakis"]),
    ("Burstein Hall", ["burstein"]),
    ("Cabot Physical Education Center", ["cabot"]),
    ("Cahners Hall", ["cahners"]),
    ("Cargill Hall", ["cargill"]),
    ("Catholic Center", ["catholic center"]),
    ("Churchill Hall", ["churchill"]),
    ("Columbus Place and Alumni Center", ["columbus place", "alumni center"]),
    ("Cullinane Hall", ["cullinane"]),
    ("Curry Student Center", ["curry student center"]),
    ("Cushing Hall", ["cushing"]),
    ("Dana Research Center", ["dana research"]),
    ("Davenport Commons", ["davenport"]),
    ("Dockser Hall", ["dockser"]),
    ("Dodge Hall", ["dodge"]),
    ("East Village", ["east village"]),
    ("Egan Research Center", ["egan"]),
    ("Ell Hall", ["ell hall"]),
    ("Fenway Center", ["fenway center"]),
    ("Forsyth Building", ["forsyth building"]),
    ("Hastings Hall", ["hastings hall"]),
    ("Hayden Hall", ["hayden"]),
    ("Hillel-Frager", ["hillel"]),
    ("Holmes Hall", ["holmes"]),
    ("Hurtig Hall", ["hurtig"]),
    ("International Village", ["international village"]),
    ("Interdisciplinary Science and Engineering Complex (ISEC)", ["isec", "interdisciplinary science"]),
    ("Kariotis Hall", ["kariotis"]),
    ("Knowles Center", ["knowles"]),
    ("Lake Hall", ["lake hall"]),
    ("Latinx Student Cultural Center", ["latinx", "latino"]),
    ("Light Hall", ["light hall"]),
    ("Loftman Hall", ["loftman"]),
    ("Marino Recreation Center", ["marino"]),
    ("Meserve Hall", ["meserve"]),
    ("Mugar Life Sciences Building", ["mugar"]),
    ("Nightingale Hall", ["nightingale"]),
    ("O'Bryant African American Institute", ["african american institute"]),
    ("Renaissance Park", ["renaissance park"]),
    ("Richards Hall", ["richards hall"]),
    ("Robinson Hall", ["robinson hall"]),
    ("Ryder Hall", ["ryder"]),
    ("Shillman Hall", ["shillman"]),
    ("Smith Hall", ["smith hall"]),
    ("Snell Engineering Center", ["snell engineering"]),
    ("Snell Library", ["snell library"]),
    ("Speare Hall", ["speare"]),
    ("Stearns Center", ["stearns"]),
    ("Stetson East", ["stetson east"]),
    ("Stetson West", ["stetson west"]),
    ("West Village F", ["west village f"]),
    ("West Village G", ["west village g"]),
    ("West Village H", ["west village h"]),
    ("White Hall", ["white hall"]),
    ("Willis Hall", ["willis hall"]),
]

NAME_COLUMNS = ("name", "official_name", "alt_name", "short_name", "loc_name", "old_name")
GENERIC_WORDS = {"hall", "center", "centre", "building", "recreation", "student", "the", "of", "and"}


def tokens(text):
    #'O'Bryant Hall' -> ['o', 'bryant', 'hall'] .
    return re.findall(r"[a-z0-9]+", str(text).lower().replace("&", " and "))


def contains_run(words, run):
    n = len(run)
    return n > 0 and any(words[i:i + n] == run for i in range(len(words) - n + 1))


def find_by_key(key, names):
    #Match a short key such as 'Marino' or 'Curry' to a full building name.
    key_words = [w for w in tokens(key) if w not in GENERIC_WORDS] or tokens(key)
    for name in names:
        words = tokens(name)
        if all(w in words for w in key_words):
            return name
    return None


def distance_m(lat1, lon1, lat2, lon2):
    #Flat-earth distance in metres, accurate enough at campus scale.
    dy = (lat2 - lat1) * 110_540
    dx = (lon2 - lon1) * 111_320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def osm_to_neu_names(rows):
    #Updating building names fetched from OSM to the NEU names.
    found, missing = [], []
    for display, aliases in BUILDINGS:
        alias_runs = [tokens(a) for a in aliases]
        best = None
        for row in rows:
            if any(contains_run(tokens(n), run) for n in row["names"] for run in alias_runs):
                d = distance_m(*CAMPUS_CENTER, row["lat"], row["lon"])
                if d <= MAX_FROM_CENTER_M and (best is None or d < best[0]):
                    best = (d, row)
        if best:
            found.append({"name": display, "lat": round(best[1]["lat"], 6), "lon": round(best[1]["lon"], 6)})
        else:
            missing.append(display)
    return found, missing


def download_building_name_osm():
    #Every named building around the campus, as rows for osm_to_neu_names().
    import osmnx as ox
    gdf = ox.features_from_point(CAMPUS_CENTER, tags={"building": True}, dist=SEARCH_RADIUS_M)
    columns = [c for c in NAME_COLUMNS if c in gdf.columns]
    points = gdf.geometry.representative_point()
    rows = []
    for i in range(len(gdf)):
        names = []
        for c in columns:
            value = gdf.iloc[i][c]
            if isinstance(value, str):
                names += [part.strip() for part in value.split(";") if part.strip()]
        if names:
            rows.append({"names": names, "lat": float(points.iloc[i].y), "lon": float(points.iloc[i].x)})
    return rows


def load_buildings(cache_file, refresh=False):
    #Buildings with positions: from the cache file, or looked up in OpenStreetMap and cached.
    if os.path.exists(cache_file) and not refresh:
        with open(cache_file, encoding="utf-8") as f:
            buildings = json.load(f)
        print(f"Buildings: {len(buildings)} read from {cache_file}")
        return buildings
    try:
        rows = download_building_name_osm()
    except Exception as err:   # no internet, osmnx missing, OpenStreetMap busy ...
        print(f"Couldn't look up the buildings in OpenStreetMap ({err}).")
        print(f"Fix that and run again, or create {cache_file} by hand "
              f'as a list of {{"name": ..., "lat": ..., "lon": ...}}.')
        return []
    found, missing = osm_to_neu_names(rows)
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(found, f, indent=1, ensure_ascii=False)
    print(f"Buildings: found {len(found)} of {len(BUILDINGS)} in OpenStreetMap, saved to {cache_file}")
    if missing:
        print("  Not found (add them to the file by hand if you need them):")
        for name in missing:
            print("   -", name)
    return found


if __name__ == "__main__":
    load_buildings(os.path.join("data", "processed", "neu_buildings.json"), refresh=True)
