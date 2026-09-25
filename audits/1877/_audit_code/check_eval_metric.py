"""Checks whether eval.py computes the paper's distance-threshold (% @ km) metric:
greps for any geocoding (Azure Maps) or haversine/great-circle distance code.
Supports finding eval-distance-metric-missing."""
import os, re, csv

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "lingli1996__GLOBE")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out", "eval_metric.csv")

eval_path = os.path.join(REPO, "examples/train/grpo/globe/eval.py")
with open(eval_path) as f:
    text = f.read()

terms = ["azure", "geocode", "geocoding", "haversine", "great_circle",
         "geopy", "nominatim", "radians", "atan2", "math.cos", "math.sin",
         "1km", "25km", "200km", "750km", "2500km", "distance"]
rows = []
for t in terms:
    n = len(re.findall(re.escape(t), text, re.IGNORECASE))
    rows.append((t, n))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["term", "occurrences_in_eval.py"])
    w.writerows(rows)

for t, n in rows:
    print(f"{t:14s} {n}")

# Also check whole GLOBE code dir
globe_dir = os.path.join(REPO, "examples/train/grpo/globe")
hit = []
for fn in os.listdir(globe_dir):
    if not (fn.endswith(".py") or fn.endswith(".sh")):
        continue
    with open(os.path.join(globe_dir, fn)) as f:
        body = f.read()
    for t in ["azure", "haversine", "great_circle", "geopy"]:
        if re.search(re.escape(t), body, re.IGNORECASE):
            hit.append((fn, t))
print("\nGeocode/distance hits across globe/ dir:", hit if hit else "NONE")
