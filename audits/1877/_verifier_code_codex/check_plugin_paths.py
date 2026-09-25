"""Checks that --external_plugins / --custom_register_path paths in the GLOBE
train scripts actually exist (supports finding train-scripts-broken-plugin-path)."""
import os, re, glob, csv

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "lingli1996__GLOBE")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out", "plugin_paths.csv")

scripts = glob.glob(os.path.join(REPO, "examples/train/grpo/globe/*.sh"))
rows = []
flag_re = re.compile(r"--(external_plugins|custom_register_path)\s+(\S+)")
for s in sorted(scripts):
    with open(s) as f:
        text = f.read()
    for m in flag_re.finditer(text):
        flag, path = m.group(1), m.group(2)
        abspath = os.path.join(REPO, path)
        exists = os.path.exists(abspath)
        rows.append((os.path.relpath(s, REPO), flag, path, exists))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["script", "flag", "referenced_path", "exists"])
    w.writerows(rows)

for r in rows:
    print(r)
print("\nReferenced 'geo/' dir exists:",
      os.path.exists(os.path.join(REPO, "examples/train/grpo/geo")))
print("Actual files live under 'globe/':",
      os.path.exists(os.path.join(REPO, "examples/train/grpo/globe/plugin.py")))
