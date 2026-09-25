#!/usr/bin/env python3
"""Read-only checks for AlphaZero-Zipf audit: missing paths.yaml, the undocumented
'../matches/' dependency (Figs 3B & 5A), and presence of plot_data/models in repo.
Supports findings: missing-paths-yaml, matches-dir-undocumented, data-not-in-repo."""
import os
import re

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "OrenNeumann__alphazero_zipfs_law")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

lines = []
def log(s):
    print(s); lines.append(s)

# 1. paths.yaml that general_utils.py opens
paths_yaml = os.path.join(REPO, "src/config/paths.yaml")
example = os.path.join(REPO, "src/config/example_paths.yaml")
log(f"[paths.yaml] exists={os.path.exists(paths_yaml)}  (code opens 'src/config/paths.yaml')")
log(f"[example_paths.yaml] exists={os.path.exists(example)}")
gu = open(os.path.join(REPO, "src/general/general_utils.py")).read()
log(f"[general_utils opens paths.yaml] count={gu.count('src/config/paths.yaml')}")
readme = open(os.path.join(REPO, "README.md")).read()
log(f"[README mentions paths.yaml] {'paths.yaml' in readme}")
log(f"[README mentions example_paths] {'example_paths' in readme}")

# 2. '../matches/' references (raw Elo tournament matrices)
log("\n[../matches/ references in src]")
for root, _, files in os.walk(os.path.join(REPO, "src")):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        for i, line in enumerate(open(p), 1):
            if "../matches" in line:
                rel = os.path.relpath(p, REPO)
                log(f"  {rel}:{i}: {line.strip()}")
log(f"[README mentions a 'matches' folder] {'matches' in readme.lower()}")

# 3. plot_data / models present in repo?
log("\n[data dirs in repo]")
for d in ["plot_data", "models", "../plot_data", "../matches"]:
    log(f"  {d}: exists={os.path.exists(os.path.join(REPO, d))}")
# Count .pkl/.npz data artifacts shipped (excluding config .npy)
data_files = []
for root, _, files in os.walk(REPO):
    if ".git" in root:
        continue
    for fn in files:
        if fn.endswith((".pkl", ".npz")):
            data_files.append(os.path.relpath(os.path.join(root, fn), REPO))
log(f"[shipped .pkl/.npz data files in repo] {len(data_files)}: {data_files}")

with open(os.path.join(OUT, "artifacts.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote out/artifacts.txt")
