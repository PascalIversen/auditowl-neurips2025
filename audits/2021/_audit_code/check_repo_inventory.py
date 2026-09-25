#!/usr/bin/env python3
"""Inventory the released RESAnything repo: list every .py, grep for an
eval/benchmark/metric harness, dataset loaders, and the README disclaimer.
Supports findings: no-eval-harness, abo-ares-dataset-missing, reimplementation-disclaimer."""
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "suikei-wang__RESAnything"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

lines = []
def log(s=""):
    lines.append(s)
    print(s)

# 1. List all python files
py_files = []
for root, _, files in os.walk(REPO):
    if "/.git" in root:
        continue
    for f in sorted(files):
        if f.endswith(".py"):
            py_files.append(os.path.relpath(os.path.join(root, f), REPO))
log("=== Python files in repo ===")
for f in sorted(py_files):
    log(f"  {f}")
log(f"total .py files: {len(py_files)}")
log("")

# 2. Grep all python source for evaluation/metric/benchmark tokens
EVAL_TOKENS = ["iou", "gIoU", "cIoU", "mIoU", "intersection", "union",
               "ground_truth", "gt_mask", "annotation", "refcoco", "reasonseg",
               "abo", "coco-tasks", "cocotask", "evaluate", "evaluation",
               "metric", "benchmark", "score_iou", "compute_iou", "Precision",
               "recall", "pr@"]
log("=== Eval/metric/benchmark token hits across .py source ===")
hits = {}
for f in py_files:
    with open(os.path.join(REPO, f), "r", errors="replace") as fh:
        src = fh.read()
    for tok in EVAL_TOKENS:
        for m in re.finditer(re.escape(tok), src, flags=re.IGNORECASE):
            ln = src.count("\n", 0, m.start()) + 1
            hits.setdefault(tok.lower(), []).append(f"{f}:{ln}")
if hits:
    for tok, locs in sorted(hits.items()):
        log(f"  {tok}: {sorted(set(locs))}")
else:
    log("  (none) -- NO iou/metric/benchmark/eval code found in any .py file")
log("")

# 3. README disclaimer + dataset-release promise (verbatim grep)
readme = os.path.join(REPO, "README.md")
with open(readme, "r", errors="replace") as fh:
    rlines = fh.readlines()
log("=== README disclaimer / dataset lines (verbatim) ===")
for i, l in enumerate(rlines, 1):
    low = l.lower()
    if ("re-implementation" in low or "protected license" in low
            or "may not be the original" in low or "release abo-ares" in low
            or "abo-ares dataset" in low or "asap" in low):
        log(f"  README.md:{i}: {l.rstrip()}")
log("")

# 4. Does any file load a benchmark dataset / produce a results table?
log("=== Driver entrypoints (if __main__) ===")
for f in py_files:
    with open(os.path.join(REPO, f), "r", errors="replace") as fh:
        src = fh.read()
    if "__main__" in src:
        log(f"  {f} has __main__")
log("")

with open(os.path.join(OUT, "repo_inventory.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
log(f"[written] {os.path.join(OUT, 'repo_inventory.txt')}")
