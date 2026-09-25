#!/usr/bin/env python3
"""Inventories the repo for the artefacts needed to reproduce Table 1
(FID/FID-random over 16 configs): training entrypoint, dataset/preprocessing,
metric-running driver (calc_metrics), config files, and per-variant generators.
Supports findings 'no-training-code', 'no-metric-driver', 'missing-variants'.
READ-ONLY: only lists/greps files."""
import os, re, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
       "code", "aigc3d__HyPlaneHead"))

py_files = []
for root, _dirs, files in os.walk(REPO):
    if os.sep + ".git" in root:
        continue
    for fn in files:
        if fn.endswith(".py"):
            py_files.append(os.path.join(root, fn))

def grep(pat):
    rx = re.compile(pat)
    hits = []
    for f in py_files:
        for i, line in enumerate(open(f, errors="ignore"), 1):
            if rx.search(line):
                hits.append(f"{os.path.relpath(f, REPO)}:{i}")
    return hits

checks = {
    "train_entrypoint (training_loop)": grep(r"def training_loop"),
    "train.py file exists": [os.path.relpath(f, REPO) for f in py_files
                             if os.path.basename(f) == "train.py"],
    "calc_metrics driver (argparse/click main calling calc_metric)":
        grep(r"calc_metric\("),
    "dataset class / ImageFolderDataset": grep(r"class .*Dataset|ImageFolderDataset|get_label"),
    "FID-random 'mode' plumbing in metric_utils": grep(r"mode == 'back'|opts\.mode"),
    "config files (*.json/*.yaml in repo root)":
        [f for f in os.listdir(REPO) if f.endswith((".json", ".yaml", ".yml")) and f != "environment.yml"],
    "generator classes": grep(r"class \w*Generator\b"),
    "area-bias / elongated split impl (renderer only)":
        [h for h in grep(r"area.?bias|elongat|spherical_cap|area_bias")
         if "renderer.py" in h or "triplane.py" in h],
    "unify-split impl (always even H//2)": grep(r"split_H, split_W = H // 2, W // 2"),
    "2+2 (two spherical) generator": grep(r"2\+2|2plus2|TwoSph|dual.*planar"),
}

result = {k: v for k, v in checks.items()}
result["VERDICT"] = {
    "training_code_present": bool(checks["train_entrypoint (training_loop)"] or checks["train.py file exists"]),
    "metric_driver_present": bool(checks["calc_metrics driver (argparse/click main calling calc_metric)"]),
    "dataset_code_present": bool(checks["dataset class / ImageFolderDataset"]),
    "config_files_present": bool(checks["config files (*.json/*.yaml in repo root)"]),
    "area_bias_split_impl_present": bool(checks["area-bias / elongated split impl (renderer only)"]),
    "two_plus_two_generator_present": bool(checks["2+2 (two spherical) generator"]),
}

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "out", "repo_completeness.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
