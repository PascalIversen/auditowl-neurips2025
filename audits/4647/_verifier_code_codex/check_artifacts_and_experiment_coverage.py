#!/usr/bin/env python3
"""Checks which paper experiments/artefacts are present in the cloned Dynam3D repo.

Deterministic checks supporting findings: missing-vln-checkpoints,
missing-released-weights, missing-reverie-navrag-eval, missing-preexplore-lifelong,
missing-noise-robustness-code, missing-ablation-toggles, missing-realworld-protocol.
Read-only. Saves a CSV to out/.
"""
import os, re, csv, glob

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "MrZihan__Dynam3D")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def walk_text_files():
    for root, dirs, files in os.walk(REPO):
        if "/.git" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith((".py", ".yaml", ".yml", ".bash", ".md", ".txt", ".json")):
                yield os.path.join(root, f)

texts = {}
for p in walk_text_files():
    try:
        with open(p, "r", errors="ignore") as fh:
            texts[p] = fh.read()
    except Exception:
        pass

def grep_any(patterns):
    hits = []
    for p, t in texts.items():
        for pat in patterns:
            if re.search(pat, t):
                hits.append((os.path.relpath(p, REPO), pat))
                break
    return hits

# 1. Released weight files actually present in repo
weight_files = []
for ext in ("*.pth", "*.pt", "*.ckpt", "*.bin", "*.safetensors"):
    for p in glob.glob(os.path.join(REPO, "**", ext), recursive=True):
        if "/.git/" not in p:
            weight_files.append(os.path.relpath(p, REPO))

# 2. Data files (gz) that are navigation datasets present in repo
data_gz = [os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "**", "*.json.gz"), recursive=True) if "/.git/" not in p]

rows = []
rows.append(("released_weight_files_in_repo", len(weight_files), ";".join(weight_files) or "NONE"))
rows.append(("nav_data_json_gz_in_repo", len(data_gz), ";".join(data_gz) or "NONE"))

# 3. REVERIE/NavRAG evaluation wiring (not just data conversion)
reverie_navrag = grep_any([r"REVERIE", r"reverie", r"NavRAG", r"navrag"])
reverie_navrag = [h for h in reverie_navrag if not h[0].startswith("discrete_to_CE") and not h[0].endswith("README.md")]
rows.append(("reverie_navrag_outside_conversion_and_readme", len(reverie_navrag), ";".join(f"{a}" for a, b in reverie_navrag) or "NONE"))

# 4. Pre-exploration / lifelong memory code
pre_life = grep_any([r"pre[-_]?explor", r"preexplor", r"[Ll]ifelong", r"lifelong_memory"])
pre_life = [h for h in pre_life if not h[0].endswith("README.md")]
rows.append(("preexplore_or_lifelong_code", len(pre_life), ";".join(a for a, b in pre_life) or "NONE"))

# 5. SLAM / depth noise robustness (Table 7)
noise = grep_any([r"SLAM", r"\bslam_noise", r"localization noise", r"orientation noise", r"depth_noise"])
rows.append(("slam_or_depth_noise_injection_code", len(noise), ";".join(a for a, b in noise) or "NONE"))

# 6. Real-world / Stretch robot protocol (Tables 4,5)
realworld = grep_any([r"Stretch", r"\bstretch\b", r"hello[_ ]robot", r"\brospy\b", r"real[-_]world"])
realworld = [h for h in realworld if not h[0].endswith("README.md") and "ultralytics" not in h[0]]
rows.append(("realworld_robot_code", len(realworld), ";".join(a for a, b in realworld) or "NONE"))

# 7. Ablation toggles in VLN model/config (Table 6 rows)
ablation = grep_any([r"use_instance", r"use_zone", r"use_pano", r"single_view", r"use_subspace", r"no_instance", r"no_zone"])
ablation = [h for h in ablation if "Dynam3D_VLN" in h[0]]
rows.append(("vln_ablation_toggles", len(ablation), ";".join(a for a, b in ablation) or "NONE"))

# 8. Checkpoint paths referenced by run scripts (to show they are not shipped)
ckpt_refs = grep_any([r"ckpt\.iter8000\.pth", r"ckpt\.iter12000\.pth", r"dynam3d\.pth", r"model_step_100000\.pt"])
rows.append(("checkpoint_paths_referenced", len(ckpt_refs), ";".join(sorted(set(a for a, b in ckpt_refs)))))

with open(os.path.join(OUT, "experiment_coverage.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["check", "count", "detail"])
    for r in rows:
        w.writerow(r)
        print(f"{r[0]}: count={r[1]} | {r[2]}")
