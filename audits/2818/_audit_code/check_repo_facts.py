#!/usr/bin/env python3
"""Deterministic checks supporting several findings: hardcoded paths, missing
deps in requirements.txt, stubbed test commands, and the unseen-task split
overlap (split.sh). Read-only; writes results to out/."""
import os, re, json, glob

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
        "NUS-HPC-AI-Lab__Recurrent-Parameter-Generation"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
res = {}

# 1) hardcoded /path/to placeholders
hits = []
for root, _, files in os.walk(REPO):
    if ".git" in root:
        continue
    for f in files:
        if f.endswith((".py", ".sh", ".json")):
            p = os.path.join(root, f)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                if "/path/to" in line:
                    hits.append((os.path.relpath(p, REPO), i, line.strip()))
res["hardcoded_path_to"] = hits

# 2) requirements.txt vs third-party imports actually used
req = open(os.path.join(REPO, "requirements.txt")).read().lower()
req_pkgs = set(l.strip().split("==")[0] for l in req.splitlines() if l.strip())
stdlib = {"os","sys","math","random","json","re","time","abc","pickle","warnings",
          "functools","itertools","collections","typing","_thread","argparse",
          "subprocess","glob","shutil","copy","datetime","pathlib"}
# map import name -> pip pkg name
alias = {"cv2":"opencv-python","PIL":"pillow","sklearn":"scikit-learn",
         "yaml":"pyyaml","mamba_ssm":"mamba-ssm"}
imports = set()
for root, _, files in os.walk(REPO):
    if ".git" in root: continue
    for f in files:
        if f.endswith(".py"):
            for line in open(os.path.join(root,f),encoding="utf-8",errors="ignore"):
                m = re.match(r"\s*(?:from|import)\s+([a-zA-Z0-9_]+)", line)
                if m:
                    imports.add(m.group(1))
third = sorted(i for i in imports if i not in stdlib and not os.path.isdir(os.path.join(REPO,i))
               and i not in {"model","dataset","workspace","train"})
# local module roots to exclude
local = {"model","dataset","workspace","train","denoiser","diffusion","mamba",
         "lstm","transformer","gatemlp","pdiff","register"}
third = [i for i in third if i not in local]
missing = []
for imp in third:
    pkg = alias.get(imp, imp).lower()
    if pkg not in req_pkgs:
        missing.append((imp, pkg))
res["third_party_imports"] = sorted(third)
res["requirements_listed"] = sorted(req_pkgs)
res["imports_missing_from_requirements"] = sorted(missing)
res["requirements_pinned"] = bool(re.search(r"==", req))

# 3) stubbed "coming soon" test commands in register.py
reg = open(os.path.join(REPO,"dataset","register.py")).read()
res["coming_soon_test_commands"] = re.findall(r'test_command = "echo .*?coming soon.*?"', reg)

# 4) split.sh overlap between held-out test classes and any class accidentally left in train
split = open(os.path.join(REPO,"dataset","condition_classinput_vittiny","split.sh")).read()
test_classes = re.findall(r"class(\d+)", split)
res["num_held_out_test_classes"] = len(test_classes)
res["held_out_test_classes"] = test_classes
res["held_out_unique"] = len(set(test_classes)) == len(test_classes)

with open(os.path.join(OUT,"repo_facts.json"),"w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
