#!/usr/bin/env python3
"""Checks (read-only) for the AMD repo audit: missing local modules/packages,
hardcoded absolute paths, missing RMSE/baseline computation. Supports findings
missing-cfgs-module, missing-parse-functools, missing-vision-score-module,
hardcoded-data-paths, missing-rmse-eval, missing-baselines, missing-colorlog-dep."""
import os, re, ast, sys, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
        "code", "kaist-ami__Automated-Model-Discovery"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

results = {}

def pyfiles():
    for root, _, files in os.walk(REPO):
        if ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)

# 1. Local modules / packages imported but absent from the repo.
# Map: module name -> expected local file/dir under REPO root.
local_module_candidates = {
    "parse_functools": "parse_functools.py",
    "vision_score_gpt2_0105_func": "vision_score_gpt2_0105_func.py",
    "cfgs": "cfgs",            # package dir
    "cfgs.asmd_cfg": os.path.join("cfgs", "asmd_cfg.py"),
}
present = {}
for mod, rel in local_module_candidates.items():
    p = os.path.join(REPO, rel)
    present[mod] = os.path.exists(p)
results["local_module_present"] = present

# Where are they imported?
import_sites = {m: [] for m in local_module_candidates}
for fp in pyfiles():
    rel = os.path.relpath(fp, REPO)
    with open(fp, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            s = line.strip()
            for m in local_module_candidates:
                base = m.split(".")[0]
                if re.match(rf"(from|import)\s+{re.escape(m)}\b", s) or \
                   re.match(rf"from\s+{re.escape(base)}\b", s):
                    if m == base or s.startswith(f"from {m}") or s.startswith(f"import {m}"):
                        import_sites[m].append(f"{rel}:{i}: {s}")
results["import_sites"] = import_sites

# 2. Hardcoded absolute paths (/home/mok, /node_data_2).
hard = []
for fp in pyfiles():
    rel = os.path.relpath(fp, REPO)
    with open(fp, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            if "/home/mok" in line or "/node_data_2" in line:
                hard.append(f"{rel}:{i}")
results["hardcoded_path_lines"] = len(hard)
results["hardcoded_path_sample"] = hard[:12]

# 2b. Does the load_gp_data path exist on this machine?
gp_path = "/home/mok/module/icml25/gpss-research/data/tsdlr_9010_csv/mok"
results["load_gp_data_path"] = gp_path
results["load_gp_data_path_exists"] = os.path.exists(gp_path)
results["repo_data_csv_count"] = len([f for f in os.listdir(os.path.join(REPO, "data"))
                                      if f.endswith("-train.csv")])

# 3. RMSE / r2_score defined but never called.
def count_calls(funcname):
    callers = []
    defs = []
    for fp in pyfiles():
        rel = os.path.relpath(fp, REPO)
        with open(fp, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if re.search(rf"\bdef\s+{funcname}\s*\(", line):
                    defs.append(f"{rel}:{i}")
                # a call is funcname( not preceded by def / word char
                for m in re.finditer(rf"(?<![\w.]){funcname}\s*\(", line):
                    if "def " not in line:
                        callers.append(f"{rel}:{i}")
    return defs, callers

for fn in ("rmse", "r2_score"):
    d, c = count_calls(fn)
    results[f"{fn}_defs"] = d
    results[f"{fn}_call_sites"] = c

# 4. Baseline implementations present?
baseline_kw = ["arima", "prophet", "boxlm", "pmdarima", "auto_arima",
               "statsmodels", "llm-sr", "llm_sr", "icsr", "sga"]
baseline_hits = {kw: [] for kw in baseline_kw}
for fp in pyfiles():
    rel = os.path.relpath(fp, REPO)
    if "prompts" in rel:   # prompts mention names but are not implementations
        continue
    with open(fp, encoding="utf-8", errors="replace") as fh:
        txt = fh.read().lower()
    for kw in baseline_kw:
        if kw in txt:
            baseline_hits[kw].append(rel)
results["baseline_impl_hits"] = baseline_hits

# 5. colorlog dependency present in requirements.txt?
req = os.path.join(REPO, "requirements.txt")
with open(req, encoding="utf-8", errors="replace") as fh:
    reqtxt = fh.read().lower()
imported_colorlog = any("colorlog" in open(fp, encoding="utf-8", errors="replace").read()
                        for fp in pyfiles())
results["colorlog_imported_in_code"] = imported_colorlog
results["colorlog_in_requirements"] = ("colorlog" in reqtxt)

with open(os.path.join(OUT, "imports_and_paths.json"), "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results, indent=2))
