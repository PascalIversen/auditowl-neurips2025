"""Checks repo for baseline code, ablation toggles, subsampling, fle_degree, and deps.
Supports findings: no-baselines-in-repo, no-ablation-code, missing-data-subsampling,
fle-degree-config-vs-paper, no-dependency-file. Read-only; writes a report to out/."""
import os, re, glob, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "nesl__GSRF")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

py_files = glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True)
yaml_files = glob.glob(os.path.join(REPO, "**", "*.yaml"), recursive=True)
sh_files = glob.glob(os.path.join(REPO, "**", "*.sh"), recursive=True)
all_text_files = py_files + yaml_files + sh_files + glob.glob(os.path.join(REPO, "*.md"))

def grep_count(patterns, files):
    hits = {}
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for p in patterns:
            for m in re.finditer(p, txt, re.IGNORECASE):
                hits.setdefault(p, []).append(os.path.relpath(f, REPO))
    return hits

report = {}

# 1. baseline implementations (paper compares vs NeRF2, WRF-GS, R2F2, FIRE)
baseline_pats = [r"\bNeRF2\b", r"\bWRF[-_ ]?GS\b", r"\bR2F2\b", r"\bFIRE\b"]
report["baseline_mentions"] = grep_count(baseline_pats, all_text_files)

# 2. ablation toggles: SH vs FLE, amplitude-only (no phase), no-fourier-loss switch
abl_pats = [r"spherical_harmonic", r"\buse_sh\b", r"\bsh_coeff", r"amplitude_only",
            r"no_phase", r"disable_phase", r"no_fourier", r"ablat"]
report["ablation_toggles"] = grep_count(abl_pats, py_files + yaml_files)

# 3. data subsampling for sparse RFID (220 instances) or CSI "30%"
sub_pats = [r"\b220\b", r"sparse", r"measurement.?density", r"0\.8.*ft", r"subsample.*train"]
report["subsampling"] = grep_count(sub_pats, py_files + yaml_files)

# 4. fle_degree values in configs
fle = {}
for f in yaml_files:
    txt = open(f).read()
    m = re.search(r"fle_degree:\s*([0-9]+)", txt)
    if m:
        deg = int(m.group(1))
        fle[os.path.relpath(f, REPO)] = {"fle_degree": deg, "num_coeffs_(deg+1)^2": (deg + 1) ** 2}
report["fle_degree_in_configs"] = fle
report["paper_claims_L3_16coeffs"] = {"L": 3, "num_coeffs": (3 + 1) ** 2}

# 5. dependency specification files
dep_files = []
for name in ["requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml",
             "setup.py", "setup.cfg", "Pipfile", "poetry.lock", "conda.yaml"]:
    if os.path.exists(os.path.join(REPO, name)):
        dep_files.append(name)
report["root_dependency_files"] = dep_files

# 6. pretrained weights present in repo?
weight_files = glob.glob(os.path.join(REPO, "**", "*.pth"), recursive=True) + \
               glob.glob(os.path.join(REPO, "weights", "**"), recursive=True)
report["weight_files_in_repo"] = [os.path.relpath(f, REPO) for f in weight_files]

# 7. timing/measurement code for training-time (Fig 4) and RFID inference latency (Fig 5)
report["time_module_used_in"] = [os.path.relpath(f, REPO) for f in py_files
                                  if re.search(r"\btime\.time\(|perf_counter", open(f).read())]

with open(os.path.join(OUT, "repo_completeness.json"), "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
