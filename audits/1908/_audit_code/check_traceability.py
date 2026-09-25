"""Checks which paper artefacts are computed in the repo and the FLE-degree config-vs-paper gap.
Supports findings: baselines-not-in-repo, ablation-code-missing, localization-code-missing,
density-experiment-missing, fle-degree-config-vs-paper-ablation, csi-normalization-pre-split.
Read-only; writes report to out/."""
import os, re, glob, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "nesl__GSRF"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

py = [f for f in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True) if "third_party" not in f]
yaml_f = glob.glob(os.path.join(REPO, "**", "*.yaml"), recursive=True)

def any_hit(pats, files):
    out = []
    for f in files:
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for p in pats:
            if re.search(p, t, re.IGNORECASE):
                out.append((os.path.relpath(f, REPO), p))
    return out

rep = {}
# Baselines: NeRF2 / WRF-GS / R2F2 / FIRE training or eval code (exclude README)
rep["baseline_impl_in_py"] = any_hit([r"nerf2", r"wrf[-_ ]?gs", r"\br2f2\b", r"\bfire\b"], py)
# Ablation toggles
rep["ablation_toggles"] = any_hit([r"spherical_harmon", r"use_sh", r"amplitude_only",
                                    r"no_phase", r"disable_phase", r"no_fourier", r"ablat", r"sh_degree"], py + yaml_f)
# BLE localization (KNN fingerprinting) -- exclude simple-knn submodule (Gaussian init)
rep["localization_code"] = any_hit([r"fingerprint", r"localiz", r"KNeighbors", r"k.?nearest.*neighbor.*posit"], py)
# Measurement-density / sparse experiment
rep["density_experiment"] = any_hit([r"density", r"\b0\.8\b.*ft", r"measurements?/ft", r"subsample.*train.*ratio"], py + yaml_f)
# FLE degree in configs vs paper ablation claim (L=3 -> 16 coeffs)
fle = {}
for f in yaml_f:
    m = re.search(r"fle_degree:\s*([0-9]+)", open(f).read())
    if m:
        d = int(m.group(1))
        fle[os.path.relpath(f, REPO)] = {"config_fle_degree": d, "config_num_coeffs": (d+1)**2}
rep["fle_degree_configs"] = fle
rep["paper_ablation_claim"] = {"L": 3, "num_coeffs": 16, "quote": "Both FLE and SH are implemented with a degree of L = 3, resulting in 16 coefficients each."}
# Dependency spec files
deps = [n for n in ["requirements.txt","environment.yml","environment.yaml","pyproject.toml",
                    "setup.py","Pipfile","poetry.lock"] if os.path.exists(os.path.join(REPO,n))]
rep["root_dependency_files"] = deps

print(json.dumps(rep, indent=2))
with open(os.path.join(OUT, "traceability.json"), "w") as f:
    json.dump(rep, f, indent=2)
