"""Checks which paper-described artefacts have NO producing code in the TCCM repo.
Supports findings: explainability-code-missing, csm-epoch-selection-missing, full-experiments-typo.
Read-only; greps the repo's *.py and *.sh for keywords and file existence. Writes a CSV summary."""
import os, re, csv, glob

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "ZhongLIFR__TCCM-NIPS")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

py_files = glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True)
sh_files = glob.glob(os.path.join(REPO, "**", "*.sh"), recursive=True)
code_files = [f for f in (py_files + sh_files) if "/datasets/" not in f]

def grep_count(patterns):
    """Return list of (file, lineno, line) where any regex pattern matches (case-insensitive)."""
    hits = []
    pats = [re.compile(p, re.I) for p in patterns]
    for f in code_files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if any(p.search(line) for p in pats):
                        hits.append((os.path.relpath(f, REPO), i, line.strip()))
        except Exception:
            pass
    return hits

rows = []

# 1. Explainability / feature-level attribution / Table 3 GMM explanation experiment
expl_hits = grep_count([r"exact.?match", r"jaccard", r"explanation", r"\battribut", r"top.?k.*feature", r"ground.?truth.*dim"])
rows.append(["explainability_attribution_code (Table 3, RQ3, Fig 1 synthetic)",
             "PRESENT" if expl_hits else "ABSENT",
             "; ".join(f"{f}:{n}" for f, n, _ in expl_hits[:5])])

# 2. Synthetic 2D contraction visualization (Figure 1)
fig1_hits = grep_count([r"contraction.*vector.*plot", r"quiver", r"2d.*synth", r"circles.*squares"])
rows.append(["figure1_2d_contraction_visualization",
             "PRESENT" if fig1_hits else "ABSENT",
             "; ".join(f"{f}:{n}" for f, n, _ in fig1_hits[:5])])

# 3. CSM / unsupervised epoch selection (Li et al. 2025b, Appendix B.6)
csm_hits = grep_count([r"\bCSM\b", r"contrast.?score", r"margin", r"select.*epoch", r"epoch.*select", r"unsupervised.*select"])
rows.append(["csm_unsupervised_epoch_selection (Appendix B.6)",
             "PRESENT" if csm_hits else "ABSENT",
             "; ".join(f"{f}:{n}" for f, n, _ in csm_hits[:5])])

# 4. Full_experiments.py referenced by run_knn.sh — does the file exist?
ref_typo = grep_count([r"Full_experiments\.py"])
file_exists = os.path.isfile(os.path.join(REPO, "Full_experiments.py"))
rows.append(["run_knn.sh references Full_experiments.py",
             "FILE_EXISTS" if file_exists else "FILE_MISSING",
             "; ".join(f"{f}:{n}" for f, n, _ in ref_typo)])

# 5. Dataset count vs paper's claim of 47
ds_count = 0
for grp in ("small", "medium", "high_dim", "large"):
    ds_count += len(glob.glob(os.path.join(REPO, "datasets", grp, "*.npz")))
rows.append(["dataset_npz_count (paper claims 47)", str(ds_count), "small+medium+high_dim+large"])

with open(os.path.join(OUT, "missing_artifacts.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["artefact", "status", "evidence"])
    w.writerows(rows)

for r in rows:
    print(r)
