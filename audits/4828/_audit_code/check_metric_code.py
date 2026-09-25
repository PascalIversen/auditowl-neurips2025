"""Checks whether the repo contains any code that COMPUTES the paper's eval
metrics (FID, ID-similarity, LPIPS as a metric) -- supports finding
'no-metric-evaluation-code'. Read-only grep over the WarpGAN main code tree
(excludes vendored editings/ and arcface eval helpers)."""
import os, re, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "Huang-KT__WarpGAN")
REPO = os.path.abspath(REPO)

# Tokens that would indicate metric COMPUTATION (not training-loss usage).
patterns = {
    "FID/Frechet/Inception": re.compile(r"\b(fid|frechet|inception_v3|InceptionV3|pytorch_fid|cleanfid|FrechetInceptionDistance)\b", re.I),
    "ID-similarity-metric": re.compile(r"\b(id_sim|id_similarity|identity_sim|cosine_sim)\b", re.I),
    "LPIPS-as-metric (eval loop)": re.compile(r"\b(lpips_metric|compute_lpips|lpips_score)\b", re.I),
    "generic-metric-aggregation": re.compile(r"\b(evaluate_metrics|compute_metrics|eval_fid|run_eval)\b", re.I),
}

# Exclude vendored / unrelated subtrees.
EXCLUDE_DIRS = {".git", "editings"}
ARCFACE_EVAL = "arcface_torch"  # IJB-C eval helper, unrelated to paper metrics

hits = {k: [] for k in patterns}
py_files = 0
for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        rel = os.path.relpath(path, REPO)
        if ARCFACE_EVAL in rel:
            continue
        py_files += 1
        try:
            txt = open(path, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for name, pat in patterns.items():
            for m in pat.finditer(txt):
                line = txt[:m.start()].count("\n") + 1
                hits[name].append(f"{rel}:{line}:{m.group(0)}")

out = {"py_files_scanned": py_files, "hits": hits,
       "any_metric_computation_found": any(v for v in hits.values())}
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "out", "metric_code.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
