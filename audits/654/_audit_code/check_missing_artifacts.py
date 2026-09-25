"""Checks existence of the result artifacts the notebook needs to regenerate
every figure/table (res/all_experiments_*.parquet/.npz) and greps the whole
repo for a Wasserstein-distance computation that the paper reports.
Supports findings res-results-missing and wasserstein-not-computed."""
import os, glob, subprocess

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "alessiomarta__transformers_equivalence_classes")

needed = [
    "res/all_experiments_results.parquet",
    "res/all_experiments_embeddings.npz",
    "res/all_experiments_embeddings_input.npz",
]
out = ["Result files the notebook (notebooks/plots_and_tables.ipynb) reads:"]
for n in needed:
    p = os.path.join(REPO, n)
    out.append(f"  {'EXISTS' if os.path.exists(p) else 'MISSING'}: {n}")

out.append("")
out.append("Any .parquet / res dir present in repo:")
res_dir = os.path.join(REPO, "res")
out.append(f"  res/ dir exists: {os.path.isdir(res_dir)}")
parquets = glob.glob(os.path.join(REPO, "**", "*.parquet"), recursive=True)
parquets = [os.path.relpath(p, REPO) for p in parquets]
out.append(f"  *.parquet files in repo: {parquets}")

out.append("")
out.append("Grep for 'wasserstein' anywhere in repo (case-insensitive):")
res = subprocess.run(["grep", "-rni", "wasserstein", REPO],
                     capture_output=True, text=True)
hits = [l for l in res.stdout.splitlines() if "/.git/" not in l]
out.append(f"  hits (excluding .git): {len(hits)}")
for h in hits:
    out.append("    " + h)

text = "\n".join(out)
print(text)
with open(os.path.join(os.path.dirname(__file__), "out", "missing_artifacts.txt"), "w") as f:
    f.write(text + "\n")
