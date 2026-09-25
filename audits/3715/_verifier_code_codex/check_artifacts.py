"""File-existence check for artefacts the experiment notebooks load/save.

Supports findings: vae-params-missing, no-reproduction-readme, no-saved-outputs.
Read-only. Run: cd _audit_code && python check_artifacts.py
"""
import json
import os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "samuel-howard__log_smoothing")
REPO = os.path.abspath(REPO)

# Paths referenced (loaded or saved) inside the notebooks.
referenced = {
    "params/vae_params.pkl": "loaded by mnist_KDE_vs_gaussian + mnist_compare_smoothing (VAE checkpoint)",
    "figures/": "every notebook savefig target",
    "samples/digit_4/gaussian_samples_mnist_digit_4.npy": "saved by mnist_compare, read by mnist_fid_computations",
    "samples/digit_4/fid_images.npy": "saved by mnist_compare, read by mnist_fid_computations",
}

rows = []
for rel, desc in referenced.items():
    p = os.path.join(REPO, rel)
    exists = os.path.exists(p)
    rows.append({"path": rel, "exists": exists, "role": desc})

# top-level inventory
top = sorted(os.listdir(REPO))

out = {"repo": REPO, "top_level": top, "referenced_artifacts": rows}
outpath = os.path.join(os.path.dirname(__file__), "out", "artifacts.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)

print("Top-level repo entries:", top)
print()
for r in rows:
    print(f"  exists={r['exists']!s:5}  {r['path']}   ({r['role']})")
print()
print("Wrote", outpath)
