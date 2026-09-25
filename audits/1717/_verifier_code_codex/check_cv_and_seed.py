"""Verify the '10-fold cross validation' claim and seeding of the 10 runs.

Supports findings: cv-is-random-resample-not-kfold, runs-unseeded.
Read-only static inspection of datasets/nsd.py and main.py. Writes out/cv_seed.txt.
"""
import os, re

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "Hosseinadeli__transformer_brain_encoder")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

nsd = open(os.path.join(REPO, "datasets", "nsd.py")).read()
mainpy = open(os.path.join(REPO, "main.py")).read()

lines = []

# 1. Is there any KFold / fold-index logic?  Look for non-overlapping fold partitioning.
has_kfold = bool(re.search(r"kfold|fold_idx|n_splits|cross_val", nsd + mainpy, re.I))
lines.append(f"KFold/fold-partition primitive present anywhere: {has_kfold}")

# 2. The split is a single shuffled 90/10; runs differ only by RNG state.
m = re.search(r"num_train = int\(np\.round\(len\(train_img_list\) / 100 \* 90\)\)", nsd)
lines.append(f"90/10 split line present: {bool(m)}")
m2 = re.search(r"if args\.run < 20:\s*\n\s*np\.random\.shuffle\(idxs\)", nsd)
lines.append(f"per-run np.random.shuffle(idxs) (no seed) present: {bool(m2)}")

# 3. Seeds: are np/torch seeds set anywhere active (not commented)?
active_seed = []
for fn, txt in [("main.py", mainpy), ("datasets/nsd.py", nsd)]:
    for i, ln in enumerate(txt.splitlines(), 1):
        s = ln.strip()
        if re.search(r"seed", s, re.I):
            commented = s.startswith("#")
            active_seed.append(f"    {fn}:{i}: {'(commented) ' if commented else '(ACTIVE) '}{s[:80]}")
lines.append("seed-related lines:")
lines.extend(active_seed)
active = [a for a in active_seed if "(ACTIVE)" in a]
lines.append(f"=> number of ACTIVE seed statements: {len(active)}")

# 4. Confirm the 10 runs are averaged (ensemble), not folds, in the eval notebook.
nb = ""
import json, glob
for p in glob.glob(os.path.join(REPO, "*.ipynb")):
    d = json.load(open(p))
    for c in d.get("cells", []):
        if c.get("cell_type") == "code":
            nb += "".join(c.get("source", [])) + "\n"
runs_pat = r"self.runs = np.arange.1, *11"
mean_pat = r"np\.mean\(lh_pred"
lines.append(f"eval notebook sets self.runs = np.arange(1,11): "
             f"{bool(re.search(runs_pat, nb))}")
lines.append(f"eval notebook averages predictions across runs (np.mean(lh_pred...)): "
             f"{bool(re.search(mean_pat, nb))}")

report = "\n".join(lines)
print(report)
open(os.path.join(OUT, "cv_seed.txt"), "w").write(report + "\n")
