"""Checks that cw_ssim.ipynb reports CW-SSIM/TEDS as the mean over NON-ZERO
(compile-successful) predictions only, supporting finding `cwssim-conditional-on-compile`.
Read-only: re-implements the exact averaging logic from the notebook on a tiny synthetic
score list and shows that failed (0.0) entries are dropped from the mean."""
import json, os, numpy as np

HERE = os.path.dirname(__file__)
NB = os.path.join(HERE, "..", "code", "newLLing__Table2LaTeX-RL", "cw_ssim.ipynb")

# Synthetic per-table CW-SSIM scores: 3 compiled (0.5,0.6,0.7), 2 failed (0.0,0.0)
scores = [0.5, 0.6, 0.7, 0.0, 0.0]

# --- replica of notebook calculate_average_and_ratio (cell 13) ---
total, valid, nonzero = 0.0, 0, 0
for s in scores:
    valid += 1
    if not np.isclose(s, 0.0):
        nonzero += 1
        total += s
avg_nonzero = total / nonzero if nonzero else 0.0
ratio = nonzero / valid if valid else 0.0
avg_all = sum(scores) / len(scores)

# confirm the notebook actually divides by non_zero_count, not valid_count
src = ""
nb = json.load(open(NB))
for c in nb["cells"]:
    s = "".join(c["source"])
    if "non_zero_count" in s and "avg = total_score" in s:
        src += s
divides_by_nonzero = "total_score / non_zero_count" in src

rows = [
    ["reported_CW-SSIM_(Average_Non-zero)", round(avg_nonzero, 4)],
    ["naive_mean_over_all_(Average_All)", round(avg_all, 4)],
    ["compile_ratio_(non_zero_ratio)", round(ratio, 4)],
    ["notebook_divides_by_non_zero_count", divides_by_nonzero],
]
out = os.path.join(HERE, "out", "nonzero_averaging.csv")
with open(out, "w") as f:
    f.write("metric,value\n")
    for k, v in rows:
        f.write(f"{k},{v}\n")
for k, v in rows:
    print(f"{k}: {v}")
print("wrote", out)
