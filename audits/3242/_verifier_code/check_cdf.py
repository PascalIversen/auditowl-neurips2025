"""Checks: does the lambda sampled in experiments.ipynb follow the paper's
CDF F(t)=1-(1-t)^q, or F(t)=1-(1-t)^(1/q)?  Supports finding `lambda-cdf-exponent-mismatch`.

Reproduces the exact sampling line from experiments.ipynb cell 3/7:
    lambd = 1 - np.exp(-np.random.exponential(x, N))
with x == q (the value plotted on the figure x-axis), and compares the
empirical CDF at a probe point t against both candidate closed forms.
Read-only; does not import or modify the repo notebooks.
"""
import numpy as np
import csv, os

np.random.seed(0)
N = 2_000_000        # large sample for a tight empirical CDF
t = 0.5              # probe point in (0,1)
outdir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(outdir, exist_ok=True)

rows = []
# q grid identical in spirit to the notebook: 0.1 .. 5
for q in [0.1, 0.5, 1.0, 2.0, 5.0]:
    lambd = 1 - np.exp(-np.random.exponential(q, N))   # exact notebook line (x==q)
    emp = np.mean(lambd <= t)                           # empirical F(t)
    paper = 1 - (1 - t) ** q                            # paper's stated formula
    recip = 1 - (1 - t) ** (1.0 / q)                    # reciprocal-exponent formula
    mean_lambda = float(np.mean(lambd))
    rows.append(dict(q=q, empirical_F=round(emp, 4),
                     paper_1_minus_1mt_pow_q=round(paper, 4),
                     recip_1_minus_1mt_pow_1overq=round(recip, 4),
                     mean_lambda=round(mean_lambda, 4)))

with open(os.path.join(outdir, "cdf_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"probe t={t}")
print(f"{'q':>5} {'empF(t)':>9} {'paper q':>9} {'1/q form':>9} {'E[lambda]':>10}")
for r in rows:
    print(f"{r['q']:>5} {r['empirical_F']:>9} {r['paper_1_minus_1mt_pow_q']:>9} "
          f"{r['recip_1_minus_1mt_pow_1overq']:>9} {r['mean_lambda']:>10}")

# verdict
emp = np.array([r['empirical_F'] for r in rows])
paper = np.array([r['paper_1_minus_1mt_pow_q'] for r in rows])
recip = np.array([r['recip_1_minus_1mt_pow_1overq'] for r in rows])
print()
print("max|emp-paper|  =", round(float(np.max(np.abs(emp - paper))), 4))
print("max|emp-recip|  =", round(float(np.max(np.abs(emp - recip))), 4))
print("E[lambda] increases with q? ",
      all(rows[i]['mean_lambda'] < rows[i+1]['mean_lambda'] for i in range(len(rows)-1)))
