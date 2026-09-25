"""Checks the CDF of the corruption rates lambda actually sampled by
experiments.ipynb (`lambd = 1 - np.exp(-np.random.exponential(q, N))`) against
the CDF stated in the paper's Appendix A, `F(t) = 1 - (1-t)**q`.
Supports finding `lambda-cdf-exponent-inverted`. Read-only; writes out/check_lambda_cdf.csv.
"""
import csv
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# The q grid actually swept by experiments.ipynb (cell at .ipynb line 331).
qs = 0.1 * np.exp(np.linspace(0, np.log(5 / 0.1), 20))
N = 200000
rng = np.random.default_rng(0)

rows = []
for q in qs:
    # verbatim reproduction of experiments.ipynb line 131 / 337
    lam = 1 - np.exp(-rng.exponential(q, N))
    emp_mean = lam.mean()
    t = 0.5
    emp_F = float(np.mean(lam <= t))          # empirical P(lambda <= 0.5)
    paper_F = 1 - (1 - t) ** q                # paper Appendix A: F(t)=1-(1-t)^q
    code_F = 1 - (1 - t) ** (1.0 / q)         # analytic CDF of 1-exp(-Exp(scale=q))
    # mean of lambda under each CDF: E[lam] = 1 - 1/(1+a) for F(t)=1-(1-t)^{1/a}? use numeric
    rows.append(
        dict(
            q=round(float(q), 4),
            emp_mean_lambda=round(emp_mean, 4),
            emp_F_at_0p5=round(emp_F, 4),
            paper_F_at_0p5=round(float(paper_F), 4),
            code_analytic_F_at_0p5=round(float(code_F), 4),
            matches_paper=bool(abs(emp_F - paper_F) < 0.01),
            matches_1_over_q=bool(abs(emp_F - code_F) < 0.01),
        )
    )

path = os.path.join(OUT, "check_lambda_cdf.csv")
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(r)
print()
print("q grid swept:", np.round(qs, 3).tolist())
print("mean(lambda) at q=0.1  :", rows[0]["emp_mean_lambda"])
print("mean(lambda) at q=5.0  :", rows[-1]["emp_mean_lambda"])
print("=> corruption INCREASES with q in the code:",
      rows[-1]["emp_mean_lambda"] > rows[0]["emp_mean_lambda"])
print()
print("Under the paper's stated F(t)=1-(1-t)^q, E[lambda] = 1/(q+1):")
print("  q=0.1 -> E[lambda] =", round(1 / 1.1, 4),
      " ; q=5.0 -> E[lambda] =", round(1 / 6.0, 4))
print("=> corruption DECREASES with q under the paper's stated CDF.")
print()
print("rows matching paper CDF   :", sum(r["matches_paper"] for r in rows), "/", len(rows))
print("rows matching 1-(1-t)^(1/q):", sum(r["matches_1_over_q"] for r in rows), "/", len(rows))
print("wrote", path)
