#!/usr/bin/env python3
"""Sanity-checks Table 6 'P-Value = 0.0' against floors for n=10 vs n=10 runs.

Supports the statistical-integrity note on Table 6: with 10 runs per method,
report whether a p-value rounding/printing as exactly 0.0 is achievable for the
common two-sample tests (Mann-Whitney U two-sided minimum, and a two-sample
t-test estimate from the reported mean/std). This is a paper-only consistency
check; no author code exists to verify which test was actually used.
"""
import math
import os
from itertools import combinations

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

lines = []

# Mann-Whitney U exact two-sided minimum p for n=m=10 (perfect separation).
n = m = 10
# number of ways to choose ranks; min U = 0 has exactly 1 arrangement of the
# combined ordering that gives complete separation among C(n+m, n) equally
# likely rank assignments under H0.
total = math.comb(n + m, n)
p_one_sided_min = 1.0 / total
p_two_sided_min = 2.0 / total
lines.append(f"Mann-Whitney U, n=m=10: C(20,10)={total}")
lines.append(f"  min one-sided p = {p_one_sided_min:.3e}")
lines.append(f"  min two-sided p = {p_two_sided_min:.3e}")
lines.append(f"  rounds to 0.0 at 1 decimal? {round(p_two_sided_min,1)==0.0}")
lines.append(f"  EXACTLY zero achievable? {p_two_sided_min==0.0}")

# Welch t-test estimate for S1: MetaPoint 31.5+/-0.61 vs Ours 70.4+/-0.53, n=10
def welch_t(m1, s1, m2, s2, n1, n2):
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m2 - m1) / se
    df = (s1**2/n1 + s2**2/n2)**2 / (
        (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    )
    return t, df

t, df = welch_t(31.5, 0.61, 70.4, 0.53, 10, 10)
lines.append("")
lines.append(f"S1 Welch t-test estimate: t={t:.2f}, df={df:.1f} (huge separation)")
lines.append("  -> p astronomically small; printing as 0.0 is plausible, NOT impossible.")

report = "\n".join(lines) + "\n"
print(report, end="")
with open(os.path.join(OUT, "table6_pvalue_floor.txt"), "w") as fh:
    fh.write(report)
