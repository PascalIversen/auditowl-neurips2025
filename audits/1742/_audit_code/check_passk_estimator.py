"""Verify the pass@k estimator in math/pass@k.py and evaluate.py equals the
Chen et al. unbiased estimator 1 - C(n-c,k)/C(n,k) (paper Eq. 2). Supports the
'items-look-fine' note that the estimator matches the paper. Read-only."""
import numpy as np
from math import comb
import os

OUT = os.path.join(os.path.dirname(__file__), "out", "passk_estimator.txt")

def repo_estimator(n, c, k):
    # verbatim logic from math/pass@k.py:58-60 and evaluate.py:29-31
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def closed_form(n, c, k):
    # paper Eq. 2: 1 - C(n-c,k)/C(n,k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

lines = []
maxdiff = 0.0
for n in [8, 32, 128, 256, 1024, 2048]:
    for c in [0, 1, 2, n // 4, n // 2, n - 1, n]:
        for k in [1, 2, 16, 128, min(256, n), n]:
            if k > n:
                continue
            a = repo_estimator(n, c, k)
            b = closed_form(n, c, k)
            maxdiff = max(maxdiff, abs(a - b))
lines.append(f"max |repo_estimator - closed_form Eq.2| over grid = {maxdiff:.3e}")
lines.append("VERDICT: estimator matches paper Eq.2" if maxdiff < 1e-9 else "VERDICT: MISMATCH")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
