"""Verify countdown/code_repair pass@k mean estimator equals the standard Chen et al. unbiased estimator 1 - C(n-c,k)/C(n,k). Supports no finding on eval correctness."""
import numpy as np
from math import comb

def get_pass_at_k_mean(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def chen(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

rows=[]
ok=True
for n in [32, 128]:
    for c in range(0, n+1):
        for k in [1,2,4,8,16,32]:
            if k>n: continue
            a=get_pass_at_k_mean(n,c,k); b=chen(n,c,k)
            if abs(a-b)>1e-9:
                ok=False
                rows.append((n,c,k,a,b))
print("all_match:", ok, "n_mismatch:", len(rows))
for r in rows[:10]:
    print(r)
