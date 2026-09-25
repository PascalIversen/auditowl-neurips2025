"""Checks recalx.calibration.ece_kl against the paper's claimed estimator.

Supports finding `ece-kl-not-paper-estimator`. The paper (Sec.5) states the
KL-divergence calibration error is computed with the consistent, asymptotically
unbiased estimator of Popordanoska et al. [51]. The repo's ece_kl is instead a
15-bin top-1 confidence binning that collapses K classes to a binary
[1-conf, conf] distribution. This script demonstrates (a) it is bin-based, not
kernel-based, and (b) it ignores the full K-dim predicted distribution.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "thomdeck__recalx"))
import numpy as np
from recalx.calibration import ece_kl

rng = np.random.default_rng(0)
n, K = 5000, 10
# random logits -> probs
logits = rng.normal(size=(n, K))
probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
labels = rng.integers(0, K, size=n)

val = ece_kl(probs, labels, n_bins=15)
val_b5 = ece_kl(probs, labels, n_bins=5)
val_b30 = ece_kl(probs, labels, n_bins=30)

# Demonstrate it only depends on top-1 confidence + correctness, not the full vector.
# Permute the non-top probabilities arbitrarily; ece_kl must be unchanged.
probs2 = probs.copy()
order = np.argsort(probs2, axis=1)
top = order[:, -1]
# scramble the non-argmax entries while preserving their sum and the max
for i in range(n):
    nontop = [j for j in range(K) if j != top[i]]
    vals = probs2[i, nontop]
    rng.shuffle(vals)
    probs2[i, nontop] = vals
val_scrambled = ece_kl(probs2, labels, n_bins=15)

out = os.path.join(os.path.dirname(__file__), "out", "ece_kl.txt")
with open(out, "w") as f:
    f.write(f"ece_kl(n_bins=15)={val:.6f}\n")
    f.write(f"ece_kl(n_bins=5)={val_b5:.6f}\n")
    f.write(f"ece_kl(n_bins=30)={val_b30:.6f}\n")
    f.write(f"ece_kl after scrambling non-top-1 probs (n_bins=15)={val_scrambled:.6f}\n")
    f.write(f"depends_on_n_bins={abs(val-val_b5)>1e-9 or abs(val-val_b30)>1e-9}\n")
    f.write(f"ignores_non_top1_distribution={abs(val-val_scrambled)<1e-9}\n")
print(open(out).read())
