"""Checks the semantics of select_samples (argsort[::-1][-top_k:]) used in
lasso_selector.py:173 and influence_selector.py:95/206/318.
Supports finding: select-samples-ordering. Read-only; prints to stdout."""
import numpy as np

# Simulate scores (smaller = more comparable per paper).
scores = np.array([0.9, 0.1, 0.5, 0.05, 0.3, 0.8, 0.02])

def select_samples(scores, top_k):
    # exact copy of repo logic
    return np.argsort(scores)[::-1][-top_k:]

for k in [1, 3, 5]:
    idx = select_samples(scores, k)
    sel = scores[idx]
    smallest_k = np.sort(scores)[:k]
    print(f"top_k={k}: selected idx={idx.tolist()} scores={sel.tolist()} "
          f"| smallest_{k}_sorted={smallest_k.tolist()} "
          f"| set_equal={set(np.round(sel,6))==set(np.round(smallest_k,6))}")
