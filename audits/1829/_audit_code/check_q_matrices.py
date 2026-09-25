"""Checks the committed OrthoTrans Q-matrices (.npy) for the self-contained
Weather dataset: shape vs seq_len/pred_len and orthonormality (Q Q^T = I).
Supports traceability of the OrthoTrans claim and the 'pre-computed orthogonal
matrix' description. Read-only.
"""
import os
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "jackyue1994__OLinear")
WDIR = os.path.join(REPO, "dataset", "weather")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

rows = []
for n in [96, 192, 336, 720]:
    f = os.path.join(WDIR, f"weather_{n}_ratio0.7.npy")
    q = np.load(f)
    # orthonormality: rows should be orthonormal eigenvectors
    qqt = q @ q.T
    err = np.abs(qqt - np.eye(q.shape[0])).max()
    rows.append((f"weather_{n}", q.shape, n, q.shape == (n, n), float(err)))

with open(os.path.join(OUT, "q_matrices.csv"), "w") as fh:
    fh.write("name,shape,expected_dim,shape_ok,max_orthonormality_err\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.2e}\n")
        print(r)
print("wrote", os.path.join(OUT, "q_matrices.csv"))
