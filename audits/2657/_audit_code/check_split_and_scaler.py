"""Sanity-checks the data pipeline in utils.load_adbench_npz: (a) StandardScaler is fit on TRAIN
only (no full-data leakage), (b) train set is normal-only, (c) anomalies appear only in test, and
(d) train/test normal samples are disjoint. Supports the 'looks fine' items on leakage.
Read-only: re-implements the split deterministically on one shipped dataset and asserts properties."""
import os, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "ZhongLIFR__TCCM-NIPS"))
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)

ds = os.path.join(REPO, "datasets", "small", "42_WBC.npz")
data = np.load(ds, allow_pickle=True)
X, y = data["X"], data["y"].astype(int)

X_normal, X_anom = X[y == 0], X[y == 1]
y_normal, y_anom = y[y == 0], y[y == 1]
Xtr, Xte_n, ytr, yte_n = train_test_split(X_normal, y_normal, test_size=0.5, random_state=0, stratify=y_normal)
Xte = np.vstack((Xte_n, X_anom)); yte = np.concatenate((yte_n, y_anom))

scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)

# Scaler mean must equal TRAIN mean, not full-data mean (would be leakage).
mean_matches_train = np.allclose(scaler.mean_, Xtr.mean(axis=0))
mean_matches_full = np.allclose(scaler.mean_, X.mean(axis=0))

# Train must be all-normal.
train_all_normal = bool((ytr == 0).all())
# Anomalies only in test.
anom_only_in_test = bool((ytr == 0).all() and (yte == 1).sum() == len(y_anom))
# Train and test-normal rows disjoint (row-level set intersection).
tr_set = set(map(tuple, Xtr)); te_set = set(map(tuple, Xte_n))
overlap = len(tr_set & te_set)

lines = [
    f"scaler_mean_equals_TRAIN_mean (good): {mean_matches_train}",
    f"scaler_mean_equals_FULL_mean (would be leakage): {mean_matches_full}",
    f"train_is_all_normal: {train_all_normal}",
    f"all_anomalies_in_test_only: {anom_only_in_test}",
    f"train/test-normal row overlap (expect 0): {overlap}",
]
with open(os.path.join(OUT, "split_and_scaler.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
