"""Regenerate simulated linear/exp data with the repo's generators (seed=42)
and compare against the shipped CSVs. Supports traceability of the simulated
datasets. Read-only w.r.t. repo (reads CSVs, writes only to _audit_code/out)."""
import sys, os
import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(__file__), "extracted",
                    "NIPS25_code_submit_final", "application")
sys.path.insert(0, REPO)
import simulate_data as sd  # noqa
sd.pd = pd  # work around module-scope pandas import (only in __main__ of the repo file)

out = []
for m in [0.1, 0.3, 0.5]:
    g = sd.DataGenerator_linear(mu=m, n_features=8, random_state=42)
    rct = g.generate_rct_data()
    ext = g.generate_external_data()
    ship_rct = pd.read_csv(os.path.join(REPO, f"dataset/simulated_data/rct_data_linear_{m}.csv"))
    ship_ext = pd.read_csv(os.path.join(REPO, f"dataset/simulated_data/rwe_data_linear_{m}.csv"))
    # align column order
    rct = rct[ship_rct.columns]
    ext = ext[ship_ext.columns]
    rct_match = np.allclose(rct.values, ship_rct.values, atol=1e-6)
    ext_match = np.allclose(ext.values, ship_ext.values, atol=1e-6)
    out.append(f"linear mu={m}: rct_reproduces={rct_match} ext_reproduces={ext_match} "
               f"(rct shape {rct.shape} vs ship {ship_rct.shape})")

for line in out:
    print(line)
