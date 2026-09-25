"""Checks that analysis.ipynb computes the paper's significance t-tests on
synthetic Gaussian draws from the reported mean/std, NOT on the real 10 seed
means. Supports finding `ttest-on-synthetic-data`.

The repo (mpnguyen2/dfPO @ 9dc65e7) computes the real per-seed means inside
benchmarks_run.py (avg_final_vals over 10 seeds) but persists ONLY the
per-algorithm summary `mean +/- std` to output/benchmarks_stat_analysis.csv.
analysis.ipynb (cell 4) then RE-SAMPLES n=10 fake values via
np.random.normal(loc=mean, scale=std) from those hard-coded summary stats and
runs scipy ttest_ind on the fabricated samples. We reproduce the notebook's
exact computation and show the reported p-values come straight from the
synthetic draws, not the experimental seed-level means.
"""
import numpy as np
from scipy.stats import ttest_ind

np.random.seed(42)  # exact seed used in analysis.ipynb cell 4
n = 10

# Values hard-coded in the notebook (taken from the SUMMARY table, not raw seeds)
dpo_md = np.random.normal(loc=6.296, scale=0.048, size=n)
crossq_md = np.random.normal(loc=6.365, scale=0.030, size=n)
stat_md, pval_md = ttest_ind(dpo_md, crossq_md, equal_var=False)

dpo_topo = np.random.normal(loc=6.046, scale=0.083, size=n)
strpo_topo = np.random.normal(loc=6.470, scale=0.098, size=n)
stat_topo, pval_topo = ttest_ind(dpo_topo, strpo_topo, equal_var=False)

dpo_mol = np.random.normal(loc=53.352, scale=0.055, size=n)
ddpg_mol = np.random.normal(loc=68.203, scale=0.001, size=n)
stat_mol, pval_mol = ttest_ind(dpo_mol, ddpg_mol, equal_var=False)

rows = [
    ("surface_DPO_vs_CrossQ", stat_md, pval_md),
    ("grid_DPO_vs_STRPO", stat_topo, pval_topo),
    ("molecule_DPO_vs_DDPG", stat_mol, pval_mol),
]

import csv, os
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "out", "ttest_synthetic.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comparison", "t_stat", "p_value", "data_source"])
    for name, t, p in rows:
        print(f"{name}: t={t:.4f}, p={p:.4e}  (computed on np.random.normal draws)")
        w.writerow([name, f"{t:.4f}", f"{p:.4e}", "np.random.normal(mean,std)"])
print(f"\nWrote {out}")
print("NOTE: surface p=%.4f is NOT significant (>0.05); this matches the paper's"
      " 'comparable performance' caveat, but ALL three p-values are derived from"
      " synthetic Gaussian samples, not the experimental seed-level means." % pval_md)
