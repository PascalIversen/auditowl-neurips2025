"""Reproduce paper Table 1 (KL_table) to verify Table 1 values. Read-only: imports repo code, runs the KL-to-zero experiment."""
import sys, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'bemigini__close-dist-rep-sim'))
sys.path.insert(0, REPO)
import numpy as np
from experiments import kl_to_zero_dissimilar_reps as k
np.random.seed(0)
d = k.kl_to_zero_dissimilar_reps(device='cpu')
import json
with open(os.path.join(os.path.dirname(__file__), 'out', 'kl_table.json'), 'w') as f:
    json.dump({kk: [float(x) for x in vv] for kk, vv in d.items()}, f, indent=2)
print("ROWS (g_length, KL, d_prob, mCCA_f, max_d_rep_f):")
for i in range(len(d['g_length'])):
    print(d['g_length'][i], round(d['KL_div'][i],4), round(d['d_prob'][i],4), round(d['mCCA_f'][i],4), round(d['max_d_rep_f'][i],4))
