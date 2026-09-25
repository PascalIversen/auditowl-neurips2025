"""Verify Theorem 4.7 bound (d_rep <= 2*M*d_prob, M=rep_dim=2) holds on constructed perturbed models (paper Fig 16/App F.6). Read-only."""
import sys, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'bemigini__close-dist-rep-sim'))
sys.path.insert(0, REPO)
import numpy as np
np.random.seed(0)
from experiments.making_constructed_models import make_even_spaced_classifier_2d, get_distances_to_perturbed_models
nc=7
m1_all_fx, m1_gy, g_m1_angles, x_angles_m1, fx_lengths = make_even_spaced_classifier_2d(
    num_classes=nc, num_inputs_per_class=500, g_length=20, min_f_length=5)
# small run: patch the loop range is 250; just call and inspect
d = get_distances_to_perturbed_models(m1_all_fx, m1_gy, g_m1_angles, x_angles_m1, fx_lengths, nc, noise_level=0.001)
import numpy as np
dprob=np.array(d['d_prob']); bound=np.array(d['d_rep_bound'])
msvds_f=d['msvds_Lfx']; msvds_g=d['msvds_Ngy']
# max d_rep = max(1-min_msvd_f, 1-min_msvd_g)
maxdrep=[]
for mf,mg in zip(msvds_f,msvds_g):
    maxdrep.append(max(1-np.min(mf), 1-np.min(mg)))
maxdrep=np.array(maxdrep)
viol=np.sum(maxdrep > bound + 1e-9)
print(f"n={len(dprob)}  bound=2*2*d_prob  violations(maxdrep>bound): {viol}")
print(f"max(maxdrep/bound ratio) = {np.max(maxdrep/np.maximum(bound,1e-12)):.4f}  (should be <=1)")
