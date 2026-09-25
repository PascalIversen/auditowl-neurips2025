"""Checks that gptq.py Helper.run_weight_correct implements paper Eq.(6):
W*(a) = W + a * W @ delta @ Xhat.T @ inv(Xhat @ Xhat.T).
Confirms the code's accumulators (H, H_delta) reproduce the closed form and
that a=1 minimizes ||W X - Wcorr Xhat||_F (Prop 5.1). Supports finding qep-formula-faithful.
"""
import numpy as np

rng = np.random.default_rng(0)
n, d, m = 5, 7, 200          # out, in, samples
W = rng.standard_normal((n, d))
X = rng.standard_normal((d, m))           # full-precision input activations
Xhat = X + 0.1 * rng.standard_normal((d, m))  # quantized (perturbed) input
delta = X - Xhat                          # paper delta = X - Xhat

# Replicate code accumulation (gptq.py add_batch_qep, ignoring the per-batch
# running-mean scaling which cancels for a single batch up to a constant factor).
# H = Xhat Xhat^T ; H_delta = delta Xhat^T   (code: self.H_delta += delta_scaled @ inp_scaled.T)
H = Xhat @ Xhat.T
H_delta = delta @ Xhat.T

# code: W += (W @ H_delta @ Hinv) * perccorr  (gptq.py:269), with small damp
damp = 0.01 * np.mean(np.diag(H))
Hd = H + damp * np.eye(d)
Hinv = np.linalg.inv(Hd)

for alpha in [0.0, 0.5, 1.0]:
    Wcorr = W + alpha * (W @ H_delta @ Hinv)
    # paper closed form (no damp): W* = W + a * W delta Xhat^T inv(Xhat Xhat^T)
    Wstar = W + alpha * (W @ delta @ Xhat.T @ np.linalg.inv(H))
    diff = np.linalg.norm(Wcorr - Wstar) / np.linalg.norm(Wstar)
    # objective ||W X - Wcorr Xhat||_F
    obj = np.linalg.norm(W @ X - Wcorr @ Xhat)
    print(f"alpha={alpha}: rel_diff_code_vs_paper={diff:.2e}  obj={obj:.4f}")

# a=1 (full correction, no damp) should drive the objective ~0 since W*Xhat = W X exactly
Wfull = W + (W @ delta @ Xhat.T @ np.linalg.inv(H))
obj_full = np.linalg.norm(W @ X - Wfull @ Xhat)
print(f"objective at alpha=1 (closed form, no damp) = {obj_full:.2e} (expect ~0)")
