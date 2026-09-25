"""Checks (1) code's MoI embedding weights equal paper Eq.(5), and (2) the
effect of computing entropy over top-k (default 20) instead of full vocab V.
Supports findings: moi-weights-match-paper (fine) and topk-truncation-vs-fullvocab."""
import numpy as np

rng = np.random.default_rng(0)

def code_weights(p, y_idx, beta, H):
    """Reproduce the effective per-token weight implied by gpu_model_runner.py
    lines 1218 & 1224-1228 on the *given support* p (already renormalized).
    posterior_probs = (p + beta*onehot)/(1+beta); embedding =
    H * sum(emb*posterior) + (1-H)*emb[y]. So weight_i =
    H*posterior_i + (1-H)*onehot_i."""
    onehot = np.zeros_like(p); onehot[y_idx] = 1.0
    posterior = (p + beta*onehot)/(1+beta)
    w = H*posterior + (1-H)*onehot
    return w

def paper_weights(p, y_idx, beta, H):
    """Paper Eq.(5): w_i = (H*p_i + (beta+1-H)*y_i)/(beta+1)."""
    onehot = np.zeros_like(p); onehot[y_idx] = 1.0
    return (H*p + (beta+1-H)*onehot)/(beta+1)

# --- Check 1: formula equivalence on a fixed support ---
max_abs_diff = 0.0
for _ in range(2000):
    k = rng.integers(2, 30)
    p = rng.random(k); p /= p.sum()
    y = int(rng.integers(0, k))
    beta = float(rng.choice([0.25, 0.5, 1, 2, 4, 8]))
    H = float(rng.random())
    d = np.abs(code_weights(p, y, beta, H) - paper_weights(p, y, beta, H)).max()
    max_abs_diff = max(max_abs_diff, d)

# --- Check 2: entropy over top-20 vs full vocab ---
# Simulate a peaked next-token distribution over a large vocab; compute
# normalized entropy the way the code does (top-20, H_max=log(20)) vs the way
# the paper defines it (full V, H_max=log(V)).
V = 50000
rows = []
for trial in range(5):
    logits = rng.normal(0, 4, V)
    full = np.exp(logits - logits.max()); full /= full.sum()
    # paper: H over full vocab, normalized by log V
    H_full = -(full*np.log(np.clip(full,1e-12,None))).sum()/np.log(V)
    # code: take top-20 logprobs, renormalize, H_max=log(20)
    top = np.sort(full)[::-1][:20]; top = top/top.sum()
    H_code = -(top*np.log(np.clip(top,1e-12,None))).sum()/np.log(20)
    rows.append((trial, round(H_full,4), round(H_code,4)))

with open("out/moi_formula_check.txt","w") as f:
    f.write(f"max_abs_weight_diff_code_vs_paper_eq5={max_abs_diff:.2e}\n")
    f.write("entropy_trial,H_paper_fullV(logV),H_code_top20(log20)\n")
    for t,a,b in rows:
        f.write(f"{t},{a},{b}\n")

print(f"max_abs_weight_diff_code_vs_paper_eq5={max_abs_diff:.2e}")
print("trial  H_paper(fullV)  H_code(top20)")
for t,a,b in rows:
    print(f"{t}      {a}          {b}")
