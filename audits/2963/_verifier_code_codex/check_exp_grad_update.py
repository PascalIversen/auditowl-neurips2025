"""Checks murmail._exp_grad_update (batch_size=1 path, used by every notebook
experiment) against the batch version and the paper's mirror-descent step:
mu_{k+1}(a|s) ~ mu_k(a|s) exp(-eta g(s,a)), g(s,a)=mu_k(a|S_k)1{S_k=s}-1{A_k=a}.
Supports finding `single-sample-md-updates-one-action`. Read-only."""
import sys, os
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "tfreihaut__Murmail")
sys.path.insert(0, os.path.abspath(REPO))
from murmail import MaxUncertaintyResponseImitationLearning as M

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# minimal instance (only needs eta, S, A1, A2 for the update methods)
S, A = 2, 3
P = np.zeros((S, A, A, S)); P[..., 0] = 1.0
expert = np.ones((S, A)) / A
inst = M.__new__(M)
inst.eta = 0.5  # small eta so the math difference is not masked by saturation
inst.S, inst.A1, inst.A2 = S, A, A

policy = np.array([[0.2, 0.3, 0.5],
                   [0.1, 0.6, 0.3]])
s, a = 0, 1  # sampled state/action

# 1) code single-sample update
single = inst._exp_grad_update(policy.copy(), s, a)

# 2) code batch update with a batch of one identical sample
batch1 = inst._batch_exp_grad_update(policy.copy(), np.array([s]), np.array([a]), 1)

# 3) paper's correct full mirror-descent step at state s
g = policy[s, :].copy()      # mu_k(a'|s)  (the 1{S_k=s} term is 1 here)
g[a] -= 1.0                  # minus 1{A_k=a}
paper = policy.copy()
paper[s, :] = policy[s, :] * np.exp(-inst.eta * g)
paper[s, :] /= paper[s, :].sum()

lines = []
lines.append(f"policy[s]            = {policy[s]}")
lines.append(f"code single-sample   = {single[s]}")
lines.append(f"code batch(size=1)   = {batch1[s]}")
lines.append(f"paper full MD step   = {paper[s]}")
lines.append("")
lines.append(f"single == paper ? {np.allclose(single[s], paper[s])}")
lines.append(f"batch  == paper ? {np.allclose(batch1[s], paper[s])}")
lines.append(f"single == batch ? {np.allclose(single[s], batch1[s])}")
# how many action entries does single-sample change (before renorm logic)?
changed = np.sum(~np.isclose(single[s] / single[s].sum(), policy[s]))
lines.append(f"actions in state s whose prob changed (single-sample) = "
             f"{int(np.sum(~np.isclose(single[s], policy[s])))} of {A}")

txt = "\n".join(lines)
print(txt)
with open(os.path.join(OUT, "exp_grad_update.txt"), "w") as f:
    f.write(txt + "\n")
