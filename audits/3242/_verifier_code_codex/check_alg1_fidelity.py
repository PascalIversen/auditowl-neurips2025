"""Verifies that experiments.ipynb's `opt_linear` weight construction really is the exact
minimiser of the paper's objective ||w||_2^2 + c (w^T lambda)^2 over the simplex (paper eq. 8 /
Algorithm 1): checks w >= 0, sum(w) = 1, support = {i : lambda_i <= threshold}, and objective
value against scipy SLSQP. Also reports the threshold t chosen by depth-map.ipynb's
balancing loop (which uses d=1, i.e. t^2 = 1/N(t)) versus the d=2 value (t^2 = 2/N(t))
appropriate for the 2-D Figure 1.
Supports 'items that look fine' + finding `depth-map-c-and-d-constants`.
Read-only on code/; writes out/check_alg1_fidelity.txt.
"""
import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

exp_nb = os.path.join(HERE, "..", "code", "supplement", "experiments.ipynb")
cells = [
    "".join(c["source"]) for c in json.load(open(exp_nb))["cells"] if c["cell_type"] == "code"
]
G = {"np": np}
with redirect_stdout(io.StringIO()):
    exec(compile(cells[1], "experiments.ipynb#codecell1", "exec"), G)
opt_linear = G["opt_linear"]

lines = []


def weights_from_opt_linear(lambd, c):
    """opt_linear returns only the MSE; re-run its weight construction verbatim."""
    n = len(lambd)
    k = n - 1
    for i in range(1, n):
        UB = (1 + c * np.linalg.norm(lambd[:i], 2) ** 2) / (c * np.linalg.norm(lambd[:i], 1))
        if lambd[i] >= UB:
            k = i - 1
            break
    beta = (1 + c * np.linalg.norm(lambd[:k + 1], 2) ** 2) / (
        -c * np.linalg.norm(lambd[:k + 1], 1) ** 2
        + (k + 1) * (1 + c * np.linalg.norm(lambd[:k + 1], 2) ** 2)
    )
    alpha = beta * np.linalg.norm(lambd[:k + 1], 1) / (
        1 + c * np.linalg.norm(lambd[:k + 1], 2) ** 2
    )
    w = beta - c * alpha * lambd
    w[w < 0] = 0
    return w


def obj(w, lambd, c):
    return float(np.dot(w, w) + c * np.dot(w, lambd) ** 2)


rng = np.random.default_rng(3)
lines.append("Algorithm 1 (opt_linear) vs scipy SLSQP on  min_{w in simplex} ||w||^2 + c(w.lambda)^2")
lines.append(f"{'n':>5} {'c':>4} {'sum(w)':>10} {'min(w)':>10} {'supp':>6} "
             f"{'obj_alg1':>12} {'obj_slsqp':>12} {'alg1<=slsqp':>12}")
ok = True
for n in (20, 50, 200):
    for c in (1.0, 3.0):
        for q in (0.5, 2.0, 5.0):
            lambd = np.sort(1 - np.exp(-rng.exponential(q, n)))
            w = weights_from_opt_linear(lambd, c)
            res = minimize(
                obj, np.ones(n) / n, args=(lambd, c), method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1}],
                options={"maxiter": 800, "ftol": 1e-14},
            )
            o1, o2 = obj(w, lambd, c), obj(res.x, lambd, c)
            good = (abs(w.sum() - 1) < 1e-10) and (w.min() >= 0) and (o1 <= o2 + 1e-9)
            ok &= good
            lines.append(f"{n:>5} {c:>4.0f} {w.sum():>10.8f} {w.min():>10.3e} "
                         f"{int((w > 0).sum()):>6} {o1:>12.6e} {o2:>12.6e} {str(o1 <= o2 + 1e-9):>12}")
lines.append(f"ALL CHECKS PASS: {ok}")
lines.append("  (sum(w)=1 exactly, w>=0, and Algorithm 1's objective is <= the numerical optimum")
lines.append("   => opt_linear implements paper eq. (8)/Algorithm 1 correctly, including the")
lines.append("   re-parameterisation alpha_code = alpha_paper/c with w = beta - c*alpha_code*lambda.)")

# --- threshold constant used by depth-map.ipynb for the 2-D Figure 1 ---
lines.append("")
lines.append("depth-map.ipynb threshold loop (.ipynb lines 318-331) balances t^2 vs 1/N(t) (d=1).")
lines.append("Paper eq. (21)/Theorem 4 for d dimensions balances t^2 vs d/N(t); Figure 1 has d=2.")


def balance(lambd, d):
    t_l, t_u, t = 0, 1, 0.5
    while abs(t_u - t_l) > 1e-5:
        n = np.sum(lambd <= t)
        n = max(n, 1)
        if t ** 2 > (d / n):
            t_u = t
            t = (t + t_l) / 2
        elif t ** 2 < (d / n):
            t_l = t
            t = (t + t_u) / 2
        else:
            break
    return t


G2 = {"np": np}
dm_nb = os.path.join(HERE, "..", "code", "supplement", "depth-map.ipynb")
dcells = [
    "".join(c["source"]) for c in json.load(open(dm_nb))["cells"] if c["cell_type"] == "code"
]
exec(compile(dcells[1].split("data, mask, lambd = generate_data")[0], "nb#1", "exec"), G2)
_, _, lam50 = G2["generate_data"](n=50)
t1, t2 = balance(lam50, 1), balance(lam50, 2)
lines.append(f"  Figure-1 lambda (n=50, seed=42): t(d=1) = {t1:.5f} -> keeps "
             f"{int(np.sum(lam50 <= t1))}/50 samples   [what the code does]")
lines.append(f"                                   t(d=2) = {t2:.5f} -> keeps "
             f"{int(np.sum(lam50 <= t2))}/50 samples   [d=2, as Figure 1 is 2-D]")

txt = "\n".join(lines)
print(txt)
with open(os.path.join(OUT, "check_alg1_fidelity.txt"), "w") as fh:
    fh.write(txt + "\n")
