"""Checks (a) the opt_linear weights sum to 1 (valid simplex vector w in Delta_n),
and (b) a reduced-scale reproduction of Fig 2 to confirm the proposed methods beat
the naive baseline. Verbatim-ports the estimator functions from experiments.ipynb
(read-only copy; the repo notebook is not modified/imported).
Supports the traceability table and the check that Fig 2's qualitative claim holds.
"""
import numpy as np, csv, os

# ---- verbatim port of experiments.ipynb estimator logic (bounded case) ----
def opt_linear_weights(lambd, c=1):
    n = len(lambd); k = n - 1
    for i in range(1, n):
        UB = (1 + c*np.linalg.norm(lambd[:i],2)**2)/(c*np.linalg.norm(lambd[:i],1))
        if lambd[i] >= UB:
            k = i - 1; break
    beta = (1 + c*np.linalg.norm(lambd[:k+1],2)**2)/(-c*np.linalg.norm(lambd[:k+1],1)**2 + (k+1)*(1+c*np.linalg.norm(lambd[:k+1],2)**2))
    alpha = beta*np.linalg.norm(lambd[:k+1],1)/(1 + c*np.linalg.norm(lambd[:k+1],2)**2)
    w = beta - c*alpha*lambd
    w[w < 0] = 0
    return w

def thresh_weights(lambd):
    tol = 1e-6; low, high = 0, 1; mid = 0.5
    while high - low > tol:
        mid = (low + high)/2
        s = np.sum(lambd <= mid)
        if s == 1/(mid**2): break
        elif s < 1/(mid**2): low = mid
        else: high = mid
    n_good = np.sum(lambd <= mid)
    w = np.zeros_like(lambd); w[:n_good] = 1.0/n_good
    return w

outdir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(outdir, exist_ok=True)

# ---- (a) weights sum to 1? ----
np.random.seed(1)
N = 10000
rows = []
for q in [0.1, 1.0, 5.0]:
    lambd = np.sort(1 - np.exp(-np.random.exponential(q, N)))
    for c in [1, 3]:
        w = opt_linear_weights(lambd, c=c)
        rows.append(dict(q=q, c=c, sum_w=round(float(w.sum()),6),
                         min_w=round(float(w.min()),6), nnz=int(np.sum(w>0))))
with open(os.path.join(outdir,"weights_sum.csv"),"w",newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
print("=== opt_linear weight-vector simplex check ===")
for r in rows: print(r)
print("all |sum_w - 1| < 1e-6 :", all(abs(r['sum_w']-1) < 1e-6 for r in rows))

# ---- (b) reduced reproduction of Fig 2(a) bounded: methods vs sample mean ----
print("\n=== reduced Fig 2(a) bounded reproduction (N=2000, trials=2000) ===")
np.random.seed(1)
N2, trials = 2000, 2000
p = 0
P = np.random.binomial(1, p, (trials, N2))       # clean = 0
Q = np.random.binomial(1, 1, (trials, N2))       # corrupt = 1
qgrid = 0.1*np.exp(np.linspace(0, np.log(5/0.1), 6))
comp = []
for x in qgrid:
    lambd = np.sort(1 - np.exp(-np.random.exponential(x, N2)))
    M = np.random.binomial(1, lambd, (trials, N2))
    Data = P*(1-M) + M*Q
    w_opt = opt_linear_weights(lambd, c=3)
    w_thr = thresh_weights(lambd)
    mse_opt = np.mean((Data @ w_opt - p)**2)
    n_good = int(np.sum(w_thr>0)); mse_thr = np.mean((Data[:,:n_good].mean(axis=1) - p)**2)
    mse_mean = np.mean((Data.mean(axis=1) - p)**2)
    comp.append((x, mse_opt, mse_thr, mse_mean))
print(f"{'q':>6} {'opt':>10} {'thresh':>10} {'mean':>10}  opt<=mean thr<=mean")
for x,o,t,m in comp:
    print(f"{x:>6.2f} {o:>10.2e} {t:>10.2e} {m:>10.2e}   {str(o<=m):>5} {str(t<=m):>5}")
print("methods <= sample-mean baseline at every q:",
      all(o<=m and t<=m for _,o,t,m in comp))
