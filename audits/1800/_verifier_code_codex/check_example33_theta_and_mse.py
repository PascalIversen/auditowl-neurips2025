"""Checks Example 3.3 (Example_Mismatch_1.py): (a) that the code's transform_mat
equals the paper's theta up to the same theta^T theta, (b) that MSE is computed on
TRAIN data (paper says test), and quantifies the train-vs-test MSE gap. Supports
findings: mse-computed-on-train (difference) and theta-matches-paper (fine)."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out = open(os.path.join(os.path.dirname(__file__), "out", "example33.txt"), "w")
def log(*a):
    print(*a); print(*a, file=out)

n = 2
target_cov = np.array([[1.0, 0.99], [0.99, 1.0]])
weights = np.linalg.cholesky(target_cov)
transform_mat = weights.T  # this is theta in code

log("transform_mat (code theta):\n", transform_mat)
log("transform_mat^T @ transform_mat (theta^T theta):\n", transform_mat.T @ transform_mat)
log("paper theta = [[1,0.99],[0,0.141]]; paper theta^T theta:")
theta_paper = np.array([[1.0, 0.99], [0.0, 0.141]])
log(theta_paper.T @ theta_paper)
log("diag of theta^T theta (must equal Vt(I) diag=1 for same per-entry MSE):",
    np.diag(transform_mat.T @ transform_mat))

# Reproduce the train-vs-test MSE gap for one rho to show train MSE != test MSE size
def gen(N, T, n, rho, sigma, tm=None):
    noise = np.random.normal(0, sigma, (N, T, n))
    pred = rho * noise + np.sqrt(1 - rho**2) * np.random.normal(0, sigma, (N, T, n))
    if tm is not None:
        pred = rho * (noise @ tm.T) + np.random.normal(0, sigma, (N, T, n)) @ \
            np.linalg.cholesky(np.eye(n) - rho**2 * tm @ tm.T).T
    return noise, pred

N, T, sigma = 80000, 100, 1
N_train = int(N*0.8)
np.random.seed(0)
rho = 0.5
noise, pred = gen(N, T, n, rho, sigma, transform_mat)
ntr, nte = noise[:N_train].reshape(-1, n), noise[N_train:].reshape(-1, n)
ptr, pte = pred[:N_train].reshape(-1, n), pred[N_train:].reshape(-1, n)
m = LinearRegression().fit(ptr, ntr)
for i in range(n):
    mse_tr = mean_squared_error(ntr[:, i], m.predict(ptr)[:, i])
    mse_te = mean_squared_error(nte[:, i], m.predict(pte)[:, i])
    log(f"dim{i}: train MSE={mse_tr:.6f}  test MSE={mse_te:.6f}  "
        f"abs diff={abs(mse_tr-mse_te):.6e}")
out.close()
