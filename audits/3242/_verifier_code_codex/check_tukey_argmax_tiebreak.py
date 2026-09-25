"""Independent copy of the cited audit check. Output is redirected to /tmp so the
audit folder remains read-only apart from this permitted verifier script and verdict.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "code", "supplement", "depth-map.ipynb")
OUT = "/tmp/codex_tukey_argmax_check"
os.makedirs(OUT, exist_ok=True)

GRID_RES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
N_DIR = 250

cells = [
    "".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"
]
G = {"np": np}
exec(compile(cells[1].split("data, mask, lambd = generate_data")[0], "nb#cell1", "exec"), G)
exec(compile(cells[2], "nb#cell2", "exec"), G)
generate_data, tukey_depth = G["generate_data"], G["tukey_depth"]

data, mask, lambd = generate_data(n=50)


def weights_uniform(lambd):
    return np.ones(len(lambd)) / len(lambd)


def weights_threshold(lambd):
    t_l, t_u, t = 0, 1, 0.5
    while abs(t_u - t_l) > 1e-5:
        n = np.sum(lambd <= t)
        E = t ** 2
        if E > (1 / n):
            t_u = t
            t = (t + t_l) / 2
        elif E < (1 / n):
            t_l = t
            t = (t + t_u) / 2
        else:
            break
    return (lambd <= t) / np.sum(lambd <= t)


def weights_optlinear(lambd):
    n = len(lambd)
    k = n - 1
    for i in range(1, n):
        UB = (1 + np.linalg.norm(lambd[:i], 2) ** 2) / np.linalg.norm(lambd[:i], 1)
        if lambd[i] >= UB:
            k = i - 1
            break
    beta = (1 + np.linalg.norm(lambd[:k + 1], 2) ** 2) / (
        -np.linalg.norm(lambd[:k + 1], 1) ** 2
        + (k + 1) * (1 + np.linalg.norm(lambd[:k + 1], 2) ** 2)
    )
    alpha = beta * np.linalg.norm(lambd[:k + 1], 1) / (
        1 + np.linalg.norm(lambd[:k + 1], 2) ** 2
    )
    w = beta - alpha * lambd
    w[w < 0] = 0
    return w


def depth_map_vectorized(data, w, grid_res, n_dir, chunk=400):
    x = np.linspace(-2, 3, grid_res)
    y = np.linspace(-2, 3, grid_res)
    X, Y = np.meshgrid(x, y)
    grid = np.c_[X.ravel(), Y.ravel()]
    angles = np.linspace(0, 2 * np.pi, n_dir)
    dirs = np.c_[np.cos(angles), np.sin(angles)]
    proj = data @ dirs.T
    out = np.empty(grid.shape[0])
    for s in range(0, grid.shape[0], chunk):
        gp = grid[s:s + chunk] @ dirs.T
        lt = proj[None, :, :] < gp[:, None, :]
        gt = proj[None, :, :] > gp[:, None, :]
        cl = (lt * w[None, :, None]).sum(axis=1)
        cr = (gt * w[None, :, None]).sum(axis=1)
        out[s:s + chunk] = np.minimum(cl, cr).min(axis=1)
    return X, Y, out.reshape(grid_res, grid_res)


rows = []
for name, wf in [
    ("(A) uniform", weights_uniform),
    ("(B) threshold", weights_threshold),
    ("(C) optimal-linear", weights_optlinear),
]:
    w = wf(lambd)
    X, Y, dm = depth_map_vectorized(data, w, GRID_RES, N_DIR)
    angles = np.linspace(0, 2 * np.pi, N_DIR)
    dirs = np.c_[np.cos(angles), np.sin(angles)]
    rng = np.random.default_rng(7)
    idx = rng.integers(0, GRID_RES, size=(200, 2))
    err = max(
        abs(tukey_depth(np.array([X[i, j], Y[i, j]]), data, w, dirs) - dm[i, j])
        for i, j in idx
    )
    mx = dm.max()
    plateau = dm >= mx - 1e-12
    i0, j0 = np.unravel_index(np.argmax(dm), dm.shape)
    p_arg = np.array([X[i0, j0], Y[i0, j0]])
    ii, jj = np.nonzero(plateau)
    p_cen = np.array([X[ii, jj].mean(), Y[ii, jj].mean()])
    cell = 5.0 / (GRID_RES - 1)
    rows.append(
        dict(
            panel=name,
            grid_res=GRID_RES,
            max_depth=round(float(mx), 6),
            plateau_cells=int(plateau.sum()),
            plateau_area_frac=round(float(plateau.mean()), 6),
            plateau_width_x=round(float(X[ii, jj].max() - X[ii, jj].min()), 4),
            plateau_width_y=round(float(Y[ii, jj].max() - Y[ii, jj].min()), 4),
            argmax_x=round(float(p_arg[0]), 4),
            argmax_y=round(float(p_arg[1]), 4),
            centroid_x=round(float(p_cen[0]), 4),
            centroid_y=round(float(p_cen[1]), 4),
            dist_argmax_to_truemean=round(float(np.linalg.norm(p_arg)), 4),
            dist_centroid_to_truemean=round(float(np.linalg.norm(p_cen)), 4),
            argmax_closer_to_truth=bool(np.linalg.norm(p_arg) < np.linalg.norm(p_cen)),
            argmax_centroid_gap=round(float(np.linalg.norm(p_arg - p_cen)), 4),
            gap_in_grid_cells=round(float(np.linalg.norm(p_arg - p_cen) / cell), 1),
            max_abs_err_vs_authors_tukey_depth=float(err),
        )
    )
    print(rows[-1])

path = os.path.join(OUT, "check_tukey_argmax_tiebreak.csv")
with open(path, "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wtr.writeheader()
    wtr.writerows(rows)
print("\nwrote", path)
