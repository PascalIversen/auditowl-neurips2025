"""Replays depth-map.ipynb's standalone cells in notebook order to show that the
threshold cell rebinds the global `N` (50 -> |{i: lambda_i <= t}|), so the next
cell's `k = N-1` / `range(1,N)` loop no longer ranges over all 50 samples.
Supports finding `depth-map-N-shadowed`. Read-only on code/; writes out/check_N_shadowing.txt.
"""
import io
import json
import os
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "code", "supplement", "depth-map.ipynb")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

cells = [
    "".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"
]

# Cell indices (code cells only, 0-based) in depth-map.ipynb:
#   0 imports | 1 generate_data + hist | 2 tukey_depth/compute_depth_map
#   3 plot_depth_colormap | 4 "Run everything": N=50; generate_data
#   5 "(A)" uniform weights | 6 "(B)" threshold weights | 7 "(C)" optimal-linear weights
#   8 plot_combined_depth_maps (the function that made Figure 1)
G = {}
lines = []

# Drop the expensive/irrelevant plotting + depth-map calls; keep all weight logic.
SKIP_PREFIXES = ("plt.", "plot_depth_colormap", "plot_combined", "X, Y, depth_map", "print(")


def run(i, strip_plot=True):
    """exec a notebook cell in shared globals G, suppressing the plotting calls."""
    src = cells[i]
    if strip_plot:
        src = "\n".join(
            ln for ln in src.split("\n") if not ln.strip().startswith(SKIP_PREFIXES)
        )
    with redirect_stdout(io.StringIO()):
        exec(compile(src, f"depth-map.ipynb#cell{i}", "exec"), G)


run(0)
run(1)
run(2)
# --- cell 4: "# --- Run everything ---"  N = 50; data, mask, lambd = generate_data(n=N)
run(4)
run(5)
lines.append(f"after cell 4  (N = 50; generate_data):        N = {G['N']!r}  len(lambd) = {len(G['lambd'])}")

# --- cell 6: "(B)" threshold weights.  Its while-loop assigns `N = np.sum(lambd <= t)`
run(6)
n_after = G["N"]
lines.append(f"after cell 6  ((B) threshold weights):        N = {n_after!r}   <-- global N CLOBBERED")
lines.append(f"              threshold t = {G['t']:.6f}, |{{i: lambda_i <= t}}| = {int(np.sum(G['lambd'] <= G['t']))}")

# --- cell 7: "(C)" optimal-linear weights, which reads the (now clobbered) global N
run(7)
w_asrun = np.array(G["w"], dtype=float)
k_asrun = G["k"]
lines.append(f"after cell 7  ((C) optimal-linear weights):   k = {k_asrun!r}  (loop ran over range(1,{n_after}))")

# --- Now the same cell 6 with the CORRECT N = len(lambd) = 50, which is what
#     plot_combined_depth_maps (cell 8, the function that produced Figure 1) uses.
G2 = dict(G)
G2["N"] = len(G["lambd"])
with redirect_stdout(io.StringIO()):
    exec(compile("\n".join(ln for ln in cells[7].split("\n") if not ln.strip().startswith(SKIP_PREFIXES)), "depth-map.ipynb#cell7", "exec"), G2)
w_fixed = np.array(G2["w"], dtype=float)
k_fixed = G2["k"]

lines.append("")
lines.append(f"cell 7 re-run with N = len(lambd) = 50:       k = {k_fixed!r}")
lines.append("")
lines.append(f"weights identical?            {np.allclose(w_asrun, w_fixed)}")
lines.append(f"nonzero weights  as-run: {int((w_asrun > 0).sum())}   corrected: {int((w_fixed > 0).sum())}")
lines.append(f"sum(w)           as-run: {w_asrun.sum():.6f}   corrected: {w_fixed.sum():.6f}")
lines.append(f"max|w_asrun - w_fixed|:       {np.abs(w_asrun - w_fixed).max():.3e}")
lines.append("")
lines.append("NOTE: cell 8 (plot_combined_depth_maps, the function that saves")
lines.append("      combined_depth_maps.png = paper Figure 1) sets `N = len(lambd)` at")
lines.append("      .ipynb line 334 BEFORE its optimal-linear loop, so Figure 1 itself")
lines.append("      is computed with the correct N. Only the standalone cells are affected.")

txt = "\n".join(lines)
print(txt)
with open(os.path.join(OUT, "check_N_shadowing.txt"), "w") as fh:
    fh.write(txt + "\n")
