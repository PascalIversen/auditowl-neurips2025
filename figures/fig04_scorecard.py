#!/usr/bin/env python3
"""Per-paper reproducibility scorecard heatmap (paper Figure 4).

Reads ONLY `analysis/data/figure_data.json` (produced by `aggregate.py`) and
renders the scorecard: one column per audited paper, one row per pass/fail
reproducibility criterion, columns sorted best -> worst. A kept MEDIUM finding
shows yellow (a tier between clean green and a high-severity red); the left-hand
% counts green+yellow as passing, so it reads as "no high-severity".

Pipeline:  python analysis/aggregate.py && python figures/fig04_scorecard.py
Output:    figures/out/fig04_scorecard.{png,pdf,svg}
"""
from __future__ import annotations
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures" / "out"
DATA = ROOT / "analysis" / "data"
FIGS.mkdir(parents=True, exist_ok=True)
D = json.loads((DATA / "figure_data.json").read_text())

# ---- look & feel (Nature-figure publication style) ------------------------
_avail = {f.name for f in fm.fontManager.ttflist}
_FONT = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
              if f in _avail), "sans-serif")
INK, MUTE = "#333333", "#767676"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_FONT, "DejaVu Sans", "Liberation Sans"],
    "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlepad": 12,
    "axes.edgecolor": "#4D4D4D", "axes.linewidth": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 11,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})

PASS, WARN, FAIL = "#8BCF8B", "#E1A33C", "#B64342"   # green / amber / vermillion
BLACK = "#3d3d3d"        # soft near-black for "no code to score" cells
PCT_X = -0.6             # right edge of the left-hand % column (bars start at 0)

# A paper with a kept MEDIUM finding (and no high) of a category shows YELLOW
# (WARN) — a middle tier between clean (green) and a high-severity hit (red).
# The left-hand % counts green + yellow as passing ("no high-severity").
SC = D["scorecard_with_med"]
labels_sc = SC["row_labels"]
M = SC["matrix"]
# drop rows we don't want to display (cells are per-row, so filter each column too)
_DROP_ROWS = {"Real implementation (not a stub)"}
_keep = [i for i, lab in enumerate(labels_sc) if lab not in _DROP_ROWS]
labels_sc = [labels_sc[i] for i in _keep]
M = [[row[i] for i in _keep] for row in M]
nrows_sc = len(labels_sc)
npap = len(M)
# "med" (yellow) ranks between green and black/red: green < yellow < no-code < red.
RANK = {True: 0, "med": 1, None: 2, False: 3}
order = sorted(range(npap), key=lambda j: tuple(RANK[v] for v in M[j]))


def cell_color_med(v):
    if v is None:
        return BLACK
    if v == "med":
        return WARN
    return PASS if v else FAIL


_MANUAL_WRAP = {"Paper–code agreement": "Paper–code\nagreement"}
wrapped_labels = [_MANUAL_WRAP.get(l, textwrap.fill(l, 22)) for l in labels_sc]
fig, ax = plt.subplots(figsize=(15.5, 0.56 * nrows_sc + 1.4))
fig.subplots_adjust(left=0.30, right=0.985, bottom=0.16, top=0.97)
for col, j in enumerate(order):
    vals = M[j]
    for r in range(nrows_sc):
        ax.add_patch(mpatches.Rectangle((col, nrows_sc - 1 - r), 0.94, 0.9,
                     facecolor=cell_color_med(vals[r]), edgecolor="white", lw=0.3))
ax.set_xlim(-4, npap)
ax.set_ylim(0, nrows_sc)
ax.set_yticks([nrows_sc - 1 - r + 0.45 for r in range(nrows_sc)])
ax.set_yticklabels(wrapped_labels, fontsize=11.5)
ax.set_xticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
# per-row tally on the LEFT: "no high-severity" = green + yellow.
for r in range(nrows_sc):
    g = sum(1 for j in range(npap) if M[j][r] in (True, "med"))
    ax.text(PCT_X, nrows_sc - 1 - r + 0.45, f"{g/npap*100:.0f}%", va="center",
            ha="right", fontsize=12, fontweight="bold", color=INK)
ax.legend(handles=[mpatches.Patch(color=PASS, label="Pass"),
                   mpatches.Patch(color=WARN, label="Medium finding"),
                   mpatches.Patch(color=FAIL, label="Fail (high-severity)"),
                   mpatches.Patch(color=BLACK, label="No code to score")],
          loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=12.5)
for ext in ("png", "pdf", "svg"):
    fig.savefig(FIGS / f"fig04_scorecard.{ext}")
plt.close(fig)
print("wrote figures/fig04_scorecard.{png,pdf,svg}")
