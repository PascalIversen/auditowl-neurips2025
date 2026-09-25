#!/usr/bin/env python3
"""Confidence-calibration figure for the auditor's findings (dot-and-whisker).

The calibration question: among findings the auditor reported at a given
confidence (low / medium / high), what fraction did the human judge correct?
A calibrated auditor's success rate rises with stated confidence.

"Correct" here = the human marked the finding "correct & relevant" OR "correct
but severity-overstated" (both are genuine, on-point findings). The denominator
is all findings in the tier; not-relevant, unsure, and false make up the rest.
The success definition is named in the y-axis label, not a title.

Confidence is a 3-level ordinal label, not a probability, so the x-axis is
categorical and there is NO identity line. Dot = empirical rate, whisker = Wilson
95% CI (bins are small, esp. 'low'), faint line connects the tiers, dashed line
marks the pooled rate, each point annotated rate% (k/n).

Output: figures/fig_human_eval_calibration.{png,pdf,svg}
Run:    python studies/reliability/calibration_panels.py   (run aggregate first)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

RB = Path(__file__).resolve().parent
DATA = RB / "data"
FIGS = RB / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- shared style (matches build_human_eval_figure.py) ----
_avail = {f.name for f in fm.fontManager.ttflist}
_FONT = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if f in _avail), "sans-serif")
INK, MUTE = "#333333", "#767676"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [_FONT, "DejaVu Sans", "Liberation Sans"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlepad": 8,
    "axes.edgecolor": "#4D4D4D", "axes.linewidth": 1.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 9.5,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})
# sequential blue ramp: low -> high confidence — matches the severity ramp in
# human_eval_panels.py so the two panels share one ordinal colour scale
RAMP = {"low": "#B5CCDA", "medium": "#3B7A9E", "high": "#21506C"}
ORDER = ["low", "medium", "high"]
SUCCESS = {"correct_relevant", "correct_wrong_severity"}   # counts as a hit


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion (returns lo, hi in [0,1])."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load() -> list[dict]:
    with (DATA / "human_eval_findings.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = load()
    stat = {}
    for c in ORDER:
        sub = [r for r in rows if r["confidence"] == c]
        stat[c] = (sum(1 for r in sub if r["verdict"] in SUCCESS), len(sub))
    K = sum(s[0] for s in stat.values())
    N = sum(s[1] for s in stat.values())
    pooled = 100.0 * K / N

    x = np.arange(len(ORDER))
    rate = [100.0 * stat[c][0] / stat[c][1] if stat[c][1] else 0.0 for c in ORDER]
    ci = [wilson(*stat[c]) for c in ORDER]
    lo = [rate[i] - 100 * ci[i][0] for i in range(len(ORDER))]
    hi = [100 * ci[i][1] - rate[i] for i in range(len(ORDER))]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.13)

    ax.plot(x, rate, "-", lw=1.3, color=MUTE, zorder=2)
    ax.errorbar(x, rate, yerr=[lo, hi], fmt="none", ecolor=INK,
                elinewidth=1.5, capsize=6, capthick=1.5, zorder=3)
    for i, c in enumerate(ORDER):
        ax.scatter(i, rate[i], s=160, color=RAMP[c], edgecolor="white", linewidth=1.6, zorder=4)
        ax.text(i + 0.10, rate[i], f"{rate[i]:.0f}%  ({stat[c][0]}/{stat[c][1]})",
                ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.axhline(pooled, ls="--", lw=1.3, color=MUTE, zorder=1)
    ax.text(-0.45, pooled + 1.5, f"pooled {pooled:.0f}%", ha="left", va="bottom",
            fontsize=8.5, color=MUTE)

    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(-0.5, 2.75)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in ORDER])
    # the success definition (incl. severity-overstated) is named on the y-axis
    ax.set_xlabel("AI auditor's self-reported confidence")
    ax.set_ylabel("Human-rated correct & relevant, or\ncorrect but severity overstated (%)")
    ax.tick_params(axis="x", length=0)

    stem = "fig_human_eval_calibration"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote figures/{stem}.{{png,pdf,svg}}  ·  "
          + " ".join(f"{c}={stat[c][0]}/{stat[c][1]} ({rate[i]:.0f}%)" for i, c in enumerate(ORDER))
          + f"  ·  pooled {K}/{N} ({pooled:.0f}%)")


if __name__ == "__main__":
    main()
