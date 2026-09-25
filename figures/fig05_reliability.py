#!/usr/bin/env python3
"""Compact 2-panel robustness figure — panels A, B only, letter titles, no caption text.

  A  detection by severity (merged bars)
  B  ensembling: recall of the real-defect set vs # runs unioned

Reuses collect() + style from stability_panels.py.
Output: figures/out/fig05_reliability.{png,pdf,svg}
Run:    python figures/fig05_reliability.py
"""
from __future__ import annotations
import sys
from pathlib import Path

from math import comb

import numpy as np
import matplotlib.pyplot as plt

RB = Path(__file__).resolve().parent.parent / "studies" / "reliability"
sys.path.insert(0, str(RB))
import stability_panels as B  # noqa: E402  (also applies the shared rcParams/style)
import analyze as A  # noqa: E402

VERM, AMB, GRN, TEAL, GREY, MUTE, INK = B.VERM, B.AMB, B.GRN, B.TEAL, B.GREY, B.MUTE, B.INK


def _det_counts(paper):
    runs = A.load_runs(RB / paper); R = len(runs)
    fr = {}
    for ri, run in enumerate(runs):
        for idx in range(len(run["findings"])):
            fr[f"r{ri+1:02d}#{idx}"] = ri
    return [len({fr[f] for f in c["fids"] if f in fr}) for c in B.MERGED[paper]], R


def _curve(ds, R):  # expected fraction of the full candidate set found by union of K reports
    return np.array([np.mean([1 - (comb(R - d, K) / comb(R, K) if R - d >= K else 0.0) for d in ds])
                     for K in range(1, R + 1)])


def main():
    D = B.collect()
    FIGS = Path(__file__).resolve().parent / "out"
    # Bump every font up substantially over the shared compact style.
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 21, "axes.labelsize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
    })
    fig, (axS, axG) = plt.subplots(1, 2, figsize=(11.6, 5.4))
    fig.subplots_adjust(wspace=0.30, bottom=0.18, top=0.90, left=0.085, right=0.985)

    # A — detection by severity (merged)
    sdm = D["sev_det"]["merged"]; svs = ["high", "medium", "low"]
    sv_means = [np.mean(sdm[s]) if sdm[s] else 0.0 for s in svs]
    axS.bar(range(3), sv_means, width=0.6, color=[VERM, AMB, GRN], edgecolor="white", zorder=3)
    for i, m in enumerate(sv_means):
        axS.text(i, m + 0.02, f"{m:.2f}", ha="center", fontsize=16, weight="bold")
    axS.axhline(np.mean(D["merged_det"]), color=MUTE, lw=1.0, ls=":")
    axS.set_xticks(range(3)); axS.set_xticklabels(["High", "Medium", "Low"], fontsize=15)
    axS.set_ylim(0, 1.0); axS.set_ylabel("Mean detection rate"); axS.set_xlabel("Severity")
    axS.set_title("a", loc="left", fontweight="bold", fontsize=22)

    # B — discovery / saturation: fraction of all distinct findings recovered vs # reports combined
    fulls = np.array([_curve(*_det_counts(p)) for p in B.PAPERS])
    m = fulls.mean(0); Kx = np.arange(1, fulls.shape[1] + 1)
    for k, row in enumerate(fulls):
        axG.plot(Kx, row, color=GREY, lw=1.7, alpha=0.6, zorder=2, label="Per paper" if k == 0 else None)
    axG.plot(Kx, m, color=TEAL, lw=2.8, marker="o", ms=6, zorder=3, label="Mean (5 papers)")
    axG.set_xticks(range(1, 11)); axG.set_ylim(0, 1.03); axG.set_xlim(0.6, 10.4)
    axG.set_xlabel("# Audit reports combined"); axG.set_ylabel("Fraction of distinct findings found")
    axG.legend(loc="lower right", fontsize=14)
    axG.set_title("b", loc="left", fontweight="bold", fontsize=22)

    for n in ("fig05_reliability.png",):
        fig.savefig(FIGS / n); fig.savefig(FIGS / n.replace(".png", ".pdf")); fig.savefig(FIGS / n.replace(".png", ".svg"))
    plt.close(fig)
    print("wrote figures/fig05_reliability.{png,pdf,svg}")


if __name__ == "__main__":
    main()
