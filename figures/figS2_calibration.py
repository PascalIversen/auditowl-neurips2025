#!/usr/bin/env python3
"""Combined human-eval figure: (a) verdict breakdown + (b) confidence calibration.

Panel a = figS2_calibration_micro  — all findings pooled (micro-average),
          one stacked bar per verdict, split low- vs medium/high-severity.
Panel b = figS2_calibration_panels    — success rate vs the auditor's
          self-reported confidence, Wilson 95% CI, pooled reference line.

Both panels are rebuilt from the same human_eval_findings.csv via the existing
single-panel builders (imported below), so this stays in sync with them.

Output: figures/figS2_calibration.{png,pdf,svg}
Run:    python figures/figS2_calibration.py  (run aggregate first)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RB = Path(__file__).resolve().parent.parent / "studies" / "reliability"
sys.path.insert(0, str(RB))

import human_eval_panels as ov          # noqa: E402  (sets shared rcParams on import)
import calibration_panels as cal      # noqa: E402

FIGS = Path(__file__).resolve().parent / "out"


def panel_verdicts(ax, rows):
    """Panel a — stacked severity bars per human verdict (micro-average)."""
    cats, pct, lab = ov.series(rows, "micro")
    x = np.arange(len(cats))
    ax.set_xlim(-0.5, len(cats) - 0.5)

    bottom = np.zeros(len(cats))                             # stack high (dark) -> low (light)
    for s in ov.SEV_STACK:
        vals = np.array([pct[k][s] for k in cats])
        ax.bar(x, vals, width=0.62, bottom=bottom, color=ov.SEV_COLOR[s],
               edgecolor="white", linewidth=1.2, zorder=3)
        bottom += vals
    for xi, k, tot in zip(x, cats, bottom):
        ax.text(xi, tot + 1.2, lab[k], ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=ov.INK)

    # Capitalized first word + tighter wrapping so the wide labels don't overlap.
    XLABELS = {
        "correct_relevant": "Correct &\nrelevant",
        "correct_not_relevant": "Correct but\nnot relevant",
        "correct_wrong_severity": "Correct but\nseverity\noverstated",
        "false": "False",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([XLABELS.get(k, ov.LABEL[k].capitalize()) for k in cats], fontsize=14)
    ax.set_ylabel("Share of findings (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="x", length=0)

    sev_h = [Patch(facecolor=ov.SEV_COLOR[s], edgecolor="white", label=ov.SEV_LABEL[s])
             for s in ("high", "medium", "low")]
    leg = ax.legend(handles=sev_h, loc="upper right", title="Severity",
                    fontsize=15, title_fontsize=16, alignment="left",
                    handlelength=1.5, handleheight=1.4)
    leg._legend_box.sep = 10                                  # keep the title clear of the swatches


def panel_calibration(ax, rows):
    """Panel b — empirical correct-rate vs stated confidence, Wilson 95% CI."""
    stat = {}
    for c in cal.ORDER:
        sub = [r for r in rows if r["confidence"] == c]
        stat[c] = (sum(1 for r in sub if r["verdict"] in cal.SUCCESS), len(sub))
    K = sum(s[0] for s in stat.values())
    N = sum(s[1] for s in stat.values())
    pooled = 100.0 * K / N

    x = np.arange(len(cal.ORDER))
    rate = [100.0 * stat[c][0] / stat[c][1] if stat[c][1] else 0.0 for c in cal.ORDER]
    ci = [cal.wilson(*stat[c]) for c in cal.ORDER]
    lo = [rate[i] - 100 * ci[i][0] for i in range(len(cal.ORDER))]
    hi = [100 * ci[i][1] - rate[i] for i in range(len(cal.ORDER))]

    ax.plot(x, rate, "-", lw=1.3, color=cal.MUTE, zorder=2)
    ax.errorbar(x, rate, yerr=[lo, hi], fmt="none", ecolor=cal.INK,
                elinewidth=1.5, capsize=6, capthick=1.5, zorder=3)
    for i, c in enumerate(cal.ORDER):
        ax.scatter(i, rate[i], s=160, color=cal.RAMP[c], edgecolor="white", linewidth=1.6, zorder=4)
        ax.text(i + 0.10, rate[i], f"{rate[i]:.0f}%  ({stat[c][0]}/{stat[c][1]})",
                ha="left", va="center", fontsize=14, fontweight="bold", color=cal.INK)
    ax.axhline(pooled, ls="--", lw=1.3, color=cal.MUTE, zorder=1)
    ax.text(-0.45, pooled + 1.5, f"Pooled {pooled:.0f}%", ha="left", va="bottom",
            fontsize=14, color=cal.MUTE)

    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(-0.5, 2.75)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in cal.ORDER])
    ax.set_xlabel("AI auditor's self-reported confidence")
    ax.set_ylabel("Human-rated correct & relevant, or\ncorrect but severity overstated (%)")
    ax.tick_params(axis="x", length=0)


def main():
    rows = ov.load()
    # Bump every font up substantially over the shared style.
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.8, 5.1))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.15, wspace=0.34)

    panel_verdicts(axA, rows)
    panel_calibration(axB, rows)
    for ax, tag in ((axA, "a"), (axB, "b")):
        ax.set_title(tag, loc="left", fontweight="bold", fontsize=19, pad=8)

    stem = "figS2_calibration"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote figures/{stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
