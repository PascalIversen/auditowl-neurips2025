#!/usr/bin/env python3
"""Human-eval figure, overall — one bar per verdict, severity-shaded, pooled.

There are TWO ways to aggregate across papers; this script emits a standalone
figure for each, plus a high-confidence-only view:

  * fig_human_eval_overall_micro     — MICRO: all findings in one box, count/80
                                       (big papers count more).
  * fig_human_eval_overall_macro     — MACRO: each paper's rate computed first,
                                       then the 5 rates averaged (every paper
                                       weighted equally).
  * fig_human_eval_overall_highconf  — MICRO, restricted to high-confidence
                                       findings (n=55).

Each bar is split by severity — low (lightest) / medium / high (dark blue),
stacked dark-at-base — on the full 0-100% scale. 'unsure' is omitted from the
bars. The same blue ramp keys the calibration panel's confidence dots.

Run:    python studies/reliability/human_eval_panels.py   (run aggregate first)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
BASE = "#3B7A9E"             # mid of a single-hue blue ramp keyed to severity

ORDER = ["correct_relevant", "correct_not_relevant", "correct_wrong_severity", "false"]
LABEL = {
    "correct_relevant": "correct &\nrelevant",
    "correct_not_relevant": "correct but\nnot relevant",
    "correct_wrong_severity": "correct but\nseverity overstated",
    "false": "false",
}


def shade(hex_color: str, f: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * f) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


# severity ramp (low -> high), shared with the calibration panel's confidence ramp
SEV_STACK = ["high", "medium", "low"]                  # bottom -> top: dark anchored to the axis
SEV_COLOR = {"low": shade(BASE, 0.62), "medium": BASE, "high": "#21506C"}
SEV_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}


def load() -> list[dict]:
    with (DATA / "human_eval_findings.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def series(rows: list[dict], agg: str):
    """Return (cats, sev_pct, bar_labels) under 'micro' or 'macro' aggregation.

    sev_pct[verdict][severity] is the % of findings at that low/medium/high tier
    (the three stack to the verdict's total share).
    """
    papers = sorted({r["paper"] for r in rows})
    sevs = ("low", "medium", "high")
    if agg == "micro":
        N = len(rows)
        c = {k: {s: 0 for s in sevs} for k in ORDER}
        for r in rows:
            if r["verdict"] in c and r["severity"] in c[r["verdict"]]:
                c[r["verdict"]][r["severity"]] += 1
        cats = [k for k in ORDER if sum(c[k].values()) > 0]
        pct = {k: {s: 100.0 * c[k][s] / N for s in sevs} for k in cats}
        lab = {k: f"{sum(pct[k].values()):.0f}%  ({sum(c[k].values())})" for k in cats}
    else:  # macro: per-paper rate first, then mean over papers
        nper = {p: sum(1 for r in rows if r["paper"] == p) for p in papers}
        c = {p: {k: {s: 0 for s in sevs} for k in ORDER} for p in papers}
        for r in rows:
            if r["verdict"] in ORDER and r["severity"] in sevs:
                c[r["paper"]][r["verdict"]][r["severity"]] += 1
        cats = [k for k in ORDER if sum(c[p][k][s] for p in papers for s in sevs) > 0]
        pct = {k: {s: float(np.mean([100.0 * c[p][k][s] / nper[p] for p in papers]))
                   for s in sevs} for k in cats}
        lab = {k: f"{sum(pct[k].values()):.0f}%" for k in cats}
    return cats, pct, lab


def build(rows: list[dict], stem: str, subtitle: str, agg: str = "micro"):
    cats, pct, lab = series(rows, agg)
    x = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.11, right=0.97)

    ax.set_xlim(-0.5, len(cats) - 0.5)

    bottom = np.zeros(len(cats))                             # stack high (dark) -> low (light)
    for s in SEV_STACK:
        vals = np.array([pct[k][s] for k in cats])
        ax.bar(x, vals, width=0.62, bottom=bottom, color=SEV_COLOR[s],
               edgecolor="white", linewidth=1.2, zorder=3)
        bottom += vals
    for xi, k, tot in zip(x, cats, bottom):
        ax.text(xi, tot + 1.2, lab[k], ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in cats], fontsize=9.5)
    ax.set_ylabel("Share of findings (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="x", length=0)

    sev_h = [Patch(facecolor=SEV_COLOR[s], edgecolor="white", label=SEV_LABEL[s])
             for s in ("high", "medium", "low")]
    leg = ax.legend(handles=sev_h, loc="upper right", title="Severity",
                    title_fontsize=9, alignment="left")
    leg._legend_box.sep = 10                                  # keep the title clear of the swatches

    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote figures/{stem}.{{png,pdf,svg}}  ·  "
          + " ".join(f"{k.split('_')[0]}…={sum(pct[k].values()):.0f}%" for k in cats))


def main():
    rows = load()
    npapers = len({r["paper"] for r in rows})
    build(rows, "fig_human_eval_overall_micro",
          f"all findings pooled · micro-average (n={len(rows)})", agg="micro")
    build(rows, "fig_human_eval_overall_macro",
          f"per-paper mean · macro-average ({npapers} papers)", agg="macro")
    hi = [r for r in rows if r["confidence"] == "high"]
    build(hi, "fig_human_eval_overall_highconf",
          f"high-confidence findings · pooled (n={len(hi)})", agg="micro")


if __name__ == "__main__":
    main()
