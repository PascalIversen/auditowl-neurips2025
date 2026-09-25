#!/usr/bin/env python3
"""Figure S4 - reviewer effort: minutes per finding and per reviewer-paper sitting.

Run: python figures/figS4_reviewer_time.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # headless: never open a window
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "studies" / "expert_review"))
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Locate the directory containing human_eval.py / gpt_eval.py, regardless
# of the working directory this notebook happens to be launched from.
_here = _ROOT / 'studies' / 'expert_review'
_candidates = [_here]
for _p in _candidates:
    if (_p / "human_eval.py").exists():
        sys.path.insert(0, str(_p))
        break
else:
    raise FileNotFoundError(
        "Could not find human_eval.py / gpt_eval.py. Run this notebook "
        "from the 'human_verification_analysis' directory (or its parent)."
    )

import human_eval as ah
import gpt_eval as ge

warnings.filterwarnings("ignore", category=FutureWarning)
ah.use_paper_style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

N_BOOT = 1500
SEED = 20260729

print("pandas", pd.__version__, "| numpy", np.__version__, "| matplotlib", mpl.__version__)

W_FULL, W_NARROW = 9.6, 6.4
PAGE_SCALE = 0.677

OUT = OUT_DIR



def save(fig, name, caption, width_cmd):
    """Write PDF + PNG to outputs/paper_figures."""
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    print(f"  wrote {name}.pdf / .png      \\includegraphics[{width_cmd}]{{figs/{name}.pdf}}")


def wilson_bar(ax, rows, xlabel, title, colour_key=None):
    """Horizontal proportion bars with Wilson intervals and n printed on the bar."""
    y = np.arange(len(rows))[::-1]
    wy = .08    # how far above bar-centre the Wilson line and its end-caps sit
    for y_, r in zip(y, rows):
        ax.barh(y_, 100 * r["p"], color=r["colour"], height=.62)
        yw = y_ + wy
        ax.plot([100 * r["lo"], 100 * r["hi"]], [yw, yw], color="#111111", lw=1.3)
        for xx in (100 * r["lo"], 100 * r["hi"]):
            ax.plot([xx, xx], [yw - .10, yw + .10], color="#111111", lw=1.3)
        ax.text(100 * r["hi"] + 2.0, y_, f"{100*r['p']:.1f}%", va="center", fontsize=7.5)
        ax.text(1.5, y_, f"{r['k']}/{r['n']}", va="center", ha="left", fontsize=9,
                color="black")
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 122)
    ax.axvline(100, color="#ccc", lw=.7, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")


print(f"figures -> {OUT}")
print(f"draw at {W_FULL} in (full) / {W_NARROW} in (narrow); both reach the page at "
      f"{PAGE_SCALE:.3f}x on a 6.5 in text block")

e1 = ah.load_workbook_findings("expert")
e2 = ah.load_workbook_findings("expert2")
e2_done = e2[e2.rated].copy()

J = ge.human_judgements(e1, e2)                        # 175 pooled judgements, 120 findings
paired = ah.build_paired(e1[e1.rated], e2[e2.rated])   # 55 double-rated findings

CORR_ORDER = ah.CORRECTNESS_ORDER
REL3 = ["Relevant", "Minor", "Trivial"]

# the headline numbers the paper leads with, asserted so a stale workbook fails here
assert (int(J.stands.sum()), int(J.stands.notna().sum())) == (171, 173)
assert (int(J.at_least_minor.sum()), int(J.at_least_minor.notna().sum())) == (143, 155)
assert int((J.correctness == "False").sum()) == 2

print(f"pooled expert judgements : {len(J)} over "
      f"{J.groupby(['paper','finding_no']).ngroups} findings, {J.paper.nunique()} papers")
print(f"double-rated subset      : {len(paired)} findings, {paired.paper.nunique()} papers")
print("\nheadline check: 171/173 stands, 143/155 at-least-minor, 2/175 false  OK")

mins = J.minutes.dropna()
sitting = J.groupby(["paper", "pass_"]).minutes.sum()          # one reviewer, one paper
sitting_first = J[J.pass_ == "first"].groupby("paper").minutes.sum()

print(f"{len(mins)} of {len(J)} judgements recorded a time; "
      f"total {mins.sum()/60:.1f} h, median {mins.median():.0f} min "
      f"(IQR {mins.quantile(.25):.0f}-{mins.quantile(.75):.0f})")
print(f"per reviewer-sitting (paper x pass, n={sitting.size}): median "
      f"{sitting.median():.0f} min   <- plotted in panel (b) and quoted in the caption")
print(f"  first passes only (n={sitting_first.size})          : median "
      f"{sitting_first.median():.0f} min")

fig, axes = plt.subplots(1, 2, figsize=(W_NARROW, 2.7), gridspec_kw=dict(wspace=.36))

ax = axes[0]
ax.hist(mins.clip(upper=60), bins=np.arange(0, 65, 5), color="#1b6f8f", alpha=.85)
ax.axvline(mins.median(), color="#b0403a", lw=1.3, ls="--")
ax.text(mins.median() + 1.8, ax.get_ylim()[1] * .88, f"median\n{mins.median():.0f} min",
        fontsize=7, color="#b0403a")
ax.set_xlabel("Minutes to adjudicate one finding\n(clipped at 60)")
ax.set_ylabel("Judgments")
ax.set_xlim(left=0)

ax.text(-0.25, 1.06, "a", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top")

ax = axes[1]
ax.hist(sitting.clip(upper=180), bins=np.arange(0, 195, 15), color="#2f6b4f", alpha=.85)
ax.axvline(sitting.median(), color="#b0403a", lw=1.3, ls="--")
ax.text(sitting.median() + 5, ax.get_ylim()[1] * .88, f"median\n{sitting.median():.0f} min",
        fontsize=7, color="#b0403a")
ax.set_xlim(left=0)
ax.set_xlabel("Minutes one reviewer spent on one\npaper's audit (clipped at 180)")
ax.set_ylabel("Reviewer-paper sittings")

ax.text(-0.25, 1.06, "b", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top")

fig.subplots_adjust(left=.12, right=.985, top=.86, bottom=.30, wspace=.36)
save(fig, "figS4_reviewer_time", "Measured reviewer effort.", "width=0.667\\linewidth")
plt.show()
