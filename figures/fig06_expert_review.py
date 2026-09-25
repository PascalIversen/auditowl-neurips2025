#!/usr/bin/env python3
"""Figure 6 - expert review: precision on the conservative pool, with the
paired inter-rater confusion matrices.

Run: python figures/fig06_expert_review.py
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

# --- one judgement per finding: the double-rated findings take the stricter verdict ----
CORR_RANK_CONS = {"True": 2, "Partially true": 1, "False": 0}
REL_RANK_CONS = {"Relevant": 2, "Minor": 1, "Trivial": 0}


def _more_conservative(a, b, rank):
    """The stricter of two verdicts on the same ordinal scale.

    Unverifiable/N/A sit outside the ordinal scale (`rank.get` is None for them), so a
    definite verdict from the other rater wins rather than being treated as tied; only
    when neither rater committed to a scale value does the result stay undecided.
    """
    ra, rb = rank.get(a), rank.get(b)
    if ra is None and rb is None:
        return a if pd.isna(b) else (b if pd.isna(a) else a)
    if ra is None:
        return b
    if rb is None:
        return a
    return a if ra <= rb else b


cons_double = paired.copy()
cons_double["correctness"] = [_more_conservative(a, b, CORR_RANK_CONS)
                              for a, b in zip(paired.correctness_r1, paired.correctness_r2)]
cons_double["relevance"] = [_more_conservative(a, b, REL_RANK_CONS)
                            for a, b in zip(paired.relevance_r1, paired.relevance_r2)]

_double_keys = set(zip(paired.paper, paired.finding_no))
cons_single = J[~J.apply(lambda r: (r.paper, r.finding_no) in _double_keys, axis=1)]

Jc = pd.concat([
    cons_double[["paper", "finding_no", "category", "correctness", "relevance"]],
    cons_single[["paper", "finding_no", "category", "correctness", "relevance"]],
], ignore_index=True)
Jc["stands"] = Jc["correctness"].map({"True": 1, "Partially true": 1, "False": 0})
Jc["fully_true"] = Jc["correctness"].map({"True": 1, "Partially true": 0, "False": 0})
Jc["at_least_minor"] = Jc["relevance"].map({"Relevant": 1, "Minor": 1, "Trivial": 0})
Jc["is_relevant"] = Jc["relevance"].map({"Relevant": 1, "Minor": 0, "Trivial": 0})

corr_counts = Jc.correctness.value_counts().reindex(CORR_ORDER).fillna(0).astype(int)
assert corr_counts.tolist() == [108, 9, 2, 1], corr_counts.to_dict()
stands_mask = Jc.correctness.isin(["True", "Partially true"])
rel_counts = Jc.loc[stands_mask, "relevance"].value_counts().reindex(
    ["Relevant", "Minor", "Trivial", "Unverifiable"]).fillna(0).astype(int)
assert rel_counts.tolist() == [73, 30, 10, 4], rel_counts.to_dict()
print(f"conservative pool: {len(Jc)} findings, one judgement each "
      f"({len(paired)} double-rated findings take the stricter of their two verdicts)")
print("correctness:", corr_counts.to_dict())
print("relevance, among the", int(stands_mask.sum()), "findings that stand:", rel_counts.to_dict())


# Correctness bars are out of all 118 findings; relevance bars are out of the 108 that
# still stand, since relevance is not asked of a finding that doesn't. An Unverifiable
# verdict counts in the denominator (it is not dropped), it just never counts as a "yes".
def prop_conservative(df, col, cluster="paper"):
    v = df[col].fillna(0)
    k, n = int(v.sum()), int(len(v))
    lo, hi = ah.wilson_ci(k, n)
    boot = ah.cluster_bootstrap(pd.DataFrame({col: v.values, cluster: df[cluster].values}),
                                cluster, lambda d: d[col].mean(), n_boot=N_BOOT, seed=SEED)
    return dict(k=k, n=n, p=k / n, lo=lo, hi=hi, boot_lo=boot["lo"], boot_hi=boot["hi"])


Jc_stands = Jc[stands_mask]
prec_rows_cons = [
    dict(label="Claim stands\n(True or Partially true)", key="stands",  colour="#1b6f8f", df=Jc),
    dict(label="Fully True",                             key="fully_true", colour="#4d93ad", df=Jc),
    dict(label="Relevance at least Minor",               key="at_least_minor", colour="#2f6b4f", df=Jc_stands),
    dict(label="Relevance = Relevant",                   key="is_relevant", colour="#8fc0a4", df=Jc_stands),
]
for r in prec_rows_cons:
    r.update(prop_conservative(r.pop("df"), r["key"]))

prec_tab_cons = pd.DataFrame(prec_rows_cons)[["label", "k", "n", "p", "lo", "hi", "boot_lo", "boot_hi"]]
prec_tab_cons["label"] = prec_tab_cons["label"].str.replace("\n", " ")
print(prec_tab_cons.round(4))

# confusion matrices for the 55 double-rated findings
cm_c = ah.confusion(paired.correctness_r1, paired.correctness_r2, CORR_ORDER)
cm_r = ah.confusion(paired.relevance_r1, paired.relevance_r2, ah.RELEVANCE_ORDER)

fig = plt.figure(figsize=(W_FULL, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.65], wspace=.5,
                      left=.09, right=.985, top=.78, bottom=.32)

# ---- left: fig 6a, recomputed on the conservative one-judgement-per-finding pool -----
ax = fig.add_subplot(gs[0, 0])
wilson_bar(ax, prec_rows_cons, "% of judged findings", "")
y = np.arange(len(prec_rows_cons))[::-1]
for y_, r in zip(y, prec_rows_cons):        # paper-clustered interval, drawn just under Wilson
    yb = y_ - .09
    ax.plot([100 * r["boot_lo"], 100 * r["boot_hi"]], [yb, yb], color="#c1440e", lw=1.3)
    for xx in (100 * r["boot_lo"], 100 * r["boot_hi"]):
        ax.plot([xx, xx], [yb - .07, yb + .07], color="#c1440e", lw=1.3)
ax.set_xlim(right=100)     # the value axis (percentage) caps at 100; % labels still draw
                           # just past the spine, in the wspace gap before subplot b
ax.plot([], [], color="#111111", lw=1.6, label="Wilson 95%")
ax.plot([], [], color="#c1440e", lw=1.6, label="Paper-clustered bootstrap 95%")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -.26), ncol=2, fontsize=7.0)
ax.set_title(f"Precision, conservative pooling", loc="left")
ax.text(-.4, 1.20, "a", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top")

# ---- right: figS3's confusion matrices (panel c of the S1-S4 composite) --------------
gs_c = gs[0, 1].subgridspec(1, 2, wspace=.55)
for j, (M, ttl, cmap, fixed_order) in enumerate([
        (cm_c, "Correctness", "Blues",   CORR_ORDER),
        (cm_r, "Relevance",   "Oranges", None)]):
    axc = fig.add_subplot(gs_c[0, j])
    if fixed_order is not None:
        # Correctness has exactly 4 possible verdicts (True / Partially true / False /
        # Unverifiable). Always reindex onto all four, on BOTH axes, so the matrix stays
        # square and a row label always lines up with the matching column label.
        Mv = M.reindex(index=fixed_order, columns=fixed_order, fill_value=0)
    else:
        keep = (M.sum(axis=1) > 0) | (M.sum(axis=0) > 0)   # drop only if empty on BOTH axes
        Mv = M.loc[keep, keep]
    assert Mv.to_numpy().sum() == M.to_numpy().sum(), f"{ttl}: reindex dropped a judgement"
    axc.imshow(np.log1p(Mv.to_numpy()), cmap=cmap, vmin=0)
    for r_ in range(Mv.shape[0]):
        for c_ in range(Mv.shape[1]):
            v = int(Mv.iloc[r_, c_])
            if v:
                axc.text(c_, r_, str(v), ha="center", va="center", fontsize=7.3,
                         color="white" if np.log1p(v) > np.log1p(Mv.to_numpy().max()) * .55
                         else "#222")
    axc.set_xticks(range(Mv.shape[1]), Mv.columns, rotation=35, ha="right", fontsize=6.7)
    axc.set_yticks(range(Mv.shape[0]), Mv.index, fontsize=6.7)
    axc.set_xlabel("Second judgement", fontsize=7.2)
    axc.set_ylabel("First judgement" if j == 0 else "", fontsize=7.2)
    axc.set_title(ttl, loc="left", fontsize=8.0)
    for s in axc.spines.values():
        s.set_visible(True)
    if j == 0:
        axc.text(-.55, 1.20, "b", transform=axc.transAxes, fontsize=11,
                 fontweight="bold", va="top")

save(fig, "fig06_expert_review",
     "Precision on the conservative pool, with the paired confusion matrices for context.",
     "width=\\linewidth")
plt.show()
