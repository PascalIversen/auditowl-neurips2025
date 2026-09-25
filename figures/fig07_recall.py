"""Regenerates synthetic_vs_real*.{pdf,png} directly from the underlying data files —
no hardcoded detection counts. Sources:
  - studies/recall/synthetic/data/matches_final.json + ANSWER_KEY_<pid>.json  (synthetic arm)
  - studies/recall/real/results.json                                     (real arm, AuditOwl)
  - studies/recall/real/baselines.json                                   (real arm, baselines)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
RECALL_DIR = os.path.join(os.path.dirname(HERE), "studies", "recall")
SYNTHETIC_ARM = os.path.join(RECALL_DIR, "synthetic")
REAL_ARM = os.path.join(RECALL_DIR, "real")

PAPERS = ["1023", "913", "2657", "1717", "1908", "2167", "2578", "4090", "1171", "3463"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------- Synthetic arm: the synthetic recall arm ----------------
def synthetic_counts():
    matches = load_json(os.path.join(SYNTHETIC_ARM, "data", "matches_final.json"))["assignments"]
    hits = {}  # (paper, sid) -> set of runs where detected
    for a in matches:
        if a["assigned_to"].startswith("seed:"):
            sid = a["assigned_to"].split(":", 1)[1]
            hits.setdefault((a["paper"], sid), set()).add(a["run"])

    r1_hit = r2_hit = union_hit = total = 0
    for pid in PAPERS:
        ak = load_json(os.path.join(SYNTHETIC_ARM, f"ANSWER_KEY_{pid}.json"))
        for s in ak["seeds"]:
            runs = hits.get((pid, s["id"]), set())
            total += 1
            if "run_01" in runs:
                r1_hit += 1
            if "run_02" in runs:
                r2_hit += 1
            if runs:
                union_hit += 1
    return [(r1_hit, total), (r2_hit, total), (union_hit, total)]


# ---------------- Real arm: the real recall arm ----------------
def real_counts_and_baselines():
    results = load_json(os.path.join(REAL_ARM, "results.json"))
    headtohead = load_json(os.path.join(REAL_ARM, "baselines.json"))

    r1_hit = r2_hit = union_hit = total = 0
    for it in results["per_item"]:
        total += 1
        if it["run_01"] == "DET":
            r1_hit += 1
        if it["run_02"] == "DET":
            r2_hit += 1
        if it["union"] == "DET":
            union_hit += 1
    auditowl = [(r1_hit, total), (r2_hit, total), (union_hit, total)]

    baseline_names = ["gpt-5", "gemini-2.5-pro", "gpt-5-mini"]
    baselines = {}
    for name in baseline_names:
        hit = n = 0
        for it in headtohead["per_item"]:
            n += 1
            hit += (it[name] or 0)
        baselines[name] = (hit, n)
    return auditowl, baselines


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": "#8a887f",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#3a3936",
    "ytick.color": "#3a3936",
    "svg.fonttype": "none",
})

ORANGE = "#eb6834"   # AuditOwl, both arms
GRAYS = ["#52514e", "#8a887f", "#c3c2b7"]  # baseline models, darkest->lightest


def make_figure(synthetic, real, baselines, real_label, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [0.62, 1]})

    # ---------------- Panel (a): synthetic only, run 1 / run 2 / union ----------------
    ax = axes[0]
    groups = ["AuditOwl\nrun 1", "AuditOwl\nrun 2", "AuditOwl\nunion"]

    x = np.array([0.0, 0.68, 1.36])
    w = 0.46
    ps = [k / n for k, n in synthetic]
    ax.bar(x, ps, width=w, color=ORANGE, label="Synthetic (seeded defects)", zorder=3)
    for xo, p, (k, n) in zip(x, ps, synthetic):
        ax.text(xo, p + 0.03, f"{k}/{n}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#0b0b0b")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11.2)
    ax.set_xlim(x[0] - w * 0.85, x[-1] + w * 0.85)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.8)
    ax.set_ylabel("Detection rate", fontsize=12)
    ax.grid(axis="y", color="#e8e6e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("(a)  Synthetic", fontsize=12, loc="left", pad=14)

    # ---------------- Panel (b): real-arm head-to-head vs baseline models ----------------
    ax2 = axes[1]
    systems = ["AuditOwl\nrun 1", "AuditOwl\nrun 2", "AuditOwl\nunion", "GPT-5", "Gemini\n2.5 Pro", "GPT-5\nmini"]
    vals = list(real) + [baselines["gpt-5"], baselines["gemini-2.5-pro"], baselines["gpt-5-mini"]]
    colors = [ORANGE, ORANGE, ORANGE, GRAYS[0], GRAYS[1], GRAYS[2]]

    xb = np.arange(len(systems))
    ps = [k / n for k, n in vals]
    ax2.bar(xb, ps, width=0.5, color=colors, zorder=3)
    for xo, p, (k, n) in zip(xb, ps, vals):
        ax2.text(xo, p + 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=11,
                  fontweight="bold", color="#0b0b0b")

    ax2.set_xticks(xb)
    ax2.set_xticklabels(systems, fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.8)
    ax2.grid(axis="y", color="#e8e6e0", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.set_title(f"(b)  Real ({real_label})", fontsize=12, loc="left", pad=14)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.16, wspace=0.28)
    fig.savefig(os.path.join(OUT, f"{out_stem}.pdf"))
    fig.savefig(os.path.join(OUT, f"{out_stem}.png"), dpi=220)
    plt.close(fig)


synthetic = synthetic_counts()

real, baselines = real_counts_and_baselines()
n = real[0][1]
print(f"real: run1={real[0]} run2={real[1]} union={real[2]}  baselines={baselines}")
make_figure(synthetic, real, baselines, f"SciCoQA, {n} most-recent", "fig07_recall")

print(f"synthetic: run1={synthetic[0]} run2={synthetic[1]} union={synthetic[2]}")
print("done")
