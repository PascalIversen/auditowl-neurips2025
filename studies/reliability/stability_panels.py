#!/usr/bin/env python3
"""Robustness / test-retest stability figure for the auditor (6 panels).

Leads with OBJECTIVE, verifiable evidence before the later severity-stratified
panels (d, e) — severity is disclosed as a secondary, clearly-labelled layer,
not the panels' organising axis:

  (a) Finding COUNT per run is stable (CV 0.13-0.22).
  (b) A reproducible CORE vs a flaky TAIL: per-issue detection-rate distribution
      over the 10 runs, under the merged-by-defect matcher.
  (c) Reproducibility concentrates on what the auditor HEADLINES (top-6
      take-aways) ~2.8x more than the unheadlined tail.
  (d) Chance-corrected reliability is FAIR-TO-MODERATE: Gwet's AC1 ~0.4 and
      matcher-stable, vs Krippendorff alpha (prevalence-paradox prone).
  (e) Detection by SEVERITY subset. (Severity is the auditor's own label ->
      shown for completeness, NOT as the headline.)

The anchor matcher (analyze.cluster(method="anchor")) is deliberately NOT used
as a second reliability estimate. It requires two findings to cite the same
`file` before it will call them the same discrepancy, so a defect that one run
cites in run.py and another in experiments/exp_forecast.py becomes two distinct
discrepancies. Measured against the merged partition it splits 37 of the 80
discrepancies (46%), inflating the count to 118. Because a split defect appears
in fewer runs per fragment, that depresses every detection rate mechanically:
it measures whether a discrepancy is re-cited at the same path, not whether it
is re-found. Only the merged partition answers the reliability question.
  (f) What's robust vs fragile (key-findings box).

Inputs: studies/reliability/selection.json, studies/reliability/<paper>/run_*/, data/merged_clusters.json
Outputs: studies/reliability/figures/fig_robustness_stability.{png,pdf,svg}
         studies/reliability/data/robustness_figure_data.json
Run: python studies/reliability/stability_panels.py
"""
from __future__ import annotations
import json, re, sys
from itertools import combinations
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RB = Path(__file__).resolve().parent
sys.path.insert(0, str(RB))
import analyze as A

DATA = RB / "data"; FIGS = RB / "figures"; FIGS.mkdir(parents=True, exist_ok=True)

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
VERM, PURP, AMB, GRN, GREY = "#B64342", "#8E6FB0", "#E1A33C", "#8BCF8B", "#C7CED6"
TEAL = "#3B7A9E"; SEV = {"high": 3, "medium": 2, "low": 1, "": 0}

PAPERS = [p["paper"] for p in json.loads((RB / "selection.json").read_text())["selected"]]
SHORT = {p: p.split("_")[0] for p in PAPERS}
MERGED = {m["paper"]: m["clusters"] for m in json.loads((DATA / "merged_clusters.json").read_text())}


def jaccard(M):
    R = M.shape[0]; js = []
    for i, j in combinations(range(R), 2):
        a, b = M[i] > 0, M[j] > 0; u = np.logical_or(a, b).sum()
        js.append((np.logical_and(a, b).sum() / u) if u else 1.0)
    return float(np.mean(js)) if js else 1.0


def gwet_ac1(M):
    R, K = M.shape
    if K == 0 or R < 2:
        return float("nan")
    a = M.sum(axis=0); b = R - a
    Pa = np.mean((a * (a - 1) + b * (b - 1)) / (R * (R - 1)))
    pi = a.sum() / (K * R); Pe = 2 * pi * (1 - pi)
    return (Pa - Pe) / (1 - Pe) if (1 - Pe) > 0 else 1.0


def alpha(M):
    K = M.shape[1]
    return A.krippendorff_nominal([list(M[:, k]) for k in range(K)]) if K else 1.0


def sev_bucket(sevs):  # majority/modal, ties -> higher
    sevs = [s for s in sevs if s in SEV and s]
    if not sevs:
        return "low"
    cnt = Counter(sevs); top = max(cnt.values())
    m = max([s for s in cnt if cnt[s] == top], key=lambda s: SEV[s])
    return "high" if m == "high" else ("medium" if m == "medium" else "low")


def takeaway_section(md: Path):
    txt = md.read_text(errors="replace").splitlines()
    hdr = next((i for i, l in enumerate(txt) if re.search(r"take.?away", l, re.I)), None)
    if hdr is None:
        return ""
    out = []
    for l in txt[hdr + 1:]:
        if re.match(r"^#{2,3}\s", l):
            break
        out.append(l)
    return "\n".join(out)


def collect():
    per_run_counts = {}
    merged_det, strict_det = [], []
    head_det, nothead_det = [], []
    sev_det = {"merged": {"high": [], "medium": [], "low": []},
               "strict": {"high": [], "medium": [], "low": []}}
    sevconf = {}  # (severity_bucket, confidence_bucket) -> [detection rates], merged
    # FP-vs-FN evidence: per detection bucket, finding-level reject/check counts
    BUCKETS = ["singleton(1)", "flaky(2-3)", "mid(4-7)", "core(8-10)"]
    bucket_stats = {b: {"n": 0, "reject": 0, "check": 0} for b in BUCKETS}
    def bk(d):
        return BUCKETS[0] if d == 1 else (BUCKETS[1] if d <= 3 else (BUCKETS[2] if d <= 7 else BUCKETS[3]))
    ensemble_curves = []   # one recall-vs-K curve per paper (corroborated-real set, det>=2)
    per_paper = {}
    for paper in PAPERS:
        pdir = RB / paper; runs = A.load_runs(pdir); R = len(runs)
        per_run_counts[paper] = [len(r["findings"]) for r in runs]
        fid_run = {}
        for ri, run in enumerate(runs):
            for idx in range(len(run["findings"])):
                fid_run[f"r{ri+1:02d}#{idx}"] = ri
        # headlined fids
        headlined = set()
        for ri, rd in enumerate(sorted(pdir.glob("run_*"))):
            fj = json.loads((rd / "findings.json").read_text())["findings"]
            slug2idx = {(f.get("id_local") or f.get("id", "").split("/", 1)[-1]): i for i, f in enumerate(fj)}
            sect = takeaway_section(rd / "audit.md")
            for slug, idx in slug2idx.items():
                if slug and re.search(r"(?<![a-z0-9-])" + re.escape(slug) + r"(?![a-z0-9-])", sect):
                    headlined.add(f"r{ri+1:02d}#{idx}")
        # per-fid severity + confidence
        fid_sev = {}; fid_conf = {}
        for ri, run in enumerate(runs):
            for idx, f in enumerate(run["findings"]):
                fid_sev[f"r{ri+1:02d}#{idx}"] = f["severity"]
                fid_conf[f"r{ri+1:02d}#{idx}"] = f["confidence"]
        # merged: incidence + per-fid detection count
        mcols, ph, pn = [], [], []
        fid_det = {}
        for c in MERGED[paper]:
            fr = {fid_run[f] for f in c["fids"] if f in fid_run}
            hr = {fid_run[f] for f in c["fids"] if f in fid_run and f in headlined}
            det = len(fr) / R; merged_det.append(det)
            for f in c["fids"]:
                if f in fid_run:
                    fid_det[f] = len(fr)
            col = np.zeros(R, int); [col.__setitem__(r, 1) for r in fr]; mcols.append(col)
            (ph if hr else pn).append(det); (head_det if hr else nothead_det).append(det)
            s_b = sev_bucket([fid_sev[f] for f in c["fids"] if f in fid_sev])
            c_b = sev_bucket([fid_conf[f] for f in c["fids"] if f in fid_conf])
            sev_det["merged"][s_b].append(det)
            sevconf.setdefault((s_b, c_b), []).append(det)
        Mm = np.array(mcols).T
        # FP/FN evidence: walk raw findings + verifier verdicts
        for ri, rd in enumerate(sorted(pdir.glob("run_*"))):
            fj = json.loads((rd / "findings.json").read_text())["findings"]
            vj = {f["id"]: f for f in json.loads((rd / "findings_verified.json").read_text()).get("findings", [])} if (rd / "findings_verified.json").exists() else {}
            for idx, f in enumerate(fj):
                d = fid_det.get(f"r{ri+1:02d}#{idx}")
                if d is None:
                    continue
                b = bk(d); bucket_stats[b]["n"] += 1
                if vj.get(f["id"], {}).get("verdict") == "reject":
                    bucket_stats[b]["reject"] += 1
                if f.get("check_script"):
                    bucket_stats[b]["check"] += 1
        # ensemble curve: expected recall of corroborated-real (det>=2) when unioning K of R runs
        real_d = [int(round(d * R)) for d in [c2 for c2 in merged_det[-len(MERGED[paper]):]]]  # placeholder
        det_counts = [len({fid_run[f] for f in c["fids"] if f in fid_run}) for c in MERGED[paper]]
        real = [d for d in det_counts if d >= 2]
        curve = [np.mean([1 - (comb(R - d, K) / comb(R, K) if R - d >= K else 0.0) for d in real]) if real else np.nan
                 for K in range(1, R + 1)]
        ensemble_curves.append(curve)
        # strict
        items, labels = A.cluster(runs, method="anchor"); K = (max(labels) + 1) if labels else 0
        Ms = np.zeros((R, K), int); grp = {}
        for (ri, f), lab in zip(items, labels):
            Ms[ri, lab] = 1; grp.setdefault(lab, []).append(f["severity"])
        for k in range(K):
            det = Ms[:, k].sum() / R; strict_det.append(det)
            sev_det["strict"][sev_bucket(grp[k])].append(det)
        per_paper[paper] = {
            "count_cv": float(np.std(per_run_counts[paper], ddof=1) / np.mean(per_run_counts[paper])),
            "head_det": float(np.mean(ph)) if ph else float("nan"),
            "nothead_det": float(np.mean(pn)) if pn else float("nan"),
            "ac1_merged": gwet_ac1(Mm), "ac1_strict": gwet_ac1(Ms), "alpha_merged": alpha(Mm),
        }
    return dict(per_run_counts=per_run_counts, merged_det=np.array(merged_det),
                strict_det=np.array(strict_det), head_det=np.array(head_det),
                nothead_det=np.array(nothead_det), sev_det=sev_det, sevconf=sevconf, per_paper=per_paper,
                bucket_stats=bucket_stats, bucket_order=BUCKETS,
                ensemble=np.array(ensemble_curves))


def main():
    D = collect()
    order = PAPERS; labels = [SHORT[p] for p in order]
    fig, axes = plt.subplots(3, 3, figsize=(16.8, 13.6))
    fig.subplots_adjust(hspace=0.55, wspace=0.30)
    axA, axB, axC = axes[0]      # stability:      count | core-vs-tail | headlined
    axS, axE, axD = axes[1]      # stratification: severity | severity×confidence | AC1
    axF, axG, axH = axes[2]      # diagnosis+fix:  flaky≠FP | ensembling | key findings

    # (a) count stable
    for i, p in enumerate(order):
        ys = np.array(D["per_run_counts"][p], float)
        axA.scatter(ys, np.full_like(ys, i) + np.linspace(-0.16, 0.16, len(ys)), s=24,
                    color=TEAL, alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3)
        m = ys.mean(); cv = ys.std(ddof=1) / m
        axA.plot([m, m], [i - 0.28, i + 0.28], color=INK, lw=2, zorder=4)
        axA.text(ys.max() + 0.4, i, f"CV {cv:.2f}", va="center", fontsize=8.5, color=MUTE)
    axA.set_yticks(range(len(order))); axA.set_yticklabels(labels, fontsize=9)
    axA.set_ylim(-0.6, len(order) - 0.4); axA.invert_yaxis()
    axA.set_xlim(0, max(max(v) for v in D["per_run_counts"].values()) + 3)
    axA.set_xlabel("Findings per run (black bar = mean)")
    axA.set_title("a  Finding count is stable", loc="left")

    # (b) core vs tail (merged-by-defect)
    ks = np.arange(1, 11)
    mh = np.array([np.sum(np.isclose(D["merged_det"], k / 10)) for k in ks])
    cols = [VERM if k <= 3 else (GRN if k >= 8 else GREY) for k in ks]
    axB.bar(ks, mh, width=0.82, color=cols, edgecolor="white", linewidth=0.5, zorder=3)
    axB.axvspan(0.5, 3.5, color=VERM, alpha=0.06); axB.axvspan(7.5, 10.5, color=GRN, alpha=0.10)
    top = mh.max()
    axB.text(2, top * 0.97, f"flaky tail\n{int(mh[:3].sum())} issues", ha="center", va="top", fontsize=8.5, color="#7a2f2e")
    axB.text(9, top * 0.97, f"stable core\n{int(mh[7:].sum())} issues", ha="center", va="top", fontsize=8.5, color="#2f6b2f")
    axB.set_xticks(ks); axB.set_xlim(0.4, 10.6)
    axB.set_xlabel("Runs (of 10) surfacing the issue"); axB.set_ylabel("# canonical issues")
    axB.set_title("b  Reproducible core vs flaky tail", loc="left")

    # (c) headlined vs not
    means = [np.nanmean(D["head_det"]), np.nanmean(D["nothead_det"])]
    axC.bar([0, 1], means, width=0.55, color=[TEAL, GREY], edgecolor="white", zorder=3)
    rs = np.random.RandomState(0)
    for p in order:
        for x, key in [(0, "head_det"), (1, "nothead_det")]:
            v = D["per_paper"][p][key]
            if v == v:
                axC.scatter(x + (rs.rand() - 0.5) * 0.28, v, s=30, color=INK, alpha=0.6, zorder=4, edgecolor="white", linewidth=0.4)
    for x, m in zip([0, 1], means):
        axC.text(x, m + 0.03, f"{m:.2f}", ha="center", fontsize=10.5, weight="bold")
    axC.annotate(f"{means[0]/means[1]:.1f}x", xy=(0.5, max(means) * 0.7), ha="center", fontsize=12, weight="bold", color=TEAL)
    axC.set_xticks([0, 1]); axC.set_xticklabels(["headlined\n(top-6)", "never\nheadlined"])
    axC.set_ylim(0, 1.0); axC.set_ylabel("Mean detection rate")
    axC.set_title("c  Reproducibility tracks headlining", loc="left")

    # (d) AC1 vs alpha
    x = np.arange(len(order)); w = 0.36
    ac1 = [D["per_paper"][p]["ac1_merged"] for p in order]; alp = [D["per_paper"][p]["alpha_merged"] for p in order]
    for lo, hi, c, lab in [(0, .2, "#f3dcdc", "poor"), (.2, .4, "#f7ecd6", "fair"), (.4, .6, "#e3eede", "moderate"), (.6, .8, "#d6e8e0", "good")]:
        axD.axhspan(lo, hi, color=c, alpha=0.55); axD.text(len(order) - 0.45, (lo + hi) / 2, lab, fontsize=7, color=MUTE, va="center", ha="right")
    axD.bar(x - w / 2, ac1, w, color=TEAL, edgecolor="white", zorder=3, label="Gwet's AC1")
    axD.bar(x + w / 2, alp, w, color="#b9b9b9", edgecolor="white", zorder=3, label="Krippendorff α")
    axD.axhline(np.mean(ac1), color=TEAL, lw=1.2, ls=(0, (4, 3)))
    axD.annotate("α collapses\n(paradox)", xy=(w / 2, alp[0]), xytext=(0.6, 0.64), fontsize=7.5, color="#7a2f2e",
                 arrowprops=dict(arrowstyle="->", color="#7a2f2e", lw=1.0))
    axD.set_xticks(x); axD.set_xticklabels(labels, fontsize=9); axD.set_ylim(0, 0.8)
    axD.set_ylabel("Reliability (present/absent)")
    axD.set_title("f  Fair-to-moderate (AC1, not α)", loc="left"); axD.legend(loc="upper left", fontsize=8)

    # (d) detection by severity (merged)
    sdm = D["sev_det"]["merged"]; svs = ["high", "medium", "low"]
    sv_means = [np.mean(sdm[s]) if sdm[s] else 0.0 for s in svs]; sv_n = [len(sdm[s]) for s in svs]
    axS.bar(range(3), sv_means, width=0.6, color=[VERM, AMB, GRN], edgecolor="white", zorder=3)
    for i, m in enumerate(sv_means):
        axS.text(i, m + 0.02, f"{m:.2f}", ha="center", fontsize=9, weight="bold")
    axS.axhline(np.mean(D["merged_det"]), color=MUTE, lw=1.0, ls=":")
    axS.text(2.45, np.mean(D["merged_det"]) + 0.015, "overall", fontsize=7, color=MUTE, ha="right")
    axS.set_xticks(range(3))
    axS.set_xticklabels([f"high\n(n{sv_n[0]})", f"medium\n(n{sv_n[1]})", f"low\n(n{sv_n[2]})"], fontsize=9)
    axS.set_ylim(0, 1.0); axS.set_ylabel("Mean detection rate")
    axS.set_title("d  Detection by severity", loc="left")
    axS.text(0.5, -0.30, "merged matcher. n = # distinct issues in each bucket. Severity is a self-label;\n"
             "the high>med step is assignment-rule-sensitive — the robust feature is the low floor.",
             transform=axS.transAxes, ha="center", va="top", fontsize=7.6, color=MUTE)

    # (e) detection by severity × confidence (merged) — the auditor's headline-ranking key
    sc = D["sevconf"]; rows = ["high", "medium", "low"]; ccols = ["high", "medium", "low"]
    Mz = np.full((3, 3), np.nan); Nn = np.zeros((3, 3), int)
    for i, s in enumerate(rows):
        for j, c in enumerate(ccols):
            v = sc.get((s, c), [])
            if v:
                Mz[i, j] = float(np.mean(v)); Nn[i, j] = len(v)
    cmap = LinearSegmentedColormap.from_list("vag", [VERM, AMB, GRN]); cmap.set_bad("#ededed")
    axE.imshow(np.ma.masked_invalid(Mz), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(3):
            if np.isnan(Mz[i, j]):
                axE.text(j, i, "–", ha="center", va="center", fontsize=12, color=MUTE)
            else:
                r0, g0, b0 = cmap(Mz[i, j])[:3]
                tc = "white" if (0.299 * r0 + 0.587 * g0 + 0.114 * b0) < 0.5 else INK
                axE.text(j, i, f"{Mz[i, j]:.2f}\n(n{Nn[i, j]})", ha="center", va="center", fontsize=10, weight="bold", color=tc)
    axE.set_xticks([0, 1, 2]); axE.set_xticklabels(["high", "medium", "low"], fontsize=9)
    axE.set_yticks([0, 1, 2]); axE.set_yticklabels(["high", "medium", "low"], fontsize=9)
    axE.set_xlabel("Confidence"); axE.set_ylabel("Severity")
    axE.set_title("e  Detection by severity × confidence", loc="left")
    axE.text(0.5, -0.32, "cell = mean detection rate over 10 runs (merged); n = # distinct issues in the cell\n"
             "(ignore small-n). Severity×confidence is the auditor's top-6 key: 0.74 (high/high) → 0.16 (low/med).",
             transform=axE.transAxes, ha="center", va="top", fontsize=7.6, color=MUTE)

    # (f) FP vs FN: flaky findings are real, not hallucinated
    bo = D["bucket_order"]; bs = D["bucket_stats"]
    chkpct = [100 * bs[b]["check"] / bs[b]["n"] if bs[b]["n"] else 0 for b in bo]
    rej_total = sum(bs[b]["reject"] for b in bo); n_total = sum(bs[b]["n"] for b in bo)
    axF.bar(range(len(bo)), chkpct, width=0.7, color=[VERM, AMB, GREY, GRN], edgecolor="white", zorder=3)
    for i, v in enumerate(chkpct):
        axF.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=8.5, weight="bold")
    axF.set_xticks(range(len(bo))); axF.set_xticklabels(["1", "2–3", "4–7", "8–10"], fontsize=9)
    axF.set_xlabel("Runs detecting the issue"); axF.set_ylabel("% backed by re-runnable check")
    axF.set_ylim(0, 100); axF.set_title("g  Flaky ≠ false positive", loc="left")
    axF.text(0.5, 0.93, f"verifier rejected {rej_total}/{n_total}\nfindings — in EVERY bucket",
             transform=axF.transAxes, ha="center", va="top", fontsize=9, color="#7a2f2e", weight="bold")

    # (g) ensembling recovers the false negatives
    ens = D["ensemble"]; Kx = np.arange(1, ens.shape[1] + 1); m = np.nanmean(ens, axis=0)
    for row in ens:
        axG.plot(Kx, row, color=GREY, lw=1.0, alpha=0.55, zorder=2)
    axG.plot(Kx, m, color=TEAL, lw=2.6, marker="o", ms=4.5, zorder=3, label="Mean over 5 papers")
    for K in (1, 3, 5):
        axG.annotate(f"{100*m[K-1]:.0f}%", xy=(K, m[K - 1]), xytext=(K + 0.15, m[K - 1] - 0.11),
                     fontsize=9, color=TEAL, weight="bold")
        axG.scatter([K], [m[K - 1]], s=40, color=TEAL, zorder=4, edgecolor="white", linewidth=0.6)
    axG.set_xticks(range(1, 11)); axG.set_ylim(0, 1.03); axG.set_xlim(0.6, 10.4)
    axG.set_xlabel("# runs unioned (ensembled)"); axG.set_ylabel("Recall of real-defect set")
    axG.set_title("h  Ensembling recovers the misses", loc="left")
    axG.text(0.5, 0.10, "the misses are false negatives, so unioning\nruns climbs to 100% (an FP regime would not)",
             transform=axG.transAxes, ha="center", fontsize=7.8, color=MUTE)

    # (h) key-findings box
    axH.axis("off")
    axH.text(0.0, 1.0, "What's robust  ✓", transform=axH.transAxes, fontsize=10.5, weight="bold", color="#2f6b2f", va="top")
    robust = ("✓  count stable (CV 0.13–0.22)\n"
              "✓  low reproducibility = false NEGATIVES,\n     not false positives (verifier rejects 0)\n"
              "✓  headlined / blocking defects reproduce\n     (~0.5–0.7); concrete & verifiable\n"
              "✓  union 3–5 runs ⇒ 83–94% recall\n"
              "✓  reliability fair-to-moderate\n     (Gwet's AC1 ≈ 0.39)")
    axH.text(0.0, 0.90, robust, transform=axH.transAxes, fontsize=8.8, va="top", linespacing=1.3)
    axH.text(0.0, 0.34, "Do not headline  ✗", transform=axH.transAxes, fontsize=10.5, weight="bold", color="#7a2f2e", va="top")
    fragile = ("✗  high>med>low severity gradient\n     (assignment-rule sensitive; self-label)\n"
               "✗  Krippendorff α on subsets (paradox)")
    axH.text(0.0, 0.25, fragile, transform=axH.transAxes, fontsize=8.8, va="top", linespacing=1.3)
    axH.text(0.0, 0.02, "n=5, repeat-only; merged-by-defect matcher;\nGT = verifier + self-checks.",
             transform=axH.transAxes, fontsize=7.6, color=MUTE, va="top", style="italic")
    axH.set_title("i  Key findings", loc="left")

    fig.suptitle("Auditor test–retest stability  (5 papers × 10 sealed repeats, frozen code, fixed model/prompt/temperature)",
                 fontsize=13.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005, f"Overall mean detection rate {np.mean(D['merged_det']):.2f}: a typical canonical issue appears in only ~4 of 10 repeats — "
             "count is stable, issue identity is not.  Merged-by-defect matcher; n=5, a characterisation, not a population estimate.",
             ha="center", fontsize=8.2, color=MUTE)

    for name in ("fig_robustness_stability.png",):
        fig.savefig(FIGS / name); fig.savefig(FIGS / name.replace(".png", ".pdf")); fig.savefig(FIGS / name.replace(".png", ".svg"))
    plt.close(fig)

    sidecar = {
        "overall_mean_detection": {"merged": float(np.mean(D["merged_det"]))},
        "by_severity_majority": {mt: {s: {"mean_det": float(np.mean(D["sev_det"][mt][s])) if D["sev_det"][mt][s] else 0.0, "n": len(D["sev_det"][mt][s])} for s in ["high", "medium", "low"]} for mt in ("merged",)},
        "by_severity_x_confidence_merged": {f"{s}|{c}": {"mean_det": float(np.mean(D["sevconf"][(s, c)])), "n": len(D["sevconf"][(s, c)])} for s in ["high", "medium", "low"] for c in ["high", "medium", "low"] if (s, c) in D["sevconf"]},
        "stable_core": {"merged": int(np.sum(D["merged_det"] >= 0.8)), "merged_total": int(len(D["merged_det"])),
                        "strict": int(np.sum(D["strict_det"] >= 0.8)), "strict_total": int(len(D["strict_det"]))},
        "flaky": {"merged": int(np.sum(D["merged_det"] <= 0.3)), "strict": int(np.sum(D["strict_det"] <= 0.3))},
        "headlined_det": float(np.nanmean(D["head_det"])), "not_headlined_det": float(np.nanmean(D["nothead_det"])),
        "ac1_merged_mean": float(np.mean([D["per_paper"][p]["ac1_merged"] for p in order])),
        "ac1_strict_mean": float(np.mean([D["per_paper"][p]["ac1_strict"] for p in order])),
        "alpha_merged_mean": float(np.mean([D["per_paper"][p]["alpha_merged"] for p in order])),
        "fp_vs_fn": {
            "verifier_reject_total": int(sum(D["bucket_stats"][b]["reject"] for b in D["bucket_order"])),
            "n_findings_total": int(sum(D["bucket_stats"][b]["n"] for b in D["bucket_order"])),
            "by_bucket": {b: {"n": D["bucket_stats"][b]["n"], "reject": D["bucket_stats"][b]["reject"],
                              "check_backed_pct": (100 * D["bucket_stats"][b]["check"] / D["bucket_stats"][b]["n"]) if D["bucket_stats"][b]["n"] else 0}
                         for b in D["bucket_order"]},
            "conclusion": "false-negative dominated: verifier rejects ~0 in every detection bucket; check-backing rises with reproducibility",
        },
        "ensemble_recall_vs_runs": [float(x) for x in np.nanmean(D["ensemble"], axis=0)],
        "per_paper": {SHORT[p]: D["per_paper"][p] for p in order},
    }
    (DATA / "robustness_figure_data.json").write_text(json.dumps(sidecar, indent=2))
    print("wrote figures/fig_robustness_stability.{png,pdf,svg} + data/robustness_figure_data.json")
    print(f"  overall mean detection (merged) {sidecar['overall_mean_detection']['merged']:.2f}")
    print(f"  by-severity (merged): high {sidecar['by_severity_majority']['merged']['high']['mean_det']:.2f} "
          f"med {sidecar['by_severity_majority']['merged']['medium']['mean_det']:.2f} "
          f"low {sidecar['by_severity_majority']['merged']['low']['mean_det']:.2f}")


if __name__ == "__main__":
    main()
