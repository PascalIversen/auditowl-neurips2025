#!/usr/bin/env python3
"""Per-paper artefact-coverage figure.

Parses the coverage table in every `audits/*/audit.md` — the
`| Paper artefact | Repo location | Computed value | Matches paper | Status |`
table — classifies each row's Status into one of five outcomes, and renders one
100%-stacked bar per paper showing the share of traced result-artefacts that are
present / partial / mismatched / MISSING (no code produces them) / unverified.

IMPORTANT framing: a row is an *artefact-claim the auditor chose to trace* (a
table, a column, a metric, an ablation, sometimes a figure) — NOT necessarily a
whole figure, and NOT the paper's full figure/table count. So this shows
"share of traced artefacts with no producing code", not "fraction of the paper's
figures that reproduce". The denominator is auditor-selected (skewed toward the
load-bearing / suspect artefacts), which is stated on the figure.

Outputs:
  analysis/data/coverage_status.json   (auditable per-paper counts)
  figures/out/fig03_code_coverage.{png,pdf,svg}

Run:
  python figures/fig03_code_coverage.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures" / "out"
DATA = ROOT / "analysis" / "data"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- look & feel: Nature-figure publication style ---------------------------
_avail = {f.name for f in fm.fontManager.ttflist}
_FONT = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
              if f in _avail), "sans-serif")
INK, MUTE = "#333333", "#767676"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_FONT, "DejaVu Sans", "Liberation Sans"],
    "font.size": 16,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 19, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlepad": 10,
    "axes.edgecolor": "#4D4D4D", "axes.linewidth": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 15,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Status colours: a present→missing severity ramp drawn from the OI palette.
COL = {
    "missing":  "#B64342",   # vermillion  — no producing code located (the focus)
    "bug":      "#3E6D8E",   # steel blue  — code present but won't run / inputs absent
    "mismatch": "#8E6FB0",   # purple      — code present but disagrees with paper
    "partial":  "#E1A33C",   # amber       — partially traceable
    "present":  "#8BCF8B",   # green       — verified / producing code present
    "na":       "#C7CED6",   # grey        — not code-checkable (human study / off-repo)
}
LABELS = {
    "missing": "Missing\n(no producing code)",
    "bug": "Bug\n(present but broken)",
    "mismatch": "Mismatch\n(code ≠ paper)",
    "partial": "Partial",
    "present": "Present / verified",
    "na": "Not code-checkable",
}
# Stack order, bottom → top. Missing on the baseline so its band height reads
# directly as "% missing" and a median reference line is meaningful.
ORDER = ["missing", "bug", "mismatch", "partial", "present", "na"]


def txt_on(hex_color: str) -> str:
    c = hex_color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "#333333"


def classify(art: str, raw: str) -> str | None:
    """Map a (artefact, Status) row to one of six outcome buckets.

    The Status cells are free text written by the auditor, so this is heuristic.
    Ordering is load-bearing:
      (1) strip leading markdown emphasis, or a bolded "**MISSING**" falls through;
      (2) the auditor's LEADING verdict wins — a cell that starts with "missing"
          stays missing even if it notes "(mechanism present)"; a cell that leads
          with "present" stays present even if it later flags a gap (no demotion
          is applied for a trailing caveat -- this leans present/na over missing,
          which is conservative relative to the paper's missing-code headline);
      (3) "not code-checkable" cues (human study / off-repo / live service / theory)
          are tested FIRST, so a "MISSING (human study)" cell is not counted as
          missing *code* and an artefact named "user study" is caught via `art`.
    `art` is the Paper-artefact column; `raw` is the Status column.
    """
    s = raw.strip()
    low = s.lower()
    # Drop wrapped-row fragments (")", ",\\", "(") and pure-punctuation cells.
    if not low or set(low) <= set("-—:.,()[]\\/ "):
        return None
    core = low.lstrip("*_`'\"~ (").strip()
    blob = (art or "").lower() + " ¦ " + low

    # 0. Not code-checkable: human studies, off-repo / live external services, theory.
    NA_CUES = ("human study", "human-study", "human eval", "human-eval", "user study",
               "user-study", "subjective", "opinion score", "listening test", "annotator",
               " rater", "human preference", "human rating", "human judg",
               "off-repo", "off repository", "external api", "live api", "live service",
               "third-party service", "closed-source api")
    if any(c in blob for c in NA_CUES):
        return "na"
    if core.startswith(("n/a", "na ", "n.a", "not applicable", "external",
                        "analytic", "theory", "out of scope", "out-of-scope")):
        return "na"

    # 1. Mismatch: code traces but disagrees with the paper.
    if "mismatch" in low or core.startswith(("mismatch", "difference", "differs")):
        return "mismatch"
    if ("✗" in s or "✘" in s) and not core.startswith(("present", "verified")):
        return "mismatch"

    leads_missing = core.startswith(("missing", "no code", "absent", "not present",
                                     "not provided"))

    # 2. Bug: producing code is PRESENT but DEFECTIVE — it crashes, errors, or will
    #    not run. This is NOT "data/inputs/weights missing": code that is present
    #    but merely lacks data is still code-traceable (-> present); the absent data
    #    is a separate missing-artifact finding, not a code-coverage failure here.
    #    Guarded by `not leads_missing` so the auditor's explicit MISSING verdict wins.
    RUN_FAIL = ("does not run", "doesn't run", "won't run", "will not run", "cannot run",
                "not runnable", "fails to run", "crash", "errors out", "import fails",
                "importerror", "traceback", "broken")
    if not leads_missing and (core.startswith("bug")
                              or any(k in low for k in RUN_FAIL)):
        return "bug"

    # 3. Missing: no producing code located.
    if leads_missing or "no producing code" in low:
        return "missing"

    # 4. Partial.
    if core.startswith("partial"):
        return "partial"

    # 5. Remaining un-checkable negatives ("unverified", "not checkable", ...).
    if core.startswith(("not ", "no ", "cannot", "un", "out of")):
        return "na"

    # 6. Present / verified.
    PRESENT = ("present", "verified", "confirm", "✓", "reproduc", "derivable",
               "re-deriv", "re deriv", "traced", "comput", "supported", "match",
               "correct", "computation present", "pipeline present", "ok ", "yes")
    if core.startswith(PRESENT) or any(k in low for k in ("present", "✓", "reproduc")):
        return "present"
    if core.startswith(("trace", "traces", "plotting", "plot ", "weakly")):
        return "partial"
    return "na"


def parse_audit(md_path: str):
    lines = open(md_path, errors="replace").read().splitlines()
    hdr = next((i for i, l in enumerate(lines)
                if "paper artefact" in l.lower() and "status" in l.lower()
                and l.count("|") >= 4), None)
    if hdr is None:
        return None
    cols = [c.strip().lower() for c in lines[hdr].split("|")]
    status_idx = max(i for i, c in enumerate(cols) if c)   # last non-empty column
    art_idx = next((i for i, c in enumerate(cols)
                    if "artefact" in c or "artifact" in c), 1)
    counts = {k: 0 for k in COL}
    for l in lines[hdr + 2:]:
        if not l.strip().startswith("|"):
            break
        cells = l.split("|")
        if len(cells) <= status_idx:
            continue
        art = cells[art_idx] if len(cells) > art_idx else ""
        cat = classify(art, cells[status_idx])
        if cat:
            counts[cat] += 1
    return counts


def collect():
    papers = []
    for md in sorted(glob.glob(str(ROOT / "audits" / "*" / "audit.md")),
                     key=lambda m: os.path.basename(os.path.dirname(m)) + "_"):  # original folder order
        if os.sep + "theory" + os.sep in md:
            continue
        num = os.path.basename(os.path.dirname(md)).split("_")[0]
        counts = parse_audit(md)
        n = sum(counts.values()) if counts else 0
        if n == 0:
            # No coverage table -> a "No code present" stub. Show it as a single
            # fully-missing (all-red) bar, but record 0 artefacts *traced* (there
            # was no code to trace, so its top-strip count is 0).
            txt = open(md, errors="replace").read().lower()
            if any(m in txt for m in ("no code present", "no author code",
                                      "no code released", "no public code",
                                      "justified_no_code")) or "no code" in txt[:600]:
                papers.append({"num": num, "n": 1, "traced": 0, "is_nocode": True,
                               "counts": {k: (1 if k == "missing" else 0) for k in COL},
                               "missing_frac": 1.0})
            continue
        papers.append({"num": num, "n": n, "traced": n, "is_nocode": False,
                       "counts": counts, "missing_frac": counts["missing"] / n})
    return papers


def render(papers):
    # Sort best → worst by share missing (so the red baseline band rises L→R).
    papers = sorted(papers, key=lambda p: (p["missing_frac"], -p["n"]))
    NP = len(papers)
    x = np.arange(NP)
    fracs = {k: np.array([p["counts"][k] / p["n"] for p in papers]) * 100 for k in COL}
    totals = np.array([p.get("traced", p["n"]) for p in papers])

    fig = plt.figure(figsize=(15.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6.2], hspace=0.10)
    ax_n = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_n)

    # --- top strip: number of artefacts traced per paper (the denominator) ---
    ax_n.bar(x, totals, width=0.92, color="#b9c0c9", edgecolor="none")
    top = max(20, totals.max())
    ax_n.set_ylim(0, top * 1.15)
    ax_n.set_yticks([0, 20])
    ax_n.set_ylabel("Artifacts\ntraced", fontsize=15, color=MUTE, rotation=0,
                    ha="right", va="center", labelpad=18)
    ax_n.tick_params(axis="y", labelsize=13, colors=MUTE, length=3)
    ax_n.tick_params(axis="x", length=0)
    plt.setp(ax_n.get_xticklabels(), visible=False)
    for sp in ("left",):
        ax_n.spines[sp].set_color("#b9c0c9")

    # --- main: 100%-stacked status bars ---
    bottom = np.zeros(NP)
    for k in ORDER:
        ax.bar(x, fracs[k], width=0.92, bottom=bottom, color=COL[k],
               edgecolor="white", linewidth=0.15, label=LABELS[k])
        bottom += fracs[k]

    med = statistics.median([p["missing_frac"] for p in papers
                             if not p.get("is_nocode")]) * 100
    n_full = sum(1 for p in papers if p["missing_frac"] == 1.0)
    n_zero = sum(1 for p in papers if p["missing_frac"] == 0.0)

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("Share of traced artifacts (%)")
    ax.set_xlim(-0.7, NP - 0.3)
    ax.set_xticks([])
    ax.set_xlabel("Audited papers", fontsize=14, labelpad=8)
    for sp in ("bottom",):
        ax.spines[sp].set_visible(True)

    # legend to the right, single column (6 classes)
    handles, labs = ax.get_legend_handles_labels()
    order_leg = [labs.index(LABELS[k]) for k in
                 ["present", "partial", "mismatch", "bug", "missing", "na"]]
    ax.legend([handles[i] for i in order_leg], [labs[i] for i in order_leg],
              loc="center left", bbox_to_anchor=(1.015, 0.5), ncol=1,
              handlelength=1.2, handletextpad=0.7, labelspacing=1.0,
              borderaxespad=0.0)

    for name in ("fig03_code_coverage.png",):
        fig.savefig(FIGS / name)
        fig.savefig(FIGS / name.replace(".png", ".pdf"))
        fig.savefig(FIGS / name.replace(".png", ".svg"))
    plt.close(fig)
    return med, n_full, n_zero


def main():
    papers = collect()
    # status totals over REAL traced artefacts only (exclude the synthetic
    # all-red no-code bars, which carry no traced artefacts).
    real = [p for p in papers if not p.get("is_nocode")]
    n_nocode = len(papers) - len(real)
    totals = {k: sum(p["counts"][k] for p in real) for k in COL}
    out = {
        "n_papers": len(papers),
        "n_nocode_papers": n_nocode,
        "n_artefacts": sum(p["traced"] for p in real),
        "totals_by_status": totals,
        "median_missing_frac": statistics.median([p["missing_frac"] for p in real]),
        "papers": papers,
    }
    (DATA / "coverage_status.json").write_text(json.dumps(out, indent=2))
    med, n_full, n_zero = render(papers)
    print(f"  {len(papers)} bars ({len(real)} with a coverage table + "
          f"{n_nocode} no-code, shown all-red), {out['n_artefacts']} traced artefact rows")
    print(f"  status totals (traced only): {totals}")
    print(f"  median missing (traced papers): {med:.0f}%   |  "
          f"{n_full} all-missing, {n_zero} none-missing")
    print("  wrote figures/fig03_code_coverage.{png,pdf,svg} + data/coverage_status.json")


if __name__ == "__main__":
    main()
