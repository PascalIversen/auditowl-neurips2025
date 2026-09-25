#!/usr/bin/env python3
"""Figure 2 — "The receipts": headline-findings figure for the NeurIPS-2025 audit.

A single landscape figure that leads with the verifiable headline numbers and the
*receipts* behind the most severe findings. Most receipts cite a re-runnable
check in the corresponding audit; a few (currently 2 of 8) instead cite a
`file:line`/paper quote directly, with no separate check script.

Data layer:
  * Aggregate counts (categories, severity, funnel, papers-affected) come from
    `analysis/data/figure_data.json` (produced by `aggregate.py`).
  * The per-paper findings-count histogram is recomputed here from
    `audits/*/findings_verified.json` using aggregate.py's `is_dropped` filter
    (drop verifier-rejected + supplement false positives) so it reconciles
    exactly with the category bars.
  * The receipt cards are a curated subset of severe findings (parsed live from
    severe_findings_display.md); each cites a real paper id and the re-runnable
    check that established it (see findings_verified.json).

Run:  python analysis/aggregate.py && python figures/fig02_finding_categories.py
Outputs: figures/out/fig02_finding_categories.{png,pdf,svg}
"""
from __future__ import annotations
import json
import glob
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures" / "out"
DATA = ROOT / "analysis" / "data"
FIGS.mkdir(parents=True, exist_ok=True)
D = json.loads((DATA / "figure_data.json").read_text())
FP_PATH = DATA / "repo_fingerprints.json"
REPO_FINGERPRINTS = json.loads(FP_PATH.read_text()) if FP_PATH.exists() else {}

# ---- look & feel (Nature-figure publication style) ------------------------
_avail = {f.name for f in fm.fontManager.ttflist}
_FONT = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
              if f in _avail), "sans-serif")
_MONO = next((f for f in ("Menlo", "DejaVu Sans Mono", "Consolas", "Courier New")
              if f in _avail), "monospace")
INK, MUTE = "#333333", "#767676"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_FONT, "DejaVu Sans", "Liberation Sans"],
    "font.size": 12, "text.color": INK,
    "axes.edgecolor": "#4D4D4D", "axes.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})
OI = {"blue": "#0F4D92", "sky": "#3775BA", "green": "#8BCF8B",
      "vermillion": "#B64342", "orange": "#E1A33C", "purple": "#8E6FB0"}
CAT_COLOR = {"missing": OI["blue"], "difference": OI["green"],
             "bug": OI["orange"], "methodology": OI["purple"]}
GREEN_D = "#2E8B57"


def shade(hex_color, t=0.30):
    """Darken toward black by fraction t (for readable text on light tints)."""
    c = hex_color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(v * (1 - t)) for v in rgb)


# =====================================================================
# Data
# =====================================================================
NP = D["n_papers"]
N_RAISED = D["n_findings"]
BY_CAT = D["by_category"]                       # missing/difference/bug/methodology
PAPERS_CAT = D["papers_by_category"]
CATLAB = D["category_labels"]
N_CAT = sum(BY_CAT.values())                    # 657 categorised


def _is_dropped(f):
    # Kept in sync with analysis/aggregate.py's is_dropped (not imported: that
    # module runs its full pipeline, including rewriting figure_data.json, as a
    # side effect of import, which this script should not silently trigger).
    return (f.get("verdict") == "reject"
            or f.get("supplement_verdict") == "false_positive"
            or f.get("provenance_verdict") == "false_positive")


# --- histogram severity switch -------------------------------------------------
# Which severities the panel-(a) prevalence histograms count. Set to None for
# ALL findings, or a subset of {"high","medium","low"} to restrict. The panel-(b)
# receipt cards are a fixed curated set and are NOT affected by this switch.
HIST_SEVERITIES = {"high", "medium"}


def _sev_ok(f):
    return HIST_SEVERITIES is None or f.get("severity") in HIST_SEVERITIES


ORDER = ["missing", "difference", "bug", "methodology"]

# --- real-code gate (mirrors aggregate.py): a paper "released code" iff a code
# URL is linked AND a cloned repo ships >=1 real code file. This drops the 13
# no-source papers -> 87 (no stub currently in the release: every code-releasing
# paper ships >=1 real code file).
CODE_EXT = {
    ".py", ".ipynb", ".m", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cu",
    ".cuh", ".java", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".sh", ".bash",
    ".r", ".jl", ".scala", ".lua", ".f90", ".f", ".mlx", ".pyx",
}
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", ".github"}


def _repo_code_files(repo):
    return sum(1 for p in repo.rglob("*")
               if not any(s in p.parts for s in _SKIP_DIRS)
               and p.is_file() and p.suffix.lower() in CODE_EXT)


def has_real_code(audit_dir):
    cl = audit_dir / "code_links.txt"
    url_ok = cl.exists() and bool(cl.read_text().strip())
    repos = list((audit_dir / "code").glob("*/")) if (audit_dir / "code").exists() else []
    fp = REPO_FINGERPRINTS.get(audit_dir.name, {})
    has_code = sum(_repo_code_files(r) for r in repos) > 0 or fp.get("has_code", False)
    return url_ok and has_code


def load_paper_rows():
    """One row per non-theory, code-releasing paper: dict(real_code,
    cc=Counter(category->n)), surviving findings only.

    Enumerated from audits/*/findings_verified.json, which only the 87
    code-releasing papers have. The 13 no-code papers are not counted here:
    doing so would require crediting each with a synthetic, unrecorded 'missing'
    finding rather than a real one, which panel (a)'s per-paper histogram has no
    way to distinguish from a genuine finding without a caption change. All four
    categories therefore share the same 87-paper denominator.
    """
    out = []
    for fp in sorted(glob.glob(str(ROOT / "audits" / "*" / "findings_verified.json"))):
        d = Path(fp).parent
        if d.parent.name == "theory":
            continue
        cc = Counter()
        for f in json.loads(Path(fp).read_text()).get("findings", []):
            if not _is_dropped(f) and _sev_ok(f):
                cc[f.get("category")] += 1
        out.append({"real_code": has_real_code(d), "cc": cc})
    return out


ROWS = load_paper_rows()
CAT_POP = {c: ROWS for c in ORDER}
PERCAT = {c: [r["cc"].get(c, 0) for r in CAT_POP[c]] for c in ORDER}
CAT_DENOM = {c: len(CAT_POP[c]) for c in ORDER}       # 87 for all four categories
CAT_HIST = {c: Counter(PERCAT[c]) for c in ORDER}
CAT_AFFECTED = {c: sum(1 for v in PERCAT[c] if v >= 1) for c in ORDER}
CAT_XMAX = {c: max(PERCAT[c]) for c in ORDER}
GLOBAL_XMAX = max(CAT_XMAX.values())          # common x-axis (and bar width) across panels
HIST_YMAX = max(max(h.values()) for h in CAT_HIST.values())   # shared y across panels

# The receipt cards are the curated severe findings, parsed live from
# severe_findings_display.md so the figure stays in sync with that file.
# Each finding -> a card: paper id + name, the one-line summary (the headline),
# and the evidence location (file:line — the re-runnable receipt).
SEVERE_MD = ROOT / "analysis" / "severe_findings_display.md"
SEC2CAT = {"MISSING": "missing", "MISMATCH": "difference",
           "BUG": "bug", "METHODOLOG": "methodology"}


def load_receipts(path):
    """Parse the display markdown into receipt-card dicts (pid, paper, cat,
    tag=short issue label, title=summary, ev=evidence-location)."""
    cat, cur, out = None, None, []
    for raw in path.read_text().split("\n"):
        line = raw.strip()
        if line.startswith("## "):                       # section -> category
            name = line[3:].upper()
            cat = next((c for k, c in SEC2CAT.items() if k in name), None)
        elif line.startswith("### "):                    # new finding header
            if cur:
                out.append(cur)
            pid, _, paper = line[4:].partition("·")
            cur = dict(pid=pid.strip().lstrip("#").strip(), paper=paper.strip(),
                       cat=cat, tag="", title="", claim="", ev="", cons="", conf="")
        elif cur is None:
            continue
        elif line.startswith("_") and "topic:" in line:  # short issue tag (chip)
            cur["tag"] = line.split("topic:")[1].strip().rstrip("_").strip()
            if "confidence:" in line:                     # high / medium / low
                cur["conf"] = line.split("confidence:")[1].split("·")[0].strip()
        elif line.startswith("**Claim:**"):                # verbatim claim paragraph
            cur["claim"] = re.sub(r"\s+", " ", line[len("**Claim:**"):]).strip()
        elif line.startswith("**Evidence:**"):
            cur["ev"] = line[len("**Evidence:**"):].split("·")[0].strip()
        elif (line.startswith("**") and line.endswith("**")
              and not line[2:].startswith(("Claim:", "Concern:", "Ask:", "Evidence:"))
              and not cur["title"]):
            cur["title"] = line[2:-2].strip()            # the one-line summary
    if cur:
        out.append(cur)
    return [r for r in out if r["cat"] in CAT_COLOR]


# --- de-identification: star the identifiable repo/model dir in evidence paths,
# and replace the model's proper name in the audit prose with [modelname]. Only
# the identifying token changes; every other word (generic dirs, filenames, code
# identifiers, line numbers) stays VERBATIM. Generic roots (src/, dataset/, the
# public open_flamingo framework, the audit's code/ clone wrapper) are not
# identifying and are left as-is.
ANON = {
    "570":  {"path": [("GC-xLSTM/", "*/")],
             "name": [("xLSTM", "[modelname]")],     # the GC-xLSTM model arch
             "sub":  [("xlstm_neural_gc.py", "*_neural_gc.py")]},
    "3457": {"path": [("code/Renovamen__ShiftDC/", "*/")],
             "name": [("ShiftDC", "[modelname]")],
             "sub":  [("run_shiftdc.py", "run_*.py")]},
    "4946": {"path": [("open_flamingo/open_flamingo/", "*/")]},  # repo+pkg dir -> *
    "4991": {"sub":  [("Wavelet Kernel-based Knowledge Retention", "[method]")]},
    "5202": {"path": [("ploi/", "*/")]},
}


def anonymize(r):
    """Apply this receipt's de-identification rules in place and return it."""
    rules = ANON.get(r["pid"])
    if not rules:
        return r
    for frm, to in rules.get("path", []):          # leading repo/model dir -> *
        r["ev"] = r["ev"].replace(frm, to)
    for frm, to in rules.get("name", []):          # model proper name -> [modelname]
        pat = r"\b" + re.escape(frm) + r"\b"
        for k in ("title", "claim", "tag"):
            r[k] = re.sub(pat, to, r[k])
    for frm, to in rules.get("sub", []):           # model-named file / coined name (all fields)
        for k in ("title", "claim", "tag", "ev"):
            r[k] = r[k].replace(frm, to)
    return r


RECEIPTS = [anonymize(r) for r in
            sorted(load_receipts(SEVERE_MD), key=lambda r: ORDER.index(r["cat"]))]

# =====================================================================
# Figure — the per-category prevalence histograms sit in a STRIP across the
#   top (one per category, shared axes for comparability); below them a
#   gallery of receipt cards, one category per row (two cards per row, or one
#   full-width card when a category has a single example).  Each card carries
#   the paper id + issue tag, the one-line summary, the VERBATIM claim
#   paragraph, and the evidence location.
#
#   ALL card text is rendered at ONE uniform point size (FS) — weight, family
#   and colour carry the hierarchy, not size.  Nothing is shrunk to fit:
#   instead each row is made exactly as tall as its tallest card needs, and the
#   whole figure height grows to suit, so the full claim always fits at FS.
# =====================================================================
FIG_W = 15.0
BY_CAT_RECEIPTS = {c: [r for r in RECEIPTS if r["cat"] == c] for c in ORDER}

# shared y-axis ticks for the histogram strip (only as many 20-steps as fit)
YTICKS = [0]
_t = 20
while _t <= HIST_YMAX:
    YTICKS.append(_t)
    _t += 20

# ---- uniform card typography & inch-based vertical metrics -----------------
FS = 10.5                                   # single font size for ALL card text
LEAD = 1.30
LINE_IN = FS / 72.0 * LEAD                   # height of one text line (inches)
GAP_HEAD, GAP_TITLE, GAP_CLAIM = 0.11, 0.11, 0.13     # inter-block gaps (inches)
TOP_INSET, BOT_INSET = 0.08, 0.09
PAD_L = 0.030                                # left text inset (card-width fraction)


def _wrap(r, w_in):
    """Width-aware line breaking (fills each line to the column) so the
    justified text below has only small, even slack to distribute."""
    title = _fit_wrap(r["title"], w_in, FS, "bold")
    claim = _fit_wrap(r["claim"], w_in, FS) if r.get("claim") else []
    ev = _fit_wrap(r["ev"], w_in, FS, family=_MONO) if r.get("ev") else []
    return title, claim, ev


def card_height_in(r, w_in):
    """Inches a card needs at FS, given its width (drives row height)."""
    title, claim, ev = _wrap(r, w_in)
    return (TOP_INSET + LINE_IN                          # header line
            + GAP_HEAD + len(title) * LINE_IN
            + GAP_TITLE + len(claim) * LINE_IN
            + GAP_CLAIM + len(ev) * LINE_IN + BOT_INSET)


# ---- width-aware wrapping + justified ("Blocksatz") body text --------------
# matplotlib has no justification, and textwrap breaks by character count —
# which under-fills proportional lines and makes justification stretch into
# ugly gaps.  So we (1) break lines by MEASURED width to fill the column, then
# (2) justify by spreading the small leftover slack across the gaps.  A
# dedicated 1x1 figure supplies font metrics before the real (auto-sized)
# figure exists.  The last line of each paragraph stays ragged.
_MFIG = plt.figure(figsize=(1, 1))
_WCACHE: dict = {}


def _w_in(s, fontsize, weight, family):
    """Rendered width of string `s` at the given style, in inches (memoised)."""
    key = (s, fontsize, weight, family)
    if key not in _WCACHE:
        kw = {"fontsize": fontsize, "fontweight": weight}
        if family:
            kw["family"] = family
        t = _MFIG.text(0, 0, s, **kw)
        bb = t.get_window_extent(renderer=_MFIG.canvas.get_renderer())
        t.remove()
        _WCACHE[key] = bb.width / _MFIG.dpi
    return _WCACHE[key]


def _fit_wrap(text, w_in, fontsize, weight="normal", family=None):
    """Greedily pack words into lines that fill the column width (measured)."""
    tw = w_in * (1 - PAD_L - 0.012)
    out, cur = [], ""
    for word in (text or "").split():
        if _w_in(word, fontsize, weight, family) > tw:    # token wider than column
            if cur:
                out.append(cur)
                cur = ""
            s = ""                                        # hard-split the long token
            for ch in word:
                if s and _w_in(s + ch, fontsize, weight, family) > tw:
                    out.append(s)
                    s = ch
                else:
                    s += ch
            cur = s
            continue
        cand = word if not cur else cur + " " + word
        if cur and _w_in(cand, fontsize, weight, family) > tw:
            out.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def _draw_par(ax, lines, cur, line_step, w_in, fontsize, color, justify,
              weight="normal", family=None):
    """Draw a paragraph at left inset PAD_L; justify all but the last line so
    every block shares the same left edge (PAD_L) and right edge (tw)."""
    tw = w_in * (1 - PAD_L - 0.012)
    tkw = {"fontsize": fontsize, "color": color, "va": "top", "fontweight": weight}
    if family:
        tkw["family"] = family
    for i, ln in enumerate(lines):
        words = ln.split()
        last = i == len(lines) - 1
        if not justify or last or len(words) < 2:
            ax.text(PAD_L, cur, ln, **tkw)
        else:
            ww = [_w_in(w, fontsize, weight, family) for w in words]
            gap = (tw - sum(ww)) / (len(words) - 1)
            space = fontsize / 72.0 * 0.28                # ~ natural word space (inches)
            # line already full — don't compress; and don't stretch a line of a
            # few long tokens into wide rivers (e.g. two identifiers) — leave it ragged
            if gap < 0 or gap > 3 * space:
                ax.text(PAD_L, cur, ln, **tkw)
            else:
                xin = 0.0
                for w, wd in zip(words, ww):
                    ax.text(PAD_L + xin / w_in, cur, w, **tkw)
                    xin += wd + gap
        cur -= line_step
    return cur


def draw_card(rect, r):
    """Render a receipt card inside `rect` (figure coords).  Vertical spacing is
    in inches converted to the card's own height, so every card reads at the
    same FS regardless of how tall the card is."""
    x, y0, w, h = rect
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    col = CAT_COLOR[r["cat"]]
    w_in, h_in = w * FIG_W, h * FIG_H
    title, claim, ev = _wrap(r, w_in)

    def f(inches):
        return inches / h_in                  # inches -> card-height fraction

    # accent bar (category colour) — the only chrome; no background fill
    ax.add_patch(Rectangle((0.006, 0.02), 0.0058, 0.96, facecolor=col, edgecolor="none"))

    cur = 1 - f(TOP_INSET)
    # header: issue tag · confidence (paper id omitted so cards do not link to a release folder)
    hx = PAD_L
    if r.get("tag"):
        ax.text(hx, cur, r["tag"].upper(), fontsize=FS, fontweight="bold",
                color=shade(col, 0.34), va="top")
        hx += (_w_in(r["tag"].upper(), FS, "bold", None) + 0.16) / w_in
    if r.get("conf"):
        ax.text(hx, cur, f"· confidence: {r['conf']}", fontsize=FS,
                color=MUTE, va="top")
    cur -= f(LINE_IN) + f(GAP_HEAD)
    # one-line summary (bold) — justified
    cur = _draw_par(ax, title, cur, f(LINE_IN), w_in, FS, INK, justify=True,
                    weight="bold")
    cur -= f(GAP_TITLE)
    # verbatim claim paragraph — justified (Blocksatz)
    cur = _draw_par(ax, claim, cur, f(LINE_IN), w_in, FS, "#474747", justify=True)
    cur -= f(GAP_CLAIM)
    # evidence location (mono — the re-runnable receipt) — left-aligned, NOT
    # justified: stretching the few wide path tokens stranded the " ; " gaps.
    cur = _draw_par(ax, ev, cur, f(LINE_IN), w_in, FS, "#5A5A5A", justify=False,
                    family=_MONO)


# ---- column geometry (fractions of FIG_W) ----------------------------------
# CL is flush with the panel labels (PANEL_X) so the cards fill the left margin
# instead of leaving an empty strip under the "(b)" heading.
CL, CR, COL_GAP = 0.014, 0.988, 0.020
HALF_W = (CR - CL - COL_GAP) / 2
FULL_W = CR - CL

# ---- per-row card lists & heights (a lone card spans the full width) -------
ROW_CARDS = [BY_CAT_RECEIPTS[c] for c in ORDER]
ROW_H_IN = []
for cards in ROW_CARDS:
    if len(cards) <= 1:
        hs = [card_height_in(r, FULL_W * FIG_W) for r in cards]
    else:
        hs = [card_height_in(r, HALF_W * FIG_W) for r in cards[:2]]
    ROW_H_IN.append(max(hs) if hs else 0.6)

# ---- top-region band heights (inches) — fixed; the card area grows ---------
# STRIP_XLAB reserves room BELOW the axes for the tick numbers + the per-axis
# x-axis label, so it doesn't collide with the (b) heading.  STRIP_NOTE is just
# the gap between the x-label band and the (b) heading.
M_TOP, HEAD1, STRIP_LABELS, STRIP_AX = 0.42, 0.24, 0.58, 1.58
STRIP_XLAB, STRIP_NOTE, HEAD2 = 0.72, 0.16, 0.22
ROW_GAP, M_BOT = 0.20, 0.30
TOP_BLOCK = (M_TOP + HEAD1 + STRIP_LABELS + STRIP_AX + STRIP_XLAB
             + STRIP_NOTE + HEAD2)
FIG_H = (TOP_BLOCK + sum(ROW_H_IN) + ROW_GAP * (len(ROW_CARDS) - 1) + M_BOT)

fig = plt.figure(figsize=(FIG_W, FIG_H))


def topf(d):
    """Distance-from-top (inches) -> figure y-fraction (va='top' anchor)."""
    return 1 - d / FIG_H


# ---- TOP STRIP (panel a): one prevalence histogram per category ------------
PANEL_X = 0.012
fig.text(PANEL_X, topf(M_TOP), "(a)", fontsize=15, fontweight="bold",
         color=INK, ha="left", va="top")


strip_lab_top = M_TOP + HEAD1
hist_top = strip_lab_top + STRIP_LABELS
L, R, hgap = 0.060, 0.985, 0.052
hw = (R - L - (len(ORDER) - 1) * hgap) / len(ORDER)
for hi, cat in enumerate(ORDER):
    col = CAT_COLOR[cat]
    hx = L + hi * (hw + hgap)
    fig.text(hx, topf(strip_lab_top), CATLAB[cat], fontsize=14.5, fontweight="bold",
             color=shade(col, 0.10), va="top")
    fig.text(hx, topf(strip_lab_top + 0.28),
             f"affects {CAT_AFFECTED[cat]} of {CAT_DENOM[cat]} papers",
             fontsize=12, color=MUTE, va="top")
    ax = fig.add_axes([hx, topf(hist_top + STRIP_AX), hw, STRIP_AX / FIG_H])
    xmax = GLOBAL_XMAX            # common x-axis -> identical bar width across panels
    xs = list(range(0, xmax + 1))
    ys = [CAT_HIST[cat].get(x, 0) for x in xs]
    bars = ax.bar(xs, ys, width=0.84, color=col, edgecolor="white", lw=0.5)
    bars[0].set_color("#BFBFBF")                  # x=0 ("none") bar, grey
    ax.set_xlim(-0.6, xmax + 0.6)
    ax.set_ylim(0, HIST_YMAX * 1.10)
    ax.set_xticks(list(range(0, xmax + 1)) if xmax < 8 else list(range(0, xmax + 1, 2)))
    ax.set_yticks(YTICKS)
    ax.tick_params(labelsize=12.5)
    # per-axis x label, padded clear of the tick numbers
    ax.set_xlabel("Findings per paper", fontsize=13, color=INK, labelpad=4)
    if hi == 0:
        ax.set_ylabel("Papers", fontsize=13.5)

# ---- CARD GALLERY (panel b): one category per row --------------------------
head2_top = hist_top + STRIP_AX + STRIP_XLAB + STRIP_NOTE
# the "(b)" letter sits a touch above the card row (cards stay at head2_top+HEAD2)
fig.text(PANEL_X, topf(head2_top - 0.12), "(b)", fontsize=15, fontweight="bold",
         color=INK, ha="left", va="top")

y = head2_top + HEAD2
for ri, cards in enumerate(ROW_CARDS):
    rh = ROW_H_IN[ri]
    bot, hfrac = topf(y + rh), rh / FIG_H
    if len(cards) == 1:
        draw_card([CL, bot, FULL_W, hfrac], cards[0])
    else:
        for ci, r in enumerate(cards[:2]):
            cx = CL if ci == 0 else CL + HALF_W + COL_GAP
            draw_card([cx, bot, HALF_W, hfrac], r)
    y += rh + ROW_GAP

for ext in ("png", "pdf", "svg"):
    fig.savefig(FIGS / f"fig02_finding_categories.{ext}")
plt.close(fig)
print(f"wrote figures/fig02_finding_categories.png (+pdf, +svg)  [{FIG_W} x {FIG_H:.1f} in]")
print(f"  {N_RAISED} findings · cats {BY_CAT} · affected {CAT_AFFECTED}")
print(f"  {len(RECEIPTS)} receipt cards from {SEVERE_MD.name}: "
      f"{ {c: sum(1 for r in RECEIPTS if r['cat'] == c) for c in ORDER} }")
print(f"  row heights (in): {[round(h, 2) for h in ROW_H_IN]}  ·  FS={FS}pt uniform")
