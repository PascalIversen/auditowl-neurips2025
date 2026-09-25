#!/usr/bin/env python3
"""Severity x confidence heatmap (paper Figure S1).

Counts every finding in audits/<paper>/findings.json (excluding audits/theory/)
by the auditor's severity x confidence labels, over the source-present papers
(after the repo-provenance correction), and renders an absolute-scale heatmap.
Colour encodes the raw finding count (no normalization), so the shading and the
printed number always agree. Both axes are the model's own labels, not ground
truth.

Outputs: figures/out/figS1_severity_confidence.{png,pdf,svg}
Run:     python figures/figS1_severity_confidence.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures" / "out"
FIGS.mkdir(parents=True, exist_ok=True)
FP_PATH = ROOT / "analysis" / "data" / "repo_fingerprints.json"
REPO_FINGERPRINTS = json.loads(FP_PATH.read_text()) if FP_PATH.exists() else {}

ORDER = ["high", "medium", "low"]

# code file extensions / skip dirs -- mirrors aggregate.py's repo_code_file_count
CODE_EXT = {
    ".py", ".ipynb", ".m", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cu",
    ".cuh", ".java", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".sh", ".bash",
    ".r", ".jl", ".scala", ".lua", ".f90", ".f", ".mlx", ".pyx",
}
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", ".github"}


def repo_code_file_count(repo: Path) -> int:
    return sum(
        1 for p in repo.rglob("*")
        if not any(s in p.parts for s in _SKIP_DIRS)
        and p.is_file() and p.suffix.lower() in CODE_EXT
    )


def has_real_code(audit_dir: Path) -> bool:
    """aggregate.py's 'Source code present' gate minus the lone filesystem stub,
    honouring the repo-provenance correction: a paper whose only cloned repo is a
    baseline/dependency (repo_provenance.json core_present == "no") released no
    author core and is excluded -- keeping this population consistent with
    aggregate.py."""
    cl = audit_dir / "code_links.txt"
    url_ok = cl.exists() and bool(cl.read_text().strip())
    repos = list((audit_dir / "code").glob("*/")) if (audit_dir / "code").exists() else []
    prov = audit_dir / "repo_provenance.json"
    if prov.exists():
        try:
            if json.loads(prov.read_text()).get("core_present") == "no":
                return False
        except Exception:
            pass
    fp = REPO_FINGERPRINTS.get(audit_dir.name, {})
    has_code = sum(repo_code_file_count(r) for r in repos) > 0 or fp.get("has_code", False)
    return url_ok and has_code


def load_counts() -> tuple[Counter, int, int]:
    cell: Counter = Counter()
    total = 0
    n_papers = 0
    for fp in sorted(glob.glob(str(ROOT / "audits" / "*" / "findings.json"))):
        adir = Path(fp).parent
        if adir.parent.name == "theory" or not has_real_code(adir):
            continue
        n_papers += 1
        data = json.loads(Path(fp).read_text())
        findings = data if isinstance(data, list) else data.get("findings", data)
        for f in findings:
            s = (f.get("severity") or "?").lower()
            c = (f.get("confidence") or "?").lower()
            cell[(s, c)] += 1
            total += 1
    return cell, total, n_papers


# ---- look & feel: mirror fig03_code_coverage.py (Nature-figure style) -----
_avail = {f.name for f in fm.fontManager.ttflist}
_FONT = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
              if f in _avail), "sans-serif")
INK, MUTE = "#000000", "#767676"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_FONT, "DejaVu Sans", "Liberation Sans"],
    "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlepad": 10,
    "axes.edgecolor": "#4D4D4D", "axes.linewidth": 1.0,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})

# white -> audit blue (#3775BA, the audit colour from the cost figure) sequential ramp
CMAP = LinearSegmentedColormap.from_list("audit_seq", ["#FFFFFF", "#3775BA"])


def main() -> None:
    cell, total, n_papers = load_counts()

    # counts[i, j] = severity ORDER[i] x confidence ORDER[j]
    counts = np.array([[cell[(s, c)] for c in ORDER] for s in ORDER], dtype=float)
    row_tot = counts.sum(axis=1)                       # per-severity n
    col_tot = counts.sum(axis=0)                       # per-confidence n
    vmax = counts.max()                                # absolute colour scale

    labels = [o.capitalize() for o in ORDER]

    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    fig.subplots_adjust(left=0.17, right=0.88, top=0.78, bottom=0.12)

    im = ax.imshow(counts, cmap=CMAP, vmin=0.0, vmax=vmax, aspect="auto")

    # annotate: raw finding count (colour encodes the same count, absolute scale)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(counts[i, j])}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=INK)

    # ticks: confidence on top with marginal n, severity on left with marginal n
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{l}\n(n={int(col_tot[j])})" for j, l in enumerate(labels)],
                       fontsize=11)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Auditor confidence", fontsize=12, labelpad=8)

    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{l}\n(n={int(row_tot[i])})" for i, l in enumerate(labels)],
                       fontsize=11)
    ax.set_ylabel("Auditor severity", fontsize=12, labelpad=8)

    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # gridlines between cells
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Findings", fontsize=10.5)
    cbar.set_ticks([0, 50, 100, 150])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=9.5)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"figS1_severity_confidence.{ext}")
    plt.close(fig)

    print(f"wrote figures/figS1_severity_confidence.{{png,pdf,svg}}  "
          f"(n={total} findings from {n_papers} source-present papers)")
    print("counts (severity row x confidence col):")
    for i, s in enumerate(labels):
        cells = "  ".join(f"{labels[j][0]}:{int(counts[i, j]):4d}" for j in range(3))
        print(f"  {s:>6} (n={int(row_tot[i]):3d}) | {cells}")


if __name__ == "__main__":
    main()
