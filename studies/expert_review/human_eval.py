"""
Minimal loading + statistics helpers needed to reproduce two figures:
figS4_reviewer_time and fig06_expert_review
(generated from a private notebook, not part of this release).

This is a trimmed, anonymised extract of the full human_eval.py module used
elsewhere in the project: it keeps only the pieces the standalone notebook actually
calls. Removed entirely: the author-response workbook, rater-identity tracking (names,
roles, anonymisation crosswalk), and every agreement statistic not used by the two
target figures (Cohen's kappa, Scott's pi, Krippendorff's alpha, Gwet's AC1 alone,
Brennan-Prediger, ICC, Bowker's test) — those live in the full module.

Two rating workbooks are read, each with one sheet per paper and one row per finding:
  expert   the first rater's pass over every paper
  expert2  a second, independent rater's blind re-rating of a subset of papers
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# 0.  Paths and vocabulary
# --------------------------------------------------------------------------

_WORKBOOK_NAMES = (
    "ratings_primary.xlsx",        # first rater's pass over every paper
    "ratings_second_rater.xlsx",   # blind re-rating of a subset, for agreement
)


def _find_project_root() -> Path:
    """Locate the folder containing the two rating workbooks.

    Searches this file's own directory and its ancestors, so the module keeps
    working however deep it ends up nested relative to the workbooks (unlike a
    fixed `parent.parent`, which only works at one specific folder depth).
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if all((candidate / name).exists() for name in _WORKBOOK_NAMES):
            return candidate
    raise FileNotFoundError(
        "Could not find the AuditOwl rating workbooks "
        f"({' / '.join(_WORKBOOK_NAMES)}) in {here} or any parent directory. "
        "Place them next to this folder (or an ancestor of it)."
    )


PROJECT_ROOT = _find_project_root()

FILES = {
    "expert": PROJECT_ROOT / _WORKBOOK_NAMES[0],
    "expert2": PROJECT_ROOT / _WORKBOOK_NAMES[1],
}

# --- discrepancy taxonomy (the audit's own four classes) -------------------
CATEGORY_MAP = {
    "missing": "missing",
    "methodology": "methodology",
    "bug": "bug",
    "mismatch": "mismatch",
    "difference": "mismatch",
}

# --- correctness -------------------------------------------------------------
# Experts were given: True / Partially true / False / Unverifiable
CORRECTNESS_MAP = {
    "true": "True",
    "partially true": "Partially true",
    "false": "False",
    "unverifiable": "Unverifiable",
}
CORRECTNESS_ORDER = ["True", "Partially true", "False", "Unverifiable"]
# Ordinal score = "how much of the claim stands". Unverifiable is *not* a point on
# this scale (it is an absence of a judgement), so it is NaN here.
CORRECTNESS_SCORE = {"True": 1.0, "Partially true": 0.5, "False": 0.0, "Unverifiable": np.nan}

# --- relevance ---------------------------------------------------------------
# Experts: Relevant / Minor / Trivial / Unverifiable / N/A (correctness False)
RELEVANCE_MAP = {
    "relevant": "Relevant",
    "minor": "Minor",
    "trivial": "Trivial",
    "unverifiable": "Unverifiable",
    "n/a (correctness false)": "N/A",
    "n/a": "N/A",
}
RELEVANCE_ORDER = ["Relevant", "Minor", "Trivial", "Unverifiable", "N/A"]
RELEVANCE_SCORE = {"Relevant": 2.0, "Minor": 1.0, "Trivial": 0.0,
                   "Unverifiable": np.nan, "N/A": np.nan}


# --------------------------------------------------------------------------
# 1.  Loading
# --------------------------------------------------------------------------

def _norm(x) -> str | float:
    """Normalise a cell to a lower-case string, or NaN when empty."""
    if x is None:
        return np.nan
    if isinstance(x, bool):                     # Excel turned "True"/"False" into bools
        return "true" if x else "false"
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    s = str(x).strip()
    return s.lower() if s and s.lower() != "nan" else np.nan


def _map(value, mapping, what):
    v = _norm(value)
    if not isinstance(v, str):
        return np.nan
    if v not in mapping:
        raise ValueError(f"unmapped {what} value: {value!r}")
    return mapping[v]


# Paper 1339 is a special case: its original findings were hand-written after a
# bad-input audit rather than produced by the pipeline, so they could not sit in
# a denominator measuring whether the auditor's findings are correct. It has
# since been properly re-audited and single-rated by R11 (no second pass); its 9
# findings are folded into ratings_primary.xlsx like any other paper's.
EXCLUDED_PAPERS = set()


def load_workbook_findings(source: str) -> pd.DataFrame:
    """Read one workbook into a tidy one-row-per-finding frame.

    Rows from EXCLUDED_PAPERS are dropped here so every consumer of this loader
    shares one denominator.
    """
    path = FILES[source]
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    rows = []
    for sheet_name, raw in sheets.items():
        if sheet_name in ("Instructions", "Overview"):
            continue
        # header row = the one whose first cell is literally "#"
        hdr_idx = None
        for i in range(min(12, len(raw))):
            if str(raw.iloc[i, 0]).strip() == "#":
                hdr_idx = i
                break
        if hdr_idx is None:
            continue
        header = [str(c).strip() if c is not None else "" for c in raw.iloc[hdr_idx]]
        body = raw.iloc[hdr_idx + 1:].copy()
        body.columns = header + [f"extra{j}" for j in range(len(body.columns) - len(header))]

        for _, r in body.iterrows():
            first = r.iloc[0]
            if first is None or (isinstance(first, float) and np.isnan(first)):
                continue
            if "FULL AUTHOR" in str(first).upper():   # verbatim-response block: stop
                break
            if not re.fullmatch(r"\d+", str(first).strip().rstrip(".0")):
                continue
            rows.append(
                {
                    "source": source,
                    "paper": str(sheet_name).strip(),
                    "finding_no": int(float(first)),
                    "issue_id": r.get("Issue id"),
                    "category_raw": r.get("Category"),
                    "location": r.get("Location"),
                    "claim": r.get("Claim"),
                    "minutes_raw": r.get("Minutes worked"),
                    "correctness_raw": r.get("Correctness"),
                    "relevance_raw": r.get("Relevance"),
                }
            )

    df = pd.DataFrame(rows)
    if EXCLUDED_PAPERS:
        keep = ~df["paper"].astype(str).isin(EXCLUDED_PAPERS)
        if not keep.all():
            df = df[keep].reset_index(drop=True)
    df["category"] = df["category_raw"].map(lambda v: _map(v, CATEGORY_MAP, "category"))
    df["correctness"] = df["correctness_raw"].map(lambda v: _map(v, CORRECTNESS_MAP, "correctness"))
    df["relevance"] = df["relevance_raw"].map(lambda v: _map(v, RELEVANCE_MAP, "relevance"))
    df["minutes"] = pd.to_numeric(df["minutes_raw"], errors="coerce")

    # derived ordinal / binary codings
    df["correctness_score"] = df["correctness"].map(CORRECTNESS_SCORE)
    df["relevance_score"] = df["relevance"].map(RELEVANCE_SCORE)
    df["stands"] = df["correctness"].map({"True": 1, "Partially true": 1, "False": 0})       # Unverifiable -> NaN
    df["fully_true"] = df["correctness"].map({"True": 1, "Partially true": 0, "False": 0})
    df["at_least_minor"] = df["relevance"].map({"Relevant": 1, "Minor": 1, "Trivial": 0})    # Unverifiable/NA -> NaN
    df["is_relevant"] = df["relevance"].map({"Relevant": 1, "Minor": 0, "Trivial": 0})
    df["rated"] = df["correctness"].notna() | df["relevance"].notna()
    return df


def build_paired(expert: pd.DataFrame, expert2: pd.DataFrame) -> pd.DataFrame:
    """Findings rated by BOTH expert raters (the inter-rater subset)."""
    keys = ["paper", "finding_no"]
    a = expert.set_index(keys)
    b = expert2.set_index(keys)
    common = a.index.intersection(b.index)
    out = pd.DataFrame(index=common).sort_index()
    for col in ["issue_id", "category", "location", "claim"]:
        out[col] = a.loc[common, col]
    for tag, src in (("r1", a), ("r2", b)):
        for col in ["correctness", "relevance", "correctness_score", "relevance_score",
                    "stands", "fully_true", "at_least_minor", "is_relevant", "minutes"]:
            out[f"{col}_{tag}"] = src.loc[common, col]
    out = out.reset_index()
    # keep only findings the second rater actually completed
    out = out[out["correctness_r2"].notna() | out["relevance_r2"].notna()].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# 2.  A single agreement helper: the confusion matrix
# --------------------------------------------------------------------------

def _clean_pair(a, b, categories=None):
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    if categories is None:
        categories = sorted(set(a) | set(b), key=str)
    return a, b, list(categories)


def confusion(a, b, categories=None) -> pd.DataFrame:
    """Cross-tabulation of two raters' verdicts on the same findings."""
    a, b, cats = _clean_pair(a, b, categories)
    M = pd.DataFrame(0, index=cats, columns=cats, dtype=int)
    for x, y in zip(a, b):
        M.loc[x, y] += 1
    M.index.name = "rater 1"
    M.columns.name = "rater 2"
    return M


# --------------------------------------------------------------------------
# 3.  Uncertainty: Wilson interval and a paper-clustered bootstrap
# --------------------------------------------------------------------------

def wilson_ci(k, n, conf=0.95):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _cluster_index_draws(cluster_labels, n_boot, seed):
    """Row-index arrays for `n_boot` cluster-resampled datasets.

    Resampling clusters (papers) rather than rows is what makes the interval honest:
    one paper contributes several correlated findings.
    """
    labels = np.asarray(cluster_labels)
    groups = {c: np.flatnonzero(labels == c) for c in pd.unique(labels)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    for row in picks:
        yield np.concatenate([groups[keys[i]] for i in row])


def cluster_bootstrap(df, cluster_col, statistic, n_boot=2000, seed=20260729, conf=0.95):
    """Paper-clustered bootstrap confidence interval for one statistic.

    `statistic` is a callable(sub_df) -> float; a resample that yields a non-finite
    value (e.g. from a degenerate draw) is simply dropped from the interval.
    """
    point = statistic(df)
    draws = [statistic(df.take(idx))
             for idx in _cluster_index_draws(df[cluster_col].to_numpy(), n_boot, seed)]
    arr = np.asarray(draws, dtype=float)
    ok = arr[np.isfinite(arr)]
    lo_q, hi_q = (1 - conf) / 2 * 100, (1 + conf) / 2 * 100
    lo, hi = (np.nan, np.nan)
    if len(ok) > 10:
        lo, hi = np.percentile(ok, [lo_q, hi_q])
    return dict(point=point, lo=float(lo), hi=float(hi), n_valid=int(len(ok)), n_boot=int(n_boot))


# --------------------------------------------------------------------------
# 4.  Plot styling
# --------------------------------------------------------------------------

def use_paper_style():
    """Matplotlib defaults matched to the manuscript figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
