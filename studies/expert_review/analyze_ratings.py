#!/usr/bin/env python3
"""Canonical analysis of the human evaluation study, sourced ONLY from
ratings_joined.xlsx (source of truth, including its own Category/Severity
columns -- not cross-referenced against audits/*/findings_verified.json).
Reproduces every number cited in the manuscript's Human Evaluation section
(main text + appendix S4.1/S4.2) so they no longer live only in an ad hoc
terminal session.

Aggregation rule (as stated in the manuscript): for double-judged findings,
take the lower rating on each axis (True > Partially true > False; Relevant >
Minor > Trivial), or the rating that was not Unverifiable.
"""
import os
import re

import numpy as np
import openpyxl
import pandas as pd
try:
    import krippendorff
except ImportError:
    krippendorff = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(HERE, "ratings_joined.xlsx")

CORR_RANK = {"True": 3, "Partially true": 2, "False": 1}
REL_RANK = {"Relevant": 3, "Minor": 2, "Trivial": 1}


def min_rule(v1, v2, rank):
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    u1, u2 = v1 not in rank, v2 not in rank
    if u1 and u2:
        return v1
    if u1:
        return v2
    if u2:
        return v1
    return v1 if rank[v1] <= rank[v2] else v2


# Paper 1339 is a special case: its original findings were hand-written after a
# bad-input audit rather than produced by the pipeline, so they could not sit in
# a denominator measuring whether the auditor's findings are correct. It has
# since been properly re-audited and single-rated by R11 (no second pass); its 9
# findings are folded into ratings_joined.xlsx like any other paper's.
EXCLUDED_PAPERS = set()


def load_joined():
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb["joined"]
    hdr = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        rows.append(dict(zip(hdr, row)))
    df = pd.DataFrame(rows)
    paper_col = hdr[0]
    dropped = df[paper_col].astype(str).isin(EXCLUDED_PAPERS).sum()
    if dropped:
        df = df[~df[paper_col].astype(str).isin(EXCLUDED_PAPERS)].reset_index(drop=True)
        print(f"(excluded {dropped} findings from paper(s) {sorted(EXCLUDED_PAPERS)}; "
              f"see EXCLUDED_PAPERS)")
    return df


def parse_minutes(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.match(r"\s*([\d.]+)", v)
        if m:
            return float(m.group(1))
    return None


def main():
    df = load_joined()
    print(f"N findings (joined workbook) = {len(df)}")

    # ---- combined verdicts under the manuscript's min-rule ----
    df["correctness"] = [
        min_rule(r["R1 correctness"], r["R2 correctness"], CORR_RANK)
        for _, r in df.iterrows()
    ]
    df["relevance"] = [
        ("N/A (correctness False)" if corr == "False"
         else min_rule(r["R1 relevance"], r["R2 relevance"], REL_RANK))
        for corr, (_, r) in zip(df["correctness"], df.iterrows())
    ]
    df["double_coded"] = df["R2 correctness"].notna()

    # ---- §4.2 headline correctness distribution ----
    vc = df["correctness"].value_counts()
    n = len(df)
    print(f"\n=== Correctness distribution (all {n}, min-rule) ===")
    for k in ("True", "Partially true", "False", "Unverifiable"):
        c = vc.get(k, 0)
        print(f"  {k:16s} {c:3d}  ({c/n*100:.1f}%)")

    correct = df[df["correctness"].isin(["True", "Partially true"])]
    print(f"\nN correct (True+Partially true) = {len(correct)}")

    # ---- relevance distribution among correct findings ----
    print("\n=== Relevance distribution among correct findings ===")
    rvc = correct["relevance"].value_counts()
    nc = len(correct)
    for k in ("Relevant", "Minor", "Trivial", "Unverifiable"):
        c = rvc.get(k, 0)
        print(f"  {k:16s} {c:3d}  ({c/nc*100:.1f}%)")

    n_false = (df["correctness"] == "False").sum()
    n_trivial = (df["relevance"] == "Trivial").sum()
    fp_rate = (n_false + n_trivial) / n
    print(f"\nEffective false-positive rate (False + Trivial)/N = "
          f"({n_false}+{n_trivial})/{n} = {fp_rate*100:.1f}%")

    # ---- who contributed the False verdicts? external (R1-only) vs author (R2) ----
    print("\n=== False verdicts: source ===")
    for _, r in df[df["correctness"] == "False"].iterrows():
        print(f"  paper={r['Paper']} issue={r['Issue id']}  "
              f"R1={r['R1 name']}/{r['R1 correctness']}  "
              f"R2={r['R2 name']}/{r['R2 correctness']}")

    # ---- double-coded set: raw agreement ----
    dc = df[df["double_coded"]].copy()
    print(f"\n=== Double-coded set: N={len(dc)} ===")
    agree_corr = (dc["R1 correctness"] == dc["R2 correctness"]).sum()
    agree_rel = (dc["R1 relevance"] == dc["R2 relevance"]).sum()
    print(f"correctness exact agreement: {agree_corr}/{len(dc)} ({agree_corr/len(dc)*100:.1f}%)")
    print(f"relevance exact agreement:   {agree_rel}/{len(dc)} ({agree_rel/len(dc)*100:.1f}%)")

    full_disagree = dc[(dc["R1 correctness"] == "True") & (dc["R2 correctness"] == "False")
                        | (dc["R1 correctness"] == "False") & (dc["R2 correctness"] == "True")]
    print(f"full (True vs False) correctness disagreements: {len(full_disagree)}")
    for _, r in full_disagree.iterrows():
        print(f"    paper={r['Paper']} issue={r['Issue id']}")

    # ---- agreement stats: Krippendorff alpha + Gwet's AC1 ----
    def gwet_ac1(r1, r2, categories):
        n = len(r1)
        k = len(categories)
        po = np.mean([a == b for a, b in zip(r1, r2)])
        pi_bar = {c: (sum(a == c for a in r1) + sum(b == c for b in r2)) / (2 * n) for c in categories}
        pe = sum(p * (1 - p) for p in pi_bar.values()) / (k - 1)
        if pe == 1:
            return 1.0
        return (po - pe) / (1 - pe)

    def alpha_nominal(r1, r2, categories):
        if krippendorff is None:
            return None
        cat_idx = {c: i for i, c in enumerate(categories)}
        data = np.array([[cat_idx[v] for v in r1], [cat_idx[v] for v in r2]], dtype=float)
        return krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")

    def bootstrap_ci(r1, r2, fn, categories, papers, n_boot=5000, seed=0):
        """Paper-clustered: resample papers with replacement, keep all of a
        resampled paper's findings together, so within-paper correlation
        (same rater pair, one sitting) is not treated as independent draws --
        matching the clustered bootstrap already used for Figure 6a."""
        rng = np.random.default_rng(seed)
        by_paper = {}
        for i, p in enumerate(papers):
            by_paper.setdefault(p, []).append(i)
        paper_ids = list(by_paper)
        vals = []
        for _ in range(n_boot):
            idx = [i for p in rng.choice(paper_ids, size=len(paper_ids), replace=True)
                   for i in by_paper[p]]
            try:
                vals.append(fn([r1[i] for i in idx], [r2[i] for i in idx], categories))
            except (ValueError, ZeroDivisionError):
                continue
        vals = [v for v in vals if v is not None]
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    r1c, r2c = dc["R1 correctness"].tolist(), dc["R2 correctness"].tolist()
    r1r, r2r = dc["R1 relevance"].tolist(), dc["R2 relevance"].tolist()
    corr_cats = ["True", "Partially true", "False", "Unverifiable"]
    rel_cats = ["Relevant", "Minor", "Trivial", "Unverifiable", "N/A (correctness False)"]

    def to_binary_corr(v):
        return "correct" if v in ("True", "Partially true") else "not-correct"

    def to_binary_rel(v):
        return "relevant" if v in ("Relevant", "Minor") else "not-relevant"

    print(f"\n=== Correctness agreement (double-coded, n={len(r1c)}) ===")
    a1 = alpha_nominal(r1c, r2c, corr_cats)
    ac1 = gwet_ac1(r1c, r2c, corr_cats)
    print(f"  4-way: Krippendorff alpha={a1}, Gwet AC1={ac1:.3f}")
    r1cb = [to_binary_corr(v) for v in r1c]
    r2cb = [to_binary_corr(v) for v in r2c]
    a1b = alpha_nominal(r1cb, r2cb, ["correct", "not-correct"])
    ac1b = gwet_ac1(r1cb, r2cb, ["correct", "not-correct"])
    print(f"  binary: Krippendorff alpha={a1b}, Gwet AC1={ac1b:.3f}")

    print(f"\n=== Relevance agreement (double-coded, n={len(r1r)}) ===")
    a2 = alpha_nominal(r1r, r2r, rel_cats)
    ac2 = gwet_ac1(r1r, r2r, rel_cats)
    print(f"  5-way: Krippendorff alpha={a2}, Gwet AC1={ac2:.3f}")
    r1rb = [to_binary_rel(v) for v in r1r]
    r2rb = [to_binary_rel(v) for v in r2r]
    a2b = alpha_nominal(r1rb, r2rb, ["relevant", "not-relevant"])
    ac2b = gwet_ac1(r1rb, r2rb, ["relevant", "not-relevant"])
    print(f"  binary: Krippendorff alpha={a2b}, Gwet AC1={ac2b:.3f}")

    print("\n=== Bootstrap 95% CIs (paper-clustered, n_boot=5000) ===")
    dc_papers = dc["Paper"].tolist()
    def alpha_fn(a, b, cats):
        return alpha_nominal(a, b, cats)
    for label, r1v, r2v, cats, fn in [
        ("correctness alpha 4-way", r1c, r2c, corr_cats, alpha_fn),
        ("correctness AC1 4-way", r1c, r2c, corr_cats, gwet_ac1),
        ("correctness alpha binary", r1cb, r2cb, ["correct", "not-correct"], alpha_fn),
        ("correctness AC1 binary", r1cb, r2cb, ["correct", "not-correct"], gwet_ac1),
        ("relevance alpha 4-way", r1r, r2r, rel_cats, alpha_fn),
        ("relevance AC1 4-way", r1r, r2r, rel_cats, gwet_ac1),
        ("relevance alpha binary", r1rb, r2rb, ["relevant", "not-relevant"], alpha_fn),
        ("relevance AC1 binary", r1rb, r2rb, ["relevant", "not-relevant"], gwet_ac1),
    ]:
        lo, hi = bootstrap_ci(r1v, r2v, fn, cats, dc_papers)
        print(f"  {label:28s} 95% CI [{lo:.3f}, {hi:.3f}]")

    # ---- timing stats ----
    print("\n=== Timing ===")
    r1_min = df["R1 minutes"].map(parse_minutes).dropna().tolist()
    r2_min = df["R2 minutes"].map(parse_minutes).dropna().tolist()
    all_min = r1_min + r2_min
    print(f"n timed judgments = {len(all_min)} (R1={len(r1_min)}, R2={len(r2_min)})")
    print(f"median = {np.median(all_min):.1f} min, mean = {np.mean(all_min):.2f} min")

    # Per-(rater, paper) sitting time (Fig. S4b) is NOT computed here: the
    # figure is rendered by figures/figS4_reviewer_time.py from the two rater
    # workbooks, and duplicating it off the joined workbook gave a different
    # median (45 vs 42) for the same quantity.

    # ---- relevance by rater type (external vs author) ----
    # Section 4.2 compares how relevant each group judged findings to be, on the
    # three-point scale. R10/R11 are the two authors, per the roster table.
    REL3 = {"Relevant": 3, "Minor": 2, "Trivial": 1}
    AUTHOR_RATERS = {"R10", "R11"}

    def _split(pairs):
        ext, auth = [], []
        for who, rel in pairs:
            v = REL3.get(rel)
            if who and v:
                (auth if str(who).strip() in AUTHOR_RATERS else ext).append(v)
        return ext, auth

    all_pairs = [(r[f"R{k} name"], r[f"R{k} relevance"])
                 for _, r in df.iterrows() for k in (1, 2)]
    ext, auth = _split(all_pairs)
    print("\n=== Relevance by rater type (3-point scale) ===")
    print(f"  full sample     external {np.mean(ext):.2f} (n={len(ext)})  "
          f"author {np.mean(auth):.2f} (n={len(auth)})")

    dbl = [r for _, r in df.iterrows()
           if r["R1 name"] and pd.notna(r["R2 name"])
           and REL3.get(r["R1 relevance"]) and REL3.get(r["R2 relevance"])]
    de, da = _split([(r[f"R{k} name"], r[f"R{k} relevance"]) for r in dbl for k in (1, 2)])
    print(f"  double-judged   external {np.mean(de):.2f}  author {np.mean(da):.2f}  "
          f"({len(dbl)} findings)")

    # ---- rater roster ----
    print("\n=== Rater roster ===")
    roster = {}

    def add(name, minutes, paper):
        if not name:
            return
        key = str(name).strip().lower()
        d = roster.setdefault(key, {"papers": set(), "findings": 0, "minutes": 0.0})
        d["papers"].add(paper)
        d["findings"] += 1
        if minutes is not None and not (isinstance(minutes, float) and np.isnan(minutes)):
            d["minutes"] += minutes

    for _, r in df.iterrows():
        add(r["R1 name"], parse_minutes(r["R1 minutes"]), r["Paper"])
        if pd.notna(r["R2 name"]):
            add(r["R2 name"], parse_minutes(r["R2 minutes"]), r["Paper"])

    rows = []
    for name, d in roster.items():
        n_find = d["findings"]
        rows.append((name, len(d["papers"]), n_find, d["minutes"], d["minutes"] / n_find))
    rows.sort(key=lambda x: -x[3])
    print(f"{'name':16s}{'#papers':>8s}{'#findings':>10s}{'time(min)':>10s}{'min/finding':>12s}")
    for name, npap, nfind, tmin, per in rows:
        print(f"{name:16s}{npap:8d}{nfind:10d}{tmin:10.1f}{per:12.1f}")

    total_papers = sum(npap for _, npap, *_ in rows)
    total_findings = sum(nfind for _, _, nfind, *_ in rows)
    total_min = sum(tmin for *_, tmin, _ in rows)
    print(f"\nsum across raters: papers(non-distinct)={total_papers} findings={total_findings} minutes={total_min:.1f}")
    n_single, n_double = len(df), int(df["double_coded"].sum())
    print(f"(sanity: findings should be {n_single} + {n_double} = {n_single + n_double} counted-with-duplication across raters)")


if __name__ == "__main__":
    main()
