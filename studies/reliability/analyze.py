#!/usr/bin/env python3
"""
analyze.py — quantify the test-retest reliability of the auditor across the
repeated audit->verify runs produced under studies/reliability/<paper>/run_NN/.

Design choices:

  * Findings are aligned across runs on OBJECTIVE ANCHORS (category + file +
    line-range overlap), never on the LLM's severity label. Severity/verdict
    agreement is reported as a *secondary*, clearly-labelled layer. This keeps
    the headline reliability numbers verifiable rather than self-referential.

  * Two independent alignment methods are computed so no metric is an artefact
    of the matcher:
        - anchor   : deterministic (category, file, overlapping lines)
        - semantic : TF-IDF cosine over title+claim+concern (sklearn), then
                     single-link clustering at a threshold
    Metrics are reported under both; large divergence is a red flag.

Outputs:
    studies/reliability/data/metrics_<paper>.json   (per paper)
    studies/reliability/data/metrics_all.json       (combined)

Usage:
    python studies/reliability/analyze.py                 # analyse all selected papers
    python studies/reliability/analyze.py --paper 1023... # one paper
    python studies/reliability/analyze.py --selftest      # validate the math, no runs needed
"""
from __future__ import annotations
import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path

import numpy as np

def _find_root(start: Path) -> Path:
    """Repo root = the ancestor holding `audits/`. Found by content rather than a
    fixed number of `.parent` hops, so this survives the file moving depth."""
    for candidate in (start, *start.parents):
        if (candidate / "audits").is_dir():
            return candidate
    raise SystemExit(f"no `audits/` directory found above {start}")


ROOT = _find_root(Path(__file__).resolve().parent)
RB = Path(__file__).resolve().parent
DATA = RB / "data"
SEV_ORD = {"low": 1, "medium": 2, "high": 3}

# -----------------------------------------------------------------------------
# loading / normalisation
# -----------------------------------------------------------------------------

def _strip_ns(fid: str) -> str:
    """Findings ids are namespaced '<dir>/<slug>'; keep the slug only."""
    return fid.split("/", 1)[1] if "/" in fid else fid


def norm_file(f: str | None) -> str:
    if not f:
        return ""
    f = str(f).strip().replace("\\", "/")
    # drop a leading clone-root segment so code/<owner>__<repo>/x == x
    parts = f.split("/")
    if parts and parts[0] == "code":
        parts = parts[1:]
        if parts and "__" in parts[0]:
            parts = parts[1:]
    return "/".join(parts)


def normalize(f: dict) -> dict:
    return {
        "slug": _strip_ns(f.get("id", "")),
        "category": f.get("category", "?"),
        "topic": f.get("topic", ""),
        "title": f.get("title", ""),
        "severity": f.get("severity", ""),
        "confidence": f.get("confidence", ""),
        "status": f.get("status", "finding"),
        "file": norm_file(f.get("file")),
        "ls": f.get("line_start"),
        "le": f.get("line_end"),
        "claim": f.get("claim", ""),
        "concern": f.get("concern", ""),
        "verdict": f.get("verdict"),   # present only in findings_verified.json
        "text": " ".join(str(f.get(k, "")) for k in ("title", "claim", "concern")),
    }


def load_runs(paper_dir: Path) -> list[dict]:
    """Return [{run, findings:[...], verified:[...]}] for run_* dirs present."""
    runs = []
    for rd in sorted(paper_dir.glob("run_*")):
        fj = rd / "findings.json"
        vj = rd / "findings_verified.json"
        if not fj.exists():
            continue
        find = json.loads(fj.read_text()).get("findings", [])
        ver = json.loads(vj.read_text()).get("findings", []) if vj.exists() else []
        runs.append({
            "run": rd.name,
            "findings": [normalize(x) for x in find],
            "verified": [normalize(x) for x in ver],
        })
    return runs

# -----------------------------------------------------------------------------
# alignment: cluster findings across runs into canonical issues
# -----------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _lines_overlap(a, b, window=10):
    als, ale = a["ls"], a["le"]
    bls, ble = b["ls"], b["le"]
    if als is None or bls is None:
        return None  # not a line-anchored comparison
    ale = ale if ale is not None else als
    ble = ble if ble is not None else bls
    return not (ale + window < bls or ble + window < als)


def anchor_match(a, b, window=10) -> bool:
    if a["category"] != b["category"]:
        return False
    if a["file"] and b["file"] and a["file"] == b["file"]:
        ov = _lines_overlap(a, b, window)
        if ov is None:        # both paper-level / no lines -> same file+category
            return True
        return ov
    return False


def semantic_sim_matrix(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(min_df=1, stop_words="english")
    X = vec.fit_transform(texts)
    return cosine_similarity(X)


def cluster(runs, method="anchor", window=10, sem_thresh=0.45):
    """Flatten findings across runs, cluster into canonical issues.

    Returns (items, labels) where items[i] = (run_idx, finding) and labels[i]
    is the canonical-issue id. Within a single run, multiple findings can fall
    into the same issue; we later collapse per-run.
    """
    items = []
    for ri, r in enumerate(runs):
        for f in r["findings"]:
            items.append((ri, f))
    n = len(items)
    uf = UnionFind(n)
    if method == "anchor":
        for i, j in combinations(range(n), 2):
            if anchor_match(items[i][1], items[j][1], window):
                uf.union(i, j)
    elif method == "semantic":
        if n >= 2:
            S = semantic_sim_matrix([it[1]["text"] for it in items])
            for i, j in combinations(range(n), 2):
                # require same category AND high text similarity
                if items[i][1]["category"] == items[j][1]["category"] and S[i, j] >= sem_thresh:
                    uf.union(i, j)
    else:
        raise ValueError(method)
    labels = [uf.find(i) for i in range(n)]
    # renumber labels 0..K-1
    remap = {l: k for k, l in enumerate(sorted(set(labels)))}
    labels = [remap[l] for l in labels]
    return items, labels


def incidence(runs, items, labels):
    """Binary matrix R x K: did run r surface canonical issue k?"""
    R = len(runs)
    K = (max(labels) + 1) if labels else 0
    M = np.zeros((R, K), dtype=int)
    for (ri, _), lab in zip(items, labels):
        M[ri, lab] = 1
    return M

# -----------------------------------------------------------------------------
# reliability statistics
# -----------------------------------------------------------------------------

def krippendorff_nominal(units):
    """Krippendorff's alpha, nominal metric, missing values allowed.

    units: list of lists; each inner list = the values assigned to one unit by
    the coders that rated it (length >= 1). Units with <2 ratings are ignored
    in the coincidence matrix (per Krippendorff).
    """
    vals = sorted({v for u in units for v in u})
    idx = {v: i for i, v in enumerate(vals)}
    V = len(vals)
    if V < 2:
        return 1.0  # everyone agrees on a single value
    o = np.zeros((V, V))
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                o[idx[u[a]], idx[u[b]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return 1.0
    # nominal delta: 0 on diagonal, 1 off-diagonal
    Do = sum(o[c, k] for c in range(V) for k in range(V) if c != k) / n
    De = sum(n_c[c] * n_c[k] for c in range(V) for k in range(V) if c != k) / (n * (n - 1))
    if De == 0:
        return 1.0
    return 1.0 - Do / De


def pairwise_jaccard(M):
    R = M.shape[0]
    js = []
    for i, j in combinations(range(R), 2):
        a, b = M[i] > 0, M[j] > 0
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        js.append(inter / union if union else 1.0)
    return np.array(js)


def bootstrap_ci(x, fn=np.mean, B=2000, seed=0):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [fn(rng.choice(x, size=len(x), replace=True)) for _ in range(B)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))

# -----------------------------------------------------------------------------
# per-paper analysis
# -----------------------------------------------------------------------------

def analyze_paper(paper_dir: Path, methods=("anchor", "semantic")) -> dict:
    runs = load_runs(paper_dir)
    R = len(runs)
    out = {"paper": paper_dir.name, "n_runs": R}
    if R == 0:
        out["error"] = "no runs found"
        return out

    # ---- count-level stability (objective, no matching needed) ----
    counts = [len(r["findings"]) for r in runs]
    highs = [sum(1 for f in r["findings"] if f["severity"] == "high") for r in runs]
    def stat(x):
        x = np.array(x, dtype=float)
        return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "cv": float(x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() else 0.0,
                "min": float(x.min()), "max": float(x.max())}
    out["counts"] = {"n_findings": stat(counts), "n_high": stat(highs),
                     "per_run_findings": counts, "per_run_high": highs}

    # ---- alignment-based stability, under each method ----
    out["alignment"] = {}
    for method in methods:
        items, labels = cluster(runs, method=method)
        M = incidence(runs, items, labels)
        K = M.shape[1]
        det = M.mean(axis=0)                       # detection rate per issue
        stable_core = int((det >= 0.8).sum())
        flaky = int((det <= 0.3).sum())
        midband = int(((det > 0.3) & (det < 0.8)).sum())
        jac = pairwise_jaccard(M)
        # presence/absence alpha: each issue is a unit, each run a coder (0/1)
        pa_units = [list(M[:, k]) for k in range(K)]
        alpha_presence = krippendorff_nominal([u for u in pa_units]) if K else 1.0
        # category alpha on co-detected issues
        # per-run share that is stable-core
        core_set = {k for k in range(K) if det[k] >= 0.8}
        run_core_share = []
        for ri in range(R):
            mine = [k for k in range(K) if M[ri, k]]
            run_core_share.append(np.mean([1.0 if k in core_set else 0.0 for k in mine]) if mine else 1.0)

        # ---- severity agreement on co-detected issues (>=50% runs), transparent ----
        sev_sd, sev_exact = [], []
        for k in range(K):
            if det[k] < 0.5:
                continue
            sevs = []
            for (ri, f), lab in zip(items, labels):
                if lab == k and f["severity"] in SEV_ORD:
                    sevs.append(SEV_ORD[f["severity"]])
            if len(sevs) >= 2:
                sev_sd.append(float(np.std(sevs, ddof=1)))
                maj = max(set(sevs), key=sevs.count)
                sev_exact.append(np.mean([1.0 if s == maj else 0.0 for s in sevs]))

        out["alignment"][method] = {
            "n_canonical_issues": K,
            "detection_rate_hist": np.histogram(det, bins=[0, .1, .3, .5, .7, .9, 1.01])[0].tolist(),
            "stable_core_ge80": stable_core,
            "flaky_le30": flaky,
            "midband": midband,
            "stable_core_fraction": float(stable_core / K) if K else 1.0,
            "mean_run_core_share": float(np.mean(run_core_share)),
            "jaccard_mean": float(jac.mean()) if len(jac) else 1.0,
            "jaccard_ci95": bootstrap_ci(jac),
            "alpha_presence_nominal": float(alpha_presence),
            "severity_within_issue_sd_mean": float(np.mean(sev_sd)) if sev_sd else 0.0,
            "severity_majority_match_rate": float(np.mean(sev_exact)) if sev_exact else 1.0,
        }

    # ---- verifier-stage stability (post-verification = what ships) ----
    if any(r["verified"] for r in runs):
        vruns = [{"run": r["run"], "findings": [f for f in r["verified"] if f["verdict"] == "keep"]}
                 for r in runs]
        survival = []
        for r in runs:
            ver = r["verified"]
            if ver:
                survival.append(np.mean([1.0 if f["verdict"] == "keep" else 0.0 for f in ver]))
        items, labels = cluster(vruns, method="anchor")
        Mk = incidence(vruns, items, labels)
        Kk = Mk.shape[1]
        detk = Mk.mean(axis=0) if Kk else np.array([])
        jak = pairwise_jaccard(Mk)
        pa_units = [list(Mk[:, k]) for k in range(Kk)]
        out["post_verify"] = {
            "survival_rate_mean": float(np.mean(survival)) if survival else float("nan"),
            "survival_per_run": [float(s) for s in survival],
            "n_canonical_issues_kept": Kk,
            "stable_core_ge80": int((detk >= 0.8).sum()) if Kk else 0,
            "stable_core_fraction": float((detk >= 0.8).sum() / Kk) if Kk else 1.0,
            "jaccard_mean": float(jak.mean()) if len(jak) else 1.0,
            "alpha_presence_nominal": float(krippendorff_nominal(pa_units)) if Kk else 1.0,
        }

    # ---- headline reproduction: does "any high-severity finding" agree? ----
    any_high = [1 if h > 0 else 0 for h in highs]
    out["headline"] = {
        "any_high_per_run": any_high,
        "any_high_agreement": float(max(any_high.count(0), any_high.count(1)) / R),
    }
    return out

# -----------------------------------------------------------------------------
# selftest: validate the math without needing any runs
# -----------------------------------------------------------------------------

def selftest():
    ok = True

    # 1) Krippendorff nominal against the canonical textbook example (alpha=0.743)
    # columns = units, rows = coders; "." = missing
    A = "1 2 3 3 2 1 4 1 2 . . .".split()
    B = "1 2 3 3 2 2 4 1 2 5 . .".split()
    C = ". 3 3 3 2 3 4 2 2 5 1 .".split()
    D = "1 2 3 3 2 4 4 1 2 5 1 .".split()
    coders = [A, B, C, D]
    units = []
    for ui in range(len(A)):
        u = [int(c[ui]) for c in coders if c[ui] != "."]
        if len(u) >= 1:
            units.append(u)
    a = krippendorff_nominal(units)
    print(f"[selftest] krippendorff nominal (expect ~0.743): {a:.4f}")
    ok &= abs(a - 0.743) < 0.01

    # 2) perfect agreement -> alpha 1, jaccard 1
    M_perfect = np.ones((10, 5), dtype=int)
    j = pairwise_jaccard(M_perfect)
    print(f"[selftest] jaccard perfect (expect 1.0): {j.mean():.4f}")
    ok &= abs(j.mean() - 1.0) < 1e-9

    # 3) self-vs-self alignment on a real findings.json -> jaccard 1, full stable core
    sel = json.loads((RB / "selection.json").read_text()) if (RB / "selection.json").exists() else None
    if sel:
        paper = sel["selected"][0]["paper"]
        fj = ROOT / "audits" / paper / "findings.json"
        if fj.exists():
            base = [normalize(x) for x in json.loads(fj.read_text())["findings"]]
            runs = [{"run": f"run_{i:02d}", "findings": base, "verified": []} for i in range(1, 11)]
            items, labels = cluster(runs, method="anchor")
            M = incidence(runs, items, labels)
            det = M.mean(axis=0)
            jac = pairwise_jaccard(M)
            print(f"[selftest] self-vs-self anchor: K={M.shape[1]} (= {len(base)} findings), "
                  f"jaccard={jac.mean():.3f} (expect 1.0), stable_core={(det>=0.8).sum()}/{M.shape[1]}")
            ok &= abs(jac.mean() - 1.0) < 1e-9
            ok &= int((det >= 0.8).sum()) == M.shape[1]

            # 4) drop one finding from half the runs -> detection rate 0.5 for it
            import copy
            runs2 = copy.deepcopy(runs)
            for i in range(0, 10, 2):
                runs2[i]["findings"] = base[1:]   # drop first finding
            items, labels = cluster(runs2, method="anchor")
            M2 = incidence(runs2, items, labels)
            det2 = sorted(M2.mean(axis=0))
            print(f"[selftest] drop-1-in-5: detection rates min={det2[0]:.2f} (expect 0.50)")
            ok &= abs(det2[0] - 0.5) < 1e-9

    print("\n[selftest]", "PASS ✅" if ok else "FAIL ❌")
    return ok

# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", help="single paper folder name (under studies/reliability/)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    DATA.mkdir(exist_ok=True)
    sel = json.loads((RB / "selection.json").read_text())
    papers = [args.paper] if args.paper else [r["paper"] for r in sel["selected"]]
    allres = []
    for p in papers:
        pd = RB / p
        if not pd.exists():
            print(f"skip {p}: no sandbox dir yet (run build_sandboxes.py + the workflow first)")
            continue
        res = analyze_paper(pd)
        (DATA / f"metrics_{p}.json").write_text(json.dumps(res, indent=2))
        allres.append(res)
        print(f"analysed {p}: {res.get('n_runs')} runs")
    (DATA / "metrics_all.json").write_text(json.dumps(allres, indent=2))
    print(f"wrote {DATA/'metrics_all.json'}")


if __name__ == "__main__":
    main()
