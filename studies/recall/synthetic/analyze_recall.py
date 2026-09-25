#!/usr/bin/env python3
"""Compute the recall metrics from the adjudicated matches.

Inputs: ANSWER_KEY_<pid>.json x10 + data/matches_final.json (falls back to
matches_auto.json with a loud warning — auto-only numbers are provisional).

Outputs: data/recall_metrics.json + data/recall_report.md with

  - detection@1: mean per-run fraction of seeds detected (primary, deployed setting)
  - detection@K: union over the K runs per paper (tests the ensembling claim)
  - per-category rates (4 cells — no severity axis, every seed is major), Wilson CIs
    (naive, seed-level) + a leave-one-paper-out range as the honest cluster-awareness
    check (10 papers)
  - mechanism-correct (strict) rates where adjudicated
  - per-seed detection counts (0..K) — the miss list for the qualitative autopsy
  - category confusion: seeded category vs reported category of matching findings
  - emergent findings per run (confabulation)
  - baseline re-detection rate
  - clean-verdict stress test: seeds detected per paper — would any seeded repo
    still look clean?

Usage:
    python studies/recall/synthetic/analyze_recall.py
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # same-dir imports (match_findings), regardless of cwd


def _load_common():
    # see match_findings.py for why this isn't sys.path.insert(0, HERE.parent) +
    # `from common import ...` (name collision with the parent dir's pilot scripts)
    spec = importlib.util.spec_from_file_location("recall_common", HERE.parent / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_common()
dump_json, load_json, wilson_ci = _common.dump_json, _common.load_json, _common.wilson_ci
from match_findings import PAPERS, baseline_findings  # noqa: E402

RUNS = HERE / "runs"
DATA = HERE / "data"
RUNS_PER_PAPER = 3


def load_answer_keys() -> list[dict]:
    keys = []
    for pid in PAPERS:
        p = HERE / f"ANSWER_KEY_{pid}.json"
        if p.exists():
            k = load_json(p)
            k["num"] = pid
            keys.append(k)
    return keys


def load_matches():
    final = DATA / "matches_final.json"
    auto = DATA / "matches_auto.json"
    if final.exists():
        return load_json(final)["assignments"], "final"
    if auto.exists():
        print("WARNING: matches_final.json missing — using UNADJUDICATED stage-A matches; "
              "all numbers provisional")
        return load_json(auto)["assignments"], "auto(PROVISIONAL)"
    raise SystemExit("run match_findings.py first")


def finding_category(pid: str, run: str, fid: str) -> str | None:
    for name in ("findings_verified.json", "findings.json"):
        p = RUNS / pid / run / name
        if p.exists():
            for f in load_json(p).get("findings", []):
                if f.get("id") == fid:
                    return f.get("category")
    return None


def rate_block(hits: int, n: int) -> dict:
    lo, hi = wilson_ci(hits, n)
    return {"k": hits, "n": n, "rate": round(hits / n, 4) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    keys = load_answer_keys()
    if not keys:
        raise SystemExit("no ANSWER_KEY_<pid>.json found — run freeze.py first")
    assignments, source = load_matches()
    # derive from the adjudicated matches rather than trusting the constant:
    # the arm was planned for RUNS_PER_PAPER but ran fewer, and a stale value
    # here would misreport the design in the committed metrics
    K = len({a["run"] for a in assignments}) or RUNS_PER_PAPER

    seed_hits: dict[tuple, set] = defaultdict(set)
    mech_hits: dict[tuple, set] = defaultdict(set)
    emergent = defaultdict(int)          # (paper, run) -> count
    baseline_hits = defaultdict(set)     # (paper, baseline id) -> runs
    confusion = defaultdict(int)         # (seeded_cat, reported_cat) -> count
    runs_seen = defaultdict(set)

    by_num = {k["num"]: k for k in keys}
    seed_meta = {(k["num"], s["id"]): s for k in keys for s in k["seeds"]}

    for a in assignments:
        paper, run, tgt = a["paper"], a["run"], a["assigned_to"]
        runs_seen[paper].add(run)
        if tgt.startswith("seed:"):
            sid = tgt.split(":", 1)[1]
            seed_hits[(paper, sid)].add(run)
            if a.get("mechanism_correct"):
                mech_hits[(paper, sid)].add(run)
            s = seed_meta.get((paper, sid))
            rep = finding_category(paper, run, a["fid"])
            if s and rep:
                confusion[(s["category"], rep)] += 1
        elif tgt.startswith("baseline:"):
            baseline_hits[(paper, tgt.split(":", 1)[1])].add(run)
        elif tgt == "emergent":
            emergent[(paper, run)] += 1

    n_unassigned = sum(1 for a in assignments if a["assigned_to"] == "unassigned")

    seeds_flat = []
    for k in keys:
        pid = k["num"]
        k_runs = sorted(runs_seen.get(pid, []))
        for s in k["seeds"]:
            det = seed_hits.get((pid, s["id"]), set())
            seeds_flat.append({
                "paper": pid, "sid": s["id"], "category": s["category"],
                "title": s["title"],
                "n_runs": len(k_runs), "n_detected": len(det),
                "detected_runs": sorted(det),
                "n_mechanism_correct": len(mech_hits.get((pid, s["id"]), set())),
                "detected_at_all": bool(det),
            })

    det1_hits = sum(s["n_detected"] for s in seeds_flat)
    det1_n = sum(s["n_runs"] for s in seeds_flat)
    detK_hits = sum(1 for s in seeds_flat if s["detected_at_all"])
    detK_n = len(seeds_flat)

    def by_field(field, value, at_k):
        sel = [s for s in seeds_flat if s[field] == value]
        if at_k:
            return rate_block(sum(1 for s in sel if s["detected_at_all"]), len(sel))
        return rate_block(sum(s["n_detected"] for s in sel),
                          sum(s["n_runs"] for s in sel))

    cats = sorted({s["category"] for s in seeds_flat})

    lopo = {}
    for k in keys:
        pid = k["num"]
        sel = [s for s in seeds_flat if s["paper"] != pid]
        n1 = sum(s["n_runs"] for s in sel)
        lopo[f"without_{pid}"] = {
            "detection_at_1": round(sum(s["n_detected"] for s in sel) / n1, 4) if n1 else None,
            "detection_at_K": round(sum(1 for s in sel if s["detected_at_all"]) / len(sel), 4)
            if sel else None,
        }

    per_paper = {}
    for k in keys:
        pid = k["num"]
        sel = [s for s in seeds_flat if s["paper"] == pid]
        det = sum(1 for s in sel if s["detected_at_all"])
        base = baseline_findings(pid)
        base_ids = {b["id"] for b in base}
        per_paper[pid] = {
            "n_seeds": len(sel), "seeds_detected_at_K": det,
            "would_look_clean": det == 0,
            "baseline_findings": len(base_ids),
            "baseline_redetected": sum(1 for b in base_ids
                                       if baseline_hits.get((pid, b))),
            "emergent_per_run": {r: emergent.get((pid, r), 0)
                                 for r in sorted(runs_seen.get(pid, []))},
        }

    metrics = {
        "source": source, "runs_per_paper": K,
        "n_seeds": detK_n, "n_trials": det1_n, "n_unassigned_findings": n_unassigned,
        "detection_at_1": rate_block(det1_hits, det1_n),
        "detection_at_K": rate_block(detK_hits, detK_n),
        "mechanism_correct_at_K": rate_block(
            sum(1 for s in seeds_flat if s["n_mechanism_correct"] > 0), detK_n),
        "by_category_at_1": {c: by_field("category", c, False) for c in cats},
        "by_category_at_K": {c: by_field("category", c, True) for c in cats},
        "leave_one_paper_out": lopo,
        "category_confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
        "per_paper": per_paper,
        "per_seed": seeds_flat,
    }
    DATA.mkdir(exist_ok=True)
    dump_json(metrics, DATA / "recall_metrics.json")

    L = []
    K_obs = max((s["n_runs"] for s in seeds_flat), default=K)
    L.append(f"# MAJOR-only seeded-defect recall — results ({source})\n")
    d1, dk = metrics["detection_at_1"], metrics["detection_at_K"]
    L.append(f"**detection@1** (single audit): {d1['k']}/{d1['n']} = **{d1['rate']:.0%}** "
             f"(95% CI {d1['ci95'][0]:.0%}-{d1['ci95'][1]:.0%})")
    L.append(f"**detection@{K_obs}** (union of {K_obs} runs): {dk['k']}/{dk['n']} = **{dk['rate']:.0%}** "
             f"(95% CI {dk['ci95'][0]:.0%}-{dk['ci95'][1]:.0%})\n")
    L.append("| category | detection@1 | detection@K |")
    L.append("|---|---|---|")
    for c in cats:
        a, b = metrics["by_category_at_1"][c], metrics["by_category_at_K"][c]
        L.append(f"| {c} | {a['k']}/{a['n']} ({a['rate']:.0%}) | {b['k']}/{b['n']} ({b['rate']:.0%}) |")
    L.append("\n## Clean-verdict stress test")
    for num, pp in per_paper.items():
        flag = "WOULD STILL LOOK CLEAN" if pp["would_look_clean"] else "defects surfaced"
        L.append(f"- paper {num}: {pp['seeds_detected_at_K']}/{pp['n_seeds']} seeds found -> {flag}; "
                 f"baseline re-detected {pp['baseline_redetected']}/{pp['baseline_findings']}; "
                 f"emergent/run {list(pp['emergent_per_run'].values())}")
    L.append("\n## Misses (0-detection seeds -> qualitative autopsy targets)")
    misses = [s for s in seeds_flat if not s["detected_at_all"]]
    if not misses:
        L.append("- none")
    for s in misses:
        L.append(f"- {s['paper']}/{s['sid']} [{s['category']}] {s['title']}")
    L.append(f"\nLeave-one-paper-out detection@1 range: "
             f"{sorted(v['detection_at_1'] for v in lopo.values())}")
    if n_unassigned:
        L.append(f"\nWARNING: {n_unassigned} findings still 'unassigned' — finish stage B.")
    (DATA / "recall_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote data/recall_metrics.json + data/recall_report.md")


if __name__ == "__main__":
    main()
