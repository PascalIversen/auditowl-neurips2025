#!/usr/bin/env python3
"""Reproduces the paper's SciCoQA real-arm numbers (Fig. 7b / Section 4.3).

Scoring follows SciCoQA's protocol, so both sides of the head-to-head are
judged the same way by the same judge model: each ground-truth variant is
compared against each of our findings (AuditOwl side) or each baseline's raw
predicted discrepancies (baseline side, pulled from SciCoQA's own released
generations) in turn with their `discrepancy_evaluation_v2` prompt, and an
item counts as detected if any pair matches. `judge_verdicts.json` and
`baselines_gpt55_judge.json` hold every per-pair verdict; `scicoqa_judge.py`
and `judge_baselines_gpt55.py` regenerate them. SciCoQA's own judge is
gpt-oss-20b; we run their protocol and prompt with GPT-5.5 (via `codex exec`)
on both sides, so AuditOwl (Claude-family) is not judged by a model from its
own family. gpt-5 and gpt-5-mini are still judged by the GPT-family GPT-5.5;
only gemini-2.5-pro's baseline score is judged by a different family.

The 10 items are the 10 most-recently-filed items in the SciCoQA real split
whose discrepancy postdates the baselines' training cutoffs (minimizing
memorization risk against the newer Opus 4.8 cutoff) AND for which SciCoQA's
own release records at least one baseline prediction. One item in the
cutoff-eligible pool (item 5) has zero recorded predictions for any baseline
model and is excluded on that basis; the next-most-recent eligible item
backfills it.

Run: python compute_recall.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    per = json.loads((HERE / "judge_verdicts.json").read_text())["per_item"]
    items = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)  # item 5 excluded: zero baseline
    # predictions exist for it in SciCoQA's own release, for any model

    for run in ("run_01", "run_02"):
        d = sum(1 for i in items
                if per[f"{run}|item{i:02d}"]["scicoqa_protocol"] == "DETECTED")
        print(f"AuditOwl {run}, n={len(items)}: {d}/{len(items)} = {100*d//len(items)}%")

    hh = json.loads((HERE / "baselines.json").read_text())
    for model in ("gpt-5", "gemini-2.5-pro", "gpt-5-mini"):
        missing = [it["item"] for it in hh["per_item"] if it[model] is None]
        if missing:
            raise SystemExit(f"{model} has no recorded prediction for {missing} -- "
                              "exclude that item from the selection instead of "
                              "silently scoring an unevaluated item as a miss")
        vals = [it[model] for it in hh["per_item"]]
        print(f"{model}, n={len(vals)}: {sum(vals)}/{len(vals)} = {100*sum(vals)//len(vals)}%")

    auditowl_cands = sum(
        len(set(p["finding"] for p in per[f"run_01|item{i:02d}"]["pairs"]))
        for i in items
    )
    print(f"AuditOwl candidates per item, run_01: {auditowl_cands}/{len(items)} = "
          f"{auditowl_cands/len(items):.1f}")

    baseline_judge = json.loads((HERE / "baselines_gpt55_judge.json").read_text())
    for model in ("gpt-5", "gpt-5-mini", "gemini-2.5-pro"):
        pairs = baseline_judge["per_baseline_per_item"][model]
        n_cands = sum(len(set(p["cand_idx"] for p in info["pairs"])) for info in pairs.values())
        print(f"{model} candidates per item: {n_cands}/{len(items)} = {n_cands/len(items):.1f}")


if __name__ == "__main__":
    main()
