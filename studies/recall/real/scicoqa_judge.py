#!/usr/bin/env python3
"""Score the SciCoQA real arm end-to-end under SciCoQA's OWN judging protocol.

Protocol, replicated from scicoqa/inference/discrepancy_eval.py:
  * each `discrepancy_description_*` GT variant is its own reference row;
  * the prediction list is exploded to one predicted discrepancy per row;
  * every (reference, predicted) pair is judged independently with the verbatim
    `discrepancy_evaluation_v2` prompt;
  * an item is DETECTED if ANY pair answers match: yes.

Judge model is GPT-5.5 (via `codex exec`), used as a neutral third-party judge
for both AuditOwl's own side (this script) and the baselines' side
(judge_baselines_gpt55.py) -- SciCoQA's own judge, gpt-oss-20b, needs a local
runtime we don't have; the protocol is theirs, the judge model is not, and
critically the SAME judge model scores both sides of the comparison.

Checkpoints incrementally to judge_verdicts.json after every completed pair, so a
crash/interrupt loses at most one in-flight pair and a re-run resumes from
disk instead of restarting from zero.

Run: python scicoqa_judge.py
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _scicoqa_gpt55_common import (
    MODEL, judge_codex, render, rowmap_1_to_10, run_dirs,
    refs_for_item, kept_findings,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "judge_verdicts.json"
ITEMS = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)  # 10 most-recently-filed items WITH
# baseline predictions available; item 5 has zero recorded predictions for any
# baseline in SciCoQA's own release (not just ours) and is excluded on that
# basis, backfilled by item 11 (the next-most-recent item with real
# predictions) rather than reported as n=9
RUNS = ("run_01", "run_02")

_lock = threading.Lock()


def build_jobs():
    rowmap = rowmap_1_to_10()
    dirs = run_dirs()
    jobs = []
    for run in RUNS:
        for i in ITEMS:
            row_id = rowmap[i]
            refs = refs_for_item(row_id)
            kept = kept_findings(dirs[i], run)
            for src, ref in refs:
                for idx, f in enumerate(kept):
                    key = f"{run}|item{i:02d}|{src}|{idx}|{f.get('id','?').split('/')[-1]}"
                    jobs.append({
                        "key": key, "run": run, "item": i, "src": src, "ref": ref,
                        "finding": f.get("id", "?").split("/")[-1],
                        "file": f.get("file"),
                        "pred_rendered": render(f),
                    })
    return jobs


def load_checkpoint():
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text())
            return data.get("_pairs_by_key", {})
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def derive_per_item(pairs_by_key: dict):
    detail = {}
    for run in RUNS:
        for i in ITEMS:
            rows = [v for k, v in pairs_by_key.items()
                    if v["run"] == run and v["item"] == i]
            rows_sorted = sorted(rows, key=lambda r: (r["src"], r["idx"]))
            det = any(r["match"] == "yes" for r in rows_sorted)
            pairs_out = [{"gt_source": r["src"], "finding": r["finding"],
                          "file": r["file"], "match": r["match"]} for r in rows_sorted]
            detail[f"{run}|item{i:02d}"] = {
                "scicoqa_protocol": "DETECTED" if det else "MISSED",
                "pairs": pairs_out,
            }
    return detail


def save(pairs_by_key: dict, total_jobs: int):
    detail = derive_per_item(pairs_by_key)
    payload = {
        "protocol": "SciCoQA discrepancy_eval.py; prompt discrepancy_evaluation_v2 verbatim",
        "judge_model_used": f"{MODEL} (via codex exec)",
        "judge_model_in_scicoqa": "gpt-oss-20b (not runnable locally)",
        "scope": "items 1,2,3,4,6,7,8,9,10,11 (item 5 excluded, no baseline predictions exist for it), both runs",
        "pairs_judged": len(pairs_by_key),
        "per_item": detail,
        "_pairs_by_key": pairs_by_key,  # internal checkpoint state; harmless extra field
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    tmp.replace(OUT)


def main():
    jobs = build_jobs()
    pairs_by_key = load_checkpoint()
    done_keys = {k for k, v in pairs_by_key.items() if v.get("match") in ("yes", "no")}
    todo = [j for j in jobs if j["key"] not in done_keys]

    print(f"{len(jobs)} total pairwise judgements planned on {MODEL}", flush=True)
    print(f"{len(done_keys)} already completed (resumed from checkpoint)", flush=True)
    print(f"{len(todo)} remaining to run\n", flush=True)

    if not todo:
        save(pairs_by_key, len(jobs))
        print("Nothing to do; checkpoint already complete.", flush=True)
        return

    completed_this_run = [0]

    def work(job):
        v, raw = judge_codex(job["ref"], job["pred_rendered"], tag=job["key"].replace("|", "_")[:60])
        return job, v

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for fut in as_completed(futs):
            job, v = fut.result()
            with _lock:
                pairs_by_key[job["key"]] = {
                    "run": job["run"], "item": job["item"], "src": job["src"],
                    "idx": job["key"].split("|")[3], "finding": job["finding"],
                    "file": job["file"], "match": v,
                }
                save(pairs_by_key, len(jobs))  # checkpoint after EVERY completed pair
                completed_this_run[0] += 1
                n = completed_this_run[0]
            if v == "yes":
                print(f"  MATCH {job['run']} item{job['item']:02d} <- {job['finding']}", flush=True)
            if n % 10 == 0:
                print(f"  ... {n}/{len(todo)} this run "
                      f"({len(pairs_by_key)}/{len(jobs)} overall)", flush=True)

    print("\n=== per-item ===", flush=True)
    detail = derive_per_item(pairs_by_key)
    for run in RUNS:
        for i in ITEMS:
            print(f"  {run} item{i:02d}  {detail[f'{run}|item{i:02d}']['scicoqa_protocol']}", flush=True)

    print("\n=== headline ===", flush=True)
    for run in RUNS:
        t = sum(1 for i in ITEMS if detail[f"{run}|item{i:02d}"]["scicoqa_protocol"] == "DETECTED")
        print(f"  {run}: {t}/10", flush=True)

    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
