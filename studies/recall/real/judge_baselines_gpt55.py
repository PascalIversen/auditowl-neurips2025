#!/usr/bin/env python3
"""Score the 3 baselines' (gpt-5, gpt-5-mini, gemini-2.5-pro) side of the
SciCoQA real-arm head-to-head with GPT-5.5 via `codex exec` -- the same
judge used for AuditOwl's own side in scicoqa_judge.py, so AuditOwl
(Claude-family) is judged by a model outside its own family. gpt-5 and
gpt-5-mini are still judged by the GPT-family GPT-5.5; only gemini-2.5-pro
is judged by a model from a different family than itself.

Raw candidate predictions come from SciCoQA's own released archive
(out/inference_discrepancy_detection_real.tar.gz, real/full/*/eval_v2/
generations.jsonl "generation" field; see _scicoqa_gpt55_common.py), matched
to our 10 items by discrepancy_id / row_id, cross-verified against paper_url.
Item 5 (row_id a9deb3ea, "05_valentyn1boreiko-llm-threat-model") has zero
candidates for all 3 baselines in the archive and is excluded from the item
set entirely (see scicoqa_judge.py); item 11 backfills it.

Each candidate prediction is judged against BOTH reference variants
(discrepancy_description_gemini / discrepancy_description_gpt), same as
AuditOwl's side. An item counts as "detected" for a baseline if ANY
(reference, candidate) pair matches.

Requires the gitignored `_scicoqa_raw_cache/` (SciCoQA's own archive,
re-download and extract out/inference_discrepancy_detection_real.tar.gz from
github.com/UKPLab/scicoqa if absent).

Checkpoints incrementally to baselines_gpt55_judge.json after every
completed pair.

Run: python judge_baselines_gpt55.py
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _scicoqa_gpt55_common import (
    MODEL, judge_codex, rowmap_1_to_10, refs_for_item,
    get_baseline_candidates, BASELINE_MODEL_DIRS,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "baselines_gpt55_judge.json"
ITEMS = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)  # see scicoqa_judge.py: item 5 has no
# baseline predictions in SciCoQA's own release, backfilled by item 11
BASELINES = ("gpt-5", "gpt-5-mini", "gemini-2.5-pro")

_lock = threading.Lock()


def build_jobs():
    rowmap = rowmap_1_to_10()
    jobs = []
    not_predicted = {}  # (baseline, item) -> True if zero candidates
    for i in ITEMS:
        row_id = rowmap[i]
        refs = refs_for_item(row_id)
        for baseline in BASELINES:
            cands = get_baseline_candidates(baseline, row_id)
            if not cands:
                not_predicted[(baseline, i)] = True
                continue
            for src, ref in refs:
                for idx, cand in enumerate(cands):
                    key = f"{baseline}|item{i:02d}|{src}|{idx}"
                    jobs.append({
                        "key": key, "baseline": baseline, "item": i, "src": src,
                        "ref": ref, "idx": idx, "cand": cand,
                    })
    return jobs, not_predicted


def load_checkpoint():
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text())
            return data.get("_pairs_by_key", {})
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def derive_per_item(pairs_by_key: dict, not_predicted: dict):
    detail = {}
    summary = {}
    for baseline in BASELINES:
        n_detected = 0
        items_detail = {}
        for i in ITEMS:
            if (baseline, i) in not_predicted:
                items_detail[f"item{i:02d}"] = {"status": "NOT_PREDICTED (0 candidates in archive)",
                                                 "pairs": []}
                continue
            rows = [v for k, v in pairs_by_key.items()
                    if v["baseline"] == baseline and v["item"] == i]
            rows_sorted = sorted(rows, key=lambda r: (r["src"], r["idx"]))
            det = any(r["match"] == "yes" for r in rows_sorted)
            if det:
                n_detected += 1
            pairs_out = [{"gt_source": r["src"], "cand_idx": r["idx"], "match": r["match"]}
                         for r in rows_sorted]
            items_detail[f"item{i:02d}"] = {
                "status": "DETECTED" if det else "MISSED", "pairs": pairs_out}
        detail[baseline] = items_detail
        summary[baseline] = {"detected": n_detected, "n": len(list(ITEMS))}
    return detail, summary


def save(pairs_by_key: dict, not_predicted: dict, total_jobs: int):
    detail, summary = derive_per_item(pairs_by_key, not_predicted)
    payload = {
        "protocol": ("Raw baseline candidate predictions from SciCoQA's own "
                     "out/inference_discrepancy_detection_real.tar.gz "
                     "(real/full/*/eval_v2/generations.jsonl 'generation' field), "
                     "judged against BOTH discrepancy_description_gemini and "
                     "discrepancy_description_gpt reference variants with the "
                     "verbatim discrepancy_evaluation_v2 prompt; item detected "
                     "if any (reference, candidate) pair matches."),
        "judge_model_used": f"{MODEL} (via codex exec)",
        "baseline_archive_dirs": BASELINE_MODEL_DIRS,
        "scope": "items 1-10, 3 baselines (gpt-5, gpt-5-mini, gemini-2.5-pro)",
        "pairs_judged": len(pairs_by_key),
        "not_predicted": [f"{b}|item{i:02d}" for (b, i) in sorted(not_predicted.keys())],
        "summary": summary,
        "per_baseline_per_item": detail,
        "_pairs_by_key": pairs_by_key,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    tmp.replace(OUT)


def main():
    jobs, not_predicted = build_jobs()
    pairs_by_key = load_checkpoint()
    done_keys = {k for k, v in pairs_by_key.items() if v.get("match") in ("yes", "no")}
    todo = [j for j in jobs if j["key"] not in done_keys]

    print(f"{len(jobs)} total pairwise judgements planned on {MODEL}", flush=True)
    print(f"not-predicted (baseline,item) pairs: {sorted(not_predicted.keys())}", flush=True)
    print(f"{len(done_keys)} already completed (resumed from checkpoint)", flush=True)
    print(f"{len(todo)} remaining to run\n", flush=True)

    if not todo:
        save(pairs_by_key, not_predicted, len(jobs))
        print("Nothing to do; checkpoint already complete.", flush=True)
        return

    completed_this_run = [0]

    def work(job):
        v, raw = judge_codex(job["ref"], job["cand"], tag=job["key"].replace("|", "_")[:60])
        return job, v

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for fut in as_completed(futs):
            job, v = fut.result()
            with _lock:
                pairs_by_key[job["key"]] = {
                    "baseline": job["baseline"], "item": job["item"], "src": job["src"],
                    "idx": job["idx"], "match": v,
                }
                save(pairs_by_key, not_predicted, len(jobs))
                completed_this_run[0] += 1
                n = completed_this_run[0]
            if v == "yes":
                print(f"  MATCH {job['baseline']} item{job['item']:02d}", flush=True)
            if n % 10 == 0:
                print(f"  ... {n}/{len(todo)} this run "
                      f"({len(pairs_by_key)}/{len(jobs)} overall)", flush=True)

    print("\n=== summary ===", flush=True)
    _, summary = derive_per_item(pairs_by_key, not_predicted)
    for b in BASELINES:
        print(f"  {b}: {summary[b]['detected']}/{summary[b]['n']}", flush=True)

    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
