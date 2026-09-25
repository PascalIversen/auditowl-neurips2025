#!/usr/bin/env python3
"""Batch 2: the 10 papers left over from the original 30-paper selection that
Heavy/Light reviewers didn't cover, plus 10 more via a seeded random draw --
together with batch 1's 30, this is where Section 3.2's "population of 40
papers" comes from (30 + 20 batch-2 papers, 10 of which overlap with batch 1's
own leftovers, = 40 distinct papers).

Selection rule (seed 20260723):
  a) the papers of the original 30-paper selection not covered in batch 1's
     Heavy/Light assignment (_selection/assignment.json's "uncovered" list)
  b) + 10 seeded-random papers from the remaining eligible pool, restricted to
     papers with >= 2 in-scope issues (familiarization amortizes), excluding
     robustness papers, batch-1's covered set, and the original 30-paper
     selection.

Writes _selection/batch2_manifest.json (papers, in-scope issue counts, seed,
the leftover/random-draw split).
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEL = os.path.join(HERE, "_selection")

SEED = 20260723
ROB = {"1333", "1829", "2371", "2578", "2657"}
IN_SCOPE = {
    ("high", "missing"), ("high", "difference"), ("high", "bug"), ("high", "methodology"),
    ("medium", "difference"), ("medium", "bug"), ("medium", "methodology"),
}


def load_all():
    papers = {}
    for f in glob.glob(os.path.join(ROOT, "audits/*/findings_verified.json")):
        pdir = os.path.basename(os.path.dirname(f))
        pid = pdir.split("_")[0]
        kept = [it for it in json.load(open(f))["findings"]
                if it.get("verdict") in ("keep", "lowered")]
        items = [it for it in kept if (it["severity"], it["category"]) in IN_SCOPE]
        papers[pid] = {"dir": pdir, "n_inscope": len(items)}
    return papers


def select(papers):
    a = json.load(open(os.path.join(SEL, "assignment.json")))
    batch1 = set(a["covered"])
    sel = json.load(open(os.path.join(SEL, "selection_manifest.json")))["selected_papers"]
    leftover = sorted([p for p in sel if p not in batch1], key=int)
    pool = sorted([p for p, v in papers.items()
                   if p not in ROB and p not in batch1 and p not in sel
                   and v["n_inscope"] >= 2], key=int)
    rng = np.random.default_rng(SEED)
    drawn = sorted(rng.choice(pool, size=10, replace=False), key=int)
    return leftover, list(drawn)


def main():
    os.makedirs(SEL, exist_ok=True)
    papers = load_all()
    leftover, drawn = select(papers)
    batch = sorted(leftover + drawn, key=int)
    manifest = {
        "seed": SEED,
        "rule": "10 leftover from batch 1's uncovered set + 10 seeded-random (>=2 in-scope issues)",
        "leftover_from_batch1": leftover,
        "random_draw": drawn,
        "papers": {p: {"dir": papers[p]["dir"], "n_inscope": papers[p]["n_inscope"]} for p in batch},
        "n_issues": sum(papers[p]["n_inscope"] for p in batch),
    }
    with open(os.path.join(SEL, "batch2_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"batch2: {len(batch)} papers, {manifest['n_issues']} in-scope issues")
    print("leftover from batch 1:", leftover)
    print("random draw:", drawn)


if __name__ == "__main__":
    main()
