#!/usr/bin/env python3
"""Seeded reviewer assignment for the expert-review study.

Staffing: 2 heavy reviewers (10 papers each) + 5 light reviewers (2 papers each).
Rules:
  - 20 distinct papers are covered. Papers carrying the rare judgment cells
    (high-severity mismatch or methodology) are always covered; the remaining
    slots are a seeded random draw from the other selected papers.
  - Heavy reviewers split the 20 papers, balanced by issue count.
  - The 10-paper overlap = each light reviewer re-rates 1 paper from each heavy
    reviewer's set, so overlap pairs span all 7 reviewers.

Reads _selection/selection_manifest.json (from make_selection.py), writes
_selection/assignment.json. Reviewer slots (Heavy-1, ..., Light-5) are
placeholders standing in for the real, anonymised reviewer identities.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEL = os.path.join(HERE, "_selection")

SEED = 20260721
N_COVERED = 20
rng = np.random.default_rng(SEED)


def load():
    manifest = json.load(open(os.path.join(SEL, "selection_manifest.json")))["selected_papers"]
    rare = {}
    for pid, info in manifest.items():
        cats = set()
        fv = os.path.join(ROOT, "audits", info["dir"], "findings_verified.json")
        for it in json.load(open(fv))["findings"]:
            if it.get("verdict") in ("keep", "lowered") and it["severity"] == "high" \
               and it["category"] in ("difference", "methodology"):
                cats.add(it["category"])
        rare[pid] = cats
    return manifest, rare


def main():
    manifest, rare = load()
    pids = sorted(manifest)
    # 4563: flagship paper (most in-scope issues); codebook examples were genericized
    # so it is safe to rate.
    must = sorted({p for p in pids if rare[p]} | {"4563"})
    rest = [p for p in pids if p not in must]
    fill = list(rng.choice(rest, size=N_COVERED - len(must), replace=False))
    covered = must + fill
    uncovered = sorted(set(pids) - set(covered))

    # split covered papers over the two heavy reviewers, balancing issue counts
    counts = {p: manifest[p]["n_inscope"] for p in pids}
    order = sorted(covered, key=lambda p: -counts[p])
    heavy = {"Heavy-1": [], "Heavy-2": []}
    for p in order:
        tgt = min(heavy, key=lambda r: (sum(counts[q] for q in heavy[r]), len(heavy[r])))
        heavy[tgt].append(p)

    # overlap: 5 papers from each heavy set, one per light reviewer
    ov1 = list(rng.choice(heavy["Heavy-1"], size=5, replace=False))
    ov2 = list(rng.choice(heavy["Heavy-2"], size=5, replace=False))
    lights = {f"Light-{i+1}": [ov1[i], ov2[i]] for i in range(5)}
    overlap = set(ov1) | set(ov2)

    with open(os.path.join(SEL, "assignment.json"), "w") as fh:
        json.dump({"seed": SEED, "covered": covered, "uncovered": uncovered,
                   "heavy": heavy, "light": lights,
                   "overlap": sorted(overlap)}, fh, indent=1)
    print(f"covered {len(covered)} papers ({sum(counts[p] for p in covered)} issues), "
          f"overlap {len(overlap)}")


if __name__ == "__main__":
    main()
