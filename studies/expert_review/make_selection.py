#!/usr/bin/env python3
"""Select the expert-review sample: the forced paper set.

Selection rule:
  1. every eligible paper with >=1 high-severity bug/mismatch/methodology finding
  2. plus every paper with >=3 high-severity missing findings not already in set 1
  (robustness papers excluded).

Reads audits/*/findings_verified.json, writes _selection/selection_manifest.json,
which make_assignment.py consumes; build_rating_workbook.py consumes
make_assignment.py's output (_selection/assignment.json) one step further
downstream.

Note: this is a live tool over the current findings, not a frozen replay. Paper
1339 was re-audited after the human study had already run, so its finding count
grew (7 in-scope at review time -> 9 today); running this script now will not
bit-reproduce that one row of the committed manifest. Every other row does.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "_selection")

ROBUSTNESS_PAPERS = {"1333", "1829", "2371", "2578", "2657"}
IN_SCOPE = {
    ("high", "missing"), ("high", "difference"), ("high", "bug"), ("high", "methodology"),
    ("medium", "difference"), ("medium", "bug"), ("medium", "methodology"),
}


def load_papers():
    papers = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "audits/*/findings_verified.json"))):
        pdir = os.path.basename(os.path.dirname(f))
        pid = pdir.split("_")[0]
        if pid in ROBUSTNESS_PAPERS:
            continue
        kept = [it for it in json.load(open(f))["findings"]
                if it.get("verdict") in ("keep", "lowered")]
        papers[pid] = (pdir, kept)
    return papers


def select(papers):
    judgment, miss_heavy = set(), set()
    for pid, (_, kept) in papers.items():
        n_miss = 0
        for it in kept:
            if it["severity"] == "high":
                if it["category"] in ("bug", "difference", "methodology"):
                    judgment.add(pid)
                elif it["category"] == "missing":
                    n_miss += 1
        if n_miss >= 3:
            miss_heavy.add(pid)
    return judgment, miss_heavy - judgment


def main():
    os.makedirs(OUT, exist_ok=True)
    papers = load_papers()
    judgment, miss_heavy = select(papers)
    selected = sorted(judgment | miss_heavy)
    manifest = {}
    for pid in selected:
        pdir, kept = papers[pid]
        findings = [it for it in kept if (it["severity"], it["category"]) in IN_SCOPE]
        manifest[pid] = {
            "dir": pdir,
            "reason": "high bug/mismatch/methodology" if pid in judgment else ">=3 high missing",
            "n_inscope": len(findings),
        }
    with open(os.path.join(OUT, "selection_manifest.json"), "w") as fh:
        json.dump({"selected_papers": manifest,
                   "n_papers": len(selected),
                   "n_inscope_issues": sum(m["n_inscope"] for m in manifest.values())}, fh, indent=1)
    print(f"selected {len(selected)} papers, "
          f"{sum(m['n_inscope'] for m in manifest.values())} in-scope issues")


if __name__ == "__main__":
    main()
