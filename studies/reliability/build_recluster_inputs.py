#!/usr/bin/env python3
"""Rebuild the per-paper input files for the merge-by-defect re-clustering pass.

`data/merged_clusters.json` is the partition behind every "merged" reliability
number, and it was produced by an LLM pass (one agent per paper) whose `/tmp`
input files were not kept. Those inputs are fully determined by data that *is*
released: the ten runs' `findings.json` per paper, indexed as `r{run}#{idx}`.

This regenerates them, so the clustering can actually be re-run: take the
prompt in `recluster_prompt.md`, feed it these files, and compare the partition
you get against the committed one. The pass is an LLM judgement and will not
reproduce byte-identically; what this makes checkable is whether a different
run of the same procedure groups the findings the same way.

Run: python studies/reliability/build_recluster_inputs.py [-o OUTDIR]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECTION = HERE / "selection.json"
MERGED = HERE / "data" / "merged_clusters.json"


def build(paper: str) -> dict:
    pdir = HERE / paper
    runs = sorted(pdir.glob("run_*"))
    findings = []
    for rd in runs:
        fp = rd / "findings.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text())
        items = data.get("findings", data)
        run_no = int(rd.name.split("_")[1])
        for idx, f in enumerate(items):
            findings.append({
                "fid": f"r{run_no:02d}#{idx}",
                "run": rd.name,
                "category": f.get("category"),
                # the prompt calls this `loc`: where the finding says the defect is
                "loc": f"{f.get('file')}:{f.get('line_start')}-{f.get('line_end')}",
                "title": f.get("title"),
                "claim": f.get("claim"),
            })
    return {"paper": paper, "n_runs": len(runs), "findings": findings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="/tmp/recluster",
                    help="where to write <paper-number>.json (default: /tmp/recluster)")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    papers = [p["paper"] for p in json.loads(SELECTION.read_text())["selected"]]
    committed = {e["paper"]: e for e in json.loads(MERGED.read_text())} if MERGED.exists() else {}

    for paper in papers:
        payload = build(paper)
        num = paper.split("_", 1)[0]
        (out / f"{num}.json").write_text(json.dumps(payload, indent=1) + "\n")

        # every fid the committed partition references must exist in the
        # regenerated input, or the indices have drifted
        have = {f["fid"] for f in payload["findings"]}
        want = {f for c in committed.get(paper, {}).get("clusters", []) for f in c["fids"]}
        missing = sorted(want - have)
        status = "ok" if not missing else f"MISSING {len(missing)}: {missing[:4]}"
        print(f"  {num:<6} {payload['n_runs']:>2} runs, {len(payload['findings']):>3} findings  {status}")

    print(f"\nwrote {len(papers)} input files to {out}")
    print("re-run the clustering with the prompt in recluster_prompt.md against these.")


if __name__ == "__main__":
    main()
