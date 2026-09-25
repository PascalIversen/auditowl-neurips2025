#!/usr/bin/env python3
"""Materialise sealed blind run sandboxes for the SEEDED trees.

Reuses studies/reliability/build_sandboxes.py verbatim (imported, not copied): same frozen
read-only code snapshot, same sealed run_NN/ layout, same AGENT/VERIFIER
instructions, same answer-key exclusions -- but pointed at
the synthetic recall arm's code_seeded/ as the source (AUDITS) and
the synthetic recall arm's runs/ as the destination (RB). code_seeded/<pid>/code/<repo>/
matches the audits/<dir>/code/<repo>/ convention build_paper() expects, so no
adaptation of build_sandboxes.py itself is needed.

Refuses to build unless ANSWER_KEY_<pid>.json + manifest.sha256.json exist for every
paper (pre-registration first!).

Usage:
    python studies/recall/synthetic/build_runs.py [--runs 3] [--clean]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
PAPERS = ["1023", "913", "2657", "2578", "2167", "1717", "1908", "4090", "1171", "3463"]
CODE_SEEDED = HERE / "code_seeded"
RUNS = HERE / "runs"


def load_bs():
    spec = importlib.util.spec_from_file_location(
        "bs", ROOT / "studies" / "reliability" / "build_sandboxes.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    bs.AUDITS = CODE_SEEDED
    bs.RB = RUNS
    return bs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean:
        if RUNS.exists():
            for dp, _dn, fn in os.walk(RUNS):
                for f in fn:
                    try:
                        (Path(dp) / f).chmod(0o644)
                    except OSError:
                        pass
            shutil.rmtree(RUNS)
            print("removed runs/")
        return

    missing = [pid for pid in PAPERS
               if not (HERE / f"ANSWER_KEY_{pid}.json").exists()]
    if missing or not (HERE / "manifest.sha256.json").exists():
        sys.exit(f"answer keys not pre-registered for {missing or '(manifest missing)'} "
                 f"-- run freeze.py --approved-by-author first")

    bs = load_bs()
    RUNS.mkdir(exist_ok=True)
    manifest = {"runs_per_paper": args.runs, "papers": []}
    for pid in PAPERS:
        info = bs.build_paper(pid, args.runs)
        manifest["papers"].append(info)
        print(f"built {pid}: {len(info['runs'])} sealed run dirs")
    (RUNS / "sandboxes.json").write_text(json.dumps(manifest, indent=2))
    n = sum(len(x["runs"]) for x in manifest["papers"])
    print(f"\n{n} sealed sandboxes -> runs/sandboxes.json")


if __name__ == "__main__":
    main()
