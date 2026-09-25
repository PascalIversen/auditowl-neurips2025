#!/usr/bin/env python3
"""
select_papers.py — deterministically draw the robustness-experiment sample.

We re-run the full audit->verify pipeline N times on a handful of papers to
measure the *test-retest reliability* of the auditor. To make that test
meaningful the auditor must have had a real codebase to reason about, so the
eligible frame is "papers for which we retrieved a non-trivial codebase".

Eligibility (a paper folder under audits/, excluding audits/theory/):
  - has a code/ subdirectory,
  - code/ holds >= MIN_SRC real source files (so the audit saw actual code),
  - has findings.json (it was audited -> we have a reference run).

Draw: uniform random, fixed seed, recorded in draw order -> selection.json.
This mirrors the project's seed=42 sampling ethos.

Run:
    python studies/reliability/select_papers.py            # draw + write selection.json
    python studies/reliability/select_papers.py --list     # just print the eligible frame
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

def _find_root(start: Path) -> Path:
    """Walk up to the repo root -- the ancestor holding `audits/`.

    Located by content rather than a fixed number of `.parent` hops, so this
    keeps working if the file moves to a different depth.
    """
    for candidate in (start, *start.parents):
        if (candidate / "audits").is_dir():
            return candidate
    raise SystemExit(f"no `audits/` directory found above {start}")


ROOT = _find_root(Path(__file__).resolve().parent)
AUDITS = ROOT / "audits"
OUT = Path(__file__).resolve().parent / "selection.json"

SEED = 42
K = 5
MIN_SRC = 5

SRC_EXT = {
    ".py", ".ipynb", ".sh", ".cpp", ".cc", ".cu", ".cuh", ".c", ".h", ".hpp",
    ".java", ".m", ".jl", ".r", ".js", ".ts", ".go", ".rs", ".scala", ".lua",
}


def count_src(code_dir: Path) -> int:
    n = 0
    for dirpath, dirnames, filenames in os.walk(code_dir):
        # skip obvious vendored/cache junk so the count reflects authored code
        dirnames[:] = [d for d in dirnames if d not in {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "site-packages", ".ipynb_checkpoints",
        }]
        for f in filenames:
            if Path(f).suffix.lower() in SRC_EXT:
                n += 1
    return n


def dir_kb(p: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(p):
        for f in filenames:
            fp = Path(dirpath) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total // 1024


def finding_stats(findings_json: Path) -> dict:
    try:
        data = json.loads(findings_json.read_text())
    except Exception:
        return {"n_findings": 0, "n_high": 0, "categories": {}}
    fs = data.get("findings", data if isinstance(data, list) else [])
    cats: dict[str, int] = {}
    n_high = 0
    for f in fs:
        cats[f.get("category", "?")] = cats.get(f.get("category", "?"), 0) + 1
        if f.get("severity") == "high":
            n_high += 1
    return {"n_findings": len(fs), "n_high": n_high, "categories": cats}


# The audited repositories are git-ignored, so a public checkout has no `code/`
# trees to count. The derived counts committed in the fingerprint snapshot let
# this script run without them, but the snapshot was captured after the
# original 78-paper draw, so re-running this does not reproduce that exact
# frame or the five-paper sample drawn from it.
_FP_PATH = ROOT / "analysis" / "data" / "repo_fingerprints.json"
_FINGERPRINTS = json.loads(_FP_PATH.read_text()) if _FP_PATH.exists() else {}


def eligible_frame() -> list[dict]:
    rows = []
    for d in sorted(AUDITS.iterdir()):
        if not d.is_dir() or d.name == "theory":
            continue
        code = d / "code"
        findings = d / "findings.json"
        if not findings.exists():
            continue
        if code.is_dir():
            nsrc = count_src(code)
        else:
            nsrc = _FINGERPRINTS.get(d.name, {}).get("n_code_files", 0)
        if nsrc < MIN_SRC:
            continue
        row = {
            "paper": d.name,
            "src_files": nsrc,
            "code_kb": dir_kb(code) if code.is_dir() else None,
            "has_verified": (d / "findings_verified.json").exists(),
            **finding_stats(findings),
        }
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print eligible frame and exit")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("-k", type=int, default=K)
    ap.add_argument("--force", action="store_true",
                    help="re-draw even if the frame has changed since selection.json")
    args = ap.parse_args()

    frame = eligible_frame()
    frame_ids = sorted(r["paper"] for r in frame)
    print(f"eligible frame: {len(frame)} papers with retrieved codebase "
          f"(>= {MIN_SRC} source files + findings.json)")

    if args.list:
        for r in sorted(frame, key=lambda x: -x["src_files"]):
            kb = f"{r['code_kb']:>9d}KB" if r["code_kb"] is not None else "     n/a "
            print(f"  {r['src_files']:5d} src  {kb}  "
                  f"{r['n_findings']:2d} find ({r['n_high']} high)  {r['paper']}")
        return

    picked_ids = random.Random(args.seed).sample(frame_ids, args.k)  # draw order
    by_id = {r["paper"]: r for r in frame}
    picked = [by_id[p] for p in picked_ids]

    # The draw is only reproducible against the frame it was made from. Repos
    # cloned after the fact (e.g. code located on a second pass) enlarge the
    # frame, so re-running would silently produce a different sample and
    # invalidate the reruns already collected under it.
    if OUT.exists() and not args.force:
        prev = json.loads(OUT.read_text())
        if prev.get("frame_size") != len(frame):
            raise SystemExit(
                f"{OUT.name} was drawn from a frame of {prev['frame_size']} papers; "
                f"this tree now yields {len(frame)}. Re-drawing would replace the "
                "recorded sample the reruns were collected under. Pass --force only "
                "if you intend to start a new reliability experiment."
            )

    sel = {
        "seed": args.seed,
        "k": args.k,
        "min_src_files": MIN_SRC,
        "frame_size": len(frame),
        "selected": picked,
    }
    OUT.write_text(json.dumps(sel, indent=2))
    print(f"\ndrew {args.k} papers (seed={args.seed}) -> {OUT.relative_to(ROOT)}")
    for r in picked:
        print(f"  - {r['paper']}")
        print(f"      {r['src_files']} src files, {r['code_kb']}KB, "
              f"{r['n_findings']} findings ({r['n_high']} high), "
              f"verified={r['has_verified']}, cats={r['categories']}")


if __name__ == "__main__":
    main()
