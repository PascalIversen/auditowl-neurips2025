#!/usr/bin/env python3
"""
build_sandboxes.py — materialise hermetic, leak-proof run sandboxes.

The whole experiment hinges on each repeated run being INDEPENDENT and BLIND to
every other run and to the original committed audit. This script enforces that
structurally (not just by prompt):

  studies/reliability/<paper>/
    code_frozen/                 one read-only snapshot of code/, shared by all
                                 runs -> guarantees byte-identical inputs
    run_01/ .. run_NN/           one sealed sandbox per run, each containing ONLY
        paper.pdf                the inputs an auditor is allowed to see — these
        paper_text.txt             MIRROR the original audit exactly (PDF is the
        metadata.txt               source of truth for quotes/equations; the .txt
        code_links.txt             is the grep-able extraction) + metadata
        code -> ../code_frozen     - the code (read-only symlink)
        auditowl/audit_prompt.md    - the protocol + schema + extractor, copied in
        findings-schema.md           so the agent never needs to leave the dir
        extract_findings.py
        AGENT_INSTRUCTIONS.md      - audit runner (with the sealed-sandbox guard)
        VERIFIER_INSTRUCTIONS.md   - verify runner

What is DELIBERATELY NOT copied into a sandbox (these are the answer key):
    audit.md, findings.json, findings_verified.json, _audit_code/, _verifier_code/

Also writes studies/reliability/sandboxes.json (the run manifest passed to the workflow).

Usage:
    python studies/reliability/build_sandboxes.py            # build all selected, 10 runs
    python studies/reliability/build_sandboxes.py --runs 10
    python studies/reliability/build_sandboxes.py --clean    # remove sandboxes (keep scripts)
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import stat
from pathlib import Path

def _find_root(start: Path) -> Path:
    """Repo root = the ancestor holding `audits/`. Found by content rather than a
    fixed number of `.parent` hops, so this survives the file moving depth."""
    for candidate in (start, *start.parents):
        if (candidate / "audits").is_dir():
            return candidate
    raise SystemExit(f"no `audits/` directory found above {start}")


ROOT = _find_root(Path(__file__).resolve().parent)
RB = Path(__file__).resolve().parent
AUDITS = ROOT / "audits"

INPUT_FILES = ["paper.pdf", "paper_text.txt", "metadata.txt", "code_links.txt"]
SHARED_FILES = {
    "auditowl/audit_prompt.md": ROOT / "auditowl/audit_prompt.md",
    "findings-schema.md": ROOT / "auditowl" / "findings_schema.md",
    "extract_findings.py": ROOT / "auditowl" / "extract_findings.py",
    "verifier-prompt.md": ROOT / "auditowl/verifier_prompt.md",
}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
               ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}

AGENT_INSTRUCTIONS = """# Audit runner — sealed sandbox

You are running inside a **sealed sandbox**: `{run}`.
Treat this directory as the entire world.

## Hard isolation rules (do not violate)
- Read ONLY files inside this directory: `paper.pdf`, `paper_text.txt`,
  `metadata.txt`, `code_links.txt`, the `code/` tree, `auditowl/audit_prompt.md`,
  `findings-schema.md`.
- Do NOT read, list, glob, or grep anything OUTSIDE this directory. In
  particular never touch `../`, any sibling `run_*/`, the `audits/` tree,
  `analysis/`, `studies/reliability/`, `_robustness_new/`, or any pre-existing
  `audit.md` / `findings.json`.
- Do NOT use the network, `git`, `gh`, or any web/search tool. Do NOT re-clone
  or fetch code. The `code/` tree is READ-ONLY.
- Write everything (audit.md, `_audit_code/`, `out/`, findings.json,
  token_cost.json) INSIDE this run directory only.

## The paper: PDF + text (use BOTH, exactly as `auditowl/audit_prompt.md` says)
- `paper.pdf` is the **source of truth** for reading, verifying, and quoting —
  only the PDF preserves equations, figures, tables, and layout. Read and quote
  from it; findings cite `file: paper.pdf`.
- `paper_text.txt` is a faithful plain-text extraction of that same PDF — use it
  to `grep`/search fast, then re-locate and confirm each quote in `paper.pdf`.
- Do NOT install PDF libraries or re-extract the PDF; the extraction is provided.

## Task
1. Audit this paper by following `auditowl/audit_prompt.md` **exactly**. The
   cloned repo is under `code/<owner>__<repo>/`; the paper is `paper.pdf` (with
   `paper_text.txt` as the searchable extraction); metadata in `metadata.txt`.
   Findings schema is `findings-schema.md`.
2. Write `audit.md` with the YAML finding blocks.
3. Extract the sidecar:
   `python extract_findings.py audit.md --out findings.json`
4. Stop. Do not verify your own findings (a separate verifier pass does that).
"""

VERIFIER_INSTRUCTIONS = """# Verify runner — sealed sandbox

You are running inside a **sealed sandbox**: `{run}`.
A prior pass in THIS directory produced `audit.md` + `findings.json`. Verify them.

## Hard isolation rules (do not violate)
- Read ONLY files inside this directory.
- Do NOT read any sibling `run_*/`, the `audits/` tree, `analysis/`, or any
  other paper's files. Do NOT use the network, `git`, `gh`, or web/search.
- `code/` is READ-ONLY. Write only `findings_verified.json` and `_verifier_code/`.

## Task
Verify the audit by following `verifier-prompt.md` **exactly**: re-open each
finding's cited `file:line` in `code/<owner>__<repo>/`, compare against the
quote, and assign `verdict` (keep / lower / reject / cannot-verify) with a
`reason` and `changed`. A finding's `file` is relative to the repo root inside
`code/`; for paper-side claims (`file: paper.pdf`) re-locate the quote in
`paper.pdf` (use `paper_text.txt` to search). Budget ~3-5 tool calls per
finding; cited-location-only. Write
`findings_verified.json` once at the end (original findings + the new fields).
"""


def _ignore(_dir, names):
    return [n for n in names if n in IGNORE_DIRS]


def freeze_code(src_code: Path, dst_frozen: Path):
    if dst_frozen.exists():
        return
    shutil.copytree(src_code, dst_frozen, ignore=_ignore, symlinks=False,
                    ignore_dangling_symlinks=True)
    # make the whole snapshot read-only so no run can mutate the shared inputs
    for dirpath, dirnames, filenames in os.walk(dst_frozen):
        for f in filenames:
            fp = Path(dirpath) / f
            try:
                fp.chmod(fp.stat().st_mode & ~0o222)
            except OSError:
                pass


def build_paper(paper: str, n_runs: int) -> dict:
    src = AUDITS / paper
    code = src / "code"
    if not code.is_dir():
        raise SystemExit(f"{paper}: no code/ to freeze")
    pdir = RB / paper
    pdir.mkdir(parents=True, exist_ok=True)
    frozen = pdir / "code_frozen"
    freeze_code(code, frozen)

    runs = []
    for i in range(1, n_runs + 1):
        rd = pdir / f"run_{i:02d}"
        rd.mkdir(exist_ok=True)
        for fn in INPUT_FILES:
            s = src / fn
            if s.exists():
                shutil.copy2(s, rd / fn)
        for name, path in SHARED_FILES.items():
            if path.exists():
                shutil.copy2(path, rd / name)
        link = rd / "code"
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
        if not link.exists():
            os.symlink(frozen.resolve(), link)
        (rd / "AGENT_INSTRUCTIONS.md").write_text(AGENT_INSTRUCTIONS.format(run=str(rd)))
        (rd / "VERIFIER_INSTRUCTIONS.md").write_text(VERIFIER_INSTRUCTIONS.format(run=str(rd)))
        runs.append(str(rd.relative_to(ROOT)))
    return {"paper": paper, "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--paper", help="only build/clean papers whose name contains this substring")
    args = ap.parse_args()

    sel = json.loads((RB / "selection.json").read_text())
    papers = [r["paper"] for r in sel["selected"]]
    if args.paper:
        papers = [p for p in papers if args.paper in p]
        if not papers:
            raise SystemExit(f"--paper {args.paper!r} matched none of {[r['paper'] for r in sel['selected']]}")

    if args.clean:
        for p in papers:
            d = RB / p
            if d.exists():
                # restore write bits so rmtree can delete the frozen snapshot
                for dp, dn, fn in os.walk(d):
                    for f in fn:
                        try:
                            (Path(dp) / f).chmod(0o644)
                        except OSError:
                            pass
                shutil.rmtree(d)
                print(f"removed {d.relative_to(ROOT)}")
        return

    manifest = {"runs_per_paper": args.runs, "papers": []}
    for p in papers:
        info = build_paper(p, args.runs)
        manifest["papers"].append(info)
        print(f"built {p}: {len(info['runs'])} sealed run dirs")
    (RB / "sandboxes.json").write_text(json.dumps(manifest, indent=2))
    n = sum(len(x["runs"]) for x in manifest["papers"])
    print(f"\n{n} sandboxes -> {(RB/'sandboxes.json').relative_to(ROOT)}")
    print("each run dir excludes audit.md/findings*.json/_audit_code/_verifier_code (the answer key)")


if __name__ == "__main__":
    main()
