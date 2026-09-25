#!/usr/bin/env python3
"""Pre-registration part 2: after author approval, apply the
approved selection to seeded trees, emit ANSWER_KEY_<pid>.json + SHA-256 manifest,
and commit. After this commit, no seed may change.

Writes seeded trees to code_seeded/<pid>/code/<repo>/ (note the intermediate "code/"
directory — unlike final_recall_study's flatter code_seeded/<pid>/<repo>/ layout,
this one matches the audits/<dir>/code/<repo>/ convention that
studies/reliability/build_sandboxes.py's build_paper() expects, so build_runs.py can reuse it
unmodified by pointing bs.AUDITS at code_seeded/ directly.

Usage: python3 freeze.py --approved-by-author   [--no-commit]
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from propose_major import resolve, tree_hashes, SEEDS, PAPERS, read_text_raw, write_text_raw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
IGNORE = {".git", "__pycache__", ".DS_Store"}
SLOT_ORDER = ["MAJOR-MISS", "MAJOR-DIFF", "MAJOR-METH", "MAJOR-BUG"]
INPUT_FILES = ["paper.pdf", "paper_text.txt", "metadata.txt", "code_links.txt"]


def apply_one(root: Path, patch: dict) -> list[dict]:
    """Apply every hunk of a seed's patch; return [{'file','line','action'}, ...] sites.
    `root` is the code/ directory (code_seeded/<pid>/code/) — hunks' `file` paths
    already start with "<repo>/", matching propose_major.py's own convention
    (its validate() resolves against WORK/pid/"code" the same way)."""
    sites = []
    for h in patch["hunks"]:
        fp = root / h["file"]
        if h.get("action") == "delete_file":
            assert fp.is_file(), f"delete_file target missing: {h['file']}"
            fp.unlink()
            sites.append({"file": h["file"], "line": None, "action": "delete_file"})
            continue
        t = read_text_raw(fp)
        n = t.count(h["old_string"])
        assert n == 1, f"{h['file']}: old_string count={n}"
        line = t[: t.index(h["old_string"])].count("\n") + 1
        write_text_raw(fp, t.replace(h["old_string"], h["new_string"]))
        sites.append({"file": h["file"], "line": line, "action": "replace"})
    return sites


def silence_check(root: Path, sites: list[dict]) -> None:
    """Light sanity check only — NOT a crash-safety gate (a seeded defect is not
    not required for this study). Just catches outright syntax errors introduced by
    a botched replace patch."""
    for s in sites:
        if s["action"] == "delete_file":
            continue
        fp = root / s["file"]
        if fp.suffix == ".py":
            r = subprocess.run([sys.executable, "-m", "py_compile", str(fp)],
                               capture_output=True)
            assert r.returncode == 0, f"py_compile fail: {fp}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved-by-author", action="store_true", required=False)
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("papers", nargs="*", help="only these pids (default: all 10)")
    args = ap.parse_args()
    if not args.approved_by_author:
        sys.exit("refusing: run only after author approval, with --approved-by-author")

    only = args.papers or PAPERS
    sel_all = json.loads((SEEDS / "selection_proposed.json").read_text())
    missing = [pid for pid in only if pid not in sel_all]
    if missing:
        sys.exit(f"no selection recorded for {missing} in selection_proposed.json")
    sel = sel_all
    # records which revision of the seeding protocol this freeze was made
    # against; empty when the protocol document is not part of the checkout
    design = HERE / "DESIGN.md"
    design_commit = subprocess.run(
        ["git", "log", "--oneline", "-1", "--", str(design)],
        capture_output=True, text=True, cwd=ROOT).stdout.strip() if design.exists() else ""
    manifest = {"design_commit": design_commit, "papers": {}}
    all_sites = 0
    for pid in only:
        src, repo = resolve(pid)
        cands = {s["sid"]: s for s in
                 json.loads((SEEDS / f"candidates_{pid}.json").read_text())["seeds"]}
        pid_root = HERE / "code_seeded" / pid
        pid_root.mkdir(parents=True, exist_ok=True)
        for fn in INPUT_FILES:
            s = src / fn
            if s.exists():
                (pid_root / fn).write_bytes(s.read_bytes())
        dst = pid_root / "code" / repo
        if dst.exists():
            for dp, _, fn in os.walk(dst):
                for f in fn:
                    try: (Path(dp) / f).chmod(0o644)
                    except OSError: pass
            shutil.rmtree(dst)
        shutil.copytree(src / "code" / repo, dst,
                        ignore=lambda d, n: [x for x in n if x in IGNORE])
        for dp, _, fn in os.walk(dst):
            for f in fn:
                (Path(dp) / f).chmod(0o644)
        key = {"paper": pid, "repo": repo,
               "design": "synthetic recall arm, four-category seeding brief", "seeds": []}
        changed_sites = []
        for slot in SLOT_ORDER:
            if slot not in sel[pid]:
                continue
            v = sel[pid][slot]
            s = cands[v["sid"]]
            sites = apply_one(pid_root / "code", s["patch"])
            changed_sites += sites
            all_sites += len(sites)
            key["seeds"].append({
                "id": v["sid"], "slot": slot,
                "category": s.get("category"),
                "sites": sites, "title": s.get("title"),
                "description": s.get("description"),
                "expected_reasoning": s.get("expected_reasoning"),
                "paper_anchor": s.get("paper_anchor"),
                "placement": s.get("placement"),
                "selection_rule": v.get("rule"),
                "no_paper_suspicion": v.get("no_paper_suspicion"),
            })
        silence_check(pid_root / "code", changed_sites)
        (HERE / f"ANSWER_KEY_{pid}.json").write_text(json.dumps(key, indent=2) + "\n")
        manifest["papers"][pid] = {"repo": repo, "n_seeds": len(key["seeds"]),
                                   "tree_sha256": hashlib.sha256(json.dumps(
                                       tree_hashes(dst), sort_keys=True).encode()).hexdigest()}
        print(f"{pid}: {len(key['seeds'])} seeds applied ({len(changed_sites)} sites), key written")
    (HERE / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest written; {all_sites} total edit sites")

    if not args.no_commit:
        files = [str(HERE / "manifest.sha256.json")] + \
                [str(HERE / f"ANSWER_KEY_{p}.json") for p in only] + \
                [str(SEEDS)]
        subprocess.run(["git", "add"] + files, cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m",
                        "Pre-register the synthetic recall arm answer keys + manifests "
                        "(author-approved; before any audit run)"],
                       cwd=ROOT, check=True)
        print(subprocess.run(["git", "log", "--oneline", "-1"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
