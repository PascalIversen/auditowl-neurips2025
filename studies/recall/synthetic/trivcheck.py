#!/usr/bin/env python3
"""Triviality gate: a fresh Codex instance reviews each SEEDED
candidate tree WITHOUT the paper. Flags are recorded per seed as a descriptive
stratification (not automatic rejection — majors are intrinsically code-only-
suspicious). Writes
trivcheck/<pid>/ trees + flags + a per-paper report seeds/trivreport_<pid>.json.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from propose_major import read_text_raw, write_text_raw

HERE = Path(__file__).resolve().parent
WORK, SEEDS, TRIV, LOGS = HERE / "work", HERE / "seeds", HERE / "trivcheck", HERE / "logs"
IGNORE = {".git", "__pycache__", ".DS_Store"}
PAPERS = ["1023", "913", "2657", "2578", "2167", "1717", "1908", "4090", "1171", "3463"]
MODEL, EFFORT, TIMEOUT = "gpt-5.6-sol", "high", 2400
_lock = threading.Lock()

PROMPT = """\
You are reviewing a research code repository under `code/`. You do NOT have the
accompanying paper — judge from the code alone.

Task: list the most suspicious defects you can find purely from reading the code:
bugs, incorrect logic, evaluation-protocol problems, internal inconsistencies,
missing/absent components a repo of this kind would be expected to have, values that
contradict the code's own comments/docs. Up to 20 items, ranked by confidence. Do not
modify any file.

Write EXACTLY one file `trivcheck_flags.json` in THIS directory (not under code/):
{"flags": [{"file": "<path relative to code/>", "line": <int approx, or null if the
            flag is about an absent file/component>,
            "reason": "<one line>", "confidence": "high|medium|low"}]}
"""


def apply_candidates(pid: str, repo: str, seeds: list) -> tuple[Path, dict]:
    """Copy clean staged code -> trivcheck tree, apply every appliable candidate.
    Returns (tree_root, {sid: {'sites': [{'file','line'}, ...]} or {'skip': reason}})."""
    td = TRIV / pid
    if td.exists():
        for dp, _, fn in os.walk(td):
            for f in fn:
                try: (Path(dp) / f).chmod(0o644)
                except OSError: pass
        shutil.rmtree(td)
    td.mkdir(parents=True)
    shutil.copytree(WORK / pid / "code", td / "code",
                    ignore=lambda d, n: [x for x in n if x in IGNORE])
    for dp, _, fn in os.walk(td / "code"):
        for f in fn:
            (Path(dp) / f).chmod(0o644)
    where = {}
    for s in seeds:
        sid = s["sid"]
        hunks = (s.get("patch") or {}).get("hunks") or []
        sites, skip = [], None
        for h in hunks:
            fp = td / "code" / h.get("file", "")
            rel = h["file"].split("/", 1)[-1] if "/" in h.get("file", "") else h.get("file", "")
            if h.get("action") == "delete_file":
                if not fp.is_file():
                    skip = "file missing"; break
                fp.unlink()
                sites.append({"file": rel, "line": None})
                continue
            if not fp.is_file():
                skip = "file missing"; break
            t = read_text_raw(fp)
            old = h.get("old_string", "")
            if not old or t.count(old) != 1:
                skip = f"old_string count={t.count(old) if old else 0}"; break
            line = t[:t.index(old)].count("\n") + 1
            write_text_raw(fp, t.replace(old, h.get("new_string", "")))
            sites.append({"file": rel, "line": line})
        where[sid] = {"skip": skip} if skip else {"sites": sites}
    return td, where


def run_paper(pid: str) -> str:
    cf = SEEDS / f"candidates_{pid}.json"
    if not cf.exists():
        return f"{pid}: no candidates file, skipped"
    rep_f = SEEDS / f"trivreport_{pid}.json"
    if rep_f.exists():
        return f"{pid}: trivreport exists, skipped"
    data = json.loads(cf.read_text())
    repo = data.get("_repo", "")
    td, where = apply_candidates(pid, repo, data.get("seeds", []))
    log = LOGS / f"{pid}.triv.jsonl"
    cmd = ["codex", "exec", "-C", str(td), "-m", MODEL,
           "--sandbox", "workspace-write", "--skip-git-repo-check",
           "--color", "never", "--json",
           "-c", f'model_reasoning_effort="{EFFORT}"',
           "-o", str(log.with_suffix(".last.md")), "-"]
    with _lock:
        print(f"[{pid}] trivcheck codex starting", flush=True)
    with log.open("w") as lf:
        subprocess.run(cmd, input=PROMPT, stdout=lf, stderr=subprocess.STDOUT,
                       text=True, timeout=TIMEOUT)
    out = td / "trivcheck_flags.json"
    flags = []
    if out.is_file():
        try:
            flags = json.loads(out.read_text()).get("flags", [])
        except Exception:
            pass

    def normf(f):
        f = str(f or "").replace("\\", "/").lstrip("./")
        if f.startswith("code/"): f = f[5:]
        if "/" in f and "__" in f.split("/")[0]: f = f.split("/", 1)[1]
        return f

    report = {"paper": pid, "n_flags": len(flags), "flags": flags, "candidates": {}}
    for sid, w in where.items():
        if "skip" in w:
            report["candidates"][sid] = {"status": "not_applied", "why": w["skip"]}
            continue
        hit = [fl for fl in flags
               if any(normf(fl.get("file")) == normf(site["file"])
                      and (site["line"] is None or fl.get("line") is None
                           or abs(int(fl.get("line") or 0) - site["line"]) <= 15)
                      for site in w["sites"])]
        report["candidates"][sid] = {
            "status": "FLAGGED" if hit else "clean",
            "sites": w["sites"],
            "matched_flags": [f.get("reason") for f in hit][:3],
        }
    rep_f.write_text(json.dumps(report, indent=2) + "\n")
    n_fl = sum(1 for c in report["candidates"].values() if c["status"] == "FLAGGED")
    return f"{pid}: {len(flags)} no-paper flags, {n_fl} candidate(s) FLAGGED"


def main():
    only = sys.argv[1:] or PAPERS
    TRIV.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for res in ex.map(run_paper, only):
            with _lock:
                print(res, flush=True)


if __name__ == "__main__":
    main()
