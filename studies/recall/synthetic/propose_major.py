#!/usr/bin/env python3
"""MAJOR-only recall study — cross-family seed proposal for 10 papers.

Per paper: stage work/<pid>/ (paper + read-only code), run Codex (gpt-5.6-sol) with
the pre-declared 4-category brief, hash-verify the staged code is
untouched, mechanically validate, write seeds/candidates_<pid>.json.
Runs 3 papers concurrently. Writes logs/ per paper. This script
implements §3 steps 1-2 only (triviality gate, collision check and approval are
separate scripts).
"""
from __future__ import annotations
import glob, hashlib, json, os, shutil, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
AUDITS = ROOT / "audits"
WORK, SEEDS, LOGS = HERE / "work", HERE / "seeds", HERE / "logs"
IGNORE = {".git", "__pycache__", ".DS_Store", ".ipynb_checkpoints"}
PAPERS = ["1023", "913", "2657", "2578", "2167", "1717", "1908", "4090", "1171", "3463"]
INPUTS = ["paper.pdf", "paper_text.txt", "metadata.txt", "code_links.txt"]
SLOTS = ["MAJOR-MISS", "MAJOR-DIFF", "MAJOR-METH", "MAJOR-BUG"]
SLOT_CATEGORY = {"MAJOR-MISS": "missing", "MAJOR-DIFF": "difference",
                  "MAJOR-METH": "methodology", "MAJOR-BUG": "bug"}


def read_text_raw(path: Path) -> str:
    """Read WITHOUT universal-newline translation. Path.read_text() silently
    normalizes \\r\\n -> \\n on read, so a patch's old_string containing a literal
    CRLF (copied verbatim from a CRLF-terminated file, e.g. 1717/engine.py) would
    never match via count()/replace() even though it's byte-for-byte correct."""
    return path.read_bytes().decode(errors="replace")


def write_text_raw(path: Path, text: str) -> None:
    path.write_bytes(text.encode())
MODEL, EFFORT, TIMEOUT = "gpt-5.6-sol", "high", 2700
_print_lock = threading.Lock()

BRIEF = """\
# Defect-seeding brief — MAJOR-only recall study (4 categories, one each)

You are the red team for an evaluation of an automated paper-code auditor. Propose
SUBSTANTIVE, REALISTIC, conclusion-invalidating defects to plant in this repository
so we can measure how many the auditor detects. Inputs here: `paper.pdf` (source of
truth), `paper_text.txt` (grep-able extraction), released code under `code/`
(READ-ONLY — do not modify it).

## The four slots (propose TWO independent candidates for EACH)

- MAJOR-MISS: there is NO CODE ANYWHERE IN THE REPO backing a major experiment/result
  the paper reports. Not a stub left running in its place — an actual absence. Pick a
  self-contained reported experiment (an ablation, a model variant, a specific
  figure/table's pipeline) and remove the file(s)/function(s) that would produce it.
  `delete_file` is allowed for this slot (see patch format below). Do NOT propose a
  stub/no-op left in place of a mechanism for this slot — that is MAJOR-BUG instead.
- MAJOR-DIFF: a full, real, coherently-implemented DIFFERENT method stands in for the
  one the paper claims — e.g. the paper says it uses a GCN but the code implements a
  CNN. Must be genuinely functional: real, differently-shaped computation, real
  gradients. If your substitute would reduce to a no-op/identity, propose it as
  MAJOR-BUG or MAJOR-MISS instead.
- MAJOR-METH: real data leakage in the MAIN pipeline — the test set overlaps/equals
  train, or hyperparameter tuning / early stopping / model selection reads the test
  set (e.g. peeking the test set during hpam tuning, or training on test).
- MAJOR-BUG: a real, substantive, HARD-TO-FIX bug in central code — not "just a
  missing import" or a trivial typo anyone would immediately dismiss as unrelated to
  the paper's methodology. Example: a wrong output activation (a ReLU where the task
  needs a linear or softmax output), silently destroying negative logits/regression
  outputs. A stub that skips a mechanism entirely (e.g. the training loop never
  updates weights — optimizer.step()/loss.backward() silently skipped) also belongs
  here, not MAJOR-MISS — the code exists and runs, it's just wrong.

## MISS vs. BUG/DIFF — apply this test before choosing a slot

Is there still code that runs and does something (even if trivial/wrong/stubbed) where
the mechanism should be? Yes -> BUG or DIFF (DIFF if the substitute is a genuine
functioning alternative; BUG if it's a no-op/trivial stand-in). No code path produces
this result at all, anywhere -> MISS.

## Silence is NOT required

Unlike a typical seeding brief, these defects do NOT need to be crash-safe. A
`delete_file` that breaks an import elsewhere, or a bug that raises a visible error
partway through a run, is ACCEPTABLE — the bar is "hard-to-fix and substantive," not
"invisible." A trivial, immediately-dismissible error (a stray typo, an unrelated
missing import) is NOT acceptable and will be rejected at approval regardless of
whether it crashes.

## Hard requirements for every candidate (violations discarded)
1. AUTHOR-PLAUSIBLE: repo's own style; a location where a real author error could
   occur; NO comments, names, or strings hinting at the change.
2. FINDABLE IN PRINCIPLE: a careful reader with paper AND code could detect it;
   give the reasoning chain in `expected_reasoning`.
3. PAPER-DEPENDENT: contradicts something the paper actually states — quote it
   verbatim in `paper_anchor`.
4. ATOMIC & NON-INTERACTING: one candidate = one defect; the four finally-chosen
   seeds will be applied TOGETHER, so candidates for different slots must not touch
   the same lines or depend on each other. Prefer a distinct file per candidate;
   never more than one candidate of the SAME slot in the same location.
5. NOT restricted to a single line or a single string-replace: you may add new code
   across multiple hunks, and one hunk may be a whole-file deletion — see the patch
   format below. Multi-file/multi-hunk seeds are fully allowed and encouraged where
   the defect naturally spans more than one site (e.g. a MISS seed removing both a
   script and the config entry that invoked it).
6. AVOID these files entirely (they host verifier-kept findings of the original
   audit; seeding them conflates attribution):
{avoid}

## Patch format (exact-string edits + optional file deletion, applied programmatically)

"patch": {{"hunks": [
   {{"action": "replace", "file": "<path relative to code/>",
     "old_string": "<verbatim excerpt occurring EXACTLY ONCE in the file>",
     "new_string": "<replacement; may be \\"\\" for an in-file deletion>"}},
   {{"action": "delete_file", "file": "<path relative to code/>"}}
]}}
`file` starts with "{repo}/". A seed's hunks apply atomically (all or nothing). Use
`delete_file` only within MAJOR-MISS candidates.

## Output
Write EXACTLY one file, `seed_candidates.json`, in THIS directory (not under code/):
{{"paper": "{pid}", "seeds": [
   {{"sid": "S01", "slot": "MAJOR-MISS", "severity": "major", "category": "missing",
     "title": "...", "description": "...", "expected_reasoning": "...",
     "paper_anchor": "<verbatim quote>", "placement": "central",
     "patch": {{"hunks": [...]}} }}, ...]}}
8 seeds total: two per slot, sids S01..S08 in slot order (MAJOR-MISS, MAJOR-DIFF,
MAJOR-METH, MAJOR-BUG). severity is always "major"; category must match the slot
(missing/difference/methodology/bug respectively). End with a one-line tally per slot.
"""


def tree_hashes(root: Path) -> dict:
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in IGNORE]
        for f in sorted(fn):
            p = Path(dp) / f
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def resolve(pid: str):
    d = sorted(glob.glob(str(AUDITS / f"{pid}")))
    assert d, f"no audit dir for {pid}"
    src = Path(d[0])
    if not (src / "code").is_dir():
        # The audited repositories are not redistributed, so a public checkout
        # has no code/ tree. Only the repo *name* is needed here, and the frozen
        # answer key already records it.
        key = Path(__file__).resolve().parent / f"ANSWER_KEY_{pid}.json"
        if key.exists():
            return src, json.loads(key.read_text())["repo"]
        raise SystemExit(f"{pid}: no code/ tree and no ANSWER_KEY_{pid}.json to "
                         "read the repository name from")
    repos = [x.name for x in (src / "code").iterdir() if x.is_dir()]
    # some audit snapshots keep a stale archive copy alongside the real repo (e.g.
    # 3463: "ganchaofan0000__DiTF" + "...DiTF.camera_ready_release_ARCHIVE") — drop
    # ARCHIVE-suffixed dirs, matching final_recall_study's ANSWER_KEY_3463.json choice.
    non_archive = [r for r in repos if "ARCHIVE" not in r]
    if len(non_archive) == 1:
        repos = non_archive
    assert len(repos) == 1, f"{pid}: expected 1 repo, got {repos}"
    return src, repos[0]


def avoid_files(src: Path, repo: str) -> list[str]:
    vj = src / "findings_verified.json"
    out = set()
    if vj.exists():
        for f in json.loads(vj.read_text()).get("findings", []):
            if str(f.get("verdict", "keep")).lower() in ("keep", "lower", "lowered", ""):
                fp = str(f.get("file") or "")
                if fp and fp != "paper.pdf" and not fp.startswith("_audit_code"):
                    rel = fp.split(f"code/{repo}/")[-1] if f"code/{repo}/" in fp else fp
                    out.add(rel)
    return sorted(out)


def stage(pid: str, src: Path, repo: str) -> Path:
    if not (src / "code" / repo).is_dir():
        # Seeding copies the audited repository into a scratch tree and edits
        # it. The audited repositories are not redistributed, so this step is
        # only runnable where they were cloned.
        raise SystemExit(
            f"{pid}: {src / 'code' / repo} not found -- seeding needs the audited "
            "repository, which is not part of the release. The seeds and answer "
            "keys it produced are committed under seeds/ and ANSWER_KEY_*.json."
        )
    wd = WORK / pid
    wd.mkdir(parents=True, exist_ok=True)
    for fn in INPUTS:
        s = src / fn
        if s.exists() and not (wd / fn).exists():
            (wd / fn).write_bytes(s.read_bytes())
    dst = wd / "code" / repo
    if not dst.exists():
        shutil.copytree(src / "code" / repo, dst,
                        ignore=lambda d, n: [x for x in n if x in IGNORE])
        for dp, _, fn in os.walk(dst):
            for f in fn:
                p = Path(dp) / f
                p.chmod(p.stat().st_mode & ~0o222)
    return wd


def validate(pid: str, repo: str, data: dict, av: list[str]) -> list[str]:
    """Light structural validation only — no crash-safety/import-safety gate
    (a seeded defect need not run silently for this study)."""
    errs = []
    seeds = data.get("seeds") or []
    per_slot = {s: 0 for s in SLOTS}
    code_root = WORK / pid / "code"
    for s in seeds:
        sid = s.get("sid", "?")
        slot = s.get("slot")
        if slot not in SLOTS:
            errs.append(f"{sid}: bad slot {slot!r}"); continue
        n_errs_before = len(errs)
        if s.get("category") != SLOT_CATEGORY[slot]:
            errs.append(f"{sid}: category {s.get('category')!r} != expected "
                        f"{SLOT_CATEGORY[slot]!r} for slot {slot}")
        for field in ("title", "description", "expected_reasoning", "paper_anchor"):
            if not str(s.get(field, "")).strip():
                errs.append(f"{sid}: empty {field}")
        hunks = (s.get("patch") or {}).get("hunks") or []
        if not hunks:
            errs.append(f"{sid}: no hunks in patch"); continue
        for h in hunks:
            f = str(h.get("file", ""))
            rel = f.split(f"{repo}/", 1)[-1] if f"{repo}/" in f else f
            if rel in av:
                errs.append(f"{sid}: seeds avoid-listed file {rel}")
            target = code_root / f
            action = h.get("action")
            if action == "delete_file":
                if not target.is_file():
                    errs.append(f"{sid}: delete_file target missing: {f}")
                continue
            if action != "replace":
                errs.append(f"{sid}: bad hunk action {action!r}"); continue
            if not target.is_file():
                errs.append(f"{sid}: file not found: {f}"); continue
            old = h.get("old_string", "")
            n = read_text_raw(target).count(old) if old else 0
            if n != 1:
                errs.append(f"{sid}: old_string occurs {n}x in {f}")
            if h.get("new_string") is None:
                errs.append(f"{sid}: missing new_string")
        if len(errs) == n_errs_before:
            per_slot[slot] += 1
    for slot, n in per_slot.items():
        if n < 1:
            errs.append(f"slot {slot}: 0 valid candidates proposed")
    return errs


def run_paper(pid: str) -> str:
    src, repo = resolve(pid)
    wd = stage(pid, src, repo)
    out = wd / "seed_candidates.json"
    dst = SEEDS / f"candidates_{pid}.json"
    if dst.exists():
        return f"{pid}: exists, skipped"
    av = avoid_files(src, repo)
    prompt = BRIEF.format(pid=pid, repo=repo,
                          avoid="\n".join(f"   - {a}" for a in av) or "   (none)")
    pre = tree_hashes(wd / "code")
    log = LOGS / f"{pid}.propose.jsonl"
    cmd = ["codex", "exec", "-C", str(wd), "-m", MODEL,
           "--sandbox", "workspace-write", "--skip-git-repo-check",
           "--color", "never", "--json",
           "-c", f'model_reasoning_effort="{EFFORT}"',
           "-o", str(log.with_suffix(".last.md")), "-"]
    with _print_lock:
        print(f"[{pid}] codex starting ({repo})", flush=True)
    with log.open("w") as lf:
        r = subprocess.run(cmd, input=prompt, stdout=lf, stderr=subprocess.STDOUT,
                           text=True, timeout=TIMEOUT)
    if tree_hashes(wd / "code") != pre:
        return f"{pid}: FATAL — codex modified the staged code tree"
    if not out.is_file():
        return f"{pid}: FAIL — no seed_candidates.json (rc={r.returncode})"
    try:
        data = json.loads(out.read_text())
    except Exception as e:
        return f"{pid}: FAIL — bad JSON: {e}"
    errs = validate(pid, repo, data, av)
    data["_validation_errors"] = errs
    data["_repo"] = repo
    for s in data.get("seeds", []):
        s.setdefault("approved", False)
    dst.write_text(json.dumps(data, indent=2) + "\n")
    n = len(data.get("seeds", []))
    return f"{pid}: {n} candidates, {len(errs)} validation error(s) -> {dst.name}"


def main():
    only = sys.argv[1:] or PAPERS
    SEEDS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for res in ex.map(run_paper, only):
            with _print_lock:
                print(res, flush=True)
    print("\nNEXT: triviality gate, then collision check, then author review of "
          "seeds/candidates_*.json")


if __name__ == "__main__":
    main()
