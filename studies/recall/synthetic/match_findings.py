#!/usr/bin/env python3
"""Match run findings against the pre-registered per-paper answer keys.

Two-stage matching:

  Stage A (this script, deterministic "anchor" pass, CATEGORY-AGNOSTIC):
    a kept finding matches a seed iff it cites ANY of the seed's sites (file, and
    when both have line anchors, overlapping within +/-window) — every seed may be
    multi-site, so "any site" is the default rule here (not a special case for one
    bonus slot, as in final_recall_study). For a `delete_file` site (and `missing`
    seeds generally) a finding also auto-matches if the deleted file's basename
    appears in the finding's title/claim/concern (missing-artifact findings often
    cite paper.pdf or a README instead of the absent file). Findings are matched
    against baseline findings the same way.

  Stage B (LLM + human adjudication): everything Stage A leaves unassigned goes to
    data/match_inputs/<pid>.json; run the prompt in data/llm_match_prompt.md
    (fixed clusters = seeds + baseline findings; conservative; "emergent" allowed),
    then hand-check every seed assignment and write data/matches_final.json in the
    SAME schema as matches_auto.json. analyze_recall.py prefers the final file.

Assignment schema (both files):
  {"assignments": [{"paper": "1023", "run": "run_01", "fid": "<finding id>",
                    "assigned_to": "seed:S03" | "baseline:<id>" | "emergent" | "unassigned",
                    "method": "anchor" | "llm" | "human",
                    "mechanism_correct": true|false|null}]}

Usage:
    python studies/recall/synthetic/match_findings.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # same-dir imports (propose_major), regardless of cwd


def _load_common():
    # NOT sys.path.insert(0, HERE.parent) + `from common import ...`: the parent dir
    # (recall_experiment/) has its OWN same-named match_findings.py/analyze_recall.py
    # (the pilot's) — inserting it into sys.path risks Python resolving "match_findings"
    # to the wrong module. Load common.py by explicit file path instead.
    spec = importlib.util.spec_from_file_location("recall_common", HERE.parent / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_common()
dump_json, load_json = _common.dump_json, _common.load_json
lines_overlap, norm_file, norm_seed_file = (
    _common.lines_overlap, _common.norm_file, _common.norm_seed_file)
from propose_major import resolve  # noqa: E402

RUNS = HERE / "runs"
DATA = HERE / "data"
WINDOW = 10
PAPERS = ["1023", "913", "2657", "2578", "2167", "1717", "1908", "4090", "1171", "3463"]


def baseline_findings(pid: str) -> list[dict]:
    """The ORIGINAL audit's kept findings for this paper — excluded from recall math,
    used only to check re-detection rate. ANSWER_KEY_<pid>.json doesn't embed these
    (freeze.py doesn't either), so read them straight from the untouched audits/ tree."""
    src, _repo = resolve(pid)
    vj = src / "findings_verified.json"
    if not vj.exists():
        return []
    out = []
    for f in load_json(vj).get("findings", []):
        # "question" status counts alongside "finding": a hedged item that still
        # correctly points a reviewer at the right place is a hit, not a non-event.
        if f.get("status", "finding") not in ("finding", "question"):
            continue
        if str(f.get("verdict", "keep")).lower() not in ("keep", "lower", "lowered", ""):
            continue
        out.append({"id": f.get("id"), "category": f.get("category"),
                    "file": f.get("file"), "line_start": f.get("line_start"),
                    "line_end": f.get("line_end"), "title": f.get("title")})
    return out

LLM_PROMPT = """\
# Fixed-cluster finding assignment (the synthetic recall arm, stage-B matcher)

You are matching reproducibility-audit findings from repeated blind audits against a
PRE-REGISTERED answer key of planted defects. Stage A already assigned the easy
cases by file/line anchor; you get the leftovers.

Read data/match_inputs/<pid>.json:
  {paper, seeds:[{sid, category, sites:[{file,line}], title, description}],
   baseline:[{id, category, file, title}],
   unassigned:[{fid, run, category, loc, title, claim, concern}]}

For EACH unassigned finding, output exactly one assignment:
  - "seed:<sid>"     — the finding describes the SAME underlying defect the seed
                       planted (same thing wrong with the same artefact/behaviour),
                       even if category, file, line, or wording differ.
                       Also set mechanism_correct: true iff the finding identifies
                       the defect's actual mechanism (not merely "something wrong
                       near that file").
  - "baseline:<id>"  — it re-detects a pre-existing (non-planted) finding.
  - "emergent"       — genuinely neither: a new observation about the repo.

BE CONSERVATIVE, exactly like a defect-level deduplicator: do not credit a seed
just because a finding touches the same file; the finding must be about the
planted problem. When torn between seed and emergent, choose emergent — recall
must not be inflated by generous matching.

Return JSON: {"paper": "<pid>", "assignments": [{"fid": "...", "run": "...",
"assigned_to": "...", "mechanism_correct": true|false|null}], "notes": "one line"}
Every input fid appears exactly once. mechanism_correct is null unless assigned_to
is a seed.
"""


def kept_findings(run_dir):
    vj = run_dir / "findings_verified.json"
    fj = run_dir / "findings.json"
    src = vj if vj.exists() else fj
    if not src.exists():
        return None
    fl = load_json(src).get("findings", [])
    out = []
    for f in fl:
        # "question" status counts alongside "finding" — see baseline_findings().
        if f.get("status", "finding") not in ("finding", "question"):
            continue
        if vj.exists() and f.get("verdict") not in ("keep", "lowered", None):
            continue
        out.append(f)
    return out


def basename_stems(path: str) -> list[str]:
    base = path.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    return [s for s in {base, stem} if len(s) >= 4]


def match_seed(seed: dict, f: dict) -> bool:
    ffile = norm_file(f.get("file"))
    for site in seed["sites"]:
        sfile = norm_seed_file(site["file"])
        if ffile != sfile:
            continue
        if site.get("action") == "delete_file" or site.get("line") is None:
            return True
        ov = lines_overlap(site["line"], site["line"],
                           f.get("line_start"), f.get("line_end"), WINDOW)
        if ov is None or ov:
            return True
    if seed["category"] == "missing":
        text = " ".join(str(f.get(k, "")) for k in ("title", "claim", "concern"))
        for site in seed["sites"]:
            sfile = norm_seed_file(site["file"])
            if any(re.search(rf"\b{re.escape(s)}\b", text) for s in basename_stems(sfile)):
                return True
    return False


def match_baseline(b: dict, f: dict) -> bool:
    if norm_file(b.get("file")) != norm_file(f.get("file")):
        return False
    ov = lines_overlap(b.get("line_start"), b.get("line_end"),
                       f.get("line_start"), f.get("line_end"), WINDOW)
    return True if ov is None else ov


def load_answer_keys() -> list[dict]:
    keys = []
    for pid in PAPERS:
        p = HERE / f"ANSWER_KEY_{pid}.json"
        if not p.exists():
            continue
        k = load_json(p)
        k["num"] = pid
        keys.append(k)
    if not keys:
        sys.exit("no ANSWER_KEY_<pid>.json found — run freeze.py first")
    return keys


def main() -> None:
    keys = load_answer_keys()
    DATA.mkdir(exist_ok=True)
    (DATA / "match_inputs").mkdir(exist_ok=True)

    assignments = []
    inputs_written = []
    incomplete = []
    for pk in keys:
        pid = pk["num"]
        pdir = RUNS / pid
        baseline = baseline_findings(pid)
        unassigned_all = []
        for rd in sorted(pdir.glob("run_*")) if pdir.is_dir() else []:
            fl = kept_findings(rd)
            if fl is None:
                incomplete.append(f"{pid}/{rd.name}")
                continue
            for f in fl:
                fid = f.get("id", "?")
                hit = next((s for s in pk["seeds"] if match_seed(s, f)), None)
                if hit:
                    assignments.append({"paper": pid, "run": rd.name, "fid": fid,
                                        "assigned_to": f"seed:{hit['id']}",
                                        "method": "anchor", "mechanism_correct": None})
                    continue
                base = next((b for b in baseline if match_baseline(b, f)), None)
                if base:
                    assignments.append({"paper": pid, "run": rd.name, "fid": fid,
                                        "assigned_to": f"baseline:{base['id']}",
                                        "method": "anchor", "mechanism_correct": None})
                    continue
                assignments.append({"paper": pid, "run": rd.name, "fid": fid,
                                    "assigned_to": "unassigned", "method": "anchor",
                                    "mechanism_correct": None})
                unassigned_all.append({
                    "fid": fid, "run": rd.name, "category": f.get("category"),
                    "loc": f"{f.get('file')}:{f.get('line_start')}",
                    "title": f.get("title"), "claim": f.get("claim"),
                    "concern": f.get("concern")})
        if unassigned_all:
            dump_json({
                "paper": pid,
                "seeds": [{"sid": s["id"], "category": s["category"],
                           "sites": s["sites"], "title": s["title"],
                           "description": s["description"]}
                          for s in pk["seeds"]],
                "baseline": baseline,
                "unassigned": unassigned_all,
            }, DATA / "match_inputs" / f"{pid}.json")
            inputs_written.append(pid)

    dump_json({"assignments": assignments}, DATA / "matches_auto.json")
    (DATA / "llm_match_prompt.md").write_text(LLM_PROMPT)

    n_seed = sum(1 for a in assignments if a["assigned_to"].startswith("seed:"))
    n_base = sum(1 for a in assignments if a["assigned_to"].startswith("baseline:"))
    n_un = sum(1 for a in assignments if a["assigned_to"] == "unassigned")
    print(f"stage A: {len(assignments)} kept findings -> {n_seed} seed-matched, "
          f"{n_base} baseline, {n_un} unassigned")
    if incomplete:
        print(f"WARNING: {len(incomplete)} run(s) without findings yet: {incomplete}")
    if inputs_written:
        print(f"stage B inputs: data/match_inputs/{{{','.join(inputs_written)}}}.json "
              f"+ data/llm_match_prompt.md")
        print("after the LLM pass + hand-check, write data/matches_final.json "
              "(same schema, no 'unassigned' left, seed matches hand-verified)")
    else:
        print("no unassigned findings — copy matches_auto.json to matches_final.json "
              "after hand-checking the seed matches")


if __name__ == "__main__":
    main()
