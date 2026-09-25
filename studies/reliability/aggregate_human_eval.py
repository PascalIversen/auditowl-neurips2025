#!/usr/bin/env python3
"""Aggregate the human-eval worksheets (HUMAN_EVAL_<paper>.md) into tidy data.

Each worksheet holds N "distinct defects" (findings merged across the 10 audit
runs). A human ticked exactly ONE verdict box per finding out of five:

    correct & relevant            -- true AND a substantive repro issue
    correct but wrong severity    -- true, worth raising, severity miscalibrated
    correct but not relevant      -- true but trivial / nitpick / acknowledged
    unsure                        -- undecidable from the frozen repo + paper
    false                         -- the claim misreads the code/paper

This script parses every studies/reliability/HUMAN_EVAL_*.md, pulls each finding's
objective metadata (category, topic, severity, confidence, detection X/Y runs)
plus the human verdict and free-text note, and writes:

    data/human_eval_findings.csv   one row per finding (tidy, for the figure)
    data/human_eval_summary.json   per-paper + pooled verdict tallies & rates

It also prints a readable rollup and fails loudly on integrity problems: a
finding with >1 ticked boxes, or a missing detection X/Y. A finding with zero
ticked boxes is not a failure — its paper is treated as not-started or
in-progress and excluded from the pooled counts rather than flagged.

Run:  python studies/reliability/aggregate_human_eval.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

RB = Path(__file__).resolve().parent
DATA = RB / "data"

# Canonical verdict keys, ordered worst -> best (used everywhere downstream).
VERDICTS = [
    ("false", "false"),
    ("unsure", "unsure"),
    ("correct but not relevant", "correct_not_relevant"),
    ("correct but wrong severity", "correct_wrong_severity"),
    ("correct & relevant", "correct_relevant"),
]
LABEL2KEY = {label: key for label, key in VERDICTS}
KEYS = [key for _, key in VERDICTS]
# "correct" = the auditor got the code/paper right, regardless of severity/relevance
CORRECT_KEYS = {"correct_relevant", "correct_wrong_severity", "correct_not_relevant"}

# A verdict box: the label phrase, then `[ ]` / `[]` / `[x]` / `[X]`.
_BOX = re.compile(
    r"(correct & relevant|correct but wrong severity|correct but not relevant|unsure|false)"
    r"\s*`\[([ xX]*)\]`"
)
_HDR = re.compile(r"^###\s+(F\d+)\s*·\s*(.*?)\s*$")
_CATTOP = re.compile(r"_category:\s*(.*?)\s*·\s*topic:\s*(.*?)_")
_SEV = re.compile(r"severity:\s*([A-Za-z]+)")
_CONF = re.compile(r"confidence:\s*([A-Za-z]+)")
_DET = re.compile(r"detection:\s*(\d+)\s*/\s*(\d+)")
_VARIED = re.compile(r"\(varied:\s*([^)]*)\)")


def paper_id(md: Path) -> str:
    m = re.search(r"HUMAN_EVAL_(\w+?)\.md$", md.name)
    return m.group(1) if m else md.stem


def parse_verdict_line(line: str) -> tuple[str | None, list[str]]:
    """Return (verdict_key or None, list_of_ticked_keys) for a Verdict line."""
    ticked = []
    for label, mark in _BOX.findall(line):
        if "x" in mark.lower():
            ticked.append(LABEL2KEY[label])
    return (ticked[0] if len(ticked) == 1 else None), ticked


def parse_finding(block: str) -> dict | None:
    """Parse one '### F.. ...' block into a record (or None if it isn't one)."""
    lines = block.splitlines()
    hm = _HDR.match(lines[0])
    if not hm:
        return None
    fid, title = hm.group(1), hm.group(2)

    cat = top = ""
    sev = conf = ""
    sev_varied = False
    det = det_total = None
    verdict = None
    ticked: list[str] = []
    notes_lines: list[str] = []
    in_notes = False

    for ln in lines[1:]:
        if (m := _CATTOP.search(ln)):
            cat, top = m.group(1).strip(), m.group(2).strip()
        if ln.lstrip().startswith("**severity:"):
            if (m := _SEV.search(ln)):
                sev = m.group(1).lower()
            if (m := _CONF.search(ln)):
                conf = m.group(1).lower()
            if (m := _DET.search(ln)):
                det, det_total = int(m.group(1)), int(m.group(2))
            sev_varied = bool(_VARIED.search(ln))
        if ln.startswith("**Verdict:**"):
            verdict, ticked = parse_verdict_line(ln)
            in_notes = False
        elif ln.startswith("**Notes:**"):
            in_notes = True
            rest = ln.split("**Notes:**", 1)[1].strip()
            if rest:
                notes_lines.append(rest)
        elif in_notes:
            if ln.strip() == "---":     # markdown rule = end of this finding block
                in_notes = False
                continue
            notes_lines.append(ln)

    return {
        "fid": fid,
        "title": title,
        "category": cat,
        "topic": top,
        "severity": sev,
        "severity_varied": sev_varied,
        "confidence": conf,
        "detection": det,
        "detection_total": det_total,
        "verdict": verdict,
        "n_ticked": len(ticked),
        "ticked": ticked,
        "notes": "\n".join(notes_lines).strip(),
    }


def parse_worksheet(md: Path) -> list[dict]:
    paper = paper_id(md)
    text = md.read_text(encoding="utf-8")
    # split on the finding headers; keep the header with its block
    blocks = re.split(r"(?=^###\s+F\d+\s)", text, flags=re.M)
    out = []
    for b in blocks:
        rec = parse_finding(b)
        if rec is not None:
            rec["paper"] = paper
            out.append(rec)
    return out


def tally(records: list[dict]) -> dict:
    counts = {k: 0 for k in KEYS}
    for r in records:
        if r["verdict"]:
            counts[r["verdict"]] += 1
    n = len(records)
    n_verdict = sum(counts.values())
    correct = sum(counts[k] for k in CORRECT_KEYS)
    return {
        "n_findings": n,
        "counts": counts,
        "correct": correct,            # auditor read code/paper right
        "false": counts["false"],
        "unsure": counts["unsure"],
        # precision-style rates over findings that received a decisive verdict
        "rate_correct": round(correct / n, 4) if n else 0.0,
        "rate_false": round(counts["false"] / n, 4) if n else 0.0,
        "rate_correct_relevant": round(counts["correct_relevant"] / n, 4) if n else 0.0,
    }


def main() -> int:
    DATA.mkdir(exist_ok=True)
    sheets = sorted(RB.glob("HUMAN_EVAL_*.md"))
    if not sheets:
        print("no HUMAN_EVAL_*.md worksheets found in", RB, file=sys.stderr)
        return 1

    all_records: list[dict] = []
    problems: list[str] = []     # genuine errors -> non-zero exit
    skipped: list[str] = []      # informational (not-yet-started / in-progress)
    for md in sheets:
        recs = parse_worksheet(md)
        if not recs:
            continue
        paper = recs[0]["paper"]
        n = len(recs)
        done = [r for r in recs if r["n_ticked"] == 1]
        blank = [r for r in recs if r["n_ticked"] == 0]
        multi = [r for r in recs if r["n_ticked"] > 1]
        if multi:                                  # >1 box ticked = real mistake
            for r in multi:
                problems.append(f"{paper} {r['fid']}: {r['n_ticked']} boxes ticked "
                                f"({r['ticked']}) — expected exactly 1")
            continue
        if not done:                               # nothing ticked = not started
            skipped.append(f"{paper}: not started (0/{n} ticked)")
            continue
        if blank:                                  # some ticked, some not = in progress
            skipped.append(f"{paper}: in progress ({len(done)}/{n} ticked, "
                           f"{len(blank)} blank) — excluded until complete")
            continue
        for r in done:                             # complete worksheet -> include it
            if r["detection"] is None:
                problems.append(f"{paper} {r['fid']}: no detection X/Y parsed")
        all_records.extend(done)

    # ---- write tidy CSV (one row per finding) ----
    csv_path = DATA / "human_eval_findings.csv"
    cols = ["paper", "fid", "verdict", "is_correct", "is_false", "detection",
            "detection_total", "severity", "severity_varied", "confidence",
            "category", "topic", "title", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in all_records:
            w.writerow([
                r["paper"], r["fid"], r["verdict"] or "",
                int(r["verdict"] in CORRECT_KEYS), int(r["verdict"] == "false"),
                r["detection"], r["detection_total"], r["severity"],
                int(r["severity_varied"]), r["confidence"],
                r["category"], r["topic"], r["title"],
                r["notes"].replace("\n", " ").strip(),
            ])

    # ---- write summary JSON (per paper + pooled) ----
    by_paper = {}
    for r in all_records:
        by_paper.setdefault(r["paper"], []).append(r)
    summary = {
        "verdict_keys": KEYS,
        "correct_keys": sorted(CORRECT_KEYS),
        "per_paper": {p: tally(recs) for p, recs in sorted(by_paper.items())},
        "pooled": tally(all_records),
    }
    json_path = DATA / "human_eval_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # ---- readable rollup ----
    print(f"included {len(by_paper)} of {len(sheets)} worksheet(s): "
          + ", ".join(f"#{p} ({len(r)})" for p, r in sorted(by_paper.items())))
    for s in skipped:
        print(f"  · skipped {s}")
    print()
    hdr = ["verdict"] + [f"#{p}" for p in sorted(by_paper)] + ["pooled"]
    print(f"{hdr[0]:<26}" + "".join(f"{h:>9}" for h in hdr[1:]))
    for key in reversed(KEYS):  # best -> worst, top down
        row = [summary["per_paper"][p]["counts"][key] for p in sorted(by_paper)]
        row.append(summary["pooled"]["counts"][key])
        print(f"{key:<26}" + "".join(f"{v:>9}" for v in row))
    tot = [summary["per_paper"][p]["n_findings"] for p in sorted(by_paper)]
    tot.append(summary["pooled"]["n_findings"])
    print(f"{'TOTAL':<26}" + "".join(f"{v:>9}" for v in tot))
    print()
    pl = summary["pooled"]
    print(f"pooled: {pl['correct']}/{pl['n_findings']} correct "
          f"({pl['rate_correct']*100:.0f}%) · "
          f"{pl['counts']['correct_relevant']}/{pl['n_findings']} correct & relevant "
          f"({pl['rate_correct_relevant']*100:.0f}%) · "
          f"{pl['false']} false · {pl['unsure']} unsure")
    print()
    print(f"wrote {csv_path.relative_to(RB.parent)}")
    print(f"wrote {json_path.relative_to(RB.parent)}")

    if problems:
        print("\n⚠ INTEGRITY PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 2
    print("\nintegrity: OK (every finding has exactly one ticked box + detection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
