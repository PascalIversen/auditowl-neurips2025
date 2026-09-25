#!/usr/bin/env python3
"""Collision check: once an author has picked one candidate per
slot for a paper (recorded in seeds/selection_proposed.json, same schema as
final_recall_study's:
    {"<pid>": {"<SLOT>": {"sid": "<sid>", "rule": "<one line>",
                           "no_paper_suspicion": <bool|None>}, ...}, ...}
selected by hand — this script does not write it), flag any two of that paper's 4
chosen seeds whose sites land <MIN_GAP lines apart in the same file. A `delete_file`
site collides with ANY other site in that file regardless of line distance (deleting
the file touches every line in it). On collision, re-propose the losing slot with the
colliding file added to its forbidden list (see re_propose.py-style targeted repair).

Usage: python3 check_collisions.py [pid ...]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from propose_major import read_text_raw

HERE = Path(__file__).resolve().parent
SEEDS = HERE / "seeds"
PAPERS = ["1023", "913", "2657", "2578", "2167", "1717", "1908", "4090", "1171", "3463"]
MIN_GAP = 50


def seed_sites(pid: str, sid: str, cmap: dict) -> list[dict]:
    """Recompute each hunk's landed (file, line|None) against the STAGED tree
    (work/<pid>/code/), matching propose_major.py's own line-numbering convention."""
    s = cmap[sid]
    code_root = HERE / "work" / pid / "code"
    if not code_root.is_dir():
        # Without the staged tree every hunk is skipped below and the gate
        # reports "no collisions" for a check it never performed. Fail loudly
        # instead: this only runs while seeding, when the tree is present.
        raise SystemExit(
            f"staged tree {code_root} not found -- cannot check for collisions. "
            "This gate runs during seeding, against the working copy the seeds "
            "were applied to; it is not reproducible from the release alone."
        )
    sites = []
    for h in (s.get("patch") or {}).get("hunks") or []:
        f = h.get("file", "")
        if h.get("action") == "delete_file":
            sites.append({"file": f, "line": None, "action": "delete_file"})
            continue
        fp = code_root / f
        if not fp.is_file():
            continue
        t = read_text_raw(fp)
        old = h.get("old_string", "")
        if old not in t:
            continue
        line = t[:t.index(old)].count("\n") + 1
        sites.append({"file": f, "line": line, "action": "replace"})
    return sites


def check_paper(pid: str) -> list[str]:
    sel_f = SEEDS / "selection_proposed.json"
    cand_f = SEEDS / f"candidates_{pid}.json"
    if not sel_f.exists() or not cand_f.exists():
        return [f"{pid}: missing selection_proposed.json or candidates file, skipped"]
    sel = json.loads(sel_f.read_text()).get(pid)
    if not sel:
        return [f"{pid}: no selection recorded, skipped"]
    cmap = {s["sid"]: s for s in json.loads(cand_f.read_text())["seeds"]}

    per_slot_sites = {}
    for slot, v in sel.items():
        per_slot_sites[slot] = seed_sites(pid, v["sid"], cmap)

    collisions = []
    slots = sorted(per_slot_sites)
    for i, slot_a in enumerate(slots):
        for slot_b in slots[i + 1:]:
            for site_a in per_slot_sites[slot_a]:
                for site_b in per_slot_sites[slot_b]:
                    if site_a["file"] != site_b["file"]:
                        continue
                    if site_a["line"] is None or site_b["line"] is None:
                        collisions.append(
                            f"{pid}: {slot_a} (delete_file) collides with {slot_b} "
                            f"in {site_a['file']} — one deletes the whole file")
                        continue
                    gap = abs(site_a["line"] - site_b["line"])
                    if gap < MIN_GAP:
                        collisions.append(
                            f"{pid}: {slot_a}:{site_a['line']} and {slot_b}:"
                            f"{site_b['line']} in {site_a['file']} — {gap} lines apart "
                            f"(< {MIN_GAP})")
    if not collisions:
        return [f"{pid}: no collisions across {len(slots)} selected slots"]
    return collisions


def main():
    only = sys.argv[1:] or PAPERS
    any_collision = False
    for pid in only:
        for line in check_paper(pid):
            print(line)
            if "collides" in line or "lines apart" in line:
                any_collision = True
    if any_collision:
        print("\nNEXT: re-propose the losing slot(s) above (re_propose.py-style, "
              "with the colliding file added to forbidden) before author approval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
