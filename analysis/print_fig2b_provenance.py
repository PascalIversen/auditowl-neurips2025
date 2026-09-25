#!/usr/bin/env python3
"""Print Figure 2b's 8 example cards next to the committed finding each one
summarizes, so the relationship between analysis/severe_findings_display.md
(hand-picked, hand-written cards) and audits/<paper>/findings.json (the
underlying evidence) is at least checkable by re-running this script, even
though the cards themselves are not machine-generated.

Matches each card to its finding by `check_script` when the card's Evidence
line names one, otherwise by (file, line_start) parsed from that line. Never
guesses: a card that can't be matched is reported as UNMATCHED rather than
paired with the wrong finding.

Run: python analysis/print_fig2b_provenance.py
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISPLAY_MD = ROOT / "analysis" / "severe_findings_display.md"

CARD_RE = re.compile(
    r"^### #(?P<paper>\d+) · .*?\n\n"
    r"\*\*(?P<headline>.+?)\*\*\n\n"
    r"_confidence: (?P<confidence>\w+) · topic: (?P<topic>.+?)_\n\n"
    r"\*\*Claim:\*\* (?P<claim>.+?)\n\n"
    r"\*\*Concern:\*\* (?P<concern>.+?)\n\n"
    r"\*\*Ask:\*\* .+?\n\n"
    r"\*\*Evidence:\*\* (?P<evidence>.+?)(?=\n\n##|\n\n### |\Z)",
    re.DOTALL | re.MULTILINE,
)
EVIDENCE_LOC_RE = re.compile(r"^([\w./-]+\.\w+):(\d+)(?:-\d+)?")
CHECK_RE = re.compile(r"check:\s*(\S+)\s*$")


def parse_cards(text: str):
    cards = []
    for m in CARD_RE.finditer(text):
        d = m.groupdict()
        evidence = d["evidence"].strip()
        loc = EVIDENCE_LOC_RE.match(evidence)
        check_m = CHECK_RE.search(evidence)
        cards.append({
            "paper": d["paper"],
            "headline": d["headline"].strip(),
            "claim": d["claim"].strip(),
            "concern": d["concern"].strip(),
            "evidence": evidence,
            "file": loc.group(1) if loc else None,
            "line_start": int(loc.group(2)) if loc else None,
            "check_script": check_m.group(1) if check_m else None,
        })
    return cards


def load_findings(paper_num: str):
    matches = glob.glob(str(ROOT / "audits" / f"{paper_num}" / "findings.json"))
    if not matches:
        return []
    return json.loads(Path(matches[0]).read_text()).get("findings", [])


def match_finding(card, findings):
    if card["check_script"]:
        hits = [f for f in findings if f.get("check_script") == card["check_script"]]
        if len(hits) == 1:
            return hits[0], "check_script"
    if card["file"] and card["line_start"] is not None:
        hits = [f for f in findings
                if f.get("file") == card["file"] and f.get("line_start") == card["line_start"]]
        if len(hits) == 1:
            return hits[0], "file:line_start"
    return None, None


def main():
    cards = parse_cards(DISPLAY_MD.read_text())
    print(f"Parsed {len(cards)} cards from {DISPLAY_MD.relative_to(ROOT)}\n")
    n_matched = 0
    for card in cards:
        findings = load_findings(card["paper"])
        finding, method = match_finding(card, findings)
        print("=" * 100)
        print(f"Paper #{card['paper']}  (matched by {method})" if finding
              else f"Paper #{card['paper']}  *** UNMATCHED ***")
        print("-" * 100)
        print(f"DISPLAY headline : {card['headline']}")
        if finding:
            print(f"FINDING title    : {finding.get('title')}")
            print(f"FINDING id       : {finding.get('id')}")
        print()
        if finding:
            claim_match = "VERBATIM" if card["claim"] == finding.get("claim") else \
                          ("PREFIX MATCH (card is truncated)"
                           if finding.get("claim", "").startswith(card["claim"].split(" [...] ")[0])
                           else "DIFFERS")
            print(f"DISPLAY claim [{claim_match}]:\n  {card['claim']}")
            print(f"FINDING claim:\n  {finding.get('claim')}")
            print()
            concern_match = "VERBATIM" if card["concern"] == finding.get("concern") else "DIFFERS"
            print(f"DISPLAY concern [{concern_match}]:\n  {card['concern']}")
            print(f"FINDING concern:\n  {finding.get('concern')}")
            n_matched += 1
        else:
            print(f"DISPLAY claim:\n  {card['claim']}")
            print(f"DISPLAY concern:\n  {card['concern']}")
            print(f"\n  no finding in audits/{card['paper']}/findings.json matched "
                  f"check_script={card['check_script']!r} or "
                  f"file:line_start={card['file']}:{card['line_start']}")
        print()
    print("=" * 100)
    print(f"{n_matched}/{len(cards)} cards matched to a specific committed finding. "
          "The bold headline on each card is always a hand-written summary, not a "
          "verbatim quote; Claim/Concern are verbatim or a marked truncation of the "
          "underlying finding where matched.")


if __name__ == "__main__":
    main()
