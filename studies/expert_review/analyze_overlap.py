#!/usr/bin/env python3
"""Reproduces the "34 findings overlapping between the author-response and
human-review subsets" statistic (paper Section 4.2, repeated in S1.3) from
overlap_comparison.csv: one row per finding that both a paper's corresponding
authors and the human-eval reviewers judged, carrying only the two verdicts
on each axis. No paper identifier, no location, no free text -- built by a
private, non-released join script (author responses carry a paper identifier
per finding, which the Ethics Statement commits to never releasing) and rows
are sorted by content so their order carries no grouping.

Run: python studies/expert_review/analyze_overlap.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "overlap_comparison.csv")

CORR_RANK = {"True": 3, "Partially true": 2, "False": 1,
             "Correct": 3, "Partially correct": 2, "Incorrect": 1}
REL_RANK = {"Relevant": 3, "Minor": 2, "Trivial": 1,
            "Minor point": 2, "Irrelevant": 1}


def main():
    with open(SRC, newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"N overlapping findings = {len(rows)}")

    corr_lower = corr_higher = corr_same = 0
    rel_lower = rel_higher = rel_same = rel_noncomp = 0
    for r in rows:
        ac, hc = r["author_correctness"], r["human_correctness"]
        if CORR_RANK[ac] < CORR_RANK[hc]:
            corr_lower += 1
        elif CORR_RANK[ac] > CORR_RANK[hc]:
            corr_higher += 1
        else:
            corr_same += 1

        ar, hr = r["author_relevance"], r["human_relevance"]
        if ar in REL_RANK and hr in REL_RANK:
            if REL_RANK[ar] < REL_RANK[hr]:
                rel_lower += 1
            elif REL_RANK[ar] > REL_RANK[hr]:
                rel_higher += 1
            else:
                rel_same += 1
        else:
            rel_noncomp += 1

    print(f"Correctness: author graded lower {corr_lower} times, higher {corr_higher} times "
          f"(same {corr_same})")
    n_comparable_rel = rel_lower + rel_higher + rel_same
    print(f"Relevance ({n_comparable_rel} of {len(rows)} carry a verdict on both sides): "
          f"author graded lower {rel_lower} times, higher {rel_higher} times (same {rel_same})")


if __name__ == "__main__":
    main()
