#!/usr/bin/env python3
"""Table S1 - per-category breakdown of the human evaluation.

Reported over all 175 individual judgements, i.e. both passes on the 14
double-rated papers, which is the same pooled body `figures/fig06_expert_review.py`
uses. Categories are the auditor's own four discrepancy classes.

Run: python studies/expert_review/table_s1_by_category.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import human_eval as ah   # noqa: E402
import gpt_eval as ge     # noqa: E402

# order and display names as they appear in the paper
ROWS = [("missing", "Missing artifact"), ("mismatch", "Paper-code mismatch"),
        ("methodology", "Methodology"), ("bug", "Bug")]


def main():
    e1 = ah.load_workbook_findings("expert")
    e2 = ah.load_workbook_findings("expert2")
    J = ge.human_judgements(e1, e2)

    print(f"Table S1 - per-category breakdown over all {len(J)} judgements\n")
    print(f"{'Category':<22}{'Judgments':>10}{'Claim stands':>14}{'>= minor rel.':>15}")
    for key, label in ROWS:
        g = J[J.category == key]
        stands = g.stands.mean() * 100
        minor = g.at_least_minor.mean() * 100
        print(f"{label:<22}{len(g):>10}{stands:>13.0f}%{minor:>14.0f}%")

    total = sum(len(J[J.category == k]) for k, _ in ROWS)
    if total != len(J):
        raise SystemExit(f"category rows sum to {total}, expected {len(J)} -- "
                         "an unrecognised category is present")


if __name__ == "__main__":
    main()
