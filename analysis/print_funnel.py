#!/usr/bin/env python3
"""Print the headline reproducibility funnel.

Reads the auditable per-paper coverage data
(analysis/data/coverage_status.json, produced by figures/fig03_code_coverage.py) and
derives every count here, so the funnel stays in sync with the audit set.

A paper's "traced artifacts" are the result-artifacts (tables, columns, metrics,
ablations) the auditor chose to check. A row counts as having "producing code"
when it was NOT flagged missing — i.e. some code that produces it was located.

Run:  python analysis/print_funnel.py
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "coverage_status.json"


def main():
    cov = json.loads(DATA.read_text())
    real = [p for p in cov["papers"] if not p.get("is_nocode")]  # have a coverage table

    funnel = [
        (cov["n_papers"],                              "empirical papers"),
        (len(real),                                    "source code present"),
        (sum(p["missing_frac"] < 0.5 for p in real),   "producing code found for >50% of traced artifacts"),
        (sum(p["missing_frac"] == 0.0 for p in real),  "producing code found for every traced artifact"),
    ]
    for count, label in funnel:
        print(f"{count:4d}  {label}")


if __name__ == "__main__":
    main()
