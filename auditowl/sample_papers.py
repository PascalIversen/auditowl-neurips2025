#!/usr/bin/env python3
"""Produce a reproducible random sample of paper row numbers.

Draws 500 distinct random numbers in the range of the NeurIPS 2025 main-track
paper list (1-based row numbers, 1..N where N = number of papers in the CSV)
and writes them, in draw order, to list.csv.

A fixed RNG seed makes the sample reproducible: re-running yields the same list.

Usage:
    python sample_papers.py
    python sample_papers.py --csv neurips_2025_main_track.csv -o list.csv \
        --count 500 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random


def count_papers(csv_path: str) -> int:
    with open(csv_path, newline="", encoding="utf-8") as f:
        # Subtract the header row.
        return sum(1 for _ in csv.reader(f)) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="neurips_2025_main_track.csv")
    parser.add_argument("-o", "--output", default="list.csv")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = count_papers(args.csv)
    if args.count > n:
        raise SystemExit(f"Cannot draw {args.count} distinct numbers from 1..{n}")

    rng = random.Random(args.seed)
    sample = rng.sample(range(1, n + 1), args.count)  # distinct, in draw order

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_number"])
        for num in sample:
            writer.writerow([num])

    print(f"Wrote {len(sample)} numbers (range 1..{n}, seed {args.seed}) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
