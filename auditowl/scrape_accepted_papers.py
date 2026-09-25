#!/usr/bin/env python3
"""Scrape the list of NeurIPS 2025 Main Conference Track papers.

Source: https://papers.nips.cc/paper_files/paper/2025

Each paper on that index page is a <li> containing an <a> whose href points to
an abstract page. The track is encoded in the href filename, e.g.
    .../<hash>-Abstract-Conference.html                  -> Main Conference Track
    .../<hash>-Abstract-Datasets_and_Benchmarks_Track.html
    .../<hash>-Abstract-Position.html

We keep only "-Abstract-Conference.html" (the main track) by default.

Usage:
    pip install requests beautifulsoup4
    python scrape_accepted_papers.py                  # -> neurips_2025_main_track.csv
    python scrape_accepted_papers.py --all            # include every track
    python scrape_accepted_papers.py -o papers.csv

Output columns: title, authors, track, url
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://papers.nips.cc"
INDEX_URL = f"{BASE_URL}/paper_files/paper/2025"

# Map the track suffix in the abstract filename to a human-readable label.
TRACK_PATTERN = re.compile(r"-Abstract-(?P<track>[A-Za-z0-9_]+)\.html$")
TRACK_LABELS = {
    "Conference": "Main Conference Track",
    "Datasets_and_Benchmarks_Track": "Datasets and Benchmarks Track",
    "Position": "Position Paper Track",
}


def fetch_index(url: str = INDEX_URL) -> str:
    resp = requests.get(url, headers={"User-Agent": "neurips-scraper/1.0"}, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_papers(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    papers: list[dict] = []
    seen: set[str] = set()

    for a in soup.select('a[href*="-Abstract-"]'):
        href = a.get("href", "")
        m = TRACK_PATTERN.search(href)
        if not m:
            continue

        url = urljoin(BASE_URL, href)
        if url in seen:
            continue
        seen.add(url)

        track_key = m.group("track")
        track = TRACK_LABELS.get(track_key, track_key.replace("_", " "))

        title = a.get_text(strip=True)

        # Authors live in the sibling <span class="paper-authors">.
        authors = ""
        container = a.find_parent(class_="paper-content") or a.parent
        if container is not None:
            span = container.find("span", class_="paper-authors")
            if span is not None:
                authors = span.get_text(strip=True)

        papers.append(
            {"title": title, "authors": authors, "track": track, "url": url}
        )

    return papers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", default="neurips_2025_main_track.csv", help="output CSV path"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include all tracks (default: main conference track only)",
    )
    args = parser.parse_args()

    print(f"Fetching {INDEX_URL} ...", file=sys.stderr)
    html = fetch_index()
    papers = parse_papers(html)

    if not args.all:
        papers = [p for p in papers if p["track"] == "Main Conference Track"]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "authors", "track", "url"])
        writer.writeheader()
        writer.writerows(papers)

    print(f"Wrote {len(papers)} papers to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
