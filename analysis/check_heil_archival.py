#!/usr/bin/env python3
"""Verifies the Discussion's Heil Bronze-tier archival count. Greps every
audited paper's raw text for archive-DOI keywords (fully automatic); the only
manual step is distinguishing the paper's own code-archive DOI from an
ordinary bibliography citation, documented inline below.

audits/*/paper_text.txt is not redistributed in the public release (same
non-redistribution policy as audits/*/code/); re-fetch a paper's text via the
abstract_url in its metadata.txt to re-run this check independently.

Run: python analysis/check_heil_archival.py
"""
import glob
import re

KEYWORDS = re.compile(
    r"codeocean|code ocean|software heritage|\bswh\b|osf\.io|figshare|"
    r"dryad|zenodo|doi\.org|10\.5281|10\.17605", re.I)

# Manually reviewed: which raw hits are the paper's OWN code on an immutable
# DOI archive, vs. an ordinary citation to someone else's work.
GENUINE_SELF_ARCHIVAL = {
    "1729":
        "Zenodo DOI 10.5281/zenodo.17424409, own RPOMDP_Benchmark code",
    "3101":
        "Zenodo DOI 10.5281/zenodo.17249361, cited alongside own GitHub repo",
}


def main():
    raw_hits = sorted(
        p.split("/")[1] for p in glob.glob("audits/*/paper_text.txt")
        if KEYWORDS.search(open(p, encoding="utf-8", errors="ignore").read()))
    genuine = [p for p in raw_hits if p in GENUINE_SELF_ARCHIVAL]
    print(f"{len(raw_hits)} papers mention an archive-DOI keyword anywhere in text")
    print(f"{len(genuine)} are the paper's own code on an immutable DOI archive:")
    for p in genuine:
        print(f"  {p}: {GENUINE_SELF_ARCHIVAL[p]}")
    assert len(raw_hits) == 14 and len(genuine) == 2, \
        "raw hit count changed -- re-review the new/dropped hits before trusting this"


if __name__ == "__main__":
    main()
