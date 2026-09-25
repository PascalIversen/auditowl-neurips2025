#!/usr/bin/env python3
"""Checks whether the two author code URLs cited in paper.pdf footnote 3 resolve.
Supports the `missing` findings: author repos are expired/unreachable.
Read-only: only HTTP GET/HEAD, writes a CSV under out/.
"""
import json
import subprocess
import csv
import os

URLS = {
    "mars-steg (Arithmetic+CoinFlip)": "https://anonymous.4open.science/r/mars-steg-A81C/",
    "neurips_encoded_rl (ToM)": "https://anonymous.4open.science/r/neurips_encoded_rl-06F5/",
    "mars-steg API files": "https://anonymous.4open.science/api/repo/mars-steg-A81C/files",
    "neurips_encoded_rl API files": "https://anonymous.4open.science/api/repo/neurips_encoded_rl-06F5/files",
    "TinyZero (dependency, expect alive)": "https://github.com/Jiayi-Pan/TinyZero",
}

out_dir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out_dir, exist_ok=True)
rows = []
for name, url in URLS.items():
    # status code
    code = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-L", url],
        capture_output=True, text=True).stdout.strip()
    # body (first 200 chars)
    body = subprocess.run(
        ["curl", "-s", "-L", url], capture_output=True, text=True).stdout.strip()[:200]
    rows.append({"name": name, "url": url, "http_status": code, "body_snippet": body})
    print(f"{code}  {name}  {url}")
    print(f"     body: {body[:120]}")

with open(os.path.join(out_dir, "repo_urls.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "url", "http_status", "body_snippet"])
    w.writeheader()
    w.writerows(rows)
print("\nWrote", os.path.join(out_dir, "repo_urls.csv"))
