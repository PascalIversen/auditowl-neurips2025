#!/usr/bin/env python3
"""Checks whether any author code/data artefact exists for paper 2941; supports finding `no-code-repo`.

Verifies: (1) no code/ dir in the audit folder, (2) code_links.txt lists no
cloned repo, (3) paper_text.txt contains no resolvable code URL. Writes a CSV
summary to out/ so a reviewer can re-run and see the same result.
"""
import csv
import os
import re

AUDIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

rows = []

# 1. code/ directory present?
code_dir = os.path.join(AUDIT_DIR, "code")
rows.append(("code_dir_exists", os.path.isdir(code_dir)))

# 2. any python/yaml/notebook anywhere under the audit folder (excluding _audit_code)?
code_files = []
for root, dirs, files in os.walk(AUDIT_DIR):
    if "_audit_code" in root:
        continue
    for f in files:
        if f.endswith((".py", ".ipynb", ".yaml", ".yml")):
            code_files.append(os.path.join(root, f))
rows.append(("author_code_files_found", len(code_files)))

# 3. code_links.txt: any non-comment, non-blank line (i.e. an actual link)?
links_path = os.path.join(AUDIT_DIR, "code_links.txt")
link_lines = []
if os.path.isfile(links_path):
    with open(links_path) as fh:
        for ln in fh:
            s = ln.strip()
            if s and not s.startswith("#"):
                link_lines.append(s)
rows.append(("code_links_non_comment_lines", len(link_lines)))

# 4. any URL in paper_text.txt?
paper_txt = os.path.join(AUDIT_DIR, "paper_text.txt")
urls = []
if os.path.isfile(paper_txt):
    with open(paper_txt) as fh:
        text = fh.read()
    urls = re.findall(r"https?://[^\s)]+", text)
rows.append(("urls_in_paper_text", len(urls)))
rows.append(("urls_list", ";".join(urls) if urls else ""))

out_csv = os.path.join(OUT, "code_present.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["check", "value"])
    for k, v in rows:
        w.writerow([k, v])

for k, v in rows:
    print(f"{k}: {v}")
print(f"\nwrote {out_csv}")
