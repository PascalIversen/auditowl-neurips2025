#!/usr/bin/env python3
"""Checks whether any author code repository exists in this audit folder.

Supports finding `no-code-repository`: the audit folder contains no `code/`
directory and `code_links.txt` lists no repositories, so there is nothing to
audit beyond the paper itself.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

lines = []
code_dir = os.path.join(ROOT, "code")
lines.append(f"code/ directory exists: {os.path.isdir(code_dir)}")
if os.path.isdir(code_dir):
    lines.append("code/ contents: " + str(os.listdir(code_dir)))

# count code source files anywhere under the audit folder (excluding _audit_code)
exts = (".py", ".ipynb", ".cpp", ".cu", ".sh", ".yaml", ".yml", ".cfg", ".toml")
found = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    if "_audit_code" in dirpath:
        continue
    for f in filenames:
        if f.endswith(exts):
            found.append(os.path.relpath(os.path.join(dirpath, f), ROOT))
lines.append(f"author source files found ({exts}): {len(found)}")
for f in found:
    lines.append("  " + f)

# inspect code_links.txt
cl = os.path.join(ROOT, "code_links.txt")
repo_links = []
if os.path.isfile(cl):
    with open(cl) as fh:
        for ln in fh:
            s = ln.strip()
            if s and not s.startswith("#"):
                repo_links.append(s)
lines.append(f"non-comment repository links in code_links.txt: {len(repo_links)}")
for s in repo_links:
    lines.append("  " + s)

report = "\n".join(lines) + "\n"
print(report, end="")
with open(os.path.join(OUT, "repo_presence.txt"), "w") as fh:
    fh.write(report)
