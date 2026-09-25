#!/usr/bin/env python3
"""Checks which dataset loader modules referenced by data_utils.py actually exist.
Supports finding `missing-dataset-loaders` (3 of 4 Table-1 datasets cannot run).
Read-only; writes report to out/dataset_modules.txt."""
import os
import re

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "Legenddddd__MEL")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out", "dataset_modules.txt")

du = os.path.join(REPO, "dataloader", "data_utils.py")
with open(du) as f:
    text = f.read()

# parse `import dataloader.X.Y as Dataset`
imports = re.findall(r"import\s+(dataloader\.[\w.]+)\s+as\s+Dataset", text)

lines = []
lines.append(f"data_utils.py references {len(imports)} dataset loader modules:")
for mod in imports:
    rel = mod.replace(".", os.sep) + ".py"
    path = os.path.join(REPO, rel)
    exists = os.path.isfile(path)
    lines.append(f"  {'EXISTS ' if exists else 'MISSING'}  {mod}  ->  {rel}")

# which datasets are configured in set_up_datasets
datasets = re.findall(r"args\.dataset == '([\w]+)'", text)
lines.append("")
lines.append(f"set_up_datasets dataset keys: {sorted(set(datasets))}")

missing = [m for m in imports if not os.path.isfile(os.path.join(REPO, m.replace('.', os.sep) + '.py'))]
lines.append("")
lines.append(f"MISSING modules: {len(missing)} of {len(imports)}")
for m in missing:
    lines.append(f"  - {m}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
