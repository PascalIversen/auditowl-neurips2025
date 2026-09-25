#!/usr/bin/env python3
"""Read-only check: does the cloned repo contain ANY of the paper-specific
artefacts (BINOTEARS, multivariate-Bernoulli binary data simulation, the
cross-entropy NOTEARS-MLP variant, Sachs data, CPDAG/MEC conversion, the
DAGMA higher-order augmentation, experiment drivers for Figs 1-6 / the Sachs
table)? Supports finding `binotears-code-absent`.

Outputs a CSV summarising grep hits per paper term, plus a file inventory.
"""
import csv
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "code" / "xunzheng__notears"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Terms that MUST appear if this were the paper's BINOTEARS code.
TERMS = [
    "binotears", "multibernoulli", "multivariate bernoulli", "bernoulli",
    "sachs", "dagma", "fges", "quasi-mcp", "quasi_mcp", "mcp",
    "cross_entropy", "cross-entropy", "cpdag", "markov equivalence",
    "interaction", "topological order", "Phi(", "extended feature",
    "second order", "higher order", "binary data", "logistic loss",
]

rows = []
py_files = sorted(p for p in REPO.rglob("*.py") if ".git" not in p.parts)
all_text = ""
for p in py_files:
    all_text += "\n" + p.read_text(encoding="utf-8", errors="ignore").lower()

for term in TERMS:
    hits = len(re.findall(re.escape(term.lower()), all_text))
    rows.append({"term": term, "hits_in_py_source": hits})

with open(OUT / "paper_term_hits.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["term", "hits_in_py_source"])
    w.writeheader()
    w.writerows(rows)

# Inventory of every tracked file.
tracked = subprocess.run(
    ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True
).stdout.strip().splitlines()
with open(OUT / "repo_inventory.txt", "w") as f:
    f.write("\n".join(tracked) + "\n")

# nonlinear.py loss check
nl = (REPO / "notears" / "nonlinear.py").read_text()
uses_squared = "squared_loss" in nl and "loss = squared_loss" in nl
uses_xent = bool(re.search(r"cross.?entropy|binary_cross|BCE", nl, re.I))

print("=== paper_term_hits.csv ===")
for r in rows:
    print(f"  {r['term']:<25} {r['hits_in_py_source']}")
print(f"\nTracked files: {len(tracked)}")
print(f"nonlinear.py uses squared_loss as the training loss : {uses_squared}")
print(f"nonlinear.py uses any cross-entropy/BCE loss         : {uses_xent}")
print("\nWritten: out/paper_term_hits.csv, out/repo_inventory.txt")
