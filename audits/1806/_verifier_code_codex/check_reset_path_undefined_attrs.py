#!/usr/bin/env python3
"""Static check: confirm `self.tol` and `self.subsampling_option` are read in
bayesqp.py but never assigned anywhere in src/ (supports finding
`reset-path-undefined-attrs`). Read-only AST/grep over the repo."""
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent / "code" / "brunzema__bayesqp" / "src" / "bayesqp"
OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

attrs = ["tol", "subsampling_option"]
reads = {a: [] for a in attrs}
writes = {a: [] for a in attrs}

for py in sorted(REPO.glob("*.py")):
    src = py.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in attrs:
                if isinstance(node.ctx, ast.Store):
                    writes[node.attr].append((py.name, node.lineno))
                else:
                    reads[node.attr].append((py.name, node.lineno))

lines = []
for a in attrs:
    lines.append(f"self.{a}: reads={reads[a]}  writes={writes[a]}")
    verdict = "UNDEFINED (read, never assigned)" if reads[a] and not writes[a] else "ok"
    lines.append(f"  -> verdict: {verdict}")

report = "\n".join(lines) + "\n"
print(report, end="")
(OUT / "reset_path_undefined_attrs.txt").write_text(report)
