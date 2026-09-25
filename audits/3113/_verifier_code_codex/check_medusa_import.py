"""Checks that models/__init__.py, eval/get_all.py, eval/plot_tree.py import a models/medusa module that is absent in the repo (supports finding missing-medusa-module). Read-only AST scan of code/."""
import os, ast

ROOT = os.path.join(os.path.dirname(__file__), "..", "code", "Xiaohao-Liu__L-MTP")

def medusa_imports(path):
    src = open(path).read()
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.module and "medusa" in n.module:
            out.append((n.lineno, n.module, [a.name for a in n.names]))
    return out

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
lines = []
for f in ["models/__init__.py", "eval/get_all.py", "eval/plot_tree.py"]:
    lines.append(f"{f} -> {medusa_imports(os.path.join(ROOT, f))}")
lines.append(f"models/medusa exists: {os.path.exists(os.path.join(ROOT,'models','medusa'))}")
lines.append(f"models/ contents: {sorted(os.listdir(os.path.join(ROOT,'models')))}")
out = "\n".join(lines)
print(out)
with open(os.path.join(os.path.dirname(__file__), "out", "medusa_import_check.txt"), "w") as fh:
    fh.write(out + "\n")
