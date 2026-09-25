"""Checks that bon/utils/power_law.py and the symbols imported by
experiments/2_plot_powerlaw.ipynb are absent (supports finding
powerlaw-notebook-dead-import). Read-only."""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "code" / "jplhughes__bon-jailbreaking"
OUT = Path(__file__).resolve().parent / "out" / "missing_powerlaw_module.txt"

lines = []

mod = REPO / "bon" / "utils" / "power_law.py"
lines.append(f"bon/utils/power_law.py exists: {mod.exists()}")

# Extract the import statement from the notebook
nb = json.loads((REPO / "experiments" / "2_plot_powerlaw.ipynb").read_text())
imported = set()
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "from bon.utils.power_law import" in src:
        # parse it
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # cell may contain non-python lines; isolate the import block
            block = src[src.index("from bon.utils.power_law import"):]
            block = block[: block.index(")") + 1]
            tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "bon.utils.power_law":
                for n in node.names:
                    imported.add(n.name)

lines.append(f"symbols imported from bon.utils.power_law: {sorted(imported)}")

# Search whole repo (excluding .git) for any definition of these symbols
defs_found = {s: [] for s in imported}
for py in REPO.rglob("*.py"):
    if ".git" in py.parts:
        continue
    try:
        tree = ast.parse(py.read_text())
    except Exception:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign)):
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for nm in names:
                if nm in defs_found:
                    defs_found[nm].append(str(py.relative_to(REPO)))

for s in sorted(imported):
    lines.append(f"  symbol {s!r} defined in repo: {defs_found[s] or 'NOWHERE'}")

result = "\n".join(lines)
OUT.write_text(result + "\n")
print(result)
