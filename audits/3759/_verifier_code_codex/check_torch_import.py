"""AST/regex check: scripts that USE torch.* but never import torch (runtime NameError). Supports finding: missing-torch-import."""
import os, re, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "tianxiao18__Lfp2vec"))
SCRIPTS = os.path.join(REPO, "script")
rows = []
for fn in sorted(os.listdir(SCRIPTS)):
    if not fn.endswith(".py"):
        continue
    src = open(os.path.join(SCRIPTS, fn)).read()
    uses_torch = bool(re.search(r"(?<![\w.])torch\.", src))
    imports_torch = bool(re.search(r"^\s*import torch(\s|$|,)", src, re.M)) or bool(re.search(r"from torch", src))
    if uses_torch and not imports_torch:
        # count occurrences and first lineno
        line = next((i+1 for i, l in enumerate(src.splitlines()) if re.search(r"(?<![\w.])torch\.", l)), None)
        rows.append({"file": f"script/{fn}", "uses_torch": True, "imports_torch": False,
                     "first_torch_use_line": line,
                     "first_torch_use": src.splitlines()[line-1].strip() if line else None})

print(json.dumps(rows, indent=2))
with open(os.path.join(os.path.dirname(__file__), "out", "torch_import.json"), "w") as f:
    json.dump(rows, f, indent=2)
