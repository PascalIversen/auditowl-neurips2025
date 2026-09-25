"""Checks that requirements.txt covers all third-party imports actually used (supports finding: incomplete-requirements)."""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "code" / "Renovamen__ShiftDC"
OUT = Path(__file__).resolve().parent / "out" / "requirements_complete.txt"

STDLIB = {
    "abc","argparse","base64","concurrent","contextlib","datetime","functools","io","json",
    "mimetypes","os","pathlib","random","re","shutil","sys","tempfile","threading","time",
    "typing","urllib","zipfile","collections","itertools","math","warnings","subprocess","ast",
}
# import name -> pip distribution name
DIST = {
    "PIL": "pillow", "dotenv": "python-dotenv", "huggingface_hub": "huggingface-hub",
    "openai": "openai", "numpy": "numpy", "pandas": "pandas", "torch": "torch",
    "transformers": "transformers", "tqdm": "tqdm", "vllm": "vllm", "datasets": "datasets",
}

imports = set()
for p in REPO.rglob("*.py"):
    if ".git" in p.parts:
        continue
    tree = ast.parse(p.read_text(errors="ignore"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports.add(node.module.split(".")[0])

third_party = sorted(i for i in imports if i not in STDLIB and i != "shiftdc")

req_text = (REPO / "requirements.txt").read_text().lower()
listed = set(re.findall(r"^([a-z0-9_\-\.]+)", req_text, flags=re.MULTILINE))

lines = [f"requirements.txt lists: {sorted(listed)}", ""]
missing = []
for imp in third_party:
    dist = DIST.get(imp, imp).lower()
    ok = dist in listed
    if not ok:
        missing.append(imp)
    lines.append(f"{'LISTED ' if ok else 'MISSING'} | import {imp:18s} -> pip '{dist}'")

lines.append("")
lines.append(f"MISSING third-party deps (imported but not in requirements.txt): {missing}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {OUT}")
