"""Checks whether any author code/data artefact exists in the audit folder.

Supports the `no-code-released` missing finding: the paper's NeurIPS checklist
promises source code, but the audit folder contains only the paper and metadata.
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the paper audit folder

# Things a code release would plausibly contain.
code_markers = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    # skip our own audit scaffolding
    if "_audit_code" in dirpath:
        continue
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext in {".py", ".ipynb", ".sh", ".yaml", ".yml", ".cpp", ".c",
                   ".java", ".js", ".ts", ".rs", ".go"} or fn in {
                   "requirements.txt", "environment.yml", "setup.py",
                   "pyproject.toml", "Dockerfile", "README.md"}:
            code_markers.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))

top_level = sorted(os.listdir(ROOT))

result = {
    "audit_folder": ROOT,
    "top_level_entries": top_level,
    "code_dir_present": os.path.isdir(os.path.join(ROOT, "code")),
    "code_markers_found": code_markers,
    "n_code_markers": len(code_markers),
    "verdict": "NO_AUTHOR_CODE" if not code_markers else "CODE_PRESENT",
}

out = os.path.join(HERE, "out", "code_presence.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
