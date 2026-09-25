"""Checks: (1) `openai` used but never defined/imported; (2) placeholder paths; (3) missing dep files. Supports findings undefined-openai-symbol, placeholder-paths, missing-requirements."""
import ast, os, glob, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "FishT0ucher__EVOREFUSE")
REPO = os.path.abspath(REPO)
out = {}

# 1. openai usage vs definition
openai_uses = {}
for f in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
    rel = os.path.relpath(f, REPO)
    src = open(f).read()
    uses = src.count("openai.generate")
    has_import = ("import openai" in src) or ("from openai" in src)
    defines = ("openai =" in src) or ("def openai" in src) or ("class openai" in src)
    if uses:
        openai_uses[rel] = {"openai.generate_calls": uses,
                            "imports_openai_module": has_import,
                            "defines_openai_symbol": defines}
out["openai_generate_usage"] = openai_uses

# 2. placeholder paths "path" / "file" / "file.jsonl"
placeholders = {}
for f in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
    rel = os.path.relpath(f, REPO)
    src = open(f).read()
    cnt = 0
    for needle in ['= "path"', '"path"', '"file"', '"file.jsonl"', "'path'", "model_name_or_path: path", "dataset: file"]:
        cnt += src.count(needle)
    if cnt:
        placeholders[rel] = cnt
out["placeholder_path_hits"] = placeholders

# 3. dependency files present?
dep_files = ["requirements.txt", "setup.py", "environment.yml", "pyproject.toml", "Pipfile"]
out["dependency_files_present"] = {d: os.path.exists(os.path.join(REPO, d)) for d in dep_files}

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "out", "checks.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
