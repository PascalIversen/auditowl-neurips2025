"""Checks that the audited paper folder contains no author code/scripts (supports missing-author-code finding)."""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

code_exts = {".py", ".ipynb", ".sh", ".m", ".cpp", ".c", ".java", ".js", ".ts"}
found_code = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    # skip the audit's own helper folder
    if "_audit_code" in dirpath:
        continue
    for f in filenames:
        ext = os.path.splitext(f)[1].lower()
        if ext in code_exts:
            found_code.append(os.path.relpath(os.path.join(dirpath, f), ROOT))

has_code_dir = os.path.isdir(os.path.join(ROOT, "code"))

result = {
    "paper_folder": os.path.basename(ROOT),
    "code_dir_exists": has_code_dir,
    "author_code_files_found": found_code,
    "all_top_level_files": sorted(os.listdir(ROOT)),
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "repo_contents.json")
with open(out_path, "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result, indent=2))
