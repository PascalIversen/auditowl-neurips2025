"""Checks whether any author code / repo artefacts exist for paper 3437 (supports `no-code-released`)."""
import os, glob, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# Anything that looks like source/data/notebooks under the paper folder.
code_exts = (".py", ".ipynb", ".R", ".r", ".m", ".jl", ".cpp", ".c", ".sh", ".yaml", ".yml", ".txt", ".csv")
found = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    # skip the audit's own scratch folder
    if os.path.basename(dirpath) in ("_audit_code", "out"):
        dirnames[:] = []
        continue
    for f in filenames:
        if f.endswith(code_exts) and f not in ("metadata.txt", "code_links.txt", "paper_text.txt", "findings.json"):
            found.append(os.path.relpath(os.path.join(dirpath, f), ROOT))

code_dir = os.path.join(ROOT, "code")
links_path = os.path.join(ROOT, "code_links.txt")
with open(links_path) as fh:
    links_body = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.strip().startswith("#")]

result = {
    "code_dir_exists": os.path.isdir(code_dir),
    "cloned_repos_listed_in_code_links": links_body,
    "candidate_code_or_data_files_in_paper_folder": found,
}
with open(os.path.join(OUT, "code_presence.json"), "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result, indent=2))
