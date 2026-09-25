"""Checks for presence of any author code/data artefacts in the audit folder.
Supports findings: no-code-released, fig1-no-compute-script."""
import os, json
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_exts = {'.py', '.ipynb', '.m', '.r', '.jl', '.cpp', '.c', '.sh'}
found = []
for dirpath, dirnames, filenames in os.walk(root):
    if '_audit_code' in dirpath:
        continue
    for f in filenames:
        ext = os.path.splitext(f)[1].lower()
        if ext in code_exts:
            found.append(os.path.relpath(os.path.join(dirpath, f), root))
# code_links.txt content
cl = os.path.join(root, 'code_links.txt')
with open(cl) as fh:
    links = [l for l in fh if l.strip() and not l.strip().startswith('#')]
result = {
    'code_dir_exists': os.path.isdir(os.path.join(root, 'code')),
    'source_files_found': found,
    'cloned_repos_listed_in_code_links': links,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'code_presence.json')
with open(out, 'w') as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result, indent=2))
