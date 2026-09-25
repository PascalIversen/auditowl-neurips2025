"""Checks which result/data inputs the plotting scripts need vs what is committed.
Supports findings: missing results .npz, missing 18/19 datasets. Read-only."""
import os, re, glob

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "Rick7117__aesp-local-pagerank")

# Datasets present in repo
ds_dir = os.path.join(REPO, "datasets")
present_datasets = sorted(os.listdir(ds_dir)) if os.path.isdir(ds_dir) else []

# results .npz files anywhere under datasets/
result_npz = glob.glob(os.path.join(ds_dir, "**", "results", "*.npz"), recursive=True)

# Plot scripts referencing a results dir
plot_scripts = sorted(glob.glob(os.path.join(REPO, "plot_*.py")) + glob.glob(os.path.join(REPO, "com_dblp_plot_*.py")))
needs_results = []
for p in plot_scripts:
    txt = open(p).read()
    # active (non-commented) lines referencing 'results'
    for i, line in enumerate(txt.splitlines(), 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        if "results" in s and (".npz" in s or "results_dir" in s or "/results" in s or "'results'" in s):
            needs_results.append((os.path.basename(p), i, s[:90]))
            break

out = os.path.join(os.path.dirname(__file__), "out", "artefact_inputs.txt")
with open(out, "w") as f:
    f.write("Datasets directories present in repo (of 19 used in paper Table 2):\n")
    f.write("  " + ", ".join(present_datasets) + "\n")
    f.write(f"  count = {len(present_datasets)}\n\n")
    f.write(f"Result .npz files committed under datasets/**/results/: {len(result_npz)}\n")
    for r in result_npz:
        f.write("  " + os.path.relpath(r, REPO) + "\n")
    f.write("\nPlot scripts that consume a results/ directory (active line):\n")
    for name, i, s in needs_results:
        f.write(f"  {name}:{i}  {s}\n")

print(open(out).read())
print("Output written to", out)
