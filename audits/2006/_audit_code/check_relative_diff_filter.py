"""Extract the exact filtering condition the '32% gap closed' average uses in the figures
notebook (cell 8). Supports finding relative-diff-filtering-32pct. Read-only."""
import os, json, csv

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "technion-cs-nlp__vlm-circuits-analysis"))
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)

nb = json.load(open(os.path.join(REPO, "figures_and_results_processing.ipynb")))
cell8 = "".join(nb["cells"][8]["source"])

# Pull the three relevant lines verbatim (stripped of cell indentation for reporting)
relevant = [ln.strip() for ln in cell8.splitlines()
            if "relative_diff = (bp_best_acc" in ln
            or "if 0 < relative_diff" in ln
            or "relative_diffs.append" in ln
            or 'Average relative diff' in ln]

csv_path = os.path.join(OUT, "relative_diff_filter.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["finding", "evidence_line"])
    for ln in relevant:
        w.writerow(["relative-diff-filtering-32pct", ln])

print("Cell 8 average-gap-closing logic (verbatim, indentation stripped):")
for ln in relevant:
    print("  ", ln)
print("\nThe averaged list only receives entries with 0 < relative_diff <= 1.0;")
print("(model,task) pairs where back-patching did not help (<=0) or where clean_v>=clean_l")
print("(no positive gap / denominator <=0) are excluded before np.mean.")
print(f"\nWrote {csv_path}")
