"""Checks Table 3's composite 'Accuracy = (MMLU + 10*MT-Bench)/2' against reported values.

Supports finding `table3-no-code` only as an arithmetic-consistency sanity check on the
paper's own reported numbers (the only deterministic check possible without code).
Read-only; writes a CSV to out/.
"""
import csv, os

# (label, MMLU, MT-Bench, reported Accuracy) from paper Table 3 (paper_text.txt:482-508)
rows = [
    ("Baseline", 80.73, 8.87, 84.71),
    ("Step1",    80.64, 8.72, 83.92),
    ("Step2",    80.29, 8.54, 82.84),
    ("Step3",    80.39, 8.30, 81.69),
    ("Step4",    79.98, 8.25, 81.24),
]

outdir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "table3_accuracy_check.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "mmlu", "mtbench", "reported_acc", "computed_acc", "abs_diff", "match_2dp"])
    for label, mmlu, mt, rep in rows:
        comp = (mmlu + 10 * mt) / 2
        diff = abs(comp - rep)
        w.writerow([label, mmlu, mt, rep, round(comp, 3), round(diff, 3), diff < 0.005])
        print(f"{label:9s} computed={comp:.3f} reported={rep:.2f} diff={diff:.3f}")
print(f"\nwrote {out}")
