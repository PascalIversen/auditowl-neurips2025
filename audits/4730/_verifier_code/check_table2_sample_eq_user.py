"""Checks whether Table 2's Sample column equals the User column for both Amazon
subsets (supports finding 'amazon-sample-equals-user-count'). Read-only; uses only
the numbers transcribed from paper Table 2 (paper.pdf, Appendix A.4)."""
import csv
import os

# Transcribed verbatim from paper.pdf Table 2 (Appendix A.4).
rows = [
    {"dataset": "Books", "user": 603668, "item": 367982, "category": 1600, "sample": 603668},
    {"dataset": "Electronics", "user": 192403, "item": 63001, "category": 801, "sample": 192403},
]

out_path = os.path.join(os.path.dirname(__file__), "out", "table2_sample_eq_user.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "user", "item", "sample", "sample_eq_user", "samples_per_user"])
    for r in rows:
        eq = r["sample"] == r["user"]
        spu = r["sample"] / r["user"]
        w.writerow([r["dataset"], r["user"], r["item"], r["sample"], eq, round(spu, 4)])
        print(f"{r['dataset']}: sample={r['sample']} user={r['user']} "
              f"sample_eq_user={eq} samples_per_user={spu:.4f}")

print(f"\nWrote {out_path}")
