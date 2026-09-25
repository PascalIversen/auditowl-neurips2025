"""Checks the paper claim 'all p-values resulted lower than 10^-3' for the
volume-ratio Welch/t-tests against the shipped tables/volume_ratio.csv
(supports finding volume-pvalue-mismatch and welch-test-mislabeled)."""
import csv, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "alessiomarta__transformers_equivalence_classes")
path = os.path.join(REPO, "tables", "volume_ratio.csv")

rows = list(csv.DictReader(open(path)))
out_lines = []
violations = []
for r in rows:
    p = float(r["volume_ratio_test_pvalue"])
    ratio = float(r["volume_ratio_test"])
    flag = "" if p < 1e-3 else "  <-- NOT < 1e-3"
    if p >= 1e-3:
        violations.append((r["dataset_name"], r["delta_multiplier"], p))
    out_lines.append(f"{r['dataset_name']:12s} delta={r['delta_multiplier']:>4} "
                     f"stat={ratio:>8} p={p}{flag}")

ratios = [float(r["volume_ratio_test"]) for r in rows]
out = []
out.append("Per-row volume-ratio test p-values (paper claims all < 1e-3):")
out += out_lines
out.append("")
out.append(f"VIOLATIONS of 'all p < 1e-3': {len(violations)}")
for v in violations:
    out.append(f"  {v}")
out.append("")
out.append(f"Reported volume_ratio_test statistics range: {min(ratios)} .. {max(ratios)}")
out.append("Paper text claims SiMExp explores '...bigger ... by an order of 10^1'.")
out.append(f"Statistics include values >= 100 (order 10^2): "
           f"{[r for r in ratios if r >= 100]}")
text = "\n".join(out)
print(text)
with open(os.path.join(os.path.dirname(__file__), "out", "volume_pvalues.txt"), "w") as f:
    f.write(text + "\n")
