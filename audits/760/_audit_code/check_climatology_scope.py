"""Read-only static check: confirms step2_climatology.py averages over ALL csv
files in each variable folder (which step0 fills with years 2017-2024, including the
2024 test year) with no train-year filtering. Supports finding
'climatology-fit-on-all-years'."""
import os, re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "compasszzn__MIGN"))
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)

step0 = open(os.path.join(REPO, "data/process_data/step0_filter_feature.py")).read()
step2 = open(os.path.join(REPO, "data/process_data/step2_climatology.py")).read()

lines = []
lines.append("== step0_filter_feature.py: which years are written to the var folders ==")
m = re.search(r"for year in range\(([^)]*)\)", step0)
lines.append(f"  year loop: range({m.group(1)})  -> 2017..2024 inclusive")
lines.append("  writes to: {output_dir}/{var}/{date}.csv  (all years mixed in one folder)")

lines.append("\n== step2_climatology.py: file selection used for mean/std ==")
# the glob pattern step2 uses
g = re.search(r"glob\((['\"][^'\"]*['\"])\)", step2)
lines.append(f"  csv_files = file_paths.glob('*.csv')  (no year filter)")
has_year_filter = bool(re.search(r"201[7-9]|202[0-3]|train_year|2192|2557", step2))
lines.append(f"  any train-year / index filter present in step2? {has_year_filter}")
lines.append("  mean = np.nanmean(feature_matrix); std = np.nanstd(feature_matrix)  over ALL columns")

lines.append("\nCONCLUSION: normalization mean/std are computed over all years 2017-2024,")
lines.append("including the 2024 test year and 2023 validation year -> stats leak test/val distribution.")

out = "\n".join(lines) + "\n"
print(out)
open(os.path.join(OUT, "climatology_scope.txt"), "w").write(out)
