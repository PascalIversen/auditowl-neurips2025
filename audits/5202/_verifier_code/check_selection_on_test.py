"""Confirms the test harness selects the reported checkpoint by TEST-set success rate,
and that training sweeps multiple n_heads / attention-dropout configs not described in the paper.
Supports findings: model-selection-on-test-coverage, undisclosed-hparam-sweep.
Read-only; greps the repo source and writes a summary to out/."""
import os, re, csv

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "Learning-for-Seq-Decision-Making__GABAR-Graph-based-action-ranking-for-planning")
REPO = os.path.abspath(REPO)

def read(p):
    with open(os.path.join(REPO, p)) as f:
        return f.read()

rows = []

# (A) main.py iterates over the three checkpoint-selection metrics and tests each on test set
mainpy = read("main.py")
metrics = re.search(r"all_model_types\s*=\s*\[([^\]]*)\]", mainpy)
rows.append(("main.all_model_types", metrics.group(1).strip() if metrics else "NOT FOUND"))
rows.append(("main.num_models_to_test", re.search(r"num_models_to_test\s*=\s*(\d+)", mainpy).group(1)))

# (B) log_model_metrics selects best by success_rate_with_monitor (= test coverage)
tu = read("ploi/test_utils.py")
# the active (last) definition
idx = tu.rfind("def log_model_metrics")
snippet = tu[idx:idx+4200]
rows.append(("active_log_model_metrics_selects_by",
             "success_rate_with_monitor (TEST coverage)" if
             "metrics.success_rate_with_monitor > best_results[planner_type]" in snippet
             else "OTHER"))
rows.append(("prints_Best_Model_line", "yes" if "Best Model for" in snippet else "no"))

# (C) sweep over n_heads and attention dropout in the canonical runner
runner = read("train_test_scripts/ltp_all_run.sh")
heads = re.search(r"heads=\(([^)]*)\)", runner)
rows.append(("ltp_all_run.sh heads_swept", heads.group(1).strip() if heads else "NOT FOUND"))
ad = re.search(r'"main"\)\s*\n\s*attn_drops=\(([^)]*)\)', runner)
rows.append(("ltp_all_run.sh attn_drops_for_main", ad.group(1).strip() if ad else "NOT FOUND"))

out = os.path.join(os.path.dirname(__file__), "out", "selection_on_test.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["item", "value"]); w.writerows(rows)

for r in rows:
    print(r)
print("\nWrote", out)
