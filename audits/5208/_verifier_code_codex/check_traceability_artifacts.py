"""Checks which result-producing/aggregation artefacts exist in the RF-Agent repo.

Supports findings: missing-aggregation-script, reward-function-coverage,
and the absence of stored result logs for Table 1 / Figs 3-6.
Read-only. Writes a summary CSV to out/.
"""
import os, glob, csv

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "deng-ai-lab__RF-Agent")
REPO = os.path.abspath(REPO)
RF = os.path.join(REPO, "RF_Agent")

rows = []

def add(name, value):
    rows.append((name, value))

# 1. Reward-function coverage for the proposed method
isaac_rf = sorted(glob.glob(os.path.join(RF, "reward_functions/isaac/RFAgent/*.py")))
bidex_rf = sorted(glob.glob(os.path.join(RF, "reward_functions/bidex/RFAgent/*.py")))
add("isaac_RFAgent_reward_funcs", len(isaac_rf))   # expect 14 (7 tasks x 2 models)
add("bidex_RFAgent_reward_funcs", len(bidex_rf))   # expect 10 tasks

# 2. Aggregation / normalization / plotting scripts that COMPUTE reported numbers
#    (Table 1 "Avg norm score"; Figs 3,4,5,6). Search whole RF_Agent tree.
def grep_count(pattern_substrings, exclude_dir="reward_functions"):
    hits = []
    for root, _, files in os.walk(RF):
        if exclude_dir in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for s in pattern_substrings:
                if s in txt:
                    hits.append((os.path.relpath(p, REPO), s))
                    break
    return hits

norm_hits = grep_count(["norm_score", "normalized", "Human - Sparse", "(Method", "human_normalized"])
add("normalization_score_script_hits", str(norm_hits))

savefig_hits = grep_count(["savefig", "plt.plot", "plt.bar", "plt.errorbar"])
add("figure_savefig_hits", str(savefig_hits))

# 3. Stored result logs (json/csv/npy) outside reward_functions
result_logs = []
for ext in ("*.json", "*.csv", "*.npy", "*.pkl"):
    for p in glob.glob(os.path.join(RF, "**", ext), recursive=True):
        if "reward_functions" in p:
            continue
        result_logs.append(os.path.relpath(p, REPO))
add("stored_result_logs", str(result_logs))

# 4. Metric reduction used by the evaluation script (max over checkpoints)
test_py = open(os.path.join(RF, "test.py"), encoding="utf-8").read()
add("test_py_uses_max_consecutive_successes",
    "max(tensorboard_logs['consecutive_successes'])" in test_py)

# 5. Seeds: search train_seed default and final-eval seed range
cfg = open(os.path.join(RF, "cfg/config_rf_agent.yaml"), encoding="utf-8").read()
add("config_train_seed_line", [l.strip() for l in cfg.splitlines() if "train_seed:" in l])
add("rfagent_eval_uses_seed_range_0_to_num_eval",
    "for i in range(cfg.num_eval):" in open(os.path.join(RF, "test.py")).read())

outp = os.path.join(os.path.dirname(__file__), "out", "traceability_artifacts.csv")
os.makedirs(os.path.dirname(outp), exist_ok=True)
with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["check", "value"])
    for r in rows:
        w.writerow(r)

for r in rows:
    print(f"{r[0]}: {r[1]}")
print(f"\nWrote {outp}")
