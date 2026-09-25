"""Check the 492-task claim and per-category counts against paper Table 2 (supports task-count traceability)."""
import json, os, collections

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "yjyddq__RiOSWorld")
RISK_DIR = os.path.join(REPO, "evaluation_risk_examples")

# Count per-category task json files (each category subfolder holds one json per task)
folder_counts = {}
total_files = 0
for d in sorted(os.listdir(RISK_DIR)):
    p = os.path.join(RISK_DIR, d)
    if os.path.isdir(p):
        n = len([f for f in os.listdir(p) if f.endswith(".json")])
        folder_counts[d] = n
        total_files += n

# test_risk.json index
with open(os.path.join(RISK_DIR, "test_risk.json")) as f:
    index = json.load(f)
index_counts = {k: len(v) for k, v in index.items()}
index_total = sum(index_counts.values())

# Paper Table 2 (Section 3): mapping repo folder -> paper subcategory count
paper_env = {"phishing_web": 56, "phishing_email": 32, "popup": 50, "recaptcha": 33, "account": 33, "induced_text": 50}
# user side: paper groups differently; we just check totals
paper_env_total = 254
paper_user_total = 238
paper_total = 492

out = {
    "folder_counts": folder_counts,
    "total_json_files": total_files,
    "index_counts": index_counts,
    "index_total": index_total,
    "paper_total": paper_total,
    "match_total_files_vs_paper": total_files == paper_total,
    "match_index_vs_paper": index_total == paper_total,
    "folder_vs_index_consistent": folder_counts == index_counts,
    "env_subcat_match": {k: (folder_counts.get(k), paper_env[k], folder_counts.get(k) == paper_env[k]) for k in paper_env},
}
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "out", "task_counts.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
