"""Checks which paper-required artefacts (data splits, trained weights, deps) are absent from the ARGRE repo. Supports findings: missing-toxicity-pairwise-data, missing-trained-weights, unpinned-missing-deps, detoxify-checkpoint-hardcoded."""
import os, re, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "xiaoyisong__ARGRE"))
out = {}

# 1. Data split files referenced by collect_hidden.py
split_refs = ["data/toxicity_pairwise/split_0.jsonl", "data/toxicity_pairwise/split_1.jsonl"]
out["data_splits"] = {p: os.path.exists(os.path.join(REPO, p)) for p in split_refs}
out["toxicity_pairwise_dir_contents"] = sorted(os.listdir(os.path.join(REPO, "data/toxicity_pairwise")))

# 2. Trained reward-model weights / exp dir
out["exp_dir_exists"] = os.path.isdir(os.path.join(REPO, "evaluation/exp"))

# 3. Dependencies imported but not in req.txt
with open(os.path.join(REPO, "req.txt")) as f:
    req = f.read().lower()
imported = set()
for root, _, files in os.walk(REPO):
    if ".git" in root:
        continue
    for fn in files:
        if fn.endswith(".py"):
            with open(os.path.join(root, fn), errors="ignore") as fh:
                for line in fh:
                    m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_]+)", line)
                    if m:
                        imported.add(m.group(1))
third_party = ["lm_eval", "wandb", "datasets", "peft", "numpy", "detoxify", "transformers",
               "torch", "accelerate", "scipy", "sklearn", "huggingface_hub", "tqdm"]
out["imported_but_not_in_req"] = sorted(
    [m for m in third_party if m in imported and m.replace("_", "-") not in req and m not in req]
)

# 4. Detoxify checkpoint path hardcoded in evaluate_model.py
em = os.path.join(REPO, "evaluation/utils/evaluate_model.py")
with open(em) as f:
    txt = f.read()
m = re.search(r"checkpoint='([^']+)'", txt)
out["detoxify_checkpoint_arg"] = m.group(1) if m else None
out["detoxify_checkpoint_exists"] = os.path.exists(os.path.join(REPO, m.group(1))) if m else None

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "out", "missing_artifacts.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
