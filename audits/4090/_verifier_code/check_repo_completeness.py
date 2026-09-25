"""Checks repo completeness vs paper claims: dataset coverage, image-domain code,
hardcoded data path, and test-set evaluation during training. Supports findings
datasets-missing, image-code-missing, hardcoded-data-path, test-set-model-selection."""
import os, re, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "wxl1122__URD")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

results = {}

# 1) Datasets present in repo data dir
data_dir = os.path.join(REPO, "Unifying_INN", "data")
present = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat") or f.endswith(".npz"))
results["datasets_present"] = present
# Paper Table 1 lists 20 tabular datasets; get_data.sh references these:
get_data = open(os.path.join(REPO, "Unifying_INN", "get_data.sh")).read()
fetch_names = sorted(set(re.findall(r"#([a-z0-9_-]+)\n", get_data)))
results["get_data_sh_datasets"] = fetch_names

# 2) Image-domain (CIFAR/MVTec) code present?
hits = []
for root, _, files in os.walk(os.path.join(REPO, "Unifying_INN")):
    if "layers/base" in root or "__pycache__" in root:
        continue
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        txt = open(p, errors="ignore").read()
        for kw in ["cifar", "mvtec", "wideresnet", "Conv2d", "pixel"]:
            if re.search(kw, txt, re.I):
                hits.append((os.path.relpath(p, REPO), kw))
results["image_domain_code_hits"] = hits

# 3) Hardcoded absolute path in loadData
ld = open(os.path.join(REPO, "Unifying_INN", "utils", "loadData.py")).read()
m = re.search(r"loadmat\((['\"].*?['\"])", ld)
results["loadData_path_expr"] = m.group(1) if m else None
results["loadData_path_is_absolute_hardcoded"] = bool(m and "/home/" in m.group(1))

# 4) main.py: does evaluation_batch run on test_loader during training, and is the
#    saved checkpoint reloaded inside eval (i.e. test metric printed each 5 epochs)?
mn = open(os.path.join(REPO, "Unifying_INN", "main.py")).read()
results["eval_called_in_training_loop"] = bool(re.search(r"if \(epoch\+1\) % 5 == 0:[\s\S]{0,200}evaluation_batch", mn))
results["no_validation_set"] = ("val" not in mn.lower().replace("evaluation", ""))
results["best_epoch_logic"] = ("best" in mn.lower())

print(json.dumps(results, indent=2))
with open(os.path.join(OUT, "repo_completeness.json"), "w") as f:
    json.dump(results, f, indent=2)
