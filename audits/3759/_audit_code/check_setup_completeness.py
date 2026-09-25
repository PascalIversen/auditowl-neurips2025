"""Checks README setup path vs repo: env filename, dataset_preprocessing dir, data dirs, weights, deps. Supports findings: readme-env-file-mismatch, missing-data-and-weights, hardcoded-cluster-paths."""
import os, re, glob, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "tianxiao18__Lfp2vec"))
out = {}

# 1. README references vs reality
readme = open(os.path.join(REPO, "README.md")).read()
out["readme_mentions_environment.yml"] = "environment.yml" in readme
out["environment.yml_exists"] = os.path.exists(os.path.join(REPO, "environment.yml"))
out["blind_localization.yml_exists"] = os.path.exists(os.path.join(REPO, "blind_localization.yml"))
out["readme_mentions_pip_requirements"] = bool(re.search(r"pip install -r requirements", readme))
out["readme_mentions_dataset_preprocessing_dir"] = "dataset_preprocessing" in readme
out["dataset_preprocessing_dir_exists"] = os.path.isdir(os.path.join(REPO, "script", "dataset_preprocessing"))

# 2. data presence: README claims data/<DS>/{lfp,raw,spectrogram}/*.pkl
data_pkls = glob.glob(os.path.join(REPO, "data", "**", "*.pkl"), recursive=True) + \
            glob.glob(os.path.join(REPO, "data", "**", "*.pickle"), recursive=True)
out["data_pickle_files_count"] = len(data_pkls)
for ds in ["Allen", "ibl", "Neuronexus"]:
    for sub in ["lfp", "raw", "spectrogram"]:
        out[f"data/{ds}/{sub}_dir_exists"] = os.path.isdir(os.path.join(REPO, "data", ds, sub))

# 3. pretrained weights
weights = []
for ext in ("*.bin", "*.safetensors", "*.pt", "*.pth", "*.ckpt"):
    weights += glob.glob(os.path.join(REPO, "**", ext), recursive=True)
out["pretrained_weight_files_count"] = len(weights)

# 4. deps in yml pip section
yml = open(os.path.join(REPO, "blind_localization.yml")).read()
for pkg in ["transformers", "evaluate", "datasets", "accelerate"]:
    out[f"yml_has_{pkg}"] = bool(re.search(rf"\b{pkg}\b", yml))

print(json.dumps(out, indent=2))
with open(os.path.join(os.path.dirname(__file__), "out", "setup_completeness.json"), "w") as f:
    json.dump(out, f, indent=2)
