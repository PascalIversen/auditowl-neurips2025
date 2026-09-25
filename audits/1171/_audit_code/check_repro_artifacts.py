#!/usr/bin/env python3
"""Checks presence of reproduction artefacts: requirements/env file, pretrained
base weights, per-dataset run commands in README, results-table.
Supports findings `no-requirements-file` and `single-run-command`.
Read-only; writes report to out/repro_artifacts.txt."""
import os
import glob

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "Legenddddd__MEL"))
OUT = os.path.join(os.path.dirname(__file__), "out", "repro_artifacts.txt")

lines = []

# 1. dependency spec
dep_patterns = ["requirements*.txt", "environment*.yml", "environment*.yaml",
                "setup.py", "setup.cfg", "pyproject.toml", "Pipfile", "conda*.txt"]
deps = []
for p in dep_patterns:
    deps += glob.glob(os.path.join(REPO, p))
lines.append("Dependency-spec files found: " + (str([os.path.basename(d) for d in deps]) if deps else "NONE"))

# 2. pretrained weights / checkpoints
weight_ext = ("*.pth", "*.pt", "*.ckpt", "*.pkl", "*.bin", "*.h5")
weights = []
for ext in weight_ext:
    weights += glob.glob(os.path.join(REPO, "**", ext), recursive=True)
weights = [w for w in weights if ".git" not in w]
lines.append("Model weight files (.pth/.pt/...) in repo: " + (str([os.path.relpath(w, REPO) for w in weights]) if weights else "NONE"))

# 3. README run commands
readme = os.path.join(REPO, "README.md")
with open(readme) as f:
    rtext = f.read()
n_python_cmds = rtext.count("python train.py")
lines.append(f"README 'python train.py' invocations: {n_python_cmds}")
# which datasets named in README
for ds in ["cub200", "StanfordDog", "StanfordCar", "Aircraft", "CUB", "Dog", "Car", "Aircraft"]:
    if ds.lower() in rtext.lower():
        lines.append(f"  README mentions dataset token '{ds}': yes")

# 4. results table in README
lines.append("README contains a results/accuracy table: " +
             ("yes" if ("Acc" in rtext or "accuracy" in rtext.lower() or "| " in rtext and "---" in rtext) else "no"))

# 5. import torch_dct (third-party dep used in helper.py)
helper = os.path.join(REPO, "models", "mel", "helper.py")
with open(helper) as f:
    htext = f.read()
lines.append("helper.py imports 'torch_dct' (unlisted dep): " + ("yes" if "import torch_dct" in htext else "no"))
net = os.path.join(REPO, "models", "mel", "Network.py")
with open(net) as f:
    ntext = f.read()
lines.append("Network.py imports 'timm' (unlisted dep): " + ("yes" if "import timm" in ntext else "no"))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
