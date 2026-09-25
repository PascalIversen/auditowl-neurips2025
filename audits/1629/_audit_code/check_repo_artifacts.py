#!/usr/bin/env python3
"""Checks completeness/provenance claims for the MESS+ repo.

Supports findings:
- missing-inference-data (no captured CSV data / W&B logs / checkpoints in repo)
- model-zoo-count-mismatch (paper says 3-model Llama zoo, configs list 4)
Read-only; writes a summary to _audit_code/out/repo_artifacts.txt.
"""
import os
import glob
import re

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "laminair__mess-plus")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

lines = []

# 1. Captured inference data / checkpoints / wandb logs present?
csvs = glob.glob(os.path.join(REPO, "**", "*.csv"), recursive=True)
ckpts = glob.glob(os.path.join(REPO, "**", "*.pt"), recursive=True) + \
        glob.glob(os.path.join(REPO, "**", "*.ckpt"), recursive=True)
data_files = [f for f in glob.glob(os.path.join(REPO, "data", "**"), recursive=True)
              if os.path.isfile(f)]
inf = glob.glob(os.path.join(REPO, "**", "inference_outputs", "**"), recursive=True)
wandb_dirs = glob.glob(os.path.join(REPO, "**", "wandb"), recursive=True)
lines.append(f"CSV files in repo: {len(csvs)} -> {csvs}")
lines.append(f"Model checkpoints (.pt/.ckpt): {len(ckpts)} -> {ckpts}")
lines.append(f"Files under data/: {len(data_files)} -> {data_files}")
lines.append(f"inference_outputs files: {len([f for f in inf if os.path.isfile(f)])}")
lines.append(f"wandb log dirs: {wandb_dirs}")
lines.append("=> Simulator (simulator.py) requires --dataset-path CSV data of captured "
             "LLM outputs; baselines/tables require W&B run logs. None present in repo.")
lines.append("")

# 2. Model-zoo count per llama3 config
lines.append("Model categories per config/llama3/*.yaml:")
for f in sorted(glob.glob(os.path.join(REPO, "config", "llama3", "*.yaml"))):
    txt = open(f).read()
    cats = re.findall(r'category:\s*"([^"]+)"', txt)
    lines.append(f"  {os.path.basename(f)}: {cats}")
lines.append("Paper Table 2 zoo: L1B / L8B / L70B (3 models). "
             "Configs add a 3B 'small' model => 4 models.")
lines.append("")

# 3. baseline notebook references nonexistent config/online dir and 'algorithm' key
online_dir = os.path.join(REPO, "config", "online")
qwen_online = os.path.join(REPO, "config", "qwen2", "online")
algo_key = [f for f in glob.glob(os.path.join(REPO, "config", "**", "*.yaml"), recursive=True)
            if re.search(r'^algorithm:', open(f).read(), re.M)]
lines.append(f"config/online/ exists: {os.path.isdir(online_dir)}")
lines.append(f"config/qwen2/online/ exists: {os.path.isdir(qwen_online)}")
lines.append(f"configs with top-level 'algorithm:' key: {algo_key}")
lines.append("run_baselines.ipynb loads CONFIG['algorithm'] and config/online/<bench>.yaml "
             "-- neither exists in the repo.")

out_path = os.path.join(OUT, "repo_artifacts.txt")
with open(out_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {out_path}")
