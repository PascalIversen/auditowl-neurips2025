#!/usr/bin/env python3
"""Deterministic completeness checks for the E-cubed-AD (E-VAD branch) repo.

Supports findings: missing-dataset-and-loader, missing-driving-thinking-weights,
stage1-broken-imports, missing-nonvad-integrations, missing-closedloop-code,
missing-crossdataset-split, missing-framework-1-2. Read-only over code/.

Run:  cd _audit_code && python check_repo_completeness.py
Writes a summary to out/repo_completeness.txt.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(
    HERE, "..", "code", "AIR-DISCOVER__E-cubed-AD__E-VAD"))
OUT = os.path.join(HERE, "out", "repo_completeness.txt")

lines = []
def log(s=""):
    print(s)
    lines.append(s)

def exists(rel):
    return os.path.exists(os.path.join(REPO, rel))

def grep_repo(pattern, exts=(".py", ".yaml", ".yml", ".sh")):
    """Case-insensitive regex search over tracked source files; returns hit count + samples."""
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for root, dirs, files in os.walk(REPO):
        # skip .git and the vendored torchvision copy (vision15/) which is a
        # third-party dependency, not author experiment code.
        if ".git" in root or os.sep + "vision15" in root or root.endswith("vision15"):
            continue
        for f in files:
            if not f.endswith(exts):
                continue
            p = os.path.join(root, f)
            try:
                with open(p, "r", errors="ignore") as fh:
                    for i, ln in enumerate(fh, 1):
                        if rx.search(ln):
                            hits.append((os.path.relpath(p, REPO), i, ln.rstrip()))
            except Exception:
                pass
    return hits

log(f"REPO = {REPO}")
log(f"exists = {os.path.isdir(REPO)}")
log("")

# ---- 1. Self-collected dataset loader module ----
log("== 1. Stage-1 dataset loader (RealCarDataset / data.real_car_dataset) ==")
log(f"  src/data dir exists          : {exists('projects/eeg_vedio/src/data')}")
log(f"  real_car_dataset.py exists   : {exists('projects/eeg_vedio/src/data/real_car_dataset.py')}")
imp = grep_repo(r"from data\.real_car_dataset import RealCarDataset")
log(f"  import sites of RealCarDataset: {len(imp)}")
for h in imp:
    log(f"     {h[0]}:{h[1]}: {h[2].strip()}")
log("")

# ---- 2. main.py broken import (training.train) ----
log("== 2. Stage-1 entrypoint main.py import target ==")
tt = grep_repo(r"from training\.train import")
log(f"  'from training.train import' sites: {len(tt)}")
for h in tt:
    log(f"     {h[0]}:{h[1]}: {h[2].strip()}")
log(f"  src/training dir exists      : {exists('projects/eeg_vedio/src/training')}")
log("")

# ---- 3. Any model weights / checkpoints committed ----
log("== 3. Committed weights / checkpoints ==")
wexts = (".pth", ".safetensors", ".ckpt", ".pt", ".pkl", ".bin")
found_w = []
for root, dirs, files in os.walk(REPO):
    if ".git" in root:
        continue
    for f in files:
        if f.endswith(wexts):
            found_w.append(os.path.relpath(os.path.join(root, f), REPO))
log(f"  weight files found: {len(found_w)}")
for w in found_w:
    log(f"     {w}")
# specific referenced weights
for w in ["ckpts/Driving_thinking_model.safetensors",
          "ckpts/eeg_encoder_base.pth",
          "ckpts/video_encoder_base.safetensors"]:
    log(f"  referenced weight present? {w}: "
        f"{exists('projects/eeg_vedio/'+w) or exists(w)}")
log("")

# ---- 4. Non-VAD integrations (UniAD / GenAD / LAW / VAD-Tiny) ----
log("== 4. Non-VAD-Base integrations referenced by Table 1 ==")
for kw in [r"\buniad\b", r"\bgenad\b", r"latent world model|\bLAW\b", r"vad[-_ ]?tiny|tiny"]:
    hits = grep_repo(kw)
    hits = [h for h in hits if "flaw" not in h[2].lower()]
    log(f"  pattern {kw!r:35}: {len(hits)} hits")
log("")

# ---- 5. Closed-loop (Bench2Drive / CARLA) code ----
log("== 5. Closed-loop simulation code (Table 2, A.6) ==")
for kw in [r"bench2drive", r"\bcarla\b", r"leaderboard", r"driving[_ ]?score|\bDS\b", r"route completion|success rate|\bSR\b"]:
    hits = grep_repo(kw)
    log(f"  pattern {kw!r:35}: {len(hits)} hits")
log("  NOTE: carla appears only as a pinned line in requirements.txt (dependency), not as code:")
for h in grep_repo(r"carla", exts=(".txt",)):
    log(f"     {h[0]}:{h[1]}: {h[2].strip()}")
log("")

# ---- 6. Cross-dataset Singapore/Boston split (A.7) ----
log("== 6. Cross-dataset geographic split (Table A.7) ==")
for kw in [r"singapore", r"boston", r"location"]:
    log(f"  pattern {kw!r:15}: {len(grep_repo(kw))} hits")
log("")

# ---- 7. Number of EAD configs / framework selection ----
log("== 7. EAD configs present ==")
cfgdir = os.path.join(REPO, "projects", "configs")
ead_cfgs = []
for root, dirs, files in os.walk(cfgdir):
    for f in files:
        if f.endswith(".py") and ("EAD" in root or "ead" in f.lower()):
            ead_cfgs.append(os.path.relpath(os.path.join(root, f), REPO))
log(f"  EAD config files: {ead_cfgs}")
# Framework wiring in head: does forward branch on framework, or always eeg_decoder?
head = os.path.join(REPO, "projects/mmdet3d_plugin/EAD/EAD_head.py")
with open(head, errors="ignore") as fh:
    htxt = fh.read()
log(f"  head references eeg_decoder(  : {'self.eeg_decoder(' in htxt}")
log(f"  head references AttnGate/TokenLearner (framework 1): "
    f"{bool(re.search(r'attngate|tokenlearner', htxt, re.I))}")
log("")

# ---- 8. Stage-1 80:10:10 split logic ----
log("== 8. Stage-1 train/val/test split logic ==")
for kw in [r"train_test_split", r"val(idation)?_set|test_set", r"0\.8\b|80:10:10", r"random_split"]:
    log(f"  pattern {kw!r:25}: {len(grep_repo(kw))} hits (in eeg_vedio scope shown below)")
sp = [h for h in grep_repo(r"train_test_split|random_split|80:10:10") ]
for h in sp:
    log(f"     {h[0]}:{h[1]}: {h[2].strip()}")
log("")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"[written] {OUT}")

# ---- machine-readable summary for citing by csv_row ----
def n(pattern, exts=(".py", ".yaml", ".yml", ".sh")):
    return len(grep_repo(pattern, exts))

rows = [
    ("check", "result", "detail"),
    ("stage1_dataset_loader_present",
     exists("projects/eeg_vedio/src/data/real_car_dataset.py"),
     "imported at train.py:18 & train_ddp.py:23; src/data absent"),
    ("driving_thinking_weight_present",
     exists("projects/eeg_vedio/ckpts/Driving_thinking_model.safetensors"),
     "required by EAD.py:105 / video_encoder.py:41 assert"),
    ("committed_weight_files", len(found_w),
     "no .pth/.safetensors/.ckpt/.pt/.pkl/.bin tracked"),
    ("uniad_integration_hits", n(r"\buniad\b"),
     "E3AD(UniAD) rows in Table 1 & Table 2"),
    ("genad_integration_hits", n(r"\bgenad\b"),
     "E3AD(GenAD) row in Table 1"),
    ("law_integration_hits",
     len([h for h in grep_repo(r"latent world model|\bLAW\b") if "flaw" not in h[2].lower()]),
     "E3AD(LAW) row in Table 1"),
    ("closedloop_carla_bench2drive_code_hits",
     n(r"bench2drive") + len(grep_repo(r"\bcarla\b", exts=(".py",))),
     "Table 2 / A.6 closed-loop DS,SR; carla only pinned in requirements.txt"),
    ("crossdataset_geo_split_code",
     "map-metadata-only",
     "singapore/boston only in nuscenes_vad_dataset.py map list; no train-on-one/test-on-other split (Table A.7)"),
    ("ead_config_files", ";".join(ead_cfgs),
     "only VAD-Base EAD config present"),
    ("framework1_attngate_tokenlearner_present",
     bool(re.search(r'attngate|tokenlearner', htxt, re.I)),
     "Framework 1 'Attach to Spatio-temporal' (Eq.5-6); head only wires eeg_decoder (Framework 3)"),
    ("stage1_split_logic_hits",
     len(grep_repo(r"train_test_split|random_split|80:10:10")),
     "80:10:10 split from paper §3.1 not implemented"),
]
import csv as _csv
csvpath = os.path.join(HERE, "out", "repo_completeness.csv")
with open(csvpath, "w", newline="") as f:
    w = _csv.writer(f)
    for r in rows:
        w.writerow(r)
log(f"[written] {csvpath}")
for i, r in enumerate(rows):
    log(f"  csv_row {i}: {r[0]} = {r[1]}")
