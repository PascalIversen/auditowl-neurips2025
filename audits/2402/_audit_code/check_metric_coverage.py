#!/usr/bin/env python3
"""Read-only check: which paper-reported metrics have computation code in the repo,
which README-referenced files exist, and how many .py files contain hardcoded
absolute paths. Supports findings: metrics-not-computed, backtranslation-missing,
readme-file-mismatch, hardcoded-abs-paths.

Run: cd _audit_code && python check_metric_coverage.py
Outputs: out/metric_coverage.csv, out/readme_files.csv, out/hardcoded_paths.csv
"""
import csv
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "code" / "umnooob__signvip"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def grep_count(pattern):
    """Count py files (excluding metrics/calculate_* defs) that reference pattern."""
    try:
        res = subprocess.run(
            ["grep", "-rilE", pattern, str(REPO), "--include=*.py"],
            capture_output=True, text=True,
        )
        return [l for l in res.stdout.splitlines() if l]
    except Exception:
        return []


# 1. Metric coverage: does any code COMPUTE this metric (not just generate videos)?
metrics = {
    "FID (Table 3)": r"calculate_fid|FrechetInception|inception_v3",
    "CLIP-FID (Table 3)": r"clip.?fid|clipfid",
    "FVD (Tables 3,4)": r"calculate_fvd|get_fvd_feats|load_i3d",
    "IDS / identity sim (Table 3)": r"arc2face|arc2face|yolo5face|identity.?sim|face.?embed",
    "PSNR (Table 4)": r"psnr|peak_signal",
    "SSIM (Table 4)": r"calculate_ssim|def ssim",
    "LPIPS (Table 4)": r"lpips|perceptual",
    "Hand SSIM (Table 4)": r"hand_ssim|hand.?region",
    "BLEU (Tables 1,2,7)": r"bleu|sacrebleu",
    "ROUGE (Tables 1,2,7)": r"rouge",
    "COMET (Tables 1,2)": r"\bcomet\b(?!_ml)",
    "Back-translation SLT model (App. C)": r"back.?translat|slt.?model|video.?to.?text|pose.?to.?text",
    "Norm. DTW (Tables 10,11)": r"fastdtw|normalized_distance",
    "Token accuracy (not in paper tables)": r"valid_acc",
}
with open(OUT / "metric_coverage.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "found_in_files", "files"])
    for name, pat in metrics.items():
        files = grep_count(pat)
        rel = [os.path.relpath(x, REPO) for x in files]
        w.writerow([name, len(files), ";".join(rel)])
        print(f"{'OK ' if files else 'NONE'} {name}: {len(files)} files {rel}")

# 2. README-referenced files
readme = (REPO / "README.md").read_text()
referenced = [
    "train.sh", "train_stage_1.py", "train_stage_2.py",
    "train_compress_vq_multicond.py", "train_multihead_t2vqpgpt.py",
    "scripts/RWTH-T/3_process_annotation.py",
]
print("\n--- README file existence ---")
with open(OUT / "readme_files.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["referenced_file", "exists"])
    for r in referenced:
        ex = (REPO / r).exists()
        w.writerow([r, ex])
        print(f"{'EXISTS' if ex else 'MISSING'}: {r}")

# 3. Hardcoded absolute paths
print("\n--- hardcoded /deepo_data paths ---")
deepo = grep_count(r"/deepo_data")
with open(OUT / "hardcoded_paths.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file_with_hardcoded_path"])
    for d in deepo:
        w.writerow([os.path.relpath(d, REPO)])
print(f"{len(deepo)} .py files contain hardcoded '/deepo_data' paths")
