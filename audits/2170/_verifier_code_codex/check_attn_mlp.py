#!/usr/bin/env python3
"""Check whether MAGNET training adds a learned conv head (AttentionScoreLayer)
that the paper's MAGNET description (LoRA + mask loss only) never mentions, and
whether that head's weights are released alongside the LoRA weights.

Supports finding: magnet-undisclosed-attn-mlp-head.
"""
import re, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "cvlab-kaist__Seg4Diff")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

coco = open(os.path.join(REPO, "configs/train_coco.yaml")).read()
sa1b = open(os.path.join(REPO, "configs/train_sa1b.yaml")).read()
train_model = open(os.path.join(REPO, "seg4diff/seg4diff_model_train.py")).read()
readme = open(os.path.join(REPO, "README.md")).read()

checks = {}
checks["train_coco.yaml USE_ATTN_MLP: True"] = bool(re.search(r"USE_ATTN_MLP:\s*True", coco))
checks["train_sa1b.yaml USE_ATTN_MLP: True"] = bool(re.search(r"USE_ATTN_MLP:\s*True", sa1b))
checks["trainer instantiates self.attn_mlp = AttentionScoreLayer()"] = (
    "self.attn_mlp = AttentionScoreLayer()" in train_model)
checks["attn_mlp params set requires_grad=True (trained)"] = bool(
    re.search(r"for name, params in self\.attn_mlp\.named_parameters\(\):\s*\n\s*params\.requires_grad = True", train_model))
checks["attn_mlp applied with residual in training forward"] = (
    "_outputs = residual + _outputs" in train_model)
checks["AttentionScoreLayer is a 3-layer Conv2d net"] = bool(
    re.search(r"nn\.Conv2d\(1, 64.*nn\.Conv2d\(64, 64.*nn\.Conv2d\(64, 1", train_model, re.S))
# README only mentions releasing lora_weights.pth, not the attn_mlp head
checks["README release mentions only lora_weights.pth"] = (
    "lora_weights.pth" in readme)
checks["README does NOT mention attn_mlp / AttentionScoreLayer weights"] = (
    "attn_mlp" not in readme and "AttentionScoreLayer" not in readme)

print("=== undisclosed AttentionScoreLayer (attn_mlp) head in MAGNET training ===")
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

with open(os.path.join(OUT, "attn_mlp.txt"), "w") as f:
    for k, v in checks.items():
        f.write(f"{int(v)}\t{k}\n")
print("\nwrote out/attn_mlp.txt")
