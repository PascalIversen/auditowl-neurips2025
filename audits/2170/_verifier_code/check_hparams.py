#!/usr/bin/env python3
"""Check paper-stated hyperparameters (LR, batch size) vs the released YAML configs.

Supports findings: train-lr-mismatch, train-batchsize-sa1b-mismatch.
Read-only: parses the repo's config YAMLs with a tiny regex (no yaml dep needed).
"""
import re, os, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "cvlab-kaist__Seg4Diff")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def grab(path, key):
    txt = open(path).read()
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([^\n#]+)", txt, re.M)
    return m.group(1).strip() if m else None

rows = []
# Paper §4.1: "trained using AdamW with lr = 1e-5"
# Paper §4.1: "per-device batch size 4 and gradient accumulation for an effective batch size of 16"
paper_lr = 1e-5
paper_eff_bs = 16

for cfg in ["configs/train_coco.yaml", "configs/train_sa1b.yaml", "configs/config.yaml"]:
    p = os.path.join(REPO, cfg)
    base_lr = grab(p, "BASE_LR")
    ims = grab(p, "IMS_PER_BATCH")
    accum = grab(p, "GRADIENT_ACCUMULATION_STEPS")
    bbm = grab(p, "BACKBONE_MULTIPLIER")
    rows.append(dict(cfg=cfg, BASE_LR=base_lr, IMS_PER_BATCH=ims,
                     GRAD_ACCUM=accum, BACKBONE_MULTIPLIER=bbm))

print("=== LR / batch-size config dump ===")
for r in rows:
    print(r)

print(f"\nPaper says: lr = {paper_lr} (1e-5); effective batch size = {paper_eff_bs}")

# LR check: paper 1e-5 vs configs
print("\n=== LR comparison ===")
for r in rows:
    if r["BASE_LR"] is None:
        continue
    cfg_lr = float(r["BASE_LR"])
    eff_lr_backbone = cfg_lr * float(r["BACKBONE_MULTIPLIER"]) if r["BACKBONE_MULTIPLIER"] else None
    print(f"{r['cfg']}: BASE_LR={cfg_lr:g}  (x backbone_mult {r['BACKBONE_MULTIPLIER']} = {eff_lr_backbone:g})"
          f"  | paper lr=1e-5  -> BASE_LR matches paper? {abs(cfg_lr-paper_lr)<1e-9}")

# Effective batch-size check (assuming 2 GPUs per paper). NB: IMS_PER_BATCH in
# detectron2 is the TOTAL images per iteration across all GPUs.
print("\n=== effective batch size (IMS_PER_BATCH * GRAD_ACCUM) ===")
for r in rows[:2]:
    if r["IMS_PER_BATCH"] is None or r["GRAD_ACCUM"] is None:
        continue
    eff = int(r["IMS_PER_BATCH"]) * int(r["GRAD_ACCUM"])
    print(f"{r['cfg']}: IMS_PER_BATCH={r['IMS_PER_BATCH']} * GRAD_ACCUM={r['GRAD_ACCUM']} "
          f"= {eff}  | paper effective bs=16 -> matches? {eff==paper_eff_bs}")

json.dump(rows, open(os.path.join(OUT, "hparams.json"), "w"), indent=2)
print("\nwrote out/hparams.json")
