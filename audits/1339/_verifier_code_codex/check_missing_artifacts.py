"""Checks which files/modules the released code imports or loads but that do not exist in the repo (supports findings: eeg-dataset-loader-absent, driving-thinking-weights-absent, stage1-entrypoint-dead-import)."""
import os, sys, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
                                    "AIR-DISCOVER__E-cubed-AD__E-VAD"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# (label, referenced-from file:line, path relative to REPO that must exist)
CHECKS = [
    ("RealCarDataset module (stage-1 EEG+video paired dataset)",
     "projects/eeg_vedio/src/run_model/train.py:18",
     "projects/eeg_vedio/src/data/real_car_dataset.py"),
    ("RealCarDataset package dir",
     "projects/eeg_vedio/src/run_model/train.py:18",
     "projects/eeg_vedio/src/data"),
    ("training.train module (main.py entrypoint)",
     "projects/eeg_vedio/src/main.py:4",
     "projects/eeg_vedio/src/training/train.py"),
    ("Driving-Thinking model weights (stage 2 loads these)",
     "projects/mmdet3d_plugin/EAD/EAD.py:105",
     "projects/eeg_vedio/ckpts/Driving_thinking_model.safetensors"),
    ("LaBraM pretrained checkpoint (stage 1)",
     "projects/eeg_vedio/cfgs/train_config.yaml:21",
     "projects/eeg_vedio/ckpts/eeg_encoder_base.pth"),
    ("Video-Swin base checkpoint (stage 1)",
     "projects/eeg_vedio/cfgs/train_config.yaml:25",
     "projects/eeg_vedio/ckpts/video_encoder_base.safetensors"),
    ("video_encoder_inference.yaml ckpt",
     "projects/eeg_vedio/cfgs/video_encoder_inference.yaml:4",
     "projects/eeg_vedio/ckpts/first_try/video_encoder_ft_1.safetensors"),
    ("any ckpts/ directory at all",
     "-", "projects/eeg_vedio/ckpts"),
    ("stage-1 clip_info.json (paired clip index)",
     "projects/eeg_vedio/cfgs/train_config.yaml:4", "projects/eeg_vedio/clip_info.json"),
    ("nuScenes train ann pkl",
     "projects/configs/EAD/EAD_based_pretrain.py:382",
     "data/nuscenes/vad_nuscenes_infos_temporal_train.pkl"),
]

rows = []
for label, ref, rel in CHECKS:
    p = os.path.join(REPO, rel)
    rows.append({"artifact": label, "referenced_from": ref,
                 "expected_path": rel, "exists": os.path.exists(p)})

print(f"REPO = {REPO}\n")
print(f"{'EXISTS':8} {'EXPECTED PATH':70} REFERENCED FROM")
print("-" * 130)
for r in rows:
    print(f"{str(r['exists']):8} {r['expected_path']:70} {r['referenced_from']}")
print()
print(f"{sum(1 for r in rows if not r['exists'])}/{len(rows)} referenced artifacts are ABSENT from the repo")

# What DOES exist under projects/eeg_vedio/src ?
src = os.path.join(REPO, "projects/eeg_vedio/src")
print("\nActual contents of projects/eeg_vedio/src/ :")
for entry in sorted(os.listdir(src)):
    print("  ", entry)

with open(os.path.join(OUT, "missing_artifacts.json"), "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 'missing_artifacts.json')}")
