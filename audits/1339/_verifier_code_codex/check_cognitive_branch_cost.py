"""Derives the input tensor the frozen Video-Swin-B cognition branch actually receives at stage 2, and compares its token count to the standard Swin3D-B forward pass (supports finding: fps-cost-inconsistent-with-wiring)."""
import os, re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
                                    "AIR-DISCOVER__E-cubed-AD__E-VAD"))

def show(rel, lo, hi, tag):
    p = os.path.join(REPO, rel)
    lines = open(p).read().splitlines()
    print(f"--- {tag}: {rel}:{lo}-{hi}")
    for i in range(lo, hi + 1):
        print(f"  {i:>5}| {lines[i-1]}")
    print()

show("projects/mmdet3d_plugin/EAD/EAD.py", 29, 35, "test-time clip tensor built here (hardcoded HxW)")
show("projects/mmdet3d_plugin/EAD/EAD.py", 171, 179, "clip goes straight into the cognition encoder, unmodified")
show("projects/eeg_vedio/src/models/video_encoder/video_encoder.py", 57, 62,
     "VideoEncoder.forward: permute then Swin3D-B, no resize/renormalise")
show("projects/mmdet3d_plugin/EAD/EAD.py", 93, 99, "the ONLY resize/ImageNet-normalise in the repo -- commented out")
show("projects/configs/EAD/EAD_based_pretrain.py", 16, 17, "img_norm_cfg the front frames have already been through")

# swin3d_b geometry, read from the vendored torchvision copy
sw = open(os.path.join(REPO, "vision15/models/video/swin_transformer.py")).read()
blk = sw[sw.index("def swin3d_b"):]
patch = re.search(r"patch_size=\[(\d+), (\d+), (\d+)\]", blk).groups()
win = re.search(r"window_size=\[(\d+), (\d+), (\d+)\]", blk).groups()
pt, ph, pw = map(int, patch)
print(f"swin3d_b geometry (vision15/models/video/swin_transformer.py): patch_size={patch}, window_size={win}")

def tokens(T, H, W):
    return (T // pt) * (H // ph) * (W // pw)

cases = [
    ("E3AD stage 2, as wired (ImageQueue 4x3x736x1280)", 4, 736, 1280),
    ("standard Swin3D-B Kinetics clip (16x3x224x224)",   16, 224, 224),
    ("the commented-out 224 centre-crop, 4 frames",       4, 224, 224),
]
print(f"\n{'INPUT CLIP':52} {'STAGE-1 TOKENS':>15}")
print("-" * 70)
vals = {}
for name, T, H, W in cases:
    n = tokens(T, H, W)
    vals[name] = n
    print(f"{name:52} {n:>15,}")

a = vals["E3AD stage 2, as wired (ImageQueue 4x3x736x1280)"]
b = vals["standard Swin3D-B Kinetics clip (16x3x224x224)"]
c = vals["the commented-out 224 centre-crop, 4 frames"]
print(f"\nAs-wired / standard-Kinetics token ratio : {a/b:6.2f}x")
print(f"As-wired / commented-out-crop  ratio    : {a/c:6.2f}x")
print("\nSwin window attention is ~linear in token count, so the per-sample cost of the")
print("cognition branch is roughly the ratio above times a full Swin3D-B forward pass,")
print("added to every nuScenes sample at both train and test time.")
print("Paper Tab. 1 reports the E3AD(VAD-Base) FPS cost as 3.8 -> 3.7 (2.6%).")
