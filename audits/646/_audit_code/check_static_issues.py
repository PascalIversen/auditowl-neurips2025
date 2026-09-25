"""Static checks for generate.py: delta_p use-before-def in non-train branch,
81-frame reshape mismatch, and unlisted runtime deps. Supports findings
delta-p-undefined-non-train, frames81-broken, deps-unlisted."""
import ast
import os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "berkegokmen1__RoPECraft")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

gen = os.path.join(REPO, "generate.py")
src = open(gen).read().splitlines()

lines = []

# 1. delta_p: assigned only inside `if train_mode:` optim loop, used in the
# always-executed no_grad block.
assign = [i + 1 for i, l in enumerate(src) if "delta_p = sum(" in l]
use = [i + 1 for i, l in enumerate(src) if "custom_image_rotary_emb + delta_p" in l]
lines.append(f"delta_p assigned at lines: {assign}")
lines.append(f"delta_p used at lines:     {use}")
lines.append("=> In train_mode=False, the assignment loop is skipped but the "
             "no_grad forward at line 952 still references delta_p -> NameError.")
lines.append("")

# 2. 81-frame path: custom_rope num_frames arg commented out -> default 13.
for i, l in enumerate(src):
    if "num_frames=13 if num_frames == 49 else 21" in l:
        lines.append(f"line {i+1}: {l.strip()}  (COMMENTED OUT)")
    if "13 if num_frames == 49 else 21" in l and "indices" in src[i-1] if i>0 else False:
        pass
for i, l in enumerate(src):
    if "indices = torch.linspace(0, gt_flow_feats.size(0) - 1, 13 if num_frames == 49 else 21)" in l:
        lines.append(f"line {i+1}: {l.strip()}  (selects 21 flow frames for 81-frame run)")
    if "num_frames: int = 13," in l:
        lines.append(f"line {i+1}: {l.strip()}  (CustomWanRotaryPosEmbed default)")
    if "latents = torch.randn(1, 16, 13, 60, 104)" in l:
        lines.append(f"line {i+1}: {l.strip()}  (hardcoded 13 latent frames)")
lines.append("=> For --frames 81: 21 flow frames feed CustomWanRotaryPosEmbed "
             "whose num_frames stays 13 (arg commented out) -> reshape(13, W, H, -1) "
             "fails since h_list has 21*W rows. Also global latents has 13 frames not 21.")
lines.append("")

# 3. unlisted deps: imports vs requirements.txt
reqs = open(os.path.join(REPO, "requirements.txt")).read()
req_names = {r.split("==")[0].strip().lower() for r in reqs.splitlines() if r.strip()}
imp_third_party = set()
for f in ("generate.py", "ftd.py"):
    t = ast.parse(open(os.path.join(REPO, f)).read())
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                imp_third_party.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            imp_third_party.add(n.module.split(".")[0])
stdlib = {"argparse","datetime","json","typing","math","os","gc","types","shutil",
          "random","pathlib"}
third = sorted(m for m in imp_third_party if m not in stdlib)
# map common import->dist
distmap = {"cv2":"opencv-python","PIL":"pillow","torch":"torch",
           "torchvision":"torchvision","numpy":"numpy","tqdm":"tqdm",
           "decord":"decord","diffusers":"diffusers","transformers":"transformers",
           "frechetdist":"frechetdist","einops":"einops","accelerate":"accelerate",
           "ftfy":"ftfy","imageio":"imageio","np":"numpy"}
lines.append("Third-party imports and whether their dist is pinned in requirements.txt:")
for m in third:
    dist = distmap.get(m, m).lower()
    status = "listed" if dist in req_names else "NOT LISTED"
    lines.append(f"  import {m:14s} -> dist {dist:16s} [{status}]")
lines.append("Note: co-tracker is loaded via torch.hub (ftd.py) -> needs internet/clone, not a pip dep.")

open(os.path.join(OUT, "static_issues.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
