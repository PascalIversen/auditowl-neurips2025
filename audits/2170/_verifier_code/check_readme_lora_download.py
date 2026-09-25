#!/usr/bin/env python3
"""Check the README's pretrained-weights download snippet: the 'SA1B-trained'
block downloads from the COCO repo_id (copy-paste bug), so following the README
yields the COCO weights for both, not the SA-1B weights.

Supports finding: readme-sa1b-download-points-to-coco.
"""
import re, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "cvlab-kaist__Seg4Diff")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

readme = open(os.path.join(REPO, "README.md")).read()

# Extract the two hf_hub_download blocks and their repo_id
repo_ids = re.findall(r'repo_id="([^"]+)"', readme)
# Find the SA1B comment block
sa1b_block = re.search(r"# Download SA1B-trained lora weights.*?repo_id=\"([^\"]+)\"", readme, re.S)
coco_block = re.search(r"# Download COCO-trained lora weights.*?repo_id=\"([^\"]+)\"", readme, re.S)

print("All repo_id occurrences in README:", repo_ids)
print("COCO block repo_id:", coco_block.group(1) if coco_block else None)
print("SA1B block repo_id:", sa1b_block.group(1) if sa1b_block else None)

bug = sa1b_block and coco_block and (sa1b_block.group(1) == coco_block.group(1) == "chyun/seg4diff-coco-lora")
print(f"\nSA1B download block points to COCO repo (copy-paste bug)? {bool(bug)}")

with open(os.path.join(OUT, "readme_lora_download.txt"), "w") as f:
    f.write(f"coco_repo_id={coco_block.group(1) if coco_block else None}\n")
    f.write(f"sa1b_repo_id={sa1b_block.group(1) if sa1b_block else None}\n")
    f.write(f"bug={int(bool(bug))}\n")
print("wrote out/readme_lora_download.txt")
