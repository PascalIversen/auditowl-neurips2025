#!/usr/bin/env python3
"""Check (a) which MLLM backbone the released code uses vs the paper's default
(Pixtral 12B), and (b) whether the code generates a bounding-box visual prompt
(paper's best/default config = bbox + mask-cropped). Supports findings:
backbone-mismatch, bbox-visual-prompt-missing."""
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "suikei-wang__RESAnything"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

lines = []
def log(s=""):
    lines.append(s); print(s)

py_yaml = []
for root, _, files in os.walk(REPO):
    if "/.git" in root:
        continue
    for f in files:
        if f.endswith((".py", ".yaml", ".yml")):
            py_yaml.append(os.path.relpath(os.path.join(root, f), REPO))

# (a) backbone
log("=== Backbone model references in code ===")
MODELS = ["pixtral", "qwen2.5-vl", "qwen2_5_vl", "qwen2-vl", "qwen", "claude", "gpt-4", "gemini"]
for f in sorted(py_yaml):
    with open(os.path.join(REPO, f), errors="replace") as fh:
        src = fh.read()
    for mod in MODELS:
        for m in re.finditer(re.escape(mod), src, flags=re.IGNORECASE):
            ln = src.count("\n", 0, m.start()) + 1
            snippet = src.splitlines()[ln-1].strip()
            log(f"  {f}:{ln}: [{mod}] {snippet}")
log("")
log("PAPER default backbone (paper_text.txt:621): 'We use Pixtral 12B [4] as the MLLM'")
log("PAPER ablation row (Table 5, paper_text.txt:873): 'Qwen 2-VL[7]' (= Qwen2-VL, NOT Qwen2.5-VL)")
log("CODE default backbone (config.yaml): Qwen/Qwen2.5-VL-7B-Instruct")
log("=> code default is neither the paper default (Pixtral 12B) nor the listed ablation (Qwen2-VL)")
log("")

# (b) bounding-box visual prompt
log("=== Bounding-box visual prompt generation in code ===")
BBOX_TOK = ["boundingrect", "cv2.rectangle", "bbox", "bounding_box", "bounding box",
            "draw_box", "x,y,w,h", "min(xs)", "boundingRect"]
found = False
for f in sorted(py_yaml):
    with open(os.path.join(REPO, f), errors="replace") as fh:
        src = fh.read()
    for tok in BBOX_TOK:
        for m in re.finditer(re.escape(tok), src, flags=re.IGNORECASE):
            ln = src.count("\n", 0, m.start()) + 1
            log(f"  {f}:{ln}: {tok}")
            found = True
if not found:
    log("  (none) -- NO bounding-box visual-prompt generation found in any source file")
log("")
log("PAPER (paper_text.txt:366-367): 'we find using two visual prompts, bounding box (V b)")
log("  and mask cropped (V m), is sufficient' -- this is the paper's chosen/default config")
log("PAPER Table 5 (paper_text.txt:855): mask+bbox = best 72.2 gIoU on RefCOCO testA")
log("CODE: candidate generation (generation.py:53) and text-mask comparison (similarity.py:65)")
log("  read ONLY from the 'cropped' dir; sam_utils saves only cropped/overlay/txt, no bbox crop.")

with open(os.path.join(OUT, "backbone_and_prompts.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
log(f"[written] {os.path.join(OUT, 'backbone_and_prompts.txt')}")
