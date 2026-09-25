#!/usr/bin/env python3
"""Verify that the released open-vocab seg eval builds the text prompt ONLY from
the GT classes present in each image (oracle class set), not the full dataset
vocabulary.

Supports finding: ovss-gt-only-prompt-oracle.
Read-only static checks on the released config + model code.
"""
import re, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "cvlab-kaist__Seg4Diff")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

cfg = open(os.path.join(REPO, "configs/eval_ovss.yaml")).read()
model = open(os.path.join(REPO, "seg4diff/seg4diff_model_ovss.py")).read()

checks = {}
checks["eval_ovss.yaml sets GT_ONLY_PROMPT: True"] = bool(
    re.search(r"GT_ONLY_PROMPT:\s*True", cfg))
# forward(): selected_idxs = idxs  (idxs == GT classes present in image)
checks["ovss.forward sets selected_idxs = idxs (GT indices)"] = (
    "selected_idxs = idxs" in model)
# prompt built from selected_idxs only
checks["prompt = ' '.join(classnames from selected_idxs)"] = bool(
    re.search(r"classnames\s*=\s*\[self\.test_class_texts\[int\(i\)\][^\n]*for i in selected_idxs\]", model)
)
# idxs come from the GT semantic map bincount (i.e. classes actually present)
checks["idxs derived from GT sem_seg bincount (get_gt_indices)"] = bool(
    re.search(r"targets\.flatten\(\)\.bincount\(\)\[:171\]\.nonzero", model))
# Confirm there is NO branch that uses the full test vocabulary as the prompt
# (search for any use of the whole test_class_texts list as the prompt)
full_vocab_prompt = bool(re.search(r'prompt\s*=\s*"\s*"\.join\(\s*self\.test_class_texts\s*\)', model)) \
    or bool(re.search(r"\.join\(\s*self\.test_class_texts\s*\)", model))
checks["NO full-vocabulary prompt path present in ovss model"] = (not full_vocab_prompt)

print("=== GT-only-prompt evidence (open-vocab eval) ===")
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

# Show the decisive code window
start = model.index("selected_idxs = idxs")
print("\n--- decisive window (seg4diff_model_ovss.py) ---")
print(model[start-40:start+360])

with open(os.path.join(OUT, "gt_only_prompt.txt"), "w") as f:
    for k, v in checks.items():
        f.write(f"{int(v)}\t{k}\n")
print("\nwrote out/gt_only_prompt.txt")
