#!/usr/bin/env python3
"""Confirm main.py exploit branch logs `chosen_model_id` without assigning it.

Supports finding: main-chosen-model-stale. In run_benchmark(), the exploration
branch assigns `chosen_model_id = len(label_cols) - 1`; the exploitation branch
computes `model_category_chosen` but never sets `chosen_model_id`, yet appends
it to MODEL_CHOSEN_LIST and logs it. So in exploitation steps the logged chosen
model is a stale value from the last exploration step (or NameError on the first
step if it is an exploit). Read-only line analysis.
"""
import os, re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "laminair__mess-plus"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

src = open(os.path.join(REPO, "main.py")).read().splitlines()

# locate run_benchmark exploit (else:) branch and the explore (if x_t==1) branch
explore_assign = [i+1 for i, l in enumerate(src)
                  if re.search(r'chosen_model_id\s*=\s*len\(label_cols\)', l)]
exploit_branch_start = [i+1 for i, l in enumerate(src)
                        if 'logger.info(f"Exploiting for step {timestamp}")' in l]
appends = [i+1 for i, l in enumerate(src) if 'MODEL_CHOSEN_LIST.append(chosen_model_id)' in l]
assigns = [i+1 for i, l in enumerate(src) if re.search(r'\bchosen_model_id\s*=', l)]

# is there any assignment to chosen_model_id inside the exploit branch (lines ~432-473)?
exploit_lo = exploit_branch_start[0] if exploit_branch_start else None
# exploit branch ends before "for instance in instances_to_propagate" (line ~474)
exploit_hi = [i+1 for i, l in enumerate(src) if 'for instance in instances_to_propagate' in l]
exploit_hi = exploit_hi[0] if exploit_hi else len(src)

assign_in_exploit = [a for a in assigns if exploit_lo and exploit_lo < a < exploit_hi]

report = [
    f"chosen_model_id assigned at lines: {assigns}",
    f"explore-branch assignment (len(label_cols)-1) at: {explore_assign}",
    f"MODEL_CHOSEN_LIST.append(chosen_model_id) at: {appends}",
    f"exploit branch span (approx): {exploit_lo}..{exploit_hi}",
    f"assignments to chosen_model_id INSIDE exploit branch: {assign_in_exploit}",
    "",
    "VERDICT: " + (
        "BUG CONFIRMED -- exploit branch appends/logs chosen_model_id with no "
        "assignment in that branch; value is stale from last exploration step "
        "(model index = largest model) or undefined on a first-step exploit."
        if not assign_in_exploit else "no bug: exploit assigns chosen_model_id"),
    "Note: simulator.py (used for the paper's main tables) instead correctly sets "
    "`chosen_model_id = self.labels.index(model_category_chosen)` (simulator.py:198).",
]
out_path = os.path.join(OUT, "main_chosen_model_bug.txt")
open(out_path, "w").write("\n".join(report) + "\n")
print("\n".join(report))
print(f"\nwrote {out_path}")
