#!/usr/bin/env python3
"""Checks which prompter methods, configs, and dependency files exist in the repo.

Supports findings: only-acavp-prompter, missing-baseline-configs, no-requirements.
Read-only; writes a summary CSV to out/.
"""
import ast
import os
import csv

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "s-enmt__ACAVP")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# 1. Prompter classes defined
prompters_src = open(os.path.join(REPO, "models", "prompters.py")).read()
tree = ast.parse(prompters_src)
prompter_classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

# 2. Config files present
config_dir = os.path.join(REPO, "configs")
configs = sorted(os.listdir(config_dir))

# 3. Dependency files present
dep_files = ["requirements.txt", "environment.yml", "pyproject.toml",
             "setup.py", "Pipfile", "poetry.lock", "conda.yml"]
dep_present = {f: os.path.exists(os.path.join(REPO, f)) for f in dep_files}

# Methods/components the paper references
paper_methods = ["VP", "EVP", "AutoVP", "Coordinator", "ACAVP"]
paper_ablation_variants = ["Affine", "Color", "Affine+Color",
                           "Resize+Additive+Color", "Affine+Additive"]
paper_overfit_techniques = ["Dropout", "MSE Loss", "Weight decay",
                            "TrivialAugment", "RandAugment", "IPMix"]

rows = []
rows.append(["prompter_classes_defined", ";".join(prompter_classes)])
rows.append(["config_files_present", ";".join(configs)])
for f, present in dep_present.items():
    rows.append([f"dep_{f}_present", present])
rows.append(["paper_methods_compared", ";".join(paper_methods)])
rows.append(["paper_ablation_variants", ";".join(paper_ablation_variants)])
rows.append(["paper_overfit_techniques", ";".join(paper_overfit_techniques)])

# Which paper methods have a prompter class?
for m in paper_methods:
    rows.append([f"prompter_class_for_{m}", m in prompter_classes])
for m in paper_methods:
    rows.append([f"config_for_{m}", f"{m}.yaml" in configs])

with open(os.path.join(OUT, "repo_completeness.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "value"])
    w.writerows(rows)

print("=== Prompter classes defined in models/prompters.py ===")
print(prompter_classes)
print("\n=== Config files in configs/ ===")
print(configs)
print("\n=== Dependency spec files present? ===")
for f, present in dep_present.items():
    print(f"  {f}: {present}")
print("\n=== Paper compares against these methods (Tables 2,3,4,8) ===")
print(paper_methods)
print("  prompter class present for each:",
      {m: (m in prompter_classes) for m in paper_methods})
print("  config present for each:",
      {m: (f"{m}.yaml" in configs) for m in paper_methods})
print("\n=== Paper ablation variants (Table 5) ===")
print(paper_ablation_variants, "-> any dedicated config?",
      [v for v in paper_ablation_variants if f"{v}.yaml" in configs])
print("\n=== Paper overfitting-mitigation techniques (Table 6) ===")
print(paper_overfit_techniques)
print("\nWrote out/repo_completeness.csv")
