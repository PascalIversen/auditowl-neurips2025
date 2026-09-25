"""Checks repo completeness: dependency spec, PDDL problems, trained weights, training-data caches.
Supports findings: missing-dependency-spec, missing-pddl-problems, missing-trained-weights.
Read-only; writes a summary CSV to out/."""
import os, csv, glob

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "Learning-for-Seq-Decision-Making__GABAR-Graph-based-action-ranking-for-planning")
REPO = os.path.abspath(REPO)

def exists_any(patterns):
    hits = []
    for p in patterns:
        hits += [h for h in glob.glob(os.path.join(REPO, p), recursive=True)
                 if "/.git/" not in h and "/VAL-master/" not in h]
    return hits

rows = []
# 1. dependency specification files
dep = exists_any(["requirements*.txt", "setup.py", "setup.cfg", "pyproject.toml",
                  "environment.yml", "environment.yaml", "Pipfile", "poetry.lock"])
rows.append(("dependency_spec_file", len(dep), ";".join(os.path.relpath(h, REPO) for h in dep) or "NONE"))

# 2. PDDL domain/problem files inside repo (excluding VAL examples)
pddl = exists_any(["**/*.pddl"])
rows.append(("pddl_problem_files_in_repo", len(pddl), ";".join(os.path.relpath(h, REPO) for h in pddl[:5]) or "NONE"))

# 3. trained model weights
weights = exists_any(["**/*.pt", "**/*.pth", "**/*.ckpt"])
rows.append(("trained_weight_files", len(weights), ";".join(os.path.relpath(h, REPO) for h in weights[:5]) or "NONE"))

# 4. cached training-data pickles (the .pkl the pipeline writes/reads)
pkls = exists_any(["**/*.pkl"])
rows.append(("training_data_pickles", len(pkls), ";".join(os.path.relpath(h, REPO) for h in pkls[:5]) or "NONE"))

# 5. FD non-optimal reference jsons (PQR denominator) present?
ref = exists_any(["cache/results/planner_data/*_non_optimal.json"])
rows.append(("fd_reference_jsons", len(ref), ";".join(os.path.relpath(h, REPO) for h in ref) or "NONE"))

out = os.path.join(os.path.dirname(__file__), "out", "repo_completeness.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["item", "count", "examples"]); w.writerows(rows)

for r in rows:
    print(r)
print("\nWrote", out)
