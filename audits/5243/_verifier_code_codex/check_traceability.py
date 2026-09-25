"""Map each paper table/figure to a producing artefact in the ClexNet repo.

Deterministic coverage check supporting the traceability table and the
'missing' findings (unos pipeline, boundary sampler, sweeps, reproducibility).
Read-only; writes out/traceability.csv.
"""
import json, os, csv, re

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "AlessandroMarchese__ClexNet")
OUT = os.path.join(os.path.dirname(__file__), "out", "traceability.csv")

nb = json.load(open(os.path.join(REPO, "CLEXNET_tutorial.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"])

def has(*pats):
    return all(re.search(p, src, re.I) for p in pats)

rows = [
    # artefact, producing-code present?, evidence
    ("Table 1 synthetic columns", "PARTIAL",
     "model zoo + instance_metrics present; numbers not pinned (random draw)"),
    ("Table 1 UNOS-PTR columns", "MISSING",
     "no UNOS/PTR/liver/feature code: " + str(not bool(re.search("UNOS|PTR|liver|AGE_DON", src, re.I)))),
    ("Table 2 Uniform reason mech", "PARTIAL", "else-branch w=1/N present"),
    ("Table 2 IPW reason mech", "PARTIAL", "KernelDensity branch present"),
    ("Table 2 Boundary Intersection", "MISSING",
     "no 'boundary'/'intersection' code: " + str(not bool(re.search("boundary|intersection", src, re.I)))),
    ("Table 3 synthetic obs-test", "PARTIAL", "Obs-Test rows present (some commented)"),
    ("Table 3 UNOS-PTR columns", "MISSING", "no UNOS code"),
    ("Table 4 Assumption-1 support", "MISSING",
     "no matching/Euclidean-support code: " + str(not bool(re.search("Euclidean|cumulative support|matching range", src, re.I)))),
    ("Fig 4 confounding sweep psi", "MISSING",
     "rho fixed at single value; no sweep loop over rho/psi"),
    ("Figs 5-6 F/sigma sweep", "MISSING",
     "F_scale param exists but no loop sweeping it: " + str(bool(re.search("F_scale", src)) and not bool(re.search(r"for .* F_scale", src)))),
    ("Table 6 / Fig 7 lambda-rho-M sweep", "MISSING",
     "single fixed configs only; no grid loop"),
    ("Hyperparam Table 7 k=3", "MISMATCH",
     "code default k_cluster=5: " + str(bool(re.search(r"k_cluster\s*=\s*5", src)))),
    ("Dependency specification", "MISSING",
     "no requirements/environment file: " + str(not any(os.path.exists(os.path.join(REPO, f))
        for f in ["requirements.txt","environment.yml","setup.py","pyproject.toml","Pipfile"]))),
]

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["paper_artefact", "status", "evidence"])
    for r in rows:
        w.writerow(r)
        print(f"{r[1]:9s} | {r[0]:38s} | {r[2]}")
print("\nwrote", OUT)
