"""Inventory the ClexNet repo and check which paper artefacts have producing code.

Supports findings: unos-semisynthetic-missing, boundary-intersection-missing,
no-requirements, traceability table. Read-only; writes out/inventory.txt.
"""
import json, os, re

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "AlessandroMarchese__ClexNet")
OUT = os.path.join(os.path.dirname(__file__), "out", "inventory.txt")

lines = []
def log(s=""):
    lines.append(s); print(s)

# 1) tracked files
tracked = []
for root, dirs, files in os.walk(REPO):
    if ".git" in root: continue
    for f in files:
        p = os.path.relpath(os.path.join(root, f), REPO)
        if ".ipynb_checkpoints" in p: continue
        tracked.append(p)
log("== Repo files (excluding .git / checkpoints) ==")
for t in sorted(tracked): log("  " + t)
log()

# 2) dependency spec present?
dep_files = ["requirements.txt","environment.yml","environment.yaml","setup.py",
             "pyproject.toml","Pipfile","conda.yaml"]
present = [d for d in dep_files if os.path.exists(os.path.join(REPO, d))]
log(f"Dependency-spec files present: {present if present else 'NONE'}")
log()

# 3) keyword presence in the only code file (the notebook)
nb = json.load(open(os.path.join(REPO, "CLEXNET_tutorial.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
keywords = {
    "UNOS / semi-synthetic real data": ["UNOS","PTR","liver","AGE_DON","COLD_ISCH","ETHCAT"],
    "one-hot encoding (paper says 76 dims after OHE)": ["OHE","OneHot","one-hot","get_dummies"],
    "boundary-intersection reason sampler (Table 2 row 3)": ["boundary","intersection"],
    "confounding sweep psi (Fig 4 / Exp 5.2)": ["psi","\\psi"],
    "F-scaling sweep sigma (Fig 5/6 / Exp A.1)": ["sigma","F(sigma)","scaling factor"],
}
log("== Keyword presence in notebook (the ONLY code file) ==")
for label, kws in keywords.items():
    counts = {k: len(re.findall(re.escape(k), src, flags=re.I)) for k in kws}
    total = sum(counts.values())
    log(f"  [{'FOUND' if total else 'ABSENT'}] {label}: {counts}")
log()

# 4) which Sample mechanism is implemented for reasons?
log("== Reason-generation mechanisms implemented in make_dataset ==")
log("  ipw_reasons branch present: " + str("ipw_reasons" in src))
log("  uniform (else w=1/N) branch present: " + str("np.full(len(Z_big_pos), 1/len(Z_big_pos))" in src))
log("  boundary-intersection branch present: " + str(bool(re.search("boundary", src, re.I))))
log()

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("\nwrote", OUT)
