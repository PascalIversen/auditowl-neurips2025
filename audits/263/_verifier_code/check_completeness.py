"""Checks repo completeness vs paper artefacts (supports findings: missing-readme,
missing-figure-code, missing-lpips-dep, ipdb-breakpoint). Read-only; writes CSV to out/."""
import os, re, csv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def exists(rel):
    return os.path.exists(os.path.join(REPO, rel))

def grep_repo(pattern, exts=(".py", ".yaml", ".toml")):
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for root, _, files in os.walk(REPO):
        if "_audit_code" in root or "/.git" in root:
            continue
        for f in files:
            if f.endswith(exts):
                p = os.path.join(root, f)
                with open(p, errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append((os.path.relpath(p, REPO), i, line.rstrip()))
    return hits

rows = []
# README referenced by pyproject.toml
rows.append(("README.md present", exists("README.md")))
# Figure 1 code: cosine-similarity histograms / Imagenette dimension experiment
rows.append(("Imagenette/cosine-sim code (Fig 1)", bool(grep_repo(r"imagenette|cosine"))))
rows.append(("two-moons histogram code (Fig 1a)", bool(grep_repo(r"sample_8gaussians.*moons|two.?moons|moons"))))
# Figure 2: velocity error vs n_samples + nearest-neighbor distance plot driver
rows.append(("nearest-neighbor distance experiment driver (Fig 2 right)", bool(grep_repo(r"nearest.?neighbor"))))
# Figure 3: hybrid model (u_star until tau then u_theta) + LPIPS
rows.append(("hybrid-model tau-switch driver (Fig 3)", bool(grep_repo(r"\bhybrid\b|\btau\b"))))
rows.append(("LPIPS dependency/usage (Fig 3 metric)", bool(grep_repo(r"lpips"))))
# CelebA training script (Fig 3/4) -- paper says pnpflow lib, not in this repo
rows.append(("CelebA training script in repo", bool(grep_repo(r"def train.*celeba|train_celeba"))))
# MNIST/FMNIST experiment (Appendix C tables)
rows.append(("MNIST/FMNIST experiment code (App. C)", bool(grep_repo(r"\bmnist\b|fmnist"))))
# live debugger breakpoint
ipdb_live = grep_repo(r"^\s*import ipdb; ipdb.set_trace\(\)")
rows.append(("live ipdb.set_trace() (uncommented)", bool(ipdb_live)))

os.makedirs(os.path.join(REPO, "_audit_code", "out"), exist_ok=True)
out = os.path.join(REPO, "_audit_code", "out", "completeness.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["check", "present_or_found"])
    for k, v in rows:
        w.writerow([k, v])
        print(f"{v!s:>5}  {k}")
if ipdb_live:
    print("\nLive ipdb breakpoints:")
    for r in ipdb_live:
        print("  ", r)
print(f"\nwrote {out}")
