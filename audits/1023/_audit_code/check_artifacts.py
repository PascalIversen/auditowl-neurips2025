"""Read-only deterministic checks for the spatiotemporal-oversquashing repo.

Supports findings:
- fosr-rgcn-missing: no FOSR rewiring or RGCN model anywhere in the repo (Table 2).
- successrate-aggregation-missing: no code computes the success-rate (MSE<0.001)
  or builds the paper tables/figures (Fig 3, Fig 4, Tab 1, Tab 2).
- multi-run-seed-commented: the `+task` (multiple seeds per config) sweep line is
  commented out in every sweep config, while the paper reports success rates / error
  bars over multiple runs.

Run: python check_artifacts.py  (prints to stdout and writes out/checks.txt)
"""
import os
import re
import subprocess

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "marshka__spatiotemporal-oversquashing")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out", "checks.txt")


def grep(pattern, exts=(".py", ".yaml", ".yml", ".md", ".ipynb")):
    hits = []
    rx = re.compile(pattern, re.IGNORECASE)
    for root, dirs, files in os.walk(REPO):
        if ".git" in root.split(os.sep):
            continue
        for fn in files:
            if not fn.endswith(exts):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append((os.path.relpath(p, REPO), i, line.rstrip()))
            except Exception:
                pass
    return hits


def main():
    lines = []
    def emit(s):
        lines.append(s)
        print(s)

    emit("=== Repo file inventory (non-.git) ===")
    for root, dirs, files in os.walk(REPO):
        if ".git" in root.split(os.sep):
            continue
        for fn in sorted(files):
            emit("  " + os.path.relpath(os.path.join(root, fn), REPO))

    emit("\n=== FOSR / RGCN search (Table 2 rewiring) ===")
    fosr = grep(r"fosr|rewir|rgcn|relational.?graph.?conv")
    emit(f"  hits: {len(fosr)}")
    for h in fosr:
        emit(f"    {h}")

    emit("\n=== Success-rate / 0.001 threshold / table-building search (Fig3/4,Tab1/2) ===")
    sr = grep(r"success.?rate|< ?0\.001|0\.001|threshold")
    # filter to interesting (exclude lr 0.001 and distance threshold which are unrelated)
    emit(f"  raw hits: {len(sr)}")
    for h in sr:
        emit(f"    {h}")

    emit("\n=== '+task' multi-seed sweep line (commented?) ===")
    task = grep(r"\+task")
    for h in task:
        commented = h[2].lstrip().startswith("#")
        emit(f"    commented={commented}  {h}")

    emit("\n=== aggregation artefacts (notebooks/csv/plot scripts) ===")
    nb = []
    for root, dirs, files in os.walk(REPO):
        if ".git" in root.split(os.sep):
            continue
        for fn in files:
            if fn.endswith((".ipynb", ".csv")) or "plot" in fn.lower() or "aggregate" in fn.lower():
                nb.append(os.path.relpath(os.path.join(root, fn), REPO))
    emit(f"  hits: {nb}")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
