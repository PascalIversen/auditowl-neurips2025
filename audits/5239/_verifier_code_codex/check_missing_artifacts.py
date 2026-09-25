"""Deterministic existence/grep checks for paper artefacts that may be absent
from BOTH released code trees (GitHub mirror + NeurIPS supplemental). Supports
the `missing` findings for: (1) Fig 16 / Sec 3.4 MNIST density-ranking, (2) the
Fig 7 / Sec 3.3 iterated SD-KDE experiment, (3) hardcoded absolute /scratch and
/pscratch paths that block out-of-the-box reproduction. Read-only."""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def all_py(root):
    for dirpath, _dirs, files in os.walk(root):
        if ".git" in dirpath:
            continue
        for fn in files:
            if fn.endswith((".py", ".sh")):
                yield os.path.join(dirpath, fn)

def grep_any(patterns):
    hits = []
    rx = [re.compile(p, re.I) for p in patterns]
    for fp in all_py(ROOT):
        try:
            with open(fp, "r", errors="ignore") as f:
                for ln, line in enumerate(f, 1):
                    for r in rx:
                        if r.search(line):
                            hits.append((os.path.relpath(fp, ROOT), ln, line.rstrip()))
        except OSError:
            continue
    return hits

lines = []
def emit(s):
    print(s)
    lines.append(s)

# 1) MNIST density-ranking by SD-KDE (Fig 16 / Sec 3.4)
emit("=== (1) MNIST density-ranking by estimated probability density ===")
rank_hits = grep_any([r"argsort", r"\brank", r"descending order", r"sort.*densit",
                      r"highest.*lowest", r"latent space"])
emit(f"  rank/argsort/latent-density hits: {len(rank_hits)}")
for h in rank_hits[:20]:
    emit(f"    {h[0]}:{h[1]}: {h[2]}")

# 2) Iterated SD-KDE (Fig 7 / Sec 3.3): multiple correction iterations + KL trend
emit("=== (2) Iterated SD-KDE (multi-step) experiment ===")
iter_hits = grep_any([r"for .*iter", r"n_iter", r"num_iter", r"iteration",
                      r"recompute.*score", r"successive"])
# exclude UCI/bank data-file false positives by restricting to .py
iter_hits = [h for h in iter_hits if h[0].endswith(".py")]
emit(f"  iteration-loop hits in .py: {len(iter_hits)}")
for h in iter_hits[:20]:
    emit(f"    {h[0]}:{h[1]}: {h[2]}")

# 3) Hardcoded absolute paths
emit("=== (3) Hardcoded absolute /scratch or /pscratch paths ===")
path_hits = grep_any([r"/scratch/", r"/pscratch/"])
emit(f"  absolute-path hits: {len(path_hits)}")
for h in path_hits:
    emit(f"    {h[0]}:{h[1]}: {h[2]}")

with open(os.path.join(OUT, "missing_artifacts.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
