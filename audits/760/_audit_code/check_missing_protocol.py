"""Checks (read-only) which paper-described protocols have NO implementation in the
core MIGN repo: the generalization half-station split (Table 4 / Fig 3) and the
Persistence baseline (Tables 1,2,4 / Figs 4,6). Supports findings
'generalization-split-missing' and 'persistence-baseline-missing'."""
import os, re, subprocess

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "compasszzn__MIGN")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# Core code = everything except vendored third-party baseline libraries.
def core_py_files():
    files = []
    for root, _, names in os.walk(REPO):
        if "/.git" in root:
            continue
        if "/baseline/spatial_temporal/pytorch_geometric_temporal" in root:
            continue
        if "/baseline/locationencoder" in root:
            continue
        for n in names:
            if n.endswith(".py"):
                files.append(os.path.join(root, n))
    return files

files = core_py_files()

def grep(patterns):
    hits = []
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for pat in patterns:
            for m in re.finditer(pat, txt):
                line = txt[:m.start()].count("\n") + 1
                hits.append((os.path.relpath(f, REPO), line, pat, m.group(0)))
    return hits

lines = []
# 1. Generalization split: random sampling of HALF the stations into disjoint train/test.
gen_hits = grep([r"randperm", r"random\.sample", r"random_split",
                 r"train_test_split", r"unseen", r"0\.5\s*\*\s*num",
                 r"half"])
lines.append("== Generalization half-station split (paper §4.3 / Table 4 / Fig 3) ==")
if gen_hits:
    for h in gen_hits:
        lines.append(f"  POSSIBLE: {h[0]}:{h[1]}  pattern={h[2]!r}  match={h[3]!r}")
else:
    lines.append("  NO station-sampling / half-split code found in core repo.")

# 2. Persistence baseline.
pers_hits = grep([r"(?i)persistence", r"(?i)last[_ ]?value", r"persist"])
lines.append("== Persistence baseline (Tables 1,2,4; Figs 4,6) ==")
if pers_hits:
    for h in pers_hits:
        lines.append(f"  POSSIBLE: {h[0]}:{h[1]}  match={h[3]!r}")
else:
    lines.append("  NO 'persistence' baseline code found in core repo.")

out = "\n".join(lines) + "\n"
print(out)
open(os.path.join(OUT, "missing_protocol.txt"), "w").write(out)
