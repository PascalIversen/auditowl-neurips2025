"""Checks two runtime blockers: np.random.random_integers removal in pinned numpy, and the get_base_path() loop that searches for a directory name not present in the clone. Supports findings: numpy-random-integers-removed, get-base-path-wrong-dirname."""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out", "runtime_blockers.txt")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
                                    "alesnow97__Spectral_Learning_POMDP"))
lines = []

# 1) numpy version + random_integers availability
lines.append(f"installed numpy version: {np.__version__}  (requirements.txt pins 2.2.6)")
lines.append(f"hasattr(np.random, 'random_integers'): {hasattr(np.random, 'random_integers')}")
if hasattr(np.random, "random_integers"):
    try:
        _ = np.random.random_integers(low=0, high=2)
        lines.append("np.random.random_integers(low=0,high=2) -> OK")
    except Exception as e:
        lines.append(f"np.random.random_integers(...) -> RAISES {type(e).__name__}: {e}")
else:
    lines.append("np.random.random_integers is ABSENT -> line 151 of simulation_spectral_estimation_error.py raises AttributeError")

# 2) get_base_path target directory present in clone?
lines.append("")
target = "NeurIPS_Average_Reward_POMDP"
lines.append(f"get_base_path() walks parents until basename == '{target}'")
lines.append(f"cloned repo dir basename = '{os.path.basename(REPO)}'")
# walk up from REPO to root, see if target ever appears
cur = REPO
found = False
chain = []
while True:
    chain.append(os.path.basename(cur) or cur)
    if os.path.basename(cur) == target:
        found = True
        break
    parent = os.path.dirname(cur)
    if parent == cur:  # reached filesystem root
        break
    cur = parent
lines.append(f"ancestor basenames from repo to root: {chain}")
lines.append(f"target '{target}' found among ancestors: {found}")
if not found:
    lines.append("=> get_base_path() reaches filesystem root without matching; "
                 "os.path.dirname('/')=='/' so the while loop never terminates (hang/infinite loop). "
                 "Plot scripts call utils.get_base_path() and cannot run from the clone as-is.")

txt = "\n".join(lines) + "\n"
with open(OUT, "w") as f:
    f.write(txt)
print(txt)
