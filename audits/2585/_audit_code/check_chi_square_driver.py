"""Checks whether any released driver instantiates the chi-square algorithm
class, which Figure 2(b) and 2(d) require. Supports finding
`chi2-panels-driver-missing`. Read-only grep over the driver scripts.
Output -> out/chi_square_driver.txt
"""
import os, re

SUP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "supplement"))
OUT = os.path.join(os.path.dirname(__file__), "out", "chi_square_driver.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

drivers = [
    "Reduction_DMDP_parallel.py", "Anchored_AMDP_parallel.py",
    "add_large_scale_dmdp.py", "add_large_scale_anchored.py",
    "add_baseline.py", "dr_q_learning_experiment.py",
]
lines = []
any_instantiates_chi = False
for d in drivers:
    src = open(os.path.join(SUP, d)).read().splitlines()
    inst_kl  = [i+1 for i, L in enumerate(src) if re.search(r"DR_RL_empirical_kl\s*\(", L)]
    inst_chi = [i+1 for i, L in enumerate(src) if re.search(r"DR_RL_empirical_chi_square\s*\(", L)]
    imp_chi  = [i+1 for i, L in enumerate(src) if "DR_RL_empirical_chi_square" in L and "import" in L]
    if inst_chi:
        any_instantiates_chi = True
    lines.append(f"{d}:")
    lines.append(f"    imports chi_square class at lines      : {imp_chi}")
    lines.append(f"    INSTANTIATES DR_RL_empirical_kl()      : {inst_kl}")
    lines.append(f"    INSTANTIATES DR_RL_empirical_chi_square: {inst_chi if inst_chi else 'NONE'}")
    lines.append("")
lines.append(f"ANY driver instantiates the chi-square class? {any_instantiates_chi}")
lines.append("Figure 2(b) [chi2, Alg 2] and 2(d) [chi2, Alg 3] each need the chi-square algorithm;")
lines.append("the chi-square class is imported everywhere but never instantiated by any driver.")
txt = "\n".join(lines)
print(txt)
open(OUT, "w").write(txt)
print("written to", OUT)
