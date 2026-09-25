"""Counts the number of runs/experiments in each released data file backing Figures 1-4, to check the paper's '10 runs' claim. Supports findings: regret-runs-not-ten, estimation-fig1-runs-not-ten."""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
                                    "alesnow97__Spectral_Learning_POMDP"))
EXP = os.path.join(REPO, "NeurIPS_experiments")
OUT = os.path.join(os.path.dirname(__file__), "out", "run_counts.txt")

lines = []

# --- Estimation-error data files (Figures 1 and 3) ---
est_files = {
    "Fig1 main S=4,A=3,O=4": "4states_3actions_4obs/pomdp1/estimation_error/1.json",
    "Fig3-left S=3,A=2,O=5": "3states_2actions_5obs/pomdp5/estimation_error/3.json",
    "Fig3-right S=5,A=3,O=5 (file1_first_two)": "5states_3actions_5obs/pomdp6/estimation_error/1_first_two_exp.json",
    "Fig3-right S=5,A=3,O=5 (file2_last_three)": "5states_3actions_5obs/pomdp6/estimation_error/2_last_three_exp.json",
}
lines.append("=== ESTIMATION ERROR (Fig 1 / Fig 3) ===")
for label, rel in est_files.items():
    p = os.path.join(EXP, rel)
    if os.path.exists(p):
        d = json.load(open(p))
        import numpy as np
        shp = np.array(d["observation_matrix_error_frobenious_norms"]).shape
        lines.append(f"{label}: num_experiments={d.get('num_experiments')} "
                     f"num_checkpoints={d.get('num_checkpoints')} obs_frob_shape={shp}")
    else:
        lines.append(f"{label}: FILE MISSING ({rel})")

# --- Regret data (Figure 2) ---
lines.append("")
lines.append("=== REGRET (Fig 2) — counting per-algorithm run indices present ===")
regret_sub = "regret/13Ep_0.04discr_10000tau1_30000tau2_0.02SMAC_30000SMT0_30000MXTO"
for pomdp in ["pomdp1", "pomdp3", "pomdp5"]:
    rdir = os.path.join(EXP, "3states_3actions_4obs", pomdp, regret_sub)
    if not os.path.isdir(rdir):
        lines.append(f"{pomdp}: DIR MISSING")
        continue
    names = os.listdir(rdir)
    def idxs(prefix):
        s = set()
        for n in names:
            if n.startswith(prefix):
                # prefix like ORACLE_  or MixedSpectral_  parse first int after prefix
                rest = n[len(prefix):]
                num = ""
                for ch in rest:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num:
                    s.add(int(num))
        return sorted(s)
    oracle = idxs("ORACLE_")
    mixed = idxs("MixedSpectral_")
    seeu = idxs("SEEU_")
    smucrl = idxs("SMUCRL_")
    lines.append(f"{pomdp}: ORACLE runs={oracle} MixedSpectral runs={mixed} "
                 f"SEEU runs={seeu} SMUCRL runs={smucrl}")

txt = "\n".join(lines) + "\n"
with open(OUT, "w") as f:
    f.write(txt)
print(txt)
