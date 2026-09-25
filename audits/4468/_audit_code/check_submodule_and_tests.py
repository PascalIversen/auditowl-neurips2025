"""Deterministic checks: (1) EG-CFG forks populated+modified; (2) MBPP guidance tests == eval tests; (3) CFG eq.13 equivalence. Supports findings: forks-present, mbpp-guidance-equals-eval, no findings on CFG math."""
import os, ast, subprocess
R = "code/boazlavon__eg_cfg"
out = []

# 1. submodule populated + EG-CFG markers in transformers fork
tf = os.path.join(R, "submodules/transformers/generation/utils.py")
out.append(f"transformers fork generation/utils.py exists: {os.path.exists(tf)}")
if os.path.exists(tf):
    txt = open(tf).read()
    out.append(f"  _eg_cfg_sample defined: {'def _eg_cfg_sample(' in txt}")
    out.append(f"  EG-CFG markers count: {txt.count('EG-CFG Modification')}")
    out.append(f"  apply_guidance call present: {'eg_cfg_injection_manager.apply_guidance(' in txt}")
for sm in ["xpython","trepan","trepan-xpy","transformers"]:
    d = os.path.join(R,"submodules",sm)
    nfiles = sum(len(f) for _,_,f in os.walk(d))
    out.append(f"submodule {sm}: dir_exists={os.path.isdir(d)} file_count={nfiles}")

# 2. MBPP: test_cases_to_prompt and test_cases_to_eval both problem['test_list']
sm = open(os.path.join(R,"eg_cfg/eg_cfg_session_manager.py")).read()
out.append("MBPP guidance test source line present (test_cases_to_prompt = problem[\"test_list\"]): "
           + str('test_cases_to_prompt = problem["test_list"]' in sm))
out.append("MBPP eval test source present (DATASET__MBPP -> test_cases_to_eval = eval_problem[\"test_list\"]): "
           + str('test_cases_to_eval = eval_problem["test_list"]' in sm))

# 3. CFG eq.13 equivalence: P*(Pc/P)^gamma == exp(logP + gamma*(logPc-logP))
import math
P, Pc, g = 0.3, 0.6, 3.0
lhs = P*((Pc)/(P))**g
rhs = math.exp(math.log(P) + g*(math.log(Pc)-math.log(P)))
out.append(f"CFG eq13 equivalence (probs-domain == log-domain): {abs(lhs-rhs)<1e-9} (lhs={lhs:.6f} rhs={rhs:.6f})")

print("\n".join(out))
open("_audit_code/out/checks.txt","w").write("\n".join(out)+"\n")
