"""Checks the output-directory bug: Fig-2/Appendix-H drivers write pickles to
subdirs they never create, so pickle.dump crashes (FileNotFoundError).
Supports finding `save-subdir-not-created`. Read-only: copies the relevant
lines and simulates the open() against a throwaway tmp tree without touching
the repo. Output -> out/save_dirs.txt
"""
import os, tempfile, re, sys

SUP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "supplement"))
OUT = os.path.join(os.path.dirname(__file__), "out", "save_dirs.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (driver, makedirs-arg-or-None, save-path-template)
cases = [
    ("Reduction_DMDP_parallel.py", "data", "data/Reduction_DMDP/kl/p_0.90_T.pkl"),
    ("Anchored_AMDP_parallel.py",  "data", "data/Anchored_AMDP/kl/p_0.90_T.pkl"),
    ("add_baseline.py",            None,   "rebuttal/baseline/T_1_10.pkl"),
    ("dr_q_learning_experiment.py",None,   "rebuttal/baseline/T_1_10.pkl"),
    ("add_large_scale_dmdp.py",    None,   "rebuttal/large_scale/T_1_10.pkl"),
    ("add_large_scale_anchored.py",None,   "rebuttal/large_scale/T_1_10.pkl"),
]

lines = []
for driver, mkd, savepath in cases:
    # confirm what the source actually contains
    src = open(os.path.join(SUP, driver)).read()
    has_mkd_data = "os.makedirs(\"data\"" in src
    has_any_mkd  = "os.makedirs" in src
    # simulate in a sandbox tmp dir
    with tempfile.TemporaryDirectory() as tmp:
        if mkd:
            os.makedirs(os.path.join(tmp, mkd), exist_ok=True)  # mimic the driver's makedirs("data")
        target = os.path.join(tmp, savepath)
        try:
            with open(target, "wb") as f:
                f.write(b"x")
            result = "WROTE OK (parent dir existed)"
        except FileNotFoundError as e:
            result = f"FileNotFoundError: {e.strerror} -> {os.path.dirname(savepath)}"
    mkd_desc = 'os.makedirs(data)' if has_mkd_data else ('some makedirs' if has_any_mkd else 'NONE')
    lines.append(f"{driver}:")
    lines.append(f"    makedirs in source : {mkd_desc}")
    lines.append(f"    save path          : {savepath}")
    lines.append(f"    open(...,'wb')     : {result}")
    lines.append("")

txt = "\n".join(lines)
print(txt)
open(OUT, "w").write(txt)
print("written to", OUT)
