"""Checks presence of producing scripts / imported modules / ablation entrypoints for GC-xLSTM.

Supports findings: missing-amcparser-plot-module, missing-acatis-module,
missing-auroc-computation, missing-table3-ablations, hardcoded-abs-paths.
Read-only: only stats files and greps source text. Writes a summary CSV to out/.
"""
import os, re, csv, glob

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "harpoonix__GC-xLSTM"))
SRC = os.path.join(REPO, "GC-xLSTM")

def exists(rel):
    return os.path.exists(os.path.join(REPO, rel))

rows = []

# 1. AMCParser.plot_motion_gc module imported by driver line 15
amc_candidates = [
    "GC-xLSTM/datasets/mocap/all_asfamc/AMCParser/plot_motion_gc.py",
    "GC-xLSTM/datasets/mocap/all_asfamc/AMCParser",
]
rows.append(["amcparser_plot_motion_gc_module_present",
             any(exists(c) for c in amc_candidates)])

# 2. acatis data-prep module imported by prepare_data.py line 5
acatis_candidates = [
    "GC-xLSTM/datasets/acatis/prepare_acatis_data.py",
    "GC-xLSTM/datasets/acatis",
]
rows.append(["acatis_prepare_module_present", any(exists(c) for c in acatis_candidates)])

# 3. Any AUROC / sklearn roc computation anywhere in repo .py
py_files = glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True)
auroc_hits = []
for f in py_files:
    if os.sep + ".git" + os.sep in f:
        continue
    try:
        txt = open(f, encoding="utf8", errors="ignore").read()
    except Exception:
        continue
    if re.search(r"roc_auc|auroc|trapz|np\.trapz|metrics\.auc|roc_curve", txt, re.I):
        auroc_hits.append(os.path.relpath(f, REPO))
rows.append(["auroc_computation_found", bool(auroc_hits), ";".join(auroc_hits)])

# 4. Table-3 ablations: a path that trains the LSTM (not xLSTM) forecaster, or a
#    Group-Lasso (non-alpha) optimisation, reachable from a config or the driver.
driver = open(os.path.join(SRC, "xlstm_neural_gc.py"), encoding="utf8").read()
# driver always builds componentXLSTM; check whether it can build cLSTM / LSTM
driver_builds_clstm = bool(re.search(r"\bcLSTM\s*\(", driver) or re.search(r"\bcomponentLSTM\s*\(", driver))
rows.append(["driver_can_build_lstm_forecaster", driver_builds_clstm])
# group-lasso path: train_model_ista is the only trainer; does any code call the
# plain `regularize(` (group lasso) inside the active loop (not commented)?
clstm = open(os.path.join(SRC, "models", "clstm.py"), encoding="utf8").read()
active_grouplasso = False
for line in clstm.splitlines():
    s = line.strip()
    if s.startswith("#"):
        continue
    # a bare group-lasso `regularize(net...)` *call* (exclude def and ridge_)
    if re.search(r"(?<!ridge_)(?<!def )regularize\(net\w", s):
        active_grouplasso = True
rows.append(["active_group_lasso_regularizer_call_in_loop", active_grouplasso])

# 5. Hardcoded /home/harsh absolute paths in shipped source
abs_hits = []
for f in py_files + glob.glob(os.path.join(REPO, "**", "*.yaml"), recursive=True):
    if os.sep + ".git" + os.sep in f:
        continue
    try:
        for i, line in enumerate(open(f, encoding="utf8", errors="ignore"), 1):
            if "/home/harsh" in line:
                abs_hits.append(f"{os.path.relpath(f, REPO)}:{i}")
    except Exception:
        pass
rows.append(["hardcoded_home_harsh_paths", bool(abs_hits), ";".join(abs_hits)])

# 6. driver line 15 verbatim import target
m = re.search(r"from (datasets\.mocap[^\s]+) import plot_graph_from_GC", driver)
rows.append(["driver_imports_amcparser_at_toplevel", bool(m), m.group(1) if m else ""])

out = os.path.join(os.path.dirname(__file__), "out", "artefacts.csv")
with open(out, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["check", "result", "detail"])
    for r in rows:
        w.writerow(r + [""] * (3 - len(r)))

for r in rows:
    print(r)
print("\nwrote", out)
