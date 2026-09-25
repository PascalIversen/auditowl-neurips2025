"""Static checks for two evaluation-code defects: missing imports in evaluate_traj_by_step.py and the resume-path label flag mismatch in safety_evaluation.py."""
import ast, os, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "yjyddq__RiOSWorld")

res = {}

# 1) evaluate_traj_by_step.py uses os.* and glob.* but never imports them
p1 = os.path.join(REPO, "evaluate", "evaluate_traj_by_step.py")
src1 = open(p1).read()
tree1 = ast.parse(src1)
imported = set()
for node in ast.walk(tree1):
    if isinstance(node, ast.Import):
        for a in node.names:
            imported.add(a.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported.add(node.module.split(".")[0])
res["evaluate_traj_by_step_imports"] = sorted(imported)
res["uses_os_dot"] = "os." in src1
res["uses_glob_dot"] = "glob." in src1
res["os_imported"] = "os" in imported
res["glob_imported"] = "glob" in imported
res["check_folder_results_will_NameError"] = (("os." in src1) and "os" not in imported) or (("glob." in src1) and "glob" not in imported)

# 2) safety_evaluation.py: get_eval_answer writes flags 'yes'/'no'/'Unknown';
#    resume branch compares safe_flag against 'safe'/'unsafe' (never produced).
p2 = os.path.join(REPO, "evaluate", "safety_evaluation.py")
src2 = open(p2).read()
res["writes_yes_no"] = ("== 'yes'" in src2) or ("=='yes'" in src2) or ("'yes'" in src2)
res["resume_compares_safe_unsafe"] = ("safe_flag.lower() == 'safe'" in src2) and ("safe_flag.lower() == 'unsafe'" in src2)
res["resume_branch_dead_for_real_flags"] = res["writes_yes_no"] and res["resume_compares_safe_unsafe"]

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "out", "eval_bugs.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
