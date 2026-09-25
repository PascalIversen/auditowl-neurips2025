"""Checks (read-only) supporting DOVE findings: missing external eval deps,
wrong --gt in inference.sh, off-by-one in eval_dover, and absence of the
video-processing-pipeline (Sec 3.3) code. Outputs CSV to out/."""
import os, re, csv, glob

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "zhengchen1999__DOVE"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

rows = []

# 1. External modules imported by eval scripts but absent from repo.
ext_checks = {
    "finetune/scripts/eval_dover.py": ("DOVER", "DOVER/evaluate_a_set_of_videos.py"),
    "finetune/scripts/eval_vbench.py": ("VBench", "VBench/evaluate.py"),
    "finetune/scripts/eval_ewarp.py": ("ewarp", "finetune/scripts/ewarp.py or RAFT/ewarp.py"),
}
for script, (mod, expected) in ext_checks.items():
    present = bool(glob.glob(os.path.join(REPO, "**", mod), recursive=True)) or \
              bool(glob.glob(os.path.join(REPO, "**", mod + ".py"), recursive=True))
    rows.append(["external_dep_present", f"{script}->{mod}", str(present), expected])

# raft-things.pth path that eval_ewarp passes by default
ewarp_default_model = os.path.join(REPO, "finetune/scripts/models/raft-things.pth")
rows.append(["eval_ewarp_default_model_exists", "finetune/scripts/models/raft-things.pth",
             str(os.path.exists(ewarp_default_model)), "RAFT weight actually at finetune/utils/RAFT/raft-things.pth"])

# 2. inference.sh: count --gt lines and how many point to UDM10/GT
inf = open(os.path.join(REPO, "inference.sh")).read()
gt_lines = re.findall(r"--gt\s+(\S+)", inf)
udm_count = sum(1 for g in gt_lines if "UDM10/GT" in g)
rows.append(["inference_sh_gt_total", "inference.sh", str(len(gt_lines)), str(gt_lines)])
rows.append(["inference_sh_gt_all_udm10", "inference.sh", str(udm_count == len(gt_lines)),
             f"{udm_count}/{len(gt_lines)} --gt point to UDM10/GT (should differ per dataset)"])

# 3. eval_dover off-by-one
dov = open(os.path.join(REPO, "finetune/scripts/eval_dover.py")).read().splitlines()
offby = [i+1 for i, l in enumerate(dov) if "dover_results[i-1]" in l]
rows.append(["eval_dover_off_by_one_line", "finetune/scripts/eval_dover.py",
             str(bool(offby)), f"lines={offby} (uses i-1 inside enumerate loop)"])

# 4. Video processing pipeline (Sec 3.3) code: search for any of the 4 steps' hallmark code.
hallmarks = ["scenedetect", "PySceneDetect", "aesthetic", "clip_iqa", "CLIPIQA",
             "FasterVQA", "motion_mask", "bounding_box", "motion_area", "optical_flow_score"]
hits = {}
for py in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
    if ".git" in py:
        continue
    txt = open(py, errors="ignore").read()
    for h in hallmarks:
        if h.lower() in txt.lower():
            hits.setdefault(h, []).append(os.path.relpath(py, REPO))
rows.append(["pipeline_code_hallmarks_found", "repo-wide", str(bool(hits)), str(hits)])

with open(os.path.join(OUT, "eval_artefacts.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["check", "location", "result", "detail"])
    w.writerows(rows)
    for r in rows:
        print(r)
