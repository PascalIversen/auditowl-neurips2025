"""Greps the whole repo for the experiments the paper reports (UniAD/GenAD/LAW/VAD-Tiny/CARLA-Bench2Drive/Singapore-Boston split/UniAD-style L2) to show none of them are implemented (supports findings: baselines-and-frameworks-absent, closed-loop-code-absent)."""
import os, re, subprocess, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
                                    "AIR-DISCOVER__E-cubed-AD__E-VAD"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# vision15/ is a vendored copy of torchvision, .git is history -> excluded from the search space
EXCLUDES = ["--exclude-dir=.git", "--exclude-dir=vision15", "--exclude-dir=__pycache__"]

PATTERNS = {
    "Tab.1/2  UniAD variant":            r"uniad|UniAD|track_head|occ_head|motion_head",
    "Tab.1    GenAD variant":            r"genad|GenAD",
    "Tab.1    LAW variant":              r"\bLAW\b|latent_world|law_model",
    "Tab.1    VAD-Tiny variant":         r"tiny|Tiny",
    "Tab.2/A.6 CARLA closed loop":       r"carla|CARLA|leaderboard|Bench2Drive|bench2drive",
    # NB: plain "location" matches torch's map_location everywhere, so match the
    # nuScenes geographic log-location names the split would have to use.
    "Tab.A.7  Singapore/Boston split":   r"singapore|Singapore|SINGAPORE|boston|Boston|BOSTON",
    "Tab.1    UniAD-style L2 (endpoint)": r"l2_endpoint|final_L2|uniad_metric",
    "Tab.3    expert/novice EEG ablation": r"expert|novice|Expert|Novice",
    "Tab.4/A.5 framework-1 AttentionGate": r"AttnGate|attention_gate|AttentionGate|TokenLearner|tokenlearner",
    "Tab.4/A.3 framework-2 ego-query xattn": r"ego_query_eeg|brain_ego_query|q_ego_brain",
    "Tab.A.2  doubled L2 loss run":      r"loss_plan_reg.*2\.0|double.*l2|l2.*double",
}

results = {}
print(f"REPO = {REPO}")
print("(searching all tracked files except .git/, vision15/ (vendored torchvision), __pycache__/)\n")
for label, pat in PATTERNS.items():
    cmd = ["grep", "-rInE", pat, REPO] + EXCLUDES
    p = subprocess.run(cmd, capture_output=True, text=True)
    hits = [l for l in p.stdout.splitlines() if l.strip()]
    results[label] = hits
    print(f"### {label}")
    print(f"    regex: {pat}")
    print(f"    hits : {len(hits)}")
    for h in hits[:6]:
        print("      ", h.replace(REPO + "/", "")[:170])
    if len(hits) > 6:
        print(f"       ... {len(hits)-6} more")
    print()

# how many runnable mmdet3d configs exist for the EAD model?
cfgdir = os.path.join(REPO, "projects/configs/EAD")
cfgs = sorted(os.listdir(cfgdir)) if os.path.isdir(cfgdir) else []
print(f"projects/configs/EAD/ contains {len(cfgs)} config(s): {cfgs}")

gi = open(os.path.join(REPO, ".gitignore")).read()
print("\n.gitignore contents (note the excluded baseline config dir):")
print(gi)

with open(os.path.join(OUT, "absent_experiments.json"), "w") as f:
    json.dump({k: v for k, v in results.items()}, f, indent=2)
print(f"wrote {os.path.join(OUT, 'absent_experiments.json')}")
