import json
from pathlib import Path


VERDICTS = {
    "missing-multi-arch-and-tables": {
        "verdict": "keep",
        "reason": "The cited InternVL helper is present, paper text reports the named architectures and tables, and scoped repo searches found no LaViLa/OpenLLaVA/InternVL drivers or named result-table/EGTEA/runtime/hallucination/overlap pipelines beyond OpenFlamingo-style metric helpers.",
        "changed": "nothing",
    },
    "missing-deps-and-data": {
        "verdict": "keep",
        "reason": "The quoted checkpoint uses hardcoded placeholder paths; the repo has no requirements/environment/setup file, README references missing environment.yml, no weights/download artifacts were found, and a repo-wide path search counted 124 placeholder/home paths.",
        "changed": "nothing",
    },
    "kl-regularizer-double-softmax": {
        "verdict": "keep",
        "reason": "The cited KL code softmaxes the already normalized gaze proportions and model attention and calls kl_div with log_target=True on probabilities; the checker ran and showed the target is distorted and not a valid log-target distribution.",
        "changed": "nothing",
    },
    "checkpoint-selected-on-train-loss": {
        "verdict": "keep",
        "reason": "The cited driver compares trainloss to best_loss and saves on that value while the validation dataloader lines are commented out.",
        "changed": "nothing",
    },
    "reg-value-hardcoded": {
        "verdict": "keep",
        "reason": "The cited train_gaze_attention code hardcodes reg_value = 100, and searches of the driver/utils show no CLI/config argument or 0/1000 code path.",
        "changed": "nothing",
    },
    "teacher-forced-eval": {
        "verdict": "keep",
        "reason": "The evaluator calls gaze_score2, whose cited lines do a teacher-forced forward pass on input_ids; data.py builds input_ids from image tokens plus annotations, and the checker confirmed no generate call in this path.",
        "changed": "nothing",
    },
    "sliding-window-random-split-leakage": {
        "verdict": "keep",
        "reason": "The loader creates consecutive windows by incrementing i by one and then randomly splits the pooled sequences with train_test_split, with no video/time grouping.",
        "changed": "nothing",
    },
}


root = Path(__file__).resolve().parents[1]
source = root / "findings.json"
target = root / "findings_verified_codex_stage1.json"

data = json.loads(source.read_text())
findings = data["findings"] if isinstance(data, dict) else data

for finding in findings:
    key = finding.get("id_local") or str(finding.get("id", "")).rsplit("/", 1)[-1]
    finding.update(VERDICTS[key])

target.write_text(json.dumps(data, indent=2) + "\n")
