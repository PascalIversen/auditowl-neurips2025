"""Checks that all distillation/teacher scripts delete the test split and report on
the validation split, and that save_best selects the max-over-epochs eval metric.
Supports findings: eval-on-validation-not-test, model-selection-on-eval-set."""
import re, os, json

ROOT = os.path.join(os.path.dirname(__file__), "..", "code", "FaisalHamman__CoD", "text-classification")
files = ["ted_no_trainer.py", "ted_no_trainer_qwen.py", "learn_filters_glue_no_trainer.py", "teacher_trainer.py", "teacher_trainer_qwen.py"]
res = {}
for f in files:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        res[f] = "MISSING FILE"; continue
    txt = open(p).read()
    res[f] = {
        "deletes_test_split": bool(re.search(r'del raw_datasets\["test"\]', txt)),
        "eval_on_validation": bool(re.search(r'eval_dataset = processed_datasets\[.*validation', txt)),
        "save_best_takes_max": bool(re.search(r'>\s*best_eval', txt)),
    }
print(json.dumps(res, indent=2))
with open(os.path.join(os.path.dirname(__file__), "out", "eval_split.json"), "w") as fh:
    json.dump(res, fh, indent=2)
