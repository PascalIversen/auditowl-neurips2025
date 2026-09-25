"""Check existence of data/dependency files the code references. Supports findings
missing-dependency-spec, missing-qa-raw-json, missing-sentiment-vl-json, missing-precomputed-results."""
import os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "technion-cs-nlp__vlm-circuits-analysis")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

checks = {
    # dependency specs
    "requirements.txt": "requirements.txt",
    "environment.yml": "environment.yml",
    "setup.py": "setup.py",
    "pyproject.toml": "pyproject.toml",
    # data files referenced in code (NOT generation-only because referenced on backpatching path)
    "data/factual_recall/qa_raw.json": "data/factual_recall/qa_raw.json",
    "data/sentiment_analysis/sentiment_vl.json": "data/sentiment_analysis/sentiment_vl.json",
    "vqa questions json": "vqa/v2_OpenEnded_mscoco_val2014_questions.json",
}

lines = []
for name, rel in checks.items():
    p = os.path.join(REPO, rel)
    exists = os.path.exists(p)
    lines.append(f"{name}: {rel} -> EXISTS={exists}")

# precomputed result artefacts the figures notebook torch.load()s
res_pt = []
for root, _, files in os.walk(os.path.join(REPO, "data")):
    for f in files:
        if f.endswith(".pt"):
            res_pt.append(os.path.join(root, f))
lines.append(f"precomputed .pt result files in data/: {len(res_pt)}")

out_path = os.path.join(OUT, "missing_files.txt")
with open(out_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")

# CSV summary for findings evidence (row 0 = dependency specs, row 1 = data jsons)
import csv as _csv
csv_path = os.path.join(OUT, "missing_files.csv")
dep_present = any(os.path.exists(os.path.join(REPO, r)) for r in
                  ["requirements.txt", "environment.yml", "setup.py", "pyproject.toml"])
qa = os.path.exists(os.path.join(REPO, "data/factual_recall/qa_raw.json"))
sv = os.path.exists(os.path.join(REPO, "data/sentiment_analysis/sentiment_vl.json"))
with open(csv_path, "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(["check", "result"])
    w.writerow(["dependency_spec_present (requirements/environment/setup/pyproject)", dep_present])
    w.writerow(["qa_raw.json present", qa, "sentiment_vl.json present", sv])

print("\n".join(lines))
print(f"\nWrote {out_path} and {csv_path}")
