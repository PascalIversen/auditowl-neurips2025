"""Checks that no ASR/probing/baseline/utility evaluation code exists in the repo (supports findings: missing-asr-eval, missing-probing, missing-baselines)."""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "code" / "Renovamen__ShiftDC"
OUT = Path(__file__).resolve().parent / "out" / "eval_code_absent.txt"

# regex keyword -> what paper artefact it would back
KEYWORDS = {
    "ASR / rejection scoring": r"\b(asr|attack_success|rejection_keyword|reject|refus)\b",
    "linear probing classifier": r"\b(logisticregression|sklearn|linear[_ ]?prob|probe|classif|confusion)\b",
    "cosine similarity (Fig 2 right)": r"cosine|cos_sim|cosine_similarity",
    "ECSO baseline": r"\becso\b",
    "AdaShield baseline": r"adash(ie|ei)ld",
    "utility benchmarks": r"\b(mmbench|mm-vet|mmvet|\bmme\b|mossbench|jailbreakv|figstep)\b",
    "binary safety classification prompt": r"is the given request harmful",
}

py_files = [p for p in REPO.rglob("*.py") if ".git" not in p.parts]
text = ""
for p in py_files:
    text += f"\n# {p.relative_to(REPO)}\n" + p.read_text(errors="ignore")

lines = []
lines.append(f"Scanned {len(py_files)} python files under {REPO.relative_to(REPO.parents[2])}")
for label, pat in KEYWORDS.items():
    hits = re.findall(pat, text, flags=re.IGNORECASE)
    lines.append(f"{'PRESENT' if hits else 'ABSENT '} | {label:42s} | matches={len(hits)}")

# also: does any script write/compute a numeric metric file?
metric_writes = re.findall(r"(asr|accuracy|score)\s*=", text, flags=re.IGNORECASE)
lines.append(f"\nnumeric-metric assignments (asr=/accuracy=/score=): {len(metric_writes)}")

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {OUT}")
