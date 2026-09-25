"""Checks which result-producing artefacts (training, DPO, metric eval) exist in the LeVo repo.
Supports findings: missing-training-code, missing-dpo-code, missing-eval-metric-code, missing-data.
Read-only; greps the repo tree for training/DPO/metric/preference-data signatures.
"""
import os, re, json, subprocess

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "code", "tencent-ailab__songgeneration"))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def py_files():
    for root, _, files in os.walk(REPO):
        if "/.git" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)

# Signatures we look for. Each maps to a paper component.
signatures = {
    "training_loop (loss.backward / training_step)": re.compile(r"\.backward\(\)|def training_step\b"),
    "DPO_loss (dpo loss / reference model logits)": re.compile(r"\bdpo[_ ]?loss\b|DPOTrainer|logsigmoid.*ref", re.I),
    "preference_data_construction": re.compile(r"win[_ ]?lose|preference[_ ]?pair|phoneme[_ ]?error.*40", re.I),
    "interpolation_merge (DNI param interpolation of 3 nets)": re.compile(r"interpolat.*(param|state_dict|model)|linear interpolation across", re.I),
    "FAD_metric": re.compile(r"\bFAD\b|frechet.*audio|audioldm_eval", re.I),
    "PER_metric": re.compile(r"phoneme.*error|\bPER\b", re.I),
    "MuQ_similarity_metric": re.compile(r"MuQ[-_ ]?MuLan|MuQ-T|MuQ-A", re.I),
    "Audiobox_aesthetic_metric": re.compile(r"audiobox|aesthetic.*(CE|CU|PC|PQ)", re.I),
    "reward_model_training": re.compile(r"reward[_ ]?model", re.I),
}

hits = {k: [] for k in signatures}
for path in py_files():
    rel = os.path.relpath(path, REPO)
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    for name, rx in signatures.items():
        for m in rx.finditer(text):
            line = text[:m.start()].count("\n") + 1
            hits[name].append(f"{rel}:{line}: {m.group(0)[:60]}")

summary = {name: {"count": len(v), "examples": v[:8]} for name, v in hits.items()}

# Data/eval-input artefacts referenced by paper
data_files = {
    "paper eval set (20 lyrics + 20 prompts, Appendix D)": None,
    "20000 LLM-generated lyrics (DPO, §3.4)": None,
    "60000 win-lose pairs (DPO, §3.4)": None,
}
present_jsonl = []
for root, _, files in os.walk(REPO):
    if "/.git" in root: continue
    for f in files:
        if f.endswith((".jsonl",)):
            p = os.path.relpath(os.path.join(root,f), REPO)
            n = sum(1 for _ in open(os.path.join(root,f), encoding="utf-8", errors="ignore"))
            present_jsonl.append(f"{p} ({n} lines)")

result = {
    "repo": os.path.relpath(REPO),
    "result_producing_code_signatures": summary,
    "jsonl_files_present": present_jsonl,
    "note": "Paper Table 1/2/3/5/6 metrics, DPO training, preference-data construction, and LeLM training are headline result producers.",
}
with open(os.path.join(OUT, "repo_completeness.json"), "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(result, indent=2))
