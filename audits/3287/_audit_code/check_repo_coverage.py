"""Enumerates repo files and greps for any code that could PRODUCE the paper's reported numbers.

Supports findings `eval-pipeline-missing`, `cv-harness-missing`, `baseline-code-missing`,
`figure-table-scripts-missing`, `dependency-spec-missing`. Read-only: lists git-tracked files and
searches for fine-tuning / cross-validation / FAISS retrieval / metric / plotting / second-order TM
tokens. Saves to out/repo_coverage.txt.
"""
import os, subprocess, re

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "code", "EESI__carmania"))
out_dir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "repo_coverage.txt")

tracked = subprocess.check_output(["git", "-C", REPO, "ls-files"], text=True).splitlines()
py = [f for f in tracked if f.endswith(".py")]

# tokens that would indicate code producing each class of paper result
patterns = {
    # NOTE: 'fine-tuning' uses a tighter pattern; the bare token 'head' was dropped because it
    # matches attention-head code (head_dim, key_value_heads, ...) and yields false positives.
    "fine-tuning (classification head/Trainer)": r"(Trainer\b|classifier|num_labels|AutoModelForSequenceClassification|fine.?tune|finetune)",
    "cross-validation (5/10-fold)": r"(KFold|StratifiedKFold|cross_val|n_splits|fold)",
    "FAISS retrieval / best-hit": r"(faiss|IndexFlat|best.?hit|knn|retriev)",
    "metrics (F1/MCC/accuracy/BLEU/perplexity)": r"(f1_score|matthews|accuracy_score|bleu|perplexity|MCC|roc_auc)",
    "plotting / figures": r"(matplotlib|plt\.|seaborn|t-?SNE|TSNE|heatmap|savefig)",
    "second-order TM (Tables 10/11)": r"(second.?order|trigram|order\s*=\s*2|n\s*=\s*3|T\(3\)|3-?gram)",
    "data download / accession": r"(wget|curl|download|datasets\.load|GRCh38|hg38|MiBiG|MEGARes|CARD|requests\.get)",
    "baselines (HyenaDNA/Caduceus/etc.)": r"(HyenaDNA|Caduceus|MetaBERTa|ConvNova|Enformer|DNABERT|NucleotideTrans)",
    "Hamming similarity / long-range retention (Fig 4)": r"(hamming|similarity|stride)",
    "beta sweep / sensitivity (Table 9)": r"(beta.*sweep|for\s+beta|grid|sweep|0\.5.*1\.0.*5\.0)",
}

dep_files = [f for f in tracked if re.search(r"(requirements.*\.txt|setup\.py|pyproject\.toml|environment\.ya?ml|Pipfile|setup\.cfg)", f)]

lines = []
lines.append(f"REPO: {REPO}")
lines.append(f"git-tracked .py files ({len(py)}): {py}")
lines.append(f"dependency-spec files: {dep_files if dep_files else 'NONE'}")
lines.append("")
lines.append("Token search across all tracked .py (excluding notebooks, which are JSON):")
for label, pat in patterns.items():
    hits = []
    for f in py:
        try:
            txt = open(os.path.join(REPO, f), encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(pat, line, re.IGNORECASE):
                hits.append(f"{f}:{i}")
    lines.append(f"  [{ 'FOUND' if hits else 'ABSENT' }] {label}: {hits if hits else '-'}")

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
