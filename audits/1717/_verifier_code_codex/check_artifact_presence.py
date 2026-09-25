"""Scan repo .py + decoded .ipynb source for keywords tied to paper artefacts.

Supports findings: missing-bert-text-modality, missing-saliency-baseline,
missing-yolo-selectivity, missing-braindive, missing-pca-feature-extraction,
ridge-baseline-fixed-penalty. Read-only; writes out/artifact_presence.txt.
"""
import json, os, re, glob

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "Hosseinadeli__transformer_brain_encoder")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# collect all source text: .py files + code cells of .ipynb
sources = {}
for p in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
    sources[os.path.relpath(p, REPO)] = open(p, encoding="utf-8", errors="ignore").read()
for p in glob.glob(os.path.join(REPO, "**", "*.ipynb"), recursive=True):
    nb = json.load(open(p, encoding="utf-8", errors="ignore"))
    txt = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            txt.append("".join(c.get("source", [])))
    sources[os.path.relpath(p, REPO)] = "\n".join(txt)

keywords = {
    "BERT backbone (Table 6)": r"\bbert\b",
    "BLIP captioning (text modality)": r"\bblip\b",
    "DeepGaze / saliency baseline (Tab1/3/5)": r"deepgaze|saliency",
    "YOLO category selectivity (Tab7-10)": r"\byolo",
    "BrainDiVE image generation (Fig S9/S10)": r"braindive|diffus",
    "PCA .fit feature extraction (PCA+regression)": r"PCA\([^)]*\)\.fit|IncrementalPCA\([^)]*\)\.fit|\.partial_fit\(",
    "Ridge / RidgeCV / alpha grid search": r"ridgecv|gridsearchcv|alphas\s*=|np\.logspace|sklearn[^\n]*Ridge",
    "k-fold / cross-validation primitive": r"kfold|cross_val|StratifiedKFold",
}

lines = []
for label, pat in keywords.items():
    hits = []
    rx = re.compile(pat, re.IGNORECASE)
    for fn, txt in sources.items():
        for i, ln in enumerate(txt.splitlines(), 1):
            if rx.search(ln):
                hits.append(f"    {fn}:{i}: {ln.strip()[:90]}")
    status = "PRESENT" if hits else "ABSENT"
    lines.append(f"[{status}] {label}")
    lines.extend(hits[:8])

report = "\n".join(lines)
print(report)
open(os.path.join(OUT, "artifact_presence.txt"), "w").write(report + "\n")
