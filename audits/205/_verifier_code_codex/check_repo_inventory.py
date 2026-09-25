"""Inventories the repo and greps for scripts that could produce paper artefacts.

Supports the traceability table and findings `no-table-pipeline`,
`no-image-temperature-learning`, `densenet-temps-placeholder`.
Checks for: code computing Table 1/2 (CE across 4 tabular + 4 image models,
two perturbation types), Table 3 sensitivity, Figures 2/3/4, regression
experiments (Appendix C), and whether image temperatures are learned anywhere.
"""
import os, re, subprocess

repo = os.path.join(os.path.dirname(__file__), "..", "code", "thomdeck__recalx")
repo = os.path.abspath(repo)

files = []
for root, _, fs in os.walk(repo):
    if ".git" in root:
        continue
    for fn in fs:
        files.append(os.path.relpath(os.path.join(root, fn), repo))

def grep(pattern):
    hits = []
    for f in files:
        p = os.path.join(repo, f)
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        if re.search(pattern, txt, re.I):
            hits.append(f)
    return hits

terms = {
    "sensitivity (Table 3 / Quantus / Sens metric)": r"sensitivit|max_sens|avg_sens|quantus",
    "remove-and-retrain / ROAR": r"retrain|roar|remove.?and.?retrain",
    "LIME explainer usage": r"\bLime\b|lime",
    "KernelSHAP / Feature Ablation": r"kernelshap|kernel_shap|FeatureAblation",
    "Popordanoska/[51] kernel CE estimator": r"popordanoska|kernel.*calibration|kde.*calibration|ece_kde",
    "image temperature learning script": r"densenet_temperatures|learn.*temperature.*image|fit.*temperature",
    "blur perturbation": r"blur|gaussian_blur",
    "tabular ResNet (Gorishniy)": r"resnet.*tabular|tabular.*resnet|FTTransformer|rtdl",
    "covertype/credit/pol datasets": r"covertype|credit|\bpol\b",
    "ViT / SigLIP models": r"\bvit\b|siglip",
    "regression / quantile CE (Appendix C)": r"quantile|regression|kuleshov",
    "ImageNet val data loader": r"imagenet.*val|ILSVRC|val.*imagenet",
}

lines = []
lines.append("=== Repo files (non-.git) ===")
for f in sorted(files):
    lines.append("  " + f)
lines.append("")
lines.append("=== Grep for artefact-producing code ===")
for label, pat in terms.items():
    hits = grep(pat)
    lines.append(f"[{'FOUND' if hits else 'ABSENT'}] {label}: {hits}")

out = os.path.join(os.path.dirname(__file__), "out", "inventory.txt")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
