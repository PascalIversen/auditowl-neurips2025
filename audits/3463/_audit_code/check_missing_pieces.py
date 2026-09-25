"""Deterministic existence checks for paper-claimed components in the released code.
Supports findings: 'channel-discard-missing', 'dinov2-integration-missing', 'pca-dead-code'."""
import os, re

ROOT = os.path.join(os.path.dirname(__file__),
    "extracted", "Supplementary Material", "code_extracted", "code")

def read(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read()

py_files = []
for dirpath, _, files in os.walk(ROOT):
    if "__pycache__" in dirpath or "__MACOSX" in dirpath:
        continue
    for fn in files:
        if fn.endswith(".py"):
            py_files.append(os.path.join(dirpath, fn))

allsrc = "".join(open(p, encoding="utf-8").read() for p in py_files)

# 1) Channel discard: paper's contribution (Table 4 +1.8% on SPair@0.10). Look for any
#    zeroing-out of channels / 'discard' logic.
print("=== channel discard ===")
print("  'discard' occurrences in .py:", len(re.findall(r"discard", allsrc, re.I)))
print("  channel-zeroing patterns ([..]=0):",
      len(re.findall(r"\[[^\]]*\]\s*=\s*0\b", allsrc)))

# 2) DINOv2 integration (the dagger rows: DiTF+DINOv2, the SOTA headline numbers).
print("=== DINOv2 integration ===")
print("  case-insensitive 'dinov2'/'dino' identifiers used as a model:",
      len(re.findall(r"dinov2", allsrc, re.I)))

# 3) PCA fusion (paper Eq.14, output dim 1280). Is pca_feature_pair ever CALLED?
eval_src = read("eval_spair.py")
defs = len(re.findall(r"def pca_feature_pair", eval_src))
calls = len(re.findall(r"(?<!def )pca_feature_pair\s*\(", eval_src))
print("=== PCA fusion ===")
print(f"  pca_feature_pair defined: {defs}, called: {calls}")
print("  q (output dim) in def:", re.search(r"def pca_feature_pair\(.*q=(\d+)", eval_src).group(1))

# 4) Modules referenced by shipped .pyc but absent as .py
print("=== pyc-only modules ===")
flux_dir = os.path.join(ROOT, "src", "flux")
for m in ["feat_flux_v1", "feat_flux_v1_class_free", "feat_flux_cross_image", "feat_dinov2"]:
    print(f"  {m}.py present: {os.path.exists(os.path.join(flux_dir, m+'.py'))}")
