"""Deterministic checks for ReplaceMe repo: missing __init__.py, thri attribute bug,
dataset handlers vs paper-claimed calibration datasets, ViT/CLIP code presence.
Supports findings: missing-init-py, thri-attribute-bug, missing-vit-clip-code,
missing-orca-generated-handler."""
import ast
import os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "mts-ai__ReplaceMe")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

lines = []

# 1. __init__.py presence
pkg = os.path.join(REPO, "ReplaceMe")
has_init = os.path.exists(os.path.join(pkg, "__init__.py"))
has_initpy = os.path.exists(os.path.join(pkg, "init.py"))
lines.append(f"ReplaceMe/__init__.py exists: {has_init}")
lines.append(f"ReplaceMe/init.py exists (misnamed?): {has_initpy}")

# 2. LowerTriangularLinear attribute check
utils_src = open(os.path.join(pkg, "utils.py")).read()
tree = ast.parse(utils_src)
ltl_attrs = set()
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "LowerTriangularLinear":
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                ltl_attrs.add(sub.attr)
lines.append(f"LowerTriangularLinear self.* attributes defined: {sorted(ltl_attrs)}")
lines.append(f"'triangular_weight' referenced in utils.py: {'triangular_weight' in utils_src}")
lines.append(f"'triangular_weight' defined as attribute: {'triangular_weight' in ltl_attrs}")

# 3. dataset handlers
handlers = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "get_calib_dataloader":
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict):
                for k in sub.keys:
                    if isinstance(k, ast.Constant):
                        handlers.append(k.value)
lines.append(f"dataset handler keys: {handlers}")
paper_calib = ["FineWeb", "SlimOrca", "orca_generated", "Arcee", "Aya", "4K SlimOrca + 4K Fineweb (fineweb_and_orca)"]
lines.append(f"paper-mentioned calibration sources: {paper_calib}")
lines.append(f"orca_generated handler present: {'orca_generated' in handlers}")
lines.append(f"Aya / 66-language handler present: {any('aya' in str(h).lower() for h in handlers)}")

# 4. ViT/CLIP code presence anywhere in repo
vit_hits = []
for root, _, files in os.walk(REPO):
    if ".git" in root:
        continue
    for f in files:
        if f.endswith((".py", ".yaml", ".yml")):
            p = os.path.join(root, f)
            txt = open(p, errors="ignore").read().lower()
            if any(t in txt for t in ["clip", "vit", "vision", "coco", "eurosat", "mimic"]):
                vit_hits.append(os.path.relpath(p, REPO))
lines.append(f"files mentioning CLIP/ViT/vision/coco/eurosat/mimic: {vit_hits}")

with open(os.path.join(OUT, "repo_structure.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
