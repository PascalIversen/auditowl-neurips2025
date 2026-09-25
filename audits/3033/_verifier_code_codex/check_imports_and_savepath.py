"""Checks (1) example.ipynb imports names absent from utils.hooks (ImportError bug),
and (2) SAETrainingConfig.save_path references a free name `save_path_base` (NameError bug).
Supports findings: example-nb-import-error, savepath-free-variable. READ-ONLY."""
import ast, json, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "surkovv__sdxl-unbox")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ---- (1) hooks.py defined top-level names vs example.ipynb imports ----
hooks_path = os.path.join(REPO, "utils", "hooks.py")
hooks_src = open(hooks_path).read()
hooks_tree = ast.parse(hooks_src)
defined = sorted(
    n.name for n in hooks_tree.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
)

nb = json.load(open(os.path.join(REPO, "example.ipynb")))
imported_from_utils = []
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "utils":
            imported_from_utils += [a.name for a in node.names]

missing = [name for name in imported_from_utils if name not in defined]

# ---- (2) save_path free variable ----
utils_path = os.path.join(REPO, "SAE", "sae_utils.py")
utils_tree = ast.parse(open(utils_path).read())
save_path_uses_self = None
field_names = set()
for cls in utils_tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name == "SAETrainingConfig":
        for item in cls.body:
            # dataclass fields are AnnAssign at class level
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_names.add(item.target.id)
            if isinstance(item, ast.FunctionDef) and item.name == "save_path":
                # find Name nodes used (not attribute) named save_path_base
                names = {n.id for n in ast.walk(item) if isinstance(n, ast.Name)}
                attrs = {n.attr for n in ast.walk(item) if isinstance(n, ast.Attribute)}
                save_path_uses_self = {
                    "uses_bare_save_path_base": "save_path_base" in names,
                    "uses_self_save_path_base": "save_path_base" in attrs,
                    "save_path_base_is_instance_field": "save_path_base" in field_names,
                }

result = {
    "hooks_defined_names": defined,
    "example_nb_imports_from_utils": imported_from_utils,
    "example_nb_missing_imports": missing,
    "example_nb_import_error": bool(missing),
    "save_path_check": save_path_uses_self,
}
with open(os.path.join(OUT, "imports_and_savepath.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
