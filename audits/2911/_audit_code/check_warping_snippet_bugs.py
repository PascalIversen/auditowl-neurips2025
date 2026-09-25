"""Checks the standalone near_equal_area_warping.py snippet for undefined
names (coordinates_sph, N) -> the file as shipped cannot run. Supports finding
'warping-snippet-undefined-names'. READ-ONLY: AST only."""
import ast, os, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
       "code", "aigc3d__HyPlaneHead"))
SRC = os.path.join(REPO, "training", "volumetric_rendering",
                   "near_equal_area_warping.py")
src = open(SRC).read()
tree = ast.parse(src)

result = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "sample_from_sphplane":
        params = {a.arg for a in node.args.args}
        assigned = {n.id for n in ast.walk(node)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        loaded = {n.id for n in ast.walk(node)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        bound = params | assigned | set(dir(__builtins__))
        # module-level names that are legitimately available
        module_names = {"torch", "math", "F", "nn", "np",
                        "cartesian_to_spherical", "denormalize_theta_phi",
                        "spherical_to_circle", "circle_polar2cartesian",
                        "cir2squ_mapping"}
        undefined = sorted(n for n in loaded
                           if n not in bound and n not in module_names)
        result = dict(
            func_line_start=node.lineno,
            func_line_end=node.body[-1].lineno,
            params=sorted(params),
            undefined_names_used=undefined,
        )

result["VERDICT"] = (
    f"BUG: undefined names referenced at runtime -> NameError: {result.get('undefined_names_used')}"
    if result.get("undefined_names_used") else "OK"
)

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "out", "warping_snippet_bugs.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
