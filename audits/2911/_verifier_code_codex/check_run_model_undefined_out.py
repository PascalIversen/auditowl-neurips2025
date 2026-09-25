#!/usr/bin/env python3
"""Checks that ImportanceRenderer.run_model uses/returns 'out' without ever
assigning it (decoder never called) -> fatal NameError. Supports finding
'run-model-undefined-out'. READ-ONLY: only parses the repo source via AST."""
import ast, os, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
       "code", "aigc3d__HyPlaneHead"))
SRC = os.path.join(REPO, "training", "volumetric_rendering", "renderer.py")

src = open(SRC).read()
tree = ast.parse(src)

result = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "run_model":
        assigned, loaded, subscript_targets, decoder_called = set(), set(), set(), False
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store):
                    assigned.add(n.id)
                elif isinstance(n.ctx, ast.Load):
                    loaded.add(n.id)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                subscript_targets.add(n.value.id)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "decoder":
                decoder_called = True
        result = dict(
            func_line_start=node.lineno,
            func_line_end=node.body[-1].lineno,
            out_assigned=("out" in assigned),
            out_used=("out" in loaded or "out" in subscript_targets),
            decoder_called=decoder_called,
            sampled_features_assigned=("sampled_features" in assigned),
            sampled_features_used=("sampled_features" in loaded),
        )

result["VERDICT"] = (
    "BUG: 'out' used/returned but never assigned; decoder never called; "
    "sampled_features computed but discarded -> run_model raises NameError"
    if (result.get("out_used") and not result.get("out_assigned")
        and not result.get("decoder_called"))
    else "OK"
)

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "out", "run_model_undefined_out.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
