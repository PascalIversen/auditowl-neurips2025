"""Checks that run-mujococar.py references args.ra_buffer_size that no add_argument defines (supports finding mujococar-undefined-ra-buffer-size)."""
import ast, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "mahaozhe__CenRA")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

results = []
for fname in ["run-mujococar.py", "run-3dpickup.py", "run-2dmaze.py"]:
    path = os.path.join(REPO, fname)
    tree = ast.parse(open(path).read())
    dests = set()
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                    dests.add(a.value[2:].replace("-", "_"))
        # args.<name> attribute accesses
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
            used.add(node.attr)
    undefined = sorted(u for u in used if u not in dests)
    results.append((fname, undefined))

with open(os.path.join(OUT, "mujococar_arg.txt"), "w") as f:
    for fname, undefined in results:
        line = f"{fname}: args.* accessed but never defined via add_argument -> {undefined}"
        print(line)
        f.write(line + "\n")
