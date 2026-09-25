"""AST check: EADHead advertises eeg_decoder=None but calls self.eeg_decoder(...) with no None-guard, so the no-cognition ablation baseline cannot be configured (supports finding: no-eeg-config-crashes)."""
import ast, os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
                                    "AIR-DISCOVER__E-cubed-AD__E-VAD"))
HEAD = os.path.join(REPO, "projects/mmdet3d_plugin/EAD/EAD_head.py")
src = open(HEAD).read()
tree = ast.parse(src)

# 1. default value of the eeg_decoder kwarg on EADHead.__init__
default = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "__init__":
        args = node.args.kwonlyargs + node.args.args
        for a, d in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
            if a.arg == "eeg_decoder":
                default = ast.unparse(d)
print(f"EADHead.__init__ default for eeg_decoder = {default}")

# 2. record, for every attribute access on self.eeg_decoder, whether any enclosing
#    statement is an `if self.eeg_decoder is not None` guard.
class Ctx(ast.NodeVisitor):
    def __init__(self):
        self.guard_depth = 0
        self.uses = []           # (lineno, kind, guarded?)
    def _is_guard(self, test):
        return "eeg_decoder" in ast.unparse(test)
    def visit_If(self, node):
        g = self._is_guard(node.test)
        if g:
            self.guard_depth += 1
        for st in node.body:
            self.visit(st)
        if g:
            self.guard_depth -= 1
        for st in node.orelse:
            self.visit(st)
    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "eeg_decoder" and \
           isinstance(f.value, ast.Name) and f.value.id == "self":
            self.uses.append((node.lineno, "CALL self.eeg_decoder(...)", self.guard_depth > 0))
        self.generic_visit(node)
    def visit_Attribute(self, node):
        if node.attr == "eeg_decoder" and isinstance(node.value, ast.Name) and node.value.id == "self":
            self.uses.append((node.lineno, "attr  self.eeg_decoder", self.guard_depth > 0))
        self.generic_visit(node)

# walk each function body separately so guard depth resets per function
c = Ctx()
c.visit(tree)
seen = {}
for lineno, kind, guarded in c.uses:
    seen.setdefault(lineno, (kind, guarded))

lines = src.splitlines()
print(f"\n{'LINE':>6}  {'GUARDED?':10} SOURCE")
print("-" * 100)
unguarded_calls = []
for lineno in sorted(seen):
    kind, guarded = seen[lineno]
    print(f"{lineno:>6}  {str(guarded):10} {lines[lineno-1].strip()[:80]}")
    if kind.startswith("CALL") and not guarded:
        unguarded_calls.append(lineno)

print()
if unguarded_calls:
    print(f"RESULT: self.eeg_decoder is CALLED without a None-guard at line(s) {unguarded_calls}.")
    print("        With the advertised default eeg_decoder=None this raises")
    print("        TypeError: 'NoneType' object is not callable.")
else:
    print("RESULT: every call site is guarded.")

# 3. is the planning-MLP input width conditional on the eeg branch?
for i, l in enumerate(lines, 1):
    if "ego_fut_dec_in_dim" in l and "=" in l and "self.embed_dims" in l:
        print(f"\nplanning-MLP input width, EAD_head.py:{i}: {l.strip()}")
        print(f"                            EAD_head.py:{i+1}: {lines[i].strip()}")
        break
print("  -> width is embed_dims*3 unconditionally (VAD upstream uses *2), i.e. the")
print("     third slot for the cognition feature is hardcoded into the checkpoint shape.")
