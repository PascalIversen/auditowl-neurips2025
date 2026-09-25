"""Checks that the regularization loop's isinstance() guard can never fire:
the trainer tests against waveletLoRAAdapter.LinearWaveletFilter, but the
modules actually inserted into the model are the run-scripts' own
LinearWaveletFilter class. Supports finding `reg-loss-never-applied`.

Pure source/AST analysis (no torch needed). Output -> out/reg_isinstance.txt
"""
import ast, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code",
                    "Applied-Machine-Learning-Lab__DEAL", "src")

def classes_defined(path):
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

def imports_of(path):
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.append((n.module, a.name))
    return out

def instantiates(path, name):
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
            return True
    return False

lines = []
adapter_classes = classes_defined("waveletLoRAAdapter.py")
t5_classes = classes_defined("T5_run_wavelet.py")
llama_classes = classes_defined("Llama3_run_wavelet.py")
trainer_imports = imports_of("uie_trainer_lora.py")

lines.append(f"waveletLoRAAdapter.py defines LinearWaveletFilter: {'LinearWaveletFilter' in adapter_classes}")
lines.append(f"T5_run_wavelet.py defines its OWN LinearWaveletFilter:    {'LinearWaveletFilter' in t5_classes}")
lines.append(f"Llama3_run_wavelet.py defines its OWN LinearWaveletFilter: {'LinearWaveletFilter' in llama_classes}")

trainer_imports_adapter_lwf = ("waveletLoRAAdapter", "LinearWaveletFilter") in trainer_imports
lines.append(f"uie_trainer_lora.py imports LinearWaveletFilter FROM waveletLoRAAdapter: {trainer_imports_adapter_lwf}")

t5_inst = instantiates("T5_run_wavelet.py", "LinearWaveletFilter")
llama_inst = instantiates("Llama3_run_wavelet.py", "LinearWaveletFilter")
lines.append(f"T5_run_wavelet.py instantiates its OWN LinearWaveletFilter:    {t5_inst}")
lines.append(f"Llama3_run_wavelet.py instantiates its OWN LinearWaveletFilter: {llama_inst}")

# Does the run script import the adapter's class at all?
t5_imports = imports_of("T5_run_wavelet.py")
llama_imports = imports_of("Llama3_run_wavelet.py")
t5_imports_adapter = ("waveletLoRAAdapter", "LinearWaveletFilter") in t5_imports
llama_imports_adapter = ("waveletLoRAAdapter", "LinearWaveletFilter") in llama_imports
lines.append(f"T5_run_wavelet imports adapter's LinearWaveletFilter:    {t5_imports_adapter}")
lines.append(f"Llama3_run_wavelet imports adapter's LinearWaveletFilter: {llama_imports_adapter}")

mismatch = (trainer_imports_adapter_lwf and t5_inst and llama_inst
            and not t5_imports_adapter and not llama_imports_adapter)
lines.append("")
lines.append(f"VERDICT: isinstance() guard in trainer can NEVER match model modules = {mismatch}")
lines.append("=> reg_loss stays 0; the lambda1/lambda2 regularization (Eq.12) is never applied.")

out = "\n".join(lines)
print(out)
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "out", "reg_isinstance.txt"), "w") as f:
    f.write(out + "\n")
