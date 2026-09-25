"""Check the random sub-circuit generator uses inclusive randint upper bounds equal to
n_layers / n_heads, allowing out-of-range layer/head indices. Supports finding
random-subcircuit-index-oob. Read-only static analysis of the source line."""
import os, re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "technion-cs-nlp__vlm-circuits-analysis"))
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)
src = open(os.path.join(REPO, "script_node_cross_modality_analysis.py")).read()

# Demonstrate python randint semantics: random.randint(a,b) includes b.
import random
random.seed(0)
n_layers = 28  # e.g. Qwen2-VL-7B decoder
samples = [random.randint(0, n_layers) for _ in range(100000)]
hit_oob = sum(1 for s in samples if s == n_layers)
maxv = max(samples)

lines = []
lines.append("Source uses random.randint(0, model.cfg.n_layers) and random.randint(0, model.cfg.n_heads):")
for m in re.finditer(r"random\.randint\(0, model\.cfg\.(n_layers|n_heads)\)", src):
    lines.append("  FOUND: " + m.group(0))
lines.append("")
lines.append("Python random.randint(a,b) is INCLUSIVE of b. Valid layer indices are 0..n_layers-1.")
lines.append(f"With n_layers={n_layers}: out of 100000 samples, max sampled index = {maxv}, "
             f"# equal to n_layers (out-of-range) = {hit_oob} ({hit_oob/1000:.2f}%)")
lines.append("Verdict: random baseline can place components at layer index == n_layers (and head == n_heads),"
             " which are out of range; such components silently match nothing in real circuits, "
             "biasing the random-baseline IoU/faithfulness downward.")

out = os.path.join(OUT, "random_subcircuit_oob.txt")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print("\nWrote", out)
