"""Extract the default run_hiflow.py config and compare cascade/LoRA settings to the paper. Supports the `difference` findings."""
import os, re, ast

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "Bujiazi__HiFlow"))
src = open(os.path.join(REPO, "run_hiflow.py")).read()

def grab(name):
    m = re.search(rf"{name}\s*=\s*(\[[^\]]*\]|[^\n,#]+)", src)
    return m.group(1).strip() if m else None

facts = {
    "loads_lora_by_default": bool(re.search(r"^\s*pipe\.load_lora_weights\(", src, re.M)),
    "lora_call_commented": bool(re.search(r"#\s*pipe\.load_lora_weights\(", src, re.M)),
    "height/width base": (grab("height"), grab("width")),
    "target_heights": grab("target_heights"),
    "target_widths": grab("target_widths"),
    "num_inference_steps_highres": grab("num_inference_steps_highres"),
    "alphas": grab("alphas"),
    "betas": grab("betas"),
}
# cascade resolutions: base then targets
print("=== run_hiflow.py default config ===")
for k, v in facts.items():
    print(f"{k}: {v}")

# Paper claim: tau = [0.6,0.3,0.3] for 1K->2K->3K->4K cascade (4 stages, 3 upscales)
# Code: target_heights=[2048,4096] -> 1K->2K->4K (3 stages, 2 upscales), no 3K
th = ast.literal_eval(facts["target_heights"])
print("\n=== cascade comparison ===")
print(f"paper cascade stages: 1K -> 2K -> 3K -> 4K  (3 upscale steps)")
print(f"code cascade stages:  1024 -> {' -> '.join(str(x) for x in th)}  ({len(th)} upscale steps)")
print(f"3K (3072) stage present in code default: {3072 in th}")
