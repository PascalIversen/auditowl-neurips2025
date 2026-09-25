"""Maps each quantitative paper artefact to repo code that COMPUTES it (not plots).
Greps the whole repo (py + ipynb sources) for the computation primitives each
artefact needs. Supports the coverage table + missing findings. READ-ONLY."""
import json, os, re

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "surkovv__sdxl-unbox")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# Gather all source text: .py files + code cells of .ipynb
blobs = {}
for root, _, files in os.walk(REPO):
    if os.sep + ".git" in root:
        continue
    for fn in files:
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, REPO)
        if fn.endswith(".py"):
            blobs[rel] = open(p, errors="ignore").read()
        elif fn.endswith(".ipynb"):
            try:
                nb = json.load(open(p, errors="ignore"))
            except Exception:
                continue
            src = []
            for c in nb.get("cells", []):
                if c.get("cell_type") == "code":
                    src.append("".join(c.get("source", [])))
            blobs[rel + "::codecells"] = "\n".join(src)

def find(pattern):
    rx = re.compile(pattern, re.I)
    return sorted(f for f, t in blobs.items() if rx.search(t))

artefacts = {
    "Fig2/Fig3/training EV (explained_variance fn)": find(r"\bexplained_variance\b|explained variance"),
    "Fig3 feature-overlap cosine similarity across timesteps": find(r"cosine_sim|F\.cosine|cosine_similarity|feature.?overlap"),
    "Fig5 LPIPS metric (RIEBench)": find(r"lpips"),
    "Fig5 CLIP-similarity metric (RIEBench)": find(r"clip.?score|clip.?sim|clip_similarity|CLIPModel|open_clip"),
    "Fig5/Fig6 grounded SAM2 masks": find(r"sam2|grounded.?sam|groundingdino|grounded"),
    "Fig5/Fig6 RIEBench / PIEBench harness": find(r"riebench|piebench|edit_categor|edit category"),
    "Fig5/Fig6 feature-selection score eq.(8)/(9)": find(r"src.*tgt|score.*concat|importance.?rank"),
    "Table1 per-block ablation LPIPS (ablate_block)": find(r"ablate_block"),
    "Table5 reconstruction (reconstruct_sae_hook)": find(r"reconstruct_sae"),
    "Table5 pixel Manhattan distance (mean/median)": find(r"manhattan|abs.*diff.*sum|L1|\.abs\(\).*sum"),
    "FLUX SAE training / layer-18 activations": find(r"flux|layer.?18|black-forest"),
    "Data collection (collect_latents)": find(r"run_with_cache|collect_latents|laion-coco"),
    "SAE training loop": find(r"training_loop_|train_sae"),
}

report = {k: v for k, v in artefacts.items()}
with open(os.path.join(OUT, "traceability.json"), "w") as f:
    json.dump(report, f, indent=2)
for k, v in report.items():
    status = "FOUND in " + ", ".join(v) if v else "*** NONE ***"
    print(f"- {k}: {status}")
