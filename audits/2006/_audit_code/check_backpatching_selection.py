"""Document that best back-patching (src,dst,window) is selected on the SAME prompt set
the reported Table 1 accuracy is measured on, with no held-out split. Supports finding
backpatching-best-config-on-eval-set. Static evidence extraction (read-only)."""
import os, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "technion-cs-nlp__vlm-circuits-analysis"))
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)

script = open(os.path.join(REPO, "script_backpatching_experiment.py")).read()
nb = json.load(open(os.path.join(REPO, "figures_and_results_processing.ipynb")))
cell8 = "".join(nb["cells"][8]["source"])

lines = []
lines.append("script_backpatching_experiment.py main():")
lines.append("  - loads vl_prompts via load_dataset(..., correct_preds_only=False)[0]  (all prompts; index [0] = original_vl_prompts)")
lines.append("  - clean baseline accuracy measured on the SAME vl_prompts")
lines.append("  - grid over layer_window_size in {5,3,1} x src_layer_range x dst_layer_range")
lines.append("    each cell calls backpatching(model, args, vl_prompts, ...) -> accuracy on the SAME vl_prompts")
lines.append("  - no discovery/eval split for back-patching (comment line ~388: 'Doesn't matter, we don't split to discovery and test here')")
lines.append("")
lines.append("Notebook cell 8 (produces Table 1 numbers):")
lines.append("  - bp_best_acc = max accuracy over all configs via topk_2d(...,k=1) then sorted(...,reverse=True)[:1]")
lines.append("  - i.e. the single best (window,src,dst) cell on the evaluation set is reported")
lines.append("")
lines.append("Confirm 'no split' comment present in script: " +
             str("we don't split to discovery and test here" in script))
lines.append("Confirm 'max' selection present in cell8: " +
             str("reverse=True" in cell8 and "topk_2d" in cell8))

out = os.path.join(OUT, "backpatching_selection.txt")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print("\nWrote", out)
