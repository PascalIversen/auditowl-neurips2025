"""Checks which paper artefacts have producing code: greps repo for Table 1 (grad norms),
Fig 4/timing, Fig 6 (SNR), LPIPS diversity, CIFAR LD3M path, and pruning-JSON dependency.
Supports findings: trace-table1-gradnorm-missing, trace-timing-missing, pruning-json-missing,
cifar-ld3m-path-missing. Read-only."""
import os, re, glob, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "Brian-Moser__prune_and_distill")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

py_files = glob.glob(os.path.join(REPO, "src", "*.py")) + [os.path.join(REPO, "get_losses_imagenet.py")]
text = {}
for f in py_files:
    try:
        text[f] = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        pass
allsrc = "\n".join(text.values())

checks = {}
# Table 1: gradient norms of Z vs T  (||dL/dZ|| across T=10..90)
checks["table1_gradnorm_code"] = bool(re.search(r"grad.*norm|norm.*grad|gradient.?norm", allsrc, re.I)
                                      and re.search(r"\bSNR\b|signal.to.noise", allsrc, re.I))
# Fig 6: SNR gradient flow analysis
checks["fig6_snr_code"] = bool(re.search(r"\bSNR\b|signal.to.noise", allsrc, re.I))
# Supp LPIPS diversity (0.386 vs 0.420)
checks["lpips_code"] = bool(re.search(r"lpips", allsrc, re.I))
# Fig 4 / timing & memory numbers: a script that REPORTS the table values (not just measures one run)
checks["timing_measure_fn"] = bool(re.search(r"time_measurement", allsrc))
checks["peak_mem_code"] = bool(re.search(r"max_memory_allocated|peak.*mem|memory_allocated", allsrc, re.I))
# CIFAR LD3M: load_ldm only handles ImageNet(cin256)/FFHQ; is there any CIFAR LDM load?
checks["ldm_cifar_load"] = bool(re.search(r"cifar.*\.ckpt|cifar.*ldm|ldm.*cifar", allsrc, re.I))
# pruning JSON dependency in the LD3M build path
checks["build_dataset_uses_json"] = bool(re.search(r"class_\{c\}_top_\{percent\}_\{order\}\.json", allsrc))

# pruning json files present in repo?
json_in_repo = glob.glob(os.path.join(REPO, "**", "class_*_top_*_*.json"), recursive=True)
checks["pruning_json_files_present"] = len(json_in_repo) > 0
checks["n_pruning_json_files"] = len(json_in_repo)

with open(os.path.join(OUT, "traceability.json"), "w") as f:
    json.dump(checks, f, indent=2)
for k, v in checks.items():
    print(f"{k}: {v}")
