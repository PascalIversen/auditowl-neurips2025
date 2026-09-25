#!/usr/bin/env python3
"""Deterministic grep-based presence checks for paper artefacts in the DiCoFlex repo.

Supports findings: hypervolume-not-computed, baselines-missing, german-credit-missing,
sensitivity-table-missing, model-selection-table-missing, std-dev-table-missing.
Read-only: only greps the repo under code/, writes a CSV to out/.
"""
import os
import re
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "code", "ofurman__DiCoFlex"))

def gather_py():
    files = []
    for root, _, names in os.walk(REPO):
        if ".git" in root:
            continue
        for n in names:
            if n.endswith(".py"):
                files.append(os.path.join(root, n))
    return files

def count_hits(pattern, files, flags=re.IGNORECASE):
    rx = re.compile(pattern, flags)
    hits = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        hits.append((os.path.relpath(f, REPO), i, line.strip()))
        except Exception:
            pass
    return hits

def main():
    files = gather_py()
    checks = {
        "hypervolume_computation": r"hypervol|pymoo|pareto|nondomin|hv\b",
        "baseline_dice":           r"\bDiCE\b|generate_dice|dice_ml",
        "baseline_cchvae":         r"cchvae|c_chvae",
        "baseline_revise":         r"\brevise\b",
        "baseline_tabcf":          r"tabcf",
        "baseline_wachter":        r"wachter",
        "baseline_carla":          r"\bcarla\b",
        "german_credit_experiment": r"german",
        "sensitivity_p_sweep_driver": r"p_values\s*=\s*\[.*0\.08|0\.25.*1\.0|sensitivity",
        "model_selection_nll_nice_realnvp": r"NICE|RealNVP|real_nvp",
        "alpha_actionability_penalty": r"\balpha\b.*=.*\b(10|1000)\b|actionab.*penalt",
    }
    rows = []
    for name, pat in checks.items():
        hits = count_hits(pat, files)
        rows.append({
            "check": name,
            "pattern": pat,
            "num_hits": len(hits),
            "sample_locations": "; ".join(f"{r[0]}:{r[1]}" for r in hits[:5]),
        })
    out = os.path.join(HERE, "out", "artifact_presence.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "pattern", "num_hits", "sample_locations"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Scanned {len(files)} .py files under {REPO}")
    for r in rows:
        print(f"{r['check']:34s} hits={r['num_hits']:3d}  {r['sample_locations']}")
    print(f"\nWrote {out}")

if __name__ == "__main__":
    main()
