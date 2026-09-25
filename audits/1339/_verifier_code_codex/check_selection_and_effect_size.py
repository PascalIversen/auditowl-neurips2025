"""Shows (a) the released config == the best-collision row of the Tab. A.4 sweep, (b) val and test read the SAME nuScenes annotation file, and (c) how few collision events the headline gain corresponds to (supports findings: tuned-on-reporting-split, single-run-no-variance)."""
import os, re, csv

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", "code", "AIR-DISCOVER__E-cubed-AD__E-VAD"))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
CFG = os.path.join(REPO, "projects/configs/EAD/EAD_based_pretrain.py")
cfg_lines = open(CFG).read().splitlines()

# ---------------------------------------------------------------- (a) sweep
# Tab. A.4, "Interact with the Planning Features" (the framework the repo ships).
# (layers, dropout, heads) -> (L2 1s,2s,3s,Avg, Col 1s,2s,3s,Avg)   [paper Tab. A.4]
SWEEP = [
    ((4, 0.10, 8), (0.33, 0.59, 0.93, 0.62), (0.06, 0.14, 0.41, 0.20)),
    ((1, 0.10, 8), (0.39, 0.70, 1.07, 0.72), (0.03, 0.14, 0.44, 0.20)),
    ((2, 0.10, 8), (0.34, 0.59, 0.91, 0.61), (0.12, 0.20, 0.32, 0.21)),
    ((6, 0.10, 8), (0.38, 0.64, 0.96, 0.66), (0.09, 0.18, 0.39, 0.22)),
    ((4, 0.05, 8), (0.38, 0.67, 1.04, 0.70), (0.07, 0.15, 0.41, 0.21)),
    ((4, 0.15, 8), (0.34, 0.61, 0.96, 0.63), (0.06, 0.17, 0.40, 0.21)),
    ((4, 0.10, 4), (0.35, 0.62, 0.96, 0.64), (0.06, 0.13, 0.36, 0.18)),
    ((4, 0.10, 2), (0.33, 0.59, 0.93, 0.62), (0.06, 0.14, 0.41, 0.20)),
]
TABLE1_E3AD_VAD_BASE = ((0.35, 0.62, 0.96, 0.64), (0.06, 0.13, 0.36, 0.18))

print("=== (a) Tab. A.4 sweep (framework 3) vs the headline Tab. 1 row ===\n")
print(f"{'layers':>7} {'dropout':>8} {'heads':>6} | {'L2 avg':>7} {'Col avg':>8}  {'':2}")
print("-" * 52)
best_col = min(r[2][3] for r in SWEEP)
best_l2 = min(r[1][3] for r in SWEEP)
for (L, D, H), l2, col in SWEEP:
    tag = []
    if col[3] == best_col: tag.append("<- min collision")
    if l2[3] == best_l2:   tag.append("<- min L2")
    print(f"{L:>7} {D:>8} {H:>6} | {l2[3]:>7.2f} {col[3]:>8.2f}  {' '.join(tag)}")

sel = [r for r in SWEEP if (r[1], r[2]) == TABLE1_E3AD_VAD_BASE]
print(f"\nTab. 1 'E3AD(VAD-Base)' row = L2 {TABLE1_E3AD_VAD_BASE[0]}, Col {TABLE1_E3AD_VAD_BASE[1]}")
print(f"  -> matches Tab. A.4 sweep row(s): {[r[0] for r in sel]}")
print(f"  -> that row has the MINIMUM collision ({best_col}) but NOT the minimum L2 "
      f"({TABLE1_E3AD_VAD_BASE[0][3]} vs best {best_l2})")

# what the shipped config actually sets for the eeg decoder
blk = "\n".join(cfg_lines[115:131])
L = int(re.search(r"num_layers=(\d+)", blk).group(1))
H = int(re.search(r"num_heads=(\d+)", blk).group(1))
D = float(re.search(r"dropout=([\d.]+)", blk).group(1))
print(f"\nprojects/configs/EAD/EAD_based_pretrain.py eeg_decoder (lines 116-131): "
      f"num_layers={L}, dropout={D}, num_heads={H}")
print(f"  -> the ONE released config is exactly the best-collision sweep row {(L, D, H)}.")

n_sweep = len(SWEEP) + 6 + 3 + 3   # Tab A.4 + Tab A.3 + Tab A.5 + the 3 frameworks in Tab 4
print(f"\nTotal configurations the appendix reports on this split: "
      f"{len(SWEEP)} (A.4) + 6 (A.3) + 3 (A.5) + 3 frameworks (Tab. 4) = {n_sweep}")

# ---------------------------------------------------------------- (b) split
print("\n=== (b) which annotation file do val and test read? ===\n")
for i, l in enumerate(cfg_lines, 1):
    if "ann_file" in l or re.match(r"\s+(val|test|train)=dict", l):
        print(f"  {i:>4}| {l.rstrip()}")
val_ann = [l for l in cfg_lines if "ann_file" in l]
print("\n  -> val and test both point at 'vad_nuscenes_infos_temporal_val.pkl':",
      sum("temporal_val.pkl" in l for l in val_ann), "of", len(val_ann), "ann_file lines.")
print("  -> there is no third, held-out split; the split the sweep is scored on IS")
print("     the split Tab. 1 reports.")

# ---------------------------------------------------------------- (c) effect
print("\n=== (c) how many collision events does the headline gain represent? ===\n")
# nuScenes val = 150 scenes; VAD scores only samples with fut_valid_flag.
DENOMS = [6019, 5000, 4819, 4000]
CASES = [
    ("VAD-Base -> E3AD(VAD-Base), avg collision", 0.22, 0.18),
    ("VAD-Base -> E3AD(VAD-Base), 3 s collision", 0.41, 0.36),
    ("UniAD -> E3AD(UniAD), avg collision",       0.31, 0.23),
    ("best vs 2nd-best sweep row (A.4), avg col", 0.20, 0.18),
]
rows = []
print(f"{'COMPARISON':46} {'delta pp':>9} " + " ".join(f"{'n='+str(d):>9}" for d in DENOMS))
print("-" * 46 + "-" * 10 + "-" * (10 * len(DENOMS)))
for name, a, b in CASES:
    d = a - b
    counts = [d / 100.0 * n for n in DENOMS]
    print(f"{name:46} {d:>9.2f} " + " ".join(f"{c:>9.1f}" for c in counts))
    rows.append([name, a, b, round(d, 3)] + [round(c, 2) for c in counts])

print("\n  -> cells are the number of scored samples whose collision verdict must flip")
print("     to produce the reported difference. Every headline open-loop gain in the")
print("     paper is a single-seed difference of ~2-5 collision events.")

with open(os.path.join(OUT, "selection_and_effect_size.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comparison", "baseline_pct", "e3ad_pct", "delta_pp"] + [f"events_at_n_{d}" for d in DENOMS])
    w.writerows(rows)
print(f"\nwrote {os.path.join(OUT, 'selection_and_effect_size.csv')}")
