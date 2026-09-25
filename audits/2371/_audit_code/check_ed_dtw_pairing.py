"""Checks that ED/DTW pair original sample i with generated sample i by index,
and that the comparison silently truncates to min(n_orig) samples, supporting
finding `ed-dtw-index-pairing`. Read-only; imports the repo's own metric code.
Outputs CSV to out/.
"""
import os, sys
import numpy as np
import csv

# NOTE: importing utils.evaluation.distance_based_measures pulls in dtaidistance
# (a DTW C-extension) which is not installed. calculate_ed itself has no such
# dependency, so we reproduce its EXACT body verbatim from
# code/SDForger__neurips_supplemental/utils/evaluation/distance_based_measures.py:13-26
def calculate_ed(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0
        for j in range(n_series):
            distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_distance_eu += distance
        distance_eu.append(total_distance_eu / n_series)
    distance_eu = np.array(distance_eu)
    average_distance_eu = distance_eu.mean()
    return average_distance_eu

rng = np.random.default_rng(0)
# Build (n_samples, length, channels) like run_TSG_evaluation feeds after transpose
n_orig, L, C = 30, 100, 1
orig = rng.standard_normal((n_orig, L, C))

# Case 1: generated identical set but PERMUTED row order.
# If ED were a set-level distribution distance it would be invariant to row order.
# Because it pairs index i<->i, a permutation changes the result.
gen_same = orig.copy()
perm = rng.permutation(n_orig)
gen_perm = orig[perm].copy()

ed_identity = calculate_ed(orig, gen_same)          # exact pairing -> 0
ed_permuted = calculate_ed(orig, gen_perm)          # same *set*, shuffled order

# Case 2: more generated than original -> silent truncation.
n_gen = 100
gen_big = rng.standard_normal((n_gen, L, C))
ed_big = calculate_ed(orig, gen_big)   # uses ori_data.shape[0]=30, ignores gen[30:]
ed_big_first30 = calculate_ed(orig, gen_big[:n_orig])
truncation_equal = np.isclose(ed_big, ed_big_first30)

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
outp = os.path.join(os.path.dirname(__file__), "out", "ed_dtw_pairing.csv")
with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "value", "note"])
    w.writerow(["ed_orig_vs_same_order", f"{ed_identity:.6f}",
                "0 because sample i paired with identical sample i"])
    w.writerow(["ed_orig_vs_same_set_shuffled", f"{ed_permuted:.6f}",
                "same SET of curves, only row order changed -> nonzero, index pairing"])
    w.writerow(["ed_30orig_vs_100gen", f"{ed_big:.6f}",
                "loop runs over ori_data.shape[0]=30"])
    w.writerow(["ed_30orig_vs_first30gen", f"{ed_big_first30:.6f}",
                "identical to above -> gen[30:100] never used"])
    w.writerow(["truncation_confirmed", str(bool(truncation_equal)), "True => 70/100 generated samples ignored"])

print(open(outp).read())
