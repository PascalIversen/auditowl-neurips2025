"""Checks that the code's group-assignment via `gain` reduces to relative_accuracy_gain>0 (i.e. ignores length), supporting the group-preference difference finding."""
import numpy as np
max_length_inc_ratio = 10
rng = np.random.default_rng(0)
mismatch = 0
for _ in range(200000):
    rel_acc = rng.uniform(-1, 1)          # long_acc - short_acc - 1/(2K)
    rel_len = rng.uniform(0.01, 5.0)      # relative length increment (>0 in practice: long>short)
    if rel_acc > 0:
        gain = rel_acc / rel_len
    else:
        gain = rel_acc * (rel_len / max_length_inc_ratio)
    # code: gain>0 -> long chosen ; gain<=0 -> short chosen
    code_choice = "long" if gain > 0 else "short"
    acc_choice = "long" if rel_acc > 0 else "short"
    if code_choice != acc_choice:
        mismatch += 1
print("samples:200000  group-choice mismatches vs sign(rel_acc):", mismatch)
print("=> group assignment depends ONLY on sign(relative_accuracy_gain); length term changes magnitude not sign")
