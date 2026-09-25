"""Check that MPTCN row-normalization weights match the paper's RN description.

Paper (Sec 4): RN divides "the input at time step t-i by min(i+1, P)".
Code (convolution_disjoint.py:99-103, 149-154) builds:
    w = 1 / arange(1, P+1) -> flip -> [1/P,...,1/2,1/1]
then in tmp() pads on the LEFT with 'replicate' (so all positions older than P get 1/P)
and multiplies the input (channel-last-axis = time, ordered oldest..newest).

This script reconstructs the per-time-step divisor the code applies for a sequence of
length T and checks it equals min(i+1, P) where i is the BACKWARD distance from t
(i=0 is the most recent step t). Supports finding: rn-matches-paper (an item that looks fine).
"""
import torch
import torch.nn.functional as F

P = 4
T = 16

# Reproduce code construction (convolution_disjoint.py:100-103)
w = 1 / torch.arange(1, P + 1, dtype=torch.float32)   # [1/1,1/2,1/3,1/4]
w = torch.flip(w, (0,)).view(1, 1, 1, -1)             # [1/4,1/3,1/2,1/1]

# tmp(): x has time on last axis ordered oldest(0)..newest(T-1). diff = T - P pad on left.
diff = T - w.size(-1)
w_full = F.pad(w, (diff, 0, 0, 0, 0, 0), mode='replicate').squeeze()  # length T

# w_full[k] multiplies the time-axis position k, where k=0 is OLDEST (t-(T-1)) and
# k=T-1 is NEWEST (t). The divisor applied = 1 / w_full.
divisor = (1.0 / w_full).round().int().tolist()

# Paper: divisor at backward distance i (i=0 newest) should be min(i+1, P).
# Code position k=T-1 is i=0 (newest). So i = (T-1) - k.
expected = [min(((T - 1) - k) + 1, P) for k in range(T)]

print("code position k (0=oldest .. T-1=newest):")
print("  applied divisor by k :", divisor)
print("  expected min(i+1,P)  :", expected)
match = divisor == expected
print("MATCH (code RN == paper 'divide t-i by min(i+1,P)'):", match)

with open(__file__.replace("check_row_norm.py", "out/row_norm.txt"), "w") as fh:
    fh.write(f"P={P} T={T}\napplied={divisor}\nexpected={expected}\nmatch={match}\n")
