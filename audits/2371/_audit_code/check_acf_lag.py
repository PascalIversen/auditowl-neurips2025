"""Checks the lagged-product slicing in acf_torch (feature_based_measures.py:98-109).
For lag i>0 the code computes x[:, i:] * x[:, :-i]; we confirm the two slices are
length-aligned (T-i each) so the elementwise product is well-defined, i.e. this is
NOT an off-by-one crash. Supports the 'looks fine' note on ACD. Read-only.
"""
import os
import numpy as np
import torch
import csv

# verbatim copy of acf_torch from
# utils/evaluation/feature_based_measures.py:98-109
def acf_torch(x, max_lag, dim=(0, 1)):
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)

B, T, D = 8, 40, 1
x = torch.randn(B, T, D)
ok = True
err = ""
try:
    out = acf_torch(x, max_lag=min(64, T), dim=(0, 1))
    shape = tuple(out.shape)
except Exception as e:
    ok = False
    shape = None
    err = str(e)

outdir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(outdir, exist_ok=True)
outp = os.path.join(outdir, "acf_lag.csv")
with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["check", "result", "note"])
    w.writerow(["acf_runs_without_error", str(ok), err])
    w.writerow(["output_shape", str(shape), "stack of max_lag autocorr values"])
print(open(outp).read())
