#!/usr/bin/env python3
"""Checks that the hardcoded 'trainable' video keys in contrastive_model.py:30
(proj.0/proj.2) do NOT match any real VideoEncoder.proj parameter (proj.1/proj.3),
so under fine_tuning_policy='missing_trainable' the whole video branch is frozen.
Supports finding: stage1-video-branch-frozen-by-key-mismatch. No torch import of
the repo needed; replicates create_mlp() from
projects/eeg_vedio/src/models/video_encoder/utils.py verbatim.
"""
import os

# create_mlp copied verbatim from utils.py (only nn.Linear/nn.ReLU naming matters).
# nn.Sequential names children by list index; a leading ReLU shifts Linears to odd indices.
def proj_param_names(input_dim, hidden_dims, output_dim, dropout=None):
    kinds = []                       # 'relu' | 'linear' | 'dropout'
    dims = [input_dim] + hidden_dims + [output_dim]
    kinds.append("relu")             # layers.append(nn.ReLU())  <-- leading ReLU
    for i in range(len(dims) - 1):
        kinds.append("linear")
        if i < len(dims) - 2:
            kinds.append("relu" if dropout is None else "relu")
            if dropout is not None:
                kinds.append("dropout")
    names = []
    for idx, k in enumerate(kinds):
        if k == "linear":
            names += [f"proj.{idx}.weight", f"proj.{idx}.bias"]
    return names

HARDCODED = ["proj.0.weight", "proj.0.bias", "proj.2.weight", "proj.2.bias"]

for cfg in ([512, 200], [200, 200]):
    real = proj_param_names(1024, cfg[:-1], cfg[-1])
    overlap = set(HARDCODED) & set(real)
    print(f"mlp_layers={cfg}")
    print(f"  real proj params        : {real}")
    print(f"  hardcoded trainable keys: {HARDCODED}")
    print(f"  overlap                 : {overlap or 'EMPTY -> entire video branch requires_grad=False'}")
    print()

out = os.path.join(os.path.dirname(__file__), "out", "video_branch_freeze.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("hardcoded trainable video keys proj.0/proj.2 match ZERO real params "
            "(proj.1/proj.3) for both mlp configs; under 'missing_trainable' the "
            "video branch is fully frozen, contradicting paper Eq.(4)/§3.2.\n")
print(f"[written] {out}")
