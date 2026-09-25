#!/usr/bin/env python3
"""Demonstrates that model/transformer.py masks attention with the constant
1e-8 (not -inf) before softmax, so a 'causal' mask still leaks future tokens.
Supports finding `causal-mask-leaks-future`. Read-only."""
import torch, torch.nn.functional as F, os, json

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# Reproduce the repo's masking on a tiny example.
# attn logits for 4 tokens; lower-triangular causal mask (token i may see <= i).
torch.manual_seed(0)
n = 4
attn = torch.randn(n, n) * 2.0           # unscaled q@k^T, as in transformer.py (no * scale)
mask = torch.triu(torch.ones(n, n, dtype=torch.long)).T.bool()  # same construction as ablation

# Repo behaviour: torch.where(mask, attn, 1e-8)
repo = F.softmax(torch.where(mask, attn, torch.tensor(1e-8)), dim=-1)
# Correct causal behaviour: masked -> -inf
correct = F.softmax(attn.masked_fill(~mask, float("-inf")), dim=-1)

# Weight that the FIRST token (row 0) assigns to FUTURE tokens (cols 1..n-1):
future_leak_repo = repo[0, 1:].sum().item()
future_leak_correct = correct[0, 1:].sum().item()

res = {
    "row0_future_weight_repo(1e-8 mask)": round(future_leak_repo, 4),
    "row0_future_weight_correct(-inf mask)": round(future_leak_correct, 4),
    "note": "With the repo's 1e-8 fill, token 0 still attends to future tokens "
            "(weight > 0); a correct causal mask gives exactly 0.",
}
with open(os.path.join(OUT, "attention_mask.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
