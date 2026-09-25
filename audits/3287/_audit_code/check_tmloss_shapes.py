"""Checks TMLoss shape consistency vs the (4x4) ground-truth bigram from the tokenizer.

Supports finding `tmloss-padcol-slice` (is the [:, :-1, :-1] slice a benign PAD-drop or a bug?).
Reproduces the einsum + slice path from carmania/loss.py against a (B,4,4) target as built
by train.py / tokenizer.encode_with_bigram, and reports whether KL is finite and shapes align.
Read-only; does not import the repo. Saves output to out/tmloss_shapes.txt.
"""
import os
import torch
import torch.nn.functional as F

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "out", "tmloss_shapes.txt")

B, L, V = 2, 9, 5  # vocab=5 (A,T,C,G,PAD); logits over full vocab
torch.manual_seed(0)
logits = torch.randn(B, L - 1, V)  # train.py feeds logits = model(input_ids[:, :-1])

# --- replicate carmania/loss.py forward ---
eps = 1e-6
probs = F.softmax(logits, dim=2)
p1, p2 = probs[:, :-1, :], probs[:, 1:, :]
pred_bigram = torch.einsum('bti,btj->bij', p1, p2)   # (B, V, V) = (B,5,5)
shape_before = tuple(pred_bigram.shape)
pred_bigram = pred_bigram[:, :-1, :-1]                # -> (B,4,4): drops last (PAD) row/col
shape_after = tuple(pred_bigram.shape)
row_sums = pred_bigram.sum(dim=-1, keepdim=True).clamp_min(1)
pred_bigram = pred_bigram / row_sums
pred_bigram = pred_bigram + eps

# ground-truth bigram as built by train.py: (N,4,4) row-normalized over axis 2
true = torch.rand(B, 4, 4)
true = true / true.sum(dim=2, keepdim=True)
true = true + eps

kl = torch.sum(true * (torch.log(true) - torch.log(pred_bigram)), dim=(-2, -1)).mean()

lines = [
    f"vocab_size V = {V}  (4 nucleotides + PAD)",
    f"pred_bigram shape before slice = {shape_before}  (B,V,V)",
    f"pred_bigram shape after  slice = {shape_after}   (B,4,4) -> matches GT (B,4,4)",
    f"GT bigram shape = {tuple(true.shape)}",
    f"KL finite = {bool(torch.isfinite(kl))}, value = {float(kl):.4f}",
    "CONCLUSION: shapes align; [:, :-1, :-1] drops the PAD row/col so pred matches the",
    "4x4 nucleotide-only ground truth. Pred row-norm uses clamp_min(1) (NOT clamp to a small",
    "epsilon): if a predicted row sums to < 1 it is left unnormalized, so pred rows need not sum to 1.",
]
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
