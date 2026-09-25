"""Compares the per-head label offset implemented in loss_function (model.py:219-224, skip_token=3) against the paper's leap positions t+k(i-1)+1 with k=2 (supports findings skip-token-mismatch and num-heads-mismatch)."""
import os
out = []
for skip in [2, 3]:
    n_head = 4
    code_offsets = [skip * (i + 1) for i in range(n_head)]
    out.append(f"CODE skip_token={skip}, n_head={n_head}: additional-head label offsets ahead = {code_offsets}")
for k in [2, 3]:
    n = 4
    paper = [k * j + 1 for j in range(n)]  # head j=0..n-1 -> t+1, t+k+1, t+2k+1, t+3k+1
    out.append(f"PAPER k={k}, n={n}: leap positions incl. NTP head = {paper} (additional heads at {paper[1:]})")
out.append("NTP head (head0) covered separately by ori_loss at offset 1 in stage-2 loss.")
txt = "\n".join(out)
print(txt)
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "out", "leap_positions.txt"), "w") as fh:
    fh.write(txt + "\n")
