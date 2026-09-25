"""Checks that KL_divergence() in train_utils_attention.py does not compute the KL
in Eq.(6) of the paper: it double-softmaxes an already-normalised gaze distribution
and passes plain probabilities (not log-probs) to F.kl_div. Supports finding
'kl-regularizer-double-softmax'. Read-only; reimplements the repo function verbatim."""
import os
import torch
import torch.nn.functional as F

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out = open(os.path.join(os.path.dirname(__file__), "out", "kl_divergence.txt"), "w")

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.write(s + "\n")

# ---- verbatim core of repo's KL_divergence (train_utils_attention.py:253-272) ----
def repo_kl(target, source):
    target_log_probs = F.softmax(target, dim=-1)   # NB: name says log, value is prob
    source_log_probs = F.softmax(source, dim=-1)
    hold = torch.nn.functional.kl_div(source_log_probs, target_log_probs,
                                      reduction='sum', log_target=True)
    if hold < 0:
        hold = 0
    return hold

# A "perfectly aligned" case: model attention == gaze target distribution.
# target_dist comes from calculate_gaze_proportions_batch -> already sums to 1.
torch.manual_seed(0)
gaze = torch.rand(256); gaze = gaze / gaze.sum()      # normalised distribution
attn = gaze.clone()                                    # identical to gaze

# Correct KL(attn || gaze) for identical distributions must be 0.
correct_kl = (attn * (attn.clamp_min(1e-12).log() - gaze.clamp_min(1e-12).log())).sum()
log("Correct KL(attn||gaze) when attn==gaze (should be 0):", float(correct_kl))

repo_val = repo_kl(gaze.unsqueeze(0), attn.unsqueeze(0))
log("Repo KL_divergence value when attn==gaze (should be 0 if correct):", float(repo_val))

# Show the double-softmax distortion: softmax of an already-normalised prob vector
sm = F.softmax(gaze, dim=-1)
log("max|softmax(gaze) - gaze| (0 would mean no distortion):", float((sm - gaze).abs().max()))

# Show that with log_target=True the 'target' is treated as log-probs although it is a prob:
# i.e. effective target = exp(softmax(gaze)), which does not sum to 1.
eff_target = torch.exp(F.softmax(gaze, dim=-1))
log("sum of effective target exp(softmax(gaze)) (should be 1 for a valid distribution):",
    float(eff_target.sum()))

out.close()
