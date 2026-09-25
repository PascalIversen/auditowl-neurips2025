"""Checks that eval_spair.py's chunk(2) on the batch dim crashes for ensemble_size=1
(the value hardwired in the provided test_spair.sh). Supports finding 'chunk2-cfg-crash'."""
import torch

# Feature saved by Featurizer4Eval.forward has batch dim = ensemble_size.
# test_spair.sh sets --ensemble_size 1, so the saved feature tensor is [1, 3072, 48, 48].
for ens in (1, 8):
    x = torch.randn(ens, 3072, 4, 4)  # small H,W for speed; batch dim is what matters
    try:
        a, b = x.chunk(2)   # eval_spair.py:175 / :196
        print(f"ensemble_size={ens}: chunk(2) OK -> a{tuple(a.shape)} b{tuple(b.shape)}")
    except ValueError as e:
        print(f"ensemble_size={ens}: chunk(2) CRASH -> ValueError: {e}")
