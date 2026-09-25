"""Tests BinnedTemperatureScaling._infer_perturbation_level wiring.

Supports finding `perturb-level-vs-single-reference`. ReCalX infers the
perturbation level of an input by counting the FRACTION OF FEATURES that differ
from a single stored reference vector (the first val sample, set via
set_unperturbed_input). During Shapley/LIME attribution Captum feeds MANY
different (and only partially perturbed) samples through forward(). For any
sample other than the reference, almost every feature already differs from the
reference, so the inferred 'perturbation level' is ~1.0 regardless of how many
features Captum actually masked -- the wrong temperature bin is selected.
This reproduces that for the Electricity setup.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "thomdeck__recalx"))
import numpy as np, torch
from recalx.calibration import BinnedTemperatureScaling
from recalx.model import MLP

rng = np.random.default_rng(0)
torch.manual_seed(0)
n, d = 200, 7  # electricity has 7 features
X = rng.normal(size=(n, d)).astype(np.float32)
means = X.mean(0)

model = MLP(d_in=d, d_out=2)
temps = np.linspace(1.0, 2.0, 10).astype(np.float32)
wrap = BinnedTemperatureScaling(model, temps, num_bins=10)
wrap.set_unperturbed_input(torch.tensor(X[0]))

# Case A: feed UNPERTURBED sample 5 (no features masked). True level should be 0.
x5 = torch.tensor(X[5:6])
lvl_5 = wrap._infer_perturbation_level(x5[0])

# Case B: feed the reference sample itself, unperturbed. True level 0.
x0 = torch.tensor(X[0:1])
lvl_0 = wrap._infer_perturbation_level(x0[0])

# Case C: reference sample with 3/7 features set to baseline (true level ~0.43)
xp = X[0].copy(); idx = [1, 3, 5]; xp[idx] = means[idx]
lvl_ref_masked = wrap._infer_perturbation_level(torch.tensor(xp))

out = os.path.join(os.path.dirname(__file__), "out", "perturb_level.txt")
with open(out, "w") as f:
    f.write(f"unperturbed NON-reference sample (true level 0.0): inferred={lvl_5}\n")
    f.write(f"unperturbed reference sample      (true level 0.0): inferred={lvl_0}\n")
    f.write(f"reference w/ 3of7 masked          (true level ~0.4): inferred={lvl_ref_masked}\n")
    f.write(f"BUG_nonref_unperturbed_misread = {lvl_5 != 0.0}\n")
print(open(out).read())
