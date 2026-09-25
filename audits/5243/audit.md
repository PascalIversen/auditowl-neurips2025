# Audit — Causal Explanation-Guided Learning for Organ Allocation (CLEXNET), NeurIPS 2025 (#5243)

## 1. Summary

The repository `AlessandroMarchese/ClexNet` (audited at commit `9271b7c`, the only
commit on `main`) contains a single Jupyter notebook (`CLEXNET_tutorial.ipynb`), a
short README, and three tutorial PNGs — nothing else. The notebook is a self-described
"tutorial on synthetic data": it defines a synthetic data generator (`make_dataset`),
an sklearn-style `ClexNetClassifier`, a small model zoo (Logistic Regression, Random
Forest, PatientNet, "ClexNet - expl only", OrganITE, ClexNet), and an evaluation cell
that prints a results table on one synthetic draw.

The paper, by contrast, reports two evaluation tracks: **synthetic** *and*
**semi-synthetic UNOS-PTR liver-offer data** (~1.1M real offers, 24k organs, 46k
patients), and three reason-generation mechanisms (uniform, IPW, boundary-intersection),
a confounding sweep (Fig. 4), an F-feasibility sweep (Figs. 5–6), and an
Assumption-1 support analysis (Table 4). The paper's availability statement (lines
477–478) claims "All code, synthetic generators and an implementation of CLEXNET are
made public."

What I did: read the paper PDF and the full notebook; ran two read-only deterministic
checks under `_audit_code/` — `check_repo_inventory.py` (file inventory, dependency-spec
presence, keyword scan for UNOS/boundary/sweep code) and `check_traceability.py`
(maps each paper table/figure to a producing artefact in the repo). Outputs in
`_audit_code/out/`. I stayed read-only on `code/`.

Headline: the public code covers only the *synthetic* half of one experiment (Table 1 /
Table 3 synthetic columns) plus the uniform/IPW reason ablation; the entire
semi-synthetic UNOS evaluation, the boundary-intersection sampler (the *winning*
ablation), the confounding sweep, the F-sweep, and the Assumption-1 analysis have **no
producing code in the repo**. There is also no dependency specification. Two
code↔paper differences (cluster count `k`, explanation-loss form) and one paper
omission (a 37-epoch explanation warm-up) are noted but are lower-severity.

## 2. Traceability table

Legend: "Repo location" is the artefact that *computes* the underlying numbers (not
merely plots them). The notebook draws fresh random data each run and reports no fixed
numbers, so even covered cells are "value not pinned" rather than numerically matched.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1, **Synthetic** columns (BCE/AUC/AUPRC/Brier × 6 models) | `CLEXNET_tutorial.ipynb` cell 19 (eval) + cell 5 (gen) | re-run produces different numbers (random draw, k=5 not 3) | ✗ (not pinned/reproducible) | PARTIAL (code present, numbers not reproducible) |
| Table 1, **Semi-synthetic UNOS-PTR** columns (all 6 models × 4 metrics) | (none) | — | — | MISSING (no UNOS pipeline) |
| Table 2, Uniform Random reason mechanism | `CLEXNET_tutorial.ipynb` cell 5 (`ipw_reasons=False` → `w=1/N`) | code path present | — | PARTIAL (no driver that sweeps the 3 mechanisms / reports Table 2) |
| Table 2, IPW reason mechanism | `CLEXNET_tutorial.ipynb` cell 5 (`ipw_reasons=True`, KDE) | code path present | — | PARTIAL |
| Table 2, **Boundary Intersection** (best row) | (none) | — | — | MISSING (sampler not implemented) |
| Table 3, **Synthetic** observational-test columns | `CLEXNET_tutorial.ipynb` cell 19 (commented-out Obs-Val/Obs-Test rows) | Obs-Test rows exist but the per-mechanism numbers not pinned | ✗ | PARTIAL |
| Table 3, **Semi-synthetic UNOS-PTR** columns | (none) | — | — | MISSING (no UNOS pipeline) |
| Table 4, Assumption-1 support (% vs Euclidean range) | (none) | — | — | MISSING (no matching/support code) |
| Fig. 4, confounding sweep over ψ (Exp 5.2) | (none) | — | — | MISSING (no ψ-sweep driver) |
| Figs. 5–6, F-feasibility sweep over σ (Exp A.1) | (none; only an `F_scale` *param* exists, no sweep) | — | — | MISSING (no σ-sweep driver) |
| Fig. 7 / Table 6, λ–ρ and M sweeps | (none; single fixed configs only) | — | — | MISSING (no sweep driver) |
| Hyperparameters Table 7 (k=3, λ=0.15, ρ=0.15, M=100, 1000 ep) | `CLEXNET_tutorial.ipynb` cells 17/19 | k default = 5 (mismatch); λ/ρ/M/epochs match | ✗ (k) | MISMATCH on k |
| Dependency specification (to rebuild env) | (none) | — | — | MISSING |

Full inventory and keyword scan: `_audit_code/out/inventory.txt`;
artefact→code map: `_audit_code/out/traceability.csv`.

## 3. Findings

## missing

```yaml finding
id: unos-semisynthetic-pipeline-missing
category: missing
topic: "result traceability / semi-synthetic evaluation"
title: "No code for the semi-synthetic UNOS-PTR evaluation (half of Tables 1 & 3, all of Table 4)"
severity: high
confidence: high
status: finding
file: code/AlessandroMarchese__ClexNet/README.md
line_start: 1
line_end: 5
quote: |
  # CLEXNET – Synthetic-Data Tutorial Notebook  🤖 🧪

  This repository contains an interactive Jupyter notebook that walks you through

  * generating a **synthetic organ–offer dataset**—complete with
claim: "The repo contains only a synthetic-data tutorial notebook; there is no data loader, one-hot encoding, feature extraction, DF construction (a CLEXNET trained on observational data used as fY), or evaluation code for the UNOS-PTR liver-offer data, despite the paper reporting a full semi-synthetic track (the right half of Table 1, the right half of Table 3, and all of Table 4 / Appendix A.3)."
concern: "The semi-synthetic UNOS results are headline evidence that CLEXNET 'outperforms existing acceptance models' on real-derived data, yet no code produces any of those numbers, so the central real-data claim cannot be reproduced or verified."
resolution: "Authors: please release the UNOS-PTR preprocessing pipeline (feature selection per Table 5, one-hot encoding to the stated 76 dims, the DF-construction step that trains a CLEXNET as fY, and the evaluation harness), or state explicitly that the semi-synthetic experiments are not reproducible from the public repo."
cross_refs: ["§5.1", "§B.2", "no-dependency-spec"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Table 1 & 3 (UNOS-PTR columns); Table 4; Appendix B.2; availability statement lines 477-478"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: boundary-intersection-sampler-missing
category: missing
topic: "ablations / reason generation"
title: "Boundary-intersection reason sampler (the winning Table 2 row) is not implemented"
severity: high
confidence: high
status: finding
file: code/AlessandroMarchese__ClexNet/CLEXNET_tutorial.ipynb
line_start: 391
line_end: 410
quote: |
  if ipw_reasons:
      # ─────────── step 2 ───────────  propensity of (X,O) in OBS
      from sklearn.neighbors import KernelDensity
  else:
      w = np.full(len(Z_big_pos), 1/len(Z_big_pos))
claim: "make_dataset implements only two reason-generation mechanisms — uniform sampling (else branch, w=1/N) and IPW (KernelDensity branch). The third mechanism the paper compares, 'boundary intersection' (Table 2, lowest-error row), has no implementation anywhere in the repo (grep for 'boundary'/'intersection' returns 0 hits)."
concern: "Table 2 concludes that boundary intersection is the best reason-generation mechanism and Section 5.3's qualitative claims rest on it, but the code that would produce that row is absent, so the ablation's headline result is unverifiable."
resolution: "Authors: please add the boundary-intersection sampler used for Table 2, row 3, or clarify how the boundary-intersection counterfactuals were generated."
cross_refs: ["§5.3"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Table 2, row 'Boundary Intersection'; Section 5.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: confounding-and-F-sweep-drivers-missing
category: missing
topic: "result traceability / sweeps"
title: "No driver code for the ψ-confounding sweep (Fig. 4), F/σ sweep (Figs. 5-6), or λ-ρ-M sweeps (Table 6 / Fig. 7)"
severity: high
confidence: high
status: finding
file: code/AlessandroMarchese__ClexNet/CLEXNET_tutorial.ipynb
line_start: 1640
line_end: 1641
quote: |
  rct_df, obs_tr, obs_val, obs_te, meta = make_dataset(
          n_obs=1_000, n_rct= 10_000, d_x=5, d_o=5, p_keep = 0.1, rho = 0.9, ipw_reasons = False, target_acceptance= 0.5, seed=42)
claim: "The notebook generates exactly one dataset (rho=0.9 fixed) and trains the model zoo once; there is no loop or script that sweeps the confounding level ψ (Fig. 4 / Exp 5.2), the feasibility-region scaling σ (Figs. 5-6 / Exp A.1), or the loss weights λ/ρ and augmentation size M (Table 6 / Fig. 7). The make_dataset signature exposes `rho` and `collider_gamma`, and ClexNetClassifier exposes `F_scale`, but no code varies them to produce the reported figures/tables."
concern: "Four reported figures/tables (Fig. 4, Figs. 5-6, Table 6, Fig. 7) characterise robustness and hyperparameter sensitivity, and none of them can be regenerated because no sweep driver exists in the repo."
resolution: "Authors: please add the sweep scripts that produced Fig. 4, Figs. 5-6, Table 6, and Fig. 7, including the exact ψ, σ, λ, ρ, and M grids and the seeds/bootstrap settings."
cross_refs: ["§5.2", "§A.1", "§C"]
check_script: _audit_code/check_traceability.py
paper_ref: "Figure 4; Figures 5-6; Table 6; Figure 7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec
category: missing
topic: "expected code completeness / environment"
title: "No dependency specification (requirements.txt / environment.yml / etc.)"
severity: medium
confidence: high
status: finding
file: code/AlessandroMarchese__ClexNet/CLEXNET_tutorial.ipynb
line_start: 39
line_end: 45
quote: |
  import numpy as np
  import pandas as pd
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import matplotlib.pyplot as plt
  from sklearn.metrics import roc_auc_score
claim: "The notebook imports numpy, pandas, torch, scikit-learn, scipy and matplotlib, but the repo ships no requirements.txt, environment.yml, setup.py, pyproject.toml, or Pipfile (verified by file-existence check); no versions are pinned anywhere."
concern: "Without pinned dependencies the environment cannot be reconstructed reliably; sklearn API drift in particular matters here (the eval cell calls `log_loss(..., eps=1e-15)` and `brier_score_loss`, whose signatures changed across sklearn versions)."
resolution: "Authors: please add a pinned dependency file (requirements.txt or environment.yml) specifying at least python, numpy, pandas, torch, scikit-learn, scipy, and matplotlib versions."
cross_refs: ["unos-semisynthetic-pipeline-missing"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "NeurIPS checklist Q5 (open access to code), lines 1537-1540"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: reported-numbers-not-reproducible
category: missing
topic: "result traceability / synthetic numbers"
title: "Synthetic Table 1/Table 3 numbers are not pinned or reproducible from the notebook"
severity: medium
confidence: medium
status: finding
file: code/AlessandroMarchese__ClexNet/CLEXNET_tutorial.ipynb
line_start: 2224
line_end: 2231
quote: |
  _set_seeds(50)

  # -------------------------------------------------------------------
  # 2)  train / test split on *observational* data
  # -------------------------------------------------------------------
  X_cols  = [c for c in obs_tr.columns if c.startswith(("X", "O"))]
  d_cols  = [c for c in obs_tr.columns if c.startswith(("dX", "dO"))]
claim: "The evaluation seeds the model zoo (`_set_seeds(50)`), but the dataset itself is generated in a separate earlier cell and the BilinearOutcome ground-truth coefficients are seeded from a fixed generator while the cohort draw uses seed=42; the random non-linear selection net `g` (cell 5, `_mlp(in_dim,16,1,l2=True)`) is initialised with no seed set at that point, so the synthetic outcome surface and therefore the reported metrics are not deterministically reproducible run-to-run, and no fixed numbers matching Table 1/3 are stored."
concern: "None of the synthetic numbers in Tables 1, 3, or 6 can be matched against a deterministic re-run; the selection network `g` is created before `_set_seeds` is invoked inside make_dataset for the very first call, leaving part of the generative pipeline unseeded."
resolution: "Authors: please pin all RNG sources (including the selection net `g` initialisation) and provide the exact seed(s) and a script that regenerates the reported synthetic table values."
cross_refs: ["reported-cluster-count-mismatch"]
check_script: _audit_code/check_traceability.py
paper_ref: "Table 1 / Table 3 / Table 6 (synthetic columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone technical bugs (crashes, wrong-axis, dead imports) were confirmed in the
parts of the notebook that run. The notebook imports are intentionally duplicated but
harmless. The unseeded selection net is routed to reproducibility under `missing`
(`reported-numbers-not-reproducible`) rather than filed as a separate bug.

## difference

```yaml finding
id: reported-cluster-count-mismatch
category: difference
topic: "hyperparameters / clustering"
title: "Code uses k=5 organ clusters by default; paper Table 7 specifies k=3"
severity: medium
confidence: high
status: finding
file: code/AlessandroMarchese__ClexNet/CLEXNET_tutorial.ipynb
line_start: 1740
line_end: 1740
quote: |
                   k_cluster     =5,
claim: "ClexNetClassifier defaults to k_cluster=5, and the model-zoo cell instantiates every ClexNet/OrganITE/PatientNet without overriding k_cluster, so all neural models cluster organs into 5 groups (KMeans n_clusters=self.k_cluster). The paper's hyperparameter table (Table 7) reports 'Organ Cluster Amount k = 3'."
concern: "The number of adversarial organ clusters directly affects the balancing term LCE; the public code does not use the value the paper reports, so a faithful re-run with the released code does not match the documented configuration."
resolution: "Authors: confirm whether the reported experiments used k=3 or k=5, and align the released default with Table 7."
cross_refs: ["reported-numbers-not-reproducible"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Table 7, 'Organ Cluster Amount k'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: explanation-loss-form-differs
category: difference
topic: "loss function / method faithfulness"
title: "Explanation loss is -log(prob-weighted score), not BCE(avg counterfactual prob, pex) as in Eq. 8"
severity: medium
confidence: medium
status: finding
file: code/AlessandroMarchese__ClexNet/CLEXNET_tutorial.ipynb
line_start: 1884
line_end: 1896
quote: |
                          # normalised weights   w_ij ∝ p_ij
                          w = p_cf / (p_cf.sum(dim=1, keepdim=True) + 1e-12)         # (B,M)

                          # weighted score:   s_i = Σ_j w_ij · p_ij   (∈[0,1])
                          p_w = (w * p_cf).sum(dim=1)                                # (B,)

                          # -------- one-sided loss: only rows where p_max < τ -------
                          active = p_max < τ
                          if active.any():
                              # log-barrier hinge on the weighted score
                              loss_vals = -torch.log(p_w[active].clamp_min(1e-12))   # (K,)
                              loss_cfx  = self.ρ_expl * loss_vals.mean()
claim: "The implemented explanation loss aggregates counterfactual acceptance probabilities with a self-weighted score p_w = Σ_j (p_ij/Σp) · p_ij and penalises -log(p_w). The paper's Eq. 8 / Algorithm 1 specify the *uniform average* avg_m p(m) and a BCE term BCE(p_avg, pex) under the guard p_max < pex."
concern: "The released loss is a different (valid, but distinct) functional form from the one the paper documents — a probability-weighted softmax-like average inside a -log barrier rather than the documented uniform-average BCE-to-target — so the paper's loss equation does not match the code's loss."
resolution: "Authors: clarify which explanation-loss form produced the reported results, and reconcile Eq. 8 / Algorithm 1 with the implemented -log(p_w) objective."
cross_refs: ["explanation-warmup-undocumented"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Equation 8; Algorithm 1 (LEXPL ← BCE(pavg, pex))"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: explanation-warmup-undocumented
category: difference
topic: "training schedule (paper omission)"
title: "Explanation loss is disabled for the first 37 epochs (expl_ep), undocumented in the paper"
severity: low
confidence: high
status: finding
file: code/AlessandroMarchese__ClexNet/CLEXNET_tutorial.ipynb
line_start: 1866
line_end: 1866
quote: |
                  if ep > self.expl_ep and self.ρ_expl > 0:
claim: "The explanation-guided augmentation loss is applied only after epoch `expl_ep` (default 37); for the first 37 epochs the model trains on BCE + adversarial loss only. Algorithm 1 in the paper describes the explanation loss as part of every negative-instance training step with no warm-up period."
concern: "An undocumented 37-epoch warm-up before the paper's signature explanation loss kicks in is a result-affecting training detail absent from the paper's described procedure."
resolution: "Authors: document the explanation-loss warm-up schedule (value of expl_ep) and confirm it was used for the reported runs."
cross_refs: ["explanation-loss-form-differs"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Algorithm 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No additional methodology finding is filed. The synthetic generator's train/val/test
split is stratified and applied *after* data construction (cell 5, `train_test_split`
with `stratify`), the feasibility box is computed from the training set only (cell 17,
`X.min(0)`/`X.max(0)`), the RCT cohort is drawn independently with the X–O link broken
(cell 5), and ground-truth `p_true` baselines are reported alongside models — these are
methodologically reasonable choices on the *synthetic* track that is present. The
serious methodological questions (does the semi-synthetic UNOS evaluation leak,
since DF is itself generated by a trained CLEXNET used as fY; whether the boundary
sampler is sound) cannot be assessed because the relevant code is absent — see
`unos-semisynthetic-pipeline-missing` and `boundary-intersection-sampler-missing`. Per
the No-Extrapolation Rule, those route to `missing`, not `methodology`.

One observation worth flagging to authors as a question rather than a finding: in the
semi-synthetic setup (paper §B.2) the unbiased test set DF is constructed by training a
CLEXNET on the observational data and using *that* model as the ground-truth fY. If the
evaluated CLEXNET shares architecture/inductive bias with the DF-generating CLEXNET,
the semi-synthetic comparison could favour CLEXNET by construction. This is a paper-prose
concern with no code to verify, so it is an open question below, not a finding.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 5          | high         | Entire UNOS semi-synthetic track, boundary sampler, all sweeps, deps absent |
| bug         | 0          | -            | No confirmed standalone runtime/wiring bug                   |
| difference  | 3          | medium       | k=5 vs paper k=3; explanation-loss form; 37-epoch warm-up    |
| methodology | 0          | -            | Synthetic track is sound; real-data validity unverifiable (absent code) |

### Top take-aways (≤6, ranked by severity × confidence)

1. **[missing] No semi-synthetic UNOS-PTR code** (`unos-semisynthetic-pipeline-missing`,
   high/high): the entire real-data half of Tables 1 & 3 and all of Table 4 have no
   producing code, so the paper's central "works on real-derived data" claim is
   unreproducible.
2. **[missing] Boundary-intersection sampler absent** (`boundary-intersection-sampler-missing`,
   high/high): the winning reason-generation mechanism in Table 2 is not implemented.
3. **[missing] No sweep drivers** (`confounding-and-F-sweep-drivers-missing`, high/high):
   Fig. 4, Figs. 5-6, Table 6, and Fig. 7 cannot be regenerated; the notebook runs one
   fixed configuration only.
4. **[missing] No dependency specification** (`no-dependency-spec`, medium/high):
   environment cannot be rebuilt; sklearn-API-sensitive calls are unpinned.
5. **[difference] k=5 vs documented k=3** (`reported-cluster-count-mismatch`,
   medium/high): released code does not use the paper's reported cluster count.
6. **[missing] Synthetic numbers not pinned/reproducible** (`reported-numbers-not-reproducible`,
   medium/medium): the selection net `g` is unseeded; no stored values match Tables 1/3/6.

### Items that genuinely look fine

- The synthetic generator splits *after* data construction with `stratify=y`
  (no train→test split leakage on the synthetic track).
- The counterfactual feasibility box is fit on the training set only
  (`X.min(0)`/`X.max(0)` in `fit`), not on full data.
- The RCT/unbiased cohort is drawn independently and the X–O confounding link is
  explicitly broken for it (cell 5).
- Ground-truth `p_true` rows are reported alongside models, giving a proper upper-bound
  reference on the synthetic track.
- Reported λ=0.15, ρ=0.15, M=100, lr=1e-3, max-epochs=1000, patience=30 match the
  notebook's ClexNet config (only k differs).

### Open questions for the authors

- Semi-synthetic DF is generated by a CLEXNET trained on the observational data and used
  as the ground-truth fY (§B.2). Does the architectural/inductive-bias overlap between
  the DF-generator and the evaluated CLEXNET advantage CLEXNET in the semi-synthetic
  comparison? (high-severity-if-true, low-confidence — no code to check.)
- Which explanation-loss form produced the reported numbers: Eq. 8's BCE(avg, pex) or
  the implemented -log(prob-weighted score)?
- Was k=3 or k=5 used for the published tables?
