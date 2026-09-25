# Code-repository audit — Paper 3242, "Robust Estimation Under Heterogeneous Corruption Rates"

## 1. Summary

This is a **theory paper** (Chaudhuri, Li, Courtade; NeurIPS 2025). Its headline
contributions are minimax rates (Theorems 1–5) and an algorithm (Algorithm 1) for
robust mean estimation / linear regression under per-sample-heterogeneous corruption
(the `λ`-contamination model). The paper states repeatedly that "the main contribution
of this work is theoretical" and that the experiments are "preliminary synthetic
evaluations" (paper.pdf §1, lines 257–263) shown only in Appendix A, Figures 1–2.

The released code (`code/supplement/`) is exactly what the NeurIPS checklist promises —
"Attached Jupyter notebooks allow reviewers to verify the plots without writing any
extra code" (paper.pdf checklist item 5):

- `experiments.ipynb` — produces Figure 2(a) (bounded mean estimation, MSE vs `q`) and
  Figure 2(b) (univariate Gaussian, squared-error quantile bands). Estimators:
  `opt_linear` (Algorithm 1 reweighting, "Optimal Linear Method"), `thresh_linear`
  ("Threshold Method"), and baselines `sample_mean` / `sample_median`.
- `depth-map.ipynb` — produces Figure 1 (weighted-Tukey-depth colormaps for three
  weighting schemes A/B/C).
- `appendix.pdf` — the paper appendix (duplicate of the appendix in `paper.pdf`).

**What I did.** I read both notebooks and the paper (main text + full appendix through
Appendix F). Because the theorems are mathematical and cannot be "reproduced" by code,
the only reproducible artefacts are the two figures, neither of which reports any scalar
number, statistical test, table, or ablation. I therefore concentrated on faithfulness
(code ↔ paper) and on whether the code runs and supports the (mild) experimental claims.
I wrote three read-only checks under `_audit_code/`:

- `check_cdf.py` (`out/cdf_check.csv`) — empirically identifies the corruption-rate
  distribution actually sampled by the notebooks.
- `check_repro.py` (`out/weights_sum.csv`) — verifies the `opt_linear` weights form a
  valid simplex vector and reproduces Figure 2(a) at reduced scale.

Findings are minor. There are **no** methodology defects, **no** bugs, and **no**
high-severity issues: a theory paper's conclusions cannot be undermined by these
illustrative experiments, and the experiments do run and do support their stated (mild)
claims. The one substantive item is that the paper mis-states the closed form of the
distribution from which the corruption rates `λ` are drawn (it is the reciprocal of the
stated exponent); the code is internally sound and matches the paper's *qualitative*
description, so this is a low-severity `difference`.

## 2. Traceability table

The paper reports **no numeric values, tables, ablations, or statistical tests** — only
two qualitative figures. Rows below therefore verify *presence + qualitative match* of
the figure-producing code.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 1 — weighted Tukey depth maps, schemes (A) uniform, (B) threshold, (C) Alg. 1 | `depth-map.ipynb` `plot_combined_depth_maps` (cell 8) | 3 depth colormaps + max-depth "estimated mean" | ✓ qualitative | Verified (runs; matches caption setup N((0,0),I) clean / N((2,2),I/5) outliers, size ∝ 1−λ) |
| Fig. 2(a) — Bounded MSE vs `q`, 3 curves (Optimal Linear / Threshold / Sample Mean) | `experiments.ipynb` cell 3 + fns `opt_linear`,`thresh_linear`,`sample_mean` | MSE(opt) < MSE(thresh) < MSE(mean) at every `q` | ✓ qualitative (reproduced, `out/`) | Verified |
| Fig. 2(b) — Univariate Gaussian squared-error, 4/5-quantile line + 15/20–17/20 band | `experiments.ipynb` cells 6–8 (`opt_linear_G`,`thresh_linear_G`,`sample_median`) | quantiles at 0.75/0.80/0.85 of sq-error | ✓ matches paper's stated quantiles | Verified (code present; quantiles as described) |
| Appendix A — "cdf given by F(t)=1−(1−t)^q" for sampling λ | `experiments.ipynb` line 131 (`lambd = 1 - np.exp(-np.random.exponential(x,N))`) | empirical F(t)=1−(1−t)^(1/q) | ✗ | MISMATCH → `difference` (see `lambda-cdf-exponent-mismatch`) |
| Fig. 1 caption — panel (B) "t from (21)" (bound t²+d/N(t), d=2) | `depth-map.ipynb` cell 8 threshold search uses `E=t**2` vs `1/N` (d=1) | t solving t²=1/N(t) | ✗ (uses d=1 in a d=2 figure) | MISMATCH → `difference` (see `depth-map-threshold-dimension`) |
| Theorems 1–5, minimax rates, Algorithm 1 correctness, lower-bound constructions | proofs in Appendix B–F (paper.pdf) | — | — | N/A — mathematical claims, not a code artefact |
| Dependency spec / README / exact run commands | (none in repo) | — | — | MISSING (see `no-dependency-or-run-instructions`) |
| Statistical tests / p-values / CIs | (none reported) | — | — | N/A — paper reports none |

## 3. Findings

### missing

```yaml finding
id: no-dependency-or-run-instructions
category: missing
topic: "code completeness / environment"
title: "No dependency spec, README, or run instructions in the repo"
severity: low
confidence: high
status: finding
file: code/supplement/experiments.ipynb
line_start: 1
line_end: 2
quote: |
  import numpy as np
  import matplotlib.pyplot as plt
claim: "The repo is three files (experiments.ipynb, depth-map.ipynb, appendix.pdf) with no requirements.txt / environment.yml, no README, and no stated commands or library versions; depth-map.ipynb additionally imports scipy.spatial and sklearn.neighbors."
concern: "The 'complete submission' baseline (dependency specification + README with the exact commands to reproduce each figure) is absent, so the environment must be reconstructed by inspection; impact is small because the dependencies are standard and the notebooks are self-contained."
resolution: "Add a requirements.txt (pin numpy/scipy/scikit-learn/matplotlib versions) and a short README stating which notebook produces which figure and how to run it."
cross_refs: []
paper_ref: "Checklist item 5 (paper.pdf): 'Attached Jupyter notebooks allow reviewers to verify the plots without writing any extra code.'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

None. The notebooks run top-to-bottom on a standard scientific-Python stack
(numpy 1.26 / scipy 1.13 / scikit-learn 1.7 / matplotlib 3.9). I explicitly checked the
two most suspicious constructs and both are correct: (i) the `opt_linear` closed-form
weights sum to 1 for every tested `λ` and `c` (`_audit_code/out/weights_sum.csv`), so
there is no simplex/normalization bug in Algorithm 1's implementation; (ii) the
`k`-selection loop's `k=i-1` followed by use of `lambd[:k+1]` cancels to the intended
active set `lambd[:i]`, i.e. no off-by-one in the resulting estimator.

### difference

```yaml finding
id: lambda-cdf-exponent-mismatch
category: difference
topic: "experimental setup / data generation"
title: "Paper states λ~CDF 1−(1−t)^q; code actually samples 1−(1−t)^(1/q)"
severity: low
confidence: high
status: finding
file: code/supplement/experiments.ipynb
line_start: 131
line_end: 131
quote: |
      lambd = 1 - np.exp(-np.random.exponential(x,N))
claim: "λ is drawn as 1−exp(−E) with E~Exponential(scale=x) and x equal to the figure's x-axis value q; this yields CDF F(t)=1−(1−t)^(1/q), whereas Appendix A states 'cdf given by F(t)=1−(1−t)^q'. Empirically (out/cdf_check.csv) the sampled CDF at t=0.5 for q=5 is 0.13, matching 1−(1−t)^(1/q)=0.129 and not the paper's 1−(1−t)^q=0.969 (max |emp−paper| = 0.93). The same line recurs at line 337 for the Gaussian experiment."
concern: "The written formula is the reciprocal-exponent of what is sampled and even contradicts the paper's own adjacent sentence 'As q increases we can expect a higher corruption rate' (true only for the 1/q form, which the code implements), so the figure axis is mislabeled relative to the stated distribution — a description error, not a computational one; the estimator receives the true λ vector regardless, so no result changes."
resolution: "Confirm the intended sampling law: the code and the qualitative statement agree on F(t)=1−(1−t)^(1/q); update the Appendix A formula accordingly (or change the sampling if 1−(1−t)^q was intended)."
cross_refs: ["depth-map-threshold-dimension"]
check_script: _audit_code/check_cdf.py
paper_ref: "Appendix A, paper.pdf: 'we sample the corruption rates λ i.i.d. from the distribution with cdf given by F(t) = 1 − (1 − t)^q. As q increases we can expect a higher corruption rate.'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: depth-map-threshold-dimension
category: difference
topic: "figure generation"
title: "Fig. 1 panel (B) threshold uses 1/N (d=1) though the figure is 2-D and (21) uses d/N(t)"
severity: low
confidence: medium
status: finding
file: code/supplement/depth-map.ipynb
line_start: 321
line_end: 323
quote: |
          N = np.sum(lambd <= t)
          E = t**2
          if E > (1/N):
claim: "The threshold that defines panel (B)'s weights is chosen by balancing t² against 1/N(t), i.e. the univariate (d=1) optimum, but the depth map is 2-D (d=2) and the caption says these weights use 'the value of t from (21)', whose balance term is d/N(t)=2/N(t)."
concern: "For a 2-D illustration the code selects a slightly different threshold than eq. (21) with d=2 would give; both are valid thresholds and this only shifts which low-corruption samples panel (B) averages, so it does not change the qualitative point of the figure."
resolution: "Either use d/N(t) with d=2 to match eq. (21), or clarify in the caption that panel (B) uses the univariate balance t²=1/N(t)."
cross_refs: ["lambda-cdf-exponent-mismatch"]
paper_ref: "Figure 1 caption and Eq. (21), paper.pdf"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

None. This is a finding-free pass for the methodology category (Rule L: reported, not
skipped). Reasoning, topic by topic:

- **Data splitting / sample independence / target leakage / pretraining contamination /
  temporal integrity / inference-time shift** — N/A. The experiments are Monte-Carlo
  simulations from a fully specified synthetic generative model (λ-contamination); there
  is no dataset, no train/test split, no learned representation, and no time dimension,
  so none of these failure modes can arise.
- **Baselines** — Appropriate and run under the same simulation/metric as the proposed
  methods. The paper uses the sample mean (bounded) and the sample median (univariate
  Gaussian, = 1-D Tukey median) as baselines and justifies each as *the* homogeneous
  minimax-robust estimator; both are computed on the identical `Data`/`λ` draws. A naive
  baseline is present, and the proposed methods beat it at every `q`
  (`_audit_code/out/…`, reproduced Fig. 2(a)).
- **Metric fit** — Consistent with the theory: the bounded case reports MSE (matching
  L(λ,D), an expectation) and the Gaussian case reports the 4/5-quantile of squared
  error with a 15/20–17/20 band (matching the paper's PAC rate `L_PAC(·,·,1/5)`, i.e. the
  0.8 quantile for δ=1/5). Error bands are defined (std for bounded, empirical quantiles
  for Gaussian).
- **Hyperparameter tuning / selective reporting** — No test-set tuning is possible (no
  test set). The one free constant `c=3` used in `opt_linear(...,c=3)` matches the
  constant that appears in the paper's own bound (Eq. (7): `‖w‖²+3(wᵀλ)²`); it is fixed,
  not tuned to the outcome.
- **Statistical integrity** — N/A. The paper reports no p-values, hypothesis tests,
  multiple comparisons, or confidence-interval claims to check.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                             |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 1          | low          | No requirements file / README / run commands (deps standard).|
| bug         | 0          | –            | Notebooks run; weights verified to sum to 1, no off-by-one.  |
| difference  | 2          | low          | Paper mis-states λ-CDF exponent; Fig.1(B) threshold uses d=1.|
| methodology | 0          | –            | Synthetic-simulation experiments; baselines & metrics sound. |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)

1. **[difference] `lambda-cdf-exponent-mismatch`** — Appendix A states the corruption
   rates are drawn from `F(t)=1−(1−t)^q`, but the code samples `F(t)=1−(1−t)^(1/q)`
   (verified empirically, `_audit_code/out/cdf_check.csv`). The code matches the paper's
   qualitative statement and the estimators receive the true λ, so no result changes —
   purely a formula/axis-labeling error. *Low severity, high confidence.*
2. **[missing] `no-dependency-or-run-instructions`** — No requirements file, README, or
   stated commands; dependencies (numpy/scipy/scikit-learn/matplotlib) are standard and
   the notebooks are self-contained, so impact is minor. *Low severity, high confidence.*
3. **[difference] `depth-map-threshold-dimension`** — Figure 1 panel (B) selects its
   threshold with the d=1 balance `t²=1/N(t)` even though the illustration is 2-D and the
   caption cites Eq. (21) (which carries `d/N(t)`). Immaterial to the figure's message.
   *Low severity, medium confidence.*

(No high- or medium-severity findings exist: the paper is theoretical, and the
illustrative experiments both run and support their stated claims.)

### Items that genuinely look fine (actively checked)

- `opt_linear` closed-form weights sum to 1 for every tested (λ, c) — no simplex/
  normalization bug (`_audit_code/out/weights_sum.csv`).
- The `k`-selection loop is not off-by-one: `k=i-1` then `lambd[:k+1]` = `lambd[:i]`, the
  intended active set.
- Figure 2(a) reproduces: Optimal Linear < Threshold < Sample Mean MSE at every `q`,
  supporting the paper's claim that the proposed methods beat the baseline and that
  reweighting improves on thresholding in the bounded case.
- Gaussian quantile reporting (0.75/0.80/0.85) matches the paper's stated
  15/20, 4/5, 17/20 quantiles and the PAC rate at δ=1/5.
- Baselines (sample mean / sample median) are the correct homogeneous minimax-robust
  estimators and are evaluated on the same draws/metric as the proposed methods.
- Randomness is seeded (`np.random.seed(1)` in both experiment cells; `seed` argument in
  `depth-map.ipynb`'s `generate_data`), so the figures are reproducible.
- The weighted Tukey depth in `depth-map.ipynb` matches Eq. (17) (min over directions of
  the smaller weighted half-count); the corruption setup matches the Figure 1 caption.
- The Fig. 2 corrupted values are fixed (bounded: 1; Gaussian: N(100,1)) rather than
  worst-case — this is a benign adversary, but it is exactly what Appendix A states, so
  code and paper agree (a disclosed limitation of a preliminary experiment, not a
  discrepancy).

### Open questions for the authors

- Confirm the intended corruption-rate law: the notebooks sample `F(t)=1−(1−t)^(1/q)`
  (which matches "higher q ⇒ higher corruption"), while Appendix A writes
  `F(t)=1−(1−t)^q`. Which is authoritative, and should the Appendix A formula be
  corrected? (`lambda-cdf-exponent-mismatch`)
- For Figure 1 panel (B), was the univariate threshold `t²=1/N(t)` intended, or the
  d=2 form `t²=2/N(t)` from Eq. (21)? (`depth-map-threshold-dimension`)
