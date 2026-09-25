# Code-Repository Audit — Paper 2279

**Title:** *Controlling the Flow: Stability and Convergence for Stochastic Gradient
Descent with Decaying Regularization* (Kassing, Weissmann, Döring).

## 1. Summary

This is primarily a **theory paper**. The headline contributions are mathematical:
proofs that regularized SGD (reg-SGD) with a vanishing Tikhonov regularization
converges (a.s. and in L²) to the minimum-norm solution of a convex problem, plus
optimal polynomial decay rates for the step-size (`q`) and regularization (`p`)
schedules (Theorems 2.3/2.4, Corollaries 2.6/2.7). These results are proved in the
appendix; no code is expected to (or does) "compute" them.

The repository (`code/openreview_supplementary.zip` → `Code_Submission/`) is a set of
**MATLAB scripts that produce the illustrative convergence figures** in Section 3.2
(Radon/X-ray tomography, Figs 3–5) and Appendix A.2 (toy example, Figs 6–11) and A.3
(ODE inverse problem, Figs 12–13). There are **no quantitative tables or reported
statistics**; every figure is a qualitative log–log convergence plot. The repo is the
author code (matches the appendix descriptions closely) and is self-contained standard
MATLAB (`lsqminnorm`, `sparse`, the built-in `mri` demo image); no requirements file is
needed for a MATLAB submission and the readme states MATLAB.

**What I did.** I extracted the supplementary zip read-only, read all 8 MATLAB sources
+ readme, and mapped each figure to its generating script. I cross-checked every
experiment constant stated in Appendix A against the hard-coded MATLAB values using a
deterministic grep script (`_audit_code/check_params.py`, output
`_audit_code/out/params.csv`). The substantive findings are all **code↔paper parameter
differences** in the illustrative experiments; none of them changes a qualitative
conclusion, and the theory itself is outside the code's scope. No leakage / split /
baseline / statistics machinery applies (synthetic data, no held-out evaluation, no
learned model, no reported metrics).

## 2. Traceability table

The paper reports **no numeric tables/statistics**; "artefacts" are the figures. The
question per Rule G is whether a script *computes* each figure's curves (not merely
plots them) and whether the configuration matches the paper text.

| Paper artefact (figure) | Repo location | Computes the curves? | Config matches paper text | Status |
|---|---|---|---|---|
| Fig. 1 (motivating toy plot) | (no dedicated script) | — | — | illustrative cartoon, no number; N/A |
| Fig. 2 (rate vs p,q, analytic) | (none — analytic formula) | — | — | analytic illustration, no simulation |
| Figs 3–4 (Radon images: base/sinogram/min-norm/recons) | `Radon/main_radon.m:172-216` | Yes (forward op + SGD/reg-SGD recon) | base image = MATLAB `mri` demo (paper unspecified) | Verified (qualitative) |
| Fig. 5 (Radon optimality gap & error) | `Radon/main_radon.m:219-272` | Yes | `N` default 10^3 vs paper "5·10^6" (readme: user may increase) | MISMATCH (N), see `radon-N-default` |
| Fig. 6 (toy L² gap & error) | `Toy_example/main_L2rates.m` | Yes | SGD step const 1 vs paper 0.1; `N`=10^4 vs 10^6 | MISMATCH, see `toy-step-const`, `iteration-count-defaults` |
| Fig. 7 (toy a.s. gap & error) | `Toy_example/main_asrates.m` | Yes | noise cov 1 vs paper 0.1²; SGD step 0.05 vs 0.1; `N`=10^4 vs 10^6 | MISMATCH, see `toy-asrates-noise`, `toy-step-const` |
| Figs 8–9 (toy, varying (p,q)) | `Toy_example/main_comparison.m` | Yes | (p,q) set matches paper; step law `q·0.2·k^{-q}` vs paper text `0.2q·k^{-1/2}`; `N`=10^5 vs 10^7 | MISMATCH (step law / N), see `toy-comparison-step` |
| Figs 10–11 (toy, varying α₁) | (no script in repo) | No | — | MISSING, see `toy-alpha1-script-missing` |
| Figs 12 (ODE gap & error) | `ODE_example/main_ODE.m:165-202` | Yes | K=63 vs 64; λ const 0.002 vs 0.001; `N`=10^4 vs 10^7 | MISMATCH, see `ode-K`, `ode-lambda-const` |
| Fig. 13 (ODE reconstruction) | `ODE_example/main_ODE.m:204-234` | Yes | KL coeff `sqrt(2*pi)` vs paper `sqrt(2)/pi` (m_ref loaded, not regenerated) | MISMATCH (regen only), see `ode-kl-coeff` |

All theorem/corollary statements (the actual contributions) are math proofs in the
appendix — no code computes them and none is expected to.

## 3. Findings

### missing

```yaml finding
id: toy-alpha1-script-missing
category: missing
topic: "result traceability"
title: "No script generates the varying-initial-step-size experiment (Figs 10–11)"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  In the final experiment, we examine the effect of the initial
  value α1 > 0 in the step-size schedule. For SGD, we set αk = α1k−1/2, while for reg-SGD we
  fix λk = k−0.111 and use the step-size schedule αk = α1k−0.667. We report both the pathwise
  optimality gap and the pathwise squared error to the minimum-norm solution for SGD (Figure 10)
  and reg-SGD (Figure 11) under varying initial step sizes α1 ∈ {0.01, 0.1, 1, 2}.
claim: "Appendix A.2 reports Figures 10 and 11 sweeping the initial step size α1 ∈ {0.01,0.1,1,2}; the three toy-example scripts (main_L2rates.m, main_asrates.m, main_comparison.m) sweep (p,q) and rates but none loops over α1 or reproduces Figs 10/11."
concern: "The α1-sweep figures cannot be reproduced from the provided code."
resolution: "Authors: please add the script that produces Figures 10–11 (the α1 sweep), or confirm it was a minor variant run off-repo."
cross_refs: []
paper_ref: "Appendix A.2, Figures 10 and 11"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

No findings. The scripts are self-contained MATLAB and run as written (constants are
literal; forward operators, gradients, and tracking logic are internally consistent).
The mislabeled `% regularization decay` / `% learning rate decay` comments in
`main_asrates.m:37-38` and `main_ODE.m:69-70` are swapped relative to the variables
they annotate, but the math (`lambdak = 1/k^p`, `step = .../k^q`) is correct — a comment
typo, not a defect, so not filed.

### difference

```yaml finding
id: ode-K
category: difference
topic: "experiment configuration"
title: "ODE example uses 63 observation points, paper states 64"
severity: low
confidence: high
status: finding
file: ODE_example/main_ODE.m
line_start: 7
line_end: 8
quote: |
  % number of observation points
  K = 2^6-1;
claim: "The ODE inverse-problem script sets the number of observation points to K = 2^6-1 = 63."
concern: "Appendix A.3 states K = 64 equidistant observation points (s_k = k/K, k=1,…,K); the code uses 63, a one-off mismatch in the discretization size."
resolution: "Authors: confirm whether K=63 (interior nodes) or K=64 was used to produce Figs 12–13, and reconcile with the appendix."
cross_refs: ["ode-lambda-const"]
check_script: _audit_code/check_params.py
paper_ref: "Appendix A.3 ('K = 64 equidistant observation points')"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ode-lambda-const
category: difference
topic: "experiment configuration"
title: "ODE regularization constant is 0.002 in code, 0.001 in paper"
severity: low
confidence: high
status: finding
file: ODE_example/main_ODE.m
line_start: 113
line_end: 114
quote: |
        % decaying regularization
        lambdak = 0.002/k^p;
claim: "The ODE script uses regularization schedule lambda_k = 0.002 * k^{-1/3}."
concern: "Appendix A.3 states lambda_k = 0.001 k^{-1/3}; the code's prefactor (0.002) is twice the reported value."
resolution: "Authors: confirm the regularization prefactor used for Figs 12–13 (0.001 vs 0.002)."
cross_refs: ["ode-K"]
check_script: _audit_code/check_params.py
paper_ref: "Appendix A.3 ('regularization λk = 0.001k−1/3')"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: toy-asrates-noise
category: difference
topic: "experiment configuration"
title: "Toy a.s.-rate script uses gradient-noise std 1, paper A.2 states 0.1"
severity: low
confidence: high
status: finding
file: Toy_example/main_asrates.m
line_start: 66
line_end: 68
quote: |
        % generate synthetic noise for gradient evaluation, same realization
        % for SGD and reg-SGD
        noisygrad = 1*randn(n,1);
claim: "main_asrates.m perturbs the gradient with N(0,1) noise (std 1, covariance 1·Id)."
concern: "Appendix A.2 states the toy example perturbs the gradient with covariance 0.1²·Id (std 0.1); the a.s.-rate script uses std 1 — 100× larger variance — and an SGD/reg-SGD step prefactor 0.05 rather than the stated 0.1."
resolution: "Authors: confirm the noise covariance and step-size prefactor used for Figure 7 (a.s. rates) vs the values quoted in Appendix A.2."
cross_refs: ["toy-step-const"]
check_script: _audit_code/check_params.py
paper_ref: "Appendix A.2 ('covariance 0.12 · Id', 'αk = 0.1k−1/2')"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: toy-step-const
category: difference
topic: "experiment configuration"
title: "Toy L²-rate SGD step prefactor is 1, paper A.2 states 0.1"
severity: low
confidence: high
status: finding
file: Toy_example/main_L2rates.m
line_start: 75
line_end: 76
quote: |
        % learning rate for SGD without reg
        step_sizek_literature = 1./k^(1/2);
claim: "main_L2rates.m sets the plain-SGD step size to 1·k^{-1/2} (and reg-SGD step to 1·k^{-q})."
concern: "Appendix A.2 states αk = 0.1k−1/2 for SGD (and αk = 0.1k−q for reg-SGD); the L²-rate script omits the 0.1 prefactor (uses 1), so the simulated step sizes differ from the reported ones."
resolution: "Authors: confirm the step-size prefactor used for Figure 6 (0.1 vs 1)."
cross_refs: ["toy-asrates-noise"]
check_script: _audit_code/check_params.py
paper_ref: "Appendix A.2 ('αk = 0.1k−1/2', 'αk = 0.1k−q')"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: toy-comparison-step
category: difference
topic: "experiment configuration"
title: "Comparison script step law q·0.2·k^{-q} differs from paper's 0.2q·k^{-1/2}"
severity: low
confidence: medium
status: finding
file: Toy_example/main_comparison.m
line_start: 111
line_end: 115
quote: |
        % learning rate for reg-SGD
        step_sizek = q*0.2/k^q;
        step_sizek2 = q2*0.2/k^q2;
        step_sizek3 = q3*0.2/k^q3;
        step_sizek4 = q4*0.2/k^q4;
claim: "The (p,q)-comparison script uses step size α_k = q·0.2·k^{-q} (decay exponent q, varying per configuration)."
concern: "Appendix A.2 (Figs 8–9) writes the step size as αk = 0.2q·k−1/2 (a fixed k^{-1/2} decay); the code instead decays with each configuration's own q, so the step-size schedule used differs from the formula printed in the paper (the (p,q) value set itself matches)."
resolution: "Authors: confirm the step-size exponent for the (p,q)-comparison experiment — fixed k^{-1/2} as written, or per-configuration k^{-q} as coded."
cross_refs: []
check_script: _audit_code/check_params.py
paper_ref: "Appendix A.2 ('αk = 0.2qk−1/2')"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ode-kl-coeff
category: difference
topic: "experiment configuration"
title: "ODE KL-basis coefficient is sqrt(2*pi) in code vs sqrt(2)/pi in paper"
severity: low
confidence: medium
status: finding
file: ODE_example/main_ODE.m
line_start: 23
line_end: 25
quote: |
  for j = 1:N_x
      V_basis(:,j) = sqrt(2*pi)*sin(j*pi*xx)';
  end
claim: "The KL basis used to (re)generate the ground-truth rhs x† scales each sine mode by sqrt(2*pi) ≈ 2.507."
concern: "Appendix A.3 defines x†(s) = Σ (√2/π) ξ_i sin(iπs) with coefficient √2/π ≈ 0.450; the code uses √(2π) ≈ 2.507, a ~5.6× different amplitude. Effect is limited because the actual run loads a saved m_ref.mat (regeneration at lines 37-38 is commented out), so it only matters if the reference is regenerated."
resolution: "Authors: confirm the basis normalization (√2/π vs √(2π)) and whether the saved m_ref.mat was produced with the paper's formula."
cross_refs: []
check_script: _audit_code/check_params.py
paper_ref: "Appendix A.3 ('x†(s) = Σ √2/π ξi sin(iπs)')"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: iteration-count-defaults
category: difference
topic: "result traceability"
title: "Default iteration counts far below the figure captions (10^3–10^5 vs 5·10^6–10^7)"
severity: low
confidence: high
status: finding
file: Radon/main_radon.m
line_start: 50
line_end: 51
quote: |
  % number of iterates
  final_iterate = 10^3;
claim: "Shipped defaults are final_iterate = 10^3 (Radon), 10^4 (toy L²/a.s., ODE), 10^5 (toy comparison)."
concern: "The figure captions report runs of length N = 5·10^6 (Fig 5), 10^6 (Figs 6–7), and 10^7 (Figs 9–12); the shipped defaults are 10^3–10^5, so running the code as-is does not reproduce the published curves. The readme explicitly notes 'the user may select an increased final iterate', so this is an acknowledged convenience default rather than a hidden discrepancy."
resolution: "No action strictly required (acknowledged); optionally ship the exact N used per figure as a comment or config so the published plots are reproducible without guessing."
cross_refs: ["radon-N-default"]
check_script: _audit_code/check_params.py
paper_ref: "Figure captions 5, 6, 7, 9, 10, 11, 12 (N values)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

No findings. This is a synthetic-data illustration of a proven theorem: there is no
train/test split, no held-out evaluation, no learned model, no baselines to tune, and
no reported statistics or p-values — the standard methodology checklists
(leakage, sample independence, pretraining contamination, temporal integrity, baseline
fairness, statistical integrity) are **N/A** here. The experiments compare SGD vs
reg-SGD on the *same* gradient-noise realization per iteration
(e.g. `main_radon.m:107-134`), which is an appropriate paired comparison for the
qualitative claim being illustrated.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 1          | low          | No script for the α1-sweep Figures 10–11.                  |
| bug         | 0          | -            | Scripts run as written; only swapped comment labels.       |
| difference  | 7          | low          | Several illustrative-experiment constants differ from App. A. |
| methodology | 0          | -            | Theory paper; synthetic illustration, no split/metric/stats. |

## 5. Closing lists

**Top take-aways** (≤6, by severity × confidence)
1. `[difference] ode-K` — ODE example uses K=63 observation points vs paper's 64 (high confidence).
2. `[difference] ode-lambda-const` — ODE regularization prefactor 0.002 vs paper 0.001 (high confidence).
3. `[difference] toy-asrates-noise` — toy a.s. script gradient-noise std 1 vs paper 0.1 (100× variance) (high confidence).
4. `[difference] toy-step-const` — toy L² SGD step prefactor 1 vs paper 0.1 (high confidence).
5. `[difference] iteration-count-defaults` — shipped N (10^3–10^5) far below caption N (5·10^6–10^7); readme-acknowledged (high confidence).
6. `[missing] toy-alpha1-script-missing` — no code reproduces the α1-sweep Figures 10–11 (medium confidence).

None of these is conclusion-changing: the paper's claims are mathematical theorems
(proved in the appendix) and the figures are qualitative convergence illustrations, so
the constant mismatches do not affect any reported number or the headline result.

**Items that genuinely look fine**
- The forward operators, gradients, and minimum-norm targets are implemented correctly
  and consistently with the objectives (Radon `radon_forward.m`; ODE finite-difference
  Laplacian `A_ellipticPDE.m` + `observation_matrix.m`; `lsqminnorm` for x∗).
- SGD vs reg-SGD use the *same* per-iteration noise realization and same
  initialization, a fair paired comparison (`main_radon.m:107-134`,
  `main_L2rates.m:78-84`).
- Radon noise covariance (0.5²) and toy-L² noise covariance (0.1²) match the paper.
- The (p,q) value set in `main_comparison.m` (lines 44-58) matches the paper's
  {(0.111,0.667),(0,0.667),(0.67,0.5),(0.111,0.29)}.
- Code is self-contained standard MATLAB with a clear readme mapping scripts→figures.

**Open questions for the authors**
- Were Figures 12–13 produced with K=63 or K=64, and λ-prefactor 0.001 or 0.002?
- Which gradient-noise covariance and step-size prefactor generated Figure 7 (a.s.
  rates) — the paper text (0.1², 0.1) or the code (1, 0.05)?
- Is there a script for the α1-sweep Figures 10–11, or were they a manual variant?
