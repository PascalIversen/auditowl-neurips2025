# Audit — Efficient Adaptive Experimentation with Noncompliance (paper 1274)

## 1. Summary

The repository (`code/CausalML__Adaptive-IV/`, single commit `f8acbac "NeurIPS Camera
Ready"`) is the authors' replication code for AMRIV, an adaptive multiply-robust IV
estimator for the ATE under noncompliance. It is a small, self-contained simulation
codebase: `models.py` (the `AMRIVExperiment` and `A2IPWExperiment` online-collection
classes), `utils.py` (learner factories), `data.py` (the synthetic one-sided-noncompliance
DGP + analytic oracles), `data_TA.py` (the TripAdvisor semi-synthetic DGP + oracles), and
two driver scripts `run_synthetic.py` (Figures 2 & 3) and `run_semi_synthetic.py`
(Figure 4). The paper has no numeric tables; all empirical artefacts are figure panels.

What I did:
- Read the paper's method (Eqs. 5–9, Algorithm 1), the experimental setup (§7, Appendix H.1/H.2),
  and matched every reported figure panel to the code that computes its values.
- Verified the influence-function formula in `models.py:_calculate_phi` against paper Eq. (9):
  exact match.
- Ran a reduced AMRIV synthetic experiment (`_audit_code/smoke_run.py`): code runs, `phi` is
  finite, and the estimate (-1.673) tracks the true ATE (-1.664).
- Checked the misspecification constant (`_audit_code/check_misspec_constant.py`).
- Audited seeding/reproducibility, cross-fitting usage (`CV` flag), dependency specification,
  and the coverage/MSE computations.

Overall the implemented estimator is faithful to Eq. (9) and recovers the true ATE. The
findings below are about (a) a core methodological component the paper foregrounds
(sequential cross-fitting) that is disabled in the code that makes the figures, (b) an
unseeded RNG that breaks per-trajectory reproducibility of the semi-synthetic experiment,
(c) absence of any dependency specification, and (d) a minor misspecification-constant
difference. None of these overturn the headline claim that adaptive assignment + AMRIV
improves efficiency/coverage over the baselines, but (a) and (c) affect faithful reproduction.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2 — optimal policy π*(x) vs δA(x) (motivating illustration, v0=4,v1=1, δA=σ(−2x)) | `run_synthetic.py:479-498` | curve recomputed from `_gamma`, `pi_star` formula | ✓ (qualitative; v0=4,v1=1 as in §"Motivating Illustration") | Verified (curve-generating code present) |
| Fig. 3a — Efficiency: normalized MSE vs T (AMRIV/AMRIV-NA/DM/DM-NA vs Oracle) | `run_synthetic.py:168-296` (run) → `mse_curve` `:38-49`, plot `:347-373` | MSE(τ̂)/MSE(Oracle) per horizon | not numerically re-run at full scale; estimator validated | Verified (compute path present) |
| Fig. 3b — Consistency: MSE ± SE vs T | `run_synthetic.py:375-405` | per-horizon MSE and SD | — | Verified (compute path present) |
| Fig. 3c — Coverage: empirical coverage of 95% CI vs T | `run_synthetic.py:407-477` (`coverage_within_exp`) | fixed-time Wald 1.96·SE coverage | ✓ (tied to Theorem 3, pointwise CLT) | Verified (compute path present) |
| Fig. 4a/b/c — same three panels, TripAdvisor semi-synthetic DGP | `run_semi_synthetic.py` (mirrors run_synthetic) + `data_TA.py` | as above | — | Verified compute path; per-seed reproducibility broken (see `semi-synthetic-unseeded-rng`) |
| §5.1 / Alg.1 step 6 — nuisances estimated "using cross-fitting" (Waudby-Smith sequential folds) | `models.py:209-243` (CV branch) | NOT executed: `CV=False` default, never set True in run scripts | ✗ (figures use non-cross-fitted branch `:245-256`) | DIFFERENCE (see `cross-fitting-disabled`) |
| §H misspecified μ̂Y(1,X) = constant E[μY(1,X)] | `run_synthetic.py:150-160` | constant = mean **compliance** `OracleMuA(z=1).mean()`=0.83, not mean outcome (0.72) | ✗ (still a constant; arbitrary value) | DIFFERENCE (see `misspec-constant-mismatch`) |
| Dependency / environment spec (Appendix H: scikit-learn, joblib, numpy, scipy, pandas, matplotlib) | (none) | — | — | MISSING (see `no-dependency-spec`) |
| Precomputed result logs for `--plot_only` | (none; `results/` not shipped) | — | — | N/A (experiments are cheap; full run regenerates) |

No quantitative tables, no statistical tests (Wilcoxon/t-test/etc.) are reported in the paper,
so STATISTICAL INTEGRITY checks are N/A beyond the coverage panels handled above.

## 3. Findings

### missing

```yaml finding
id: no-dependency-spec
category: missing
topic: "expected code completeness / dependencies"
title: "No dependency specification (no requirements.txt / environment.yml / setup)"
severity: medium
confidence: high
status: finding
file: code/CausalML__Adaptive-IV/README.md
line_start: 10
line_end: 16
quote: |
  ## Replication Code for Paper

  Use the following commands to replicate the figures from the "Efficient Adaptive Experimentation with Noncompliance" paper:

  * For Figure 2 & 3: ```python run_synthetic.py```

  * For Figure 4: ```python run_semi_synthetic.py```
claim: "The repo ships no requirements.txt, environment.yml, setup.py, or pyproject.toml; the README gives run commands but pins no package versions, while the code imports numpy, scikit-learn, scipy, pandas, joblib, matplotlib (and optionally torch)."
concern: "The environment cannot be rebuilt to the versions used for the paper; scikit-learn RandomForest defaults and RNG behaviour change across versions, so figures are not guaranteed reproducible."
resolution: "Add a pinned requirements.txt / environment.yml listing numpy, scikit-learn, scipy, pandas, joblib, matplotlib (and torch if used) with the versions used to produce the figures."
cross_refs: []
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: semi-synthetic-unseeded-rng
category: bug
topic: "reproducibility / seeding"
title: "Semi-synthetic DGP uses global np.random for confounder/treatment/noise; per-seed RNG unused"
severity: medium
confidence: high
status: finding
file: code/CausalML__Adaptive-IV/data_TA.py
line_start: 117
line_end: 129
quote: |
      rng = np.random.default_rng(seed)
      draw_x   = x_factory(seed)              # reuse same RNG for X

      def g():
          # 1. covariate
          X = draw_x()
          nu = np.random.uniform(U_low, U_high)
          
          A1 = np.random.binomial(1, coef_Z*scipy.special.expit(0.4*X[0] + nu))
          A0 = np.random.binomial(1, .006)

          eps0 = np.random.lognormal(0, sigma_0)
          eps1 = np.random.lognormal(0, sigma_1)
claim: "`make_synthetic_iv_dgp(seed)` creates `rng = np.random.default_rng(seed)` (line 117) but `g()` draws the latent confounder nu, both potential treatments A1/A0, and both noise terms eps0/eps1 from the *global* `np.random`, never from `rng`; only the covariates X go through the seeded RNG."
concern: "The per-trajectory `seed` controls only covariates, so the latent confounder, treatments, and outcome noise are not reproducible from the seed; combined with joblib parallelism this makes Figure 4 non-deterministic, contradicting the intent of the seed argument (the synthetic `data.py` correctly uses `rng.binomial`/`rng.uniform`)."
resolution: "Replace the `np.random.*` calls in `data_TA.py:g()` with `rng.*` so the trajectory is fully determined by `seed`, matching `data.py`."
cross_refs: []
check_script: _audit_code/smoke_run.py
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: cross-fitting-disabled
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Sequential cross-fitting (foregrounded in §5.1 / Alg.1) is disabled in the code that makes the figures"
severity: medium
confidence: high
status: finding
file: code/CausalML__Adaptive-IV/models.py
line_start: 209
line_end: 245
quote: |
        # ---------------- cross-validation branch ----------------
        if self.CV and t >= 1:                     
            fold0 = np.zeros(t + 1, dtype=bool)
            fold0[::2] = True                  
            fold1 = ~fold0

            # train on opposite folds
            self.muY0_fold0.fit(X_t[fold0 & (Z_t == 0)], Y_t[fold0 & (Z_t == 0)])
claim: "AMRIVExperiment defaults to `CV=False` (models.py:16) and neither run_synthetic.py nor run_semi_synthetic.py ever set `CV=True`, so the sequential cross-fitting branch (models.py:209-243) never runs; all reported figures use the non-cross-fitted plug-in branch (models.py:245-256) that fits delta/s on the full history."
concern: "The paper presents sequential cross-fitting (Waudby-Smith et al., §5.1 'Unbiased two-stage estimation via cross-fitting' and Algorithm 1 step 6 'using cross-fitting') as a core ingredient for unbiased two-stage variance estimation, but the figures are produced without it; the non-CV branch is itself a valid plug-in procedure, so this is a faithfulness gap rather than an invalid method."
resolution: "Clarify whether the reported figures used cross-fitting; if not, state that the non-cross-fitted plug-in variant was used, or enable `CV=True` in the run scripts. Note the CV branch also carries a `# TODO: fix this` comment (models.py:237)."
cross_refs: []
tags: [reforms:3, lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: misspec-constant-mismatch
category: difference
topic: "misspecification ablation"
title: "Synthetic AMRIV-MS uses mean compliance as the constant μ̂Y(1,X), not mean outcome"
severity: low
confidence: high
status: finding
file: code/CausalML__Adaptive-IV/run_synthetic.py
line_start: 149
line_end: 154
quote: |
  data_gen = make_synthetic_iv_dgp(f=my_f, d=d, seed=0)
  X_test = np.array([data_gen()[0] for i in range(10000)])
  ms_const1 = OracleMuA(z=1).predict(X_test).mean()
  miss_factories = {
      "muY0": make_rf_factory(regression=True, n_estimators=100, max_depth=5, min_samples_leaf=5),
      "muY1": lambda **_: MSY(ms_const1),
claim: "The misspecified μ̂Y(1,X) constant `ms_const1` is set to the mean of `OracleMuA(z=1)` (the mean compliance score, ≈0.83 per _audit_code/check_misspec_constant.py), whereas Appendix H says μ̂Y(1,X) is replaced by the constant E[μY(1,X)] (≈0.72)."
concern: "The constant fed to the deliberately-misspecified outcome model is the wrong quantity relative to the paper's description; since the variant's purpose is only to be a constant (mis)model, the specific value is unlikely to change conclusions, but the procedure differs from the text (the semi-synthetic version instead uses OracleMuY(z=1)+1.5)."
resolution: "Set the synthetic misspecified constant to E[μY(1,X)] = OracleMuY(z=1).predict(X_test).mean() as described, or update Appendix H to describe the constant actually used."
cross_refs: []
check_script: _audit_code/check_misspec_constant.py
tags: [reforms:6]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

No methodology findings. The implemented influence function matches paper Eq. (9) exactly,
nuisances are fit on past data only (no leakage), the assignment policy / truncation match
Eqs. (5)/(7), the coverage panels compute a valid pointwise CLT interval consistent with
Theorem 3, and the estimator recovers the true ATE in a smoke run.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 1          | medium       | No dependency / environment specification. |
| bug         | 1          | medium       | Semi-synthetic DGP draws confounder/treatment/noise from unseeded global np.random. |
| difference  | 2          | medium       | Cross-fitting (a foregrounded method component) disabled in figure-producing code; misspec constant uses wrong quantity. |
| methodology | 0          | -            | Implemented estimator matches Eq. (9); no leakage or invalid procedure found. |

### Top take-aways
- **[difference] `cross-fitting-disabled`** (medium/high): the sequential cross-fitting the
  paper foregrounds in §5.1 and Algorithm 1 is never executed (`CV=False`); figures use the
  non-cross-fitted plug-in branch.
- **[bug] `semi-synthetic-unseeded-rng`** (medium/high): Figure 4's DGP is not reproducible
  from its `seed` because nu/treatments/noise use the global `np.random`, not the seeded `rng`.
- **[missing] `no-dependency-spec`** (medium/high): no requirements/environment file, so the
  exact scikit-learn/numpy versions (which affect RF behaviour and RNG) are unspecified.
- **[difference] `misspec-constant-mismatch`** (low/high): synthetic AMRIV-MS constant is the
  mean compliance score, not the mean outcome described in Appendix H.

### Items that genuinely look fine
- The (R)EIF `phi` formula in `models.py:_calculate_phi` (lines 139-153) matches paper Eq. (9)
  term-for-term (weight, 1/δA, residual, +δ).
- Nuisances used in `phi_t` come from the previous refit batch (past data Ht−1), not the
  current point — no train/test leakage in the online estimator.
- The plug-in assignment policy and truncation (`_assign`, lines 99-114) match Eqs. (5)/(7)
  with `kt = 2/0.999^t`.
- Synthetic DGP (`data.py`) matches Appendix H.1: `X~Unif(0,2)^d`, `δA=σ(2X[1])`, `f`, `g`,
  v0=4, v1=0.25, u=−2 all reproduced; per-seed RNG used correctly there.
- The smoke run recovers the true ATE (estimate -1.673 vs true -1.664), confirming the
  estimator is wired correctly.

### Open questions for the authors
- Did the figures in the submission use cross-fitting (`CV=True`) or the default non-cross-fitted
  branch? (Drives `cross-fitting-disabled`.)
- Is the analytic `OracleSigma` in `data.py`/`data_TA.py` the exact residual variance
  Var(Y−Aδ|Z=z,X) (it omits the explicit compliance-variance/covariance terms shown in the
  paper's variance decomposition)? Filed as a question, not a finding, since the Oracle is only
  a reference curve and AMRIV uses estimated `s0/s1`.
