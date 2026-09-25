# Audit: "Maximizing the Value of Predictions in Control: Accuracy Is Not Enough" (NeurIPS 2025, paper 1800)

## Summary

This is primarily a **theory paper** on the "prediction power" of forecasts in
(LQR / general) optimal control. Its empirical content is a set of small,
self-contained **illustrative simulations** that accompany Examples 3.3, 3.4 and
A.2 and produce the paper's six figures. The cited repo
(`yihenglin97/Prediction-Power`, single commit `3832054`, no tag) contains seven
short NumPy/scikit-learn scripts (893 LOC total), a `run_prediction_power_exp.sh`
driver, a pinned `env.yml`, and the six output figure PDFs under `Figures/`. There
are no datasets, trained models, or real-world benchmarks — all data is synthetic
Gaussian noise generated in-script.

I read the paper (Section 3, Appendices A.5–A.6, Figure captions) and every script,
mapped each of the six figures to the script that computes it, and ran two
deterministic checks under `_audit_code/`:
`check_example33_theta_and_mse.py` (confirms the code's θ matches the paper's θ and
that θᵀθ has unit diagonal — i.e. the "same per-entry MSE" construction is correct —
and measures the train-vs-test MSE gap). Outputs in `_audit_code/out/`.

Overall the repo is complete, runs end-to-end, and is methodologically sound for
its purpose (illustrating a theoretical phenomenon, not making a generalization
claim). The only findings are two **low-severity code↔paper differences** that do
not affect any conclusion: (1) Figure 4's MSE is computed on the training set
although the paper says it is on a held-out test set, and (2) the M-GAPS decaying
learning rate used in code differs from the formula stated in Appendix A.6.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 1 (Ex. 3.4): M-GAPS / π¹ avg-cost improvement vs P(1)/T | `Example_Policy_Optimization_1.py:50-137` → `Figures/average_cost_improvement_current_prediction_0.5.pdf` | curves (not single number) | ✓ (qualitative) | Verified |
| Fig. 2 (Ex. 3.4): M-GAPS / π² avg-cost improvement vs P(2)/T | `Example_Policy_Optimization_2.py:52-143` → `Figures/average_cost_improvement_next_prediction_0.5.pdf` | curves | ✓ (qualitative) | Verified |
| Fig. 3 (Ex. 4.4): expected Q functions at x=0, t=T−1 | `prediction_power_intuition.py:23-49` → `Figures/prediction_improvement_intuition.pdf` | EQ0=u²+1, (u±1)² | ✓ | Verified (matches closed forms in paper) |
| Fig. 4 (Ex. 3.3): MSE–ρ curve, predictors I vs θ | `Example_Mismatch_1.py:71-79,98-138` → `Figures/1_step_mse_comparison.pdf` | MSE on **train** set | ≈ (on-train vs paper's "test"; diff ~1e-3) | MISMATCH (low) — see `mse-fig4-computed-on-train` |
| Fig. 5 (Ex. 3.3): control cost–ρ curve, empirical + theoretical | `Example_Mismatch_1.py:81-96,140-149` → `Figures/1_step_total_cost_comparison.pdf` | empirical cost over N_test trajectories + theory line | ✓ | Verified |
| Fig. 6 (Ex. A.2): MSE–time curve, predictors 1 vs 2 | `Example_Mismatch_2.py:39-125` → `Figures/multi_step_mse_predictor_A_B.pdf` | MSE on test set per t | ✓ (qualitative; unseeded) | Verified |
| θ construction: θᵀθ unit diagonal ⇒ same per-entry MSE as I | `Example_Mismatch_1.py:41-43` (`check_example33_theta_and_mse.py`) | diag(θᵀθ)=[1,1]; θ matches paper [[1,.99],[0,.141]] | ✓ | Verified |
| M-GAPS LR η_t = (1+t/1000)^{-0.5} (App. A.6) | `Example_Policy_Optimization_1.py:68`, `..._2.py:70` | `1e-3·(1+t/100)^{-0.5}` | ✗ | MISMATCH (low) — see `mgaps-lr-formula-mismatch` |

All six numbered figures trace to a script that computes (not merely plots) their
values. The paper reports no tables, no headline scalar metrics, and no statistical
tests, so there is nothing further to trace.

## missing

No missing-artefact findings. The repo contains a dependency spec (`env.yml`,
fully pinned), all figure-producing scripts, a driver shell script, a README with
install + run instructions, and the six output figures. Synthetic data is generated
in-script (no external dataset to ship). No pretrained models are needed (everything
trains trivially via `sklearn.LinearRegression` / a short gradient loop). Nothing in
the data/code-availability statement is unresolved.

## bug

No technical bugs found. Scripts are internally consistent; the dead transformed
global `noise = noise @ weights.T` at `Example_Mismatch_1.py:49` is shadowed by the
locally regenerated `noise` inside `test_one_step_optimal_control_corr` and does not
affect any output (the function correctly controls the plain N(0,I) disturbance that
the predictor is built from). N/A for shape/axis/off-by-one classes after review.

## difference

```yaml finding
id: mse-fig4-computed-on-train
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Figure 4 MSE-rho curve computed on training data, paper says test set"
severity: low
confidence: high
status: finding
file: Example_Mismatch_1.py
line_start: 71
line_end: 79
quote: |
      model = LinearRegression()
      model.fit(pred_train.reshape((-1, n)), noise_train.reshape((-1, n)))
      reg_pred = model.predict(pred_train.reshape((-1, n)).reshape((-1, n)))
      metric_record = []

      for i in range(n):
          mse = mean_squared_error(noise_train.reshape((-1, n))[:, i], reg_pred[:, i])
          r2 = r2_score(noise_train.reshape((-1, n))[:, i], reg_pred[:, i])
          metric_record.append([mse, r2])
claim: "The MSE/R^2 plotted in Figure 4 are computed by predicting on pred_train and comparing to noise_train, i.e. on the training split, while pred_test/noise_test are used only for the control-cost simulation."
concern: "Appendix A.5.1 states the MSE-rho curve is plotted on a 16000-sample held-out test set; the code reports in-sample (training) MSE instead, a code-vs-paper mismatch."
resolution: "Authors: confirm whether Figure 4 should report test-set MSE (predict on pred_test vs noise_test); with 64000 training samples the two values differ only at ~1e-3 (see _audit_code/out/example33.txt), so the figure and the equal-MSE conclusion are unaffected."
cross_refs: []
check_script: _audit_code/check_example33_theta_and_mse.py
paper_ref: "Appendix A.5.1, Figure 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: mgaps-lr-formula-mismatch
category: difference
topic: "hyperparameters"
title: "M-GAPS decaying learning rate in code differs from Appendix A.6 formula"
severity: low
confidence: high
status: finding
file: Example_Policy_Optimization_1.py
line_start: 67
line_end: 69
quote: |
          # use a decay learning rate
          lr = 1e-3 / np.sqrt(1 + i / 100)
          pol_opt.policy_update(lr=lr)
claim: "Both policy-optimization scripts use the step-size 1e-3 * (1 + t/100)^{-0.5} (same line in Example_Policy_Optimization_2.py:70)."
concern: "Appendix A.6 states the decaying learning rate sequence is eta_t = (1 + t/1000)^{-0.5}; the code uses a different time constant (100 vs 1000) and an explicit 1e-3 prefactor not shown in the paper formula."
resolution: "Authors: reconcile the learning-rate schedule used to generate Figures 1-2 with the formula reported in Appendix A.6 (the code's schedule is itself a valid decaying step size, so the convergence illustration stands)."
cross_refs: []
paper_ref: "Appendix A.6 ('decaying learning rate sequence eta_t = (1 + t/1000)^{-0.5}')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodological findings. The empirical sections are illustrations of a proven
theoretical phenomenon, not generalization claims, so leakage / sample-independence /
baseline-fairness concerns are structurally limited:

- **Data splitting / leakage** — train/test splits are on i.i.d. synthetic Gaussian
  samples regenerated each call (`Example_Mismatch_1.py:64-69`,
  `Example_Mismatch_2.py:29-47`); no shared units across splits, no preprocessing
  fitted on the full set. The equal-MSE construction is verified correct
  (`_audit_code/out/example33.txt`: diag(θᵀθ)=[1,1], θ matches the paper).
- **Baselines** — the no-prediction optimal policy ū=−Kx (the relevant lower bound)
  and the theoretical optimal cost are both plotted alongside M-GAPS in every figure
  (`Example_Policy_Optimization_1.py:84-93,128`).
- **Temporal integrity** — N/A: the "time step" axis is a control horizon over
  synthetic noise, not real time-series data with look-ahead risk.
- **Pretraining contamination** — N/A: no pretrained model or external embeddings.
- **Statistical integrity** — N/A: no significance tests or CIs are claimed; Figures
  1-2 show 25/75-percentile bands over 30 trials, consistent with Appendix A.6.

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 0          | -            | Complete, self-contained repo (env.yml, scripts, figures).  |
| bug         | 0          | -            | Scripts run and are internally consistent.                  |
| difference  | 2          | low          | Fig.4 MSE on train not test; M-GAPS LR differs from A.6.    |
| methodology | 0          | -            | Synthetic illustrations of a proven result; splits sound.   |

## Top take-aways

1. (`difference`, low) Figure 4's MSE-ρ curve is computed on the **training** split,
   not the held-out test set the paper describes — numerically negligible
   (~1e-3 gap) and does not affect the equal-MSE conclusion.
   [`mse-fig4-computed-on-train`]
2. (`difference`, low) The M-GAPS decaying learning rate in code
   (`1e-3·(1+t/100)^{-0.5}`) differs from Appendix A.6's stated
   `(1+t/1000)^{-0.5}`; the code schedule is still a valid decaying step.
   [`mgaps-lr-formula-mismatch`]

## Items that genuinely look fine

- **θ construction (core of Example 3.3)**: the code's θ matches the paper's
  [[1,0.99],[0,0.141]] and θᵀθ has unit diagonal, so predictors I and θ provably
  share per-entry MSE while differing in prediction power — verified
  (`_audit_code/out/example33.txt`).
- **Figure 3 closed forms** match the paper's expected-Q expressions exactly
  (EQ0=u²+1, (u±1)²).
- **Full figure coverage**: all six numbered figures trace to a computing script;
  `run_prediction_power_exp.sh` runs all five figure scripts.
- **Baselines and theory lines** (no-prediction ū=−Kx, theoretical optimal cost)
  are plotted alongside the method in every relevant figure.
- **Dependencies** are fully pinned in `env.yml`; the repo is self-contained with no
  external data, weights, or APIs.

## Open questions for the authors

- None high-severity. The two differences above are clarification requests, not
  blocking issues; both leave every paper conclusion intact.
- Minor (not filed as a finding): `Example_Mismatch_2.py` (Figure 6) sets no random
  seed (`:34`), unlike the other scripts; with N=200000 samples the curve ordering
  is robust, but a seed would make Figure 6 bit-reproducible.
