# Audit — *Spectral Learning for Infinite-Horizon Average-Reward POMDPs* (NeurIPS 2025, #1576)

## Summary

The repository (`alesnow97/Spectral_Learning_POMDP`) is the official implementation of a
theory-driven RL paper. Its empirical contribution is two figure families: estimation-error
of the *Mixed Spectral Estimation* algorithm (Figs. 1, 3) and a regret comparison of
*Mixed Spectral UCRL* vs. SEEU and SM-UCRL (Figs. 2, 4). The repo ships the algorithm
implementations (`strategy/`), the POMDP environment (`environment/`), policies
(`policies/`), two experiment drivers (`run_estimation_error_experiments.py`,
`run_regret_experiments.py`), two plotting scripts (`plots/`), and the *raw saved JSON
outputs* of the experiments under `NeurIPS_experiments/`. There is no held-out-test /
train-test concept — this is a simulation paper (the "data" are synthetic POMDP rollouts),
so data-splitting / leakage / pretraining / temporal checklists are N/A.

I verified the data backing each figure (`_audit_code/check_run_counts.py`), the
confidence-interval helper that both plot scripts use
(`_audit_code/check_ci_formula.py`), and two concrete runtime blockers
(`_audit_code/check_runtime_blockers.py`). The paper's numbers are *plots*, not tabulated
scalars, so the traceability check is about whether the figure-generating data and the
plotting pipeline are present, runnable, and consistent with the captions. The substantive
findings concern (1) an inverted/mislabelled confidence band in the only error-bar code,
(2) two runtime blockers in the released scripts, and (3) the captioned "10 runs" not being
backed by the released data for the headline instances.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 1 estimation error, S=4,A=3,O=4, "10 runs, 95% c.i." | data `4states_3actions_4obs/pomdp1/estimation_error/1.json`; plotted by `plots/run_estimation_err_plot.py:plot_estimation_error` | curves present; data has `num_experiments=5` | runs ✗ (5≠10); CI band ✗ | MISMATCH (see `est-runs-not-ten`, `ci-band-inverted`) |
| Fig. 2 regret, S=3,A=3,O=4, "10 runs, 95% c.i." | data `3states_3actions_4obs/pomdp{1,3,5}/regret/13Ep_...`; plotted by `plots/run_regret_plot.py:plot_regret` | regret = oracle−algo, cumsum; data has 5 runs (pomdp3/5) or 3 (pomdp1) | curves ✓; runs ✗ (≤5≠10); CI band ✗ | MISMATCH (see `regret-runs-not-ten`, `ci-band-inverted`) |
| Fig. 3-left estimation error, S=3,A=2,O=5, "10 runs" | data `3states_2actions_5obs/pomdp5/estimation_error/3.json` | `num_experiments=10`, 30 ckpts | runs ✓ | Verified (data side) |
| Fig. 3-right estimation error, S=5,A=3,O=5, "10 runs" | data `5states_3actions_5obs/pomdp6/.../1_first_two_exp.json`+`2_last_three_exp.json` | 2+3 = 5 runs total | runs ✗ (5≠10); filenames don't match plot loader | MISMATCH (see `est-runs-not-ten`); also plot loader reads `1_first_exps`/`2_last_exps` |
| Fig. 4 regret violating Ass. 6.1, S=3,A=3,O=4 | not located as a distinct saved directory (only one regret dir name per pomdp) | — | — | QUESTION (see `fig4-violating-data-not-located`) |
| Regret hyperparams: SEEU τ1=1e5, τ2=3e5; MS-UCRL T0=3e5 (App. J.2) | `run_regret_experiments.py:74-82` uses τ1=1e4, τ2=3e4, T0=3e4 | code values, confirmed in saved dir name `..._10000tau1_30000tau2_..._30000MXTO` | ✗ | MISMATCH (see `regret-hparams-differ`) |
| Estimation-error driver instance (main paper Fig. 1 = S=4,A=3,O=4) | `run_estimation_error_experiments.py:34-36` defaults S=3,A=2,O=5 | produces Fig. 3-left instance, not Fig. 1 | ✗ (default ≠ headline) | MISMATCH (low; see `est-runs-not-ten` cross-ref) |
| Any single scalar / table / statistical test | (none in paper) | — | — | N/A — paper reports only figures |

## missing

```yaml finding
id: fig4-violating-data-not-located
category: missing
topic: "result traceability"
title: "Figure 4 (regret under violated Assumption 6.1) data not separately identifiable in repo"
severity: low
confidence: low
status: question
file: paper.pdf
quote: |
  Figure 4: Regret comparison on a POMDP with S “ 3, A “ 3, O “ 4 violating Assumption 6.1 (10
  runs, 95 %c.i.).
claim: "Appendix J reports a second regret experiment (Fig. 4) on an instance that violates Assumption 6.1, but the repo holds only one regret directory name per POMDP (13Ep_0.04discr_10000tau1_30000tau2_0.02SMAC_30000SMT0_30000MXTO) under 3states_3actions_4obs/pomdp{1,3,5}, with no marker distinguishing the assumption-violating instance."
concern: "The data backing Fig. 4 cannot be unambiguously located, so that experiment is not independently reproducible from the released artefacts."
resolution: "Authors: indicate which saved POMDP instance / directory corresponds to the Assumption-6.1-violating Fig. 4, or add the missing run data."
cross_refs: ["regret-runs-not-ten"]
paper_ref: "Appendix J.2, Figure 4"
tags: [reforms:2, heil:bronze]
```

## bug

```yaml finding
id: ci-band-inverted
category: bug
topic: "statistical integrity / plotting"
title: "ci2() uses t.ppf(1-conf) (negative quantile) -> error band inverted, conf level wrong"
severity: medium
confidence: high
status: finding
file: plots/run_regret_plot.py
line_start: 17
line_end: 27
quote: |
  def ci2(mean, std, n, conf=0.85):
      # Calculate the t-value
      t_value = t.ppf(1 - conf, n - 1)

      # Calculate the margin of error
      margin_error = t_value * std / math.sqrt(n)

      # Calculate the lower and upper bounds of the confidence interval
      lower_bound = mean - margin_error
      upper_bound = mean + margin_error
      return lower_bound, upper_bound
claim: "Both plot scripts compute their shaded 'confidence interval' as mean ± t.ppf(1-conf, n-1)·std/sqrt(n); for conf=0.85 (regret) and conf=0.90 (estimation error) the quantile t.ppf(1-conf,·) is NEGATIVE, so lower_bound lands above the mean and upper_bound below it."
concern: "The error bands in every figure (Figs. 1-4) are computed with an inverted (and wrong-level) t-multiplier, so the captioned '95% c.i.' bands are neither 95% nor correctly oriented; the only quantification of statistical significance in the paper is mis-computed."
resolution: "Use the two-sided multiplier t.ppf(1-(1-0.95)/2, n-1) (i.e. t.ppf(0.975, n-1)) for a true 95% CI; verified numerically that the current call returns negative t-values (e.g. -1.19 for n=5 vs the correct +2.78)."
cross_refs: ["ci-conf-level-mismatch", "ci-hardcoded-n"]
check_script: _audit_code/check_ci_formula.py
paper_ref: "Figures 1-4 captions, '95 %c.i.'"
tags: [stats:auc-ci, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: get-base-path-wrong-dirname
category: bug
topic: "reproducibility / plotting entrypoint"
title: "get_base_path() searches for dir 'NeurIPS_Average_Reward_POMDP' absent from the clone -> infinite loop"
severity: medium
confidence: high
status: finding
file: utils.py
line_start: 146
line_end: 153
quote: |
  def get_base_path():
      current_path = os.getcwd()

      while not os.path.basename(current_path) == "NeurIPS_Average_Reward_POMDP":
          parent_path = os.path.dirname(current_path)
          current_path = parent_path

      return current_path
claim: "Both plot scripts build every data path via os.path.join(utils.get_base_path(), ...); get_base_path() walks parent directories until one is named 'NeurIPS_Average_Reward_POMDP', a name that does not occur anywhere in the released repository (its root is 'Spectral_Learning_POMDP')."
concern: "Run from the published repo the loop reaches the filesystem root, where os.path.dirname('/')=='/', so it never terminates (hangs); the plotting pipeline that produces every figure cannot run as shipped without editing this hardcoded name."
resolution: "Replace the hardcoded directory name with a repo-relative path resolution (e.g. relative to __file__) or rename the loop target to the actual repo directory."
cross_refs: []
check_script: _audit_code/check_runtime_blockers.py
tags: [reforms:4, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: numpy-random-integers-removed
category: bug
topic: "dependencies / runtime"
title: "np.random.random_integers used but removed in the pinned numpy 2.2.6"
severity: medium
confidence: medium
status: finding
file: simulations/estimation_error/simulation_spectral_estimation_error.py
line_start: 151
line_end: 151
quote: |
            initial_state = np.random.random_integers(low=0, high=self.num_states - 1)
claim: "The estimation-error experiment seeds the initial state with np.random.random_integers, but requirements.txt pins numpy==2.2.6, and np.random.random_integers was removed in numpy 2.0."
concern: "Under the declared environment, regenerating the estimation-error data (Figs. 1, 3) raises AttributeError at this line, so the headline estimation experiment cannot be re-run with the pinned dependencies."
resolution: "Replace with np.random.randint(0, self.num_states) (or np.random.default_rng().integers). Note: I could only execute under a locally available numpy 1.26.4, where the deprecated function still exists; the removal in numpy 2.x is the basis for medium confidence — authors should confirm by installing requirements.txt verbatim."
cross_refs: []
check_script: _audit_code/check_runtime_blockers.py
tags: [reforms:4, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: regret-runs-not-ten
category: difference
topic: "result traceability / sample size"
title: "Fig. 2 captioned '10 runs' but released regret data has at most 5 runs"
severity: medium
confidence: high
status: finding
file: plots/run_regret_plot.py
line_start: 292
line_end: 292
quote: |
    num_experiments = 5
claim: "The released regret data under 3states_3actions_4obs/pomdp3 and pomdp5 contain 5 run indices (0-4) per algorithm, pomdp1 only 3 (0-2), and the plot script's default num_experiments is 5; Fig. 2 (and Fig. 4) caption states '10 runs, 95% c.i.'."
concern: "The reported 10-run average / CI is not backed by the released data, which contain at most 5 runs; the error bars and means a reviewer can reproduce differ from those claimed."
resolution: "Authors: release the full 10 runs used for Fig. 2/Fig. 4, or correct the caption to the number of runs actually used."
cross_refs: ["ci-band-inverted", "est-runs-not-ten"]
check_script: _audit_code/check_run_counts.py
paper_ref: "Figure 2 caption; Figure 4 caption"
tags: [forensics:hidden-iteration, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: est-runs-not-ten
category: difference
topic: "result traceability / sample size"
title: "Fig. 1 and Fig. 3-right captioned '10 runs' but backing data hold 5 runs"
severity: medium
confidence: high
status: finding
file: _audit_code/out/run_counts.txt
line_start: 2
line_end: 5
quote: |
  Fig1 main S=4,A=3,O=4: num_experiments=5 num_checkpoints=30 obs_frob_shape=(5, 30)
  Fig3-left S=3,A=2,O=5: num_experiments=10 num_checkpoints=30 obs_frob_shape=(10, 30)
  Fig3-right S=5,A=3,O=5 (file1_first_two): num_experiments=2 num_checkpoints=30 obs_frob_shape=(2, 30)
  Fig3-right S=5,A=3,O=5 (file2_last_three): num_experiments=3 num_checkpoints=30 obs_frob_shape=(3, 30)
claim: "Fig. 1 (main, S=4,A=3,O=4) data file 1.json reports num_experiments=5; the Fig. 3-right (S=5,A=3,O=5) data are split across 1_first_two_exp.json (2 runs) + 2_last_three_exp.json (3 runs) = 5 runs; both figure captions state '10 runs, 95% c.i.' (Fig. 3-left, S=3,A=2,O=5, genuinely has 10)."
concern: "Two of the three estimation-error figures average over 5 runs, not the captioned 10, so the curves and CI bands a reviewer can regenerate from the release do not match the stated sample size."
resolution: "Authors: release the remaining runs or correct the captions; also note the run-estimation driver defaults to S=3,A=2,O=5, not the main-paper S=4,A=3,O=4 instance."
cross_refs: ["regret-runs-not-ten", "ci-hardcoded-n"]
check_script: _audit_code/check_run_counts.py
paper_ref: "Figure 1 caption; Figure 3 caption"
tags: [forensics:hidden-iteration, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ci-conf-level-mismatch
category: difference
topic: "statistical integrity"
title: "Plot scripts use conf=0.90 / 0.85 while every caption states '95% c.i.'"
severity: low
confidence: high
status: finding
file: plots/run_estimation_err_plot.py
line_start: 17
line_end: 17
quote: |
  def ci2(mean, std, n, conf=0.90):
claim: "The estimation-error plotter calls ci2 with conf=0.90 and the regret plotter with conf=0.85, but all figure captions report '95% c.i.'."
concern: "Even setting aside the inverted-sign defect (see ci-band-inverted), the nominal confidence level coded does not match the 95% claimed in the captions."
resolution: "Authors: align the coded confidence level with the captioned 95% (conf argument and the t-quantile)."
cross_refs: ["ci-band-inverted"]
check_script: _audit_code/check_ci_formula.py
paper_ref: "Figures 1-4 captions"
tags: [stats:auc-ci, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ci-hardcoded-n
category: difference
topic: "statistical integrity"
title: "Estimation-error CI uses a hardcoded n (10/20), not the actual run count"
severity: low
confidence: high
status: finding
file: plots/run_estimation_err_plot.py
line_start: 67
line_end: 68
quote: |
    num_considered_episodes = 10
    confidence = 10
claim: "plot_estimation_error passes a literal confidence=10 as the sample size n to ci2 (plot_custom_estimation_error passes 20), instead of the true number of runs in the loaded data (5 for Fig. 1; 5 for Fig. 3-right)."
concern: "The CI width (std/sqrt(n) and the t degrees of freedom) is computed against the wrong n, so even the band magnitude is inconsistent with the underlying sample."
resolution: "Authors: pass observation_matrix_error_frobenious_norms.shape[0] as n (the code even has this commented out at lines 80 and 215)."
cross_refs: ["ci-band-inverted", "est-runs-not-ten"]
check_script: _audit_code/check_ci_formula.py
paper_ref: "Figure 1 / Figure 3 captions"
tags: [stats:auc-ci, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: regret-hparams-differ
category: difference
topic: "experimental settings"
title: "Regret hyperparameters in code are 10x smaller than Appendix J.2 reports"
severity: low
confidence: high
status: finding
file: run_regret_experiments.py
line_start: 73
line_end: 82
quote: |
    # SEEU
    tau_1 = 10000
    tau_2 = 30000

    # SM UCRL
    min_action_prob_smucrl = 0.02

    ############################################
    # SET EXPERIMENT LENGTH
    initial_episode_length = 30000
claim: "Appendix J.2 reports SEEU tau_1=1e5, tau_2=3e5 and Mixed Spectral UCRL / SM-UCRL initial episode T0=3e5, but the driver sets tau_1=1e4, tau_2=3e4, T0=3e4; the saved data directory name (13Ep_0.04discr_10000tau1_30000tau2_0.02SMAC_30000SMT0_30000MXTO) confirms the code values, not the paper values, produced the released results."
concern: "The hyperparameters that generated the released regret curves differ by a factor of 10 from those stated in the paper, so the paper's described setting and the reproducible setting disagree."
resolution: "Authors: reconcile the Appendix J.2 values with the driver / saved-directory values, and state which produced Fig. 2."
cross_refs: ["regret-runs-not-ten"]
paper_ref: "Appendix J.2 (SEEU/SM-UCRL/Mixed Spectral UCRL hyperparameters)"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fig3-caption-internal-inconsistency
category: difference
topic: "figure labelling"
title: "Figure 3 prose says right instance is S=5,A=5,O=5; caption and data say S=5,A=3,O=5"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  For the experiment on the right, we consider a larger POMDP instance (S “ 5, A “ 5, O “ 5)
claim: "Appendix J.1 prose describes the right-hand Fig. 3 instance as S=5, A=5, O=5, but the Fig. 3 caption and the released data directory (5states_3actions_5obs) both use A=3."
concern: "An internal labelling inconsistency about the instance size of a reported experiment; the released data fix the ambiguity (A=3) but the text disagrees."
resolution: "Authors: correct the prose to S=5, A=3, O=5 to match the caption and the released instance."
cross_refs: []
paper_ref: "Appendix J.1 (text vs Figure 3 caption)"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: data-generation-unseeded
category: methodology
topic: "reproducibility / seeding"
title: "Data-generating experiments set no RNG seed; only plot scripts seed"
severity: low
confidence: high
status: finding
file: simulations/estimation_error/simulation_spectral_estimation_error.py
line_start: 148
line_end: 151
quote: |
        for n in range(num_experiments):
            print("Experiment_n: " + str(n))

            initial_state = np.random.random_integers(low=0, high=self.num_states - 1)
claim: "Neither experiment driver nor either simulation class sets a numpy/Python RNG seed (grep finds np.random.seed only in the two plotting scripts); POMDP instances, policies, initial states, and rollouts are all drawn from the unseeded global RNG."
concern: "The released figures cannot be regenerated bit-for-bit, and rerunning yields a fresh random POMDP each time, so the specific instances behind Figs. 1-4 are not reproducible from the code alone (mitigated only because the raw outputs are shipped under NeurIPS_experiments/)."
resolution: "Authors: seed all RNG sources at the start of each experiment and record per-figure seeds, or document that reproducibility relies on the shipped JSON outputs."
cross_refs: ["numpy-random-integers-removed"]
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                             |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 1          | low          | Fig. 4 (assumption-violating regret) data not separately identifiable (question). |
| bug         | 3          | medium       | Inverted/mislabelled CI band; plot path loop hangs; numpy API removed. |
| difference  | 6          | medium       | "10 runs" not backed by data (≤5); regret hparams 10x off; CI level/n; Fig.3 size. |
| methodology | 1          | low          | Data-generation code is unseeded (raw outputs shipped, so mitigated). |

## Top take-aways (≤6, ranked by severity × confidence)

1. **[bug] `ci-band-inverted`** — The only error-bar code (`ci2`, used by both plot scripts) computes the band with `t.ppf(1-conf, n-1)`, a negative quantile, so every captioned "95% c.i." band is inverted and at the wrong confidence level (0.85/0.90). Deterministically verified.
2. **[difference] `regret-runs-not-ten` / `est-runs-not-ten`** — Figs. 1, 2 and 3-right are captioned "10 runs" but the released data contain at most 5 (regret pomdp1 only 3). The headline averages/CIs are not reproducible at the stated n.
3. **[bug] `get-base-path-wrong-dirname`** — `utils.get_base_path()` loops on a directory name (`NeurIPS_Average_Reward_POMDP`) absent from the clone; both plotting scripts hang when run as shipped.
4. **[bug] `numpy-random-integers-removed`** — `np.random.random_integers` (removed in numpy 2.0) is called while `requirements.txt` pins numpy 2.2.6, so regenerating the estimation-error data crashes under the declared environment (confirmed only on local numpy 1.26.4 → medium confidence).
5. **[difference] `regret-hparams-differ`** — The regret driver and the saved-data directory name use tau_1/tau_2/T0 that are 10x smaller than Appendix J.2 states.
6. **[difference] `ci-conf-level-mismatch` / `ci-hardcoded-n`** — Coded CI confidence (0.90/0.85) and a hardcoded n (10/20) both contradict the captioned 95% over the actual run count.

## Items that genuinely look fine

- The regret computation logic (`plots/run_regret_plot.py:183-243`): regret = oracle reward − algorithm reward over the reward channel `[2]` of `collected_samples`, then cumulative-sum and averaged — methodologically appropriate for cumulative regret.
- The three competing algorithms (Mixed Spectral UCRL, SEEU, SM-UCRL) and the oracle are each implemented and invoked in `simulation_spectral_regret.py`, run on the *same* POMDP instance under the same horizon — a fair common-instance comparison.
- The raw experiment outputs are actually shipped (`NeurIPS_experiments/`), so results can be inspected without rerunning; Fig. 3-left (S=3,A=2,O=5) genuinely has the claimed 10 runs.
- `requirements.txt` exists and is fully pinned; README documents project structure and the reproduce command.

## Open questions for the authors

- `fig4-violating-data-not-located`: which released directory corresponds to the Assumption-6.1-violating regret experiment (Fig. 4)?
- `numpy-random-integers-removed`: does the estimation-error driver actually run under the pinned numpy 2.2.6, or is an older numpy required?
- Were the headline Figs. 1/2 produced from 10 runs that were not released, or from the 5 runs present?
