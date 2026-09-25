# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (paper 1829)

## 1. Summary

The repo `jackyue1994__OLinear` is the official implementation of OLinear, a linear
multivariate time-series forecaster that (a) transforms the input/output series into a
"decorrelated" domain using an orthogonal matrix `Q` derived from the temporal Pearson
correlation matrix (OrthoTrans), and (b) replaces self-attention with a row-normalized linear
layer (NormLin). It is built on the Time-Series-Library / iTransformer codebase: a generic
`run.py` driver, `experiments/exp_forecast.py` training/eval loop, `data_provider/` loaders, and
per-dataset shell scripts under `scripts/`. Headline results are Tables 2 (long-term) and 3
(short-term); statistical robustness is reported in Appendix Tables 13–15 / 26 (std, 99% CIs,
Student's t-test over 7 seeds).

What I did (read-only):
- Read the model (`model/OLinear.py`, `model/OLinear_C.py`, and all variant models), the data
  loaders (`data_provider/data_loader.py`, `data_factory.py`), the training/eval loop
  (`experiments/exp_forecast.py`), the driver (`run.py`), the Q-matrix generator
  (`dataset/Generate_corrmat.ipynb`), `requirements.txt`, and a sample of the `scripts/`.
- Confirmed the train/valid/test split is chronological and the `StandardScaler` is fit on the
  training portion only; confirmed RevIN is per-instance; confirmed the orthogonal matrices `Q`
  are computed on the training slice only (matches the paper's "Let X_train denote the training
  set"). These are methodologically sound and are NOT findings.
- Wrote `_audit_code/check_stats_and_selection.py` to grep the whole repo for any
  std / confidence-interval / t-test computation, to confirm `requirements.txt` is unpinned, and
  to anchor the `run.py` test-set selection lines. Output in
  `_audit_code/out/stats_and_selection.json`.

Bottom line: the core method, splitting, scaling, and OrthoTrans precomputation are sound and
faithful to the paper. The substantive issues are (i) the statistical-robustness artefacts
(std, CIs, t-test) reported in the appendix have NO computation anywhere in the repo, (ii) for
multi-run experiments the driver reports the single best-on-test run rather than a mean, and
(iii) the dependency spec is unpinned and contains a non-installable package name.

## 2. Traceability table (Rule G)

| Paper artefact | Repo location | Computes value? | Status |
|---|---|---|---|
| Table 2 long-term MSE/MAE (per dataset) | `scripts/OLinear*/<DS>_orthoLinear.sh` → `run.py` → `exp_forecast.py:793` `metric()` | Yes (single fixed-seed run, `--itr 1`) | Traceable (raw large-dataset CSVs + their Q-matrices not shipped; download link in README) |
| Table 3 short-term MSE/MAE | `scripts/OLinear*/*_S2*.sh`, `more_datasets/*.sh` → same path | Yes | Traceable; small-dataset CSVs + Q-matrices ARE shipped (`dataset/<DS>/`) |
| Table 2/3 "1st Count" rows | (none) | No (derived from table cells; baselines from cited papers) | Not separately checkable |
| Table 13 std over 7 seeds (OLinear) | `scripts/OLinear/robust/*.sh` log per-run mse/mae; aggregation (none) | No code computes std | MISSING (see `missing-significance-stats-code`) |
| Table 26 std over 7 seeds (OLinear-C) | `scripts/OLinear_C/robust/*.sh`; aggregation (none) | No | MISSING (same) |
| Table 14 99% CIs (3×std, 7 seeds) | (none) | No | MISSING (same) |
| Table 15 Student's t-test p-values vs iTransformer | (none) | No t-test code in repo | MISSING (same) |
| Ablations: var/temp, attn variants, basis (Fig 9–14, Tabs 25/27…) | `scripts/ablation/*`, `model/OLinear_ablation_*`, `model/OLinear_attn_var.py`, `model/orthoLinear_basis/*` | Yes | Traceable |
| Robustness to limited Q-matrix training data (App I.10) | `dataset/Generate_corrmat.ipynb` `base_ratio` loop | Yes (generates `ratio<train_ratio` Q-matrices) | Traceable |
| Efficiency (FLOPs/mem, Table 1 / App) | `experiments/exp_forecast.py:32` `compute_model_stats` (fvcore) | Yes | Traceable |

## 3. Findings

## missing

```yaml finding
id: missing-significance-stats-code
category: missing
topic: "statistical integrity / result traceability"
title: "No code computes the per-seed std, 99% CIs, or t-test p-values (Tables 13/14/15/26)"
severity: medium
confidence: high
status: finding
file: run.py
line_start: 582
line_end: 594
quote: |
                # log into txt
                mse_mse_string = (f'mse:{mse:.5f}, mae:{mae:.5f}, lamda1:{lamda1:.2f}, '
                                  f'git_multi_stage:{args.git_multi_stage}, decoder_cat:{args.decoder_cat_num}, '
                                  f'alpha1:{args.alpha}, loss_fun_alpha1:{args.lossfun_alpha}')
                print(mse_mse_string)
                with open(log_txt, 'a') as f:
                    f.write(f'------------ {setting} -------------' + '\n' + '\n')
                    args_dict = vars(args)
                    for k, v in sorted(args_dict.items()):
                        f.write(f'{k}: {v}, ')
                    f.write('\n\n')
                    f.write('\t' + mse_mse_string + '\n\n')
                    f.write('--------------------------------- Ends -----------------------------\n\n')
claim: "run.py writes each run's mse/mae to a text log, but no script in the repo aggregates the 7 per-seed runs into a standard deviation, a 99% confidence interval, or a Student's t-test p-value. A whole-repo grep (_audit_code/check_stats_and_selection.py) finds zero genuine std/CI/t-test computations (all keyword hits are filenames like 'model_stats.txt' or LayerNorm/positional-encoding '.std()' calls)."
concern: "Tables 13, 14, 15, and 26 (standard deviations, 99% CIs, and the OLinear-vs-iTransformer Student's t-test with p<0.05) are headline robustness/significance artefacts, yet nothing in the released code produces them, so they cannot be reproduced or checked from the repo."
resolution: "Authors: please add the script that reads the per-seed logs and computes the standard deviations, the 99% CIs, and the Student's t-test p-values reported in Tables 13/14/15/26, and state whether the per-method central value entering the t-test is the mean over the 7 seeds."
cross_refs: ["best-of-itr-test-selection"]
check_script: _audit_code/check_stats_and_selection.py
paper_ref: "Tables 13, 14, 15, 26; App F, H.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unpinned-dependencies
category: missing
topic: "expected code completeness / dependencies"
title: "requirements.txt is fully unpinned (no versions for torch/numpy/etc.)"
severity: medium
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 13
quote: |
  pandas
  scikit-learn
  numpy
  matplotlib
  torch
  fvcore
  einops
  thop
  timm
  reformer_pytorch
  openpyxl
  seaborn
  pywt
claim: "Every dependency is listed without a version constraint (no ==, >=, ~=)."
concern: "Numerical results (and even importability, given reformer_pytorch/timm API churn) depend on specific versions of torch/numpy; with no pins the exact environment that produced the tables cannot be rebuilt."
resolution: "Authors: pin the versions (e.g. a frozen `pip freeze` or `torch==x.y`, `numpy==x.y`) used to produce the reported results."
cross_refs: ["requirements-pywt-bad-name"]
check_script: _audit_code/check_stats_and_selection.py
paper_ref: "README 'Install Pytorch and necessary dependencies'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: requirements-pywt-bad-name
category: bug
topic: "dependencies / installability"
title: "requirements.txt lists 'pywt' but the PyPI package is 'PyWavelets'; pip install fails"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 13
line_end: 13
quote: |
  pywt
claim: "The file lists `pywt` as a dependency, but `pywt` is the import name; the installable PyPI package is `PyWavelets`. `pip install pywt` returns 'No matching distribution found for pywt' (verified in this sandbox)."
concern: "The README's documented install step `pip install -r requirements.txt` aborts on this line, so a fresh environment cannot be created as instructed; `pywt` is imported by the wavelet-basis ablation models (model/orthoLinear_basis/OLinear_wavelet*.py)."
resolution: "Replace `pywt` with `PyWavelets` in requirements.txt."
cross_refs: ["unpinned-dependencies"]
paper_ref: "README install instructions"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: early-stopping-patience-mismatch
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Paper states early-stopping patience 10; scripts use patience 5 / 8"
severity: low
confidence: high
status: finding
file: scripts/OLinear/robust/ETTh1_orthoLinear.sh
line_start: 64
line_end: 64
quote: |
      --patience 8 \
claim: "The paper says training halts 'if the validation performance does not improve for 10 consecutive epochs' (App D), but the released scripts pass --patience 8 (robust ETTh1) or --patience 5 (e.g. scripts/OLinear/ECL_orthoLinear.sh), never 10. Early stopping itself is on validation loss (exp_forecast.py:547), which is sound."
concern: "A reproducer following the paper's stated patience of 10 would train longer than the scripts and could obtain different numbers; the discrepancy is small but is a concrete paper-vs-code mismatch in a training detail."
resolution: "Authors: reconcile the stated patience (10) with the per-dataset script values (5/8), or note that patience was tuned per dataset."
cross_refs: []
paper_ref: "Appendix D, Implementation details"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: best-of-itr-test-selection
category: methodology
topic: "hyperparameter tuning / model selection on test set"
title: "Multi-run driver reports the single best-on-test run (min MSE+MAE over --itr runs)"
severity: medium
confidence: medium
status: finding
file: run.py
line_start: 578
line_end: 580
quote: |
                if mse + mae <= best_mse + best_mae:
                    best_lamda1 = lamda1
                    best_mse, best_mae, best_ii = mse, mae, ii
claim: "For multi-run experiments (the robust scripts pass --itr 7 --fix_seed 0), run.py evaluates every run on the TEST set and keeps best_mse/best_mae = the minimum-(MSE+MAE) run; this best-of-N value is what is printed and written to best_log_txt / best_log_dataset_txt. (The inner test_batch_size loop is inert in the shipped scripts because test_batch_size_list has one element.)"
concern: "Selecting the run with the lowest test error across 7 seeds is test-set-based model selection; if these best-of-7 values are the central values reported for OLinear in the robustness/significance tables (13/14/15/26), the std would be understated and the t-test against iTransformer (compared on possibly mean values) would be biased. Per-task hyperparameters (lr/D/L/batch) are also chosen from sets (App D) without a stated validation-only criterion."
resolution: "Authors: confirm whether the reported per-seed central values are the mean over the 7 runs or the best-on-test run; if best-on-test, re-report using the mean and recompute the t-test."
cross_refs: ["missing-significance-stats-code", "seq-inter-test-stage-selection"]
check_script: _audit_code/check_stats_and_selection.py
paper_ref: "Tables 13/14/15/26; App D, F, H.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: seq-inter-test-stage-selection
category: methodology
topic: "model selection on test set (latent)"
title: "test() selects the intermediate stage with lowest test MSE+MAE (inert for shipped models)"
severity: low
confidence: low
status: question
file: experiments/exp_forecast.py
line_start: 849
line_end: 858
quote: |
        mse_list.append(mse)
        mae_list.append(mae)

        mse_plus_mae = [a + b for a, b in zip(mse_list, mae_list)]

        # print min_mse
        # print(mse_list, mae_list)
        min_index = mse_plus_mae.index(min(mse_plus_mae))
        mse = mse_list[min_index]
        mae = mae_list[min_index]
claim: "When a model returns multiple intermediate-stage outputs (self.args.seq_inter defaults to 1), test() computes MSE+MAE per stage on the TEST set and reports the argmin stage. For every OLinear model in this repo (OLinear, OLinear_C, all ablation/basis variants) forward() returns a single tensor, so mse_list==[mse] and min_index==0 — the selection is a no-op for the released models."
concern: "The reported metric is, by construction, the test-set-minimum over decoder stages; it is harmless here only because the shipped models emit one stage, but the same code path would pick the best-on-test stage for any multi-stage model and is a latent test-selection mechanism."
resolution: "Authors: confirm no reported OLinear number used a multi-stage configuration through this path; consider selecting the stage on validation, not test."
cross_refs: ["best-of-itr-test-selection"]
paper_ref: "Tables 2, 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 2          | medium       | No std/CI/t-test code for Tables 13/14/15/26; deps fully unpinned |
| bug         | 1          | low          | requirements lists non-installable `pywt` (should be PyWavelets) |
| difference  | 1          | low          | Early-stopping patience 10 (paper) vs 5/8 (scripts) |
| methodology | 2          | medium       | Multi-run driver reports best-on-test run; latent test-stage selection (inert) |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. `missing-significance-stats-code` (missing, med/high): the std, 99% CIs, and Student's t-test p-values in Tables 13/14/15/26 have no computing script anywhere in the repo.
2. `best-of-itr-test-selection` (methodology, med/med): for `--itr 7` robust runs the driver reports the single best-on-test run (min MSE+MAE), a test-set selection that, if used for the reported central values, biases the std and the significance test.
3. `unpinned-dependencies` (missing, med/high): requirements.txt has no version pins, so the exact result-producing environment cannot be rebuilt.
4. `requirements-pywt-bad-name` (bug, low/high): `pip install -r requirements.txt` fails on the `pywt` line (package is `PyWavelets`).
5. `early-stopping-patience-mismatch` (difference, low/high): paper says patience 10, scripts use 5/8.
6. `seq-inter-test-stage-selection` (methodology/question, low/low): a latent test-set stage-selection path that is inert for the released single-stage models.

### Items that genuinely look fine
- Train/valid/test split is chronological and the `StandardScaler` is fit on the training segment only (`data_provider/data_loader.py:51-71, 268-311, 563-581`); no scaler leakage.
- RevIN is strictly per-instance (statistics taken from each input window; `layers/RevIN.py:41-62`); no cross-sample test leakage.
- The orthogonal matrices `Q` (OrthoTrans) are computed from the TRAINING slice only — `dataset/Generate_corrmat.ipynb` cell 0 uses `A = data[train_length-...:train_length, ...]` and cell 1 uses `A_ori[0:int(len_a*ratio)]` with `ratio = 0.7*base_ratio ≤ 0.7`. This matches the paper's "Let X_train denote the training set" (paper §4.2). No future-data leakage in the transform.
- Early stopping selects the checkpoint on validation loss, not test loss (`exp_forecast.py:542-556`); test loss is only printed (line 543/546), which is not leakage.
- Shipped small datasets (cars, covid, DowJones, ILI, nasdaq, power, SP500, unemployment, weather, website, wiki) include their precomputed Q-matrix `.npy` files, so the short-term tables for those datasets are runnable offline.

### Open questions for the authors
- Are the per-method central values in Tables 13/14/15/26 the mean over 7 seeds or the best-on-test run that `run.py` reports? (drives `best-of-itr-test-selection` severity)
- Where is the script that computes the standard deviations, 99% CIs, and the Student's t-test in Tables 13/14/15/26?
- The Q-matrix generator (`dataset/Generate_corrmat.ipynb`) contains a hardcoded absolute Windows path (`D:\SystemData\xunn\...` in cell 2) and requires manual per-dataset edits of `file_path`/`train_ratio`/`time_lag`; for the large benchmarks (ECL/ETT/Traffic/PEMS/Solar/Exchange) neither the raw CSVs nor the Q-matrices are shipped — can a turnkey generation script be provided?
