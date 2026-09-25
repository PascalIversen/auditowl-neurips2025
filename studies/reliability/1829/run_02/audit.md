# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (paper 1829)

## 1. Summary

The repo `code/jackyue1994__OLinear/` is the official implementation of OLinear, a
linear time-series forecaster that (a) decorrelates the series with an orthogonal
matrix `Qi`/`Qo` derived from the temporal Pearson correlation matrix
(OrthoTrans), and (b) replaces cross-variate self-attention with a row-L1-
normalized Softplus weight matrix (NormLin). The repo contains the OLinear model
family (`model/OLinear*.py`), a Time-Series-Library-style data provider, a
training/eval driver (`run.py`, `experiments/exp_forecast.py`), a notebook that
generates the `Q` matrices (`dataset/Generate_corrmat.ipynb`), and `scripts/` for
the main and ablation experiments. It does **not** contain any baseline model
(iTransformer, PatchTST, RLinear, DLinear, Leddam, TimesNet, …), nor the headline
benchmark datasets (ECL, Traffic, Solar, PEMS, ETT, Exchange).

What I did (read-only):
- Read the paper method (§3–§5) and implementation details (Appendix D) and the
  README; mapped each main/appendix table to repo code.
- Read the data loaders, the `Q`-matrix generation notebook, the model forward
  passes (`OLinear.py`, `OLinear_C.py`), the NormLin layer
  (`layers/Transformer_EncDec.py::LinearEncoder`), the metric code
  (`utils/metrics.py`), the training/eval loop (`experiments/exp_forecast.py`),
  and the run driver (`run.py`), plus a sample of the shell scripts.
- Wrote two deterministic checks under `_audit_code/`:
  - `check_missing_data.py` → `_audit_code/out/missing_data.csv`: of 18 distinct
    `--root_path` dataset dirs referenced by the scripts, **7 are MISSING** from
    the repo (ECL/electricity, traffic, Solar, PEMS, ETT-small, exchange_rate,
    METR_LA); 11 (the short-term "more datasets") are present.
  - `check_plugin_models.py`: the model registry holds **only OLinear variants**
    (no paper baseline appears), and the six CLI plug-in flags
    (`iTrans_ortho_trans`, `PatchTST_ortho_trans`, `DLinear_ortho_trans`,
    `iTrans_linear`, `PatchTST_linear`, `Leddam_attnLinear`) are **never consumed**
    by any model.

Positive checks (things that look correct): the OrthoTrans `Q` matrices and the
across-variate correlation matrix are computed from the **training portion only**
(notebook slices `data[0:train_length]`), the `StandardScaler` is fit on train
only, early stopping selects the best-**validation** checkpoint, and the NormLin
layer matches Eq. 3/4 exactly. No train/test leakage was found in the active code
path.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 long-term MSE/MAE (OLinear) | `run.py` + `model/OLinear.py` + `scripts/OLinear/*.sh` | not runnable here | — | Code present, but datasets+Q-mats absent (see `missing-headline-datasets-and-qmats`) |
| Table 2/3 baseline columns (iTrans, PatchTST, DLinear, Leddam, …) | (none) | — | — | MISSING (no baseline model code; paper says values taken from prior work) |
| Table 3 short-term (cars/covid/…/wiki) | `scripts/OLinear/more_datasets/*.sh` + `dataset/<ds>/` | not run | — | Code+data PRESENT (reproducible) |
| Table 4 / 20 basis ablation | `model/orthoLinear_basis/*` + `scripts/ablation/basis/*.sh` | not run | — | Code present |
| Table 5 / 21 OrthoTrans→iTrans/PatchTST/RLinear | (none) | — | — | MISSING (no baseline model, plugin flags unconsumed, no scripts) |
| Table 6 / 22 CSL/ISL ablation | `model/OLinear_ablation_var_temp.py` + `scripts/ablation/var_temp/*.sh` | not run | — | Code present |
| Table 7 / 25 OLinear vs OLinear-C | `model/OLinear_C.py` + `scripts/OLinear_C/*.sh` | not run | — | Code present (datasets partly absent) |
| Table 8 / 23 NormLin vs attention variants | `model/OLinear_attn_var.py` + `scripts/ablation/attn_var/*.sh` | not run | — | Code present |
| Table 9 / 24 NormLin→iTrans/PatchTST | (none) | — | — | MISSING (no baseline model, plugin flags unconsumed) |
| Table 10 / 33 NormLin on large TS models | (none) | — | — | MISSING (no LTM code) |
| Table 13 / 26 std over 7 seeds | `scripts/.../robust/*.sh` (itr 7) produce per-run logs only | — | — | MISSING std/aggregation script |
| Table 14 99% confidence intervals | (none) | — | — | MISSING (no CI code) |
| Table 15 Student's t-test significance | (none) | — | — | MISSING (no scipy / t-test code) |
| Table 35 R²/r/MASE scale-free metrics | `utils/metrics.py:34-104` | computed by `metric()` | partial | MASE denominator non-standard (see `mase-nonstandard-denominator`) |
| Table 1 FLOPs/Memory NormLin vs MHSA | analytic table; `compute_model_stats` measures empirical FLOPs | — | — | Analytic (no per-number script needed) |

## 3. Findings

## missing

```yaml finding
id: missing-plugin-and-baseline-code
category: missing
topic: "result traceability / generality experiments"
title: "No baseline model code; OrthoTrans/NormLin plug-in tables have no implementation"
severity: high
confidence: high
status: finding
file: experiments/exp_basic.py
line_start: 13
line_end: 26
quote: |
        self.model_dict = {
            'OLinear': OLinear,
            'OLinear_C': OLinear_C,
            'OLinear_attn_var': OLinear_attn_var,
            'OLinear_ablation_var_temp': OLinear_ablation_var_temp,
            'OLinear_ablation_lin_design': OLinear_ablation_lin_design,
            'OLinear_no_Q_neither': OLinear_no_Q_neither,
            'OLinear_FFT': OLinear_FFT,
            'OLinear_wavelet': OLinear_wavelet,
            'OLinear_wavelet2': OLinear_wavelet2,
            'OLinear_Legendre': OLinear_Legendre,
            'OLinear_Laguerre': OLinear_Laguerre,
            'OLinear_cheby': OLinear_cheby,
        }
claim: "The model registry contains only OLinear variants; no iTransformer, PatchTST, RLinear, DLinear, Leddam or large-TS-model is in the repo, and the CLI plug-in flags (iTrans_ortho_trans, PatchTST_ortho_trans, DLinear_ortho_trans, iTrans_linear, PatchTST_linear, Leddam_attnLinear) defined in run.py:74-80 are never consumed by any model (verified by _audit_code/check_plugin_models.py)."
concern: "The paper's headline 'generality' claims — OrthoTrans applied to iTransformer/PatchTST/RLinear (Table 5/21) and NormLin plugged into Transformer forecasters and large TS models (Tables 9/24 and 10/33) — cannot be reproduced because neither the baseline models nor the driver scripts exist in the repo."
resolution: "Authors: please add the baseline-model implementations and the scripts that produce Tables 5, 9, 10 (and full Tables 21, 24, 33), or point to where the plug-in experiments were run."
cross_refs: ["§5.2", "Table 5", "Table 9", "Table 10"]
check_script: _audit_code/check_plugin_models.py
paper_ref: "Tables 5, 9, 10; README 'Generality' section"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-headline-datasets-and-qmats
category: missing
topic: "data / artefact availability"
title: "Main benchmark datasets and their Q-matrix .npy files are absent from the repo"
severity: medium
confidence: high
status: finding
file: scripts/OLinear/ECL_orthoLinear.sh
line_start: 27
line_end: 31
quote: |
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --q_mat_file electricity_${seq_len}_ratio0.7.npy \
      --q_out_mat_file electricity_${pred_len}_ratio0.7.npy \
      --q_channel_file electricity_COV_channel_ratio0.70.npy \
claim: "The main scripts reference ./dataset/electricity/electricity.csv and pre-computed Q matrices (e.g. electricity_96_ratio0.7.npy), but ./dataset/electricity/ and 6 other headline benchmark dirs (traffic, Solar, PEMS, ETT-small, exchange_rate, METR_LA) and all their .npy Q matrices are absent from the repo (verified by _audit_code/check_missing_data.py: 7 of 18 referenced dataset dirs MISSING)."
concern: "The numbers in the headline long-term tables (Table 2/17) cannot be reproduced from the repo as shipped; the CSVs are pointed to external Google-Drive/Tsinghua-Cloud links (unverifiable here) and the author-generated Q matrices for those datasets are not included and must be regenerated by hand via the notebook."
resolution: "Authors: include the Q-matrix .npy files for all reported datasets (or a script that generates them automatically for every dataset), and confirm the external dataset links resolve."
cross_refs: ["missing-headline-datasets-and-qmats-notebook"]
check_script: _audit_code/check_missing_data.py
paper_ref: "Table 2; Appendix B"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-stats-significance-code
category: missing
topic: "statistical integrity"
title: "No code computes the reported std dev, 99% CIs, or Student's t-test"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "Table 13: Robustness of OLinear performance. Standard deviations are calculated over seven random seeds"
claim: "The repo has no script that aggregates the 7 random-seed runs into mean±std (Tables 13, 26), computes the 99% confidence intervals (Table 14), or performs the Student's t-test significance comparison (Table 15); there is no scipy import and no t-test/CI/std-aggregation code anywhere in the repo (grep for scipy/ttest/confidence returns nothing in *.py)."
concern: "Reported statistical artefacts (std, CIs, p-values) are not traceable to any computation in the repo, so the significance/robustness claims cannot be independently checked."
resolution: "Authors: provide the aggregation/statistics script (mean±std, CI, t-test) that consumes the per-seed run logs and produces Tables 13–15 and 26."
cross_refs: ["§I", "Table 13", "Table 14", "Table 15"]
paper_ref: "Tables 13, 14, 15, 26"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: requirements-unpinned
category: missing
topic: "dependencies / environment"
title: "requirements.txt lists no version for any dependency"
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
claim: "Every dependency is listed without a version pin (no ==, >=, or ~=)."
concern: "The exact runtime environment cannot be reconstructed; future versions of torch/numpy/scikit-learn may change numerics or break the code, hampering reproduction of the reported numbers."
resolution: "Authors: pin versions (e.g. a frozen pip freeze / environment.yml) for at least torch, numpy, scikit-learn, pandas, fvcore."
cross_refs: []
paper_ref: "Appendix D"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-headline-datasets-and-qmats-notebook
category: missing
topic: "reproducibility / Q-matrix generation"
title: "Q-matrix notebook is single-file/hardcoded-path and not wired into the pipeline"
severity: low
confidence: high
status: finding
file: dataset/Generate_corrmat.ipynb
line_start: 1
line_end: 1
quote: |
  file_path = r'dataset\Solar\solar_AL.xlsx'  # replace with csv or excel file;
claim: "The only way to produce the Q matrices for the (absent) benchmark datasets is to manually edit hardcoded per-file Windows paths in Generate_corrmat.ipynb (file_path, save_path, train_ratio comment '0.6 for ETT and PEMS') and rerun each cell once per dataset; there is no automated batch generation."
concern: "Regenerating the missing Q matrices requires error-prone manual edits (and the right train_ratio per dataset, which is implicit), making faithful reproduction of OrthoTrans inputs fragile."
resolution: "Authors: convert the notebook into a parameterised script driven by the dataset config so Q matrices are generated automatically with the correct train ratio per dataset."
cross_refs: ["missing-headline-datasets-and-qmats"]
paper_ref: "§4.2; README step 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No active bug found in the reported code path. (See "Items that look fine" for two
test-time selection mechanisms that are *dead code* for the released
single-stage / single-batch-size configs.)

## difference

```yaml finding
id: earlystop-patience-mismatch
category: difference
topic: "training protocol"
title: "Scripts use early-stopping patience 5 (some 8), paper states 10 epochs"
severity: low
confidence: high
status: finding
file: scripts/OLinear/ECL_orthoLinear.sh
line_start: 61
line_end: 61
quote: |
      --patience 5 \
claim: "The released scripts mostly set --patience 5 (count across scripts/: 73 occurrences of 5, 19 of 8, 31 of 10), while the paper states early stopping 'halts training if the validation performance does not improve for 10 consecutive epochs'."
concern: "The early-stopping criterion that selects the reported checkpoint differs from the protocol stated in the paper; since it determines which epoch's weights are evaluated, it can change the reported MSE/MAE."
resolution: "Authors: clarify the patience actually used for the main results, and reconcile with the '10 consecutive epochs' claim in Appendix D."
cross_refs: []
paper_ref: "Appendix D, Implementation details"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: mase-nonstandard-denominator
category: difference
topic: "metrics"
title: "Reported 'MASE' scales by the test-target 1-step naive error, not the in-sample seasonal naive"
severity: low
confidence: medium
status: finding
file: utils/metrics.py
line_start: 81
line_end: 102
quote: |
    if y_naive is None:
        y_naive = y_true
    ...
    # naive MAE [batch, channel]
    mae_naive = np.mean(np.abs(y_true_flat[:, 1:, :] - y_naive_flat[:, :-1, :]), axis=1)

    # avoid potential error
    mae_naive[mae_naive < 1e-5] = np.nan

    # MASE
    mase = np.nanmean(mae_model / mae_naive)
claim: "MASE is computed with the denominator = mean |y_true[t] - y_true[t-1]| over the *test prediction horizon itself* (y_naive defaults to y_true), rather than the conventional in-sample (training-period) seasonal/naive forecast error."
concern: "The quantity labelled 'MASE' in Table 35 is computed differently from the standard MASE definition, so the absolute values are not comparable to MASE reported elsewhere; the denominator depends on the test target window."
resolution: "Authors: state the exact MASE definition used (denominator source and seasonality), and confirm the baseline MASE values in Table 35 use the identical definition."
cross_refs: ["missing-plugin-and-baseline-code"]
paper_ref: "Table 35"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding rises to `finding` status. The orthogonal `Q` matrices, the
across-variate correlation matrix, and the `StandardScaler` are all fit on the
training portion only (no look-ahead), checkpoint selection uses validation loss,
and the splits are standard chronological train/val/test. One transparency item is
recorded as a question below.

```yaml finding
id: hp-selection-criterion-undocumented
category: methodology
topic: "hyperparameter tuning"
title: "Per-dataset hyperparameters are tuned over ranges but the selection criterion is unstated"
severity: low
confidence: medium
status: question
file: paper.pdf
quote: |
  "with the initial learning rate selected from {10−4, 2 × 10−4, 5 × 10−4}. The model dimension D is chosen from {128, 256, 512}, ... The block number L is chosen from {1, 2, 3}."
claim: "The paper selects learning rate, model dimension D, batch size and block number L from discrete ranges per dataset (the chosen values are hardcoded in each script), but neither the paper nor the code states whether selection used the validation set or the test set."
concern: "If the per-dataset hyperparameter choice were made on the test split, the comparison would be optimistic; the repo's early-stopping uses validation loss, but the outer HP choice criterion is not documented."
resolution: "Authors: confirm that learning rate / D / L / batch size were selected on the validation set (not the test set), and describe the selection criterion."
cross_refs: ["earlystop-patience-mismatch"]
paper_ref: "Appendix D"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | Baseline/plug-in code absent (Tables 5/9/10); headline datasets+Q-mats absent; no std/CI/t-test code; deps unpinned |
| bug         | 0          | -            | No active bug; multi-stage / multi-batch-size test selection is dead code for released configs |
| difference  | 2          | low          | Early-stop patience (5 vs paper's 10); 'MASE' uses non-standard denominator |
| methodology | 1 (question)| low         | HP selection criterion (val vs test) undocumented; no leakage found in active path |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing]** No baseline models and no driver scripts for the OrthoTrans/NormLin plug-in tables (Tables 5/9/10) — the headline "generality" claims are not reproducible (`missing-plugin-and-baseline-code`, high/high).
2. **[missing]** Headline benchmark datasets + their author-generated Q-matrices are absent; main long-term tables not reproducible as shipped (`missing-headline-datasets-and-qmats`, medium/high).
3. **[missing]** No code computes the reported standard deviations, 99% CIs, or t-test significance (`missing-stats-significance-code`, medium/high).
4. **[missing]** `requirements.txt` pins no versions (`requirements-unpinned`, medium/high).
5. **[difference]** Early-stopping patience in scripts is mostly 5, paper states 10 epochs (`earlystop-patience-mismatch`, low/high).
6. **[difference]** Quantity labelled MASE uses a non-standard (test-target) denominator (`mase-nonstandard-denominator`, low/medium).

### Items that genuinely look fine
- OrthoTrans `Qi`/`Qo` and the across-variate correlation matrix are computed from the **training portion only** (`Generate_corrmat.ipynb` slices `data[0:train_length]`); no look-ahead leakage into the transform.
- `StandardScaler` is fit on the train split only (`data_loader.py:69`, `:308`).
- Early stopping selects the best-**validation** checkpoint (`utils/tools.py` EarlyStopping uses `val_loss`); per-epoch test loss is logged but never used for selection.
- NormLin matches paper Eq. 3/4: `A = F.normalize(F.softplus(weight_mat), p=1, dim=-1); A @ values` (`Transformer_EncDec.py:136-141`).
- The short-term "more datasets" experiments (cars, covid, DowJones, ILI, nasdaq, power, SP500, unemployment, weather, website, wiki) ship with both CSVs and Q-matrices and are self-contained.
- The min-over-stages test-set selection (`exp_forecast.py:852-858`) and the min-over-test-batch-sizes selection (`run.py:568-576`) are inert for the released configs (models return a single tensor, not a tuple; `test_batch_size_list` has one element), so they do not affect reported numbers.

### Open questions for the authors
- Were per-dataset hyperparameters (LR, D, L, batch size) selected on the validation set or the test set? (`hp-selection-criterion-undocumented`)
- Do the external dataset download links (Google Drive / Tsinghua Cloud / Appendix B) still resolve, and will Q-matrices be shipped for all datasets?
- Exact MASE definition and whether baselines in Table 35 used the identical computation.
