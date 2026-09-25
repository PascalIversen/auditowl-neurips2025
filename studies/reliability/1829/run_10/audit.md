# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (paper 1829)

## Summary

The repository `code/jackyue1994__OLinear/` is the official implementation of the
paper (README header confirms this and the model/scripts match the paper's
OLinear, OrthoTrans and NormLin descriptions). It is a fork of the
Time-Series-Library / iTransformer codebase: `run.py` is the entrypoint,
`experiments/exp_forecast.py` holds the train/eval loop, `data_provider/` holds
the chronological dataset loaders, `model/` holds the OLinear variants, and
`scripts/` holds per-dataset `.sh` driver scripts.

What I verified by reading and by running read-only checks under `_audit_code/`:
- **OrthoTrans basis derivation is leakage-free.** The orthogonal matrices are
  built from the *training portion only* (`dataset/Generate_corrmat.ipynb` slices
  `A = data[train_length-... : train_length]` with `train_ratio` 0.7, or 0.6 for
  ETT/PEMS), and the scripts request `ratioX` files matching each dataset's train
  fraction (`ETTh1_..._ratio0.6.npy`, `electricity_..._ratio0.7.npy`). This
  matches the paper's statement that `X_train` is used (paper §4.2).
- **Standard chronological splits, train-only scaler.** All `data_loader.py`
  dataset classes fit `StandardScaler` on the train slice only and split
  train/valid/test chronologically (no shuffling of the test loader). No
  preprocessing is fit on full data.
- **NormLin matches the paper's equation.** `layers/newLinear.py:27` and
  `layers/Transformer_EncDec.py:136-138` implement
  `RowNormL1(Softplus(W)) x`, exactly Eq. (NormLin) in §4.3.
- **No test-set selection in the headline path.** All OLinear models return a
  single tensor (not a tuple), so the multi-stage "best-stage by test MSE+MAE"
  block (`exp_forecast.py:799-857`) is inactive; `test_batch_size_list` has one
  entry in the shipped scripts and RevIN normalises per-instance so the
  best-batch-size loop is a no-op; `find_best`/lamda1-sweep defaults to off; test
  loss is printed each epoch but early stopping uses validation loss only
  (`exp_forecast.py:542-547`).
- **No hardcoded absolute paths** in the `.py` run path.

The defects I did find concern reproducibility of the *generality / robustness*
claims and packaging, plus one broken ablation script, not the core OLinear SOTA
numbers.

I ran: `_audit_code/check_model_registry.py`,
`_audit_code/check_plugin_baselines.py` (outputs in `_audit_code/out/`).

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 long-term MSE/MAE (OLinear column) | `model/OLinear.py` + `experiments/exp_forecast.py:793` (`metric`) via `scripts/OLinear/*.sh` | MSE/MAE in z-norm space | n/a (not re-run) | Code present (datasets external) |
| Table 2 baseline columns (iTrans, PatchTST, …) | (none) | — | — | Not in repo (cited benchmark numbers) |
| Table 1 FLOPs/memory NormLin vs MHSA | `experiments/exp_forecast.py:32-152` (`compute_model_stats`, needs `--model_stats_mode`, CUDA) | FLOPs/mem | n/a | Code present (GPU-only) |
| Table 3 short-term (S1/S2) | `model/OLinear.py` + `scripts/OLinear/*_S2.sh` | MSE/MAE | n/a | Code present |
| Table 4 transformation-basis ablation | `model/orthoLinear_basis/OLinear_{FFT,wavelet,wavelet2,cheby,Laguerre,Legendre}.py` + `scripts/ablation/basis/*.sh` | MSE/MAE | n/a | Mostly present; 3 scripts crash on bad model name (see `basis-script-unregistered-model`) |
| Table 5 OrthoTrans plug-in to iTransformer/PatchTST/RLinear | (none) | — | — | MISSING (no baseline models) |
| Table 9 NormLin replacing self-attention in iTransformer/PatchTST | (none) | — | — | MISSING (no baseline models) |
| Table 10 NormLin in Timer (large model) | (none) | — | — | MISSING (no Timer model) |
| Table 6 model ablation (NormLin/linear along var/temp) | `model/OLinear_ablation_var_temp.py` + `scripts/ablation/var_temp/*.sh` | MSE/MAE | n/a | Code present |
| Compare-with-attention (NormLin vs attn variants) | `model/OLinear_attn_var.py` + `scripts/ablation/attn_var/*.sh` | MSE/MAE | n/a | Code present |
| Table 13 / 26 std over 7 random seeds | run via `--itr N --fix_seed 0` (no aggregation script) | std | — | MISSING aggregation/seeding (see `no-stat-test-or-std-code` & `multi-seed-unseeded`) |
| Table 14 99% confidence intervals + Student's t-test (OLinear vs 2nd-best) | (none) | — | — | MISSING (no statistical-test code) |
| Figure 3 / Table 41 efficiency | `experiments/exp_forecast.py:32-152` | timings | n/a | Code present (GPU-only) |

## missing

```yaml finding
id: plugin-generality-models-missing
category: missing
topic: "result traceability / generality experiments"
title: "Plug-in/generality baselines (iTransformer, PatchTST, RLinear, Timer) absent from repo"
severity: high
confidence: high
status: finding
file: experiments/exp_basic.py
line_start: 3
line_end: 7
quote: |
  from model import OLinear, OLinear_C, OLinear_attn_var, OLinear_ablation_var_temp, OLinear_ablation_lin_design, \
      OLinear_no_Q_neither

  from model.orthoLinear_basis import OLinear_FFT, OLinear_wavelet, OLinear_wavelet2, OLinear_Legendre, OLinear_Laguerre, \
      OLinear_cheby
claim: "The model registry (model_dict) and the model/ directory contain only OLinear variants and basis ablations; there is no iTransformer, PatchTST, RLinear, DLinear or Timer implementation, and no script invokes one (verified by _audit_code/check_plugin_baselines.py)."
concern: "The paper's generality claims — Table 5 (OrthoTrans plugged into iTransformer/PatchTST/RLinear), Table 9 (NormLin replacing self-attention in iTransformer/PatchTST), and Table 10 (NormLin in Timer) — are headline contributions stated in the abstract ('consistently enhances Transformer-based forecasters') but cannot be reproduced because the baseline model code with OrthoTrans/NormLin integrated is not in the repository."
resolution: "Authors: please add the iTransformer/PatchTST/RLinear/DLinear/Timer model files (with the OrthoTrans and NormLin variants) and the driver scripts that produce Tables 5, 9 and 10."
cross_refs: ["§Model analysis (Table 5)", "§NormLin as a plug-in (Table 9)", "§Table 10"]
check_script: _audit_code/check_plugin_baselines.py
paper_ref: "Table 5, Table 9, Table 10; abstract"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-stat-test-or-std-code
category: missing
topic: "statistical integrity / result traceability"
title: "No code for the reported t-test, 99% CIs, or 7-seed standard deviations"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "We perform a significance test with 7 random seeds using Student's t-test to compare OLinear and"
claim: "The paper reports standard deviations over seven random seeds (Tables 13, 26), 99% confidence intervals (Table 14), and a Student's t-test between OLinear and the second-best method, but the repository contains no script that computes any of these (no scipy/ttest/std-aggregation code; grep over the repo finds none)."
concern: "Per result-traceability, every reported statistical test should trace to a script that computes it; here the significance test, CIs and per-seed std are produced off-repo and cannot be verified or reproduced."
resolution: "Authors: please provide the script that aggregates the 7-seed runs into the reported std / 99% CIs and computes the Student's t-test."
cross_refs: ["multi-seed-unseeded"]
paper_ref: "Tables 13, 14, 26; 'significance test with 7 random seeds using Student's t-test'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unpinned-dependencies
category: missing
topic: "expected code completeness / dependencies"
title: "requirements.txt has zero version pins; 'pywt' is the wrong PyPI package name"
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
claim: "None of the 13 dependencies carries a version constraint (0 lines contain '=='), and line 13 lists 'pywt' although the importable module 'pywt' is distributed on PyPI as 'PyWavelets' (so `pip install -r requirements.txt` fails on that line)."
concern: "Unpinned versions make the environment non-rebuildable (numpy/torch/timm API drift can change results or break the code), and the wrong package name causes the documented install command to error out."
resolution: "Authors: pin tested versions (e.g. torch==X, numpy==Y, …) and replace 'pywt' with 'PyWavelets'."
cross_refs: []
check_script: _audit_code/out/requirements_check.txt
paper_ref: "README 'pip install -r requirements.txt'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: benchmark-datasets-and-qmats-not-shipped
category: missing
topic: "data availability"
title: "Major benchmark datasets and their Q-matrices not in repo (behind external links)"
severity: low
confidence: medium
status: question
file: README.md
line_start: 47
line_end: 51
quote: |
  2. Some datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).  The others can be obtained from the links in Appendix B. 
  Some datasets are already in the folder ```./dataset```. 
  The python script ```Generate_corrmat.ipynb``` for generating Pearson correlation matrices and Q matrices is also provided in ```./dataset```.
claim: "Only the 'more datasets' (power, nasdaq, website, DowJones, wiki, ILI, SP500, cars, weather, covid, unemployment) and their precomputed Q-matrices ship in ./dataset; the main benchmarks reported in Table 2 (ECL/electricity, Traffic, Solar, PEMS03/04/07/08, ETT*, Exchange, METR) and their Q-matrix .npy files are not present and must be downloaded plus regenerated via the notebook (which requires per-dataset manual edits of file_path/train_ratio)."
concern: "The abstract states 'code and datasets are available'; for the headline-table datasets the user must fetch data from external drives and hand-edit/run the notebook to regenerate every Q-matrix, so the headline runs are not turnkey-reproducible from the repo alone."
resolution: "Authors: confirm the Google-Drive/Tsinghua links resolve and include all benchmark datasets, and consider shipping (or scripting non-interactively) the Q-matrix generation for the main benchmarks."
cross_refs: []
paper_ref: "Abstract ('code and datasets are available'); Table 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: basis-script-unregistered-model
category: bug
topic: "ablation scripts"
title: "Basis-ablation scripts pass unregistered model name 'OLinear_wavelet_concat' → KeyError"
severity: low
confidence: high
status: finding
file: scripts/ablation/basis/ECL_orthoLinear_basis.sh
line_start: 6
line_end: 21
quote: |
  model_names=(OLinear_FFT OLinear_wavelet_concat OLinear_wavelet2 OLinear_cheby OLinear_Laguerre OLinear_Legendre)
claim: "The active model_names array lists 'OLinear_wavelet_concat' (and 'OLinear_wavelet2_concat' in the Solar/PEMS03 basis scripts), and the loop `for ((l = 1; l < 3; l++))` passes index 1 = 'OLinear_wavelet_concat' to `--model`; that key is not in experiments/exp_basic.py model_dict (verified by _audit_code/check_model_registry.py), so `self.model_dict[self.args.model]` raises KeyError and the run crashes."
concern: "Three of the basis-ablation driver scripts (ECL, Solar, PEMS03) will crash as written; additionally the loop bounds skip OLinear_FFT and the three polynomial bases, so this script does not reproduce the full Table 4 row set."
resolution: "Authors: change the model names to the registered 'OLinear_wavelet'/'OLinear_wavelet2' (or register the *_concat variants) and fix the loop bounds to cover all six bases."
cross_refs: []
check_script: _audit_code/check_model_registry.py
paper_ref: "Table 4 (transformation-basis ablation)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: multi-seed-unseeded
category: difference
topic: "reproducibility / seeding"
title: "Multi-seed robustness runs are never explicitly seeded (fix_seed must be 0)"
severity: low
confidence: medium
status: finding
file: run.py
line_start: 467
line_end: 471
quote: |
  if args.fix_seed:
      fix_seed = 2023  # 2023  # if args.task_name == 'forecasting' else 2021
      random.seed(fix_seed)
      torch.manual_seed(fix_seed)
      np.random.seed(fix_seed)
claim: "Seeding only happens once when fix_seed is truthy, and run.py:414 requires fix_seed=0 whenever itr>1; the `for ii in range(args.itr)` loop (run.py:512) never re-seeds per iteration, so the 7-seed robustness runs (Tables 13/14/26) use whatever non-deterministic global RNG state exists rather than a fixed, recorded sequence of seeds."
concern: "The paper reports std/CIs 'over seven random seeds', but the code does not set or record those seeds, so the exact reported dispersion cannot be reproduced and the single-seed main results use a fixed seed=2023 rather than being averaged over seeds."
resolution: "Authors: set and log an explicit per-run seed inside the itr loop (e.g. seed = base+ii) so the 7-seed statistics are reproducible."
cross_refs: ["no-stat-test-or-std-code"]
paper_ref: "Tables 13, 14, 26 ('seven random seeds')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The procedures actually implemented are sound:
chronological splits with a train-only scaler, an OrthoTrans basis fitted on the
training portion only (matching the paper), per-instance RevIN, validation-based
early stopping, and metrics (MSE/MAE in z-normalised space) consistent with the
benchmark protocol the cited baselines also use. N/A for pretraining
contamination (the released OLinear runs train from scratch; the Timer
fine-tuning experiment that does involve a pretrained LTM is itself not in the
repo — see `plugin-generality-models-missing`).

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 4          | high         | Generality/plug-in baselines, stat-test code, deps, datasets |
| bug         | 1          | low          | Basis-ablation scripts crash on unregistered model name    |
| difference  | 1          | low          | 7-seed robustness runs are not explicitly seeded           |
| methodology | 0          | -            | Core pipeline is leakage-free and faithful to the paper    |

## Top take-aways

1. (missing, high) `plugin-generality-models-missing` — iTransformer/PatchTST/RLinear/Timer
   implementations behind the paper's generality claims (Tables 5, 9, 10) are absent.
2. (missing, medium) `no-stat-test-or-std-code` — no code computes the reported
   Student's t-test, 99% CIs, or 7-seed std (Tables 13/14/26).
3. (missing, medium) `unpinned-dependencies` — requirements.txt has no version
   pins and lists the wrong PyPI name `pywt`.
4. (bug, low) `basis-script-unregistered-model` — 3 basis-ablation scripts pass
   an unregistered `--model` name and crash.
5. (difference, low) `multi-seed-unseeded` — robustness runs are not explicitly
   seeded, so reported dispersion is not reproducible.
6. (missing, low) `benchmark-datasets-and-qmats-not-shipped` — main-benchmark
   data and Q-matrices require external download + manual notebook runs.

## Items that genuinely look fine

- OrthoTrans Q-matrix is computed from the **training portion only** (notebook
  slices `[train_length-...:train_length]`; scripts request `ratio0.6`/`ratio0.7`
  files matching each dataset's train fraction) — no test-set leakage into the
  basis, consistent with paper §4.2.
- Chronological train/valid/test splits with `StandardScaler` fit on train only
  (`data_provider/data_loader.py`); test loader not shuffled.
- NormLin implementation `RowNormL1(Softplus(W))x` matches the paper equation.
- Early stopping uses validation loss; test loss is only printed, not used to
  select checkpoints/epochs/hparams (no test leakage).
- The multi-stage "best stage by test MSE+MAE" and best-test-batch-size loops are
  inert for the released models (single-tensor output; per-instance RevIN; single
  test batch size), so reported numbers are not test-set-selected.
- All Table 4 / Table 6 / attention-comparison ablation models are present and
  registered.

## Open questions for the authors

- `plugin-generality-models-missing`: will the iTransformer/PatchTST/RLinear/Timer
  + OrthoTrans/NormLin code and scripts be released so Tables 5/9/10 can be
  reproduced?
- `no-stat-test-or-std-code` / `multi-seed-unseeded`: which exact seeds were used
  for the 7-seed runs, and what script produced the t-test and 99% CIs?
