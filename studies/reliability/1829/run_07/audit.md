# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (paper 1829)

## Summary

The repo `code/jackyue1994__OLinear/` is the official implementation of OLinear, a
Time-Series-Library-style codebase (`run.py` driver, `experiments/exp_forecast.py`,
`data_provider/`, `model/`, `scripts/`). It contains the OLinear model
(`model/OLinear.py`), the OLinear-C variant (`model/OLinear_C.py`), the model and
basis ablation variants, per-dataset shell scripts, a notebook
(`dataset/Generate_corrmat.ipynb`) that pre-computes the OrthoTrans `Q` matrices and
channel-correlation matrices, and 11 of the 24 benchmark datasets (csv + Q matrices)
bundled under `dataset/`.

What I checked / ran (read-only, under `_audit_code/`):
- `check_qmat_shapes.py` — confirmed bundled temporal Q-matrix `.npy` shapes match
  the requested seq_len/pred_len and the channel-corr file matches feature count.
- `check_channel_files_exist.py` — cross-referenced every script's `--q_channel_file`
  against files actually present for bundled datasets (44/50 referenced files missing).
- `check_models_and_keys.py` — parsed `model_dict` vs the model names requested by
  scripts (two undefined keys) and checked for baseline/plug-in model files (none).
- Read the OrthoTrans construction (paper §4.2 + notebook), data splitting
  (`data_provider/data_loader.py`), training/eval/early-stopping
  (`experiments/exp_forecast.py`), the driver/seed logic (`run.py`), and metrics
  (`utils/metrics.py`).

Methodologically the core pipeline is sound: train/val/test splits are standard (70/10/20
for `custom`, 60/20/20 for `custom2`, 12/4/4-month for ETT); the `StandardScaler` is fit
on the training segment only; and the OrthoTrans `Q` matrices / channel-correlation
matrices are computed from the *training portion only* (notebook slices to
`[train_length-len:train_length]` / `[0:int(len*0.7)]`), matching the paper's claim that
`Xtrain` is used (paper §4.2). I found no train/test leakage. Early stopping uses
validation loss only (test loss is printed but not used for selection). The findings below
are about reproducibility completeness (missing baseline/plug-in code, missing channel
matrices for most bundled datasets, unpinned deps) plus two minor code bugs.

## Traceability table (Rule G)

| Paper artefact | Repo location | Computed? | Matches | Status |
|---|---|---|---|---|
| Table 2 long-term MSE/MAE for **OLinear** | `run.py` + `experiments/exp_forecast.py:793` (`metric(...)`), scripts `scripts/OLinear/*.sh` | yes (writes `result_long_term_forecast.txt`) | not run (no GPU) | Code present; numbers not re-run |
| Table 3 short-term MSE/MAE for **OLinear** | scripts `scripts/OLinear/more_datasets/*`, `_S2` | yes | not run | Code present |
| Tables 2/3/8 **11 baselines** (TimeMixer, iTransformer, PatchTST, CARD, …) | `model/` has no baseline model files | no | — | MISSING (baseline code absent) — see `baselines-not-in-repo` |
| Table 5 / §5.2 **OrthoTrans as plug-in** (iTransformer, PatchTST, RLinear) | `run.py` flags `iTrans_ortho_trans`/`PatchTST_ortho_trans`/`DLinear_ortho_trans` exist but no such model in `model_dict` | no | — | MISSING — see `plugin-generality-code-missing` |
| Tables 9/10 **NormLin as plug-in** for Transformers / LTMs | no baseline/LTM model in repo | no | — | MISSING (folded into `plugin-generality-code-missing`) |
| Table 7 **OLinear-C** for cars/covid/DowJones/ILI/nasdaq/power/SP500/wiki/website/unemployment | `model/OLinear_C.py` present, but `*_COV_channel_*.npy` absent for all bundled datasets except weather | partial | — | MISSING channel matrices — see `channel-corr-matrices-missing` |
| Table 7 **OLinear-C** for weather | `model/OLinear_C.py` + `dataset/weather/weather_COV_channel_ratio0.70.npy` | yes | not run | Code+data present |
| Table 4 basis ablation (Fourier/Legendre/Laguerre/Chebyshev) | `model/orthoLinear_basis/OLinear_{FFT,Legendre,Laguerre,cheby}.py` + `scripts/ablation/basis/*` | yes | not run | Code present |
| Table 4 basis ablation (**Haar/Meyer wavelet**) on ECL/PEMS03/Solar | `scripts/ablation/basis/{ECL,PEMS03,Solar}_*basis.sh` request `OLinear_wavelet_concat`/`OLinear_wavelet2_concat` not in `model_dict` | crashes (KeyError) | — | BUG — see `undefined-wavelet-concat-model-key` |
| Table 6 CSL/ISL ablation | `model/OLinear_ablation_var_temp.py`, `scripts/ablation/var_temp/*` | yes | not run | Code present |
| Table 8 NormLin vs attention | `model/OLinear_attn_var.py`, `scripts/ablation/attn_var/*` | yes | not run | Code present |
| Table 1 NormLin vs MHSA FLOPs/memory | `experiments/exp_forecast.py:32` `compute_model_stats` (`--model_stats_mode`) | yes (FLOPs via fvcore) | not run | Code present |
| Table 13 std over 7 random seeds | `scripts/*/robust/*.sh` (`--itr 7 --fix_seed 0`) | yes (7 unseeded runs) | not run | Code present (seeds not recorded → exact values not reproducible, std is) |
| ECL/Traffic/Solar/PEMS/ETT/Exchange/METR datasets | not bundled; README points to Google Drive / Tsinghua Cloud / Appendix B | n/a | — | External standard datasets (documented fetch) |

## missing

```yaml finding
id: plugin-generality-code-missing
category: missing
topic: "result traceability / generality experiments"
title: "No code for OrthoTrans/NormLin plug-in experiments (iTransformer, PatchTST, RLinear)"
severity: medium
confidence: high
status: finding
file: experiments/exp_basic.py
line_start: 13
line_end: 25
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
claim: "The model registry contains only OLinear and its own variants/ablations; no iTransformer / PatchTST / RLinear / DLinear / TimeMixer model is registered or present in model/, although run.py defines flags iTrans_ortho_trans, PatchTST_ortho_trans, DLinear_ortho_trans (run.py:78-80)."
concern: "One of the paper's three headline contributions — that OrthoTrans/NormLin as a plug-in 'consistently improves the performance of existing forecasters' (Table 5, Tables 9-10) — has no runnable code in the repo, so the generality results cannot be reproduced."
resolution: "Provide the modified baseline models (iTransformer/PatchTST/RLinear with OrthoTrans and NormLin plug-ins) and their scripts, or point to where they live."
cross_refs: ["baselines-not-in-repo", "§5.2 Generality", "Table 5"]
check_script: _audit_code/check_models_and_keys.py
paper_ref: "Section 5.2 'OrthoTrans as a plug-in', Table 5; Tables 9-10"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: channel-corr-matrices-missing
category: missing
topic: "data/artefact availability (OLinear-C)"
title: "Channel-correlation matrices required by OLinear-C absent for all bundled datasets except weather"
severity: medium
confidence: high
status: finding
file: scripts/OLinear_C/more_datasets/cars_orthoLinear.sh
line_start: 35
line_end: 39
quote: |
      --q_mat_file cars_${seq_len}_ratio0.7.npy \
      --q_out_mat_file cars_${pred_len}_ratio0.7.npy \
      --q_channel_file cars_COV_channel_ratio0.70.npy \
      --model_id Cars_OLinear_${seq_len}_${pred_len}_${d_model}_${lr}_corr_only \
      --model $model_name \
claim: "OLinear-C (model/OLinear_C.py:42-45) requires a per-dataset channel-correlation matrix `*_COV_channel_ratio0.70.npy`, but of the 50 script references to bundled datasets only weather's file exists; 44/50 referenced channel files are absent (cars, covid, DowJones, ILI, nasdaq, power, SP500, unemployment, website, wiki)."
concern: "Table 7 (OLinear-C results) and the var-corr / attn-var ablations cannot be reproduced out-of-the-box on the bundled datasets, because the channel matrices the scripts load are not shipped (they must be regenerated via dataset/Generate_corrmat.ipynb, which the README presents as optional)."
resolution: "Ship the `*_COV_channel_ratio0.70.npy` files for all bundled datasets, or state in the README that Generate_corrmat.ipynb must be run first to produce them."
cross_refs: ["channel-file-assert-typo", "Table 7"]
check_script: _audit_code/check_channel_files_exist.py
paper_ref: "Table 7 (OLinear vs OLinear-C)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "baselines / fair comparison"
title: "None of the 11 baseline models are present in the repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  We carefully choose 11 well-acknowledged state-of-the-art forecasting models as our
  baselines
claim: "The paper compares OLinear against 11 baselines (Tables 2, 3, 8), but model/ contains no baseline implementation (checked: iTransformer, PatchTST, DLinear, TimeMixer, Leddam, CARD, Fredformer, FITS, FilterNet, TimesNet all ABSENT); only OLinear variants are registered."
concern: "Baseline numbers cannot be reproduced from this repo and there is no script that runs baselines under the same split/metric/tuning budget, so the head-to-head comparison is not independently checkable here."
resolution: "Either include the baseline code/scripts used to produce Tables 2/3/8, or cite the exact source (repo+commit) and configuration for each baseline number."
cross_refs: ["plugin-generality-code-missing"]
check_script: _audit_code/check_models_and_keys.py
paper_ref: "Section 5.1 'Baselines'; Tables 2, 3, 8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unpinned-dependencies
category: missing
topic: "environment reproducibility"
title: "requirements.txt lists no versions; 'pywt' is not an installable PyPI name"
severity: low
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
claim: "All dependencies are unpinned (no versions), and the final entry `pywt` is not a valid PyPI distribution name (the package is `PyWavelets`, import name `pywt`); `pip install -r requirements.txt` will fail on `pywt`."
concern: "The exact environment cannot be rebuilt from the spec, and a literal `pip install -r requirements.txt` errors out on `pywt`, which is used by the wavelet-basis ablation (model/orthoLinear_basis/OLinear_wavelet.py:10)."
resolution: "Pin versions (especially torch/numpy) and replace `pywt` with `PyWavelets` (or `pywavelets`)."
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: undefined-wavelet-concat-model-key
category: bug
topic: "ablation scripts"
title: "Basis-ablation scripts request model keys 'OLinear_wavelet_concat'/'OLinear_wavelet2_concat' not in model_dict"
severity: low
confidence: high
status: finding
file: scripts/ablation/basis/ECL_orthoLinear_basis.sh
line_start: 6
line_end: 6
quote: |
  model_names=(OLinear_FFT OLinear_wavelet_concat OLinear_wavelet2 OLinear_cheby OLinear_Laguerre OLinear_Legendre)
claim: "The active model_names arrays in scripts/ablation/basis/{ECL,PEMS03,Solar}_*basis.sh include `OLinear_wavelet_concat` (and `OLinear_wavelet2_concat` in PEMS03/Solar), but model_dict (experiments/exp_basic.py:13-25) only defines `OLinear_wavelet`/`OLinear_wavelet2`; exp_forecast.py:218 indexes `self.model_dict[self.args.model]` with no default."
concern: "Running these basis-ablation scripts as shipped raises `KeyError: 'OLinear_wavelet_concat'` at model construction, so the wavelet rows of Table 4 cannot be produced via these scripts."
resolution: "Rename the model_names entries to `OLinear_wavelet`/`OLinear_wavelet2`, or add the `*_concat` variants to model_dict."
cross_refs: []
check_script: _audit_code/check_models_and_keys.py
paper_ref: "Table 4 (wavelet rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: channel-file-assert-typo
category: bug
topic: "error handling"
title: "OLinear-C guard for missing channel file uses `assert ValueError(...)` (always truthy, never raises)"
severity: low
confidence: high
status: finding
file: model/OLinear_C.py
line_start: 42
line_end: 47
quote: |
        assert configs.q_channel_file is not None, 'configs.q_channel_file should not be None in orthoLinear'
        q_channel_file = os.path.join(configs.root_path, configs.q_channel_file)
        if os.path.isfile(q_channel_file):
            self.channel_corr_mat = torch.from_numpy(np.load(q_channel_file)).to(torch.float32).to(device)
        else:
            assert ValueError('self.channel_corr_mat should not be None in orthoLinear_corr_only')
claim: "When the channel-correlation file is absent, the intended error path is `assert ValueError(...)`, which asserts the truthiness of a (truthy) exception object and therefore never raises; `self.channel_corr_mat` is left unset and the code later fails with an opaque AttributeError at OLinear_C.py:74."
concern: "The defensive check does not do what it intends (it should `raise ValueError`), so a missing channel matrix produces a confusing downstream crash instead of the intended clear message — compounding the missing-file issue (see channel-corr-matrices-missing)."
resolution: "Replace `assert ValueError(...)` with `raise ValueError(...)`."
cross_refs: ["channel-corr-matrices-missing"]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

No `difference` findings. The implemented procedures I checked (train-only Q/scaler,
70/10/20 or 60/20/20 splits, validation-based early stopping, MASE as defined in Eq. 22,
metrics on standardized data) match the paper's descriptions where both are present.

## methodology

No `methodology` findings. I specifically checked for leakage and found none:
the OrthoTrans `Q` matrices and channel-correlation matrices are computed from the
training portion only (notebook cells slice to the train segment;
`ratio0.7`/`ratio0.6` in the filenames match the actual train fraction of the
corresponding loaders, verified for `custom` (0.7) and `custom2` (0.6)); the
`StandardScaler` is fit on the train segment only (`data_loader.py:69-70, 166-167,
308-310, 466-468`); early stopping uses validation loss, not test loss
(`exp_forecast.py:542-547`). The per-stage / per-test-batch-size "best of"
selection in `exp_forecast.py:852-857` and `run.py:568-576` does not bias the
reported OLinear/OLinear-C numbers because (a) both models return a single output
(no intermediate stages → single-element mse_list) and (b) all scripts pass exactly
one `--test_batch_size` (verified: only `test_batch_size 16` appears).

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                |
|-------------|------------|--------------|----------------------------------------------------------------|
| missing     | 4          | medium       | Plug-in/baseline code absent; OLinear-C channel matrices absent; deps unpinned |
| bug         | 2          | low          | Undefined wavelet model keys crash 3 basis scripts; dead assert guard |
| difference  | 0          | -            | Implemented procedure matches paper where both present         |
| methodology | 0          | -            | No leakage found; Q/scaler train-only, val-based early stopping |

## Top take-aways (≤6)

1. **[missing] plugin-generality-code-missing** — no code for the OrthoTrans/NormLin
   plug-in (generality) experiments (Table 5, Tables 9-10), a headline contribution.
2. **[missing] channel-corr-matrices-missing** — OLinear-C (Table 7) channel matrices
   ship only for weather; 44/50 referenced `*_COV_channel_*.npy` are absent.
3. **[missing] baselines-not-in-repo** — none of the 11 baseline models are in the repo,
   so Tables 2/3/8 comparisons aren't reproducible here.
4. **[missing] unpinned-dependencies** — unpinned deps and an invalid `pywt` name break
   `pip install -r requirements.txt`.
5. **[bug] undefined-wavelet-concat-model-key** — 3 basis-ablation scripts crash with
   KeyError on undefined model keys.
6. **[bug] channel-file-assert-typo** — `assert ValueError(...)` never raises, masking the
   missing-channel-file error.

## Items that genuinely look fine

- **No train/test leakage in OrthoTrans.** Q matrices and channel-correlation matrices
  are built from the training portion only (notebook slices; filename ratios match the
  loaders' train fractions).
- **Scaler fit on train only** across all dataset loaders.
- **Early stopping** uses validation loss; test loss is printed but never used to select
  checkpoint/epoch/hyperparameter.
- **Bundled datasets are self-contained for OLinear**: csv + temporal Q matrices present
  with correct shapes for all bundled datasets (and weather additionally has the channel
  matrix, so OLinear-C runs on weather).
- **MASE/R²/Pearson** implementations (utils/metrics.py) match the paper's Eq. 22 definitions.
- **Seed handling**: main results use a fixed seed (2023); robustness (Table 13) uses
  `--itr 7 --fix_seed 0` for 7 random-state runs — a real multi-run harness.

## Open questions for the authors

- Where is the code for the plug-in/generality experiments (iTransformer/PatchTST/RLinear
  with OrthoTrans and NormLin)? The `*_ortho_trans` flags exist but no model consumes them.
- For OLinear-C results on non-weather bundled datasets, were the channel matrices
  regenerated via Generate_corrmat.ipynb, and can they be shipped for reproducibility?
- For ETT/PEMS, the notebook comment says to use `ratio=0.6`; can you confirm the shipped
  (non-bundled) ETT/PEMS Q matrices used 0.6 (train fraction), not the 0.7 default, to
  avoid including validation data in the Q computation?
