# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (paper 1829)

## 1. Summary

The repository `code/jackyue1994__OLinear/` is the official implementation of OLinear,
a linear time-series forecaster built on the Time-Series-Library / iTransformer
framework. The headline contributions are (a) **OrthoTrans**, a data-adaptive
orthogonal transform whose matrices `Q_i` (T×T) and `Q_o` (τ×τ) are derived from the
*training set's* temporal Pearson correlation matrix and pre-computed offline as
`.npy` files, and (b) **NormLin**, a row-L1-normalised Softplus linear layer used in
the Cross-Series Learner. The repo contains the model (`model/OLinear.py`,
`model/OLinear_C.py`), the training/eval harness (`run.py`,
`experiments/exp_forecast.py`), data loaders, ablation models (basis / attn-var /
var-temp), the Q-matrix generator notebook (`dataset/Generate_corrmat.ipynb`), and a
subset of the 24 benchmark datasets (cars, covid, DowJones, ILI, nasdaq, power, SP500,
unemployment, website, wiki, weather). The large standard benchmarks (ECL, Traffic,
Solar, ETT, PEMS, Exchange, METR-LA) and their Q-matrices are not shipped; the README
points to Google-Drive / Tsinghua-Cloud downloads and Appendix B links for them.

What I did (scripts under `_audit_code/`, outputs under `_audit_code/out/`):
- **`check_q_matrix_train_only.py`** — recomputed the committed temporal Q-matrix
  `dataset/weather/weather_96_ratio0.7.npy` from (i) the first-70% training portion and
  (ii) the full series, replicating `Generate_corrmat.ipynb` Cell 0. The committed file
  matches the **train-only** recomputation exactly (max |Δ| = 0.000000) and differs from
  the full-series version (max |Δ| = 0.47). => OrthoTrans is correctly computed from
  training data only (no look-ahead leakage in the transform).
- **`check_channel_matrix_train_only.py`** — same check for the channel-correlation
  matrix `weather_COV_channel_ratio0.70.npy`; matches train-only (Δ=0.000000), differs
  from full-series (Δ=0.10).
- **Smoke test** (inline): constructed `model.OLinear.Model` with the weather config and
  ran a forward pass on CPU; output shape (4,96,21) as expected, ~0.34 M params. The
  headline model is runnable.

Routing notes: scaler and both correlation transforms are fit on training data only;
early stopping / checkpoint selection use *validation* loss (sound). The two real
concerns are (1) the run.py driver reporting the **best of N runs/configs selected on
the test set** whenever `itr>1` (used by the `robust/` seed-sweep scripts), and (2)
missing baseline-model code for the OrthoTrans-as-plug-in generality table (Table 5).

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches | Status |
|---|---|---|---|---|
| Table 2 long-term MSE/MAE (OLinear) | `run.py` + `experiments/exp_forecast.py:560-892` (`OLinear`) | yes (model+eval present; datasets via download) | not run (no GPU/data) | Present; data via external links |
| Table 3 short-term MSE/MAE (OLinear) | same; datasets cars/covid/.../weather shipped | yes | not run | Present |
| Table 2/3 baseline columns (TimeMixer, iTransformer, PatchTST, DLinear, …) | (none) — only OLinear variants in `experiments/exp_basic.py:13-26` | no | — | Baseline code absent (numbers cited from prior work) |
| Table 5 OrthoTrans plug-in (iTransformer/PatchTST/RLinear) | (none) — args exist (`run.py:78-80`) but no model/script | no | — | MISSING (see `orthotrans-plugin-baselines-missing`) |
| Table 6 CSL/ISL design ablation (NormLin vs Attn) | `model/OLinear_attn_var.py`, `model/OLinear_ablation_var_temp.py` + scripts | yes | not run | Present |
| §5.2 basis ablation (Fourier/Haar/wavelet/Cheby/…) | `model/orthoLinear_basis/*` + `scripts/ablation/basis/*` | partially | — | 3/7 scripts reference undefined model (see `basis-script-undefined-model`) |
| Table 1 FLOPs/memory (analytic) | analytic formula; `compute_model_stats` measures empirical | yes | n/a | Present |
| Table 13 std over 7 seeds | `robust/*` scripts (`itr 7`); no std-aggregation script in repo | partial | — | std must be computed offline; driver only logs per-seed + "global best" |
| OrthoTrans transform from X_train (paper §4.2) | `dataset/Generate_corrmat.ipynb` Cell 0/1; verified against committed npy | yes | ✓ (Δ=0.0) | Verified train-only |

## 3. Findings

## missing

```yaml finding
id: orthotrans-plugin-baselines-missing
category: missing
topic: "generality / plug-in experiments"
title: "OrthoTrans plug-in code for iTransformer/PatchTST/RLinear (Table 5) absent"
severity: medium
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
claim: "The model registry contains only OLinear and its ablation variants; no iTransformer, PatchTST, RLinear, or DLinear implementation exists in model/, and no script sets --iTrans_ortho_trans 1 / --PatchTST_ortho_trans 1 / --DLinear_ortho_trans 1 (the flags exist in run.py:78-80 but are dead)."
concern: "Table 5's generality result ('OrthoTrans yields average MSE improvements of 5.1% and 10.1% for iTransformer and PatchTST') cannot be reproduced because the baseline forecasters OrthoTrans is plugged into are not in the repo."
resolution: "Authors: please add the iTransformer/PatchTST/RLinear model implementations and the plug-in run scripts used to generate Table 5 (and Table 6's plug-in rows)."
cross_refs: ["Table 5"]
paper_ref: "Section 5.2 'OrthoTrans as a plug-in'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: basis-script-undefined-model
category: bug
topic: "ablation scripts"
title: "Basis-ablation scripts (ECL/PEMS03/Solar) call undefined model OLinear_wavelet_concat"
severity: low
confidence: high
status: finding
file: scripts/ablation/basis/ECL_orthoLinear_basis.sh
line_start: 6
line_end: 6
quote: |
  model_names=(OLinear_FFT OLinear_wavelet_concat OLinear_wavelet2 OLinear_cheby OLinear_Laguerre OLinear_Legendre)
claim: "Three of seven basis-ablation scripts (ECL, PEMS03, Solar) list model names 'OLinear_wavelet_concat' / 'OLinear_wavelet2_concat' that are not keys of experiments/exp_basic.py:model_dict (only 'OLinear_wavelet' and 'OLinear_wavelet2' exist) and are not defined anywhere in model/orthoLinear_basis/."
concern: "Running these scripts crashes with a KeyError at self.model_dict[self.args.model] when it reaches the wavelet entry, so the wavelet-basis comparison for those datasets cannot be reproduced as scripted."
resolution: "Authors: rename 'OLinear_wavelet_concat'/'OLinear_wavelet2_concat' to the shipped 'OLinear_wavelet'/'OLinear_wavelet2' in the ECL/PEMS03/Solar basis scripts, or add the missing model files."
cross_refs: []
paper_ref: "Section 5.2 'Comparison with other transformation bases'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

(no `difference` findings — the model/eval procedures implemented match the paper's description where checkable.)

## methodology

```yaml finding
id: itr-best-of-n-selected-on-test
category: methodology
topic: "selective reporting / test-set selection"
title: "Driver reports best of N runs/configs selected by test MSE+MAE"
severity: medium
confidence: medium
status: finding
file: run.py
line_start: 568
line_end: 580
quote: |
                for test_bs in sorted(test_batch_size_list):
                    print('>>>>>>>testing : {} (test_batch_size: {})<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.
                          format(setting, test_bs))
                    mse0, mae0 = exp.test(setting, test=args.test_mode, test_batch_size=test_bs)

                    if mse0 < mse:
                        mse, mae = mse0, mae0
                        best_batch_size = test_bs
                print(f'\tbest_test_batch_size: {best_batch_size}, best_mse: {mse:.5f}, best_mae: {mae:.5f}')

                if mse + mae <= best_mse + best_mae:
                    best_lamda1 = lamda1
                    best_mse, best_mae, best_ii = mse, mae, ii
claim: "exp.test() returns the TEST-set MSE/MAE; across the itr loop (and any lamda1 sweep / multiple test_batch_size) the driver keeps the run with the lowest test MSE+MAE as best_mse/best_mae, which is what gets logged to best_log.txt / best_results/. The robust/ scripts run this with --itr 7 --fix_seed 0 (e.g. scripts/OLinear/robust/Weather_orthoLinear.sh:51,57), i.e. 7 seeds, reporting the best of 7 by test score."
concern: "Selecting the best of multiple training runs (or hyper-parameter settings) by test-set MSE/MAE is test-set selection; if any reported number is taken from best_results/ rather than a fixed single seed or a mean over seeds, it overstates performance — and the per-seed values needed for the Table 13 std are not aggregated by any script in the repo."
resolution: "Authors: confirm the main Tables 2/3 numbers come from the single-seed itr=1 main scripts (not the itr=7 robust best-of-7), and that Table 13's mean/std is computed over all 7 logged per-seed runs rather than the 'global best' the driver prints. Consider selecting by validation, not test."
cross_refs: ["itr-test-batch-size-selection"]
paper_ref: "run.py driver loop; Appendix F Table 13 (std over 7 seeds)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: itr-test-batch-size-selection
category: methodology
topic: "test-set selection (latent)"
title: "Latent: reported metric is the min over test_batch_size choices (test-set min)"
severity: low
confidence: medium
status: question
file: run.py
line_start: 463
line_end: 463
quote: |
    test_batch_size_list = [args.test_batch_size]
claim: "When more than one test batch size is supplied, run.py:573-575 keeps the lowest test MSE across them. In the shipped scripts test_batch_size_list ends up with a single element (line 463 plus the dedup append at 486-487), so the selection does not actually trigger; the mechanism is latent."
concern: "Were a second test batch size ever added, the reported number would be the per-batch-size minimum chosen on the test set, a mild test-set selection; with drop_last=False this can shift the metric slightly."
resolution: "Authors: confirm no reported number used more than one test_batch_size; if so, fix the metric to a single deterministic batch size."
cross_refs: ["itr-best-of-n-selected-on-test"]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 1 | medium | Plug-in baseline models (iTransformer/PatchTST/RLinear) for Table 5 absent. |
| bug | 1 | low | 3 basis-ablation scripts call an undefined model name (KeyError). |
| difference | 0 | - | Implemented model/eval match the paper where checkable. |
| methodology | 2 | medium | Driver reports best-of-N selected on test set (itr>1); latent test-batch-size min. |

## 5. Closing lists

**Top take-aways** (≤6, ranked):
1. [methodology] `itr-best-of-n-selected-on-test` — the run.py driver keeps the best of
   N runs/configs by **test** MSE+MAE; the `robust/` scripts use `--itr 7 --fix_seed 0`,
   and no script aggregates a mean/std. Verify which numbers in Tables 2/3/13 come from
   this path. (severity medium, confidence medium)
2. [missing] `orthotrans-plugin-baselines-missing` — Table 5's OrthoTrans-plug-in
   generality result has no baseline-model code or scripts in the repo. (medium/high)
3. [bug] `basis-script-undefined-model` — ECL/PEMS03/Solar basis-ablation scripts call
   `OLinear_wavelet_concat`, which is not in the model registry → KeyError. (low/high)
4. [methodology] `itr-test-batch-size-selection` — latent test-set min over test batch
   sizes; not active in shipped scripts. (low, question)

**Items that genuinely look fine** (actively checked):
- OrthoTrans transform is computed from the **training set only**: the committed
  `weather_96_ratio0.7.npy` reproduces exactly from the first-70% training portion and
  differs from the full series (`_audit_code/check_q_matrix_train_only.py`,
  `out/q_matrix_train_check.csv`). Same for the channel matrix
  (`check_channel_matrix_train_only.py`). No look-ahead leakage in the transform.
- StandardScaler is fit on training data only (`data_provider/data_loader.py:69-71,
  166-168, 308-311`); test windows reuse it. Standard, sound.
- Early stopping and checkpoint selection use **validation** loss, not test
  (`experiments/exp_forecast.py:542,547`; `utils/tools.py:69-86`). Test loss is only
  printed, never used to pick a checkpoint/epoch — no early-stop-on-test leakage.
- NormLin matches the paper: `F.normalize(F.softplus(weight_mat), p=1, dim=-1)`
  (`layers/Transformer_EncDec.py:136-138`), i.e. RowNormL1(Softplus(W)) as in Eq. 3.
- The headline OLinear model builds and runs (CPU smoke test, output shape correct).
- For the plain OLinear model the per-"stage" test-set argmin in
  `exp_forecast.py:852-858` is inert (single output → min_index=0); it only matters for
  tuple-returning variants.

**Open questions for the authors:**
- Are the main long/short-term tables (Tables 2/3) produced by the single-seed `itr=1`
  main scripts, and is Table 13's std the mean/SD over all 7 logged per-seed runs (not
  the driver's "global best")? (ties to `itr-best-of-n-selected-on-test`)
- Where are the iTransformer/PatchTST/RLinear implementations and plug-in run scripts
  for Table 5/6? (ties to `orthotrans-plugin-baselines-missing`)
