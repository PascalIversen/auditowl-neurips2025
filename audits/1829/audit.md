# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (paper 1829)

## 1. Summary

The repo (`jackyue1994__OLinear`, single commit `f168e01`) is the official implementation of OLinear, a
linear time-series forecaster that operates in an "orthogonally transformed domain". It is a fork of the
Time-Series-Library / iTransformer codebase: `run.py` parses ~200 args, `experiments/exp_forecast.py`
drives train/vali/test, `data_provider/data_loader.py` defines the standard TSLib temporal-split datasets,
and `model/OLinear.py` / `model/OLinear_C.py` implement the two headline models. OrthoTrans is realised by
pre-computed orthonormal matrices `Qi`/`Qo` loaded from `.npy` files (`model/OLinear.py:24-42`), generated
offline by `dataset/Generate_corrmat.ipynb`. NormLin is implemented as `LinearEncoder` in
`layers/Transformer_EncDec.py:102-138` (softplus + L1-normalised learnable weight matrix).

What I did:
- Read the model, data loader, experiment driver, `run.py`, metrics, NormLin layer, and the OLinear scripts.
- Verified the temporal split and scaling logic (train-only `StandardScaler.fit`) and that early stopping uses
  validation loss, not test loss.
- Confirmed the multi-stage "best stage" selection and multi-`test_batch_size` selection in `exp_forecast.test`
  / `run.py` are inactive for the main OLinear/OLinear-C models and the released scripts (single tensor output,
  single test batch size, `itr 1`, no `find_best`).
- Wrote `_audit_code/check_q_matrices.py`: confirmed the committed Weather Q-matrices are square with the
  expected dim (96/192/336/720) and orthonormal (max |QQ^T − I| ≤ 6.6e-7).
- Inspected `Generate_corrmat.ipynb`: the temporal and channel correlation matrices are computed from
  training-set rows only (no test leakage in the precomputed transform).
- Cross-checked the data/code-availability statement and Appendix B dataset list against the repo contents.

Overall: the implemented procedure is methodologically sound and faithful to the paper; the OrthoTrans
matrices are train-only and orthonormal as claimed, splits are temporal, and metric/early-stopping wiring is
clean. Findings are reproducibility-level (unpinned deps, Q-matrix regeneration friction, a latent device bug).

## 2. Traceability table

Headline metrics are MSE/MAE on standardized data, produced by `run.py` → `Exp_Forecast.test`
(`experiments/exp_forecast.py:560-892`, `metric()` in `utils/metrics.py:107`). The repo commits no
result logs, so values cannot be byte-compared without retraining; the *computation* is present and traced.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 / Table 17 long-term MSE & MAE (per dataset/horizon) | `experiments/exp_forecast.py:793,877` via `utils/metrics.py:18,14` | computed at runtime (not logged) | n/a (no committed output) | Traced (pipeline present; not re-run) |
| Abstract "SOTA on 24 benchmarks / 140 tasks" | scripts under `scripts/OLinear`, `scripts/OLinear_C` (one per dataset/horizon) | n/a | n/a | Traced (driver scripts present) |
| Table 1 NormLin FLOPs/memory vs MHSA | `experiments/exp_forecast.py:32-152` (`compute_model_stats`, fvcore) | computed at runtime | n/a | Traced |
| OrthoTrans = eigvecs of temporal Pearson corr matrix, train-only (§4.2) | `dataset/Generate_corrmat.ipynb` cell 0; loaded `model/OLinear.py:29,38` | Q square + orthonormal (err≤6.6e-7) | ✓ | Verified (`_audit_code/out/q_matrices.csv`) |
| App. I.10 "robust when Q computed from limited training data" (Table 36) | `Generate_corrmat.ipynb` `base_ratio` loop (0.1–1.0) | n/a | n/a | Traced (generation supports subsets) |
| App. I.9 extra metrics R2 / r / MASE | `utils/metrics.py:34-104` | computed at runtime | n/a | Traced |
| More metrics ± std (e.g. Weather:720 = 0.333±2e-3) | (none — multi-seed std) | — | — | MISSING (scripts use `itr 1`, single seed 2023; no std harness) |
| Var/temp, attn-variant, basis ablations | `scripts/ablation/*`, `model/OLinear_ablation_*`, `model/orthoLinear_basis/*` | computed at runtime | n/a | Traced (ablation models + scripts present) |
| Main long-term datasets (ECL/Traffic/Solar/PEMS/ETT/Exchange/METR-LA) | CSV + Q-matrices NOT in repo; README points to Drive/Tsinghua + notebook | — | — | MISSING-from-repo (documented external download) |

## 3. Findings

## missing

```yaml finding
id: requirements-unpinned
category: missing
topic: "dependencies / reproducibility"
title: "requirements.txt lists packages with no version pins"
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
claim: "The only dependency specification pins no versions (no torch / numpy / fvcore versions, no Python version)."
concern: "Numeric results from a forecasting model can shift across torch/numpy versions, and an unpinned environment cannot be faithfully rebuilt to reproduce the reported MSE/MAE."
resolution: "Authors: provide a pinned environment (exact torch/numpy/fvcore versions and Python version, e.g. a frozen requirements.txt or environment.yml)."
cross_refs: []
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: qmatrix-regen-notebook-hardcoded-paths
category: missing
topic: "result traceability / orthogonal transform inputs"
title: "Q-matrices for the main benchmarks are not committed; regeneration notebook has hardcoded paths"
severity: medium
confidence: high
status: finding
file: dataset/Generate_corrmat.ipynb
line_start: 1
line_end: 1
quote: |
  file_path = r'D:\SystemData\xunn\Desktop\codes\iTransformer-main\dataset\Solar\solar_AL.xlsx'
claim: "OLinear requires per-dataset/per-horizon orthonormal Q-matrices (`q_mat_file`, `q_out_mat_file`); for the main long-term benchmarks (ECL, Traffic, Solar, PEMS, ETT, Exchange, METR-LA) neither the CSVs nor the .npy Q-matrices are in the repo, and the only generator is a notebook that hardcodes a Windows absolute path and requires manual per-dataset edits (file_path, train_ratio 0.7 vs 0.6)."
concern: "Reproducing the headline tables for the main benchmarks requires downloading external data AND manually re-running an un-parameterised notebook with edited absolute paths, so the inputs to OrthoTrans are not turnkey-reproducible from the repo."
resolution: "Authors: ship the precomputed Q-matrices for all benchmarks (or a single CLI script that takes a dataset path and emits the matrices), and replace the hardcoded notebook paths with relative paths."
cross_refs: ["qmatrix-orthonormal-ok"]
paper_ref: "Section 4.2 (OrthoTrans); README 'Usage' step 2"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: multiseed-std-harness-absent
category: missing
topic: "statistical integrity / variance reporting"
title: "Reported ±std values have no multi-seed harness; released scripts run a single fixed seed"
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
claim: "Every released OLinear/OLinear-C script uses `--itr 1` and `--fix_seed 1`, so a single run with the fixed seed 2023 is performed; there is no loop over multiple seeds."
concern: "Appendix I.8 reports mean±std over runs (e.g. Weather:720 = 0.333±2e-3), but the repo provides no script that varies the seed to produce those standard deviations, so the reported variance is not reproducible from the released code."
resolution: "Authors: provide the multi-seed driver (or seed list) used to compute the reported ±std values."
cross_refs: []
tags: [reforms:7, forensics:hidden-iteration]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: qmat-hardcoded-cuda0
category: bug
topic: "device handling"
title: "OrthoTrans Q-matrix is hardcoded to cuda:0, ignoring --gpu / --devices"
severity: low
confidence: high
status: finding
file: model/OLinear.py
line_start: 28
line_end: 29
quote: |
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_mat = torch.from_numpy(np.load(q_mat_dir)).to(torch.float32).to(device)
claim: "The Q-matrices are placed on a hardcoded `cuda:0`, whereas the model itself is moved to `cuda:{args.gpu}` (experiments/exp_basic.py:37); under `--gpu N` (N>0) or multi-GPU the transform tensor and the model would live on different devices."
concern: "With a non-zero `--gpu` or `--use_multi_gpu`, the einsum between the model activations (`cuda:N`) and `self.Q_mat` (`cuda:0`) would raise a device-mismatch error; the released single-GPU scripts mask this by setting CUDA_VISIBLE_DEVICES to one device so both resolve to cuda:0."
resolution: "Use the experiment's device (`self.device` / `next(self.parameters()).device`) instead of a literal `cuda:0` when loading Q_mat / Q_out_mat."
cross_refs: []
tags: [lones:stage-4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: inactive-test-stage-and-batchsize-selection
category: difference
topic: "evaluation procedure (paper omission)"
title: "Test code contains best-stage and best-test-batch-size selection logic (inactive for main model)"
severity: low
confidence: high
status: finding
file: experiments/exp_forecast.py
line_start: 852
line_end: 858
quote: |
        mse_plus_mae = [a + b for a, b in zip(mse_list, mae_list)]

        # print min_mse
        # print(mse_list, mae_list)
        min_index = mse_plus_mae.index(min(mse_plus_mae))
        mse = mse_list[min_index]
        mae = mae_list[min_index]
claim: "test() selects the stage minimising (MSE+MAE) over intermediate-stage outputs, and run.py (run.py:568-576) selects the best test_batch_size by test MSE; both are selections made on the test set and not described in the paper."
concern: "If exercised these would be test-set selection (optimistic bias), but for the released OLinear / OLinear-C scripts they are inert: the models return a single tensor (model/OLinear.py:152, model/OLinear_C.py:162) so `mse_list` has one element (min_index=0), and each script passes a single `--test_batch_size`, so the batch-size list has one entry."
resolution: "Authors: confirm the headline numbers use only the final-stage output at a single test batch size (no per-test-set selection); consider removing or guarding the selection code to avoid accidental misuse for tuple-returning variants."
cross_refs: []
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. Items actively checked and found sound:
- **Splits / leakage**: all dataset classes use temporal (ordered) train/valid/test borders and fit
  `StandardScaler` on the training span only (`data_provider/data_loader.py:69-71, 268-311, 563-581`); no
  shuffled split, no scaler fit on full data.
- **OrthoTrans inputs**: `Generate_corrmat.ipynb` cell 0 builds the temporal correlation matrix from
  `A = data[train_length-...:train_length]` (train rows only); cell 1's channel matrix uses the first
  `train_ratio*base_ratio` fraction (train-only at base_ratio=1). Q-matrices verified orthonormal
  (`_audit_code/out/q_matrices.csv`). No future statistics leak into the transform.
- **Early stopping / model selection**: checkpoint selection uses validation loss
  (`experiments/exp_forecast.py:547`); `test_loss` is printed (line 543) but never feeds early stopping.
- **Metric fit**: MSE/MAE on standardized data is the standard long-term-forecasting protocol shared by all
  baselines; extra scale-free metrics (R2, r, MASE) are also computed.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 3          | medium       | Unpinned deps; Q-matrices/notebook not turnkey; no std harness |
| bug         | 1          | low          | Q-matrix hardcoded to cuda:0 (latent under multi-/non-0 GPU)   |
| difference  | 1          | low          | Inactive test-set selection logic (paper omission)            |
| methodology | 0          | -            | Splits, scaler, OrthoTrans inputs, early stopping all sound   |

## Top take-aways (≤6, ranked)

1. **[missing, medium]** `requirements.txt` is fully unpinned — the environment cannot be reliably rebuilt
   (`requirements-unpinned`).
2. **[missing, medium]** Main-benchmark Q-matrices/CSVs are absent and the only generator
   (`Generate_corrmat.ipynb`) has hardcoded Windows paths needing manual per-dataset edits
   (`qmatrix-regen-notebook-hardcoded-paths`).
3. **[difference, low]** Test code contains best-stage / best-test-batch-size selection, inactive for the
   main models/scripts but a latent optimistic-bias risk for tuple-returning variants
   (`inactive-test-stage-and-batchsize-selection`).
4. **[missing, low]** No multi-seed harness backs the reported ±std (`multiseed-std-harness-absent`).
5. **[bug, low]** Q-matrix hardcoded to `cuda:0` would mismatch the model device under `--gpu N`/multi-GPU
   (`qmat-hardcoded-cuda0`).

## Items that genuinely look fine

- Temporal splits with train-only `StandardScaler.fit` across all dataset loaders (no leakage).
- OrthoTrans Q-matrices are orthonormal and built from training rows only — faithful to §4.2 and App. I.10.
- Early stopping / checkpointing keyed on validation loss; test loss is logging-only.
- NormLin (`LinearEncoder`, softplus + L1 normalisation) matches the paper's description.
- Driver scripts exist for the main results and the var/temp, attention-variant, and basis ablations.
- All datasets are public with documented sources (Appendix B); the newer small datasets are committed.

## Open questions for the authors

- Do the headline Table 2/17 numbers use only the final-stage output at one test batch size (confirming the
  `min_index`/best-batch-size selection never influenced reported MSE/MAE)?
- What seed set / driver produced the Appendix I.8 mean±std values?
- Can the precomputed Q-matrices for all benchmarks be released (or a parameterised generator) so OrthoTrans
  is reproducible without manual notebook edits?
