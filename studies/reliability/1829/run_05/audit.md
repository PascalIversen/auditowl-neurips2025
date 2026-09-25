# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (#1829)

## 1. Summary

The repo `jackyue1994__OLinear` is the official implementation of OLinear, a
linear forecasting model that (a) transforms the series into a decorrelated
domain via an orthogonal matrix `Qi`/`Qo` derived from the eigendecomposition of
the temporal Pearson correlation matrix (OrthoTrans) and (b) learns
cross-/intra-series representations with a row-normalized linear layer
("NormLin"). The repo contains the model code (`model/OLinear.py`,
`model/OLinear_C.py`, ablation variants), data loaders
(`data_provider/data_loader.py`), the training/eval harness
(`run.py`, `experiments/exp_forecast.py`), the metric code (`utils/metrics.py`),
per-dataset shell scripts under `scripts/`, a notebook that generates the Q and
correlation matrices (`dataset/Generate_corrmat.ipynb`), and the precomputed
`.npy` Q-matrices plus CSVs for ~9 "extra" datasets.

What I did (read-only; no GPU so no training was run):
- Traced the data-splitting and scaling logic (chronological train/val/test;
  scaler fit on train only) and the OrthoTrans matrix generation (notebook) to
  confirm whether the transformation leaks test statistics. It does not — the Q
  and channel-correlation matrices are built from the training segment only.
- Verified the NormLin layer matches Eq. 3 (Softplus then row-wise L1 norm) and
  that OLinear-C uses a pure `Softmax(CorrMatv)` (no learnable additive term).
- Ran three deterministic `_audit_code/` checks: (1) grep the whole `*.py` tree
  for any statistical-test / std-over-seeds / confidence-interval code
  (`check_stats_and_seed_code.py`); (2) probe which headline datasets and
  precomputed Q-matrices referenced by `scripts/OLinear/*.sh` actually exist
  (`check_missing_data_qmat.py`); (3) check whether `requirements.txt` pins any
  versions (`check_requirements_pinned.py`).

Headline outcome: the model code is faithful to the paper and the core
transformation is leakage-free, but several *reproduction artefacts* are
missing — the standard-deviation / t-test / confidence-interval tables
(Tables 13/14/15/26) have no computing code, there is no 7-seed sweep mechanism
(a single hardcoded seed), dependencies are entirely unpinned, and the
precomputed Q-matrices for the main long-term benchmarks are not shipped.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 / 17 long-term MSE/MAE (OLinear) | `run.py` + `experiments/exp_forecast.py:560-892` + `utils/metrics.py:14-19` | recomputable (needs data+Q-mats) | n/a (cannot run) | Code present; data/Q-mats absent (see `missing-headline-data-and-qmats`) |
| Table 3 / 18-19 short-term MSE/MAE | same harness, `scripts/OLinear/*_S2.sh` | recomputable for shipped datasets | n/a | Code present |
| NormLin def. (Eq. 3): Softplus + row-wise L1 | `layers/Transformer_EncDec.py:136-138`, `layers/newLinear.py:27` | matches | ✓ | Verified |
| OLinear-C = Softmax(CorrMatv) | `model/OLinear_C.py:73-77` (`WeightTrans='none'`), `layers/Transformer_EncDec.py:418-420` | pure softmax, no learnable add | ✓ | Verified |
| OrthoTrans `Qi/Qo` from **train** only | `dataset/Generate_corrmat.ipynb` CELL 0/2 (`data[0:train_length]`) | train-only | ✓ | Verified (no leakage) |
| MASE (Eq. 22, lag-1 naive on target window) | `utils/metrics.py:71-104` | matches | ✓ | Verified |
| Table 13 std over 7 seeds (OLinear) | (none) | — | — | MISSING (`no-significance-test-code`) |
| Table 26 std over 7 seeds (OLinear-C) | (none) | — | — | MISSING (`no-significance-test-code`) |
| Table 14 99% confidence intervals | (none) | — | — | MISSING (`no-significance-test-code`) |
| Table 15 Student's t-test p-values | (none) | — | — | MISSING (`no-significance-test-code`) |
| "seven random seeds" robustness runs | `run.py:467-471` single seed=2023, no sweep | — | — | MISSING (`no-seed-sweep-code`) |
| Dependency spec for env rebuild | `requirements.txt` (0/13 pinned) | — | — | MISSING (`unpinned-dependencies`) |

## 3. Findings

## missing

```yaml finding
id: no-significance-test-code
category: missing
topic: "statistical integrity / result traceability"
title: "No code computes the std/t-test/CI tables (Tables 13, 14, 15, 26)"
severity: medium
confidence: high
status: finding
file: experiments/exp_forecast.py
line_start: 786
line_end: 793
quote: |
        mse_list = []
        mae_list = []

        if self.imp_mode:
            mae, mse, rmse, mape, mspe = metric(preds_array[mask_mat], trues_array[mask_mat])
            f = open("result_imputation.txt", 'a')
        else:
            mae, mse, rmse, mape, mspe, r2, pear, mase = metric(preds_array, trues_array)
claim: "The evaluation path computes a single per-run MSE/MAE/etc. via metric(); no script in the repo aggregates multiple runs into standard deviations, performs a Student's t-test, or computes confidence intervals."
concern: "Tables 13 (std over 7 seeds), 14 (99% CIs), 15 (Student's t-test p-values such as 5.12E-11), and 26 (OLinear-C std) are statistical claims with no code that produces them, so these robustness/significance results are not reproducible from the repo."
resolution: "Provide the script that runs the 7 seeds, collects per-seed metrics, and computes the standard deviations, 99% CIs, and Student's t-test p-values reported in Tables 13/14/15/26."
cross_refs: ["no-seed-sweep-code"]
check_script: _audit_code/check_stats_and_seed_code.py
paper_ref: "Tables 13, 14, 15, 26; Appendix F/I.7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-seed-sweep-code
category: missing
topic: "reproducibility / seeding"
title: "'Seven random seeds' claimed, but code uses one hardcoded seed with no sweep"
severity: medium
confidence: high
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
claim: "When fix_seed is set (all released scripts use --fix_seed 1), the seed is the constant 2023; there is no mechanism that iterates over seven distinct seeds, and --itr defaults to 1 (the only itr-loop varies a loss hyperparameter, not the seed)."
concern: "The paper repeatedly reports robustness 'over seven random seeds' (Tables 13/14/15/26, §F), but the provided code only ever runs a single fixed seed, so the seven-seed runs that underpin those tables cannot be reproduced."
resolution: "Add a seed argument or a seed-list loop (and document the seven seed values used), so reviewers can reproduce the per-seed metrics behind Tables 13/14/15/26."
cross_refs: ["no-significance-test-code"]
check_script: _audit_code/check_stats_and_seed_code.py
paper_ref: "Tables 13, 14, 15, 26; Appendix F"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-headline-data-and-qmats
category: missing
topic: "expected code completeness / data availability"
title: "Headline long-term datasets and their precomputed Q-matrices are absent"
severity: medium
confidence: high
status: finding
file: model/OLinear.py
line_start: 24
line_end: 32
quote: |
        q_mat_dir = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        if not os.path.isfile(q_mat_dir):
            q_mat_dir = os.path.join(configs.root_path, q_mat_dir)
        assert os.path.isfile(q_mat_dir)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_mat = torch.from_numpy(np.load(q_mat_dir)).to(torch.float32).to(device)

        assert (self.Q_mat.ndim == 3 if self.Q_chan_indep else self.Q_mat.ndim == 2)
        assert (self.Q_mat.shape[0] == self.enc_in if self.Q_chan_indep else self.Q_mat.shape[0] == self.seq_len)
claim: "The model requires a precomputed Q-matrix .npy on disk (assert os.path.isfile). For 14 of 23 OLinear scripts — including the core benchmarks ECL/electricity, ETT(h1/h2/m1/m2), Traffic, Solar, PEMS03/04/07/08, Exchange, METR-LA — neither the data CSV/NPZ nor the referenced Q-matrix file (e.g. electricity_96_ratio0.7.npy) is present in the repo."
concern: "The headline tables (Table 2/17 etc.) cannot be reproduced out-of-the-box: the scripts crash at model construction because the Q-matrix files are missing, and the underlying datasets are not shipped."
resolution: "Ship the precomputed Q-matrices for the main benchmarks (or a one-command script that regenerates them from the downloaded data), since the data CSVs are linked in the README but the Q-matrices are not provided for these datasets."
cross_refs: []
check_script: _audit_code/check_missing_data_qmat.py
paper_ref: "Table 2 / Table 17 (long-term forecasting)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unpinned-dependencies
category: missing
topic: "dependencies / environment"
title: "requirements.txt pins no versions and no Python version is given"
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
claim: "All 13 dependencies are listed as bare package names with no version constraints, and no Python/CUDA/torch version is specified anywhere."
concern: "The numerical results depend on the torch/numpy versions (e.g. eigendecomposition sign/order conventions in eigh, default dtypes); an unpinned environment cannot be reliably rebuilt and may not reproduce the reported metrics."
resolution: "Pin exact versions (torch, numpy, scikit-learn, pandas, etc.) and state the Python version, e.g. via a frozen environment file."
cross_refs: []
check_script: _audit_code/check_requirements_pinned.py
paper_ref: "Appendix D (Implementation details)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No formal YAML finding is filed here. The Q-matrix generation notebook
`dataset/Generate_corrmat.ipynb` hardcodes an author-machine absolute Windows
path in the temporal-correlation cell
(`file_path = r'D:\SystemData\xunn\Desktop\codes\iTransformer-main\dataset\Solar\solar_AL.xlsx'`)
and uses per-dataset relative paths that must be edited by hand in the other
cells, and there is no driver that regenerates all Q-matrices. This is a
low-severity reproducibility friction that compounds
`missing-headline-data-and-qmats` (the precomputed Q-matrices for the main
benchmarks are not shipped, so a reviewer must re-run this notebook). It is left
as prose rather than a structured code finding because the evidence lives in a
`.ipynb` whose only stable anchor is JSON-escaped notebook source, which cannot
be cited as a reliable verbatim code quote.

## difference

(none)

## methodology

```yaml finding
id: test-set-selection-machinery
category: methodology
topic: "hyperparameter tuning / test-set leakage"
title: "Driver/test code can select best result by test-set MSE+MAE (dormant in released scripts)"
severity: low
confidence: medium
status: question
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
claim: "The driver evaluates on the test set (exp.test, flag='test') and keeps the configuration with the lowest test mse+mae across test_batch_size values and across a lamda1 hyperparameter sweep (lines 530-580); exp_forecast.py:852-858 similarly picks the min-(mse+mae) 'stage' on the test set."
concern: "If activated, selecting the reported metric by minimising test-set mse+mae over batch sizes / lamda1 / stages is test-set tuning; the reported number would be the best over several test-set evaluations rather than a held-out estimate."
resolution: "Confirm that the released numbers used a single fixed lamda1 (1.0), a single test_batch_size, and the single-output OLinear/OLinear-C model (so this selection is a no-op); if any sweep was used for headline numbers, re-select configurations on validation, not test."
cross_refs: []
paper_ref: "Appendix D; Tables 2/3/17"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

Note on the question above: in the released `scripts/OLinear/*.sh` and
`scripts/OLinear_C/*.sh`, `--find_best` is never passed (default `False`), each
script sets a single `--test_batch_size`, and both `model/OLinear.py:152` and
`model/OLinear_C.py:162` return a single tensor (not a multi-stage tuple), so
`mse_list`/`mae_list` hold one element and the `min`-selection at
`exp_forecast.py:852-858` is a no-op. The selection machinery is therefore
**dormant for the headline configs**; it is filed as a `question` because the
mechanism exists and would constitute test-set selection if enabled.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 4          | medium       | std/t-test/CI tables, 7-seed sweep, headline data/Q-mats, unpinned deps |
| bug         | 0          | -            | notebook hardcoded paths noted in prose (low, not structured) |
| difference  | 0          | -            | model code is faithful to the paper                        |
| methodology | 1          | low          | test-set-selection machinery present but dormant (question)|

## 5. Closing lists

**Top take-aways** (≤ 6, ranked by severity × confidence):
1. `no-significance-test-code` (missing, medium/high): Tables 13/14/15/26
   (std, 99% CI, Student's t-test p-values) have no computing code.
2. `no-seed-sweep-code` (missing, medium/high): "seven random seeds" claimed,
   but the code uses a single hardcoded seed (2023) with no sweep.
3. `missing-headline-data-and-qmats` (missing, medium/high): 14/23 OLinear
   scripts reference datasets and precomputed Q-matrices that are absent, so the
   main long-term tables are not runnable out-of-the-box.
4. `unpinned-dependencies` (missing, low/high): no version pins, no Python
   version — environment cannot be reliably rebuilt.
5. `test-set-selection-machinery` (methodology question, low/medium): driver/test
   code can select the best result by test-set MSE+MAE, but is dormant in the
   released configs.
6. (prose, bug, low) Q-matrix generation notebook hardcodes author-machine
   paths and has no all-dataset driver — compounds the missing Q-matrices.

**Items that genuinely look fine** (actively checked):
- OrthoTrans is leakage-free: temporal Q (`Generate_corrmat.ipynb` CELL 0/2)
  uses `data[0:train_length]` and channel-corr (CELL 1) uses the training prefix
  only; the StandardScaler in every loader is fit on the train segment only
  (`data_provider/data_loader.py:68-71, 165-168, 300-311, 558-581`).
- Train/val/test splits are chronological (e.g. 12/4/4 months for ETT,
  0.7/0.1/0.2 or 0.6/0.2/0.2 elsewhere) — appropriate for time series.
- NormLin matches Eq. 3 (Softplus then row-wise L1 norm):
  `layers/Transformer_EncDec.py:136-138`, `layers/newLinear.py:27`.
- OLinear-C uses a pure `Softmax(CorrMatv)` with no learnable additive term
  (`WeightTrans='none'`): `model/OLinear_C.py:73-77`.
- MSE/MAE/MASE definitions in `utils/metrics.py` match the paper (Eq. 22);
  metrics are computed on the standardized scale (`--inverse 0`), consistent
  with the iTransformer/TSLib baselines being compared against.

**Open questions for the authors:**
- Where is the code that runs the seven seeds and produces the standard
  deviations, 99% confidence intervals, and Student's t-test p-values in
  Tables 13/14/15/26?
- Were the headline numbers produced with a single fixed `lamda1`, a single
  `test_batch_size`, and the single-output model (so the test-set-selection loop
  in `run.py:568-580` / `exp_forecast.py:852-858` was a no-op)?
