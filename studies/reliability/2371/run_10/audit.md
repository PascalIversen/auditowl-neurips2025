# Audit — Forging Time Series with Language (SDForger), paper #2371

## Summary

SDForger generates synthetic multivariate time series by (1) embedding windowed
signals into a low-dimensional tabular space with FastICA or functional PCA,
(2) encoding each embedding row as a Fill-In-The-Middle text prompt and
fine-tuning an autoregressive LLM (GPT-2 by default), (3) sampling new embedding
rows and decoding them back to the time-series space. The paper reports
similarity metrics (Table 1: MDD/ACD/SD/KD/ED/DTW/SHAP-RE) and downstream TTM
forecasting utility (Table 2: RMSE/MASE/WQL) across 12 datasets and against 5
generative baselines (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4).

I audited the NeurIPS supplemental reproduction package
`code/SDForger__neurips_supplemental/` (the full pipeline: generation +
similarity + TTM utility, with 16 dataset CSVs) and cross-referenced the
open-source `code/IBM__fms-dgt/` databuilder. I read every script under
`sources/` and `utils/`, the two configs, the conditional-generation notebook,
and the `data/` tree. I ran read-only deterministic checks under `_audit_code/`
(no repo file was modified, no network used): a reshape test for the shapelet
metric, and a static traceability scan for baseline code, a multi-dataset
driver, the Figure-2 kNN accuracy, and the TTM channel-averaging. I did not
execute the LLM/TTM pipeline (requires GPU + the external `tsfm_public`/granite
download), so utility numbers were checked for traceability and metric
correctness, not re-run.

The generation, embedding, decoding, filtering, similarity metrics, and the
TTM train/val/test split are present and look methodologically reasonable
(forward, non-overlapping consecutive train/val/test segments for the
multivariate setting; scaler fit on train only). The main gaps are
reproduction-completeness ones: none of the 5 baselines are implemented in the
repo, there is no driver to run the 12-dataset aggregation, and the Section-6
Figure-2 headline (kNN accuracy 0.81) has no computing code.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 SDF-ICA/FPC similarity (MDD/ACD/SD/KD/ED/DTW/SHR), 12 datasets | `utils/evaluation/TSG_evaluation.py` + `run_TSG_evaluation.py` | metric code present; only ecl (MS) / bikesharing (UV) configs provided | partial | Metric code present; per-dataset aggregation not driven |
| Table 1 / Table 2 baseline rows (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) | (none) | — | — | MISSING (no baseline code) |
| Table 2 TTM RMSE/MASE/WQL (bikesharing/etth1/traffic) | `utils/evaluation/utils_ttm.py:evaluate_ttm_model` | metric code present; averages over all forecast channels | unverified | Metric code present; channel scope unverified |
| Tables D.3/D.4/D.10–D.14 per-dataset similarity | `run_TSG_evaluation.py` | one row per run; no multi-dataset loop | partial | No driver to produce all rows |
| Fig. 2 / §6 conditional gen, kNN accuracy = 0.81 | `notebook/conditional_generation.ipynb` | generation + plotting only; no classifier | — | MISSING (no accuracy code) |
| Table D.1 variance retained (k=3/5/7, FICA/FPC) | `utils/augmentation/sdforger_augmentation.py:86-110,165-173` | `var_explained` computed per channel | plausible | Computed (single config per run) |
| Table D.2 filtering rejection stats | `utils/augmentation/sdforger.py:521-565` | discard/NaN/norm stats computed in-loop | plausible | Computed but not exported to a table |
| SHAP-RE for multivariate (>1 channel) | `utils/evaluation/shapelet_based_measures.py:20-21` | reshape crashes for C>1 | n/a | Latent bug; not exercised by reported (C=1) tables |

## missing

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "baselines / result reproduction"
title: "No baseline-model code in repo; Table 1/2 competitor numbers cannot be reproduced"
severity: high
confidence: high
status: finding
file: utils/evaluation/utils_evaluation.py
line_start: 67
line_end: 71
quote: |
  # Function to save results to CSV
  def save_results(df_results, csv_file_path, result_row):
      row_count = len(df_results) + 1
      df_results.loc[row_count] = result_row
      df_results.to_csv(csv_file_path, index=False)
claim: "The evaluation scripts only generate, evaluate, and save SDForger's own results; no implementation, wrapper, config, or driver exists for any of the five baselines (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) that Tables 1 and 2 compare against."
concern: "Every comparative claim in Tables 1-2 (and D.3/D.4/D.10-D.14) depends on baseline numbers that no code in the package produces, so the central 'outperforms existing generative models' conclusion is not reproducible from the artefact."
resolution: "Authors: provide the baseline-generation scripts/configs (or pinned forks) and a driver that produces every baseline row under the same split, scaling, and metric pipeline as SDForger."
cross_refs: ["no-multidataset-driver"]
check_script: _audit_code/check_traceability.py
paper_ref: "Table 1, Table 2; Section 4 Baselines"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-multidataset-driver
category: missing
topic: "result reproduction / experiment harness"
title: "No driver to run the 12-dataset aggregation behind Table 1 and the appendix tables"
severity: medium
confidence: high
status: finding
file: sources/run_TSG_evaluation.py
line_start: 71
line_end: 80
quote: |
  if __name__ == "__main__":

      # Set paths for output, model, and data input
      OUTPUT_PATH = os.path.join('output')
      os.makedirs(OUTPUT_PATH, exist_ok=True)

      # Set output results file
      # csv_file_path = os.path.join(str(OUTPUT_PATH), data + '_TSG_evaluation.csv')
      csv_file_path = os.path.join(str(OUTPUT_PATH), 'TSG_evaluation.csv')
      df_results = initialize_results_file_tsg(csv_file_path)
claim: "Each script run evaluates a single config (one dataset, one setting). Only the ecl (multisample) and bikesharing (univariate/multivariate) configs are shipped; there is no loop or batch script over the 12 datasets, settings, embedding dims, and seeds that the aggregated Table 1 and appendix Tables D.3/D.4/D.10-D.14 summarise."
concern: "The aggregated and per-dataset numbers cannot be regenerated without the user re-deriving 12+ dataset configurations and the averaging/normalisation/ranking logic, none of which is provided."
resolution: "Authors: provide the per-dataset configs (data_name, channels, lengths, k) and the aggregation/normalisation/ranking script that turns per-run CSVs into Table 1 and the appendix tables."
cross_refs: ["baselines-not-in-repo"]
check_script: _audit_code/check_traceability.py
paper_ref: "Table 1; Appendix Tables D.3, D.4, D.10-D.14"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fig2-knn-accuracy-missing
category: missing
topic: "result traceability / Section 6"
title: "Figure-2 conditional-generation accuracy (0.81) has no computing code"
severity: medium
confidence: high
status: finding
file: notebook/conditional_generation.ipynb
line_start: 1
line_end: 1
quote: |
  ### SDForger Augmentation
claim: "The conditional_generation.ipynb notebook performs text-conditioned generation and plotting (the only artefact for Section 6), but contains no longitudinal kNN classifier, no accuracy computation, and no scikit-fda usage; a static scan for KNeighbors/accuracy_score/skfda/0.81 across all notebooks and scripts returns nothing."
concern: "The Section-6 headline number 'accuracy of 0.81 in identifying the generated curves' cannot be reproduced because the classifier that produces it is absent from the package."
resolution: "Authors: add the longitudinal kNN classification script (training data, class labels, fit/predict, accuracy) that produces the 0.81 figure."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "Section 6, Figure 2 ('accuracy of 0.81')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: shapelet-reshape-multivariate-crash
category: bug
topic: "similarity metrics / SHAP-RE"
title: "SHAP-RE reshape assumes single channel; crashes for any C>1 input"
severity: low
confidence: high
status: finding
file: utils/evaluation/shapelet_based_measures.py
line_start: 20
line_end: 24
quote: |
  train_data  = orig_data.reshape(orig_data.shape[0], orig_data.shape[1])
  test_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1])
  train_y  = np.random.rand(orig_data.shape[0])
  test_y = np.random.rand(gen_data.shape[0])
  n_train, p = train_data.shape
claim: "calculate_shapelet_recons_err reshapes a (n_samples, length, n_channels) array to (n_samples, length), which is only size-consistent when n_channels==1; for C>1 numpy raises 'cannot reshape array of size ... into shape (n,length)'."
concern: "The SHAP-RE column is reported in Table 1 only for the single-channel similarity settings, so the reported numbers are unaffected, but the metric silently cannot be applied to multivariate data and would crash if used there."
resolution: "Authors: confirm SHAP-RE was computed per channel for single-channel data only, or generalise the reshape to handle C>1 (e.g. flatten or loop over channels)."
cross_refs: []
check_script: _audit_code/check_shapelet_reshape.py
paper_ref: "Table 1, SHR column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: config-k-vs-paper-univariate
category: difference
topic: "hyperparameters / configs"
title: "Shipped univariate config uses k=5 and L0=2000-but-1-sample, paper says k=3"
severity: low
confidence: high
status: finding
file: sources/config/config.yaml
line_start: 4
line_end: 15
quote: |
  data_train_channels: 
  - cnt
  data_train_params:
  - 2000
  - 1
  evaluation:
    generated_data_path: output/generated_data.npy
    train_data_path: output/train_data.npy
  save_results: true
  sdforger_augmentation_strategy: univariate
  sdforger_batch: 32
  sdforger_embedding_dim: 5
claim: "The default univariate example config sets sdforger_embedding_dim: 5, whereas the paper fixes k=3 for the multisample and univariate settings (Section 4 'Parameter settings'). The repo's README is consistent with the paper text (k=3), so the example config differs from the paper's reported setting."
concern: "A user running the shipped config out-of-the-box reproduces a k=5 run, not the k=3 setting that the paper's main univariate results use; the discrepancy is a config default, and the code path supports k=3 via the same field."
resolution: "Authors: set the example config defaults to the paper's reported k (3) for the main settings, or document that the shipped configs are illustrative rather than the exact Table-1 settings."
cross_refs: []
paper_ref: "Section 4 'Parameter settings' (k=3); README"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ttm-metric-averages-all-channels
category: methodology
topic: "utility evaluation / metric scope"
title: "TTM RMSE/MASE/WQL averaged over all forecast channels, not just the target"
severity: medium
confidence: low
status: question
file: utils/evaluation/utils_ttm.py
line_start: 335
line_end: 357
quote: |
  for sample in range(0, true_val.shape[0]):
        list_rmse_per_channel = []
        list_mase_per_channel = []
        list_wql_per_channel = []
        list_h1_per_channel = []
        for channel in range(0, pred_val.shape[2]):
            rmse = compute_rmse(true_val[sample, :, channel], pred_val[sample, :, channel])
            mase = compute_mase(np.array(true_val[sample, :, channel]), np.array(pred_val[sample, :, channel]))
            wql = compute_wql(np.array(true_val[sample, :, channel]), np.array(pred_val[sample, :, channel]))
            h1 = compute_h1_distance(np.array(true_val[sample, :, channel]), np.array(pred_val[sample, :, channel]))
            list_rmse_per_channel.append(rmse)
            list_mase_per_channel.append(mase)
            list_wql_per_channel.append(wql)
            list_h1_per_channel.append(h1)
        rmse_list.append(list_rmse_per_channel)
        mase_list.append(list_mase_per_channel)
        wql_list.append(list_wql_per_channel)
        h1_list.append(list_h1_per_channel)

    # Step 7: Calculate the average metrics across all windows
    avg_rmse = np.mean(rmse_list)
claim: "evaluate_ttm_model loops over every channel in pred_val (range(0, pred_val.shape[2])) and averages the per-channel RMSE/MASE/WQL uniformly, whereas Table 2 frames the task as forecasting a single target (e.g. bikesharing target: count, controls: temperature, humidity)."
concern: "If pred_val carries all input channels (target + controls), the reported per-dataset metric mixes the target's error with control-channel errors, so the Table-2 numbers would not be the pure target-forecast metric the caption implies; whether pred_val.shape[2]==1 (target-only) depends on tsfm internals I could not execute, hence a question."
resolution: "Authors: confirm whether pred_val contains only the target channel or all channels; if all channels, restrict the metric to prediction_channel_indices (the target) so it matches the Table-2 caption."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "Table 2 (target/control framing)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|--------------------------------|
| missing     | 3          | high         | No baseline code; no 12-dataset driver; no Fig-2 accuracy code |
| bug         | 1          | low          | SHAP-RE reshape crashes for C>1 (not hit by reported tables) |
| difference  | 1          | low          | Shipped univariate config k=5 vs paper k=3 |
| methodology | 1          | medium       | TTM metric may average target+control channels (question) |

## Top take-aways

- **baselines-not-in-repo** (`missing`, high): no code for any of the 5 baselines;
  Table 1/2 comparison numbers are not reproducible from the package.
- **no-multidataset-driver** (`missing`, medium): no harness produces the 12-dataset
  aggregation / per-dataset appendix tables; only 2-3 example configs shipped.
- **fig2-knn-accuracy-missing** (`missing`, medium): Section-6 headline accuracy 0.81
  has no computing code in the notebook or scripts.
- **ttm-metric-averages-all-channels** (`methodology`, medium / question): RMSE/MASE/WQL
  averaged over all forecast channels; target-only scope unverified.
- **shapelet-reshape-multivariate-crash** (`bug`, low): SHAP-RE reshape only valid for
  single-channel input.
- **config-k-vs-paper-univariate** (`difference`, low): shipped univariate config uses
  k=5, paper reports k=3.

## Items that genuinely look fine

- **Multivariate train/val/test split** (`utils_preprocess_data.py:339-393`): consecutive
  forward segments (train before val before test), windowed within each segment; the
  StandardScaler is fit on train only and applied to val/test — no scaler leakage and
  no temporal look-ahead across the split.
- **Dependency pinning**: `sdforgerpy310cuda.yaml` pins python=3.10.17, torch==2.4.1,
  transformers==4.46.3 and the full dependency set; the environment is rebuildable.
- **Similarity metric definitions** (`feature_based_measures.py`, `distance_based_measures.py`,
  `shapelet_based_measures.py`): MDD/ACD/SD/KD/ED/DTW/SHAP-RE are implemented and adopted
  from the cited Ang et al. (2023) / Zheng et al. (2016) benchmarks as the paper states.
- **Seeding** (`utils/generals.py`, `sdforger.set_seed`): random/numpy/torch/HF seeds set;
  FastICA uses random_state=0; generation reproducibility is supported.
- **In-generation IQR norm filtering** (`sdforger.py:521-565`): matches the Appendix-A.3
  q1-3*IQR <= N <= q3+3*IQR criterion across channels.

## Open questions for the authors

- Does `pred_val` in `evaluate_ttm_model` contain only the target channel (`cnt`/`HUFL`/
  `junction1`) or all input channels? This determines whether Table-2 metrics are
  target-only (ttm-metric-averages-all-channels).
- Where are the baseline-generation scripts and the aggregation/ranking code that turn
  per-run CSVs into Tables 1-2 and D.3/D.4/D.10-D.14?
- Was SHAP-RE computed per single channel for all reported datasets (so the
  multivariate reshape path was never exercised)?
