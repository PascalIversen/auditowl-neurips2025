# Audit — SDForger: Forging Time Series with Language (NeurIPS 2025, paper 2371)

## 1. Summary

The workspace contains two author repositories. `code/IBM__fms-dgt/` is the
public open-source release of the SDForger **generation** databuilder only
(`fms_dgt/public/databuilders/time_series/`, authored by Rousseau, Boschi,
Salwala — matching the paper). `code/SDForger__neurips_supplemental/` is the
NeurIPS supplemental and is the load-bearing reproduction package: it contains
the generation pipeline (`utils/augmentation/`), the similarity metrics
(MDD/ACD/SD/KD/ED/DTW/SHAP-RE in `utils/evaluation/`), the TTM utility
evaluation (`run_TTM_evaluation.py`, `utils/evaluation/utils_ttm.py`), two
example configs, and 15 dataset CSVs. I confirmed the supplemental README
matches the paper, read every Python source file in the supplemental and the
generation databuilder, read the conditional-generation notebook, and read the
paper's Tables 1–2, the evaluation methodology (§4), and Appendices A–C.

I ran read-only checks under `_audit_code/` (`check_metric_facts.py`) to
confirm code-level facts behind the findings without executing the heavy
LLM-generation/TTM-finetuning pipeline (which needs an A100 / network model
downloads, both out of scope in this sealed sandbox). Output is in
`_audit_code/out/metric_facts.txt`.

Headline structural facts: the proposed-method generation and metric *functions*
are present and look sound, but (a) **none** of the five baseline generative
models whose numbers populate Tables 1–2 are implemented anywhere in either
repo, (b) there is **no** driver/aggregation script that produces Table 1's
"Normalized Average" or "Average Rank" columns and **no** per-dataset configs for
the 12 datasets, and (c) the Section-6 conditional-generation headline
("accuracy of 0.81") has no computing script. Several metric implementations
also deviate from their stated definitions (ED scaling, MASE/WQL).

## 2. Result-traceability table

| Paper artefact | Repo location (computes value?) | Matches paper | Status |
|---|---|---|---|
| Table 1 — SDForger MDD/ACD/SD/KD/ED/DTW/SHR (raw scores) | `utils/evaluation/{feature,distance,shapelet}_based_measures.py` via `run_TSG_evaluation.py` (functions present; example config only) | not run here | Present (functions), no per-dataset driver/configs |
| Table 1 — **baseline** rows (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) | (none — 0 files reference these models) | — | **MISSING** (F01) |
| Table 1 — "Norm. Avg." (normalized [0–1] per group) + "Avg. Rank" columns | (none — no normalization/rank script) | — | **MISSING** (F02) |
| Table 2 — SDForger RMSE/MASE/WQL | `utils/evaluation/utils_ttm.py` (`compute_rmse/mase/wql`) via `run_TTM_evaluation.py` | functions present | Present (functions), see F05/F06 on MASE/WQL definitions |
| Table 2 — **baseline** rows (TimeVAE…LS4, + "+OD") | (none) | — | **MISSING** (F01) |
| Table 2 — "Avg. Rank" column | (none) | — | **MISSING** (F02) |
| §6 / Fig. 2 — kNN classifier "accuracy of 0.81" | notebook produces the Fig. 2 plots only; no kNN/accuracy code anywhere | — | **MISSING** (F03) |
| Appendix Tables D.1–D.14 (per-dataset scores, ablations, timings, LLM comp.) | (no aggregation/ablation drivers; metric functions only) | — | **MISSING** (subset of F02) |
| ED definition (Appendix B.2: data in [0,1]) | `distance_based_measures.py:13-26` (StandardScaler upstream, not [0,1]) | code z-scores | MISMATCH → difference (F07) |
| ED/DTW pairing ("each original series and its generated") | `distance_based_measures.py:20,32` pairs `ori[i]` vs `gen[i]` by raw index | — | methodology (F04) |

## 3. Findings

## missing

```yaml finding
id: baselines-not-implemented
category: missing
topic: "baselines / result traceability"
title: "No baseline (TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4) code; Table 1 & 2 comparison rows unreproducible"
severity: high
confidence: high
status: finding
file: code/SDForger__neurips_supplemental/utils/evaluation/TSG_evaluation.py
line_start: 10
line_end: 49
quote: |
  def tsg_evaluation(original_dataset, generated_dataset):

      print(original_dataset.shape)

      result = {'MDD': None,
              'ACD': None,
              'SD': None,
              'KD': None,
              'ED': None,
              'DTW':None,
              'SHAP-RE':None}
claim: "The evaluation pipeline only ever consumes `original_dataset` and `generated_dataset` (SDForger's own output); no script in either repo implements, runs, or imports the five baseline generators (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) whose numbers fill the baseline rows of Tables 1 and 2 (grep over all .py files returns 0 references — see _audit_code/out/metric_facts.txt)."
concern: "The paper's headline claim that SDForger 'outperforms existing generative models' rests entirely on baseline numbers that no provided code can reproduce, so the comparison is unverifiable from the artefact."
resolution: "Authors: provide the baseline generation code (or the exact forks/commits and configs used) so the competitor rows of Tables 1–2 can be regenerated under the same split/metric/preprocessing."
cross_refs: ["aggregation-and-configs-missing"]
check_script: _audit_code/check_metric_facts.py
paper_ref: "Tables 1 and 2 (baseline rows); §4 'Baselines'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: aggregation-and-configs-missing
category: missing
topic: "result traceability / experiment protocol"
title: "No per-dataset configs and no normalization/rank aggregation script for Tables 1–2 and Appendix D"
severity: medium
confidence: high
status: finding
file: code/SDForger__neurips_supplemental/sources/run_TSG_evaluation.py
line_start: 149
line_end: 157
quote: |
      print(f'\nTSG Evaluation')
      result = tsg_evaluation( original_dataset, generated_dataset )
      
      # result_shivani = shivani_evaluation( original_dataset, generated_dataset, device, result_path_visualization = os.path.join(OUTPUT_PATH, str(target)), 
      #                         combined_name = generated_data_path.split("/")[-2] + suffix )
      
      print('\nSaving sdforger')
      save_results(df_results, csv_file_path, values_to_save + 
                  [result['MDD'], result['ACD'], result['SD'], result['KD'], result['ED'], result['DTW'], result['SHAP-RE']])
claim: "The TSG evaluation writes one row of raw metrics for a single (dataset,config) run; only two example configs exist (`config.yaml`, `config_ttm.yaml`) for bikesharing/ecl, and no script normalizes scores to [0,1] per metric group or computes the average-rank column that Table 1/Table 2 report."
concern: "The reported aggregated tables (12 datasets, normalized averages, ranks) and Appendix Tables D.1–D.14 cannot be reproduced because the per-dataset drivers and the aggregation/normalization/ranking step are absent."
resolution: "Authors: add the per-dataset config set (or a manifest) and the script that aggregates raw CSV rows into the normalized-average and average-rank columns of Tables 1–2 and Appendix D."
cross_refs: ["baselines-not-implemented"]
check_script: _audit_code/check_metric_facts.py
paper_ref: "Table 1 'Norm. Avg.'/'Avg. Rank'; Appendix D"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: section6-accuracy-no-script
category: missing
topic: "result traceability"
title: "Section 6 kNN-classifier accuracy 0.81 has no computing script"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2).
claim: "The conditional_generation.ipynb notebook generates the Figure-2 channel-conditioned curves but contains no classifier; grep for KNeighbors/accuracy_score/skfda/n_neighbors across the whole supplemental returns 0 code matches (the value 0.81 appears only inside a stored notebook output cell, not as a computation)."
concern: "The headline number quantifying the multimodal/text-conditioning experiment (accuracy = 0.81) is not produced by any provided code, so it cannot be verified."
resolution: "Authors: include the kNN training/evaluation script (data split, k, distance) that produces the 0.81 accuracy reported in §6."
cross_refs: []
check_script: _audit_code/check_metric_facts.py
paper_ref: "§6 'Shaping time series with language'; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: shapre-collapses-channel-axis
category: bug
topic: "evaluation metric / SHAP-RE"
title: "SHAP-RE reshape silently collapses the channel axis (valid only for univariate)"
severity: low
confidence: medium
status: question
file: code/SDForger__neurips_supplemental/utils/evaluation/shapelet_based_measures.py
line_start: 20
line_end: 23
quote: |
      train_data  = orig_data.reshape(orig_data.shape[0], orig_data.shape[1])
      test_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1])
      train_y  = np.random.rand(orig_data.shape[0])
      test_y = np.random.rand(gen_data.shape[0])
claim: "`calculate_shapelet_recons_err` receives data shaped (n_samples, length, n_channels) but reshapes to (n_samples, length); this only equals the intended 2-D matrix when n_channels==1, and would raise or mis-flatten for multi-channel input."
concern: "The shapelet reconstruction metric drops the channel dimension; it is correct for the univariate/multisample similarity tables (1 channel) but would silently misbehave if applied to multivariate series."
resolution: "Authors: confirm SHAP-RE was only computed on single-channel series, or add an explicit per-channel loop so the reshape is not relied upon."
cross_refs: []
check_script: _audit_code/check_metric_facts.py
paper_ref: "Appendix B.2 'Shapelet-based Reconstructions'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## difference

```yaml finding
id: ed-scaling-mismatch
category: difference
topic: "evaluation metric / Euclidean Distance"
title: "Appendix B.2 says ED uses [0,1]-scaled data; code z-scores with StandardScaler"
severity: low
confidence: high
status: finding
file: code/SDForger__neurips_supplemental/utils/augmentation/utils_preprocess_data.py
line_start: 234
line_end: 237
quote: |
          arr_train = np.array(preprocessed_train[var])
          scaler = StandardScaler()
          scaled_arr_train = scaler.fit_transform(arr_train)
          scaled_preprocessed_train.append(scaled_arr_train)
claim: "Preprocessing standardizes each channel with StandardScaler (zero-mean/unit-variance), but Appendix B.2 states for ED that 'the input time series has been preprocessed to fit within the range of [0, 1]'; no MinMax/[0,1] scaling exists in the repo (see _audit_code/out/metric_facts.txt)."
concern: "ED (and the [0,1] assumption it relies on for a deterministic value-wise comparison) is computed on z-scored rather than [0,1]-scaled data, so the metric's stated normalization does not match the implementation; the metric itself remains a valid distance."
resolution: "Authors: clarify whether ED in Tables 1/Appendix D was computed on z-scored data, and reconcile the Appendix B.2 [0,1] statement with the StandardScaler implementation."
cross_refs: ["ed-dtw-index-pairing"]
check_script: _audit_code/check_metric_facts.py
paper_ref: "Appendix B.2 'Euclidean Distance'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: mase-naive-denominator
category: difference
topic: "evaluation metric / MASE"
title: "MASE scaled by the forecast horizon's own first differences, not the in-sample naive error"
severity: low
confidence: medium
status: finding
file: code/SDForger__neurips_supplemental/utils/evaluation/utils_ttm.py
line_start: 247
line_end: 251
quote: |
  def compute_mase(true, pred):
      """Compute MASE (Mean Absolute Scaled Error)"""
      numerator = np.mean(np.abs(true - pred))
      denominator = np.mean(np.abs(true[1:] - true[:-1]))  # Naive one-step ahead forecast
      return numerator / denominator if denominator != 0 else np.nan
claim: "The MASE denominator is the mean absolute one-step difference of the held-out forecast window's ground truth, computed per forecast horizon; the standard MASE scales by the naive forecast error estimated on the in-sample/training series."
concern: "Using the test horizon's own differences as the naive scale changes the metric's meaning and comparability versus the textbook MASE, though it is applied uniformly across all methods."
resolution: "Authors: confirm this in-horizon naive scaling is intended and applied identically to every method/baseline in Table 2, and note the deviation from standard MASE."
cross_refs: ["wql-single-prediction"]
check_script: _audit_code/check_metric_facts.py
paper_ref: "Table 2 (MASE columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ed-dtw-index-pairing
category: methodology
topic: "evaluation metric / similarity"
title: "ED and DTW pair real sample i with generated sample i by raw array index (no matching)"
severity: medium
confidence: medium
status: finding
file: code/SDForger__neurips_supplemental/utils/evaluation/distance_based_measures.py
line_start: 17
line_end: 32
quote: |
  for i in range(n_samples):
      total_distance_eu = 0
      for j in range(n_series):
          distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
          total_distance_eu += distance
      distance_eu.append(total_distance_eu / n_series)
  ...
  def calculate_dtw(ori_data,comp_data):
      distance_dtw = []
      n_samples = ori_data.shape[0]
      for i in range(n_samples):
          distance = multi_dtw_distance(ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True)
claim: "ED and DTW compare the i-th original window directly to the i-th generated sample by position in the array, with no nearest-neighbour matching, sorting, or alignment between the two sets (grep for match/argmin/sort/linear_sum_assignment over the file returns none; see _audit_code/out/metric_facts.txt)."
concern: "Generated samples are drawn independently and are not in correspondence with specific real windows, so an index-based pairing makes the ED/DTW distances depend on arbitrary generation order rather than on genuine set-to-set similarity, potentially mis-ranking methods on the distance-based metrics."
resolution: "Authors: confirm whether ED/DTW use the same index-pairing for every method and whether this matches the TSGBench (Ang et al. 2023) protocol; if a matching step (e.g. nearest-neighbour / Hungarian) was intended, add it."
cross_refs: ["ed-scaling-mismatch"]
check_script: _audit_code/check_metric_facts.py
paper_ref: "Appendix B.2 'Euclidean Distance' / 'Dynamic Time Warping'; Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: wql-single-prediction
category: methodology
topic: "evaluation metric / WQL"
title: "WQL evaluates every quantile against one point prediction and is not normalized"
severity: medium
confidence: medium
status: finding
file: code/SDForger__neurips_supplemental/utils/evaluation/utils_ttm.py
line_start: 253
line_end: 259
quote: |
  def compute_wql(true, pred, quantiles=[0.1, 0.5, 0.9]):
      """Compute Weighted Quantile Loss (WQL)"""
      total_loss = 0
      for q in quantiles:
          errors = true - pred
          total_loss += np.mean(np.maximum(q * errors, (q - 1) * errors))
      return total_loss / len(quantiles)
claim: "WQL reuses the single point forecast `pred` as the quantile forecast for all of q in {0.1,0.5,0.9} and averages the pinball losses by number of quantiles; it never uses distinct quantile predictions and is not normalized by sum|true| as in the standard weighted quantile loss."
concern: "With one shared point prediction the pinball terms reduce to a symmetric scaled absolute error (effectively a rescaled MAE), so the reported WQL does not measure probabilistic-forecast calibration and is mislabeled relative to the standard WQL definition."
resolution: "Authors: clarify what WQL is intended to measure here given TTM outputs a point forecast; if a true quantile loss was intended, supply the quantile forecasts and the normalization."
cross_refs: ["mase-naive-denominator"]
check_script: _audit_code/check_metric_facts.py
paper_ref: "Table 2 (WQL columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 3          | high         | Baselines, aggregation/configs, and §6 accuracy have no code |
| bug         | 1          | low          | SHAP-RE channel-axis reshape (question; univariate only)    |
| difference  | 2          | low          | ED scaling and MASE naive-scale differ from stated defs     |
| methodology | 2          | medium       | ED/DTW index-pairing; WQL collapses to scaled MAE           |

## 5. Closing lists

**Top take-aways** (≤6, ranked by severity × confidence):
1. (missing) `baselines-not-implemented` — none of the 5 competitor generators are in the repo, so every baseline row in Tables 1–2 is unreproducible; the "outperforms baselines" claim cannot be checked. [high/high]
2. (missing) `aggregation-and-configs-missing` — no per-dataset configs and no normalization/rank aggregation, so the aggregated Tables 1–2 and Appendix D cannot be regenerated. [medium/high]
3. (missing) `section6-accuracy-no-script` — the §6 conditional-generation accuracy 0.81 has no computing code. [medium/high]
4. (methodology) `ed-dtw-index-pairing` — ED/DTW compare real and generated samples by arbitrary array index with no matching. [medium/medium]
5. (methodology) `wql-single-prediction` — WQL applies all quantiles to one point prediction and is unnormalized, reducing to a scaled MAE. [medium/medium]
6. (difference) `ed-scaling-mismatch` — ED's stated [0,1] preprocessing is actually z-scoring. [low/high]

**Items that genuinely look fine**:
- The SDForger generation pipeline (FPC/FastICA embedding, textual FIM encoding, fine-tuning, multinomial sampling, norm-IQR filtering, diversity stopping) is implemented and matches the §3/Appendix A description.
- MDD/ACD/SD/KD feature-based metrics in `feature_based_measures.py` match their Appendix B.1 definitions (histogram MDD, lagged ACF, standardized 3rd/4th moments).
- Dependencies are pinned (`sdforgerpy310{cuda,mps}.yaml`) and the README gives exact reproduce commands for the two provided example settings.
- Seeding is set across random/numpy/torch/transformers (`utils/generals.py`), and the SDForger class re-seeds before fit/generate.
- The cloned `IBM__fms-dgt` time_series databuilder README lists the paper's authors and matches the SDForger method (correct author repo, generation-only).

**Open questions for the authors**:
- Was the SHAP-RE metric ever computed on multivariate (multi-channel) series, given the reshape collapses the channel axis? (`shapre-collapses-channel-axis`)
- Do ED/DTW use index-pairing for every method, and does that follow TSGBench? (`ed-dtw-index-pairing`)
- Are the MASE in-horizon naive scaling and the point-prediction WQL the intended, uniformly-applied metric definitions for Table 2? (`mase-naive-denominator`, `wql-single-prediction`)
