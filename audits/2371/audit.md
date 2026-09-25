# Audit — SDForger: Forging Time Series with Language (paper 2371)

## 1. Summary

The repository under `code/SDForger__neurips_supplemental/` is the NeurIPS supplemental
reproduction package for SDForger. It contains the SDForger *generation* pipeline
(periodicity-aware segmentation → FPC/FastICA embedding → GPT-2 fine-tuning on
fill-in-the-middle text prompts → decoding) and the *similarity* and *utility* evaluation
code (MDD/ACD/SD/KD feature metrics, ED/DTW distance metrics, SHAP-RE shapelet metric, and
a TTM-forecasting utility harness). The three entry points are
`sources/run_data_augmentation.py`, `sources/run_TSG_evaluation.py`, and
`sources/run_TTM_evaluation.py`, each driven by a single YAML config
(`sources/config/config.yaml`, `sources/config/config_ttm.yaml`). All 12 benchmark
datasets from Appendix C are bundled under `data/`. A second clone,
`code/IBM__fms-dgt/`, is the general fms-dgt framework that hosts the public databuilder;
I focused the audit on the supplemental package, which is the package the paper's repro
checklist points at and which produces the paper's numbers.

What I did: I read every script in the supplemental package and mapped each reported
quantity to the code that would compute it. I wrote two read-only checks under
`_audit_code/`: (a) `check_ed_dtw_pairing.py` reproduces `calculate_ed` verbatim and shows
the distance metric pairs original-window *i* with generated-sample *i* by row index and
silently truncates to the smaller of the two sample counts; (b) `check_acf_lag.py`
confirms the lagged-product slicing in `acf_torch` runs without an off-by-one. I also
ran a numpy reshape check confirming the SHAP-RE metric crashes on multivariate input.
I could not execute the full pipeline (it needs `torch`/`transformers`/`tsfm_public`/
`dtaidistance` and a GPT-2 download), so the run-level findings below are static plus the
targeted unit checks.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 raw similarity scores for **SDF-ICA/SDF-FPC** (MDD/ACD/SD/KD/ED/DTW/SHR) | `utils/evaluation/TSG_evaluation.py:10-53` via `run_TSG_evaluation.py` | per-run row | n/a (not run) | Present (metric code) |
| Table 1 raw similarity scores for **baselines** (TimeVAE, TimeVQVAE, RtsGAN, SdeGAN, LS4) | (none — no baseline generation code) | — | — | MISSING |
| Table 1 **Norm. Avg.** (Feat./Dist.) and **Rank** columns | (none — no aggregation/normalization/ranking script) | — | — | MISSING |
| Tables D.10–D.14 per-dataset similarity (12 datasets) | metric code present; multi-dataset driver (none) | — | — | Partial (no driver) |
| Table 2 TTM RMSE/MASE/WQL for bikesharing/etth1/traffic | `utils/evaluation/utils_ttm.py:301-379` via `run_TTM_evaluation.py` | per-run row | n/a (not run) | Present (only bikesharing config shipped) |
| Table D.6 / D.7 (k ablation), D.9 (LLM backbone) | same metric code, different config | — | — | Present (config-driven) |
| Table D.5 / D.8 generation time | (no timing harness in repo) | — | — | MISSING (timing not measured by shipped code) |
| Table D.2 filtering / rejection rates | `utils/augmentation/sdforger.py:521-565` (filtering); no rate-reporting script | — | — | Partial (filter present, no rate dump) |
| Fig. 2 / §6 "accuracy of 0.81" (longitudinal kNN identifying generated curves) | (none — notebook plots only, no classifier) | — | — | MISSING |

## 3. Findings

## missing

```yaml finding
id: no-aggregation-or-baseline-code
category: missing
topic: "result traceability"
title: "No code produces Table 1 normalized averages, ranks, or any baseline numbers"
severity: high
confidence: high
status: finding
file: utils/evaluation/utils_evaluation.py
line_start: 31
line_end: 46
quote: |
  def initialize_results_file_tsg(csv_file_path):

      if os.path.exists(csv_file_path):
          df_results = pd.read_csv(csv_file_path)
      else:
          df_results = pd.DataFrame(columns=[
              'key',
              'seed', 'data', 'target', 'augmentation', 
              'n points train', 'n samples train', 'min train windows',
              'train windows', 'windows length', 'overlap', 'period', 
              'sdforge llm', 'learning rate', 'train splitting',
              'permute', 'init value', 'embedding type', 
              'embedding dim', 'generated samples', 'var requested', 
              'var explained', 'MDD', 'ACD', 'SD', 'KD', 'ED', 'DTW', 'SHAP-RE'
          ])
      return df_results
claim: "The TSG evaluation writes one CSV row of raw metrics per (dataset, model) invocation; the repo contains no script that produces the 'Norm. Avg.' (Feat./Dist.) or 'Rank' columns of Table 1, and no code that generates the five baseline competitors (TimeVAE, TimeVQVAE, RtsGAN, SdeGAN, LS4) whose rows fill most of Tables 1 and 2."
concern: "The headline comparison ('SDForger outperforms existing generative models … average rank') depends on normalization and ranking across baselines, none of which can be reproduced from the shipped code — the NeurIPS checklist item 5 explicitly asks for scripts to reproduce results 'for the new proposed method and baselines'."
resolution: "Provide the aggregation/normalization/ranking script that turns per-run metric CSVs into Table 1's Norm. Avg. and Rank columns, and the baseline-generation code (or the baseline-generated .npy outputs) so each Table 1/2 row is reproducible."
cross_refs: ["§5.1", "Table 1", "Table 2"]
paper_ref: "Table 1 (Norm. Avg., Rank columns); Table 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fig2-knn-accuracy-missing
category: missing
topic: "result traceability"
title: "No code computes the 0.81 kNN accuracy reported for text-conditioned generation (Fig. 2)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2).
claim: "Section 6 reports a quantitative result (accuracy 0.81 from a longitudinal kNN classifier) for the text-conditioned generation experiment, but the only artefact for that experiment is notebook/conditional_generation.ipynb, which standardizes, augments, and plots the three channels and contains no classifier, no 'accuracy', and no kNN call (grep over its 14 code cells returns 0 hits for accuracy/KNeighbors/knn/classif/score)."
concern: "A reported numeric claim has no computing artefact in the repo, so the 0.81 cannot be reproduced or verified."
resolution: "Authors: please add the script/cell that builds the longitudinal kNN classifier, trains it on real data, and computes the 0.81 identification accuracy from the conditional generations."
cross_refs: ["§6", "Figure 2"]
paper_ref: "Section 6, 'Shaping time series with language'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: ed-dtw-silent-truncation
category: bug
topic: "evaluation / distance metrics"
title: "ED and DTW loop over n_original, silently discarding surplus generated samples"
severity: medium
confidence: high
status: finding
file: utils/evaluation/distance_based_measures.py
line_start: 13
line_end: 22
quote: |
  def calculate_ed(ori_data,gen_data):
      n_samples = ori_data.shape[0]
      n_series = ori_data.shape[2]
      distance_eu = []
      for i in range(n_samples):
          total_distance_eu = 0
          for j in range(n_series):
              distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
              total_distance_eu += distance
          distance_eu.append(total_distance_eu / n_series)
claim: "Both calculate_ed (line 17) and calculate_dtw (line 30-31) iterate range(ori_data.shape[0]) and index gen_data[i], so only the first n_original generated samples are ever compared."
concern: "Under the paper's protocol the model generates ~100 samples while there are ~30 training windows (config minimum_windows_number=30, min/max_generations=100), so ED/DTW evaluate only the first 30 of 100 generated curves and ignore 70% of the synthetic set; check_ed_dtw_pairing.py confirms calculate_ed(orig, gen[:30]) == calculate_ed(orig, gen) exactly when 100 generated rows are supplied."
resolution: "Authors: confirm how many generated curves enter the reported ED/DTW; if generation count exceeds the original count the surplus is dropped — clarify whether the .npy fed to evaluation was pre-truncated to the window count, or fix the loop to aggregate over all generated samples."
cross_refs: ["ed-dtw-index-pairing"]
check_script: _audit_code/check_ed_dtw_pairing.py
paper_ref: "Table 1, ED/DTW columns; Appendix B.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: shapelet-multivariate-reshape-crash
category: bug
topic: "evaluation / shapelet metric"
title: "SHAP-RE reshape collapses only when channels==1; crashes for multivariate input"
severity: low
confidence: high
status: finding
file: utils/evaluation/shapelet_based_measures.py
line_start: 20
line_end: 21
quote: |
      train_data  = orig_data.reshape(orig_data.shape[0], orig_data.shape[1])
      test_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1])
claim: "calculate_shapelet_recons_err reshapes (n_samples, length, channels) to (n_samples, length), which is only valid when channels==1; for channels>1 numpy raises ValueError: cannot reshape array of size N into shape (n,length) (verified directly with a (30,250,3) array)."
concern: "The SHAP-RE (SHR) metric cannot be computed on any multi-channel series; it is usable only because the Table 1 settings that report SHR (multisample ecl, univariate) are single-channel, so the metric is latently broken for the multivariate/multichannel case."
resolution: "Authors: confirm SHR was only ever computed on single-channel data; if a multivariate SHR was reported anywhere, the reshape would have failed — reshape should flatten or loop over channels."
cross_refs: []
paper_ref: "Table 1, SHR column; Appendix B.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: ed-preprocessing-standardscaler-not-0-1
category: difference
topic: "preprocessing / evaluation"
title: "Paper says inputs scaled to [0,1] for ED; code uses StandardScaler (zero-mean/unit-var)"
severity: low
confidence: high
status: finding
file: utils/augmentation/utils_preprocess_data.py
line_start: 234
line_end: 237
quote: |
        arr_train = np.array(preprocessed_train[var])
        scaler = StandardScaler()
        scaled_arr_train = scaler.fit_transform(arr_train)
        scaled_preprocessed_train.append(scaled_arr_train)
claim: "The only scaler applied to the series before embedding and evaluation is sklearn StandardScaler (mean 0, unit variance per column), not min-max [0,1] scaling — grep over the package finds no MinMaxScaler."
concern: "Appendix B.2 states 'the input time series has been preprocessed to fit within the range of [0, 1]' as the basis for ED being a deterministic similarity; the actual preprocessing is standardization, which does not bound values to [0,1], so the paper's stated ED normalization does not match the code (the standardization is itself valid, only the description differs)."
resolution: "Authors: correct the Appendix B.2 description to standardization, or apply [0,1] scaling if that is what produced the reported ED magnitudes."
cross_refs: []
paper_ref: "Appendix B.2, Euclidean Distance"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ed-dtw-index-pairing
category: methodology
topic: "evaluation / distance metrics"
title: "ED/DTW pair original window i with generated sample i by index, not by similarity"
severity: medium
confidence: medium
status: finding
file: utils/evaluation/distance_based_measures.py
line_start: 17
line_end: 22
quote: |
      for i in range(n_samples):
          total_distance_eu = 0
          for j in range(n_series):
              distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
              total_distance_eu += distance
          distance_eu.append(total_distance_eu / n_series)
claim: "ED (and DTW, line 30-32) compute a per-index distance between original sample i and generated sample i; check_ed_dtw_pairing.py shows that feeding the SAME SET of curves in a shuffled row order changes ED from 0 to 13.67, i.e. the metric depends entirely on arbitrary row alignment."
concern: "SDForger generation is unconditional, so generated row i has no correspondence to original window i; pairing them by index makes ED/DTW a distance between unrelated curves rather than a distribution-level similarity, and the value can be made arbitrarily large or small by reordering — weakening the ED/DTW evidence for 'high-quality' synthesis (note: this index-paired ED/DTW is the definition the paper itself gives in Appendix B.2, inherited from Ang et al. 2023 / TSGBench, and is applied identically to all baselines, so it does not by itself overturn the relative ranking — hence medium/medium)."
resolution: "Authors: clarify whether ED/DTW use an optimal matching/nearest-neighbour assignment between real and generated sets; if it is raw index pairing, justify why a per-index distance over unconditionally generated samples is meaningful."
cross_refs: ["ed-dtw-silent-truncation"]
check_script: _audit_code/check_ed_dtw_pairing.py
paper_ref: "Appendix B.2; Table 1 ED/DTW"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ttm-mase-denominator-from-test-window
category: methodology
topic: "evaluation / utility metric"
title: "MASE scaled by the forecast window's own naive error, not the training-set naive error"
severity: low
confidence: medium
status: question
file: utils/evaluation/utils_ttm.py
line_start: 247
line_end: 251
quote: |
  def compute_mase(true, pred):
      """Compute MASE (Mean Absolute Scaled Error)"""
      numerator = np.mean(np.abs(true - pred))
      denominator = np.mean(np.abs(true[1:] - true[:-1]))  # Naive one-step ahead forecast
      return numerator / denominator if denominator != 0 else np.nan
claim: "MASE scales the forecast error by the mean absolute one-step difference of the *future/test* window's own ground-truth values (true[1:]-true[:-1]), not by the in-sample naive error of the training series as in the standard MASE definition."
concern: "Using the test window's own volatility as the scaling denominator makes MASE a per-window-normalized error rather than the conventional training-anchored MASE; it is applied identically to all methods and to zero-shot, so cross-method ranking is preserved, but the absolute MASE values are not comparable to standard MASE in the literature."
resolution: "Authors: confirm the intended MASE definition; if the standard seasonal/in-sample naive denominator was intended, the metric should use the training series, not the forecast window."
cross_refs: []
paper_ref: "Table 2, MASE columns"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 2          | high         | No baseline-generation / aggregation-ranking code; Fig. 2 0.81 accuracy has no script. |
| bug         | 2          | medium       | ED/DTW silently use only the first n_original generated samples; SHR reshape crashes on multivariate. |
| difference  | 1          | low          | Paper says [0,1] scaling for ED; code uses StandardScaler. |
| methodology | 2          | medium       | ED/DTW pair samples by arbitrary row index; non-standard MASE denominator (question). |

## 5. Closing lists

**Top take-aways** (≤6, severity × confidence):
1. `no-aggregation-or-baseline-code` (missing, high/high) — the cross-method comparison that supports the headline claim (normalized averages, ranks, and all five baselines in Tables 1–2) is not reproducible from the shipped code.
2. `ed-dtw-silent-truncation` (bug, medium/high) — ED/DTW evaluate only the first n_original of the ~100 generated curves, discarding most of the synthetic set.
3. `ed-dtw-index-pairing` (methodology, medium/medium) — ED/DTW measure distance between index-aligned but uncorrelated real/generated curves (matches the paper's own B.2 definition, applied equally to baselines).
4. `fig2-knn-accuracy-missing` (missing, medium/high) — the 0.81 identification accuracy in §6 has no computing artefact.
5. `shapelet-multivariate-reshape-crash` (bug, low/high) — SHR is latently broken for >1 channel; only works because reported SHR is single-channel.
6. `ed-preprocessing-standardscaler-not-0-1` (difference, low/high) — preprocessing is standardization, not the [0,1] scaling the ED definition claims.

**Items that genuinely look fine**:
- FastICA/FPC encode→decode are consistent inverses: FPC reconstruction (`sdforger_augmentation.py:280-282`) multiplies by train std and adds train mean, the exact inverse of the standardize-then-project embedding (lines 83,106); FastICA reconstruction (line 298) is the correct `emb @ mixing.T + mean` inverse of `fit_transform`.
- `acf_torch` lagged-product slicing (`feature_based_measures.py:104`) is length-aligned (x[:, i:] and x[:, :-i] are both T−i); `check_acf_lag.py` runs it without an off-by-one.
- All 12 benchmark datasets from Appendix C are bundled under `data/` (appliances, bikesharing, carboncaptureplant, ecl, etth1, traffic, and the monash_* set), so the data-availability claim resolves.
- The norm-based IQR filtering in `sdforger.py:531-554` implements the q1−3·IQR ≤ N ≤ q3+3·IQR rule described in Appendix A.3 verbatim.
- Seeds are set globally (`generals.py:11-17`) and within SDForger (`sdforger.py:834-840`); no test-set leakage into TTM model selection was found (early stopping is on a separate `val_data` split, not the test set — `run_TTM_evaluation.py:113,221`).

**Open questions for the authors**:
- Were the per-run metric CSVs aggregated by an external (un-shipped) notebook into the Norm. Avg. / Rank columns, and were the baseline outputs produced off-repo? (`no-aggregation-or-baseline-code`.)
- How many generated curves actually entered the reported ED/DTW, given the loop truncates to the original window count? (`ed-dtw-silent-truncation`.)
- Is the MASE denominator (`ttm-mase-denominator-from-test-window`) the intended definition, or should it use the training-series naive error?
