# Audit — SDForger: Forging Time Series with Language (NeurIPS 2025, #2371)

## Summary

SDForger generates synthetic time series by embedding signals into a low-dimensional
tabular form (FastICA / FPC), encoding rows as fill-in-the-middle text, fine-tuning an
autoregressive LLM (GPT-2 by default), and decoding sampled embeddings back to series.
The workspace ships two repos under `code/`:

- `code/IBM__fms-dgt/` — the public GitHub release. It contains only the SDForger
  *generation* databuilder (`fms_dgt/public/databuilders/time_series/`) inside the large
  general fms-dgt framework. It does NOT contain the paper's evaluation, baselines, or
  experiment drivers.
- `code/SDForger__neurips_supplemental/` — the NeurIPS supplemental ZIP, which is the
  actual reproduction package for the paper's numbers. It contains generation
  (`sources/run_data_augmentation.py` + `utils/augmentation/`), the similarity evaluation
  (`sources/run_TSG_evaluation.py` + `utils/evaluation/{distance,feature,shapelet}_based_measures.py`)
  used for Table 1, and the TTM utility evaluation (`sources/run_TTM_evaluation.py` +
  `utils/evaluation/utils_ttm.py`) used for Table 2, plus 15 dataset CSVs under `data/`.

What I did: read the paper (PDF + text extraction) Sections 3–6 and Appendices A–D; read
the supplemental generation, similarity-eval, and TTM-eval code end-to-end; and wrote two
read-only checks under `_audit_code/` (`check_eval_facts.py`, `check_missing_artifacts.py`)
that (a) reproduce the ED/DTW array-pairing and StandardScaler scaling logic and the
shapelet reshape constraint, and (b) scan the package for baseline-generation and
conditional-classifier code. I did not (and cannot, without GPUs / model downloads) re-run
the full generation or TTM training.

The core generation and similarity/utility evaluation code is present, runs as a coherent
pipeline, seeds all RNGs (`utils/generals.py`), and the TTM utility splits are temporally
disjoint (train/val/test from non-overlapping segments, scaler fit on train only) — so I
found no leakage in the utility evaluation. The principal gaps are (1) the comparison
baselines (TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4) have no generation code in the package,
(2) the Section 6 / Figure 2 conditional-generation classifier (the "0.81 accuracy") is
not in the package, and (3) the similarity metrics ED/DTW pair real window *i* with
synthetic sample *i* by raw array index, which is an arbitrary correspondence for
unconditional generation (and the ED scaling in code differs from the metric's own paper
definition).

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — SDForger MDD/ACD/SD/KD (feature-based) | `utils/evaluation/feature_based_measures.py` (`calculate_mdd/acd/sd/kd`) via `run_TSG_evaluation.py` | code present (not re-run) | n/a (not re-run) | Present, not re-executed |
| Table 1 — SDForger ED/DTW/SHR (distance-based) | `utils/evaluation/distance_based_measures.py`, `shapelet_based_measures.py` | code present (not re-run) | n/a | Present; ED/DTW pairing + ED scaling concerns (see `ed-dtw-index-pairing`, `ed-scaling-vs-paper-def`) |
| Table 1 — baseline rows (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) | (none — no baseline generation code in package) | — | — | MISSING (`baselines-generation-missing`) |
| Table 2 — TTM RMSE/MASE/WQL (SDForger, OD, 0-shot, OD+gen) | `utils/evaluation/utils_ttm.py` (`evaluate_ttm_model`, `compute_rmse/mase/wql`) via `run_TTM_evaluation.py` | code present (not re-run) | n/a | Present, not re-executed; WQL on point forecast (see `wql-point-forecast-degenerate`) |
| Table 2 — baseline rows (generated / +OD) | (none — no baseline generation code) | — | — | MISSING (cross-ref `baselines-generation-missing`) |
| §6 / Figure 2 — "accuracy of 0.81" longitudinal-kNN identifying generated curves | (none — no classifier code in `sources/`, `utils/`, or the notebook) | — | — | MISSING (`conditional-knn-eval-missing`) |
| Appendix D tables D.1–D.14 (variance retained, rejection rates, per-dataset, ablations, LLM/time) | partly derivable from `info_dict*.json` + per-config runs; no aggregation/plotting scripts in package | — | — | Aggregation scripts not shipped (cross-ref `baselines-generation-missing` for any baseline columns) |
| 12 experiment datasets (Appendix C) | `code/SDForger__neurips_supplemental/data/*.csv` (15 CSVs present) | files present | ✓ | Verified present |

## missing

```yaml finding
id: baselines-generation-missing
category: missing
topic: "baselines / reproducibility"
title: "No generation code for the five baselines compared in Tables 1 and 2"
severity: high
confidence: high
status: finding
file: sources/run_TSG_evaluation.py
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
claim: "The TSG and TTM evaluation drivers only ever read SDForger-generated data ('generated_data.npy') and the original data; there is no code in the package that generates, loads, or wraps the five competitor models (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) whose rows fill every comparison in Tables 1 and 2."
concern: "The paper's central claim that SDForger 'outperforms existing generative models' in Tables 1–2 cannot be reproduced from this package because the baseline numbers are produced by code that is not included."
resolution: "Authors: provide (or link, with commit) the baseline-generation scripts and the exact configs used so the Table 1/2 competitor rows can be reproduced under the same split, scaling, and metric code."
cross_refs: ["§5.1", "§5.2"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Table 1, Table 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: conditional-knn-eval-missing
category: missing
topic: "result traceability"
title: "Section 6 / Figure 2 conditional kNN accuracy (0.81) has no code in the package"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2).
claim: "Section 6 reports a 0.81 longitudinal-kNN accuracy for the text-conditioned generation experiment, but no kNN / classifier / scikit-fda code exists anywhere in the supplemental (the only related file, notebook/conditional_generation.ipynb, performs conditional generation and plotting but contains no classifier)."
concern: "The headline number quantifying the conditional-generation capability (Figure 2) is not reproducible from the released code."
resolution: "Authors: provide the longitudinal-kNN evaluation script (training data, classifier settings, train/test protocol) that produces the 0.81 accuracy."
cross_refs: ["§6"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 6, Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No clear scientific-correctness bug found. The similarity code runs as a coherent
pipeline; the only run-shape limitation (shapelet RE assumes single-channel input,
`utils/evaluation/shapelet_based_measures.py:20-21` reshapes `(N, L, C)` to `(N, L)`,
which only succeeds when `C==1`) is consistent with how the per-channel similarity tables
are organised and did not surface as an error in the cited single-channel settings, so I
record it as context rather than a standalone bug. `0` findings.

## difference

```yaml finding
id: ed-scaling-vs-paper-def
category: difference
topic: "evaluation consistency / preprocessing"
title: "ED computed on standard-scaled data, but its paper definition assumes [0,1] scaling"
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
claim: "All series are standard-scaled per timestamp (zero mean, unit variance) before generation and evaluation; the Euclidean Distance metric is then computed on these standard-scaled values."
concern: "Appendix B.2 defines ED stating 'the input time series has been preprocessed to fit within the range of [0,1]', whereas the code (and the protocol in Appendix C.2) use StandardScaler, so the ED values are not on the [0,1] basis the metric definition assumes (the protocol and metric-definition sections of the paper are themselves inconsistent)."
resolution: "Authors: reconcile the ED definition (B.2, '[0,1]') with the actual standard-scaling used (C.2 and code); confirm which scaling produced the Table 1 ED column."
cross_refs: ["ed-dtw-index-pairing"]
check_script: _audit_code/check_eval_facts.py
paper_ref: "Appendix B.2 (ED), Appendix C.2 (protocol)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ed-dtw-index-pairing
category: methodology
topic: "similarity metrics / evaluation validity"
title: "ED and DTW pair real window i with synthetic sample i by raw array index"
severity: medium
confidence: medium
status: finding
file: utils/evaluation/distance_based_measures.py
line_start: 13
line_end: 26
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

      distance_eu = np.array(distance_eu)
      average_distance_eu = distance_eu.mean()
      return average_distance_eu
claim: "ED (and DTW, line 28-37, identically) iterates i over ori_data.shape[0] and computes the distance between real sample i and generated sample i, i.e. it pairs them by raw array position. In the reported setting there are 30 real windows and 100 generated samples (paper App. C.2), so only the first 30 generated samples are scored and each is matched to an unrelated real window by index."
concern: "For unconditional generation there is no natural correspondence between a specific real window and a specific synthetic sample, so a position-based 1:1 ED/DTW is an arbitrary pairing (a set-to-set distance such as nearest-neighbour or optimal assignment is the usual choice); the resulting Table 1 ED/DTW values—and 70% of the generated samples being ignored—depend on generation order rather than distributional closeness."
resolution: "Authors: justify the index-based real↔synthetic pairing for ED/DTW, or recompute these metrics with a correspondence-free formulation (e.g. nearest-neighbour or assignment) and confirm rankings are unchanged."
cross_refs: ["ed-scaling-vs-paper-def"]
check_script: _audit_code/check_eval_facts.py
paper_ref: "Appendix B.2 (ED/DTW), Appendix C.2 (30 real / 100 generated)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: wql-point-forecast-degenerate
category: methodology
topic: "utility metrics / WQL"
title: "WQL computed from a single point forecast, not quantile forecasts"
severity: low
confidence: high
status: finding
file: utils/evaluation/utils_ttm.py
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
claim: "The Weighted Quantile Loss in Table 2 is computed by applying the pinball loss at quantiles 0.1/0.5/0.9 to the SAME single point forecast `pred` (TTM's deterministic prediction), rather than to per-quantile forecasts."
concern: "WQL is meant to score a probabilistic (multi-quantile) forecast; evaluating each quantile against one point prediction degenerates to a fixed reweighting of the absolute error and does not measure calibration, so the Table 2 'WQL' column is not the standard quantity, though it is applied identically across all training sources."
resolution: "Authors: clarify whether TTM produces quantile outputs; if not, state that 'WQL' here is a point-forecast pinball average so it is not over-interpreted as a probabilistic-forecast score."
cross_refs: []
paper_ref: "Table 2 (WQL columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 2 | high | Baseline-generation code absent for all Table 1/2 competitors; §6 conditional-kNN (0.81) eval absent. |
| bug | 0 | - | No scientific-correctness bug; shapelet RE single-channel reshape noted as context only. |
| difference | 1 | low | ED computed on standard-scaled data; paper B.2 ED definition assumes [0,1]. |
| methodology | 2 | medium | ED/DTW use arbitrary index pairing of real↔synthetic; WQL from a point forecast. |

## Top take-aways

1. **(missing, high)** `baselines-generation-missing` — none of the five competitor models
   (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) have generation code in the package, so the
   core "outperforms existing models" comparison in Tables 1–2 is not reproducible.
2. **(missing, medium)** `conditional-knn-eval-missing` — the Section 6 / Figure 2 "0.81
   accuracy" longitudinal-kNN evaluation has no code anywhere in the package.
3. **(methodology, medium)** `ed-dtw-index-pairing` — ED/DTW pair real window *i* with
   synthetic sample *i* by raw index and ignore 70% of generated samples (30 real vs 100
   generated), an arbitrary correspondence for unconditional generation.
4. **(difference, low)** `ed-scaling-vs-paper-def` — ED is computed on standard-scaled data
   while its paper definition (App. B.2) assumes [0,1] scaling (internal paper inconsistency;
   code follows the App. C.2 protocol).
5. **(methodology, low)** `wql-point-forecast-degenerate` — Table 2 "WQL" is the pinball loss
   of a single point forecast across quantiles, not a probabilistic-forecast WQL.

## Items that genuinely look fine

- **RNG seeding is complete** (`utils/generals.py:11-17`): Python `random`, NumPy, Torch,
  HF `set_seed`, and CUDA are all seeded from one config seed.
- **TTM utility splits are leakage-free**: train/val/test come from non-overlapping segments
  (`utils_preprocess_data.py:339-357`) and the StandardScaler is fit on train only and merely
  `.transform`-applied to val/test (`utils_preprocess_data.py:235-254`); the paper states the
  same (App. C.2, "scaling computed on the training set ... applied across all splits").
- **Datasets are shipped**: 15 dataset CSVs are present under `data/`, covering the 12
  benchmark datasets listed in Appendix C.
- **Decoding is the genuine inverse of the embedding** (FastICA mixing / FPC basis
  reconstruction in `sdforger_augmentation.py:277-299`), consistent with §3.3.

## Open questions for the authors

- Where is the baseline-generation code and the exact configs used for the Table 1/2
  competitor rows? (high severity, currently high confidence it is absent from this package.)
- Is the index-based real↔synthetic pairing in ED/DTW intentional, and do the Table 1
  rankings survive a correspondence-free ED/DTW? (medium severity, medium confidence.)
- Does TTM emit quantile forecasts, or is the Table 2 "WQL" deliberately a point-forecast
  pinball average? (low severity.)
