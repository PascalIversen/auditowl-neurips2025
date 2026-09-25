# Audit — SDForger: Forging Time Series with Language (paper 2371)

## 1. Summary

The workspace ships two author repositories. `code/IBM__fms-dgt/` is the
open-source release: it contains only the **generation** databuilder
(`fms_dgt/public/databuilders/time_series/`). `code/SDForger__neurips_supplemental/`
is the NeurIPS supplemental ZIP — the fuller reproduction package containing
SDForger generation (`utils/augmentation/sdforger*.py`), the similarity metrics
(`utils/evaluation/{feature,distance,shapelet}_based_measures.py`), the TTM
utility evaluation (`utils/evaluation/utils_ttm.py`), driver scripts
(`sources/run_*.py`), pinned conda envs, and 16 dataset CSVs. The paper's
headline tables are Table 1 (similarity: MDD/ACD/SD/KD/ED/DTW/SHR), Table 2
(TTM utility: RMSE/MASE/WQL), and Figure 2 (text-conditioned generation,
classifier accuracy 0.81).

I read the paper PDF (Sections 3–6, Appendices A–D) and every `.py` file in the
supplemental, plus the notebook (`conditional_generation.ipynb`). I ran five
deterministic checks under `_audit_code/` (read-only re-implementations of the
cited code on synthetic arrays): the shapelet reshape, the ED/DTW index pairing,
the WQL point-forecast degeneracy, the hardcoded-42 seed override, and a
traceability grep over the supplemental `.py` files. I did not retrain any model
or download weights (no network).

What works: the seven per-dataset similarity metrics (MDD, ACD, SD, KD, ED, DTW,
SHR) and the four utility metrics (RMSE, MASE, WQL, Sobolev-H1) are each
implemented and wired into the driver scripts that write the result CSVs;
dependencies are pinned; the generation/filtering/decoding pipeline is complete.
The defects are: (a) several headline/aggregate artefacts are **not computed by
any repo script** (normalized-average and average-rank columns of Tables 1 & 2,
the Figure-2 classifier accuracy, and all baseline generators), and (b) two
metric implementations have methodological problems (ED/DTW pair real and
synthetic samples by array index even though synthetic samples have no
correspondence to specific real windows; WQL is computed from a single point
forecast for all quantiles). The hardcoded internal seed undermines the
"averaged across 5 seeds" claim for the generation stage.

## 2. Result-traceability table

| Paper artefact | Repo location | Computed? | Matches | Status |
|---|---|---|---|---|
| Table 1: MDD per model/dataset | `utils/evaluation/feature_based_measures.py:88-94` (`calculate_mdd`) via `run_TSG_evaluation.py` | yes | not run (no weights) | Verified-present |
| Table 1: ACD | `feature_based_measures.py:155-161` (`calculate_acd`) | yes | — | Verified-present |
| Table 1: SD | `feature_based_measures.py:183-191` (`calculate_sd`) | yes | — | Verified-present |
| Table 1: KD | `feature_based_measures.py:215-223` (`calculate_kd`) | yes | — | Verified-present |
| Table 1: ED | `distance_based_measures.py:13-26` (`calculate_ed`) | yes (but index-paired) | — | METHODOLOGY (see `ed-dtw-index-pairing`) |
| Table 1: DTW | `distance_based_measures.py:28-37` (`calculate_dtw`) | yes (index-paired) | — | METHODOLOGY (see `ed-dtw-index-pairing`) |
| Table 1: SHR (SHAP-RE) | `shapelet_based_measures.py:6-40` (`calculate_shapelet_recons_err`) | yes (univariate only) | — | Verified-present (limitation `shapelet-univariate-only`) |
| Table 1/2: "Norm. Avg." columns | (none) | NO | — | MISSING (`aggregates-not-in-repo`) |
| Table 1/2: "Avg. Rank" column | (none) | NO | — | MISSING (`aggregates-not-in-repo`) |
| Table 1/2: baseline rows (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) | (none) | NO generation code | — | MISSING (`baselines-not-in-repo`) |
| Table 2: RMSE/MASE/WQL/H1 | `utils/evaluation/utils_ttm.py:243-379` via `run_TTM_evaluation.py` | yes | — | Verified-present |
| Table 2: WQL specifically | `utils_ttm.py:253-259` | yes (degenerate) | — | METHODOLOGY (`wql-point-forecast`) |
| Fig. 2: classifier accuracy 0.81 (longitudinal kNN) | (none; notebook plots only) | NO | — | MISSING (`fig2-accuracy-not-in-repo`) |
| Table D.1 variance retained | `sdforger_augmentation.py:86-110 / 152-173` | yes (printed) | — | Verified-present |
| Table D.2 filtering stats (5 seeds) | `sdforger.py:521-566` (filtering); seed handling `sdforger.py:86,834-840` | partial | — | METHODOLOGY (`sdforger-seed-hardcoded-42`) |
| "averaged across 5 seeds" (Table D.2) | config `seed` exists but is overridden | NO real seed control on generation | — | see `sdforger-seed-hardcoded-42` |

## 3. Findings

## missing

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "result traceability / baselines"
title: "No generation code for any of the 5 baselines in Tables 1 and 2"
severity: high
confidence: high
status: finding
file: out/traceability_grep.csv
csv_row: 3
quote: |
  baseline_generation,0,False,
claim: "A grep over every .py file in the supplemental finds zero references to TimeVAE, TimeVQVAE, RTSGAN/RtsGAN, SDEGAN/SdeGAN or LS4; only SDForger generation is present. The baseline rows in Tables 1 and 2 therefore cannot be reproduced from this repo."
concern: "The central claim is that SDForger matches or beats these baselines, but the numbers the comparison rests on are produced by code that is absent, so a reviewer cannot reproduce or check the competitor scores."
resolution: "Authors: provide the baseline generation scripts (or exact commands / configs and pinned forks) used to produce the TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4 rows, and the synthetic outputs that were fed into the shared evaluation."
cross_refs: ["aggregates-not-in-repo"]
check_script: _audit_code/check_traceability_grep.py
paper_ref: "Tables 1 and 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: aggregates-not-in-repo
category: missing
topic: "result traceability / aggregation"
title: "Normalized-average and average-rank columns of Tables 1 & 2 not computed by any script"
severity: medium
confidence: high
status: finding
file: out/traceability_grep.csv
csv_row: 1
quote: |
  rank_or_normalized_avg,0,False,
claim: "No .py file in the supplemental computes a normalized average (Norm. Avg. Feat./Dist. columns) or an average rank (Rank / Avg. Rank columns) across models; the evaluation scripts only emit raw per-metric CSV rows (utils_evaluation.py:31-71)."
concern: "The 'Norm. Avg.' and 'Avg. Rank' columns are how the paper substantiates 'consistently strong and balanced performance' and 'top average rank', yet the normalization scheme and ranking are done off-repo and cannot be re-derived or checked."
resolution: "Authors: include the script that turns the raw metric CSVs into the normalized [0-1] averages and the average-rank columns, documenting the normalization (min-max over which set of models/datasets?) used in Tables 1 and 2."
cross_refs: ["baselines-not-in-repo"]
check_script: _audit_code/check_traceability_grep.py
paper_ref: "Table 1 (Norm. Avg., Rank); Table 2 (Avg. Rank)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fig2-accuracy-not-in-repo
category: missing
topic: "result traceability / Figure 2"
title: "Figure-2 classifier accuracy 0.81 has no computing code in the repo"
severity: medium
confidence: high
status: finding
file: out/traceability_grep.csv
csv_row: 2
quote: |
  knn_classifier_fig2,0,False,
claim: "The paper reports accuracy 0.81 from a 'longitudinal k-nearest neighbor classifier (Ramos-Carreno et al., 2024)' identifying generated curves (Section 6, Fig. 2), but a grep finds no KNeighbors / classifier / skfda / .score() call anywhere in the supplemental .py files; the conditional_generation.ipynb only generates and plots the curves (cells 14-17) and never fits or scores a classifier."
concern: "A reported quantitative result (the 0.81 accuracy demonstrating channel-conditioned generation) is not backed by any runnable code, so it cannot be reproduced or verified."
resolution: "Authors: add the script/notebook cell that trains the longitudinal kNN classifier on real data and evaluates it on the conditionally generated curves to obtain accuracy 0.81, and list the scikit-fda dependency it needs."
cross_refs: []
check_script: _audit_code/check_traceability_grep.py
paper_ref: "Section 6 'Shaping time series with language'; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

(No standalone runtime bug owns its own finding. The shapelet `reshape` would
crash on multivariate input but the metric is only reported for univariate /
multisample (single-channel) data, so it is filed as a methodology limitation
below rather than a bug affecting reported numbers.)

## difference

(No pure code↔paper faithfulness difference rises to a finding once the
methodology and missing items above are owned. The `config.yaml`
`sdforger_embedding_dim: 5` vs paper's k=3 is excluded: the CLI/config supports
k=3, the paper's setting — see audit-prompt EVALUATION CONSISTENCY exclusions.)

## methodology

```yaml finding
id: ed-dtw-index-pairing
category: methodology
topic: "similarity metrics / sample correspondence"
title: "ED and DTW compare the i-th real window to the i-th synthetic sample by array index"
severity: high
confidence: medium
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
claim: "ED (and identically calculate_dtw at lines 28-37) loops i over range(ori_data.shape[0]) and subtracts gen_data[i] from ori_data[i], i.e. it pairs the i-th original window with the i-th generated sample by array position. Synthetic samples are produced independently by the LLM (sdforger_augmentation.py) with no correspondence to any specific real window, and the generated count (100) differs from the original count (30 windows), so 70 of the 100 generated samples are silently never compared. My check confirms only ori.shape[0]=30 pairs are formed and that the score changes when the generated rows are permuted."
concern: "Pairing unrelated real and synthetic instances by index makes ED/DTW measure an essentially arbitrary alignment rather than distributional similarity, so the reported ED/DTW values (and the 'TimeVQVAE excels on distance metrics / SDForger balanced' narrative) are not well-defined comparisons."
resolution: "Authors: clarify how real and generated samples are matched for ED/DTW (nearest-neighbor assignment? sorted? full pairwise min as in TSGBench?), and report whether the metric is invariant to the ordering of the generated set; if index pairing is intended, justify it given the count mismatch."
cross_refs: []
check_script: _audit_code/check_ed_dtw_pairing.py
paper_ref: "Appendix B.2; Table 1 ED, DTW columns"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: wql-point-forecast
category: methodology
topic: "utility metrics / WQL"
title: "WQL computed from a single point forecast for all quantiles"
severity: medium
confidence: medium
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
claim: "compute_wql recomputes errors = true - pred inside the quantile loop using the same single point prediction `pred` for every quantile q; the caller passes pred_val = predictions[0] (the TTM point forecast, utils_ttm.py:325). A true Weighted Quantile Loss needs distinct predicted quantiles q. My check confirms the returned value is a deterministic function of the point-error vector alone."
concern: "The Table-2 WQL column is not the standard weighted quantile loss (which requires quantile forecasts); it is a fixed reweighting of point errors, so it does not measure forecast-distribution calibration as the metric name implies."
resolution: "Authors: confirm whether TTM produced quantile forecasts; if only point forecasts are available, either drop WQL or relabel it, and clarify how the reported WQL values should be interpreted."
cross_refs: []
check_script: _audit_code/check_wql_point_forecast.py
paper_ref: "Table 2 WQL columns; Section 5.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: sdforger-seed-hardcoded-42
category: methodology
topic: "reproducibility / seeding"
title: "SDForger internal seed hardcoded to 42, overriding the config seed used for '5 seeds' averaging"
severity: medium
confidence: high
status: finding
file: utils/augmentation/sdforger.py
line_start: 86
line_end: 87
quote: |
        self.seed = kwargs['seed'] if 'seed' in kwargs else 42
        self.set_seed()
claim: "sdforger_augmentation.py:224 constructs `SDForger(model_path=llm, text_template=..., float_type=...)` with no seed kwarg, so SDForger.__init__ sets self.seed=42 and immediately calls self.set_seed() (sdforger.py:834-840), which runs random.seed(42)/np.random.seed(42)/torch.manual_seed(42) — overwriting the set_seed(SEED=54) the driver had called in run_data_augmentation.py:73. The LLM train/val split (sdforger.py:273) and column permutations then use 42 regardless of config; FastICA is additionally fixed at random_state=0 (sdforger_augmentation.py:165). My check reproduces the override."
concern: "Table D.2 reports statistics 'averaged across 5 seeds' and the abstract leans on robustness, but the dominant generation randomness (LLM finetuning split, sampling, permutation) is pinned to a constant 42, so the 5 seeds vary far less than implied and the multi-seed averages may understate variance."
resolution: "Authors: pass the config seed into SDForger (or remove the hardcoded 42) and confirm which randomness sources actually differ across the 5 seeds reported in Table D.2."
cross_refs: []
check_script: _audit_code/check_seed_override.py
paper_ref: "Appendix Table D.2 ('averaged across 5 seeds')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: shapelet-univariate-only
category: methodology
topic: "similarity metrics / SHAP-RE"
title: "SHAP-RE reshape collapses channels and only supports univariate (C=1) data"
severity: low
confidence: high
status: finding
file: utils/evaluation/shapelet_based_measures.py
line_start: 20
line_end: 23
quote: |
      train_data  = orig_data.reshape(orig_data.shape[0], orig_data.shape[1])
      test_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1])
      train_y  = np.random.rand(orig_data.shape[0])
      test_y = np.random.rand(gen_data.shape[0])
claim: "calculate_shapelet_recons_err reshapes a 3-D (n_samples, time, channels) array to 2-D (n_samples, time), which is only valid when channels==1; my check shows it raises 'cannot reshape array of size 3750 into shape (5,250)' for a (5,250,3) input. The dummy labels train_y/test_y are also drawn with un-seeded np.random.rand (the surrounding RNG state is the constant 42 from the SDForger seed)."
concern: "SHR is reported only for the single-channel multisample/univariate settings (Table 1), so this does not invalidate reported numbers, but the metric cannot be applied to multivariate data and its SIDL training depends on random label vectors whose only seeding is the incidental global state."
resolution: "Authors: confirm SHR was only ever computed on single-channel data; if so, document the limitation, and confirm the random label vectors do not affect the unsupervised reconstruction error reported."
cross_refs: ["sdforger-seed-hardcoded-42"]
check_script: _audit_code/check_shapelet_reshape.py
paper_ref: "Appendix B.2; Table 1 SHR column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | Baselines, Table 1/2 aggregates, and Fig-2 accuracy have no computing code. |
| bug         | 0          | -            | No runtime bug affects a reported number (shapelet crash only off univariate path). |
| difference  | 0          | -            | Config k=5 vs paper k=3 excluded (CLI supports k=3). |
| methodology | 4          | high         | ED/DTW index-pairing; WQL point-forecast; hardcoded seed; SHR univariate-only. |

## 5. Closing lists

### Top take-aways (ranked)
1. `baselines-not-in-repo` (missing, high) — the entire baseline comparison in Tables 1 and 2 has no generation code in the repo.
2. `ed-dtw-index-pairing` (methodology, high) — ED/DTW pair real and synthetic samples by array index with no correspondence and ignore 70% of generated samples.
3. `aggregates-not-in-repo` (missing, medium) — the normalized-average and average-rank columns (the paper's headline summaries) are not computed by any script.
4. `wql-point-forecast` (methodology, medium) — the Table-2 WQL column is a reweighting of point errors, not a real quantile loss.
5. `fig2-accuracy-not-in-repo` (missing, medium) — Figure-2's 0.81 classifier accuracy has no code.
6. `sdforger-seed-hardcoded-42` (methodology, medium) — the SDForger seed is fixed at 42, undermining the '5 seeds' averaging.

### Items that genuinely look fine
- The seven similarity metrics (MDD, ACD, SD, KD, ED, DTW, SHR) and four utility metrics (RMSE, MASE, WQL, Sobolev-H1) are each implemented and wired into driver scripts that emit the result CSVs.
- `acf_torch` (feature_based_measures.py:98-109) is not a shape/broadcast bug: for lag i>0, `x[:, i:]` and `x[:, :-i]` have matching length T-i; the ACD runs and returns the expected shape (verified in `_audit_code/check_acf_torch.py`).
- Dependencies are fully pinned (`sdforgerpy310cuda.yaml`, `sdforgerpy310mps.yaml`); the data CSVs for the example settings are shipped under `data/`.
- The TTM train/val/test ranges are consecutive, non-overlapping segments of the series (utils_preprocess_data.py:339-357), so the utility evaluation has a genuine held-out test set.
- The generation pipeline (segmentation, ICA/FPC embedding, FIM text encoding, LLM finetuning with early stopping, IQR-norm filtering, decoding) is complete and self-consistent.

### Open questions for the authors
- For ED/DTW: what is the intended correspondence between real and synthetic samples, and is the metric invariant to the order of the generated set? (`ed-dtw-index-pairing`)
- Did TTM produce quantile forecasts, or is WQL intentionally computed from a point forecast? (`wql-point-forecast`)
- Which randomness sources actually varied across the "5 seeds" in Table D.2 given the hardcoded internal seed of 42? (`sdforger-seed-hardcoded-42`)
