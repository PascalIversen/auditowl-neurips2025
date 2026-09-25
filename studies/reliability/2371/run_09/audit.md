# Audit — Forging Time Series with Language (SDForger), paper #2371

## Summary

SDForger generates synthetic multivariate time series by (i) embedding windows
into tabular coefficients via FastICA / FPC, (ii) text-encoding the coefficients
and fine-tuning an autoregressive LLM (GPT-2) to generate new coefficient rows,
(iii) decoding back to series. Two author repos are provided:

- `code/SDForger__neurips_supplemental/` — the **full reproduction package**:
  generation (`utils/augmentation/`), the similarity metrics for Table 1
  (`utils/evaluation/{distance,feature,shapelet}_based_measures.py`), the TTM
  utility eval for Table 2 (`utils/evaluation/utils_ttm.py`), configs, and the
  12 experiment datasets under `data/`.
- `code/IBM__fms-dgt/fms_dgt/public/databuilders/time_series/` — the public
  open-source release of the **generation step only** (no evaluation, no
  baselines). It is a parallel re-implementation of the same FICA/FPC + LLM
  pipeline; the supplemental package is the one that produces the paper tables.

What I did: read the paper (PDF + text) including Appendices A–D; read every
script in the supplemental package and the public databuilder README; and ran
three small read-only checks under `_audit_code/` (numpy 1.26.4) reproducing the
exact ED loop, the WQL formula, and the generated/original count mismatch. I did
NOT run the full GPT-2/TTM pipeline (needs A100-class GPU + the unshipped
`tsfm_public` package).

Headline finding-free items: the SDForger generation pipeline, the period-aware
segmentation, the train/val/test split for the TTM utility experiment
(consecutive, forward-in-time, scaler fit on train only), and the FastICA/FPC
embedding + decoding all look correct and faithful to the paper. The concerns
below are concentrated in the **evaluation metrics** and in **missing
artefacts** for results the code cannot reproduce.

## Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Table 1 SDF-ICA/FPC rows: MDD, ACD, SD, KD | `utils/evaluation/feature_based_measures.py` via `run_TSG_evaluation.py` | yes (SDForger rows) | not run | Code present (metric caveats below) |
| Table 1 SDF rows: ED, DTW | `utils/evaluation/distance_based_measures.py:13-37` | yes | not run | Code present — index-pairing concern (F01) |
| Table 1 SDF rows: SHR (SHAP-RE) | `utils/evaluation/shapelet_based_measures.py:6-40` | yes (C=1 only) | not run | Code present |
| Table 1 **baseline** rows (TimeVAE, TimeVQVAE, RtsGAN, SdeGAN, LS4) | (none) | no | — | MISSING baseline generation code (F04) |
| Table 2 SDF / OD / 0-shot / OD+Gen rows: RMSE, MASE, WQL | `utils/evaluation/utils_ttm.py:301-379` via `run_TTM_evaluation.py` | yes | not run | Code present — WQL mislabel (F02), MASE scaling (F03) |
| Table 2 **baseline** rows (TimeVAE…LS4, +OD) | (none) | no | — | MISSING baseline generation code (F04) |
| §6 / Fig. 2: kNN classifier accuracy = 0.81 on conditional generation | `notebook/conditional_generation.ipynb` (plots only) | no | — | MISSING classifier code (F05) |
| Appendix Tables D.1–D.14 (variance retained, rejection rates, k/LLM/efficiency ablations, per-dataset scores) | partial: D.1 variance derivable from embedding code; D.6/D.7 via re-running with different `k`; rejection stats from `sdforger.generate` filtering | partial | not run | Mostly re-runnable for SDForger; no driver scripts shipped (F06) |

## missing

```yaml finding
id: baseline-generation-code-absent
category: missing
topic: "baselines / reproducibility"
title: "No baseline generator code; Table 1 & 2 baseline rows not reproducible"
severity: high
confidence: high
status: finding
file: code_links.txt
line_start: 1
line_end: 14
quote: |
  # AUTHOR / PRIMARY repositories (cloned):
  # 1. GitHub open-source release — the SDForger *generation* databuilder only.
claim: "Both shipped repos contain only SDForger generation + the SDForger evaluation harness; neither contains generation code or generated outputs for the five baselines (TimeVAE, TimeVQVAE, RtsGAN, SdeGAN, LS4) that Tables 1 and 2 compare against."
concern: "The headline claim that SDForger 'outperforms existing generative models' rests on baseline numbers that cannot be reproduced or audited from the released code; only SDForger's own rows can be regenerated."
resolution: "Authors: release the baseline generation scripts/configs (or the generated baseline `.npy` files) so the full Table 1 / Table 2 comparison can be reproduced under the same split, metrics, and preprocessing."
cross_refs: ["F01", "F02"]
paper_ref: "Tables 1 and 2; §4 Baselines"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: section6-knn-accuracy-no-code
category: missing
topic: "result traceability"
title: "§6 / Fig. 2 kNN accuracy = 0.81 has no classifier code in repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2).
claim: "The repo's only conditional-generation artefact is notebook/conditional_generation.ipynb, which plots the channel-conditioned curves (Fig. 2) but contains no kNN classifier, no skfda/KNeighbors call, and no accuracy computation; a repo-wide grep finds none."
concern: "The quantitative §6 claim (accuracy 0.81) traces to no script that computes it, so it cannot be verified or reproduced from the released code."
resolution: "Authors: provide the longitudinal-kNN evaluation script (train/test setup, distance, number of curves) used to obtain the 0.81 figure."
cross_refs: []
check_script: _audit_code/check_ed_index_pairing.py
paper_ref: "Section 6; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ttm-tsfm-revision-unpinned
category: missing
topic: "dependencies / reproducibility"
title: "TTM utility eval depends on unpinned tsfm_public + moving model revision"
severity: low
confidence: medium
status: finding
file: code/SDForger__neurips_supplemental/README.md
line_start: 37
line_end: 42
quote: |
  To run TTM evaluation:
  ```shell
  git clone "https://github.com/ibm-granite/granite-tsfm.git" 
  cd granite-tsfm
  pip install ".[notebooks]"
  ```
claim: "Table 2 (TTM utility) requires `tsfm_public`, which the README tells users to `git clone` HEAD with no commit/tag, and config_ttm.yaml sets `TTM_MODEL_REVISION: main` (a moving reference); neither is pinned in the conda env files (which omit tsfm_public)."
concern: "The TTM forecasting backbone and its preprocessing can change on the granite-tsfm `main` branch, so Table 2 numbers are not pinned to a reproducible dependency state."
resolution: "Authors: pin the granite-tsfm commit and the TTM model revision used for the paper."
cross_refs: ["F02"]
paper_ref: "Table 2; §5.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-table-driver-scripts
category: missing
topic: "result traceability / drivers"
title: "No driver scripts to regenerate the aggregate paper tables / appendices"
severity: low
confidence: medium
status: question
file: code/SDForger__neurips_supplemental/README.md
line_start: 57
line_end: 66
quote: |
  ## Example: Univariate Settings + TSG evaluation

  The default configuration for univariate augmentation is [here](sources/config/config.yaml). It correponds to the univariate augmentation of the variable ```cnt``` of the bikesharing dataset.

  First, run data augmentation. Then, run TSG evaluation:
claim: "The repo ships single-config entrypoints that produce one CSV row per run; there is no harness that sweeps the 12 datasets / both embeddings / multiple settings and aggregates them into the normalized averages and ranks of Tables 1, D.3, D.4, D.6–D.14."
concern: "Reproducing the aggregate tables requires re-running each config manually and re-implementing the normalization/ranking step, which is not provided."
resolution: "Authors: provide the aggregation/normalization/ranking script and the per-dataset config list used to build Tables 1 and D.*."
cross_refs: []
paper_ref: "Table 1; Appendix D"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## bug

```yaml finding
id: config-seed-overridden-by-42
category: bug
topic: "reproducibility / seeding"
title: "Config `seed` silently overridden by hardcoded 42 inside SDForger"
severity: low
confidence: high
status: finding
file: code/SDForger__neurips_supplemental/utils/augmentation/sdforger.py
line_start: 86
line_end: 87
quote: |
        self.seed = kwargs['seed'] if 'seed' in kwargs else 42
        self.set_seed()
claim: "sdforger_augmentation() constructs `SDForger(model_path=..., text_template=..., float_type=...)` without passing `seed`, so `self.seed` defaults to 42; `set_seed()` then re-seeds random/numpy/torch to 42, overriding the global `set_seed(SEED)` (config seed=54) for the LLM-shuffle (`dataset.sample(random_state=self.seed)`), the 80/20 train/val split (`train_test_split(seed=self.seed)`), and all generation sampling."
concern: "Changing the config `seed` does not actually change the core stochastic steps (fine-tuning shuffle/split, multinomial generation), so the exposed seed knob is misleading and the reported seed=54 runs are not what the field implies."
resolution: "Authors: thread the config seed into the SDForger constructor (or document that 42 is always used), and confirm which randomness was varied across reported runs."
cross_refs: []
check_script: _audit_code/check_wql_is_half_mae.py
paper_ref: "§4 Parameter settings"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: wql-is-half-mae
category: difference
topic: "evaluation metrics"
title: "Table 2 'WQL' column is 0.5*MAE, not a Weighted Quantile Loss"
severity: medium
confidence: high
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
claim: "TTM emits a single point forecast (`predictions[0]`), and compute_wql feeds that same point prediction for q=0.1, 0.5, 0.9; with one prediction the averaged pinball loss over a symmetric quantile set equals exactly 0.5*MAE (verified, ratio = 0.500000)."
concern: "WQL is a probabilistic metric requiring distinct quantile/probabilistic forecasts; as implemented it is a deterministic rescaling of MAE, so the 'WQL' column does not measure what its name claims (though it is applied symmetrically to all rows, so rankings are unaffected)."
resolution: "Authors: either compute WQL from genuine quantile forecasts, or rename the column (e.g. to 0.5*MAE / MAE) to reflect what is computed."
cross_refs: ["F03"]
check_script: _audit_code/check_wql_is_half_mae.py
paper_ref: "Table 2, WQL columns"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: mase-scaled-on-forecast-window
category: difference
topic: "evaluation metrics"
title: "MASE denominator uses forecast-window naive error, not training history"
severity: low
confidence: high
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
claim: "The MASE scaling factor (denominator) is the one-step naive error computed on the per-sample *forecast horizon's* ground truth (`true[1:]-true[:-1]`), not on the in-sample training series as the standard MASE definition requires."
concern: "Scaling on the (short, 96-step) forecast window rather than the training history changes the magnitude and the '1.0 = naive' interpretation of MASE; the values are valid as a relative score across rows but are not standard MASE."
resolution: "Authors: confirm whether MASE was intended to scale by the training-period naive error, and clarify the per-window scaling in the paper."
cross_refs: ["F02"]
paper_ref: "Table 2, MASE columns"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ed-dtw-index-pairing
category: methodology
topic: "evaluation metrics / similarity"
title: "ED & DTW pair original[i] with generated[i] by arbitrary index"
severity: medium
confidence: high
status: finding
file: code/SDForger__neurips_supplemental/utils/evaluation/distance_based_measures.py
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
claim: "ED (and DTW, same loop at lines 28-37) compute the distance between the i-th original window and the i-th generated window by row index, although the two sets are produced independently and have no correspondence. The loop runs over `ori_data.shape[0]` (=30), so when 100 instances are generated it silently uses only the first 30 generated rows (verified) and would IndexError if fewer than 30 were generated."
concern: "An index-pairing distance is not a valid set-to-set similarity: it is order-dependent (re-shuffling the generated rows changes ED/DTW, verified) and discards most generated samples, so the reported absolute ED/DTW similarity values (Table 1) are not well-defined as a generation-quality metric (it is applied symmetrically across methods, limiting—but not removing—the impact on rankings)."
resolution: "Authors: clarify the intended pairing; ED/DTW similarity to a sample set should use a matching (e.g. nearest-neighbour / optimal assignment) or a distributional distance, and should use all generated samples. Confirm whether row order carries meaning here."
cross_refs: ["F04"]
check_script: _audit_code/check_ed_index_pairing.py
paper_ref: "Table 1, ED and DTW columns; Appendix B.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 4          | high         | Baseline gen code absent; §6 kNN, table drivers, TTM dep   |
| bug         | 1          | low          | Config `seed` overridden by hardcoded 42 in SDForger       |
| difference  | 2          | medium       | "WQL" is 0.5*MAE; MASE scaled on forecast window           |
| methodology | 1          | medium       | ED/DTW pair samples by arbitrary index, discard extras     |

## Top take-aways

1. **(missing, high)** No baseline generation code or outputs — Tables 1 & 2
   baseline rows, hence the "outperforms baselines" claim, are not reproducible
   from the release. [`baseline-generation-code-absent`]
2. **(methodology, medium)** ED & DTW in Table 1 pair original[i] with
   generated[i] by arbitrary index, are order-dependent, and silently use only
   the first 30 of 100 generated samples. [`ed-dtw-index-pairing`]
3. **(difference, medium)** The Table 2 "WQL" column is exactly 0.5*MAE on a
   single point forecast, not a Weighted Quantile Loss. [`wql-is-half-mae`]
4. **(missing, medium)** The §6 conditional-generation accuracy = 0.81 (Fig. 2)
   has no classifier code in the repo. [`section6-knn-accuracy-no-code`]
5. **(difference, low)** MASE is scaled by the forecast-window naive error, not
   the training history → non-standard MASE. [`mase-scaled-on-forecast-window`]
6. **(bug, low)** The config `seed` is silently overridden by a hardcoded 42
   inside SDForger, so it does not control fine-tuning shuffle/split or
   generation. [`config-seed-overridden-by-42`]

## Items that genuinely look fine

- **Train/val/test split for the TTM utility experiment** (Table 2):
  consecutive, forward-in-time segments (train before val before test), per-
  timestamp StandardScaler fit on train and applied to val/test — no temporal
  leakage (`utils/augmentation/utils_preprocess_data.py:323-395`).
- **Multisample split**: train/val/test drawn from disjoint sample columns via
  `random.sample` — no sample overlap (`...:278-320`).
- **SDForger generation pipeline**: FastICA/FPC embed → text encode → GPT-2
  fine-tune (early stopping on a 20% validation split) → decode, matching the
  paper's Section 3 method; in-generation filtering and stopping criterion match
  Appendix A.3/A.4.
- **Dependencies pinned** for the core env (`sdforgerpy310cuda.yaml`:
  torch==2.4.1, transformers==4.46.3, dtaidistance==2.3.13, etc.).
- **Datasets shipped**: all 12 benchmark datasets are present under `data/`.
- **MDD/ACD/SD/KD** feature-based metrics are computed over the full sample sets
  (distribution-level), not index-paired, and look consistent with Appendix B.1.

## Open questions for the authors

- Was ED/DTW row-index pairing intentional, and does the generated-sample order
  carry meaning? (`ed-dtw-index-pairing`)
- Which sources of randomness were varied across the reported runs, given the
  hardcoded seed 42? (`config-seed-overridden-by-42`)
- Which granite-tsfm commit and TTM model revision produced Table 2?
  (`ttm-tsfm-revision-unpinned`)
