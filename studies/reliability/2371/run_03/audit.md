# Audit — SDForger (NeurIPS 2025 #2371)

## 1. Summary

The paper introduces **SDForger**, an LLM-based synthetic time-series generator
(FastICA/FPC embedding → text → GPT-2 fine-tuning → decode), and compares it to
five generative baselines on similarity metrics (Table 1) and a downstream TTM
forecasting utility task (Table 2).

Two repos are provided. `code/IBM__fms-dgt/` is the public open-source release
and only contains the **generation** databuilder
(`fms_dgt/public/databuilders/time_series/`), no evaluation. The fuller
reproduction artefact is the NeurIPS supplemental,
`code/SDForger__neurips_supplemental/`, which contains the generation pipeline
(`utils/augmentation/`), the similarity metrics (`utils/evaluation/`), the TTM
utility evaluation (`utils/evaluation/utils_ttm.py`), three driver scripts
(`sources/run_data_augmentation.py`, `run_TSG_evaluation.py`,
`run_TTM_evaluation.py`), two configs, a conditional-generation notebook, and 16
dataset CSVs.

What I did: read the three drivers and all of `utils/`; mapped every Table-1/2 and
headline number to code; ran one deterministic probe under `_audit_code/`
(`check_ed_dtw_pairing.py`) that reproduces the Euclidean-distance pairing logic
verbatim (the repo module could not be imported directly because `dtaidistance`
is not installed in the sandbox); grepped both repos for any baseline-model code
and for the Section-6 kNN-classifier code.

Headline findings: (i) **none of the five baselines** (TimeVAE, TimeVQVAE,
RtsGAN, SdeGAN, LS4) have any code in either repo, so the comparative claims in
Tables 1/2 are not reproducible; (ii) the ED/DTW similarity metrics pair
synthetic sample *i* with real sample *i* by array index — synthetic samples have
no correspondence to specific real samples, and the extra synthetic rows are
silently dropped; (iii) the SHAP-RE column of Table 1 is computed with unseeded
numpy randomness (the TSG driver never seeds), so it is non-deterministic; (iv)
the Section-6 "accuracy 0.81" number has no code; (v) generation forces
`bf16=True` which contradicts the `float32` config and breaks the README's MPS
path. The SDForger generation + its own SDForger-row metrics are present and look
internally coherent.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1, SDF-ICA/FPC rows (MDD,ACD,SD,KD,ED,DTW,SHR) | `utils/evaluation/TSG_evaluation.py` + `feature_/distance_/shapelet_based_measures.py` | recomputable (needs run) | n/a (no weights shipped) | Present (SDForger only) |
| Table 1, baselines TimeVAE/TimeVQVAE/RtsGAN/SdeGAN/LS4 | (none) | — | — | MISSING (no baseline code) |
| Table 1, ED & DTW columns | `utils/evaluation/distance_based_measures.py:13-37` | index-paired distance | — | Computed but index-paired (see ed-dtw-index-pairing) |
| Table 1, SHR column | `utils/evaluation/shapelet_based_measures.py` | non-deterministic | — | Computed but unseeded (see shap-re-unseeded) |
| Table 2, SDF rows (RMSE,MASE,WQL) | `sources/run_TTM_evaluation.py` + `utils/evaluation/utils_ttm.py` | recomputable (needs run) | n/a | Present (SDForger only) |
| Table 2, baselines (+GEN, +OD rows) | (none) | — | — | MISSING (no baseline code) |
| Table 2/utils, WQL metric | `utils/evaluation/utils_ttm.py:253-259` | un-normalised mean pinball | — | Present, non-canonical (see wql-not-normalised) |
| Section 6 / Fig. 2, kNN accuracy = 0.81 | notebook generates curves only | — | — | MISSING (no classifier code) |
| Table D.5, generation-time comparison | (none) | — | — | MISSING (no timing script) |
| Table D.2, filtering / rejection stats | `utils/augmentation/sdforger.py:521-565` (in-loop only, not logged out) | partial | — | Partially present |
| TTM utility metrics over channels | `utils/evaluation/utils_ttm.py:335-358` | mean over all forecast channels | unclear (paper says "target: count") | Question (see ttm-metric-all-channels) |

## 3. Findings

## missing

```yaml finding
id: baselines-absent
category: missing
topic: "baselines / result traceability"
title: "No code for any of the five baselines in Tables 1 and 2"
severity: high
confidence: high
status: finding
file: paper.pdf
line_start: 455
line_end: 463
quote: |
  Baselines
  We evaluated SDForger’s performance against several baseline models for synthetic time
  series generation, covering different approaches. Variational autoencoders: TimeVAE (Desai et al.,
  2021) ... TimeVQVAE
  (Lee et al., 2023) ... RTSGAN (Pei et al., 2021) ... SDEGAN (Kidger et al., 2021) ... and a diffusion-based
  model: LS4 (Zhou et al., 2023) ... Hyperparameters for all baseline
  competitors follow those reported in their original papers, except for SdeGAN, for which we fix the
  number of training iterations to 1000 to balance convergence and computational cost.
claim: "Tables 1, 2, D.3, D.4, D.5, D.6, D.7 compare SDForger to TimeVAE, TimeVQVAE, RtsGAN, SdeGAN, and LS4, but a recursive grep of both shipped repos finds no class, script, config, or fetch logic for any of these five models; only SDForger generation + evaluation exists."
concern: "The paper's central claim that SDForger 'outperforms existing generative models' rests on baseline numbers that cannot be regenerated from the released code, so the comparison is not reproducible."
resolution: "Authors: release the baseline-generation scripts / configs (or the generated baseline arrays) used to produce the non-SDForger rows of Tables 1 and 2, with hyperparameters and commit pins."
cross_refs: ["§5.1", "§5.2"]
paper_ref: "Tables 1 and 2; Baselines paragraph p.8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: section6-knn-accuracy-no-code
category: missing
topic: "result traceability"
title: "Section-6 kNN classifier accuracy 0.81 has no code"
severity: medium
confidence: high
status: finding
file: paper.pdf
line_start: 939
line_end: 940
quote: |
  For instance, using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2). These
claim: "The conditional-generation notebook (notebook/conditional_generation.ipynb, 19 cells) only generates and plots the channel-conditioned curves of Figure 2; no cell trains or scores a k-nearest-neighbour classifier, and no skfda/accuracy/score call exists anywhere in either repo."
concern: "The reported 0.81 channel-identification accuracy — the quantitative evidence for the textual-conditioning claim of Section 6 — cannot be reproduced from the released code."
resolution: "Authors: add the longitudinal-kNN evaluation code (classifier construction, train/test split on real vs. generated curves, accuracy computation) that yields the 0.81 figure."
cross_refs: ["§6"]
paper_ref: "Section 6, Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ttm-deps-unpinned
category: missing
topic: "dependencies / reproducibility"
title: "tsfm_public (TTM) absent from env files; installed from unpinned main"
severity: medium
confidence: high
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
claim: "The TTM utility evaluation imports tsfm_public, but neither sdforgerpy310cuda.yaml nor sdforgerpy310mps.yaml lists tsfm/granite-tsfm; the README installs it from an unpinned `main`, and config_ttm.yaml sets TTM_MODEL_REVISION: main, so both the library and the pretrained TTM checkpoint are moving targets."
concern: "Table 2's utility numbers depend on an unpinned dependency and an unpinned model revision, so the exact environment that produced them cannot be rebuilt."
resolution: "Authors: pin a granite-tsfm commit and a concrete TTM model revision (not `main`), and add tsfm to the environment file."
cross_refs: ["ttm-metric-all-channels"]
paper_ref: "Table 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: generation-time-table-no-code
category: missing
topic: "result traceability / efficiency"
title: "Table D.5 generation-time comparison has no timing script"
severity: low
confidence: medium
status: finding
file: paper.pdf
line_start: 1865
line_end: 1868
quote: |
  Table D.5: Average generation time: baselines Average time (in seconds) required to generate synthetic
  univariate time series for the bikesharing dataset across three targets: count, temperature, and humidity.
  We report results for two input sequence lengths: 250 and 500. All models were evaluated under the same
  computational constraints (-mem 20G -cores 1+1 -gpu v100) using a single NVIDIA V100 GPU.
claim: "Table D.5 reports per-model wall-clock generation times for SDForger and all five baselines, but no timing/benchmark harness exists in either repo (the drivers do not record generation time, and there is no baseline code to time)."
concern: "The efficiency claim ('SDForger is substantially faster than all competitors, often by one to two orders of magnitude') cannot be reproduced from the code."
resolution: "Authors: release the timing harness and the baseline runners used for Table D.5."
cross_refs: ["baselines-absent"]
paper_ref: "Table D.5; §5.3 Generation efficiency"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: bf16-hardcoded-vs-float32-config
category: bug
topic: "fine-tuning configuration"
title: "TrainingArguments forces bf16=True, contradicting float32 config and breaking MPS path"
severity: medium
confidence: medium
status: finding
file: code/SDForger__neurips_supplemental/utils/augmentation/sdforger.py
line_start: 305
line_end: 305
quote: |
                bf16=True,
claim: "The GPT-2 fine-tuning TrainingArguments hardcode bf16=True (line 305, inside the TrainingArguments constructor), while the model is loaded with torch_dtype=float32 (the config default sdforger_float_type: float32 → self.dtype=torch.float32, lines 100-101) and, on macOS, _train_args sets use_mps_device=True (line 49)."
concern: "bf16 mixed-precision is unconditional regardless of the float_type config and is unsupported on MPS, so the README's documented MPS install path errors out and the float32 setting is effectively ignored on the CUDA path."
resolution: "Authors: gate bf16 on device capability / float_type (e.g. bf16=torch.cuda.is_bf16_supported()) instead of hardcoding True, so the MPS path runs and float32 is honoured."
cross_refs: []
paper_ref: "Parameter settings p.8 (float settings)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: wql-not-normalised
category: difference
topic: "evaluation metric"
title: "WQL implemented as un-normalised mean pinball loss"
severity: low
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
claim: "compute_wql averages the pinball loss over the 3 quantiles and returns it with no normalisation by the sum/mean of |true|, unlike the canonical Weighted Quantile Loss (e.g. GluonTS/Chronos)."
concern: "The reported 'WQL' is a raw pinball-loss average rather than the canonical scale-normalised WQL, so the column is not directly comparable to WQL values reported elsewhere; it stays internally consistent across rows of Table 2 because every method uses the same function."
resolution: "Authors: confirm whether WQL is intentionally un-normalised; if it follows the standard definition, divide total_loss by the sum of absolute target values."
cross_refs: []
paper_ref: "Table 2 (WQL column)"
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
title: "ED and DTW pair synthetic sample i with real sample i by array index"
severity: medium
confidence: high
status: finding
file: code/SDForger__neurips_supplemental/utils/evaluation/distance_based_measures.py
line_start: 13
line_end: 37
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

  def calculate_dtw(ori_data,comp_data):
      distance_dtw = []
      n_samples = ori_data.shape[0]
      for i in range(n_samples):
          distance = multi_dtw_distance(ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True)
          distance_dtw.append(distance)

      distance_dtw = np.array(distance_dtw)
      average_distance_dtw = distance_dtw.mean()
      return average_distance_dtw
claim: "Both distance metrics loop i over the real samples and compute the distance between real row i and generated row i (gen_data[i]); my probe (_audit_code/check_ed_dtw_pairing.py) reproduces this and shows (a) reordering the synthetic block changes ED from 0 to 5.64 even when the two sets are identical, and (b) with 30 real vs 100 generated rows the result equals using only the first 30 generated rows."
concern: "Synthetic instances are generated independently and have no correspondence to specific real instances, so pairing real[i] with gen[i] measures the distance between arbitrary unrelated curves (a distributional/closest-match metric is intended); additionally the paper generates 100 synthetic instances but ED/DTW silently use only the first 30."
resolution: "Authors: confirm the intended pairing; ED/DTW for set-vs-set generation should use a matching (e.g. nearest-neighbour / optimal assignment) rather than index alignment, and should not discard generated rows beyond the real count."
cross_refs: ["§B.2"]
check_script: _audit_code/check_ed_dtw_pairing.py
paper_ref: "Table 1 (ED, DTW columns); Appendix B.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: shap-re-unseeded
category: methodology
topic: "reproducibility / statistical integrity"
title: "SHAP-RE (SHR) computed with unseeded numpy randomness in the TSG driver"
severity: medium
confidence: high
status: finding
file: code/SDForger__neurips_supplemental/utils/evaluation/shapelet_based_measures.py
line_start: 22
line_end: 36
quote: |
    train_y  = np.random.rand(orig_data.shape[0])
    test_y = np.random.rand(gen_data.shape[0])
    n_train, p = train_data.shape
    n_test, _ = test_data.shape


    A_rand_init = np.random.randn(n_test, K)
    q = int(np.ceil(p * r))
    runid = f'l_{lambdas}_K_{K}_q_{q}'
    # Train SIDL on original dataset
    S, A, Offsets, F_obj = USIDL(train_data, train_y, lambdas, K, q, c, epsilon, maxIter, maxInnerIter, runid)

    # Learn sparse coding on generated test set with dictionary learned from original dataset
    A_test = A_rand_init
    Offsets_test = np.random.randint(0, p - q, (n_test, K))
claim: "calculate_shapelet_recons_err and USIDL initialise the shapelet dictionary (utils_shapelet.py:26-28), the coefficient matrix, and the offsets with np.random.randn / randint / permutation; run_TSG_evaluation.py reads SEED but never calls set_seed (unlike run_data_augmentation.py), so these draws are unseeded each evaluation run."
concern: "The SHR column of Table 1 — by far the highest-variance metric (e.g. LS4 SHR = 160.4 vs single-digit values for SDForger) — is non-deterministic across evaluation runs, so the reported SHR values (and the per-method SHR ordering) are not reproducible."
resolution: "Authors: seed the SHAP-RE computation (and add set_seed in run_TSG_evaluation.py), or report SHR mean ± std over multiple random initialisations."
cross_refs: ["ed-dtw-index-pairing"]
paper_ref: "Table 1 (SHR column); Appendix B.2 Shapelet-based Reconstructions"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ttm-metric-all-channels
category: methodology
topic: "utility evaluation"
title: "TTM RMSE/MASE/WQL averaged over all forecast channels, not the stated target only"
severity: low
confidence: low
status: question
file: code/SDForger__neurips_supplemental/utils/evaluation/utils_ttm.py
line_start: 335
line_end: 358
quote: |
      for sample in range(0, true_val.shape[0]):
          list_rmse_per_channel = []
          list_mase_per_channel = []
          list_wql_per_channel = []
          list_h1_per_channel = []
          for channel in range(0, pred_val.shape[2]):
              rmse = compute_rmse(true_val[sample, :, channel], pred_val[sample, :, channel])
              ...
      # Step 7: Calculate the average metrics across all windows
      avg_rmse = np.mean(rmse_list)
claim: "evaluate_ttm_model loops over every channel of the prediction tensor and averages RMSE/MASE/WQL over all of them, whereas Table 2 labels each dataset with a single 'target' channel (e.g. bikesharing target: count, controls: temperature, humidity)."
concern: "If the TTM prediction tensor contains the control channels as well as the target, the reported metrics would be an average over target+controls rather than the target-only metric the table caption implies; I cannot confirm the tensor's channel content without running TTM (depends on prediction_channel_indices behaviour inside tsfm_public)."
resolution: "Authors: clarify whether pred_val contains only the target channel; if it also contains control channels, restrict the metric loop to the target channel(s) to match the table caption."
cross_refs: ["ttm-deps-unpinned"]
paper_ref: "Table 2 (target/control specification)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | All five baselines absent; Section-6 0.81 and Table D.5 timing have no code; TTM deps unpinned. |
| bug         | 1          | medium       | bf16=True hardcoded vs float32 config; breaks MPS path. |
| difference  | 1          | low          | WQL implemented as un-normalised mean pinball loss. |
| methodology | 3          | medium       | ED/DTW index-pairing; SHAP-RE unseeded; TTM metric channel scope (question). |

## 5. Closing lists

**Top take-aways** (≤6, by severity × confidence):
1. `baselines-absent` (missing, high) — no code for TimeVAE/TimeVQVAE/RtsGAN/SdeGAN/LS4; the comparative claim of Tables 1/2 is not reproducible.
2. `ed-dtw-index-pairing` (methodology, medium/high) — ED and DTW pair real and synthetic curves by array index and discard the extra generated rows.
3. `shap-re-unseeded` (methodology, medium/high) — SHAP-RE is non-deterministic; the TSG driver never seeds.
4. `section6-knn-accuracy-no-code` (missing, medium/high) — the 0.81 channel-identification accuracy has no classifier code.
5. `bf16-hardcoded-vs-float32-config` (bug, medium) — fine-tuning forces bf16, contradicting the float32 config and the documented MPS path.
6. `ttm-deps-unpinned` (missing, medium) — TTM library + model revision both `main`, not in the env file.

**Items that genuinely look fine**:
- SDForger generation pipeline (FastICA/FPC embed → text → GPT-2 fine-tune → decode) is fully present and internally consistent (`utils/augmentation/`).
- Train/val/test segmentation is temporally consecutive and non-overlapping, and StandardScaler is fit on train then applied to val/test (no scaling leakage): `utils_preprocess_data.py:223-258, 323-395`.
- The in-generation norm/IQR filtering and duplicate/NaN removal described in Appendix A.3 are implemented (`sdforger.py:482-565`).
- Feature-based metrics MDD/ACD/SD/KD are present and computed per the Appendix-B definitions (`feature_based_measures.py`).
- The dependency env files pin concrete versions (e.g. `dtaidistance==2.3.13`) for the generation/similarity side.
- The shapelet 2-D `reshape` (`shapelet_based_measures.py:20-21`) only assumes a single channel, which is consistent with SHR being reported only for the single-channel multisample/univariate settings.

**Open questions for the authors**:
- `ttm-metric-all-channels` — does the TTM prediction tensor contain only the target channel, or target+controls? (Determines whether Table 2 metrics are target-only.)
- Were the baseline arrays generated off-repo with a separate (unreleased) codebase, and can they be shared?
- Is the un-normalised WQL intentional, or should it follow the canonical scale-normalised definition?
