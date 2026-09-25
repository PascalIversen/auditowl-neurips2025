# Audit — SDForger: Forging Time Series with Language (paper 2371)

## Summary

The workspace contains two author artefacts. `code/IBM__fms-dgt/` is the public
open-source release, but it is **generation-only**: it ships the SDForger
time-series databuilder (`fms_dgt/public/databuilders/time_series/`) with no
similarity metrics, no utility/TTM evaluation, and no baselines.
`code/SDForger__neurips_supplemental/` is the NeurIPS supplemental package and is
the artefact that actually backs the paper's numbers: it contains the SDForger
augmentation pipeline (`utils/augmentation/`, `sources/run_data_augmentation.py`),
the similarity metrics MDD/ACD/SD/KD/ED/DTW/SHAP-RE
(`utils/evaluation/`, `sources/run_TSG_evaluation.py`), and the TTM utility
evaluation RMSE/MASE/WQL (`sources/run_TTM_evaluation.py`), plus the 15 dataset
CSVs and two example configs.

What I did: read both repos; mapped every numbered table/figure/headline number
to code; ran the repo's own `calculate_shapelet_recons_err` three times on
identical inputs to test determinism (`_audit_code/check_shr_nondeterminism.py`);
re-implemented the ED loop to measure how many generated samples it consumes and
whether the result depends on generated-sample ordering
(`_audit_code/check_ed_dtw_pairing.py`). I did not retrain any model (requires
GPUs / external `granite-tsfm`).

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 SDForger similarity (MDD/ACD/SD/KD/ED/DTW/SHR) | `utils/evaluation/TSG_evaluation.py:10-53` driven by `sources/run_TSG_evaluation.py` | not re-run (needs LLM fine-tune) | — | Present (code), see SHR/ED notes |
| Table 1 baselines (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) | (none) | — | — | MISSING (no baseline code in either repo) |
| Table 1 SHR (SHAP-RE) column + Dist. Norm. Avg. | `utils/evaluation/shapelet_based_measures.py:6-40` | nondeterministic: 8.10/7.96/8.05 on identical inputs | — | Present but UNSEEDED (see shr-unseeded) |
| Table 1 ED / DTW columns | `utils/evaluation/distance_based_measures.py:13-37` | re-implemented loop | — | Present; arbitrary index pairing, uses 30 of 100 gen (see ed-dtw-pairing) |
| Table 2 TTM utility (RMSE/MASE/WQL), 4 sources × 3 datasets | `sources/run_TTM_evaluation.py` + `utils/evaluation/utils_ttm.py:301-379` | not re-run (needs granite-tsfm + GPU) | — | Present (code) |
| Table 2 baselines (same 5 generative models) | (none) | — | — | MISSING (no baseline code) |
| §6 / Fig. 2 "accuracy of 0.81" longitudinal kNN classifier | `notebook/conditional_generation.ipynb` (plot only) | — | — | MISSING (no kNN/accuracy code) |
| Table D.1 variance retained per k | `utils/augmentation/sdforger_augmentation.py:86-110,152-173` | not re-run | — | Present (code) |
| Table D.2 rejection/filtering stats | `utils/augmentation/sdforger.py` generate/filtering | not re-run | — | Present (code) |

## missing

```yaml finding
id: baselines-no-code
category: missing
topic: "baselines"
title: "No code for any of the 5 generative baselines in Tables 1 & 2"
severity: high
confidence: high
status: finding
file: utils/evaluation/TSG_evaluation.py
line_start: 10
line_end: 53
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
claim: "The evaluation harness only scores one (original, generated) pair; neither repo contains code to generate or run the five baselines (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) whose numbers populate Tables 1 and 2. A grep for these names across all .py files returns nothing."
concern: "The paper's headline claim that SDForger 'outperforms existing generative models' and the per-baseline numbers in Tables 1-2 cannot be reproduced or checked from the provided code."
resolution: "Authors: release the baseline-generation scripts (or the generated baseline outputs and the exact hyperparameters/commits used), so the comparison can be reproduced under the same split and metrics."
cross_refs: ["§5", "Table 1", "Table 2"]
paper_ref: "Section 5 'Results'; Tables 1 and 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: knn-081-accuracy-no-code
category: missing
topic: "result traceability"
title: "Section 6 'accuracy of 0.81' kNN classifier has no code"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  For instance, using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2).
claim: "The conditional-generation notebook (notebook/conditional_generation.ipynb) only produces the Figure 2 plot of original vs synthetic curves; no script in either repo trains the longitudinal kNN classifier or computes the 0.81 accuracy. A grep for 'knn', 'neighbor', 'classifier', 'accuracy', 'skfda', 'Ramos' across all .py files returns nothing."
concern: "The quantitative claim backing the multimodal/text-conditioning experiment (0.81 accuracy) cannot be traced to any computation in the repo."
resolution: "Authors: provide the script that trains the longitudinal kNN classifier and computes the 0.81 identification accuracy."
cross_refs: ["knn-081-accuracy-no-code"]
paper_ref: "Section 6 'Shaping time series with language'; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone bugs (crashes / wrong-axis / off-by-one independent of method)
were confirmed. The SHR-nondeterminism and ED/DTW-pairing issues are routed to
`methodology` because the code runs and does what it intends; the problem is
that what it intends is not a sound/reproducible measurement.

## difference

```yaml finding
id: mase-denominator-horizon-not-train
category: difference
topic: "evaluation metrics"
title: "MASE scaled by forecast-horizon naive error, not training-set naive error"
severity: low
confidence: high
status: finding
file: utils/evaluation/utils_ttm.py
line_start: 247
line_end: 251
quote: |
  def compute_mase(true, pred):
      """Compute MASE (Mean Absolute Scaled Error)"""
      numerator = np.mean(np.abs(true - pred))
      denominator = np.mean(np.abs(true[1:] - true[:-1]))  # Naive one-step ahead forecast
      return numerator / denominator if denominator != 0 else np.nan
claim: "MASE here divides the mean absolute forecast error by the mean absolute first difference of the *future (test) values being forecast*, rather than by the in-sample one-step naive error of the training series as in the standard MASE definition (Hyndman & Koehler 2006)."
concern: "The denominator is computed from the same window being scored, so the reported 'MASE' is a non-standard scaling; it is applied uniformly to all methods so within-table rankings are still internally comparable, but the absolute MASE values are not the conventional metric."
resolution: "Authors: confirm whether MASE was intended to scale by the training-series naive error; if the horizon-based scaling is intended, label it as a scaled MAE / custom MASE."
cross_refs: ["Table 2"]
paper_ref: "Table 2 (MASE columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: shr-unseeded-nondeterministic
category: methodology
topic: "statistical integrity / reproducibility"
title: "SHAP-RE similarity metric is unseeded and nondeterministic"
severity: medium
confidence: high
status: finding
file: utils/evaluation/shapelet_based_measures.py
line_start: 22
line_end: 37
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
      A_test, Offsets_test, F_all_1 = update_A_par(test_data, S, A_test, Offsets_test, lambdas, maxIter, epsilon)
claim: "The SHAP-RE (SHR) metric initializes the SIDL dictionary (S, A, Offsets) and the test sparse-coding (A_rand_init, Offsets_test) with np.random.* and is never seeded; sources/run_TSG_evaluation.py reads SEED but never calls set_seed/np.random.seed before tsg_evaluation. Running the repo's own calculate_shapelet_recons_err three times on identical inputs returned 8.1048, 7.9609, 8.0526."
concern: "The SHR column in Table 1 — and the distance-based normalized average and average-rank that include it — depend on uncontrolled RNG state, so the reported SHR values (and any 'SDForger best on SHR' / ranking conclusions drawn from them) are not reproducible."
resolution: "Authors: seed the RNG (or average over multiple seeds with a reported variance) before computing SHAP-RE, and report the spread; confirm which seed produced the Table 1 SHR values."
cross_refs: ["Table 1"]
check_script: _audit_code/check_shr_nondeterminism.py
paper_ref: "Table 1 (SHR column); Appendix B.2 'Shapelet-based Reconstructions'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ed-dtw-pairing
category: methodology
topic: "evaluation metrics"
title: "ED/DTW pair samples by arbitrary index and use only 30 of 100 generated"
severity: medium
confidence: medium
status: finding
file: utils/evaluation/distance_based_measures.py
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
claim: "ED and DTW compute distance between original sample i and generated sample i by positional index, looping only over ori_data.shape[0]. Per the paper's protocol (Appendix C.2: I=30 training instances, 100 synthetic generated; config min=max=100), this consumes only the first 30 of the 100 generated samples and pairs each original window with an arbitrary generated sample, even though SDForger's generated samples (sdforger_augmentation.py) have no per-sample correspondence to the originals."
concern: "The paper's ED definition pairs 'each original series and its generated' counterpart, but the code's index pairing is arbitrary and discards 70% of the generated set, so ED/DTW do not measure per-sample reconstruction fidelity and depend on the (meaningless) ordering of the generated array; _audit_code/check_ed_dtw_pairing.py shows ED uses 30/100 samples and varies across random generated-orderings."
resolution: "Authors: clarify how original and generated samples are matched for ED/DTW (e.g. nearest-neighbour matching as in TSGBench), why only 30 of 100 generated samples are scored, and whether the reported values are order-invariant."
cross_refs: ["Table 1", "shr-unseeded-nondeterministic"]
check_script: _audit_code/check_ed_dtw_pairing.py
paper_ref: "Appendix B.2 'Euclidean Distance' / 'Dynamic Time Warping'; Table 1 ED/DTW columns"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 2          | high         | No baseline code (Tables 1-2); no kNN-0.81 code (§6)        |
| bug         | 0          | -            | None confirmed                                             |
| difference  | 1          | low          | MASE scaled by horizon naive error, not training naive     |
| methodology | 2          | medium       | SHR unseeded/nondeterministic; ED/DTW arbitrary pairing    |

## Top take-aways

1. (missing, high) No code generates or runs the 5 baselines that the
   "outperforms existing generative models" claim and Tables 1-2 rest on
   (`baselines-no-code`).
2. (methodology, medium) The SHR similarity metric is unseeded and returns
   different values on identical inputs (8.10/7.96/8.05), so the Table 1 SHR
   column and the distance-norm-average/rank that include it are not
   reproducible (`shr-unseeded-nondeterministic`).
3. (methodology, medium) ED/DTW pair original and generated samples by arbitrary
   index and score only 30 of 100 generated samples, so they do not measure the
   per-sample fidelity the paper's formula describes (`ed-dtw-pairing`).
4. (missing, medium) The §6 "accuracy of 0.81" kNN identification number has no
   computing code in the repo (`knn-081-accuracy-no-code`).
5. (difference, low) MASE is scaled by the forecast-horizon naive error rather
   than the standard training-set naive error (`mase-denominator-horizon-not-train`).

## Items that genuinely look fine

- All 12 (15 CSV) datasets are present under `data/`, and FastICA/FPC embedding,
  period-aware segmentation, generation, and the SDForger fine-tune pipeline are
  fully implemented (`utils/augmentation/`, `sources/run_data_augmentation.py`).
- Generation is seeded (`utils/generals.py:set_seed` called in
  `run_data_augmentation.py:73,124`); the seed gap is specific to the SHR
  evaluation metric, not the data generation.
- MDD/ACD/SD/KD feature-based metrics and the TTM utility harness
  (RMSE/WQL/H1, train/val/test split logic) are present and deterministic given
  a fixed model.
- Dependencies are pinned in `sdforgerpy310{cuda,mps}.yaml`; the external
  `granite-tsfm` dependency for TTM is documented in the README.

## Open questions for the authors

- Which seed produced the Table 1 SHR values, and what is the run-to-run spread?
- How are original/generated samples matched for ED/DTW, and why are only 30 of
  100 generated samples used?
- Where are the baseline-generation scripts (or the generated baseline outputs)
  used for Tables 1 and 2?
