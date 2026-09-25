# Audit — Forging Time Series with Language (SDForger), paper #2371

## 1. Summary

SDForger transforms time series into compact functional (FastICA / PCA) embeddings,
encodes them as text, fine-tunes an autoregressive LLM (GPT-2 in the paper), and
samples new textual embeddings that are decoded back into synthetic series. The
paper claims SDForger "outperforms existing generative models in many scenarios,
both in similarity-based evaluations and downstream forecasting tasks" (abstract).

Two author artefacts are provided:
- `code/IBM__fms-dgt/` — the public open-source release containing the SDForger
  *generation* databuilder (`fms_dgt/public/databuilders/time_series/`). This is
  the generation method only.
- `code/SDForger__neurips_supplemental/` — the NeurIPS supplemental reproduction
  package: generation (`utils/augmentation/`), the similarity metrics
  (`utils/evaluation/`: MDD/ACD/SD/KD/ED/DTW/SHAP-RE) and the TTM utility
  evaluation (`utils/evaluation/utils_ttm.py`), driver scripts under `sources/`,
  config files, and 15 dataset CSVs under `data/`.

What I did. I read the evaluation entry points (`sources/run_TSG_evaluation.py`,
`sources/run_TTM_evaluation.py`, `sources/run_data_augmentation.py`), the metric
implementations, and the preprocessing/splitting code, and cross-referenced them
against the paper's Section 4 (evaluation methodology), Appendix B (metric
definitions), Tables 1–2, and Section 6 (text conditioning). I ran two read-only
checks under `_audit_code/`: (a) a reproduction of the repo's `calculate_ed` loop
to confirm that ED/DTW pair original and generated curves by array index and are
order-dependent; (b) a call to the repo's actual `calculate_shapelet_recons_err`
five times on identical inputs to quantify the non-determinism of the SHAP-RE
metric (the evaluation entry point never seeds the RNG).

Findings: the similarity metrics ED and DTW pair synthetic and real curves by
arbitrary array position even though synthetic curves are unconditioned samples
(methodology); the SHAP-RE metric is computed with an unseeded RNG and varies
~13% run-to-run (methodology); and the code for all five baselines
(TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4) that every headline comparison in Tables 1–2
rests on, plus the k-NN classifier behind the 0.81 accuracy in Section 6, is
absent from both repos (missing).

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1, SDForger MDD/ACD/SD/KD (feature-based) | `utils/evaluation/feature_based_measures.py` (via `run_TSG_evaluation.py`) | recomputable (re-run req.) | not run (needs GPT-2 generation) | Present (code exists) |
| Table 1, SDForger ED / DTW (distance-based) | `utils/evaluation/distance_based_measures.py` | recomputable, but index-paired (see `ed-dtw-positional-pairing`) | n/a | Present but metric questionable |
| Table 1, SDForger SHR (SHAP-RE) | `utils/evaluation/shapelet_based_measures.py` | non-deterministic (see `shapre-unseeded-nondeterministic`) | n/a | Present but non-reproducible |
| Table 1, BASELINE columns (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) | (none) | — | — | MISSING (no baseline code) |
| Table 2, SDForger RMSE/MASE/WQL | `utils/evaluation/utils_ttm.py` (via `run_TTM_evaluation.py`) | recomputable (re-run req.) | not run | Present (code exists) |
| Table 2, BASELINE rows (GAN-based etc.) | (none) | — | — | MISSING (no baseline code) |
| §6 / Fig. 2: k-NN classifier accuracy = 0.81 | (none — not in `sources/`, `utils/`, or notebook) | — | — | MISSING (no classifier script) |
| Appendix D ablations (k, LLM choice, timing, rejection rates) | partially via configs (k, llm are params); no driver aggregating the ablation tables | — | — | Partial / not aggregated |

Notes on "recomputable (re-run req.)": no generated outputs, fine-tuned weights, or
results CSVs are committed, so reproducing the exact SDForger numbers requires
re-running nondeterministic GPT-2 fine-tuning and generation. Dependencies are
pinned (`sdforgerpy310cuda.yaml` / `sdforgerpy310mps.yaml`), so the environment is
reproducible; this is expected for a generative-model paper and is not filed as a
standalone finding.

## 3. Findings

## missing

```yaml finding
id: baseline-code-absent
category: missing
topic: "baselines / result traceability"
title: "No code for any of the 5 baselines underpinning Tables 1 and 2"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  and a diffusion-based model: LS4 (Zhou et al., 2023), which
  generates sequences via a learned reverse-time diffusion process. Hyperparameters for all baseline
  competitors follow those reported in their original papers
claim: "Section 4 'Baselines' lists five baselines (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) that fill every comparison column of Tables 1 and 2; a case-insensitive search of both author repos (code/IBM__fms-dgt and code/SDForger__neurips_supplemental) finds no implementation, config, or invocation of any of these five baselines, and the supplemental contains no baseline training/generation/eval scripts."
concern: "The headline claim 'outperforms state-of-the-art baselines' (abstract, conclusions) cannot be reproduced because none of the baseline numbers in Tables 1–2 can be regenerated from the released code, and the reproducibility checklist answers 'Yes' to providing code to faithfully reproduce the main results."
resolution: "Authors: release the baseline training/generation/evaluation scripts (or exact forks + commit hashes and hyperparameter configs) used to produce the TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4 columns of Tables 1 and 2."
cross_refs: ["§4 Baselines", "Table 1", "Table 2", "ed-dtw-positional-pairing"]
paper_ref: "Section 4 'Baselines'; Tables 1 and 2; reproducibility checklist Q on code access"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: knn-accuracy-script-missing
category: missing
topic: "result traceability"
title: "No script computes the 0.81 k-NN accuracy reported in Section 6 / Figure 2"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  For instance, using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2).
claim: "Section 6 reports a longitudinal k-NN classifier accuracy of 0.81; no script or notebook in either repo imports a k-NN classifier (e.g. skfda/KNeighbors), computes an accuracy_score, or otherwise produces this number — the conditional_generation.ipynb notebook only visualizes channels."
concern: "A reported quantitative result (0.81 accuracy) supporting the text-conditioning claim has no computing artefact in the repo, so it cannot be reproduced or checked."
resolution: "Authors: add the classifier-evaluation script (data split, classifier config, label construction) that produces the 0.81 accuracy in Figure 2."
cross_refs: ["§6", "Figure 2"]
paper_ref: "Section 6 'Shaping time series with language'; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone `bug` findings. The evaluation and generation scripts parse their
YAML configs and run via the documented commands; the defects found are
methodological (the procedures the code intends to run), not breakages of the
code's own intent.

## difference

No `difference` findings. Where the code and paper could be compared (ED/DTW
definition in Appendix B, the temporal train/val/test split, the GPT-2 / k / lr
hyperparameters in the configs), the code matches the paper's description; the
concerns are about validity of the described-and-implemented procedure, which
routes to `methodology`.

## methodology

```yaml finding
id: ed-dtw-positional-pairing
category: methodology
topic: "evaluation metrics"
title: "ED and DTW pair synthetic and real curves by array index, not as distributions"
severity: medium
confidence: high
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
claim: "ED and DTW compare the i-th original window with the i-th generated curve by array position and average over i. The synthetic curves are independent samples from the fine-tuned LLM (sdforger_augmentation.py: model.generate(...)), not conditioned on or aligned with any specific original window, so the pairing is arbitrary; with the univariate config (30 train windows, 100 generations) the loop also silently uses only the first 30 generated curves and ignores the other 70."
concern: "A distance between two unordered sets of curves must be invariant to the ordering of the generated samples and must use all generated samples; this metric is neither, so the ED/DTW 'similarity' numbers in Table 1 depend on the arbitrary order/index of the generated array rather than measuring set-level similarity."
resolution: "Authors: clarify how synthetic curves are matched to real ones for ED/DTW (nearest-neighbour / optimal assignment / per-sample identity?) and confirm whether all generated curves are used; if matched by index, justify why that is meaningful for unconditioned samples."
cross_refs: ["Table 1", "Appendix B.2"]
check_script: _audit_code/check_ed_positional_pairing.py
paper_ref: "Appendix B.2 'Euclidean Distance' / 'Dynamic Time Warping'; Table 1 ED/DTW columns"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: shapre-unseeded-nondeterministic
category: methodology
topic: "evaluation metrics / reproducibility"
title: "SHAP-RE metric uses an unseeded RNG; varies ~13% run-to-run on identical inputs"
severity: medium
confidence: high
status: finding
file: utils/evaluation/shapelet_based_measures.py
line_start: 6
line_end: 39
quote: |
  def calculate_shapelet_recons_err(orig_data, gen_data, K=20, lambdas =0.1, r=0.25):

      c = 100
      epsilon = 1e-5
      maxIter = 1e3
      maxInnerIter = 5

      # Reverse the transpose operation
      # orig_data = orig_data.transpose(2, 0, 1)
      # gen_data = gen_data.transpose(2, 0, 1)
      orig_data = orig_data
      gen_data = gen_data


      train_data  = orig_data.reshape(orig_data.shape[0], orig_data.shape[1])
      test_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1])
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

      test_recons_error_sidl = unsup_obj(test_data, S, A_test, Offsets_test, 0) / n_test
claim: "calculate_shapelet_recons_err uses np.random (A_rand_init, Offsets_test, plus S/A/Offsets in USIDL and np.random.permutation inside update_A_par) without any seeding, and sources/run_TSG_evaluation.py never calls set_seed; calling the repo's own function five times on identical inputs (check_shapre_nondeterminism.py) gives SHAP-RE values [1.62, 1.75, 1.81, 1.72, 1.59] with relative spread ~0.13."
concern: "The SHR column of Table 1 is a single reported number per model, but the metric is non-deterministic with ~13% run-to-run variation on fixed inputs, so the reported SHR values (and the resulting model ranking on that metric) are not reproducible and could change the comparison."
resolution: "Authors: seed all RNGs before the SHAP-RE computation (or average over multiple seeds and report variance), and state how many SIDL restarts / which seed produced the reported SHR values."
cross_refs: ["Table 1", "Appendix B.2"]
check_script: _audit_code/check_shapre_nondeterminism.py
paper_ref: "Appendix B.2 'Shapelet-based Reconstructions'; Table 1 SHR column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 2 | high | No baseline code for Tables 1–2; no script for the 0.81 k-NN accuracy. |
| bug | 0 | - | Scripts run as intended; no defect contradicting the code's own intent. |
| difference | 0 | - | Where comparable, code matches the paper's described procedure. |
| methodology | 2 | medium | ED/DTW index-pair unordered samples; SHAP-RE is unseeded / non-deterministic. |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing] baseline-code-absent** (high/high) — none of the five baselines
   (TimeVAE, TimeVQVAE, RTSGAN, SDEGAN, LS4) that every comparison in Tables 1–2
   relies on has code in either repo, so the headline "outperforms state-of-the-art
   baselines" claim is not reproducible.
2. **[methodology] shapre-unseeded-nondeterministic** (medium/high) — the SHAP-RE
   metric is computed with an unseeded RNG and the entry point never seeds; it
   varies ~13% run-to-run on identical inputs, making the reported SHR column and
   its ranking non-reproducible.
3. **[methodology] ed-dtw-positional-pairing** (medium/high) — ED and DTW pair the
   i-th real window with the i-th (arbitrary) generated curve and ignore unmatched
   generations, so these "similarity" numbers depend on array ordering rather than
   measuring set-level similarity.
4. **[missing] knn-accuracy-script-missing** (medium/high) — the 0.81 longitudinal
   k-NN accuracy in Section 6 / Figure 2 has no computing script in either repo.

### Items that genuinely look fine
- **Temporal train/val/test split (utility eval).** `preprocess_uni_multi_variate_data`
  (utils/augmentation/utils_preprocess_data.py:323-395) carves train, then val,
  then test as consecutive, ordered slices (`get_data(start, end_train)`,
  `(end_train, end_val)`, `(end_val, end_test)`), respecting temporal order; the
  StandardScaler is `fit` on train and only `transform`-ed on val/test
  (standardscale_train_val_test, lines 223-258) — no scaler leakage.
- **Dependency specification.** Both `sdforgerpy310cuda.yaml` and
  `sdforgerpy310mps.yaml` pin exact versions (e.g. `dtaidistance==2.3.13`,
  `datasets==3.5.0`), so the environment is reproducible.
- **Utility metrics applied uniformly.** RMSE/MASE/WQL/H1 in utils_ttm.py are the
  same functions for all four training sources (zero-shot, real, synthetic,
  combined), so even the slightly unusual WQL-on-a-point-forecast does not bias the
  Table 2 comparison.
- **Repo provenance.** `code/IBM__fms-dgt/fms_dgt/public/databuilders/time_series/`
  and `code/SDForger__neurips_supplemental/` both correspond to this paper
  (SDForger generation + evaluation), matching `code_links.txt`.

### Open questions for the authors
- For ED/DTW: is index pairing intentional, and were baselines scored identically
  (so the ranking is still fair even if the absolute metric is order-dependent)?
- Which seed / how many SIDL restarts produced the reported SHR values, and were
  baselines' SHR computed under the same protocol?
- Were the baseline scripts simply omitted from the supplemental, or were baseline
  numbers taken from the original baseline papers / a separate codebase?
