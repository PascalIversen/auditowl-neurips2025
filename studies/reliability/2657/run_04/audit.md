# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (paper 2657)

## 1. Summary

The repository `code/ZhongLIFR__TCCM-NIPS/` is the official author code for the NeurIPS 2025
paper (README cites the exact title, OpenReview id `jDYuadVajk`, arXiv 2510.18328, and the same
author list as `metadata.txt`; owner `ZhongLIFR` matches first author *Zhong Li*). TCCM is a
flow-matching-inspired, semi-supervised tabular anomaly detector: it trains an MLP `f_theta(x,t)`
to predict the contraction vector `-x` (Eq. 4) and scores test points by the residual norm
`||f_theta([x;Embed(1)]) + x||_2` (Eq. 5). The repo contains the method (`FMAD/`), 30+ baselines
(`baselines/` + PyOD), the 47 ADBench `.npz` datasets, driver scripts, and aggregation/plot code.

What I did:
- Read the paper (`paper.pdf` via `paper_text.txt`) for all headline numbers, the protocol
  (Appendix B.3/B.5/B.6), the ablations (Appendix D.3), robustness/interpretability (D.4),
  and statistical tests (D.5).
- Read every Python file at the repo root, `FMAD/`, and `utils.py`; skimmed `baselines/`.
- Verified the split/scaling logic (`utils.py:load_adbench_npz`) is leakage-free
  (StandardScaler fit on train only, applied to test).
- Wrote two deterministic checks in `_audit_code/`:
  - `check_csm_and_epochs.py` → confirms **no CSM / epoch-selection code** exists and that the
    per-dataset epochs are hardcoded across 47 branches (values 1…5000).
    Output: `_audit_code/out/csm_and_epochs.json`.
  - `check_epoch_coverage.py` → confirms all 47 dataset names resolve to a branch
    (no `UnboundLocalError` for the shipped datasets).
    Output: `_audit_code/out/epoch_coverage.json`.
- Grepped the whole repo for the producing code of every appendix figure / statistical test.

Headline conclusions: the **main accuracy and efficiency results (Figs 2, 3; Tables 6–13) and
the contamination study (Fig 15) are fully traceable and the core method is methodologically
sound** (clean split, train-only scaling, deterministic single-pass score). The defects are
about *completeness*: the paper's described **unsupervised epoch-selection (CSM) procedure is
absent** (epochs are hardcoded with no provenance), and the **explainability** claim (title's
"Explainable"), several appendix ablations (Figs 16/17/18), the theory-validation figures
(Figs 1, 6, representation-collapse §C.3), and the **statistical-significance tests (Friedman /
Nemenyi, Figs 21/22)** have **no code in the repo**.

## 2. Result-traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Fig 2a/2b: AUPRC/AUROC rank box plots (45 detectors × 47 datasets × 5 seeds) | `FullExperiments.py:310-312` (AUC/PR per run) + `AggregateResults.py:206-272` (ranks) | Yes | n/a (rank plot) | Verified |
| Tables 6–13: full AUROC/AUPRC tables | `AggregateResults.py:173,269` (per-seed/aggregate CSVs) | Yes | n/a | Verified |
| Fig 3a/3b/3c: mean inference/training/total time, large datasets | `FullExperiments.py:173,184` (train/test time) + `AggregateResults.py:225-230` | Yes | n/a | Verified |
| Headline "1573× / 4864× / 85× faster inference" | derived from Fig 3 timing columns (`AggregateResults.py` `Mean_TestTime`) | Yes (from timings) | plausible | Verified (timing pipeline present) |
| "10,575 runs = 45×47×5" | 45 models in `FullExperiments.MODEL_NAMES`, 47 `.npz` files, seeds 0–4 in `bash_files/run_main.sh` | Yes | 45×47×5=10575 ✓ | Verified |
| Inductive-setting results (Appendix E) | `bash_files/run_semisupervise.sh` + `force_inductive_models` | Yes | n/a | Verified |
| Fig 12: time-embedding ablation | `AblationStudies.py:255-350` | Yes | n/a | Verified |
| Fig 13: sensitivity to fixed t | `AblationStudies.py:142-212` | Yes | n/a | Verified |
| Fig 14: noise-injection ablation | `AblationStudies.py:360-449` | Yes | n/a | Verified |
| Fig 15a/15b: contamination robustness (TCCM + top-10) | `ContaminationStudies.py` + `ProcessContamination.py` | Yes | n/a | Verified |
| Robustness PGD attack curves (`combined_FP_*.pdf`, App D.4) | `RobustnessStudy.py:240-315` | Yes | n/a | Verified |
| Epochs per dataset ("only data-dependent hyperparameter", App B.3/B.6) | hardcoded `FMAD/functions.py:73-268` | Hardcoded | — | MISMATCH-PROCESS (see `missing-csm-epoch-selection`) |
| Unsupervised CSM epoch-selection T(f) (App B.6) | (none) | No | — | MISSING |
| Statistical significance: Friedman + Nemenyi tests, Figs 21/22 (App D.5) | (none) | No | — | MISSING |
| Fig 4 + App D.4.2: feature-level explainability (MNIST 1-vs-7; synthetic attribution validation) | (none) | No | — | MISSING |
| Fig 16: standardization ablation (App D.3) | (none) | No | — | MISSING |
| Fig 17: time-weighted / interpolated-input ablation (App D.3) | (none) | No | — | MISSING |
| Fig 18: AE+TimeEmbedding comparison (App D.3) | (none) | No | — | MISSING |
| Fig 1: synthetic ring/two-moons/blobs visualization | (none) | No | — | MISSING (illustrative) |
| Fig 6: GMM mismatch-assumption validation (App C.2/D) | (none; `RobustnessStudy.py` makes GMM data only for PGD) | No | — | MISSING |
| §C.3: representation-collapse empirical verification | (none) | No | — | MISSING |

## 3. Findings

## missing

```yaml finding
id: missing-csm-epoch-selection
category: missing
topic: "hyperparameter selection / result traceability"
title: "Unsupervised CSM epoch-selection procedure is absent; epochs are hardcoded"
severity: high
confidence: high
status: finding
file: FMAD/functions.py
line_start: 73
line_end: 78
quote: |
  def determine_FMAD_hyperparameters(dataset_name_raw):
      dataset_name = dataset_name_raw.lower()
      if "census" in dataset_name:
          epoch_size = 5
          batch_size = 1024
          learning_rate = 0.005
claim: "The number of training epochs — the paper's 'only data-dependent hyperparameter' — is returned from a per-dataset if/elif table of hardcoded constants (47 branches, values 1..5000); no code computes the Improved Contrast Score Margin (CSM) criterion T(f) the paper says was used to choose these epochs."
concern: "Appendix B.6 claims epochs were chosen by an automated unsupervised CSM tuning method (Li et al. 2025b), but that selection code is not in the repo, so the headline TCCM numbers cannot be reproduced from scratch and the per-dataset epoch values have no verifiable provenance."
resolution: "Authors: please release the CSM epoch-selection script, the candidate-epoch search ranges per dataset, and the exact data partition on which T(f) was evaluated."
cross_refs: ["leakage-csm-on-test-scores"]
check_script: _audit_code/check_csm_and_epochs.py
paper_ref: "Appendix B.6 'Unsupervised Epoch Selection Strategy'; B.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-statistical-significance-tests
category: missing
topic: "statistical integrity / result traceability"
title: "No code for the Friedman/Nemenyi tests and critical-difference diagrams (Figs 21, 22)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  We report the results using critical difference diagrams (see Figure 21 and 22). For AUPRC, the
  Nemenyi test indicates that there are no statistically significant differences among the top-performing
  group, which includes TCCM (ranked 5.8), DTE-NonParametric, LUNAR, KDE, AutoEncoder, ICL,
claim: "The paper performs Friedman + Nemenyi post-hoc significance testing and reports critical-difference diagrams (Figs 21/22) for its main ranking conclusions (App D.5), but a repo-wide grep finds no Friedman/Nemenyi/critical-difference/post-hoc code in any non-baseline script."
concern: "The statistical-significance result the NeurIPS checklist (item 7) points to as supporting the main conclusion has no producing code, so the significance claim cannot be reproduced from the artefact."
resolution: "Authors: please provide the script that runs the Friedman/Nemenyi tests and renders the critical-difference diagrams from the aggregated rank CSVs."
cross_refs: []
paper_ref: "Appendix D.5; Figures 21, 22"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-explainability-code
category: missing
topic: "explainability"
title: "Feature-level explainability (Fig 4 MNIST; App D.4.2 attribution validation) has no code"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  As shown in Figure 4, the model highlights the
  additional horizontal stroke that distinguishes 7 from 1, demonstrating that the learned importance
  scores align well with human-interpretable cues.
claim: "Explainability is one of the five headline claims (the title's 'Explainable'); the paper shows a MNIST 1-vs-7 feature-attribution figure (Fig 4) and a controlled synthetic attribution-validation study (App D.4.2), but no script in the repo computes or visualizes per-feature residual attributions, the MNIST setup, or the synthetic validation."
concern: "A central contribution of the paper has no reproducible artefact: nothing in the code produces the feature-wise importance maps or quantitatively validates them, so the explainability claim is unverifiable from the repo."
resolution: "Authors: please add the explainability scripts (MNIST 1-vs-7 attribution figure and the App D.4.2 synthetic attribution-accuracy experiment)."
cross_refs: []
paper_ref: "Section 5 'Explainability'; Figure 4; Appendix D.4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-appendix-ablations
category: missing
topic: "ablations"
title: "Ablations for Figs 16 (standardization), 17 (time-weighted sampling), 18 (AE+TE) absent"
severity: low
confidence: high
status: finding
file: AblationStudies.py
line_start: 594
line_end: 609
quote: |
      results = run_time_embedding_ablation()
      plot_time_embedding_ablation(results)
      # ======================================================== #
      # ======================================================== #
      """
      Ablation Study 3: Training with or without Noise
      """
      noise_results = run_noise_ablation(data_configs, seed_list)
      plot_noise_ablation(noise_results)
      # ======================================================== #
      # ======================================================== #
      """
      Ablation Study 4: Contamination Ratio Sensitivity (Per-Dataset Max)
      """
      results, contamination_levels_dict= run_training_contamination_ablation_dynamic_fixed_split()
claim: "AblationStudies.py only produces Figs 12, 13, 14 and the contamination figure; the appendix also reports Fig 16 (effect of z-score standardization), Fig 17 (effect of interpolated time-dependent inputs z_t=tz), and Fig 18 (TCCM vs AE+TimeEmbedding), none of which have code (repo grep for 'standardiz'/'interpolat'/'AE+TE'/Figure_16-18 finds nothing in non-baseline files)."
concern: "Three appendix ablations that justify TCCM design choices (normalization, no time-interpolation, residual learning vs AE) cannot be reproduced because their experiment code is not in the repo."
resolution: "Authors: please add the scripts producing Figures 16, 17, and 18."
cross_refs: []
paper_ref: "Appendix D.3; Figures 16, 17, 18"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-theory-validation-figs
category: missing
topic: "theory validation / result traceability"
title: "No code for Fig 1 (synthetic illustration), Fig 6 (GMM mismatch), or §C.3 collapse check"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  Figure 6 shows boxplots
  of the score distributions for normals and anomalies across d ∈{2, 5, 10, 15, 20}. In all cases, anomalous points exhibit consistently higher scores than normal points, with AUROC values exceeding
  0.9 regardless of dimension.
claim: "Figure 1 (ring/two-moons/blobs vector-field visualization), Figure 6 (boxplots validating the anomaly-mismatch assumption of Props 3-5 over d in {2,5,10,15,20}), and the §C.3 representation-collapse tracking are described as empirical validations, but no script produces them (RobustnessStudy.py generates GMM data only for the PGD attack curves, not these figures)."
concern: "The empirical evidence backing the theoretical propositions and the conceptual illustration is not reproducible from the repo."
resolution: "Authors: please add the scripts for the synthetic illustration (Fig 1), the mismatch-validation boxplots (Fig 6), and the collapse analysis (§C.3)."
cross_refs: []
paper_ref: "Figure 1; Figure 6; Appendix C.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: run-knn-wrong-filename
category: bug
topic: "driver scripts"
title: "run_knn.sh invokes non-existent Full_experiments.py (actual file FullExperiments.py)"
severity: low
confidence: high
status: finding
file: bash_files/run_knn.sh
line_start: 27
line_end: 28
quote: |
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
claim: "run_knn.sh calls `python Full_experiments.py` (underscore + lowercase 'e'), but the only matching file at the repo root is `FullExperiments.py`; the four invocations in this script will fail with 'can't open file'."
concern: "The KNN driver script crashes on every dataset because of the filename typo; it is not used by the README's main roadmap, so impact is limited, but the script is non-functional as shipped."
resolution: "Rename the invocation to `FullExperiments.py` (the other bash scripts already use the correct name)."
cross_refs: []
paper_ref: "n/a (repo driver script)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: epoch-hparam-no-default-branch
category: bug
topic: "hyperparameter selection"
title: "determine_FMAD_hyperparameters has no else branch; unknown dataset -> undefined locals"
severity: low
confidence: high
status: finding
file: FMAD/functions.py
line_start: 259
line_end: 268
quote: |
      elif "hepatitis" in dataset_name:
          epoch_size = 1
          batch_size = 512
          learning_rate = 0.005

      return {
          "epochs": epoch_size,
          "learning_rate": learning_rate,
          "batch_size": batch_size
              }
claim: "The function is a long if/elif chain with no final `else`; if a dataset name matches no branch, `epoch_size`/`batch_size`/`learning_rate` are never assigned and the `return` raises UnboundLocalError. All 47 shipped datasets do resolve (verified), so it does not affect the paper, but any new/renamed dataset crashes."
concern: "A reviewer running TCCM on a dataset whose name is not in the table gets an opaque UnboundLocalError instead of a sensible default; the substring matching is also collision-prone for future dataset names."
resolution: "Add an `else` branch with default epochs/batch/lr (and prefer exact-name matching over substring `in`)."
cross_refs: ["missing-csm-epoch-selection"]
check_script: _audit_code/check_epoch_coverage.py
paper_ref: "Appendix B.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: torch-version-mismatch
category: difference
topic: "dependencies / environment"
title: "requirements.txt pins torch==1.13.1 but paper states PyTorch 2.0"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "requirements.txt pins `torch==1.13.1`, while Appendix B.3 states the implementation 'is based on Python 3.9.21 with PyTorch 2.0'."
concern: "The pinned environment does not match the reported one; for this MLP it is unlikely to change results, but it is a reproducibility-environment inconsistency."
resolution: "Authors: confirm which PyTorch version produced the reported numbers and align requirements.txt with the paper."
cross_refs: []
paper_ref: "Appendix B.3 'Hardware and Software'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: leakage-csm-on-test-scores
category: methodology
topic: "hyperparameter tuning / data leakage"
title: "Epoch-selection criterion needs anomalies, but only test data contains them — possible test tuning"
severity: medium
confidence: low
status: question
file: paper.pdf
quote: |
  where ˆµO, ˆσ2
  O denote the mean and variance of anomaly scores for the top-k predicted anomalies, and
  ˆµI, ˆσ2
  I correspond to the remaining n −k presumed inliers. For each candidate epoch, we compute
  T(f) and select the configuration maximizing this criterion.
claim: "The CSM epoch-selection criterion T(f) is computed over a set partitioned into 'top-k predicted anomalies' vs 'presumed inliers'. In the semi-supervised setup the training set is normal-only (utils.py:34 `X_normal`), so a set containing anomalies is the *test* set; selecting epochs to maximize the anomaly/inlier margin on test outputs would tune the model on the test set."
concern: "If the per-dataset epoch counts (the only data-dependent hyperparameter, which strongly affects scores) were selected by maximizing T(f) on test-set anomaly scores, the main AUROC/AUPRC results would be optimistically biased by test-set tuning; the absence of any held-out validation split in the code (utils.py only makes train/test) makes this impossible to rule out."
resolution: "Authors: state explicitly which data the CSM criterion was evaluated on. If it used the test split, re-select epochs on a held-out subset of training normals (or a labeled validation fold) and report the impact on the headline numbers."
cross_refs: ["missing-csm-epoch-selection"]
paper_ref: "Appendix B.6; utils.py:33-43"
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | CSM epoch-selection, significance tests, explainability, 3 ablations, theory-validation figs all absent |
| bug         | 2          | low          | run_knn.sh filename typo; no default branch in epoch table (47 shipped datasets still resolve) |
| difference  | 1          | low          | requirements pins torch 1.13.1 vs paper's PyTorch 2.0 |
| methodology | 1          | medium       | CSM epoch selection may have been computed on test-set scores (question; main pipeline itself is leakage-free) |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing] `missing-csm-epoch-selection`** (high/high): the paper's unsupervised CSM epoch-selection
   procedure is not in the repo; epochs are hardcoded per dataset (1…5000) with no provenance code,
   so the headline numbers are not reproducible from scratch.
2. **[missing] `missing-statistical-significance-tests`** (med/high): no Friedman/Nemenyi or
   critical-difference code (Figs 21/22) backing the main ranking-significance claim.
3. **[missing] `missing-explainability-code`** (med/high): the title-level "Explainable" claim
   (Fig 4 MNIST + App D.4.2 attribution validation) has no producing code.
4. **[methodology] `leakage-csm-on-test-scores`** (med/low, question): the described epoch-selection
   criterion requires anomalies, which only exist in the test split; possible test-set tuning that
   cannot be ruled out because no validation split exists in the code.
5. **[missing] `missing-appendix-ablations`** (low/high): Figs 16/17/18 ablations absent.
6. **[bug] `run-knn-wrong-filename`** (low/high): run_knn.sh calls a non-existent file.

### Items that genuinely look fine
- **Train/test split + scaling are leakage-free**: `utils.py:37-48` fits StandardScaler on
  X_train only and applies it to X_test; the stratified 50/50 normal split uses `random_state=seed`
  so each of the 5 seeds is a genuinely distinct run (matches paper §5.1).
- **Anomaly score matches Eq. 5 exactly**: `FMAD/FlowMatchingAD.py:48-50` sets t=1 and computes
  `torch.norm(f_xt + X, dim=1)` (per-sample, correct axis).
- **Main accuracy/efficiency pipeline is traceable**: `FullExperiments.py` computes per-run AUROC/AUPRC
  and train/test timings; `AggregateResults.py` aggregates to the ranks/timings behind Figs 2 & 3.
- **10,575-runs arithmetic checks out** (45 detectors × 47 datasets × 5 seeds).
- **Datasets are shipped** (47 `.npz` files present) and all resolve in the epoch table.
- **Baselines use documented default configs via PyOD / source repos** (Appendix B.3), and a
  force-inductive variant is provided for a fair semi-supervised comparison.

### Open questions for the authors
- On which data partition was the CSM epoch-selection criterion T(f) evaluated — a held-out subset
  of training normals, a labeled validation fold, or the test set? (`leakage-csm-on-test-scores`)
- Will the CSM selection script, candidate-epoch ranges, and the significance-test / explainability /
  appendix-ablation scripts be released? (`missing-*`)
- Which PyTorch version produced the reported numbers — 1.13.1 (requirements) or 2.0 (paper)?
