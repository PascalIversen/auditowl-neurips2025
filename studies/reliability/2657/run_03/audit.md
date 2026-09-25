# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (NeurIPS 2025, paper 2657)

## 1. Summary

The repository `code/ZhongLIFR__TCCM-NIPS/` is the **official author code** (README header,
arXiv 2510.18328, OpenReview `jDYuadVajk`, author Zhong Li = GitHub `ZhongLIFR`; matches the
paper). TCCM is a one-step flow-matching anomaly detector for tabular data, evaluated on the
ADBench suite (47 datasets, 45 detectors). The core method is `FMAD/FlowMatchingAD.py`
(class `TCCM`) + `FMAD/functions.py` (the `FlowMatching` MLP, sinusoidal time embedding, and
per-dataset hyperparameter table). Drivers: `FullExperiments.py` (main AUROC/AUPRC benchmark,
Figure 2), `AggregateResults.py` (ranking aggregation), `AblationStudies.py` (Appendix D.3
ablations), `RobustnessStudy.py` (Appendix D.4 PGD evasion attacks on a synthetic GMM),
`ContaminationStudies.py` + `ProcessContamination.py` (contaminated-training robustness). All 47
ADBench `.npz` datasets are bundled under `datasets/`.

What I ran (read-only, under `_audit_code/`):
- `check_dataset_files.py` — all 47 datasets used by the benchmark are present (47/47).
- `check_hyperparam_coverage.py` — every one of the 47 dataset names is matched by a branch of
  `determine_FMAD_hyperparameters`, so no `UnboundLocalError` fall-through occurs in practice (47/47).
- `check_first_match.py` — verified the if/elif substring chain routes each dataset to the
  intended branch (e.g. `cardio`→2000 vs `cardiotocography`→1; `annthyroid`→2000 vs `thyroid`→10);
  no mis-routing.
- `check_missing_experiments.py` — whole-repo keyword scan: the explainability/feature-attribution
  validation (Table 3), the statistical tests (Appendix D.5 Friedman/Nemenyi/CD), and the CSM
  unsupervised epoch-selection procedure (Appendix B.3/B.6) have **zero** implementing code.

Headline data computations (AUROC, AUPRC, train/test time → Figures 2 and 3) ARE present and
traceable through `FullExperiments.py` → `AggregateResults.py`. The principal gaps are
**missing code** for three paper artefacts and a description↔code mismatch around how the only
data-dependent hyperparameter (training epochs) was selected.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2 AUROC/AUPRC rank box plots over 47 datasets, 45 detectors | `FullExperiments.py:310-312` (per-run AUROC/AUPRC) + `AggregateResults.py:220-239,270-273` (mean + ranks) | values computed; per-detector mean ranks → `Results_*_rank.csv` | n/a (not run) | Data-computed; box-plot PDF render not in code (see `readme-pdf-output-not-produced`) |
| Fig. 3a mean inference time + "Avg. Slowdown ×1573/×4864/×85" | `FullExperiments.py:179-184` (test_time) + `AggregateResults.py:229-230` (Mean_TestTime) | test-time recorded & aggregated; slowdown ratio derived | n/a | Verified (computation present) |
| Table 5 per-dataset #Epochs (only data-dependent hyperparam) | `FMAD/functions.py:73-268` (`determine_FMAD_hyperparameters`, hardcoded) | hardcoded constants; match Table 5 | ✓ (values match) | The hardcoded VALUES match Table 5, but the **selection procedure** that produced them (CSM, Li et al. 2025b) is absent → see `epoch-selection-csm-code-missing` |
| Table 3 explainability: ExactMatch / Jaccard / AUROC / AUPRC on synthetic GMM | (none) | — | — | MISSING (no feature-attribution / top-k / ExactMatch / Jaccard code anywhere) |
| Appendix D.5 statistical tests: Friedman + Nemenyi critical-difference | (none) | — | — | MISSING (no friedman/nemenyi/CD code anywhere) |
| Appendix D.3 ablations (Figs 12-15): time-embedding, fixed-t, noise, contamination | `AblationStudies.py` | computed | n/a | Verified (computation present) |
| Appendix D.4 empirical robustness (PGD evasion, `combined_FP_*.pdf`) | `RobustnessStudy.py` | computed | n/a | Verified (computation present) |
| Contamination robustness (top-10 models) | `ContaminationStudies.py` + `ProcessContamination.py` | computed | n/a | Computed; mild preprocessing-order concern → `contam-rescale-on-prescaled-data` (question) |
| Main benchmark train/test split (normal-only train, scaler on train) | `utils.py:19-53` | — | — | Verified sound (no scaler leakage; split seed varies per run) |

## 3. Findings

## missing

```yaml finding
id: epoch-selection-csm-code-missing
category: missing
topic: "hyperparameter tuning / reproducibility"
title: "Unsupervised CSM epoch-selection procedure absent; only hardcoded per-dataset epochs remain"
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
claim: "The number of training epochs — described in the paper as TCCM's only data-dependent hyperparameter, selected per dataset by the unsupervised Improved Contrast Score Margin (CSM) criterion of Li et al. (2025b) — is supplied to TCCM purely as a hardcoded per-dataset lookup table; the entire selection procedure is absent from the repo."
concern: "The headline benchmark numbers depend on these epoch values, yet the code that produced them cannot be reproduced or audited, and one cannot verify whether the (unspecified) data used by the CSM criterion touched the test set."
resolution: "Authors: please release the CSM-based epoch-selection script, and state explicitly which data split (train-only vs the test set) the CSM criterion T(f) was evaluated on for each candidate epoch."
cross_refs: ["epoch-values-chosen-ad-hoc", "§B.3", "§B.6", "Table 5"]
check_script: _audit_code/check_missing_experiments.py
paper_ref: "Appendix B.3 (lines 2660-2667) and B.6 'Unsupervised Epoch Selection Strategy' (lines 2760-2788); Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: explainability-validation-code-missing
category: missing
topic: "result traceability / explainability"
title: "Table 3 explainability validation (ExactMatch, Jaccard) has no implementing code"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  "Exact Match: The proportion of anomalies for which the predicted top-k features exactly
  coincide with the ground-truth anomalous dimensions ... Jaccard Index: The average
  intersection-over-union (IoU) between predicted and true anomalous dimensions"
claim: "The paper reports a quantitative explainability experiment (Table 3: near-perfect ExactMatch and Jaccard across d in {5,10,15,20,25}) validating TCCM's feature-level attribution, but no script in the repo computes top-k feature attribution, ExactMatch, or Jaccard."
concern: "Explainability is one of the three headline claims in the title/abstract; the only quantitative evidence for it (Table 3) is not reproducible from the released code."
resolution: "Authors: please release the synthetic-GMM feature-attribution experiment that computes the per-feature residual scores, top-k selection, ExactMatch, and Jaccard reported in Table 3."
cross_refs: ["§D.4.2", "Table 3"]
check_script: _audit_code/check_missing_experiments.py
paper_ref: "Appendix D.4.2 (lines 4719-4793); Table 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: statistical-tests-code-missing
category: missing
topic: "statistical integrity"
title: "Friedman + Nemenyi critical-difference tests (Appendix D.5) not implemented in repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "we apply the Friedman test ... to determine whether there is any statistically significant
  difference in performance rankings among all methods. If the null hypothesis is rejected, we
  proceed with the Nemenyi post hoc test ... Two methods are considered significantly
  different if their average ranks differ by at least the critical difference (CD)."
claim: "The paper conducts Friedman + Nemenyi critical-difference significance tests over the 45-detector / 47-dataset rankings (Appendix D.5), but a whole-repo scan finds no code for friedman, nemenyi, or critical-difference."
concern: "The reported significance of TCCM's ranking advantage cannot be reproduced or checked from the released code."
resolution: "Authors: please release the script that runs the Friedman test and Nemenyi/CD post-hoc analysis on the per-dataset mean ranks."
cross_refs: ["§D.5"]
check_script: _audit_code/check_missing_experiments.py
paper_ref: "Appendix D.5 'Statistical Tests' (lines 4794-4808)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: readme-pdf-output-not-produced
category: bug
topic: "reproduction instructions"
title: "README claims AggregateResults.py emits Rank_ROC.pdf / Rank_PR.pdf, but it only writes CSVs"
severity: low
confidence: high
status: finding
file: AggregateResults.py
line_start: 269
line_end: 273
quote: |
      df_aggregated.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}.csv"))
      order_roc = df_aggregated.groupby("Detector")["Rank_ROC"].mean().sort_values()
      order_roc.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank.csv"))
      order_pr = df_aggregated.groupby("Detector")["Rank_PR"].mean().sort_values()
      order_pr.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank_PR.csv"))
claim: "README (Reproduction Roadmap line 98; 'Result Aggregation' lines 199-204) says `python AggregateResults.py --semi_only` produces `Rank_ROC.pdf` and `Rank_PR.pdf` (the Figure 2 box plots), but the script contains no matplotlib/seaborn import or savefig call and only writes CSV files."
concern: "Following the README does not reproduce the Figure 2 PDFs; the box-plot rendering code for the main figure is not in the repo (the underlying rank values, however, are computed and saved to CSV)."
resolution: "Authors: include the box-plot script that turns the rank CSVs into Rank_ROC.pdf / Rank_PR.pdf, or correct the README to state only CSVs are produced."
cross_refs: ["Figure 2"]
paper_ref: "Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: epoch-values-chosen-ad-hoc
category: difference
topic: "hyperparameter tuning"
title: "Code comments indicate manual epoch choice, contradicting the 'principled CSM-based selection' claim"
severity: medium
confidence: medium
status: finding
file: FMAD/functions.py
line_start: 127
line_end: 134
quote: |
  elif "skin" in dataset_name:
      epoch_size = 110  # Chosen first from "100 or 1"
      batch_size = 1024
      learning_rate = 0.005
  elif "celeba" in dataset_name:
      epoch_size = 2  # Chosen first from "100 or 1"
      batch_size = 1024
      learning_rate = 0.005
claim: "For the `skin` and `celeba` datasets the hardcoded epoch values carry the comment 'Chosen first from \"100 or 1\"', indicating the epoch was picked manually from a couple of candidate values."
concern: "The paper states all epoch values come from a principled unsupervised CSM criterion (Li et al. 2025b); these comments suggest at least some values were chosen ad hoc, a description↔code mismatch about how the headline hyperparameter was set."
resolution: "Authors: clarify whether every Table 5 epoch was selected by the CSM criterion, or whether some were chosen manually (and if so, on what basis)."
cross_refs: ["epoch-selection-csm-code-missing", "Table 5", "§B.6"]
check_script: _audit_code/check_first_match.py
paper_ref: "Appendix B.6 'Hardcoding for Simplicity' (lines 2784-2788); Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: requirements-torch-version-mismatch
category: difference
topic: "dependencies / environment"
title: "requirements.txt pins torch==1.13.1 but the paper states PyTorch 2.0"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "The pinned dependency is torch==1.13.1, whereas the paper's environment description states the implementation is based on PyTorch 2.0."
concern: "A reviewer rebuilding the environment from requirements.txt gets a different major PyTorch version than the one the paper says was used; results could differ subtly."
resolution: "Authors: align requirements.txt with the PyTorch version actually used (state whether 1.13.1 or 2.0 produced the reported numbers)."
cross_refs: ["§B.3"]
paper_ref: "Appendix B.3 'Hardware and Software' (line 2686: 'Our implementation is based on Python 3.9.21 with PyTorch 2.0')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: contam-rescale-on-prescaled-data
category: methodology
topic: "preprocessing / data splitting (contamination study)"
title: "Contamination study re-splits already-scaled data without re-fitting the scaler"
severity: low
confidence: medium
status: question
file: ContaminationStudies.py
line_start: 160
line_end: 172
quote: |
      X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed) # By default, we use 0

      X_all = np.vstack([X_train, X_test])
      y_all = np.concatenate([y_train, y_test])

      X_normal = X_all[y_all == 0]
      X_abnormal = X_all[y_all == 1]

      X_train_normal, X_test_normal = train_test_split(X_normal, test_size=0.5, random_state=42, stratify=None)
      X_train_abnormal_full, X_test_abnormal = train_test_split(X_abnormal, test_size=0.5, random_state=42, stratify=None)

      X_test = np.vstack([X_test_normal, X_test_abnormal])
      y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_abnormal))])
claim: "The contamination study loads data via load_adbench_npz (which StandardScaler-fits on the original train normals and transforms the rest), then re-stacks the already-scaled train+test arrays into X_all and performs a brand-new 50/50 split without re-fitting the scaler on the new training subset."
concern: "The features in the new train/test split were scaled with statistics computed from the original train-normal subset (a different partition), so the contamination-study test points are normalized using statistics partly derived from points that are now in the test set — a mild preprocessing-before-split irregularity (no label leakage)."
resolution: "Authors: confirm whether the scaler should be re-fitted on each contamination-study training subset; quantify whether re-fitting changes the Figure 15 trends."
cross_refs: ["§D.3"]
paper_ref: "Appendix D.3 contamination ablation (Figure 15)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 3          | high         | Epoch-selection (CSM), Table-3 explainability, and D.5 stat-tests code all absent |
| bug         | 1          | low          | README promises Figure-2 PDFs that AggregateResults.py never writes    |
| difference  | 2          | medium       | Ad-hoc epoch comments vs CSM claim; torch 1.13.1 vs paper's PyTorch 2.0 |
| methodology | 1          | low          | Contamination study re-splits already-scaled data (question)           |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing] `epoch-selection-csm-code-missing`** (high/high) — TCCM's only data-dependent
   hyperparameter (#epochs) is hardcoded; the unsupervised CSM selection procedure the paper
   credits for it is entirely absent, so the headline benchmark cannot be fully reproduced and the
   data CSM uses (train vs test) cannot be verified.
2. **[missing] `explainability-validation-code-missing`** (high/high) — Table 3, the only
   quantitative evidence for the "Explainable" headline claim, has no implementing code.
3. **[missing] `statistical-tests-code-missing`** (medium/high) — Friedman/Nemenyi/CD significance
   analysis (Appendix D.5) is not in the repo.
4. **[difference] `epoch-values-chosen-ad-hoc`** (medium/medium) — code comments ('Chosen first
   from "100 or 1"') suggest some epochs were picked manually, contradicting the CSM-selection claim.
5. **[difference] `requirements-torch-version-mismatch`** (low/high) — requirements.txt pins
   torch 1.13.1 while the paper states PyTorch 2.0.
6. **[bug] `readme-pdf-output-not-produced`** (low/high) — README claims AggregateResults.py
   emits the Figure-2 PDFs, but it only writes CSVs.

### Items that genuinely look fine
- **Train/test split & scaling** (`utils.py:19-53`): trains on normal-only data, fits the
  StandardScaler on train and only transforms test — no preprocessing leakage; the split seed
  varies across the 5 runs, so the reported mean±std reflects genuine resampling.
- **Seeding** (`utils.py:9-16`): torch, cuda, numpy and random are all seeded.
- **Dataset availability**: all 47 ADBench datasets used by the benchmark are bundled (47/47).
- **Hyperparameter table coverage**: every dataset name resolves to a defined branch; no
  `UnboundLocalError`, and substring routing (cardio vs cardiotocography, annthyroid vs thyroid)
  is correct.
- **Core method ↔ Algorithm 1/2**: `FlowMatchingAD.py` matches the paper's training loss
  (||f(x,t)+x||², t~U(0,1)) and the one-step inference score (||f(x,1)+x||₂).
- **Baseline configs**: baselines use PyOD/source defaults as the paper states; TCCM and baselines
  share the same split/metric.

### Open questions for the authors
- On what data partition is the CSM criterion T(f) evaluated when selecting epochs — the
  normal-only training set, or a set containing the test anomalies? (If the latter, epoch
  selection touches the test set and would become a `methodology` concern.) See
  `epoch-selection-csm-code-missing`.
- Were all Table-5 epoch values selected by the CSM criterion, or some chosen manually? See
  `epoch-values-chosen-ad-hoc`.
- Should the StandardScaler be re-fit per contamination-study training subset? See
  `contam-rescale-on-prescaled-data`.
