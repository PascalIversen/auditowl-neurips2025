# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (paper 2657)

## 1. Summary

The repository `code/ZhongLIFR__TCCM-NIPS/` is the **official author code** (README cites the
exact NeurIPS 2025 paper, OpenReview id `jDYuadVajk`, and the method "TCCM / Time-Conditioned
Contraction Matching"). It contains the core method (`FMAD/FlowMatchingAD.py`, `FMAD/functions.py`),
the main benchmark driver (`FullExperiments.py`), result aggregation (`AggregateResults.py`),
ablation studies (`AblationStudies.py`), empirical robustness (`RobustnessStudy.py`), contamination
studies (`ContaminationStudies.py` + `ProcessContamination.py`), 13 baseline implementations under
`baselines/`, the 47 ADBench `.npz` datasets, pinned `requirements.txt`, and reproduction bash
scripts under `bash_files/`.

What I did (all read-only; helper scripts under `_audit_code/`):
- Read the paper's method (§3), scoring (Eq. 5), theory (§4, Propositions 1–2), and the experiment
  setup (§5.1 and Appendix B.3/B.4), and matched each against the code.
- Verified the train/test split + scaling logic in `utils.py:load_adbench_npz` (no leakage; scaler
  fit on train only; anomalies test-only).
- Confirmed the TCCM training loss (Eq. 4) and inference score (Eq. 5) are implemented faithfully.
- `_audit_code/check_hyperparam_coverage.py`: confirmed all 47 datasets map to a defined epoch in
  `determine_FMAD_hyperparameters` (no UnboundLocalError; substring ordering is safe).
- `_audit_code/check_missing_artifacts.py`: grepped for code producing each paper figure/result.
  Figures 12–15 (ablations) have code; **Figure 4 (explainability), Figure 16 (feature
  normalization ablation), Figure 17 (time-interpolated inputs ablation), and the
  Autoencoder+Time-Embedding comparison have no code in the repo.**

Overall the core method and main benchmark pipeline are sound and faithful to the paper; the data
split has no leakage and the metrics (AUROC+AUPRC) fit the imbalanced task. The findings concern
(a) absent code for several paper artefacts (explainability figure, two ablations, the unsupervised
epoch-selection method), and (b) the per-dataset hyperparameter tuning applied only to TCCM.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2 AUPRC/AUROC rank box plots (rank 5.8 / 5.7) | `FullExperiments.py` (per-(dataset,model) AUROC/AUPRC) → `AggregateResults.py:206-272` (ranks/means) | Yes | Cannot rerun full grid; computation path present | Verified (path) |
| Per-(dataset,model) AUROC / AUPRC (Appendix tables) | `FullExperiments.py:310-311` (`roc_auc_score`, `average_precision_score`) | Yes | n/a | Verified (path) |
| Fig. 3 mean run-time / "1573× / 85× faster inference" | runtimes recorded `FullExperiments.py:173,184`; aggregated `AggregateResults.py:229-230` (Mean_TestTime). No script computes the speedup *ratio* or plots Fig. 3 | Partial (raw times yes; ratio/plot no) | — | PARTIAL |
| Fig. 4 explainability (MNIST digit '1', AUROC 0.76, feature-level map) | (none) | No | — | MISSING |
| Eq. 4 training loss | `FMAD/FlowMatchingAD.py:30-34` | Yes | ✓ | Verified |
| Eq. 5 anomaly score `‖f(x,1)+x‖₂` | `FMAD/FlowMatchingAD.py:48-50` | Yes | ✓ | Verified |
| Train/test split (50% of normals train; scaler on train) | `utils.py:37-48` | Yes | ✓ | Verified |
| Prop. 2 GMM discriminative setup | `RobustnessStudy.py:33-74` (disjoint normal/anom modes) | Yes | ✓ | Verified |
| Robustness Fig. (combined_FP_True/False) PGD attack | `RobustnessStudy.py:240-315` | Yes | ✓ | Verified |
| Ablation Fig. 12 (time embedding) | `AblationStudies.py:255-350` | Yes | ✓ | Verified |
| Ablation Fig. 13 (sensitivity to t) | `AblationStudies.py:142-212` | Yes | ✓ | Verified |
| Ablation Fig. 14 (noise injection) | `AblationStudies.py:360-449` | Yes | ✓ | Verified |
| Ablation Fig. 15 (training contamination) | `AblationStudies.py:461-580`; full study `ContaminationStudies.py` | Yes | ✓ | Verified |
| Ablation Fig. 16 (feature normalization) | (none) | No | — | MISSING |
| Ablation Fig. 17 (time-interpolated inputs) | (none) | No | — | MISSING |
| Ablation: Autoencoder + Time Embedding comparison | (none) | No | — | MISSING |
| TCCM epochs (Table 5) via "unsupervised selection (Li et al. 2025b)" | hardcoded results only in `FMAD/functions.py:73-268`; selection method not implemented | No (results only) | — | MISSING (procedure) |

## 3. Findings

## missing

```yaml finding
id: explainability-figure-no-code
category: missing
topic: "result traceability / explainability"
title: "No code produces the explainability result (Figure 4, MNIST digit, AUROC 0.76)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  (3) Explainability: TCCM enables feature-level
  attribution for anomaly scores, supporting interpretable diagnosis—an aspect largely absent in
  existing deep anomaly detection models (see Figure 4).
claim: "Explainability is one of the paper's four headline contributions (RQ3); Figure 4 shows feature-level anomaly explanations on MNIST with a reported AUROC of 0.76, but a repo-wide grep for any explanation/visualization/MNIST-image script returns nothing (see _audit_code/out/missing_artifacts.txt)."
concern: "A headline contribution (feature-level explainability) and its figure/number cannot be reproduced because the code that generates them is absent from the repository."
resolution: "Authors: please add the script that produces Figure 4 (the MNIST digit-1 explanation and the 0.76 AUROC) and the feature-attribution visualization."
cross_refs: []
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "§1 contribution (3); §5 Explainability; Figure 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unsupervised-epoch-selection-not-implemented
category: missing
topic: "hyperparameter tuning / reproducibility"
title: "Unsupervised epoch-selection method not in repo; only hardcoded epoch results"
severity: medium
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/FMAD/functions.py
line_start: 73
line_end: 80
quote: |
  def determine_FMAD_hyperparameters(dataset_name_raw):
      dataset_name = dataset_name_raw.lower()
      if "census" in dataset_name:
          epoch_size = 5
          batch_size = 1024
          learning_rate = 0.005
      elif "backdoor" in dataset_name:
          epoch_size = 200
claim: "The paper states the number of training epochs is chosen per-dataset via 'the unsupervised hyperparameter selection method proposed by Li et al. (2025b), which requires no access to anomaly labels'; the repo contains only the resulting hardcoded epoch values (e.g. comments such as 'Chosen first from \"100 or 1\"' at lines 128/132), not the selection procedure itself."
concern: "The per-dataset epoch values—the only data-dependent TCCM hyperparameter—cannot be reproduced and the label-free claim cannot be verified, because the selection code that would produce these numbers without anomaly labels is absent."
resolution: "Authors: please include the implementation (or a runnable wrapper) of the unsupervised epoch-selection method so the Table 5 epoch values are reproducible and the no-label criterion is checkable."
cross_refs: ["tccm-only-per-dataset-tuning"]
paper_ref: "Appendix B.3, 'The number of training epochs is determined ... unsupervised hyperparameter selection method'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-ablation-figures-16-17-ae
category: missing
topic: "ablations"
title: "Figures 16, 17 and Autoencoder+TimeEmbedding ablations claimed but absent from code"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  (4) Feature normalization (Figure 16): z-score normalization is generally beneficial and improves robustness; (5) Time-interpolated
  inputs (Figure 17): interpolation offers no gain and may add noise; (6) Comparison with Autoencoder
  +Time Embedding: confirms that TCCM learns a time-conditioned velocity field rather than a
  reconstruction mapping.
claim: "The paper lists six ablation/sensitivity studies; AblationStudies.py implements Figures 12-15 only. A repo-wide grep finds no code for Figure 16 (feature normalization), Figure 17 (time-interpolated inputs), or the Autoencoder+Time-Embedding comparison (see _audit_code/out/missing_artifacts.txt). Note the 'interpolated_x' at AblationStudies.py:46-47 belongs to the noise-injection study (Fig. 14), not a Fig. 17 study."
concern: "Three of the paper's claimed ablation analyses have no reproducing code, so those conclusions (e.g. 'z-score normalization improves robustness', 'interpolation offers no gain') cannot be verified from the repo."
resolution: "Authors: please add the scripts that produce the Figure 16, Figure 17, and Autoencoder+Time-Embedding ablation results."
cross_refs: []
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "§5 (4) Ablation Studies; Appendix D.3; Figures 16-17"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: run-knn-wrong-filename
category: bug
topic: "reproduction scripts"
title: "run_knn.sh invokes nonexistent Full_experiments.py (actual file is FullExperiments.py)"
severity: low
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/bash_files/run_knn.sh
line_start: 28
line_end: 28
quote: |
                nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
claim: "The helper script run_knn.sh calls `python Full_experiments.py`, but the repository file is `FullExperiments.py` (camel-case); `Full_experiments.py` does not exist (verified by ls), so every invocation in this script fails with a file-not-found error."
concern: "The standalone KNN reproduction script cannot run as written; it errors immediately on the wrong filename (low impact since KNN is also runnable through the main FullExperiments.py model index)."
resolution: "Rename the invoked file to `FullExperiments.py` in run_knn.sh, or remove the unused helper script."
cross_refs: []
paper_ref: "README Reproduction Roadmap (KNN reviewer check)"
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
title: "requirements.txt pins torch==1.13.1 but paper and README state PyTorch 2.0"
severity: low
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "requirements.txt pins torch==1.13.1, whereas Appendix B.3 of the paper states 'Our implementation is based on Python 3.9.21 with PyTorch 2.0'. Both are internally valid PyTorch versions for this MLP; only the recorded environment differs."
concern: "The reported software environment (PyTorch 2.0) does not match the pinned dependency (1.13.1), a minor reproducibility inconsistency that could change low-order numerical results."
resolution: "Authors: confirm which PyTorch version produced the reported numbers and align requirements.txt with the paper's stated environment."
cross_refs: []
paper_ref: "Appendix B.3 Hardware and Software"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: tccm-only-per-dataset-tuning
category: methodology
topic: "baselines / hyperparameter tuning fairness"
title: "TCCM gets per-dataset epoch tuning while all baselines use default hyperparameters"
severity: medium
confidence: medium
status: finding
file: code/ZhongLIFR__TCCM-NIPS/FullExperiments.py
line_start: 254
line_end: 266
quote: |
      elif mname == "TCCM":
          from FMAD.functions import determine_FMAD_hyperparameters
          hyparam = determine_FMAD_hyperparameters(dataset_name)
          parameters.update({"n_features": X_train.shape[1]})
          parameters.update(hyparam)
              
      elif mname in additional_models:
          if mname == "MCM":
              parameters = {"n_features": X_train.shape[1]}
          elif mname == "GOAD":
              parameters = {"n_epoch": 25}
          else:
              parameters = {}
claim: "TCCM is configured with per-dataset tuned epochs (and lr/batch) via determine_FMAD_hyperparameters, while all baselines run with default configurations (the paper confirms 'For all baseline detectors, we use their default configurations and hyperparameters'); only MCM/GOAD get a single fixed param, none get per-dataset tuning."
concern: "Tuning a key training-budget hyperparameter per dataset for the proposed method only—while deep baselines keep defaults—can bias the head-to-head ranking comparison (Figure 2) in TCCM's favour; the asymmetry's fairness hinges on the epoch-selection truly using no anomaly labels, which is unverifiable because the selection code is absent (see unsupervised-epoch-selection-not-implemented)."
resolution: "Authors: either apply an equivalent label-free per-dataset selection budget to the deep baselines, or report TCCM with a single fixed epoch budget to show the ranking is robust to this asymmetry; and release the selection code so the label-free claim is checkable."
cross_refs: ["unsupervised-epoch-selection-not-implemented"]
paper_ref: "Appendix B.3 (epochs via unsupervised selection; baselines use defaults)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 3 | medium | Explainability Fig. 4, two ablations (Fig. 16/17 + AE comparison), and the unsupervised epoch-selection procedure have no code. |
| bug | 1 | low | run_knn.sh calls a nonexistent filename. |
| difference | 1 | low | torch pin (1.13.1) disagrees with paper's stated PyTorch 2.0. |
| methodology | 1 | medium | TCCM per-dataset tuned; baselines default (paper-acknowledged, but tuning code absent). |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. [missing] `explainability-figure-no-code` — no code produces the explainability Figure 4 / 0.76 AUROC, a headline RQ3 contribution (medium / high).
2. [methodology] `tccm-only-per-dataset-tuning` — only TCCM gets per-dataset epoch tuning; baselines use defaults, a fairness concern for the Figure 2 ranking (medium / medium).
3. [missing] `unsupervised-epoch-selection-not-implemented` — the label-free epoch-selection method is not in the repo; only hardcoded results remain, so the "no anomaly labels" claim is unverifiable (medium / high).
4. [missing] `missing-ablation-figures-16-17-ae` — Figures 16, 17 and the Autoencoder comparison have no reproducing code (low / high).
5. [difference] `torch-version-mismatch` — requirements pin torch 1.13.1 vs paper's PyTorch 2.0 (low / high).
6. [bug] `run-knn-wrong-filename` — `run_knn.sh` references `Full_experiments.py` which does not exist (low / high).

### Items that genuinely look fine
- **Train/test split has no leakage** (`utils.py:33-48`): normals split 50/50, anomalies test-only, `StandardScaler` fit on `X_train` and only transformed on `X_test`.
- **Core method faithful to the paper**: training loss (Eq. 4) at `FMAD/FlowMatchingAD.py:30-34` and the one-step score `‖f(x,1)+x‖₂` (Eq. 5) at lines 48-50 match exactly; `tfixed=1` matches the paper default.
- **Metrics fit the imbalanced task**: both AUROC and AUPRC computed (`FullExperiments.py:310-311`), with correct orientation (higher residual = anomaly, positive class = label 1).
- **Genuine 5-seed replication**: seeds 0-4 each drive a fresh `train_test_split(random_state=seed)`, so the 5 runs use different splits (not just NN init noise).
- **All 47 datasets covered** by `determine_FMAD_hyperparameters` with safe substring ordering (no crash; verified `_audit_code/out/hyperparam_coverage.txt`).
- **Robustness GMM setup matches Proposition 2** (`RobustnessStudy.py:33-74`): normal modes {-3,0,3} vs anomaly modes {-9,9} are disjoint as assumed; PGD attack and FP/FN plots are implemented.
- **Dependencies pinned** (`requirements.txt`), README has results table + exact reproduction commands.

### Open questions for the authors
- Were the per-dataset TCCM epoch values selected strictly without anomaly labels (i.e. without consulting test AUROC/AUPRC)? Releasing the selection code would resolve this.
- Can the Figure 2 ranking be reproduced with a single fixed epoch budget for TCCM, to demonstrate the tuning asymmetry does not drive the result?
- Where is the script for the explainability Figure 4 (MNIST) and the AUROC 0.76 number?
