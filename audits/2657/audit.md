# Code-repository audit — TCCM (NeurIPS 2025, paper 2657)

## 1. Summary

The repo `code/ZhongLIFR__TCCM-NIPS/` is the official code for "Scalable, Explainable
and Provably Robust Anomaly Detection with One-Step Flow Matching" (TCCM). It contains
the TCCM model (`FMAD/`), 13 author-ported baselines plus the PyOD/ADBench baseline
wiring (`baselines/`, `utils.py`), driver scripts for the main benchmark
(`FullExperiments.py` + `bash_files/run_main.sh`), an inductive-mode variant
(`run_semisupervise.sh`), result aggregation/ranking (`AggregateResults.py`), ablations
(`AblationStudies.py`), empirical adversarial robustness (`RobustnessStudy.py`), and
contaminated-training robustness (`ContaminationStudies.py`, `ProcessContamination.py`).
All 47 ADBench `.npz` datasets are shipped.

I read the method and every driver script, mapped the paper's figures/tables/claims to
producing code, and ran three deterministic checks under `_audit_code/`:
`check_missing_artifacts.py` (keyword/file-existence grep over all `*.py`/`*.sh` for the
explainability code, the unsupervised epoch-selection code, the `Full_experiments.py`
reference, and dataset count), and `check_split_and_scaler.py` (re-implements
`load_adbench_npz` on a shipped dataset to confirm no scaling leakage and a clean
normal-only train split). The data pipeline is clean. The principal gaps are that the
two non-detection headline contributions — explainability (RQ3 / Table 3) and the
disclosed unsupervised epoch-selection protocol — have no producing code in the repo.

The audited tree is a moving `main` (no submission tag); the README "Updates" log shows
edits dated 2026-05-11, after acceptance.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2a/b — AUROC & AUPRC rank distributions across 47 datasets, 45 detectors | `FullExperiments.py:310-312` (per-run AUROC/AUPRC) → `AggregateResults.py:206-271` (ranks) | not run (3-day CPU benchmark) | n/a (logic present) | Verified-present |
| Per-dataset AUROC/AUPRC tables (Appendix) | `FullExperiments.py:310-316`, `AggregateResults.py:220-269` | not run | n/a | Verified-present |
| Training/inference time comparison (scalability, Fig. 3) | `FullExperiments.py:160-188,313` (train/test timing) | not run | n/a | Verified-present |
| Inductive-mode (`_semisup`) reviewer experiment | `utils.py:28-36`, `run_semisupervise.sh` | not run | n/a | Verified-present |
| Ablation: time-embedding variants (Fig. 12) | `AblationStudies.py:255-349` | not run | n/a | Verified-present |
| Ablation: sensitivity to fixed `t` (Fig. 13) | `AblationStudies.py:142-211` | not run | n/a | Verified-present |
| Ablation: noise injection (Fig. 14) | `AblationStudies.py:360-448` | not run | n/a | Verified-present |
| Ablation: training contamination (Fig. 15) | `AblationStudies.py:461-579` | not run | n/a | Verified-present |
| Contaminated-training robustness, top-10 models | `ContaminationStudies.py:23-36,...`, `ProcessContamination.py` | not run | n/a | Verified-present |
| Empirical adversarial robustness (PGD FN/FP, Figs 19/20) | `RobustnessStudy.py:240-320` | not run | n/a | Verified-present |
| **Explainability: ExactMatch / Jaccard (Table 3, RQ3, Appendix D.4.2)** | (none) | — | — | **MISSING (no explanation code)** |
| **Fig. 1 — synthetic 2D contraction-field visualization** | (none) | — | — | **MISSING** |
| Unsupervised epoch-selection (CSM, Appendix B.6; Table 5 epoch values) | hardcoded only: `FMAD/functions.py:73-268` | — | n/a | MISSING (selection protocol not in repo) |

## 3. Findings

## missing

```yaml finding
id: explainability-code-missing
category: missing
topic: "explainability / result traceability"
title: "No code for the explainability contribution (Table 3 ExactMatch/Jaccard, Fig. 1)"
severity: high
confidence: high
status: finding
file: FMAD/FlowMatchingAD.py
line_start: 40
line_end: 53
quote: |
    def decision_function(self, X_test):
        """
        Compute the anomaly scores of X_test
        """
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)

        with torch.no_grad():
            t = torch.ones(X.shape[0], 1, device=X.device, dtype=torch.float32)  # Set t to 1
            f_xt = self.model(X, t)  # Predict contraction vectors
            anomaly_scores = torch.norm(f_xt + X, dim=1)  # compute the anomaly score, based on Equation 5.

        anomaly_scores = anomaly_scores.cpu().numpy()
        return anomaly_scores
claim: "The model only ever reduces the residual f(x,1)+x to a scalar L2 norm; nothing in the repo computes the per-feature residual components, the top-k feature attribution, or the ExactMatch/Jaccard metrics, and no synthetic-GMM explanation experiment or 2D contraction visualization exists. _audit_code/check_missing_artifacts.py reports ABSENT for explainability/attribution code and for the Fig. 1 visualization."
concern: "'Explainable' is in the paper title and is headline contribution RQ3; Table 3 reports near-perfect ExactMatch/Jaccard, but no code produces the feature-level attributions or those numbers, so this contribution is unreproducible from the repo."
resolution: "Provide the synthetic-GMM explanation experiment that computes top-k feature attributions from the residual vector and the ExactMatch/Jaccard values in Table 3, plus the Fig. 1 visualization code."
cross_refs: ["§D.4.2", "Table 3", "Figure 1"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Abstract; §5 RQ3; Appendix D.4.2, Table 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: csm-epoch-selection-missing
category: missing
topic: "hyperparameter tuning / protocol"
title: "Disclosed unsupervised epoch-selection (CSM) protocol absent; epochs hardcoded"
severity: medium
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
claim: "Per-dataset epoch counts (ranging 1..5000) are hardcoded in a long if/elif chain; the unsupervised Contrast-Score-Margin (CSM) selection method from Li et al. (2025b) that the paper says produced these values (Appendix B.6) is not in the repo. _audit_code/check_missing_artifacts.py reports ABSENT for any CSM/epoch-selection code."
concern: "TCCM is the only model with a tuned hyperparameter (epochs); without the selection code a reviewer cannot verify the values are the label-free CSM optima the paper claims rather than chosen with reference to test AUROC, leaving the fairness of the tuned-vs-default comparison unverifiable."
resolution: "Release the CSM epoch-selection script (candidate-epoch search + T(f) criterion) so the hardcoded Table 5 epoch values can be reproduced without anomaly labels."
cross_refs: ["epoch-selection-asymmetry", "§B.3", "§B.6", "Table 5"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Appendix B.3 and B.6; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: run-knn-wrong-script-name
category: bug
topic: "driver scripts"
title: "run_knn.sh invokes non-existent Full_experiments.py (file is FullExperiments.py)"
severity: low
confidence: high
status: finding
file: bash_files/run_knn.sh
line_start: 28
line_end: 28
quote: |
                nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
claim: "run_knn.sh calls `python Full_experiments.py` in all four dataset-group blocks, but the file is named `FullExperiments.py`; `Full_experiments.py` does not exist in the repo (_audit_code/check_missing_artifacts.py: FILE_MISSING)."
concern: "Every invocation in run_knn.sh fails with `No such file or directory`, so this auxiliary KNN-only runner produces no output; the main benchmark (run_main.sh) uses the correct name and is unaffected."
resolution: "Rename the four references in run_knn.sh from `Full_experiments.py` to `FullExperiments.py`."
cross_refs: []
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "n/a (helper script, not in README roadmap)"
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
title: "Paper states PyTorch 2.0; requirements.txt pins torch==1.13.1"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "requirements.txt pins torch==1.13.1, whereas the paper states the implementation uses PyTorch 2.0."
concern: "Minor reproducibility/description discrepancy; the model is a plain MLP and both runs are CPU-only, so results are unlikely to change, but the documented environment does not match the shipped one."
resolution: "Align the pinned torch version with the version actually used for the reported runs."
cross_refs: []
paper_ref: "Appendix B.3 (Hardware and Software)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: epoch-selection-asymmetry
category: methodology
topic: "baselines / tuning fairness"
title: "TCCM gets per-dataset epoch selection; all baselines run at library defaults"
severity: medium
confidence: medium
status: question
file: FullExperiments.py
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
claim: "Only TCCM receives per-dataset tuned hyperparameters (via determine_FMAD_hyperparameters); every other detector is instantiated with empty/default parameters, consistent with the paper's statement that baselines 'use their default configurations and hyperparameters'."
concern: "Asymmetric tuning (the proposed method's only tunable knob is optimized per dataset while deep baselines keep library defaults, e.g. fixed epoch counts) can favor TCCM in the headline ranking; the paper argues the selection is label-free (CSM), but that code is absent (see csm-epoch-selection-missing), so the fairness cannot be confirmed."
resolution: "Confirm the epoch selection used no anomaly labels / test scores, and report TCCM under a fixed default epoch budget (or apply the same label-free budget search to deep baselines) to show the #1 ranking is not an artifact of asymmetric tuning."
cross_refs: ["csm-epoch-selection-missing", "§B.3"]
paper_ref: "Appendix B.3 (Configurations; 'For all baseline detectors, we use their default configurations')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|-----------------------------------------------------------------------|
| missing     | 2          | high         | Explainability contribution (Table 3, Fig. 1) and CSM epoch-selection protocol have no code |
| bug         | 1          | low          | run_knn.sh calls a misnamed script; main benchmark unaffected          |
| difference  | 1          | low          | requirements.txt torch==1.13.1 vs paper's "PyTorch 2.0"               |
| methodology | 1          | medium       | TCCM tuned per-dataset vs default baselines (filed as question; selection claimed label-free) |

### Top take-aways (≤6, by severity × confidence)
1. **[missing] Explainability is unreproducible** — no code computes per-feature attributions, ExactMatch/Jaccard (Table 3), or Fig. 1, despite "Explainable" being a title-level contribution (`explainability-code-missing`, high/high).
2. **[missing] The unsupervised epoch-selection protocol is absent** — epochs are hardcoded per dataset; the CSM selection method that purportedly produced them is not shipped (`csm-epoch-selection-missing`, medium/high).
3. **[methodology] Tuning asymmetry** — only TCCM is hyperparameter-tuned per dataset while baselines use defaults; fairness of the #1 ranking hinges on the (absent) label-free claim (`epoch-selection-asymmetry`, medium/medium, question).
4. **[bug] run_knn.sh is broken** — references non-existent `Full_experiments.py` (`run-knn-wrong-script-name`, low/high).
5. **[difference] torch version mismatch** — requirements pin 1.13.1, paper says 2.0 (`torch-version-mismatch`, low/high).

### Items that genuinely look fine
- **No scaling/leakage in the split**: `StandardScaler` is fit on the training (normal-only) subset and applied to the test set; train is all-normal, all anomalies are test-only, and train/test-normal rows are disjoint (`_audit_code/check_split_and_scaler.py`, verified on a shipped dataset).
- **Anomaly-score direction is correct**: `decision_function` returns `||f(x,1)+x||` (higher = more anomalous) and `roc_auc_score(y_test, scores)` uses y=1 for anomalies — consistent (`FMAD/FlowMatchingAD.py:50`, `FullExperiments.py:310`).
- **Main benchmark + aggregation traceable**: AUROC/AUPRC are computed per run in `FullExperiments.py` and ranked across 5 seeds in `AggregateResults.py`; Fig. 2 is produced by code present in the repo.
- **Ablations, contamination, and adversarial-robustness studies all have producing scripts** (`AblationStudies.py`, `ContaminationStudies.py`, `RobustnessStudy.py`).
- **All 47 ADBench datasets are shipped** (`_audit_code/check_missing_artifacts.py`: count = 47, matching the paper).

### Open questions for the authors
- Did the per-dataset epoch values in `determine_FMAD_hyperparameters` come solely from the label-free CSM criterion, with no reference to test AUROC/AUPRC? (Comments such as `# Chosen first from "100 or 1"` warrant clarification.) See `epoch-selection-asymmetry` / `csm-epoch-selection-missing`.
- Will the explainability experiment (Table 3, Appendix D.4.2) and Fig. 1 visualization code be released? See `explainability-code-missing`.
