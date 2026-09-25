# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (paper 2657)

## 1. Summary

The cloned repo `code/ZhongLIFR__TCCM-NIPS/` is the official author code for this
NeurIPS 2025 paper (README title, author `ZhongLIFR` = first author Zhong Li, and
the OpenReview/arXiv links all match). Note the repo name says "TCCM" while the
paper title says "One-Step Flow Matching"; TCCM ("Time-Conditioned Contraction
Matching") is the method name, so this is consistent.

What I did:
- Read the core method (`FMAD/FlowMatchingAD.py`, `FMAD/functions.py`), the data
  loader (`utils.py`), and the four experiment drivers (`FullExperiments.py`,
  `AblationStudies.py`, `RobustnessStudy.py`, `ContaminationStudies.py`) plus
  result aggregation (`AggregateResults.py`, `ProcessContamination.py`) and the
  `bash_files/` runners.
- Verified the training loss (Eq. 4) and inference score (Eq. 5) against the code:
  both faithful (the score is `||f(x,t=1) + x||_2`; loss target is `-x`).
- Ran deterministic checks under `_audit_code/`:
  - `check_inventory.py`: 47 unique datasets present (matches paper); scanned all
    23 `.py` files for stat-test / explanation / epoch-selection code.
  - `check_model_indices.py`: reconstructed the 52-entry `MODEL_NAMES` order used
    by the bash runners.
  - `check_hyperparam_coverage.py`: all 47 datasets resolve in
    `determine_FMAD_hyperparameters`; verified the hardcoded epochs equal Table 5.
- Confirmed all main scripts `py_compile` cleanly.

Headline result: the empirical-accuracy, scalability, robustness, ablation, and
contamination pipelines are present and traceable to code. The two gaps that
matter most are (a) the **explainability** contribution — a headline claim in the
paper title and a dedicated quantitative table (Table 3) and two figures (Fig. 1,
Fig. 4) — has **no code at all**; and (b) the **Friedman/Nemenyi statistical
tests** (Appendix D.5, Fig. 21/22) have no code. Methodologically, the proposed
method's single most impactful hyperparameter (training epochs, ranging 1–5000)
is tuned per dataset while all baselines use default configurations.

## 2. Result-traceability table

Detector AUROC/AUPRC for every (dataset, model) pair is computed in
`FullExperiments.py:310-311`; per-dataset/per-seed rankings (Figure 2) are
computed in `AggregateResults.py:206-272`. Figures 3/7/8 timing numbers come from
the `~ExecTime/~TrainTime/~TestTime` columns aggregated there. I could not run the
full 10,575-run benchmark, so "Matches paper" is "n/a (not re-run)" for numeric
values; the column instead records whether the *computation* exists.

| Paper artefact | Repo location | Computes value? | Status |
|---|---|---|---|
| Fig. 2a/2b detector ranks (TCCM 5.8 / 5.7), 45 detectors × 47 datasets × 5 seeds | `FullExperiments.py:310-311` (AUROC/AUPRC), `AggregateResults.py:206-272` (ranks) | yes | Present (not re-run) |
| Fig. 3a/3b/3c inference/train/total time speedups (1573×, 4864×, 85.9×, …) | `FullExperiments.py:173,182-184` (times), `AggregateResults.py:225-231` (mean/std) | yes (values); plotting absent | Present (not re-run) |
| Tables 6–13 full AUROC/AUPRC tables | `AggregateResults.py:269` (`Results_{split}.csv`) | yes | Present (not re-run) |
| Fig. 21/22 Friedman + Nemenyi critical-difference diagrams (Appendix D.5) | (none) | no | **MISSING (no stat-test code)** |
| Table 3 explanation accuracy (ExactMatch / Jaccard, 5D–25D) | (none) | no | **MISSING (no explanation code)** |
| Fig. 1 learned contraction-vector visualisation (synthetic 2D) | (none) | no | **MISSING** |
| Fig. 4 MNIST 1-vs-7 feature explanation (AUROC 0.76) | (none) | no | **MISSING** |
| Figs. 12/13/14 ablations (time embedding / fixed-t / noise) | `AblationStudies.py:255-449,142-212` | yes | Present (not re-run) |
| Contamination figure (Appendix D, Fig "Contamination_Figure_TCCM") | `AblationStudies.py:461-580`; `ContaminationStudies.py` + `ProcessContamination.py` | yes | Present (not re-run) |
| Figs. 19/20 empirical robustness (PGD FN/FP attacks) | `RobustnessStudy.py:240-320` | yes | Present (not re-run) |
| Table 5 per-dataset #Epochs / batch size | `FMAD/functions.py:73-268` (hardcoded) | values match Table 5 | Verified (values equal) |
| "unsupervised CSM-based epoch selection" (App. B.6) producing Table 5 epochs | (none) | no | **MISSING (only final values hardcoded)** |

## 3. Findings

## missing

```yaml finding
id: explanation-code-absent
category: missing
topic: "explainability / result traceability"
title: "No code for the explainability contribution (Table 3, Fig. 1, Fig. 4)"
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
claim: "The model only returns a per-sample scalar anomaly score (the L2 norm collapses the per-feature residual vector). No code anywhere in the repo computes the per-feature residual components, the ExactMatch/Jaccard explanation metrics of Table 3, the synthetic-GMM explanation experiment, the 2D contraction-vector visualisation (Fig. 1), or the MNIST 1-vs-7 feature attribution (Fig. 4)."
concern: "Explainability is a headline contribution (it is in the paper title and has a dedicated quantitative table); none of its reported numbers or figures can be reproduced because the producing code is absent."
resolution: "Provide the script(s) that (a) build the synthetic GMM with known shifted dimensions, (b) compute per-feature residual attributions, and (c) compute ExactMatch and Jaccard against ground-truth dimensions, plus the Fig. 1 / Fig. 4 generators."
cross_refs: []
check_script: _audit_code/check_inventory.py
paper_ref: "Table 3 (App. D.4.2); Fig. 1; Fig. 4; Eq. 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: statistical-tests-absent
category: missing
topic: "statistical integrity / result traceability"
title: "Friedman + Nemenyi tests and critical-difference diagrams have no code"
severity: medium
confidence: high
status: finding
file: AggregateResults.py
line_start: 270
line_end: 273
quote: |
    order_roc = df_aggregated.groupby("Detector")["Rank_ROC"].mean().sort_values()
    order_roc.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank.csv"))
    order_pr = df_aggregated.groupby("Detector")["Rank_PR"].mean().sort_values()
    order_pr.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank_PR.csv"))
claim: "The aggregation script computes mean per-detector ranks and stops there; a repo-wide grep finds no Friedman, Nemenyi, critical-difference, posthoc, scikit_posthocs, or autorank usage in any of the 23 .py files."
concern: "The paper reports a Friedman omnibus test, a Nemenyi post-hoc test, and two critical-difference diagrams (Fig. 21, Fig. 22, App. D.5) as the basis for its significance claims, but no code computes them, so those statistical conclusions cannot be reproduced."
resolution: "Provide the script that runs the Friedman and Nemenyi tests on the per-dataset mean ranks and renders the CD diagrams."
cross_refs: []
check_script: _audit_code/check_inventory.py
paper_ref: "Appendix D.5; Figures 21 and 22"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: csm-epoch-selection-absent
category: missing
topic: "hyperparameter tuning"
title: "Unsupervised CSM epoch-selection protocol absent; only final epochs hardcoded"
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
claim: "The number of training epochs is returned by a per-dataset if/elif lookup of hardcoded constants; the values equal Table 5 (verified by _audit_code/check_hyperparam_coverage.py), but the CSM (Improved Contrast Score Margin) selection procedure that the paper says produced these values is nowhere in the repo (grep for 'csm'/'contrast score margin'/'select.*epoch' returns nothing)."
concern: "The paper (App. B.6) claims epochs were chosen by a principled label-agnostic CSM criterion; with only the final hardcoded values present, a reviewer cannot verify that the selection was actually unsupervised rather than tuned on test performance."
resolution: "Provide the CSM-based epoch-search script (candidate range + the T(f) criterion over which data) so the selected epoch values can be regenerated."
cross_refs: ["epoch-selection-test-dependence"]
check_script: _audit_code/check_hyperparam_coverage.py
paper_ref: "Appendix B.6 'Unsupervised Epoch Selection Strategy'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: run-knn-wrong-filename-and-index
category: bug
topic: "reproduction scripts"
title: "run_knn.sh calls a non-existent file and the wrong model index"
severity: low
confidence: high
status: finding
file: bash_files/run_knn.sh
line_start: 23
line_end: 29
quote: |
            for j in {50..50}; do # 1 models
                core_id=$(( (i * 1 + j) % MAX_CORES ))

                # Run the process
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
            done
claim: "run_knn.sh invokes `python Full_experiments.py` (file does not exist; the real driver is `FullExperiments.py`, verified by _audit_code/check_inventory.py) and uses model index 50, which is `INNE_semisup`, not KNN (KNN is index 16 and KNN_semisup is index 51)."
concern: "The script as written cannot run (missing file) and, even if the filename were fixed, would evaluate INNE rather than KNN; the standalone KNN runner is effectively non-functional."
resolution: "Rename the called file to FullExperiments.py and set the model index to the intended KNN entry (16 for transductive KNN, or 51 for KNN_semisup)."
cross_refs: ["semisup-knn-index-omitted"]
check_script: _audit_code/check_model_indices.py
paper_ref: "README 'Reproduction Roadmap'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: semisup-knn-index-omitted
category: bug
topic: "reproduction scripts"
title: "Inductive-mode runner skips KNN_semisup (index 51)"
severity: low
confidence: high
status: finding
file: bash_files/run_semisupervise.sh
line_start: 23
line_end: 23
quote: |
            for j in {45..50}; do # 7 models
claim: "The loop comment says '7 models' but {45..50} iterates only 6 indices (45-50); the seven force-inductive models occupy indices 45-51, so KNN_semisup (index 51) is never run (verified by _audit_code/check_model_indices.py). This pattern repeats for every split in the file."
concern: "The appendix inductive-mode comparison claims semi-supervised variants of all seven transductive detectors, but the runner omits the KNN inductive variant, so that row/column would be missing from the results unless run by the (broken) run_knn.sh."
resolution: "Change the loop bound to {45..51} so KNN_semisup is included, or confirm KNN_semisup was run separately."
cross_refs: ["run-knn-wrong-filename-and-index"]
check_script: _audit_code/check_model_indices.py
paper_ref: "Appendix on inductive-mode experiments (Table 5 inductive variants)"
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
title: "requirements.txt pins torch 1.13.1 but paper/README state PyTorch 2.0"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "The pinned PyTorch version is 1.13.1, whereas the paper (App. B.3, Hardware and Software) and README setup section state PyTorch 2.0 / Python 3.9.21."
concern: "A reviewer rebuilding the environment from requirements.txt gets a different major PyTorch version than the one the paper says was used; minor, but it is a stated-vs-pinned discrepancy."
resolution: "Reconcile the pin with the reported PyTorch 2.0 (or state that results are version-insensitive)."
cross_refs: []
paper_ref: "Appendix B.3 'Hardware and Software'; README 'Setup'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: asymmetric-epoch-tuning
category: methodology
topic: "baselines / hyperparameter tuning"
title: "TCCM's key hyperparameter is tuned per dataset while baselines use defaults"
severity: high
confidence: medium
status: finding
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
claim: "Only TCCM receives per-dataset hyperparameters (epochs spanning 1–5000 and batch size 512/1024 via determine_FMAD_hyperparameters). Every other detector is instantiated with its source-default configuration (AutoEncoder verbose=0 and DeepSVDD/MCM n_features are structural, not tuning; GOAD n_epoch=25 is a single fixed constant). The paper confirms 'For all baseline detectors, we use their default configurations and hyperparameters.'"
concern: "Tuning the proposed method's most impactful hyperparameter per dataset while leaving baselines at defaults is an asymmetric comparison that can inflate TCCM's relative ranking — the headline effectiveness claim (best average rank)."
resolution: "Either tune each baseline's primary hyperparameter per dataset with the same label-agnostic budget, or report TCCM with a single fixed epoch count across datasets, to make the comparison symmetric."
cross_refs: ["csm-epoch-selection-absent"]
check_script: _audit_code/check_hyperparam_coverage.py
paper_ref: "Appendix B.3 (TCCM hyperparameters; 'default configurations' for baselines); Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: epoch-selection-test-dependence
category: methodology
topic: "hyperparameter tuning / data leakage"
title: "Epoch-selection criterion appears to require the (anomaly-containing) test set"
severity: medium
confidence: low
status: question
file: utils.py
line_start: 33
line_end: 43
quote: |
    # Train using only normal samples
    X_normal, X_anomalous = X[y == 0], X[y == 1]
    y_normal, y_anomalous = y[y == 0], y[y == 1]

    X_train, X_test_normal, y_train, y_test_normal = train_test_split(
        X_normal, y_normal, test_size=test_size, random_state=random_state, stratify=y_normal
    )

    # Test set contains both normal and abnormal data
    X_test = np.vstack((X_test_normal, X_anomalous))
    y_test = np.concatenate((y_test_normal, y_anomalous))
claim: "Training data is normal-only (no anomalies). The paper's CSM epoch criterion T(f) compares top-k 'predicted anomalies' against presumed inliers, which requires a set that actually contains anomalies — i.e. the test set, since X_train has none. The selection code is absent (see csm-epoch-selection-absent), so I cannot confirm which split it used."
concern: "If the per-dataset epoch count was selected by maximising a criterion computed on the anomaly-containing test set, then model selection is test-set-dependent even though it is label-agnostic, which can bias the reported metrics upward."
resolution: "Authors: clarify on which data the CSM criterion was evaluated for epoch selection (held-out portion of training normals vs the test set) and release that code."
cross_refs: ["csm-epoch-selection-absent", "asymmetric-epoch-tuning"]
paper_ref: "Appendix B.6 (CSM criterion T(f)); utils.load_adbench_npz"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|-----------------------------------------------------------------------|
| missing     | 3          | high         | Explainability (Table 3 / Fig. 1 / Fig. 4), stat-tests, CSM selection all absent |
| bug         | 2          | low          | KNN runner broken; inductive runner skips KNN_semisup                 |
| difference  | 1          | low          | torch pin 1.13.1 vs stated PyTorch 2.0                                 |
| methodology | 2          | high         | TCCM tuned per dataset vs default baselines; possible test-dependent epoch selection |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing] explanation-code-absent** — the title-level "Explainable" claim, Table 3 (ExactMatch/Jaccard), Fig. 1 and Fig. 4 have no producing code (high / high).
2. **[methodology] asymmetric-epoch-tuning** — only TCCM gets per-dataset epoch/batch tuning (epochs 1–5000); all baselines use defaults, which can inflate the headline ranking (high / medium).
3. **[missing] statistical-tests-absent** — Friedman/Nemenyi tests and CD diagrams (App. D.5, Fig. 21/22) have no code (medium / high).
4. **[missing] csm-epoch-selection-absent** — the "unsupervised CSM epoch selection" that produced Table 5 is not in the repo; only the final values are hardcoded (medium / high).
5. **[methodology] epoch-selection-test-dependence** — the CSM criterion needs anomalies, which only exist in the test set; possible test-dependent selection, unverifiable (medium / low; filed as question).
6. **[bug] run-knn-wrong-filename-and-index** — run_knn.sh calls a non-existent file and the wrong model index (low / high).

### Items that genuinely look fine
- Training loss (Eq. 4) and inference score (Eq. 5) are implemented faithfully (`FMAD/FlowMatchingAD.py:33-34,48-50`); the `nn.MSELoss` vs `E[||·||]` difference shares the same minimiser (benign, not filed).
- Data loading has no train→test leakage: `StandardScaler` is fit on training normals only and applied to test (`utils.py:46-48`).
- All 47 datasets are present and uniquely partitioned across the four size splits; the hardcoded epoch values exactly match Table 5 (verified deterministically).
- Ablation (Fig. 12/13/14), empirical-robustness (Fig. 19/20), and contamination experiments all have present, runnable driver code that computes the reported metrics.
- Seeding is comprehensive (`utils.set_seed` covers torch/numpy/random/cuDNN) and five seeds are used throughout.

### Open questions for the authors
- On which data split was the CSM epoch-selection criterion computed (held-out training normals vs the test set)? (epoch-selection-test-dependence)
- Were the inductive-mode results for KNN_semisup actually produced, given the runner skips index 51? (semisup-knn-index-omitted)
- Can the explanation-accuracy (Table 3), Fig. 1, and Fig. 4 generation scripts be released? (explanation-code-absent)
