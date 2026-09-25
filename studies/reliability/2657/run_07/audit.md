# Audit — Paper 2657: TCCM (Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching)

## 1. Summary

The cloned repo `code/ZhongLIFR__TCCM-NIPS/` is the **official author code** (README,
citation, and OpenReview/arXiv links all match paper 2657; repo owner `ZhongLIFR` = first
author *Zhong Li*). It implements TCCM (Time-Conditioned Contraction Matching), a one-step
flow-matching anomaly detector, plus 44 baselines, evaluated on 47 ADBench datasets in a
semi-supervised setting (train on 50% of normal data, test on the rest + all anomalies).

What I did (all read-only):
- Read the core method (`FMAD/FlowMatchingAD.py`, `FMAD/functions.py`), the main driver
  (`FullExperiments.py`), data loading/splitting (`utils.py`), the aggregation
  (`AggregateResults.py`), and the ablation/robustness/contamination scripts.
- Cross-checked the paper's protocol (semi-supervised split, StandardScaler on train only)
  against `utils.load_adbench_npz` — these **match** and contain no obvious leakage.
- Wrote three deterministic checks under `_audit_code/`:
  - `check_missing_artifacts.py` — greps the whole repo for the procedures the paper
    describes (CSM epoch selection, explainability/attribution, Friedman/Nemenyi tests).
  - `check_epochs_match_table5.py` — confirms the hardcoded per-dataset epochs equal paper
    Table 5.
  - `check_model_index_ranges.py` — maps bash model-index loops to `MODEL_NAMES`.
- All 47 datasets are bundled (`datasets/{small,medium,high_dim,large}`, 12/15/9/11) and
  `requirements.txt` pins all dependencies — data and environment are reproducible.

Headline picture: the **main effectiveness/scalability pipeline is present and traceable**.
The gaps are (a) the *unsupervised epoch-selection procedure* that the paper says produced
TCCM's only data-dependent hyperparameter is **absent** (only a hardcoded lookup remains),
and (b) two whole paper contributions/analyses — **explainability** (RQ3, Fig. 4, Table 3)
and the **statistical-significance tests** (App. D.5, Figs. 21–22) — have **no implementing
code** at all.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2a/2b TCCM rank AUPRC=5.8 / AUROC=5.7, all 45×47×5 runs | `FullExperiments.py` (per run) + `AggregateResults.py` (ranks) | Yes (re-run needed) | not re-run | Present / traceable |
| Fig. 3a/3b/3c runtime (inference / train / total) | `FullExperiments.py:153-188` records `train_time`/`test_time`; `AggregateResults.py:225-230` aggregates | Yes | not re-run | Present / traceable |
| Tables 6–13 per-dataset AUROC/AUPRC (all 45 detectors) | `FullExperiments.py` + `AggregateResults.py` | Yes | not re-run | Present / traceable |
| Table 5 #Epochs per dataset (TCCM hyperparams) | `FMAD/functions.py:73-268` (hardcoded) | values match (verified) | ✓ | Present (values) — see `missing-csm-epoch-selection` for the *selection method* |
| "Unsupervised epoch selection via CSM, label-agnostic" (App. B.6) | (none) | — | — | **MISSING** (no CSM / selection code) |
| Fig. 4 MNIST explainability (1 vs 7, AUROC=0.76) | (none) | — | — | **MISSING** (no MNIST / attribution code) |
| Table 3 explanation accuracy (ExactMatch / Jaccard), App. D.4.2 | (none) | — | — | **MISSING** (no synthetic attribution code) |
| App. D.5 / Figs. 21–22 Friedman + Nemenyi critical-difference | (none) | — | — | **MISSING** (no statistical-test code) |
| Fig. 23 inductive (force-inductive) ranking, incl. KNN_semisup | `FullExperiments.py` via `bash_files/run_semisupervise.sh` | partial | — | KNN_semisup not launched — see `knn-semisup-omitted-from-bash` |
| Ablations Figs. 12–17 (time embed, fixed t, noise, contamination) | `AblationStudies.py` | Yes | not re-run | Present / traceable |
| Empirical robustness Figs. (PGD FN/FP attacks) | `RobustnessStudy.py` | Yes | not re-run | Present / traceable |
| Contamination robustness | `ContaminationStudies.py` + `ProcessContamination.py` | Yes | not re-run | Present / traceable |

## missing

```yaml finding
id: missing-csm-epoch-selection
category: missing
topic: "hyperparameter tuning / result traceability"
title: "Unsupervised CSM epoch-selection procedure (Table 5 values) is absent; only a hardcoded lookup remains"
severity: high
confidence: high
status: finding
file: FMAD/functions.py
line_start: 73
line_end: 82
quote: |
  def determine_FMAD_hyperparameters(dataset_name_raw):
      dataset_name = dataset_name_raw.lower()
      if "census" in dataset_name:
          epoch_size = 5
          batch_size = 1024
          learning_rate = 0.005
      elif "backdoor" in dataset_name:
          epoch_size = 200
          batch_size = 1024
          learning_rate = 0.005
claim: "The only data-dependent TCCM hyperparameter (training epochs) is set by a hardcoded per-dataset if/elif lookup table running to line 268; inline comments at lines 128 and 132 read 'epoch_size = 110  # Chosen first from \"100 or 1\"' and 'epoch_size = 2  # Chosen first from \"100 or 1\"'. No code anywhere in the repo computes the 'Improved Contrast Score Margin (CSM)' criterion, examines training-loss curves, or searches a candidate-epoch space."
concern: "The paper (App. B.6, Table 5) attributes these epoch values to an unsupervised, label-agnostic CSM selection procedure (Li et al. 2025b); that procedure is what makes the headline effectiveness numbers fair, yet no implementation of it exists in the repo, so the claim that epochs were chosen without any access to test/anomaly information cannot be reproduced or verified (the inline comments 'Chosen first from \"100 or 1\"' suggest manual choice)."
resolution: "Authors: please release the CSM epoch-selection script (the T(f) criterion and the per-candidate-epoch search), and confirm whether the Table-5 epochs were ever selected using test-set AUROC/AUPRC rather than the unsupervised criterion."
cross_refs: ["§B.3", "§B.6", "Table 5"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Appendix B.6 'Unsupervised Epoch Selection Strategy'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-explainability-code
category: missing
topic: "explainability"
title: "No code for the explainability contribution (Fig. 4 MNIST, Table 3 ExactMatch/Jaccard)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  We use digit 1 as the normal class and digit 7
  as the anomaly (achieving an AUROC of 0.76). As shown in Figure 4, the model highlights the
  additional horizontal stroke that distinguishes 7 from 1
claim: "Explainability is one of the three headline contributions (RQ3). The paper reports a MNIST feature-attribution figure (Fig. 4, AUROC=0.76) and a quantitative synthetic-GMM attribution study (Table 3: ExactMatch / Jaccard over d in {5,10,15,20,25}, App. D.4.2). A repo-wide grep finds no code that loads MNIST as 1-vs-7, computes per-feature importance/attribution, or computes ExactMatch / Jaccard against ground-truth shifted dimensions."
concern: "A full headline contribution and its quantitative validation (Table 3, near-perfect ExactMatch/Jaccard) have no implementing artefact, so neither Fig. 4 nor Table 3 can be reproduced."
resolution: "Authors: please add the MNIST explanation script and the synthetic GMM attribution-accuracy script (top-k feature selection from the residual vector, ExactMatch and Jaccard computation)."
cross_refs: ["Figure 4", "Table 3", "§D.4.2"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 5 (3) Explainability; Appendix D.4.2; Table 3; Figure 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-statistical-tests
category: missing
topic: "statistical integrity"
title: "Friedman + Nemenyi critical-difference tests (App. D.5, Figs. 21-22) have no code"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  We further perform statistical significance testing using the
  Friedman (Friedman, 1937) and Nemenyi tests (Nemenyi, 1963) to assess whether the observed
  ranking differences are statistically meaningful; detailed results are provided in Appendix D.5.
claim: "The paper relies on Friedman + Nemenyi tests and critical-difference diagrams (Figs. 21-22) to support its significance claims. A repo-wide grep for 'friedman', 'nemenyi', 'critical difference', 'posthoc', 'wilcoxon', 'scikit_posthocs' returns zero hits; `AggregateResults.py` computes only mean ranks, not any hypothesis test."
concern: "The statistical tests that the paper cites as evidence for/against ranking significance (and which conclude no significant difference among the top group) cannot be reproduced from the released code."
resolution: "Authors: please release the script that computes the Friedman statistic, the Nemenyi post-hoc test, and the critical-difference diagrams from the per-seed ranks."
cross_refs: ["§D.5", "Figure 21", "Figure 22"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 5 (1); Appendix D.5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: knn-semisup-omitted-from-bash
category: bug
topic: "reproducibility / experiment harness"
title: "run_semisupervise.sh loop {45..50} omits KNN_semisup (index 51); run_knn.sh calls a nonexistent script"
severity: medium
confidence: high
status: finding
file: bash_files/run_semisupervise.sh
line_start: 23
line_end: 28
quote: |
            for j in {45..50}; do # 7 models
                core_id=$(( (i * 6 + j) % MAX_CORES ))

                # Run the process
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python FullExperiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
claim: "MODEL_NAMES has 52 entries (verified by AST). The 7 force-inductive models occupy indices 45-51, with KNN_semisup at index 51. run_semisupervise.sh loops j in {45..50} (only 6 indices), so KNN_semisup (51) is never launched even though the comment says '7 models'. The auxiliary run_knn.sh invokes `python Full_experiments.py` (line 28 etc.), but the file is named `FullExperiments.py` and `Full_experiments.py` does not exist; its loop {50..50} targets index 50 (INNE_semisup), not KNN."
concern: "The inductive-evaluation appendix (Fig. 23) reports a KNN_semisup result, but as scripted no run produces it, and the dedicated run_knn.sh fails with FileNotFoundError; the supplementary inductive figure cannot be reproduced as released."
resolution: "Change the loop to {45..51} (and fix the comment / core_id divisor) and rename `Full_experiments.py` to `FullExperiments.py` in run_knn.sh."
cross_refs: ["Figure 23", "§E"]
check_script: _audit_code/check_model_index_ranges.py
paper_ref: "Appendix E (Results under the Inductive Evaluation Setting); Figure 23"
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
title: "Paper and README state PyTorch 2.0, requirements.txt pins torch==1.13.1"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "requirements.txt pins torch==1.13.1, whereas the paper (App. B.3, 'Our implementation is based on Python 3.9.21 with PyTorch 2.0') and the README (Setup) state PyTorch 2.0 / Python 3.9.21."
concern: "Minor environment inconsistency; unlikely to change results for this simple MLP, but a reproducer following the paper's stated PyTorch 2.0 may see slightly different numerics than the pinned 1.13.1 environment."
resolution: "Authors: align requirements.txt with the reported PyTorch version (or note that 1.13.1 is the actually-used version)."
cross_refs: ["§B.3"]
paper_ref: "Appendix B.3 (Hardware and Software)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding filed as a confirmed defect. The main protocol is sound:
- Semi-supervised split (50% normal train / rest+anomalies test) with `train_test_split`
  stratified on the normal class, split **varying per seed** (`random_state=seed`), so the
  5 runs are genuinely different splits (`utils.py:19-39`).
- `StandardScaler` is **fit on train only** and applied to test (`utils.py:46-48`) — no
  preprocessing leakage.
- Transductive baselines `fit(X_test)` (`FullExperiments.py:167`) is the standard
  transductive protocol and is explicitly disclosed (paper §D.2.2); these models are
  **excluded** from the headline `--semi_only` aggregation (`AggregateResults.py:126-128`,
  default `semi_only=True`), so they do not inflate TCCM's reported rank.

One open methodology *question* (not a confirmed finding) is recorded below in
"Open questions", cross-referencing `missing-csm-epoch-selection`.

## Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | CSM epoch-selection, all explainability code, and all statistical-test code are absent |
| bug         | 1          | medium       | run_semisupervise.sh omits KNN_semisup; run_knn.sh calls nonexistent script |
| difference  | 1          | low          | torch pinned to 1.13.1 vs paper/README "PyTorch 2.0" |
| methodology | 0          | -            | main split/scaler/transductive handling is sound; see open question |

## Top take-aways (ranked)

1. **[missing] `missing-csm-epoch-selection` (high / high)** — TCCM's only data-dependent
   hyperparameter (epochs) is a hardcoded table; the unsupervised CSM selection the paper
   credits for it has no code, so the fairness/label-agnosticism of the headline numbers
   cannot be verified.
2. **[missing] `missing-explainability-code` (medium / high)** — an entire headline
   contribution (explainability: Fig. 4 MNIST, Table 3 ExactMatch/Jaccard) is uncoded.
3. **[missing] `missing-statistical-tests` (medium / high)** — Friedman/Nemenyi significance
   tests (App. D.5, Figs. 21-22) are uncoded.
4. **[bug] `knn-semisup-omitted-from-bash` (medium / high)** — the inductive-evaluation
   appendix figure (Fig. 23) cannot be reproduced as scripted (KNN_semisup not launched;
   run_knn.sh references a nonexistent file).
5. **[difference] `torch-version-mismatch` (low / high)** — torch 1.13.1 pinned vs paper's
   PyTorch 2.0.

## Items that genuinely look fine

- **Data splitting / scaler**: no leakage; scaler fit on train only; per-seed splits differ.
- **Datasets shipped**: all 47 ADBench `.npz` files are bundled (12/15/9/11).
- **Dependencies**: fully pinned in `requirements.txt`.
- **Core method matches the algorithm**: training loss `||f(x,t)+x||^2` (Eq. 4) and inference
  score `||f(x,1)+x||_2` (Eq. 5) match `FMAD/FlowMatchingAD.py:30-37,48-50` and Algorithms 1-2.
- **Table 5 epoch values** equal the hardcoded epochs (verified deterministically).
- **Effectiveness/scalability/ablation/robustness/contamination pipelines** are present and
  traceable to the figures they produce.
- **Transductive vs inductive handling** is disclosed and the headline aggregation excludes
  transductive `fit(X_test)` models.

## Open questions for the authors

- (cross-ref `missing-csm-epoch-selection`) Were the Table-5 epoch values ever selected by
  inspecting test-set AUROC/AUPRC (e.g., the epoch-sensitivity sweep of Fig. 5), or strictly
  via the unsupervised CSM criterion with no label/test access? If the former, the per-dataset
  epoch choice would be test-set tuning (a `methodology` concern); the released code cannot
  distinguish these because the selection procedure is absent.
- The abstract states TCCM is "outperforming state-of-the-art methods", while App. D.5 reports
  the Nemenyi test finds **no statistically significant difference** among the top group
  (TCCM, DTE-NonParametric, LUNAR, KDE, ...). This is internal to the paper (not a code
  defect); please confirm the headline framing is intended as the accuracy-vs-speed *balance*
  rather than a significant accuracy win.
