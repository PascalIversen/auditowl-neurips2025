# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (paper 2657)

## 1. Summary

The cloned repo `code/ZhongLIFR__TCCM-NIPS/` is the official author code (owner
`ZhongLIFR` = first author Zhong Li; README cites this exact NeurIPS 2025 paper).
TCCM is a one-step flow-matching anomaly detector for tabular data, evaluated on
47 ADBench datasets against 44 baselines (AUROC/AUPRC ranks, Figure 2), plus
scalability (Figure 3), explainability (Figure 4 / Table 3), ablations
(Appendix D.3), empirical robustness (Appendix D.4), and contamination
(Appendix D) studies.

What I did (read-only):
- Read the core model (`FMAD/FlowMatchingAD.py`, `FMAD/functions.py`), the data
  loader/split (`utils.py`), and all experiment drivers (`FullExperiments.py`,
  `AggregateResults.py`, `AblationStudies.py`, `RobustnessStudy.py`,
  `ContaminationStudies.py`, `ProcessContamination.py`) and the `bash_files/`.
- Cross-checked the paper's Table 5 per-dataset epochs against the hardcoded
  `determine_FMAD_hyperparameters` table (they match).
- Ran three deterministic checks under `_audit_code/`:
  - `check_semisup_index_range.py` — confirms `run_semisupervise.sh` launches
    6 of the 7 force-inductive models (off-by-one).
  - `check_split_and_scaler.py` — confirms training is normal-only,
    train/test normal rows are disjoint, and the StandardScaler is fit on the
    training split only (no test-statistic leakage).
- Greps for explainability code (ExactMatch / Jaccard / top-k feature
  attribution / MNIST 1-vs-7) and statistical-test code (Friedman / Nemenyi):
  none found.

Overall: the core model and the main AUROC/AUPRC benchmark, ablations,
robustness, and contamination experiments are present and the data-splitting is
methodologically clean. The main gaps are (a) the explainability experiments — a
headline contribution — have no code; (b) the unsupervised epoch-selection
procedure that produced the per-dataset epoch values is not implemented (only
its hardcoded outputs are shipped); (c) the statistical-significance tests have
no code; and (d) two driver-script defects (one off-by-one that drops a baseline,
one dead filename).

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2a/2b AUPRC/AUROC rank box plots (47 datasets, 45 detectors) | `FullExperiments.py` (scores+AUROC/AUPRC) → `AggregateResults.py` (`Rank_PR`/`Rank_ROC`, mean ranks) | rank values in CSV | n/a (rank distribution) | Verified (values computed; PDF not generated, see fig2-pdf-not-produced) |
| TCCM avg rank 5.8 (AUPRC), 5.7 (AUROC) | `AggregateResults.py:270-272` (`order_roc`/`order_pr` mean ranks) | mean rank per detector | plausibly | Verified (computation present) |
| Full per-dataset AUROC/AUPRC tables (appendix) | `AggregateResults.py:269` `Results_{split}.csv` | mean/std AUC,PR | plausibly | Verified |
| Fig. 3 inference/training run time (1573×/85× speedups) | `FullExperiments.py:173,182` (train/test time) → `AggregateResults.py:225-230` `Mean_TestTime` etc. | time values in CSV | plausibly | Verified (values computed; plot script not in repo) |
| Table 5 per-dataset #Epochs / batch size | `FMAD/functions.py:73-268` `determine_FMAD_hyperparameters` | epoch/batch constants | ✓ (census=5, spambase=5000, musk=5, …) | Verified |
| Epoch-selection protocol (CSM criterion, App. B.6) that PRODUCES Table 5 | (none) | — | — | MISSING (see epoch-selection-protocol-absent) |
| Fig. 4 MNIST 1-vs-7 explanation, AUROC 0.76 | (none) | — | — | MISSING (see explainability-code-absent) |
| Table 3 ExactMatch / Jaccard explanation accuracy (App. D.4.2) | (none) | — | — | MISSING (see explainability-code-absent) |
| App. D.5 Friedman + Nemenyi tests, Figs 21/22 CD diagrams | (none) | — | — | MISSING (see stat-tests-code-absent) |
| Ablation: time embedding (Fig 12) | `AblationStudies.py:255-350` | AUROC/AUPRC bars | plausibly | Verified |
| Ablation: fixed-t sensitivity (Fig 13) | `AblationStudies.py:142-212` | AUROC/AUPRC vs t | plausibly | Verified |
| Ablation: noise injection (Fig 14) | `AblationStudies.py:360-449` | AUROC/AUPRC bars | plausibly | Verified |
| Ablation: training-contamination (Fig "Contamination_TCCM") | `AblationStudies.py:461-580` | AUROC/AUPRC vs ratio | plausibly | Verified |
| App. D.4 empirical robustness (PGD on synthetic GMM, FP/FN) | `RobustnessStudy.py` | AUROC/AUPRC vs eps | plausibly | Verified (synthetic only; real data raises NotImplementedError) |
| Contamination robustness top-10 models (App. D) | `ContaminationStudies.py` + `ProcessContamination.py` | AUROC/AUPRC vs ratio | plausibly | Verified |

## 3. Findings

## missing

```yaml finding
id: explainability-code-absent
category: missing
topic: "result traceability / explainability"
title: "No code for the explainability experiments (Fig 4 MNIST, Table 3 ExactMatch/Jaccard)"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  We use digit 1 as the normal class and digit 7
  as the anomaly (achieving an AUROC of 0.76). As shown in Figure 4, the model highlights the
  additional horizontal stroke that distinguishes 7 from 1
claim: "Explainability is one of the paper's three headline contributions (title: 'Explainable'); it is validated by the MNIST 1-vs-7 case study (Fig 4, AUROC 0.76) and the synthetic ExactMatch/Jaccard study (Table 3, App. D.4.2). No script in the repo loads MNIST 1/7, computes per-feature residual importance, selects top-k features, or computes ExactMatch/Jaccard; greps for 'jaccard', 'exact match', 'importance', and digit/MNIST attribution return nothing in the author code."
concern: "A headline contribution and its two reported result artefacts (Figure 4 and Table 3) cannot be reproduced because no code computes the feature-level attributions or the explanation-accuracy metrics."
resolution: "Authors: please add the explanation-attribution code (top-k feature selection from the residual vector, ExactMatch/Jaccard computation, the MNIST 1-vs-7 experiment) used for Figure 4 and Table 3."
cross_refs: ["§5.1", "App. D.4.2", "Table 3", "Figure 4"]
paper_ref: "Section 5.1 Effectiveness/Explainability; Appendix D.4.2; Table 3; Figure 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: epoch-selection-protocol-absent
category: missing
topic: "hyperparameter selection"
title: "Unsupervised epoch-selection procedure (CSM) not in repo; only hardcoded epochs shipped"
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
claim: "The paper (App. B.3/B.6) states the per-dataset training epochs were chosen by an unsupervised hyperparameter-selection method (Improved Contrast Score Margin / CSM criterion, Li et al. 2025b) that 'requires no access to anomaly labels' and is computed from the model-output distribution. The repo contains no implementation of this CSM search (greps for 'csm', 'contrast', 'margin', 'epoch select' return nothing); only the resulting per-dataset epoch constants are hardcoded in a long if/elif chain."
concern: "The data-dependent hyperparameter selection that the paper presents as 'principled and largely automated' cannot be reproduced or audited; a reviewer cannot check whether the epoch search was label-free or what data it was run on."
resolution: "Authors: please release the CSM-based epoch-selection script (including which dataset it evaluates the criterion on) so the Table 5 epoch values can be reproduced."
cross_refs: ["possible-test-set-epoch-selection", "App. B.3", "App. B.6", "Table 5"]
paper_ref: "Appendix B.3 (Configurations) and B.6 (Unsupervised Epoch Selection Strategy); Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stat-tests-code-absent
category: missing
topic: "statistical integrity"
title: "Friedman + Nemenyi significance tests (App. D.5, Figs 21/22) have no code"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  We report the results using critical difference diagrams (see Figure 21 and 22). For AUPRC, the
  Nemenyi test indicates that there are no statistically significant differences among the top-performing
  group, which includes TCCM (ranked 5.8)
claim: "The paper reports Friedman + Nemenyi tests on detector ranks (App. D.5) with critical-difference diagrams (Figs 21, 22). No script in the repo computes a Friedman or Nemenyi test or a CD diagram (greps for 'friedman', 'nemenyi', 'critical difference' return nothing in the author code); AggregateResults.py only computes mean ranks."
concern: "The reported significance test (and its conclusion that differences in the top group are not significant) cannot be reproduced from the released code."
resolution: "Authors: please add the script that performs the Friedman/Nemenyi tests and produces Figures 21/22 from the per-seed rank tables."
cross_refs: ["App. D.5"]
paper_ref: "Appendix D.5 Statistical Tests; Figures 21 and 22"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: semisup-loop-off-by-one
category: bug
topic: "experiment driver / reproduction"
title: "run_semisupervise.sh runs 6 of 7 force-inductive models; INNE_semisup skipped"
severity: medium
confidence: high
status: finding
file: bash_files/run_semisupervise.sh
line_start: 23
line_end: 23
quote: |
            for j in {45..50}; do # 7 models
claim: "MODEL_NAMES indices 45-51 are the seven force-inductive (semisup) baselines ABOD_semisup … INNE_semisup. The loop `for j in {45..50}` is inclusive 45..50 = 6 values, so index 51 (INNE_semisup) is never launched, despite the inline comment '7 models'. Confirmed by _audit_code/check_semisup_index_range.py. The same `{45..50}` appears at all four dataset blocks (lines 23, 58, 98, 138)."
concern: "INNE_semisup never produces a results file; under the default `--semi_only` aggregation it is back-filled with AUC=-1 (AggregateResults.py:198) and pinned to the worst rank for every dataset, distorting the 45-detector rank distribution (Figure 2) that includes TCCM."
resolution: "Change the loop bound to `{45..51}` (and the inline comment) so all seven force-inductive baselines run, then regenerate the Figure 2 rankings; or confirm INNE_semisup was run separately and supply its results."
cross_refs: ["Figure 2"]
check_script: _audit_code/check_semisup_index_range.py
paper_ref: "Figure 2 (45 anomaly detectors over 47 datasets)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: run-knn-dead-filename
category: bug
topic: "experiment driver"
title: "run_knn.sh invokes non-existent Full_experiments.py"
severity: low
confidence: high
status: finding
file: bash_files/run_knn.sh
line_start: 28
line_end: 28
quote: |
                nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
claim: "run_knn.sh calls `python Full_experiments.py` (underscore, lower-case 'e') at every dataset block, but the actual file is `FullExperiments.py` (camelCase, no underscore); no `Full_experiments.py` exists in the repo. The script would fail with 'No such file or directory' for every job."
concern: "This helper script cannot run as written; it will silently produce no KNN results (errors land only in per-job logs)."
resolution: "Rename the invocation to `python FullExperiments.py`. (KNN/KNN_semisup are also reachable via the main scripts, so impact on the paper is limited.)"
cross_refs: []
paper_ref: "Reproduction (KNN baseline)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: goad-nonzero-default-epochs
category: difference
topic: "baselines / hyperparameters"
title: "GOAD baseline given n_epoch=25 in code; paper says baselines use default configs"
severity: low
confidence: high
status: finding
file: FullExperiments.py
line_start: 263
line_end: 264
quote: |
        elif mname == "GOAD":
            parameters = {"n_epoch": 25}
claim: "FullExperiments.py overrides GOAD's number of epochs to 25; GOAD's own default is n_epoch=1 (baselines/goad.py:50,124). The paper states 'For all baseline detectors, we use their default configurations and hyperparameters as provided by their source implementations.'"
concern: "The paper's blanket claim of default baseline hyperparameters is contradicted for GOAD; the override raises GOAD's training (more epochs is generally favourable to the baseline), so it does not unfairly disadvantage GOAD, but the description is inaccurate."
resolution: "Authors: state that GOAD used n_epoch=25 (not its default 1), and confirm whether any other baseline deviates from defaults."
cross_refs: []
paper_ref: "Appendix B.3, 'For all baseline detectors, we use their default configurations…'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fig2-pdf-not-produced
category: difference
topic: "result aggregation / outputs"
title: "AggregateResults.py never writes Rank_ROC.pdf / Rank_PR.pdf the README promises"
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
claim: "The README's reproduction roadmap states `python AggregateResults.py --semi_only` produces `Rank_ROC.pdf` and `Rank_PR.pdf`, but the script imports no plotting library and writes only CSV files (confirmed: no matplotlib/savefig/.pdf in AggregateResults.py). The Figure 2 box plots themselves are not generated by the released code."
concern: "The plot artefact promised by the README is not produced; only the underlying rank values are written. (Per Rule G a plotting script is not required, since the values that drive Figure 2 are computed.)"
resolution: "Authors: either add the box-plot generation to AggregateResults.py or correct the README to list only the CSV outputs."
cross_refs: ["semisup-loop-off-by-one", "Figure 2"]
paper_ref: "README Reproduction Roadmap; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: possible-test-set-epoch-selection
category: methodology
topic: "hyperparameter tuning / leakage"
title: "Cannot rule out epoch selection on the test set (CSM criterion needs anomalies)"
severity: medium
confidence: low
status: question
file: paper.pdf
quote: |
  Within this space, we apply the unsupervised hyperparameter tuning
  method introduced by Li et al. (2025b), based on the Improved Contrast Score Margin (CSM) criterion.
  This criterion evaluates the margin between top-k predicted anomalous and normal samples solely
  from the distribution of model outputs, without requiring any ground-truth labels
claim: "The CSM criterion that selects the per-dataset epoch count is computed from the separation between top-k predicted anomalies and the rest of a dataset's scores. The training split in utils.load_adbench_npz contains only normal samples, so a meaningful 'top-k anomalous vs inlier' margin must be evaluated on data that contains anomalies — i.e. the test split. If epochs were chosen by maximising CSM on the test set, that is test-set-dependent model selection (a form of leakage) even though no labels are used. The selection code is not in the repo (see epoch-selection-protocol-absent), so the data it ran on cannot be confirmed."
concern: "If the per-dataset epochs (Table 5) were tuned to maximise score separation on the test set, the headline AUROC/AUPRC could be optimistically biased for TCCM relative to baselines that used fixed defaults."
resolution: "Authors: confirm on which split the CSM epoch-selection criterion was evaluated (held-out normal-only data vs the test set), and release that code; if a separate held-out fold was used, state how it was constructed."
cross_refs: ["epoch-selection-protocol-absent"]
paper_ref: "Appendix B.6 Epoch Selection Protocol"
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | Explainability experiments (headline) have no code; epoch-selection procedure and stat-tests also absent. |
| bug         | 2          | medium       | Off-by-one drops INNE_semisup from semisup runs; run_knn.sh uses a dead filename. |
| difference  | 2          | low          | GOAD given n_epoch=25 vs claimed defaults; promised Fig-2 PDFs not generated. |
| methodology | 1          | medium       | Possible test-set epoch selection — `question`, low confidence (selection code absent). |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. `[missing]` explainability-code-absent — Figure 4 and Table 3 (a headline "Explainable" contribution) have no code; not reproducible. (high / high)
2. `[bug]` semisup-loop-off-by-one — `{45..50}` runs 6 of 7 force-inductive baselines; INNE_semisup is dropped and back-filled to worst rank, distorting the Figure 2 rank distribution. (medium / high)
3. `[missing]` epoch-selection-protocol-absent — the unsupervised CSM epoch-selection that produced the only data-dependent hyperparameter is not implemented; only hardcoded epochs shipped. (medium / high)
4. `[methodology]` possible-test-set-epoch-selection — cannot rule out that epochs were tuned on the test set; needs author clarification. (medium / low, question)
5. `[missing]` stat-tests-code-absent — Friedman/Nemenyi tests and CD diagrams (App. D.5) have no code. (low / high)
6. `[bug]` run-knn-dead-filename — run_knn.sh calls a non-existent `Full_experiments.py`. (low / high)

### Items that genuinely look fine
- Data splitting (`utils.load_adbench_npz`): training is normal-only, train/test
  normal rows are disjoint, and the StandardScaler is fit on the training split
  only — no test-statistic leakage (verified, `_audit_code/check_split_and_scaler.py`).
- Table 5 per-dataset epoch/batch values match the hardcoded
  `determine_FMAD_hyperparameters` constants exactly.
- Core model (`FMAD/FlowMatchingAD.py`) matches Algorithms 1/2 and Eqs 4/5:
  MSE to target `dx_dt = -x`, score `||f(x,1)+x||_2` at fixed t=1.
- Main benchmark numbers and run-times are computed (not just plotted) in
  `FullExperiments.py` → `AggregateResults.py`; per-dataset AUROC/AUPRC tables and
  rank CSVs are produced.
- Dependencies are pinned in `requirements.txt` (minor: it lists torch==1.13.1
  while the paper says PyTorch 2.0 — trivial, not filed).
- Contamination train-anomalies and test-anomalies are disjoint halves
  (random_state=42), so the contamination study does not leak test anomalies.

### Open questions for the authors
- On which data split was the CSM epoch-selection criterion evaluated?
  (possible-test-set-epoch-selection / epoch-selection-protocol-absent)
- Was INNE_semisup actually included in the reported 45-detector ranking, given
  run_semisupervise.sh never launches it? (semisup-loop-off-by-one)
- Where is the code for Figure 4 / Table 3 and the Friedman/Nemenyi tests?
