# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (paper 2657)

## 1. Summary

The repository `code/ZhongLIFR__TCCM-NIPS/` is the official author code for this NeurIPS 2025
paper (README header and citation match the paper title/authors). TCCM is a one-step
flow-matching semi-supervised anomaly detector for tabular data. The core method is a tiny MLP
(`FMAD/FlowMatchingAD.py`, `FMAD/functions.py`) trained with an MSE "contraction" objective
(`f(x,t) ≈ −x`) and scored by `‖f(x,1)+x‖₂`. The repo also ships:

- `FullExperiments.py` — main benchmark driver (45 models × 47 ADBench datasets × 5 seeds),
  records AUROC/AUPRC and train/test/total wall-clock time per run.
- `AggregateResults.py` — aggregates per-seed `.npz` results into mean/std AUROC/AUPRC, per-dataset
  ranks, and mean run-times (CSV outputs).
- `AblationStudies.py` — fixed-`t` sensitivity, time-embedding, noise-injection, contamination
  ablations (Figures 12–14, Contamination_Figure_TCCM).
- `RobustnessStudy.py` — synthetic GMM PGD evasion-attack robustness (Appendix D.4.1, Fig 19/20).
- `ContaminationStudies.py` + `ProcessContamination.py` — contaminated-training robustness for top-10
  models.
- All 47 ADBench `.npz` datasets are committed; `requirements.txt` is fully pinned.

What I did: read every Python source file and all five bash drivers; located the headline claims in
`paper.pdf` (verified against `paper_text.txt` line anchors); and ran three deterministic checks under
`_audit_code/` — `check_epochs.py` (hardcoded epochs vs Table 5), `check_scripts.py` (bash-script
file/index sanity), with outputs in `_audit_code/out/`. The method code is complete, runnable and
methodologically sound. The two substantive gaps are (a) the **explainability** experiments — one of the
three title contributions, with quantitative Table 3 and Figure 4 — have **no code at all**, and (b) the
**unsupervised epoch-selection procedure** that the paper says produced the per-dataset epochs is
described but not implemented (the epochs are hardcoded). Several minor bash-script defects prevent
reproduction of the inductive-KNN supplementary row.

## 2. Result-traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Fig 2a/2b: AUROC/AUPRC rank box plots over 47 datasets | `FullExperiments.py` (AUROC/AUPRC per run) + `AggregateResults.py:220-272` (mean + ranks) | values computed; **plot PDF not produced** | n/a (no seeds run here) | Verified (values); plotting absent |
| Fig 3a/3b/3c: mean inference/training/total runtime, large datasets | `FullExperiments.py:296-313` (times recorded) + `AggregateResults.py:225-231` (Mean_*Time) | values computed; **plot PDF not produced** | n/a | Verified (values); plotting absent |
| §5.2 "1.50s on census", "4864.76× faster than KDE", "85.91× faster than LUNAR" | `FullExperiments.py` timing + `AggregateResults.py` Mean_*Time | computable from outputs | not re-run | Traceable |
| Table 5: per-dataset #Epochs / batch / lr | `FMAD/functions.py:73-268` (hardcoded) | yes | ✓ exact match (see `_audit_code/out/check_epochs.txt`) | Verified |
| Unsupervised epoch-selection (CSM criterion) that *produced* Table 5 epochs | (none) | — | — | **MISSING (no code)** |
| Fig 4: MNIST digit feature-wise explanations | (none) | — | — | **MISSING (no code)** |
| Table 3 (App D.4.2): ExactMatch / Jaccard / AUROC / AUPRC of feature attribution | (none) | — | — | **MISSING (no code)** |
| Fig 12 time-embedding ablation | `AblationStudies.py:255-350` | yes | not re-run | Traceable |
| Fig 13 fixed-`t` sensitivity | `AblationStudies.py:142-212` | yes | not re-run | Traceable |
| Fig 14 noise-injection ablation | `AblationStudies.py:360-449` | yes | not re-run | Traceable |
| Fig 19/20 (D.4.1) GMM PGD robustness | `RobustnessStudy.py:240-320` | yes | not re-run | Traceable |
| Contamination robustness (top-10 models) | `ContaminationStudies.py` + `ProcessContamination.py` | yes | not re-run | Traceable |
| Fig 1 conceptual 2D contraction-field visualization | (none) | — | — | MISSING (illustrative only) |
| Prop 1 (Lipschitz) / Prop 2 (GMM discrim.) proofs | paper analytical; partial empirical via `RobustnessStudy.py` | n/a | n/a | Analytical |

## 3. Findings

## missing

```yaml finding
id: explainability-experiments-no-code
category: missing
topic: "explainability / result traceability"
title: "No code for the feature-level explanation results (Fig 4, Table 3) — a headline contribution"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  Table 3: Quantitative evaluation of explanation accuracy on synthetic GMM anomalies.
  Setting ExactMatch Jaccard AUROC AUPRC
  5D 1.000 1.000 1.000 1.000
claim: "Explainability is one of the paper's three titular contributions; §5 (Fig 4, MNIST digit explanations) and Appendix D.4.2 (Table 3: ExactMatch=Jaccard=1.000 across 5D-25D synthetic GMM) report quantitative feature-attribution accuracy, but no script, function, or notebook in the repo computes per-feature importance, the Exact-Match/Jaccard metrics, or the MNIST explanation figure (grep for 'explain|feature.?wise|attribution|imshow|exactmatch|jaccard' returns no implementation)."
concern: "A headline quantitative result (Table 3) and the qualitative Fig 4 cannot be reproduced or checked because the explanation-generation and evaluation code is entirely absent from the repository."
resolution: "Authors: please add the script that (i) computes the per-feature residual-velocity importance scores, (ii) produces the MNIST explanations of Fig 4, and (iii) computes the Exact-Match / Jaccard numbers of Table 3 on the synthetic GMM data."
cross_refs: ["§5 (Explainability)", "Appendix D.4.2"]
check_script: _audit_code/check_scripts.py
paper_ref: "Figure 4; Table 3 (Appendix D.4.2)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unsupervised-epoch-selection-missing
category: missing
topic: "hyperparameter tuning"
title: "CSM unsupervised epoch-selection procedure described in paper is not implemented; epochs hardcoded"
severity: medium
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/FMAD/functions.py
line_start: 73
line_end: 134
quote: |
  def determine_FMAD_hyperparameters(dataset_name_raw):
      dataset_name = dataset_name_raw.lower()
      if "census" in dataset_name:
          epoch_size = 5
          batch_size = 1024
          learning_rate = 0.005
  ...
      elif "skin" in dataset_name:
          epoch_size = 110  # Chosen first from "100 or 1"
          batch_size = 1024
          learning_rate = 0.005
      elif "celeba" in dataset_name:
          epoch_size = 2  # Chosen first from "100 or 1"
          batch_size = 1024
          learning_rate = 0.005
claim: "The number of training epochs — the paper's only data-dependent hyperparameter — is returned by a hardcoded per-dataset if/elif ladder. The values match Table 5 exactly (verified: _audit_code/out/check_epochs.txt), but the 'unsupervised hyperparameter selection method ... based on the Improved Contrast Score Margin (CSM)' that the paper (Appendix B.6) says generated them is nowhere in the repo (grep for 'CSM|contrast.?score|margin|epoch.?select|unsupervised' finds no implementation)."
concern: "The paper presents the per-dataset epoch counts as the output of a principled label-agnostic CSM protocol, but only the final hardcoded numbers are shipped, so the claim that the epochs were chosen without using labels (and the whole tuning protocol) is unverifiable; in-code comments like 'Chosen first from \"100 or 1\"' suggest manual rather than automated selection."
resolution: "Authors: please release the CSM-based epoch-search script (candidate-epoch grid + T(f) criterion + selection) so reviewers can reproduce the Table 5 epoch values and confirm no labels were used."
cross_refs: ["§B.6", "Table 5"]
check_script: _audit_code/check_epochs.py
paper_ref: "Appendix B.6 'Unsupervised Epoch Selection Strategy'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: run-knn-wrong-script-name
category: bug
topic: "reproduction scripts"
title: "run_knn.sh calls non-existent Full_experiments.py and the wrong model index (50≠KNN)"
severity: low
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/bash_files/run_knn.sh
line_start: 23
line_end: 25
quote: |
            for j in {50..50}; do # 1 models
                core_id=$(( (i * 1 + j) % MAX_CORES ))

                # Run the process
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
claim: "run_knn.sh invokes `python Full_experiments.py`, but the file is named `FullExperiments.py` (camelCase, no underscore); the call fails with 'can't open file Full_experiments.py'. Moreover model index 50 maps to INNE_semisup, not KNN/KNN_semisup (KNN_semisup is index 51). Verified in _audit_code/out/check_scripts.txt."
concern: "The 'run KNN' helper script crashes immediately and, even if the filename were fixed, would run INNE rather than KNN, so the inductive-KNN benchmark row cannot be produced as written."
resolution: "Authors: rename the call to `FullExperiments.py` and change the model index to 51 (KNN_semisup) / 5 (transductive KNN) as intended."
cross_refs: ["semisup-missing-knn-index"]
check_script: _audit_code/check_scripts.py
paper_ref: "README 'Inductive-mode reviewer check'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: semisup-missing-knn-index
category: bug
topic: "reproduction scripts"
title: "run_semisupervise.sh range {45..50} omits KNN_semisup (index 51)"
severity: low
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/bash_files/run_semisupervise.sh
line_start: 28
line_end: 28
quote: |
            for j in {45..50}; do # 7 models
claim: "The loop comment says '7 models' but {45..50} is only six indices (ABOD/COF/LOF/PCA/KPCA/INNE _semisup). force_inductive index 51 = KNN_semisup is never launched, and run_knn.sh (the helper meant to cover KNN) runs index 50 (INNE_semisup) instead. So the force-inductive KNN result is produced by no provided script. Verified in _audit_code/out/check_scripts.txt."
concern: "One of the reviewer-requested inductive-mode rows (force-inductive KNN) cannot be reproduced from the shipped scripts because no driver ever runs model index 51."
resolution: "Authors: change the range to {45..51}, or confirm KNN_semisup was intentionally excluded."
cross_refs: ["run-knn-wrong-script-name"]
check_script: _audit_code/check_scripts.py
paper_ref: "README 'Additional Inductive-Mode Experiments'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: worker-error-tuple-arity
category: bug
topic: "error handling"
title: "model_worker error path puts a 2-tuple but consumer unpacks 4 values"
severity: low
confidence: medium
status: question
file: code/ZhongLIFR__TCCM-NIPS/FullExperiments.py
line_start: 119
line_end: 121
quote: |
    except Exception as e:
        print("ERROR!")
        queue.put(("ModERROR", str(e)))
claim: "If `model_worker` itself raises (e.g. `limit_memory` fails) it enqueues a 2-element tuple here, but the consumer `run_model_with_timeout` always unpacks 4 elements (`status, result, train_time, test_time = queue.get()`, FullExperiments.py:134), which would raise ValueError. In normal operation `train_and_eval` catches its own exceptions and returns a 4-tuple, so this path is only reached if memory-limit setup throws."
concern: "A failure during worker setup would crash the parent with an unhandled ValueError instead of recording a clean error row, but the trigger is narrow and does not affect successful runs."
resolution: "Authors: make the error enqueue a 4-tuple (e.g. `('ModERROR', str(e), None, None)`) to match the consumer arity."
cross_refs: []
paper_ref: "n/a (engineering)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## difference

```yaml finding
id: aggregate-no-figure-plots
category: difference
topic: "result figures"
title: "README claims AggregateResults.py produces Rank_ROC.pdf/Rank_PR.pdf, but it only writes CSVs"
severity: low
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/AggregateResults.py
line_start: 270
line_end: 273
quote: |
    order_roc = df_aggregated.groupby("Detector")["Rank_ROC"].mean().sort_values()
    order_roc.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank.csv"))
    order_pr = df_aggregated.groupby("Detector")["Rank_PR"].mean().sort_values()
    order_pr.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank_PR.csv"))
claim: "The README 'Reproduction Roadmap' and 'Result Aggregation' sections state that `python AggregateResults.py --semi_only` produces the AUROC/AUPRC box-plot PDFs `Rank_ROC.pdf` and `Rank_PR.pdf` (Figure 2), but the script ends at line 273 having written only CSVs; it imports no matplotlib/seaborn and contains no savefig/boxplot call (confirmed by grep)."
concern: "The Figure-2 box plots cannot be regenerated by the documented command; only the underlying ranking values (which Fig 2 visualizes) are produced, so the figure-generation step is undocumented/absent (the values themselves remain traceable)."
resolution: "Authors: add the box-plot plotting code to AggregateResults.py (or ship the plotting script), or correct the README to state that only CSVs are produced."
cross_refs: []
paper_ref: "Figure 2; README 'Result Aggregation and Visualization'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: torch-version-paper-vs-repo
category: difference
topic: "environment / dependencies"
title: "Paper states PyTorch 2.0; requirements.txt pins torch==1.13.1"
severity: low
confidence: high
status: finding
file: code/ZhongLIFR__TCCM-NIPS/requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "Appendix B.3 (Hardware and Software) states 'Our implementation is based on Python 3.9.21 with PyTorch 2.0', but the shipped requirements.txt pins torch==1.13.1 (and the README badge says Python 3.9)."
concern: "The reported software environment does not match the one needed to rebuild the repo's environment; both are individually valid but a reviewer following the paper would install a different PyTorch than the code was pinned against."
resolution: "Authors: reconcile the PyTorch version between the paper text and requirements.txt."
cross_refs: []
paper_ref: "Appendix B.3 'Hardware and Software'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding. The semi-supervised protocol in `utils.load_adbench_npz` is sound:
training uses only normal samples (50% of normals), the test set is the held-out normals plus all
anomalies, and the `StandardScaler` is fit on train and applied to test (no scaler leakage). Transductive
baselines are fit on the test set by design (the documented ADBench transductive protocol), and the
authors additionally provide `force_inductive` variants for a fair inductive comparison. The only
data-dependent hyperparameter (epochs) is disclosed in Table 5; its selection procedure being absent
from the code is filed as `missing` (see `unsupervised-epoch-selection-missing`), not `methodology`,
because nothing in the code can be shown to use labels.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 2          | high         | Explainability code absent (headline); CSM epoch-selection absent |
| bug         | 3          | low          | run_knn.sh broken; run_semisupervise omits KNN_semisup; worker tuple arity (question) |
| difference  | 2          | low          | Fig-2 plotting not in AggregateResults; torch version paper≠repo |
| methodology | 0          | -            | Split/scaling/transductive protocol checked and sound        |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing, high/high]** No code produces the explainability results — Fig 4 (MNIST) and Table 3
   (ExactMatch=Jaccard=1.000) — even though "Explainable" is a titular contribution
   (`explainability-experiments-no-code`).
2. **[missing, medium/high]** The unsupervised CSM epoch-selection protocol (Appendix B.6) that the
   paper says produced the per-dataset epochs is not implemented; epochs are hardcoded in
   `FMAD/functions.py` (values match Table 5) (`unsupervised-epoch-selection-missing`).
3. **[bug, low/high]** `run_knn.sh` calls non-existent `Full_experiments.py` and the wrong model index
   (`run-knn-wrong-script-name`).
4. **[bug, low/high]** `run_semisupervise.sh` range `{45..50}` never launches KNN_semisup (index 51)
   (`semisup-missing-knn-index`).
5. **[difference, low/high]** README says `AggregateResults.py` emits Rank_ROC.pdf/Rank_PR.pdf
   (Figure 2) but it only writes CSVs (`aggregate-no-figure-plots`).
6. **[difference, low/high]** Paper says PyTorch 2.0; repo pins torch==1.13.1
   (`torch-version-paper-vs-repo`).

### Items that genuinely look fine
- Semi-supervised split + StandardScaler fit-on-train (no scaler leakage), `utils.py:19-54`.
- 47 ADBench datasets are committed and split into small/medium/high_dim/large matching the paper.
- `requirements.txt` is fully version-pinned; all 13 additional baselines have scoring methods.
- Hardcoded TCCM epochs match Table 5 exactly (`_audit_code/out/check_epochs.txt`).
- Core TCCM training/scoring matches Algorithm 1/2 and Eq. 4/5 (`FMAD/FlowMatchingAD.py`).
- Contamination and synthetic-robustness pipelines (`ContaminationStudies.py`,
  `RobustnessStudy.py`, `ProcessContamination.py`) compute and plot their reported values.

### Open questions for the authors
- Were the Table 5 epochs truly chosen by the label-agnostic CSM criterion, or partly by hand (the
  comments "Chosen first from '100 or 1'" suggest manual choice)? Please release the selection script.
- Was KNN_semisup (force-inductive KNN) intentionally excluded from the inductive-mode experiments,
  or is it a script oversight?
