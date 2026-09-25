# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (paper 2657)

## 1. Summary

The repository `code/ZhongLIFR__TCCM-NIPS/` is the official author code for the
NeurIPS 2025 paper (README title and BibTeX match `metadata.txt`; first author
"Zhong Li" matches the GitHub handle `ZhongLIFR`). It implements TCCM
(Time-Conditioned Contraction Matching), a one-step flow-matching semi-supervised
anomaly detector, and benchmarks it on 47 ADBench datasets against 44 baselines.

I verified the provenance, read the core method (`FMAD/FlowMatchingAD.py`,
`FMAD/functions.py`), the data loader/split (`utils.py`), the main driver
(`FullExperiments.py`), the aggregator (`AggregateResults.py`), the ablation,
robustness and contamination runners, and all five `bash_files/*.sh`. I
cross-checked the per-dataset epoch table against Table 5 of the paper, counted
datasets (47) and baselines (44) programmatically, and ran two `_audit_code/`
checks: a static arity check of the multiprocessing queue protocol and a
filesystem/grep sweep for the claimed-but-absent components (CSM epoch selection,
Friedman/Nemenyi tests, feature-attribution / MNIST explanation code).

The data split (train on 50% of normals, test on held-out normals + all anomalies,
StandardScaler fit on train only) is sound and faithful to the paper. The headline
results pipeline (Figure 2 rank box-plots, Figure 3 runtimes) is fully present and
traceable. The principal gaps are *missing* artefacts for claimed procedures and
sub-results: the unsupervised epoch-selection mechanism, the Friedman/Nemenyi
significance tests, and the feature-level explanation / MNIST experiment (Figure 4)
have no producing code. The only methodological concern is that TCCM's single key
hyperparameter (training epochs) is tuned per dataset (values 1–5000) while all
baselines run on fixed library defaults.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2a/2b detector rank box-plots (AUROC/AUPRC across 47 datasets) | `FullExperiments.py` (scores) → `AggregateResults.py:206-272` (ranks) | yes | rank methodology matches | Verified |
| TCCM avg rank AUROC=5.7 / AUPRC=5.8 | `AggregateResults.py:270-273` (rank means) | mechanism present; values not shipped | n/a (no precomputed CSV in repo) | Traceable, not re-run |
| Fig. 3a/3b/3c runtimes (train/test/total seconds) | `FullExperiments.py:160-188,313` (timings) → `AggregateResults.py:225-231` | yes | matches | Verified |
| 47 ADBench datasets, 4 size groups | `datasets/{small,medium,high_dim,large}` (47 .npz), `AggregateResults.py:6-16` | yes | 47 confirmed | Verified |
| 44 baselines + TCCM | model dicts in `FullExperiments.py:18-91` | yes | 44 confirmed | Verified |
| Table 5 per-dataset #Epochs / batch size | `FMAD/functions.py:73-268` | yes | values match Table 5 | Verified (values), but see F-missing-epoch-selection |
| Unsupervised CSM epoch-selection protocol (App. B.6) | (none) | — | — | MISSING (no selection code; only a hardcoded lookup) |
| Friedman + Nemenyi tests, critical-difference diagrams (Fig. 21/22, App. D.5) | (none) | — | — | MISSING (no statistical-test code) |
| Fig. 4 MNIST feature-level explanation (digit1 normal / digit7 anomaly, AUROC 0.76) | (none) | — | — | MISSING (no attribution / MNIST script) |
| App. D.4.2 synthetic attribution-accuracy study | (none) | — | — | MISSING (no script) |
| Empirical robustness (Fig. combined_FP_*, App. D.4) | `RobustnessStudy.py` (full PGD pipeline) | yes | matches | Verified |
| Ablations: time embedding (Fig.12), fixed-t (Fig.13), noise (Fig.14), contamination | `AblationStudies.py:142-609` | yes | matches | Verified |
| Contamination robustness, top-10 models (App. D / Fig.15) | `ContaminationStudies.py`, `ProcessContamination.py` | yes | matches | Verified |
| Proposition 1 (Lipschitz) robustness guarantee | theoretical (paper); empirically probed by `RobustnessStudy.py` | n/a | n/a | N/A (analytical) |

## 3. Findings

## missing

```yaml finding
id: epoch-selection-mechanism-absent
category: missing
topic: "hyperparameter tuning"
title: "Unsupervised CSM epoch-selection protocol is absent; epochs are a hardcoded per-dataset lookup"
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
claim: "TCCM's number of training epochs (the paper's only data-dependent hyperparameter) is supplied by a hardcoded if/elif lookup of 47 dataset-specific integer values (1 to 5000); no code implements the unsupervised epoch-selection procedure the paper says produced them."
concern: "The paper (App. B.6, lines 2760-2788) states epochs were chosen by an unsupervised, label-free 'Improved Contrast Score Margin (CSM)' criterion of Li et al. (2025b); a repo-wide grep finds no CSM / top-k / candidate-epoch / selection code, so the values cannot be reproduced or verified to be label-free, leaving open whether they were tuned against test-set AUROC."
resolution: "Authors: please release the CSM-based epoch-selection script (candidate space, scoring, per-dataset chosen value) so the hardcoded numbers in determine_FMAD_hyperparameters can be reproduced without anomaly labels."
cross_refs: ["asymmetric-epoch-tuning-vs-baselines", "B.6", "Table 5"]
check_script: _audit_code/check_missing_components.py
paper_ref: "Appendix B.6 'Unsupervised Epoch Selection Strategy'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: friedman-nemenyi-tests-missing
category: missing
topic: "statistical integrity"
title: "Friedman / Nemenyi significance tests and critical-difference diagrams have no code"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "First, we apply the Friedman test ... If the null hypothesis is rejected, we proceed with the Nemenyi post hoc test ... We report the results using critical difference diagrams (see Figure 21 and 22)."
claim: "The paper reports Friedman + Nemenyi post-hoc significance tests on detector rankings across 47 datasets and shows critical-difference diagrams (Figures 21, 22)."
concern: "A repo-wide grep finds no Friedman, Nemenyi, critical-difference, or post-hoc test code (the only scipy.stats import is invgamma in a baseline), so the reported significance analysis cannot be reproduced from the artefact."
resolution: "Authors: please add the script that runs the Friedman/Nemenyi tests on the aggregated ranks and renders Figures 21/22."
cross_refs: ["D.5"]
check_script: _audit_code/check_missing_components.py
paper_ref: "Appendix D.5 'Statistical Tests'; Figures 21, 22"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: explanation-mnist-code-missing
category: missing
topic: "explainability"
title: "Feature-level explanation and the MNIST illustration (Fig. 4, AUROC 0.76) have no producing code"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "We use digit 1 as the normal class and digit 7 as the anomaly (achieving an AUROC of 0.76). As shown in Figure 4, the model highlights the additional horizontal stroke that distinguishes 7 from 1 ... the residual vector itself encodes per-feature contributions to the anomaly score"
claim: "Explainability is a headline contribution: the paper reports a feature-wise attribution derived from the residual velocity field, illustrated on MNIST (digit 1 normal vs digit 7 anomaly, AUROC 0.76, Fig. 4) and validated by a synthetic study (App. D.4.2)."
concern: "No script in the repo computes per-feature importance, loads MNIST, or produces the Figure-4 explanation or the App. D.4.2 attribution-accuracy study; a grep for explain/attribution/feature-wise/heatmap/mnist returns only the dataset-name lookup table, so the explainability claims and the 0.76 number are not reproducible."
resolution: "Authors: please add the attribution / MNIST-explanation script and the App. D.4.2 synthetic-validation code."
cross_refs: []
check_script: _audit_code/check_missing_components.py
paper_ref: "Section 5 'Explainability'; Figure 4; Appendix D.4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: worker-error-tuple-arity-mismatch
category: bug
topic: "experiment driver"
title: "Worker exception path puts a 2-tuple but the parent unpacks 4 values"
severity: low
confidence: high
status: finding
file: FullExperiments.py
line_start: 119
line_end: 121
quote: |
      except Exception as e:
          print("ERROR!")
          queue.put(("ModERROR", str(e)))
claim: "On any exception raised inside model_worker outside train_and_eval's own try (e.g. limit_memory / OOM kill / import-time failure), the worker enqueues a 2-tuple ('ModERROR', str(e)), but run_model_with_timeout unpacks it as `status, result, train_time, test_time = queue.get()` (FullExperiments.py:134), which raises ValueError: not enough values to unpack (expected 4, got 2). Same pattern in ContaminationStudies.py:59 vs :72."
concern: "Instead of recording a failed run and continuing, the per-(dataset,model) process crashes on the unpack, so a model that fails for these reasons may silently produce no result rather than an explicit error row."
resolution: "Make the except branch enqueue a 4-tuple, e.g. queue.put(('ModERROR', str(e), None, None))."
cross_refs: []
check_script: _audit_code/check_queue_tuple_arity.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: run-knn-wrong-filename
category: bug
topic: "reproduction scripts"
title: "run_knn.sh invokes Full_experiments.py, which does not exist (file is FullExperiments.py)"
severity: low
confidence: high
status: finding
file: bash_files/run_knn.sh
line_start: 28
line_end: 28
quote: |
                nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
claim: "All four python invocations in run_knn.sh call `Full_experiments.py`; the actual driver is named `FullExperiments.py` (no underscore), so every job logs `python: can't open file 'Full_experiments.py'` and produces no result."
concern: "run_knn.sh is non-functional as written; the script is not in the README reproduction roadmap and KNN is also covered by run_main.sh, so impact is limited, but the auxiliary script cannot run."
resolution: "Rename the invoked file to FullExperiments.py in run_knn.sh (4 occurrences)."
cross_refs: []
check_script: _audit_code/check_missing_components.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: torch-version-mismatch
category: difference
topic: "environment / dependencies"
title: "requirements.txt pins torch==1.13.1 but the paper states PyTorch 2.0"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 7
line_end: 7
quote: |
  torch==1.13.1
claim: "The pinned dependency is torch==1.13.1, whereas the paper's Hardware and Software paragraph states the implementation is 'based on Python 3.9.21 with PyTorch 2.0'."
concern: "The exact environment that produced the reported numbers is ambiguous; the discrepancy is unlikely to change results (the model is a small MLP + sinusoidal embedding), so this is a low-severity faithfulness gap, not a correctness issue."
resolution: "Authors: reconcile the pinned torch version with the reported PyTorch 2.0, or confirm which version produced the paper's numbers."
cross_refs: []
paper_ref: "Appendix B.3 'Hardware and Software'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: asymmetric-epoch-tuning-vs-baselines
category: methodology
topic: "baselines / fair comparison"
title: "TCCM's epochs tuned per dataset (1-5000) while all baselines use fixed library defaults"
severity: medium
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
claim: "For TCCM the driver loads a per-dataset epoch count (47 hand-set values spanning 1 to 5000) via determine_FMAD_hyperparameters; for every other model parameters stays `{}` or a single fixed value, i.e. PyOD / source defaults, consistent with the paper's statement that for baselines 'we use their default configurations and hyperparameters'."
concern: "TCCM's single most impactful hyperparameter is adapted to each dataset while competitors are not, so the rank comparison is not under an equal tuning budget; the paper argues epochs were chosen label-free and that performance is on a stable plateau (Fig. 5), but several chosen values are very small (1, 2, 3, 5) where the plateau argument is weakest, and the selection cannot be checked because the mechanism is absent (see epoch-selection-mechanism-absent)."
resolution: "Authors: report TCCM under a single fixed epoch budget (matching the baselines' fixed-default regime) or apply the same dataset-wise unsupervised selection to the tunable baselines, and confirm the chosen small-epoch values do not lie on the steep part of the epoch-sensitivity curve."
cross_refs: ["epoch-selection-mechanism-absent"]
check_script: _audit_code/check_missing_components.py
paper_ref: "Appendix B.3 (baseline defaults); Table 5; Figure 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 3 | high | Epoch-selection mechanism, significance tests, and explanation/MNIST code all absent |
| bug | 2 | low | Worker error-tuple arity mismatch; run_knn.sh wrong filename |
| difference | 1 | low | torch pin (1.13.1) vs paper's PyTorch 2.0 |
| methodology | 1 | medium | TCCM epochs tuned per dataset vs fixed-default baselines |

## 5. Closing lists

### Top take-aways (ranked by severity x confidence)
1. [missing] `epoch-selection-mechanism-absent` (high/high) — the unsupervised CSM epoch-selection that the paper says produced TCCM's per-dataset epochs is not in the repo; only a hardcoded 47-entry lookup remains, so the choice cannot be reproduced or shown to be label-free.
2. [missing] `friedman-nemenyi-tests-missing` (medium/high) — the reported significance tests and CD diagrams (Fig. 21/22) have no code.
3. [missing] `explanation-mnist-code-missing` (medium/high) — the headline explainability result (Fig. 4 MNIST, AUROC 0.76) and App. D.4.2 validation have no producing script.
4. [methodology] `asymmetric-epoch-tuning-vs-baselines` (medium/medium) — TCCM's epochs are tuned per dataset (1-5000) while baselines use fixed defaults.
5. [bug] `worker-error-tuple-arity-mismatch` (low/high) — worker exception path enqueues a 2-tuple but the parent unpacks 4, crashing on certain failures instead of logging them.
6. [bug] `run-knn-wrong-filename` (low/high) — run_knn.sh calls a non-existent `Full_experiments.py`.

### Items that genuinely look fine
- Data split (`utils.py:19-54`): train on 50% of normals only, test = held-out normals + all anomalies, StandardScaler fit on train and applied to test — no preprocessing leakage; matches the paper's described protocol exactly.
- Anomaly score `||f(x,1)+x||` and training target `dx/dt = -x` (`FMAD/FlowMatchingAD.py:33-50`) implement Eq. 4/5 and Algorithms 1/2 faithfully.
- Rank-aggregation methodology (`AggregateResults.py:206-272`): per-dataset ranks from per-seed mean metrics, then averaged — matches the Figure-2 description.
- Runtime/scalability logging (`FullExperiments.py`) traces to Figure 3.
- Dataset count (47) and baseline count (44) verified programmatically and match the paper.
- Seeds 0-4 vary both the train/test split (`random_state=seed`) and model init via `set_seed`.
- Transductive baselines correctly fit on X_test (their canonical transductive use), with a separate forced-inductive variant set for the appendix study.

### Open questions for the authors
- Were the 47 hardcoded epoch values produced strictly by the label-free CSM criterion, or did any selection step consult test-set AUROC? (Decides whether `asymmetric-epoch-tuning-vs-baselines` is benign tuning or test-set leakage.)
- README line ~218 mentions editing `run_everything.sh` to change MEMORY_LIMIT/TIME_LIMIT, but no such file exists; which script is intended?
- The contamination study fixes the test split at `random_state=42` for all five seeds (`ContaminationStudies.py:168-169`), so the five repetitions vary only in model init — is that the intended variance source for the App. D contamination figure?
