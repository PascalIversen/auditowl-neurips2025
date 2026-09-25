# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (NeurIPS 2025, paper 2657)

## 1. Summary

The repository `code/ZhongLIFR__TCCM-NIPS/` is the official author code (README cites
the exact paper title, authors, OpenReview/arXiv links). It implements TCCM
(Time-Conditioned Contraction Matching), a one-step flow-matching anomaly detector,
and benchmarks it against 44 baselines on 47 ADBench datasets. Core method:
`FMAD/FlowMatchingAD.py` (training objective `MSE(f(x,t), -x)`, score `||f(x,1)+x||`)
and `FMAD/functions.py` (network + per-dataset hyperparameters). Experiment drivers:
`FullExperiments.py` (main Figure 2 benchmark), `RobustnessStudy.py` (synthetic PGD
robustness, App. D.4), `AblationStudies.py` (App. D.3 ablations), `ContaminationStudies.py`
(App. contamination), `AggregateResults.py` (ranks → Figure 2 / Figure 23), driven by
`bash_files/*.sh`.

What I did. I read every Python file and bash script, the README, and the relevant
paper sections (Experiment Setup §, App. B.3 hyperparameters + Table 5, App. B.5
algorithm, App. B.6 epoch-selection protocol, App. D.3 ablations, App. D.4 robustness/
explanation, App. D.5 statistical tests, App. E inductive setting). I wrote three
read-only checks under `_audit_code/`:
- `check_table5_epochs.py` — confirms all 47 per-dataset epoch values in the code match
  paper Table 5, and that NO label-free / CSM epoch-selection logic exists in the repo.
- `check_semisup_model_count.py` — confirms `run_semisupervise.sh` runs only 6 of 7
  force-inductive models, dropping `KNN_semisup` (index 51).
- `check_hyperparam_dispatch.py` — confirms all 47 dataset names resolve to a defined
  epoch (no substring-collision crash).

Headline result (Figure 2): TCCM has the best average rank over 47 datasets vs 44
baselines (AUROC rank 5.7, AUPRC rank 5.8). The paper itself reports that under the
Nemenyi post-hoc test these differences are NOT statistically significant within the
top group (App. D.5) — disclosed by the authors.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Per-run AUROC/AUPRC (all 45×47×5 runs) | `FullExperiments.py:310-312` | yes (`roc_auc_score`, `average_precision_score`) | n/a (raw) | Verified |
| Figure 2 average ranks (5.7 / 5.8) | `AggregateResults.py:206-272` | yes (rank over mean-over-seeds) | plausible | Verified (logic present) |
| Table 5 per-dataset #Epochs (47 rows) | `FMAD/functions.py:73-268` | yes (hardcoded) | 47/47 MATCH (`_audit_code/out/table5_epochs.csv`) | Verified |
| App. B.6 unsupervised (CSM) epoch-selection protocol | (none) | no | — | MISSING (no selection code) |
| Figure 3 runtimes / speedups (1573.39× etc.) | `AggregateResults.py:225-230` (Mean_*Time) | values yes; ratios+plot no | plausible | Partial (values traceable) |
| Figure 4 MNIST explanation (digit 1 vs 7) | (none) | no | — | MISSING |
| Table 3 explanation accuracy (ExactMatch/Jaccard, 5–25D) | (none) | no | — | MISSING |
| App. D.5 Friedman + Nemenyi tests, CD diagrams (Fig 21/22) | (none) | no | — | MISSING |
| App. D.3 Fig 12 time-embedding ablation | `AblationStudies.py:255-298` | yes | plausible | Verified (logic present) |
| App. D.3 Fig 13 fixed-t sensitivity | `AblationStudies.py:142-212` | yes | plausible | Verified (logic present) |
| App. D.3 Fig 14 noise injection | `AblationStudies.py:360-405` | yes | plausible | Verified (logic present) |
| App. D.3 Fig 16 feature-normalization ablation | (none) | no | — | MISSING |
| App. D.3 Fig 17 time-interpolated-inputs ablation | (none) | no | — | MISSING |
| App. D.4 synthetic PGD robustness (Fig 19/20) | `RobustnessStudy.py:240-320` | yes | plausible | Verified (logic present) |
| App. E inductive-setting ranks incl. `KNN_semisup` (Fig 23, KNN_semisup ranked ~3rd/4th) | `run_semisupervise.sh:23` | NO for KNN_semisup | — | BUG (index 51 never run) |

## 3. Findings

## missing

```yaml finding
id: csm-epoch-selection-protocol-missing
category: missing
topic: "hyperparameter selection / reproducibility"
title: "Paper's label-free (CSM) epoch-selection protocol is absent from the code"
severity: medium
confidence: high
status: finding
file: FMAD/functions.py
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
claim: "The only data-dependent hyperparameter (#epochs, ranging 1–5000 across datasets) is selected by a long hardcoded if/elif chain; no code computes the Improved Contrast Score Margin (CSM) criterion the paper says was used (App. B.6: 'we apply the unsupervised hyperparameter tuning method ... based on the Improved Contrast Score Margin (CSM) criterion ... For each candidate epoch, we compute T(f) and select the configuration maximizing this criterion')."
concern: "Because the selection step is not in the repo, a reviewer cannot verify the per-dataset epoch counts were chosen label-free as claimed (App. B.6); the values could equally have been chosen by inspecting test AUROC — and the code comments at FMAD/functions.py:128 and :133 ('Chosen first from \"100 or 1\"') hint at manual trial rather than the described automated criterion."
resolution: "Authors: please add the CSM-based epoch-search script (over the candidate space, using only model outputs / no labels) that produced the Table 5 epoch values, so the label-free claim is reproducible."
cross_refs: ["asymmetric-per-dataset-epoch-tuning"]
check_script: _audit_code/check_table5_epochs.py
paper_ref: "Appendix B.6 'Unsupervised Epoch Selection Strategy'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: statistical-tests-missing
category: missing
topic: "result traceability / statistical integrity"
title: "Friedman/Nemenyi significance tests and CD diagrams (App. D.5) have no code"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  "We report the results using critical difference diagrams (see Figure 21 and 22). For AUPRC, the Nemenyi test indicates that there are no statistically significant differences among the top-performing group"
claim: "The paper performs Friedman and Nemenyi post-hoc tests and renders critical-difference diagrams (Figures 21–22), but no script in the repo computes Friedman/Nemenyi/CD; a repo-wide grep finds only an unrelated scipy.stats.invgamma import in a baseline."
concern: "The reported significance verdicts (and the CD diagrams) cannot be reproduced from the released code."
resolution: "Authors: please add the script that runs the Friedman + Nemenyi tests on the per-dataset rank matrix and draws the CD diagrams."
cross_refs: []
paper_ref: "Appendix D.5 'Statistical Tests', Figures 21–22"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: explanation-accuracy-table3-missing
category: missing
topic: "result traceability / explainability"
title: "Table 3 explanation-accuracy and Figure 4 MNIST explanation have no code"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  Table 3: Quantitative evaluation of explanation accuracy on synthetic GMM anomalies.
claim: "Table 3 reports ExactMatch/Jaccard of feature attributions on synthetic GMM data (5D–25D), and Figure 4 shows MNIST digit-1-vs-7 explanations, but no script computes per-feature attributions, ExactMatch, Jaccard, or the MNIST explanation (grep for 'jaccard', 'exactmatch', 'importance', 'mnist' explanation code returns nothing relevant)."
concern: "The explainability claims (a headline contribution) are not reproducible from the released code; the per-feature residual decomposition is described but not implemented as a runnable artefact."
resolution: "Authors: please add the script computing the per-feature residual attribution and the ExactMatch/Jaccard metrics in Table 3, plus the MNIST explanation in Figure 4."
cross_refs: []
paper_ref: "Table 3; Figure 4; Appendix D.4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-figs-16-17-missing
category: missing
topic: "ablations"
title: "Feature-normalization (Fig 16) and time-interpolated-inputs (Fig 17) ablations have no code"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  "(4) Feature normalization (Figure 16): z-score normalization is generally beneficial and improves robustness; (5) Time-interpolated inputs (Figure 17): interpolation offers no gain and may add noise"
claim: "AblationStudies.py implements the time-embedding (Fig 12), fixed-t (Fig 13), noise-injection (Fig 14) and contamination ablations, but contains no routine for the feature-normalization ablation (Fig 16) or the time-interpolated-inputs ablation (Fig 17); a grep for normalization/interpolation ablation code finds none."
concern: "Two of the six reported ablation studies cannot be reproduced from the released code."
resolution: "Authors: please add the Figure 16 (normalization variants) and Figure 17 (time-interpolated inputs) ablation scripts."
cross_refs: []
paper_ref: "Section 'Ablation Studies and Sensitivity Analysis'; Figures 16, 17 (App. D.3)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: semisup-knn-not-run
category: bug
topic: "experiment driver / reproducibility"
title: "run_semisupervise.sh runs only 6 of 7 force-inductive models, dropping KNN_semisup (a top-ranked detector)"
severity: medium
confidence: high
status: finding
file: bash_files/run_semisupervise.sh
line_start: 23
line_end: 23
quote: |
            for j in {45..50}; do # 7 models
claim: "The loop `for j in {45..50}` iterates indices 45..50 (6 values) but the comment says '7 models'; the 7 force_inductive models occupy MODEL_NAMES indices 45..51, so index 51 = KNN_semisup is never launched. This is repeated in all four dataset blocks (lines 23, 58, 98, 138)."
concern: "App. E / Figure 23 report KNN_semisup as a top detector (AUROC rank ~3rd, AUPRC rank ~4th — TCCM's closest competitor in the inductive setting), yet the provided script cannot produce its result files, so the inductive-setting figure is not reproducible as released."
resolution: "Change the loop bound to `{45..51}` (or `for j in $(seq 45 51)`) so KNN_semisup (index 51) is included; confirm whether the published Figure 23 KNN_semisup numbers were produced by a different invocation."
cross_refs: ["semi-only-default-vs-figure2-mode"]
check_script: _audit_code/check_semisup_model_count.py
paper_ref: "Appendix E, Figure 23 (KNN_semisup ranked ~3rd/4th)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: semi-only-default-vs-figure2-mode
category: difference
topic: "evaluation consistency (README vs paper figure)"
title: "Documented aggregation command (--semi_only) reproduces App. E Fig 23, not main-text Fig 2"
severity: low
confidence: medium
status: finding
file: AggregateResults.py
line_start: 102
line_end: 103
quote: |
    argp.add_argument("--semi_only", action="store_true", help="If true, only semi-supervised models are considered.")
    argp.set_defaults(semi_only=True)
claim: "AggregateResults defaults to semi_only=True, which drops the 7 transductive models and includes the 7 force_inductive (_semisup) variants. The README's reproduction roadmap lists `python AggregateResults.py --semi_only` for the main Rank_ROC.pdf / Rank_PR.pdf. But the main-text Figure 2 leaderboard lists the transductive detectors (KNN, ABOD, LOF, PCA, COF, KPCA, INNE) and labels them 'Classical (Transductive)', i.e. it corresponds to the NON-semi_only (default-mode) aggregation; the _semisup variants only appear in App. E Figure 23."
concern: "A user following the README's documented `--semi_only` command obtains the appendix inductive-setting ranking (Fig 23), not the headline Figure 2; the mapping from command to figure is ambiguous (the README does note removing the flag, but does not say which figure each mode yields)."
resolution: "Authors: please state explicitly which command/flag reproduces Figure 2 vs Figure 23 (Figure 2 appears to require running AggregateResults WITHOUT --semi_only after run_main.sh)."
cross_refs: ["semisup-knn-not-run"]
paper_ref: "Figure 2 (transductive detectors shown) vs Appendix E Figure 23 (_semisup variants)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: asymmetric-per-dataset-epoch-tuning
category: methodology
topic: "baselines / fair comparison"
title: "TCCM gets per-dataset epoch tuning (1–5000) while all baselines use single fixed defaults"
severity: medium
confidence: medium
status: question
file: FullExperiments.py
line_start: 254
line_end: 258
quote: |
    elif mname == "TCCM":
        from FMAD.functions import determine_FMAD_hyperparameters
        hyparam = determine_FMAD_hyperparameters(dataset_name)
        parameters.update({"n_features": X_train.shape[1]})
        parameters.update(hyparam)
claim: "For TCCM the code selects a dataset-specific epoch count (per FMAD/functions.py, ranging from 1 to 5000 across the 47 datasets); for every baseline the code passes no per-dataset hyperparameters (parameters={} or a single fixed value such as GOAD n_epoch=25), matching the paper's statement 'For all baseline detectors, we use their default configurations and hyperparameters as provided by their source implementations.'"
concern: "TCCM's principal training hyperparameter is adapted per dataset while baselines' analogous hyperparameters are not, which can advantage TCCM in the head-to-head ranking; even if the adaptation is label-free as claimed, no baseline is given an equivalent per-dataset adaptation budget."
resolution: "Authors: either apply the same label-free per-dataset selection budget to the deep baselines' key hyperparameters, or report TCCM under a single fixed epoch count to confirm the ranking advantage is not an artefact of asymmetric tuning. Filed as a question because the epoch-selection code is absent (see csm-epoch-selection-protocol-missing) so the leakage-vs-fair question cannot be settled from the repo."
cross_refs: ["csm-epoch-selection-protocol-missing"]
paper_ref: "Appendix B.3 (TCCM tuning) vs 'For all baseline detectors, we use their default configurations'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 4          | medium       | CSM epoch-selection protocol, statistical tests, Table 3/Fig 4 explanations, Fig 16/17 ablations have no code |
| bug         | 1          | medium       | run_semisupervise.sh drops KNN_semisup (index 51), a top-ranked inductive-setting detector |
| difference  | 1          | low          | README's `--semi_only` aggregation maps to App. E Fig 23, not main Fig 2 |
| methodology | 1 (question)| medium      | Per-dataset epoch tuning for TCCM vs fixed defaults for all baselines   |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[bug]** `run_semisupervise.sh:23` uses `for j in {45..50}` (comment says "7 models") — index 51 = `KNN_semisup` is never run, yet App. E Figure 23 reports it as a top-3/top-4 detector; the inductive-setting figure is not reproducible as released. (`semisup-knn-not-run`)
2. **[missing]** The label-free CSM epoch-selection protocol (App. B.6) that justifies the per-dataset epoch values is absent; only the hardcoded results (Table 5) are in the repo. (`csm-epoch-selection-protocol-missing`)
3. **[methodology, question]** TCCM gets per-dataset epoch tuning (1–5000) while baselines use single fixed defaults; fair-comparison concern that cannot be settled from the repo. (`asymmetric-per-dataset-epoch-tuning`)
4. **[missing]** Table 3 explanation-accuracy metrics and the Figure 4 MNIST explanation — a headline "explainability" contribution — have no computing code. (`explanation-accuracy-table3-missing`)
5. **[missing]** Friedman/Nemenyi tests and CD diagrams (App. D.5, Fig 21/22) have no code. (`statistical-tests-missing`)
6. **[difference]** The README's documented `--semi_only` aggregation reproduces App. E Fig 23, not the main-text Fig 2. (`semi-only-default-vs-figure2-mode`)

### Items that genuinely look fine
- **Split & no preprocessing leakage**: `utils.py:37-48` does a stratified 50/50 split of normal data, builds the test set from held-out normals + all anomalies, and fits the `StandardScaler` on `X_train` only (then transforms test) — matches the paper's stated protocol with no train/test scaler leakage.
- **All 47 datasets and 45 main models present/wired**: 12+15+9+11 = 47 unique `.npz` files; `run_main.sh` runs `j in {0..44}` = 45 detectors.
- **Per-dataset epochs match Table 5 exactly**: 47/47 MATCH (`_audit_code/out/table5_epochs.csv`).
- **No hyperparameter-dispatch crash**: all 47 dataset names resolve to defined epochs despite substring matching (cardiotocography checked before cardio); `_audit_code/out/hyperparam_dispatch.csv`.
- **Seeding is comprehensive**: `utils.set_seed` seeds torch/cuda/numpy/random and sets cuDNN deterministic; runs repeated over 5 seeds.
- **Dependencies pinned**: `requirements.txt` pins all versions (note torch==1.13.1 vs paper's "PyTorch 2.0" — a benign documentation mismatch, method does not depend on a 2.x-only feature).
- **TCCM method matches Algorithm 1/2 and Eqs. 4–5**: `FMAD/FlowMatchingAD.py` trains `MSE(f(x,t), -x)` and scores `||f(x,1)+x||`, exactly as the pseudo-code and equations specify.

### Open questions for the authors
- (high-severity / lower-confidence) Were the per-dataset epoch counts in Table 5 selected purely from the label-free CSM criterion (no test-AUROC inspection)? The selection code is not in the repo, and comments at `FMAD/functions.py:128,133` suggest manual choices. (`csm-epoch-selection-protocol-missing`, `asymmetric-per-dataset-epoch-tuning`)
- Which exact command reproduces Figure 2 vs Figure 23, and were the Figure 23 `KNN_semisup` numbers produced by a script other than the released `run_semisupervise.sh`? (`semisup-knn-not-run`, `semi-only-default-vs-figure2-mode`)
