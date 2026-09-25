# Audit — TCCM: Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching (paper 2657)

## 1. Summary

The cloned repo `code/ZhongLIFR__TCCM-NIPS/` is the official author code (owner
`ZhongLIFR` = first author Zhong Li; README title and BibTeX match the paper).
It implements **TCCM** (Time-Conditioned Contraction Matching), a one-step
flow-matching anomaly detector, plus 44 baselines (vendored under `baselines/`
and via PyOD), benchmark drivers, and four study scripts (ablation, robustness,
contamination, aggregation). The 47 ADBench `.npz` datasets are bundled under
`datasets/`. `requirements.txt` is pinned.

What I did: read the paper (PDF + text extraction), the core method
(`FMAD/FlowMatchingAD.py`, `FMAD/functions.py`), the data pipeline (`utils.py`),
the benchmark driver (`FullExperiments.py`), the aggregation/ranking pipeline
(`AggregateResults.py`), the four study scripts, and the bash runners. I wrote
three deterministic checks under `_audit_code/`:
`check_model_index_range.py` (model-index ↔ bash-loop coverage),
`check_csm_absent.py` (search for the unsupervised epoch-selection method), and
`check_hparam_substring_collisions.py` (substring-dispatch safety in
`determine_FMAD_hyperparameters`). The split (50/50 on normal data),
StandardScaler-fit-on-train, 5-seed protocol, and the Figure-2 ranking pipeline
all match the paper's description; the data pipeline shows no leakage. Two
defects: (1) the unsupervised CSM epoch-selection procedure the paper relies on
is not in the repo (epochs are hardcoded constants); (2) an off-by-one shell
range silently drops one baseline (`INNE_semisup`) from the inductive-mode
supplementary experiment that the paper reports.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2a/2b rank distributions (AUPRC/AUROC, 45 detectors × 47 datasets × 5 seeds = 10,575 runs) | `FullExperiments.py` (per-run AUROC/AUPRC at L310-311) + `AggregateResults.py` (mean over seeds + per-dataset rank, L206-272) | Yes | Pipeline matches text | Verified (code present; raw results gitignored, not shipped) |
| TCCM avg rank 5.8 (AUPRC) / 5.7 (AUROC) | `AggregateResults.py:270-273` | Computes ranks | Cannot re-run at scale | Verified-by-construction |
| TCCM per-dataset #Epochs (Table 5) | `FMAD/functions.py:73-268` (`determine_FMAD_hyperparameters`) | Hardcoded constants | Values match Table 5 exactly | Verified (values) but see `csm-epoch-selection-missing` for the selection procedure |
| Fig. 3 efficiency (1573× / 4864× / 85× faster inference) | `FullExperiments.py:179-184` (test-time timing) + `AggregateResults.py:229-230` | Times measured & aggregated | Cannot re-run at scale | Verified-by-construction (code present) |
| Fig. 5 epoch-sensitivity (Appendix B.6) | `AblationStudies.py:142-211` (`run_fixed_t_sensitivity_analysis`) — actually plots t-sensitivity (Fig.13); epoch-sensitivity Fig.5 has no dedicated runner | partial | — | Question (no script clearly produces Fig. 5 epoch sweep) |
| Figs 12-14 ablations (time embedding / noise / contamination) | `AblationStudies.py:255-448` (savefig L349, L448, plus t-sens L211) | Yes | Matches | Verified (code present) |
| Fig. 15 training-contamination | `ContaminationStudies.py` + `ProcessContamination.py` | Yes | Matches | Verified (code present) |
| Appendix D.4 empirical robustness (combined_FP_*.pdf) | `RobustnessStudy.py:240-320` (PGD) | Yes | Matches | Verified (code present) |
| Appendix E inductive setting, incl. `INNE_semisup` (Fig. 23a/b) | `bash_files/run_semisupervise.sh` indices {45..50} | Partial | `INNE_semisup` (idx 51) never run | MISMATCH → `semisupervise-range-off-by-one` |
| Prop. 1 Lipschitz robustness; Prop. 2 discriminative behaviour | paper §4 (theory) | N/A (analytical) | — | N/A (no code artefact) |

## 3. Findings

## missing

```yaml finding
id: csm-epoch-selection-missing
category: missing
topic: "hyperparameter tuning / reproducibility"
title: "Unsupervised CSM-based epoch selection (the only data-dependent hyperparameter) is not in the repo"
severity: medium
confidence: high
status: finding
file: FMAD/functions.py
line_start: 73
line_end: 77
quote: |
  def determine_FMAD_hyperparameters(dataset_name_raw):
      dataset_name = dataset_name_raw.lower()
      if "census" in dataset_name:
          epoch_size = 5
          batch_size = 1024
claim: "The per-dataset training-epoch counts that TCCM uses are returned as hardcoded constants by a long if/elif chain keyed on the dataset name; no code computes them."
concern: "The paper states (Appendix B.6, Table 5) the epoch count—'the only data-dependent hyperparameter'—is chosen by the unsupervised Improved Contrast Score Margin (CSM) criterion of Li et al. (2025b) without using labels, but that selection procedure exists nowhere in the repo (a whole-repo grep finds zero CSM / contrast-score / epoch-selection code), so the fairness claim 'no label information used to optimize hyperparameters' cannot be verified."
resolution: "Authors: please add the CSM-based epoch-selection script (the candidate-epoch sweep and the T(f) computation) that produced the Table 5 values, or confirm it was run off-repo and provide it."
cross_refs: ["asymmetric-tuning-budget", "Appendix B.6", "Table 5"]
check_script: _audit_code/check_csm_absent.py
paper_ref: "Appendix B.6 'Unsupervised Epoch Selection Strategy'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: semisupervise-range-off-by-one
category: bug
topic: "experiment driver / inductive-mode experiment"
title: "run_semisupervise.sh loops j in {45..50}, silently skipping INNE_semisup (index 51)"
severity: low
confidence: high
status: finding
file: bash_files/run_semisupervise.sh
line_start: 23
line_end: 23
quote: |
              for j in {45..50}; do # 7 models
claim: "The inductive-mode reviewer-check driver iterates model indices 45..50 (6 models) across all four dataset blocks, but MODEL_NAMES index 51 is INNE_semisup; the comment says '7 models' yet only 6 are launched."
concern: "INNE is one of the seven transductive detectors the paper explicitly converts to inductive mode (Appendix E), and INNE_semisup appears in the reported inductive-setting ranking (Fig. 23a/b), so the shipped script cannot reproduce that detector's inductive results."
resolution: "Change the four `for j in {45..50}` loops to `{45..51}` so all seven force-inductive models (including INNE_semisup) are run."
cross_refs: ["Appendix E.1"]
check_script: _audit_code/check_model_index_range.py
paper_ref: "Appendix E, list 'ABOD, COF, LOF, PCA, KPCA, KNN, and INNE'; Fig. 23 ranking incl. INNE_semisup"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

(none)

## methodology

```yaml finding
id: asymmetric-tuning-budget
category: methodology
topic: "baselines / tuning fairness"
title: "TCCM uses per-dataset tuned epochs while all baselines use source defaults"
severity: medium
confidence: low
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
claim: "TCCM receives a dataset-specific epoch count (ranging 1–5000 across datasets, Table 5); every other model is instantiated with empty/default parameters (`parameters = {}`), and the paper states baselines 'use their default configurations and hyperparameters as provided by their source implementations'."
concern: "Tuning one design hyperparameter per dataset for the proposed method while leaving baselines at defaults is an asymmetric tuning budget that can inflate the proposed method's relative ranking; the paper argues this is fair because the epoch is chosen unsupervised, but that selection code is absent (see csm-epoch-selection-missing), so the asymmetry cannot be independently judged."
resolution: "Authors: report TCCM with a single fixed epoch budget across datasets (or apply the same unsupervised selection budget to tunable baselines), and quantify how much the per-dataset epoch tuning changes TCCM's average rank."
cross_refs: ["csm-epoch-selection-missing"]
paper_ref: "Appendix B.3 'For all baseline detectors, we use their default configurations'; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 1          | medium       | Unsupervised CSM epoch-selection procedure absent; epochs hardcoded |
| bug         | 1          | low          | Off-by-one shell range drops INNE_semisup from inductive-mode run   |
| difference  | 0          | -            | Split, scaler, metrics, and ranking pipeline match the paper        |
| methodology | 1          | medium       | Asymmetric tuning (TCCM per-dataset epochs vs default baselines) — question |

## 5. Closing lists

**Top take-aways** (≤6, ranked by severity × confidence):
1. (`missing`) The unsupervised CSM epoch-selection method that justifies TCCM's per-dataset epochs as label-free is not in the repo; epochs are hardcoded constants (`csm-epoch-selection-missing`). Medium / high.
2. (`methodology`) TCCM is per-dataset tuned (epochs 1–5000) while baselines run at defaults — possible asymmetric tuning, filed as a question pending the absent selection code (`asymmetric-tuning-budget`). Medium / low.
3. (`bug`) `run_semisupervise.sh` runs only 6 of the 7 force-inductive models; INNE_semisup (index 51) is never executed though it is reported in Appendix E (`semisupervise-range-off-by-one`). Low / high.

**Items that genuinely look fine** (actively checked):
- Train/test split: `utils.py:37-39` splits ONLY the normal data 50/50; anomalies go entirely to test (`utils.py:42-43`). No anomaly leaks into training.
- Preprocessing: StandardScaler is `fit_transform` on X_train and `transform` on X_test (`utils.py:46-48`) — no fit-on-full-data leakage, matching the paper.
- Hardcoded-epoch substring dispatch: `check_hparam_substring_collisions.py` shows the two name collisions (annthyroid⊃thyroid, cardiotocography⊃cardio) are resolved correctly because the specific key precedes the shorter one; returned epochs match Table 5.
- Table 5 epoch values are byte-consistent with the hardcoded values in `FMAD/functions.py`.
- Seeding: `utils.set_seed` seeds torch, cuda, numpy, random and sets cuDNN deterministic; 5 seeds [0..4] used as stated.
- Efficiency comparison fairness: the headline competitors (DTE-NonParametric, KDE, LUNAR) are NOT in `transductive_models`, so they train on X_train like TCCM (`FullExperiments.py:166-172`); inference time is measured around `decision_function` only.
- Dependencies: `requirements.txt` is pinned; vendored baselines only import numpy/scipy/sklearn/torch/tqdm/matplotlib/seaborn (all listed) and PyOD models via `pyod` (listed).

**Open questions for the authors**:
- Provide the CSM epoch-selection script (candidate-epoch sweep + T(f) margin) that produced Table 5, to confirm label-agnostic selection (`csm-epoch-selection-missing`).
- Quantify the effect of per-dataset epoch tuning on TCCM's average rank versus a single fixed epoch budget (`asymmetric-tuning-budget`).
- `requirements.txt` pins `torch==1.13.1` while the README/Appendix B.3 say PyTorch 2.0; which was used for the reported numbers? (Minor; not filed as a finding.)
- Is there a dedicated script for the Fig. 5 per-dataset epoch sweep, or was it produced ad hoc? (Minor traceability question.)
