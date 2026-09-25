# Human-eval worksheet — #2657 · 2657

**16 distinct defects** (the 10 PDF+text audit runs' findings, merged by defect). Detection = how many of the 10 runs surfaced the defect (high = robust; 1 = one run only). Severity & confidence are the auditor's own labels (spread shown where runs disagreed); the wording/quote is taken from the highest-confidence run that cited code.

Tick **one** box per defect (put an `x`):

- **correct & relevant** — true *and* a substantive reproducibility issue worth raising
- **correct but wrong severity** — true and worth raising, but the severity label is miscalibrated (e.g. an out-of-the-box crash with a trivial fix tagged high that's really low/medium)
- **correct but not relevant** — technically true but trivial / nitpick / already acknowledged
- **unsure** — can't decide without resources beyond the frozen repo + paper
- **false** — the claim misreads the code/paper and does not hold

Frozen code: `2657/code_frozen/`  ·  paper: `audits/2657/paper.pdf`

---

### F01 · TCCM's training-epoch count (the paper's only data-dependent hyperparameter) is a hardcoded 47-branch if/elif table (values 1..5000); no code implements the unsupervised CSM / Improved Contrast Score Margin selection the paper claims (some values commented 'Chosen first from "100 or 1"')

_category: Missing code / data · topic: hyperparameter tuning_

**severity: medium  (varied: high, medium)  ·  confidence: high  (varied: high, medium)  ·  detection: 10/10 runs**

- **Claim:** The number of training epochs — the paper's only data-dependent hyperparameter — is returned by a hardcoded per-dataset if/elif ladder. The values match Table 5 exactly (verified: _audit_code/out/check_epochs.txt), but the 'unsupervised hyperparameter selection method ... based on the Improved Contrast Score Margin (CSM)' that the paper (Appendix B.6) says generated them is nowhere in the repo (grep for 'CSM|contrast.?score|margin|epoch.?select|unsupervised' finds no implementation).
- **Concern:** The paper presents the per-dataset epoch counts as the output of a principled label-agnostic CSM protocol, but only the final hardcoded numbers are shipped, so the claim that the epochs were chosen without using labels (and the whole tuning protocol) is unverifiable; in-code comments like 'Chosen first from "100 or 1"' suggest manual rather than automated selection.
- **Ask:** Authors: please release the CSM-based epoch-search script (candidate-epoch grid + T(f) criterion + selection) so reviewers can reproduce the Table 5 epoch values and confirm no labels were used.
- **Evidence:** `code/ZhongLIFR__TCCM-NIPS/FMAD/functions.py:73-134` · paper: Appendix B.6 'Unsupervised Epoch Selection Strategy'; Table 5
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r10#1)
- **Quoted at `code/ZhongLIFR__TCCM-NIPS/FMAD/functions.py:73-134`:**
```
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
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
in the paper the authors write "we ultimately fix the selected epoch across all seeds… for simplicity and reproducibility.". But if reproducibility is the issue, why not push the code for the selection?
---

### F02 · Explainability — a titular contribution (RQ3) — has no code: no per-feature residual attribution, no MNIST 1-vs-7 figure (Fig 4, AUROC 0.76), no ExactMatch/Jaccard metrics (Table 3); the model returns only a scalar L2 score

_category: Missing code / data · topic: explainability / result traceability_

**severity: high  (varied: high, medium, low)  ·  confidence: high  ·  detection: 9/10 runs**

- **Claim:** The model only returns a per-sample scalar anomaly score (the L2 norm collapses the per-feature residual vector). No code anywhere in the repo computes the per-feature residual components, the ExactMatch/Jaccard explanation metrics of Table 3, the synthetic-GMM explanation experiment, the 2D contraction-vector visualisation (Fig. 1), or the MNIST 1-vs-7 feature attribution (Fig. 4).
- **Concern:** Explainability is a headline contribution (it is in the paper title and has a dedicated quantitative table); none of its reported numbers or figures can be reproduced because the producing code is absent.
- **Ask:** Provide the script(s) that (a) build the synthetic GMM with known shifted dimensions, (b) compute per-feature residual attributions, and (c) compute ExactMatch and Jaccard against ground-truth dimensions, plus the Fig. 1 / Fig. 4 generators.
- **Evidence:** `FMAD/FlowMatchingAD.py:40-53` · paper: Table 3 (App. D.4.2); Fig. 1; Fig. 4; Eq. 5
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r09, r10  (representative: r05#0)
- **Quoted at `FMAD/FlowMatchingAD.py:40-53`:**
```
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
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F03 · No code for the Friedman + Nemenyi post-hoc significance tests / critical-difference diagrams (Figs 21-22, App D.5); aggregation stops at mean per-detector ranks

_category: Missing code / data · topic: statistical integrity_

**severity: medium  (varied: medium, low)  ·  confidence: high  ·  detection: 7/10 runs**

- **Claim:** The paper conducts Friedman + Nemenyi critical-difference significance tests over the 45-detector / 47-dataset rankings (Appendix D.5), but a whole-repo scan finds no code for friedman, nemenyi, or critical-difference.
- **Concern:** The reported significance of TCCM's ranking advantage cannot be reproduced or checked from the released code.
- **Ask:** Authors: please release the script that runs the Friedman test and Nemenyi/CD post-hoc analysis on the per-dataset mean ranks.
- **Evidence:** `paper.pdf` · paper: Appendix D.5 'Statistical Tests' (lines 4794-4808)
- **Found in runs:** r01, r03, r04, r05, r06, r07, r09  (representative: r03#2)
- **Quoted at `paper.pdf`:**
```
"we apply the Friedman test ... to determine whether there is any statistically significant
difference in performance rankings among all methods. If the null hypothesis is rejected, we
proceed with the Nemenyi post hoc test ... Two methods are considered significantly
different if their average ranks differ by at least the critical difference (CD)."
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
The inputs for these tests are produced by AggregateResults.py but not code for the tests (easy to fix because the code would be a standard testing code). Severity wise it could be argued for low.
---

### F04 · requirements.txt pins torch==1.13.1 while the paper (App B.3) and README state PyTorch 2.0 / Python 3.9.21

_category: Paper–code mismatch · topic: environment / dependencies_

**severity: low  ·  confidence: high  ·  detection: 7/10 runs**

- **Claim:** The pinned dependency is torch==1.13.1, whereas the paper's Hardware and Software paragraph states the implementation is 'based on Python 3.9.21 with PyTorch 2.0'.
- **Concern:** The exact environment that produced the reported numbers is ambiguous; the discrepancy is unlikely to change results (the model is a small MLP + sinusoidal embedding), so this is a low-severity faithfulness gap, not a correctness issue.
- **Ask:** Authors: reconcile the pinned torch version with the reported PyTorch 2.0, or confirm which version produced the paper's numbers.
- **Evidence:** `requirements.txt:7` · paper: Appendix B.3 'Hardware and Software'
- **Found in runs:** r01, r02, r03, r04, r05, r07, r10  (representative: r01#5)
- **Quoted at `requirements.txt:7`:**
```
torch==1.13.1
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
I do not think the difference in the version will affect the results at all. Maybe it was just changed for some tests. Very minor issue. Readme doesn't state PyTorch 2.0 but paper does indeed.
---

### F05 · run_semisupervise.sh loops `for j in {45..50}` (6 indices) but comments '7 models'; the 7th force-inductive model at index 51 is never launched

_category: Technical bug · topic: reproducibility / experiment harness_

**severity: low  (varied: medium, low)  ·  confidence: high  ·  detection: 6/10 runs**

- **Claim:** MODEL_NAMES has 52 entries (verified by AST). The 7 force-inductive models occupy indices 45-51, with KNN_semisup at index 51. run_semisupervise.sh loops j in {45..50} (only 6 indices), so KNN_semisup (51) is never launched even though the comment says '7 models'. The auxiliary run_knn.sh invokes `python Full_experiments.py` (line 28 etc.), but the file is named `FullExperiments.py` and `Full_experiments.py` does not exist; its loop {50..50} targets index 50 (INNE_semisup), not KNN.
- **Concern:** The inductive-evaluation appendix (Fig. 23) reports a KNN_semisup result, but as scripted no run produces it, and the dedicated run_knn.sh fails with FileNotFoundError; the supplementary inductive figure cannot be reproduced as released.
- **Ask:** Change the loop to {45..51} (and fix the comment / core_id divisor) and rename `Full_experiments.py` to `FullExperiments.py` in run_knn.sh.
- **Evidence:** `bash_files/run_semisupervise.sh:23-28` · paper: Appendix E (Results under the Inductive Evaluation Setting); Figure 23
- **Found in runs:** r05, r06, r07, r08, r09, r10  (representative: r07#3)
- **Quoted at `bash_files/run_semisupervise.sh:23-28`:**
```
for j in {45..50}; do # 7 models
    core_id=$(( (i * 6 + j) % MAX_CORES ))

    # Run the process
    echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
    nohup taskset -c "$core_id" python FullExperiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
can be easily fixed but correct and a reproducibility nuisance. It is a problem for a appendix figure. borderline correct but not relevant.
---

### F06 · run_knn.sh invokes `python Full_experiments.py` but the driver is `FullExperiments.py` (camelCase) — file-not-found on every call; model index 50 also maps to INNE_semisup, not KNN

_category: Technical bug · topic: reproduction scripts_

**severity: medium  ·  confidence: high  ·  detection: 6/10 runs**

- **Claim:** run_knn.sh invokes `python Full_experiments.py` (file does not exist; the real driver is `FullExperiments.py`, verified by _audit_code/check_inventory.py) and uses model index 50, which is `INNE_semisup`, not KNN (KNN is index 16 and KNN_semisup is index 51).
- **Concern:** The script as written cannot run (missing file) and, even if the filename were fixed, would evaluate INNE rather than KNN; the standalone KNN runner is effectively non-functional.
- **Ask:** Rename the called file to FullExperiments.py and set the model index to the intended KNN entry (16 for transductive KNN, or 51 for KNN_semisup).
- **Evidence:** `bash_files/run_knn.sh:23-29` · paper: README 'Reproduction Roadmap'
- **Found in runs:** r01, r02, r04, r05, r09, r10  (representative: r05#3)
- **Quoted at `bash_files/run_knn.sh:23-29`:**
```
for j in {50..50}; do # 1 models
    core_id=$(( (i * 1 + j) % MAX_CORES ))

    # Run the process
    echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
    nohup taskset -c "$core_id" python Full_experiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
done
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F07 · Only TCCM gets per-dataset tuned epochs (1..5000) via determine_FMAD_hyperparameters; every baseline runs its source-default config — fair-comparison concern (paper does state baselines use defaults)

_category: Methodology / validity · topic: baselines / fair comparison_

**severity: medium  (varied: high, medium)  ·  confidence: medium  (varied: medium, low)  ·  detection: 5/10 runs**

- **Claim:** For TCCM the driver loads a per-dataset epoch count (47 hand-set values spanning 1 to 5000) via determine_FMAD_hyperparameters; for every other model parameters stays `{}` or a single fixed value, i.e. PyOD / source defaults, consistent with the paper's statement that for baselines 'we use their default configurations and hyperparameters'.
- **Concern:** TCCM's single most impactful hyperparameter is adapted to each dataset while competitors are not, so the rank comparison is not under an equal tuning budget; the paper argues epochs were chosen label-free and that performance is on a stable plateau (Fig. 5), but several chosen values are very small (1, 2, 3, 5) where the plateau argument is weakest, and the selection cannot be checked because the mechanism is absent (see epoch-selection-mechanism-absent).
- **Ask:** Authors: report TCCM under a single fixed epoch budget (matching the baselines' fixed-default regime) or apply the same dataset-wise unsupervised selection to the tunable baselines, and confirm the chosen small-epoch values do not lie on the steep part of the epoch-sensitivity curve.
- **Evidence:** `FullExperiments.py:254-266` · paper: Appendix B.3 (baseline defaults); Table 5; Figure 5
- **Found in runs:** r01, r02, r05, r06, r08  (representative: r01#6)
- **Quoted at `FullExperiments.py:254-266`:**
```
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
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
I would even argue for severity high. The comparison is not fair because the baselines are not tuned per dataset while the model is.
---

### F08 · The CSM epoch criterion compares top-k 'predicted anomalies' vs presumed inliers, but the training split is normal-only — the criterion needs anomalies, implying it is computed on the test set (leakage)

_category: Methodology / validity · topic: hyperparameter tuning / data leakage_

**severity: medium  ·  confidence: low  ·  detection: 3/10 runs**

- **Claim:** Training data is normal-only (no anomalies). The paper's CSM epoch criterion T(f) compares top-k 'predicted anomalies' against presumed inliers, which requires a set that actually contains anomalies — i.e. the test set, since X_train has none. The selection code is absent (see csm-epoch-selection-absent), so I cannot confirm which split it used.
- **Concern:** If the per-dataset epoch count was selected by maximising a criterion computed on the anomaly-containing test set, then model selection is test-set-dependent even though it is label-agnostic, which can bias the reported metrics upward.
- **Ask:** Authors: clarify on which data the CSM criterion was evaluated for epoch selection (held-out portion of training normals vs the test set) and release that code.
- **Evidence:** `utils.py:33-43` · paper: Appendix B.6 (CSM criterion T(f)); utils.load_adbench_npz
- **Found in runs:** r04, r05, r09  (representative: r05#7)
- **Quoted at `utils.py:33-43`:**
```
# Train using only normal samples
X_normal, X_anomalous = X[y == 0], X[y == 1]
y_normal, y_anomalous = y[y == 0], y[y == 1]

X_train, X_test_normal, y_train, y_test_normal = train_test_split(
    X_normal, y_normal, test_size=test_size, random_state=random_state, stratify=y_normal
)

# Test set contains both normal and abnormal data
X_test = np.vstack((X_test_normal, X_anomalous))
y_test = np.concatenate((y_test_normal, y_anomalous))
```

**Verdict:**   correct & relevant `[]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[x]`   false `[ ]`

**Notes:**
"the criterion needs anomalies" is not true. It uses predicted anomalies. However, if epoch selection was test-set-dependent, it could inflate results so the claim might be relevant. Partly correct, partly false... unsure. The agent formulates as a question with low confidence which is good but the main finding is a bit misformulated. This issue could be easily resolved if the code would be complete.
---

### F09 · README says `AggregateResults.py --semi_only` produces Rank_ROC.pdf / Rank_PR.pdf (Fig 2 box plots), but the script imports no plotting library and writes only CSVs

_category: Technical bug · topic: reproduction instructions_

**severity: low  ·  confidence: high  ·  detection: 3/10 runs**

- **Claim:** README (Reproduction Roadmap line 98; 'Result Aggregation' lines 199-204) says `python AggregateResults.py --semi_only` produces `Rank_ROC.pdf` and `Rank_PR.pdf` (the Figure 2 box plots), but the script contains no matplotlib/seaborn import or savefig call and only writes CSV files.
- **Concern:** Following the README does not reproduce the Figure 2 PDFs; the box-plot rendering code for the main figure is not in the repo (the underlying rank values, however, are computed and saved to CSV).
- **Ask:** Authors: include the box-plot script that turns the rank CSVs into Rank_ROC.pdf / Rank_PR.pdf, or correct the README to state only CSVs are produced.
- **Evidence:** `AggregateResults.py:269-273` · paper: Figure 2
- **Found in runs:** r03, r09, r10  (representative: r03#3)
- **Quoted at `AggregateResults.py:269-273`:**
```
df_aggregated.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}.csv"))
order_roc = df_aggregated.groupby("Detector")["Rank_ROC"].mean().sort_values()
order_roc.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank.csv"))
order_pr = df_aggregated.groupby("Detector")["Rank_PR"].mean().sort_values()
order_pr.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank_PR.csv"))
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
plotting code is missing
---

### F10 · AblationStudies.py implements only Figs 12-15; no code for Fig 16 (feature normalization), Fig 17 (time-interpolated inputs z_t=tz), or Fig 18 (TCCM vs AE+TimeEmbedding)

_category: Missing code / data · topic: ablations_

**severity: low  ·  confidence: high  (varied: high, medium)  ·  detection: 3/10 runs**

- **Claim:** AblationStudies.py only produces Figs 12, 13, 14 and the contamination figure; the appendix also reports Fig 16 (effect of z-score standardization), Fig 17 (effect of interpolated time-dependent inputs z_t=tz), and Fig 18 (TCCM vs AE+TimeEmbedding), none of which have code (repo grep for 'standardiz'/'interpolat'/'AE+TE'/Figure_16-18 finds nothing in non-baseline files).
- **Concern:** Three appendix ablations that justify TCCM design choices (normalization, no time-interpolation, residual learning vs AE) cannot be reproduced because their experiment code is not in the repo.
- **Ask:** Authors: please add the scripts producing Figures 16, 17, and 18.
- **Evidence:** `AblationStudies.py:594-609` · paper: Appendix D.3; Figures 16, 17, 18
- **Found in runs:** r02, r04, r06  (representative: r04#3)
- **Quoted at `AblationStudies.py:594-609`:**
```
results = run_time_embedding_ablation()
plot_time_embedding_ablation(results)
# ======================================================== #
# ======================================================== #
"""
Ablation Study 3: Training with or without Noise
"""
noise_results = run_noise_ablation(data_configs, seed_list)
plot_noise_ablation(noise_results)
# ======================================================== #
# ======================================================== #
"""
Ablation Study 4: Contamination Ratio Sensitivity (Per-Dataset Max)
"""
results, contamination_levels_dict= run_training_contamination_ablation_dynamic_fixed_split()
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
correct but the grep evidence cited in the claim is not good evidence.
---

### F11 · model_worker enqueues a 2-tuple on out-of-try exceptions (OOM / limit_memory failure), but run_model_with_timeout always unpacks 4 elements → the real error is masked by a ValueError

_category: Technical bug · topic: experiment driver_

**severity: low  ·  confidence: high  (varied: high, medium)  ·  detection: 2/10 runs**

- **Claim:** On any exception raised inside model_worker outside train_and_eval's own try (e.g. limit_memory / OOM kill / import-time failure), the worker enqueues a 2-tuple ('ModERROR', str(e)), but run_model_with_timeout unpacks it as `status, result, train_time, test_time = queue.get()` (FullExperiments.py:134), which raises ValueError: not enough values to unpack (expected 4, got 2). Same pattern in ContaminationStudies.py:59 vs :72.
- **Concern:** Instead of recording a failed run and continuing, the per-(dataset,model) process crashes on the unpack, so a model that fails for these reasons may silently produce no result rather than an explicit error row.
- **Ask:** Make the except branch enqueue a 4-tuple, e.g. queue.put(('ModERROR', str(e), None, None)).
- **Evidence:** `FullExperiments.py:119-121` · paper: n/a
- **Found in runs:** r01, r10  (representative: r01#3)
- **Quoted at `FullExperiments.py:119-121`:**
```
except Exception as e:
    print("ERROR!")
    queue.put(("ModERROR", str(e)))
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
correct and borderline relevant
---

### F12 · determine_FMAD_hyperparameters is an if/elif chain with no final else; an unmatched dataset name would raise UnboundLocalError (all 47 shipped datasets do resolve)

_category: Technical bug · topic: hyperparameter selection_

**severity: low  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** The function is a long if/elif chain with no final `else`; if a dataset name matches no branch, `epoch_size`/`batch_size`/`learning_rate` are never assigned and the `return` raises UnboundLocalError. All 47 shipped datasets do resolve (verified), so it does not affect the paper, but any new/renamed dataset crashes.
- **Concern:** A reviewer running TCCM on a dataset whose name is not in the table gets an opaque UnboundLocalError instead of a sensible default; the substring matching is also collision-prone for future dataset names.
- **Ask:** Add an `else` branch with default epochs/batch/lr (and prefer exact-name matching over substring `in`).
- **Evidence:** `FMAD/functions.py:259-268` · paper: Appendix B.3
- **Found in runs:** r04  (representative: r04#6)
- **Quoted at `FMAD/functions.py:259-268`:**
```
elif "hepatitis" in dataset_name:
    epoch_size = 1
    batch_size = 512
    learning_rate = 0.005

return {
    "epochs": epoch_size,
    "learning_rate": learning_rate,
    "batch_size": batch_size
        }
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
This is rather petty. A very minor code quality issue which should not be a finding.
---

### F13 · FullExperiments.py overrides GOAD's n_epoch to 25 (GOAD's own default is 1), contradicting the paper's 'baselines use default configurations' statement

_category: Paper–code mismatch · topic: baselines / hyperparameters_

**severity: low  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** FullExperiments.py overrides GOAD's number of epochs to 25; GOAD's own default is n_epoch=1 (baselines/goad.py:50,124). The paper states 'For all baseline detectors, we use their default configurations and hyperparameters as provided by their source implementations.'
- **Concern:** The paper's blanket claim of default baseline hyperparameters is contradicted for GOAD; the override raises GOAD's training (more epochs is generally favourable to the baseline), so it does not unfairly disadvantage GOAD, but the description is inaccurate.
- **Ask:** Authors: state that GOAD used n_epoch=25 (not its default 1), and confirm whether any other baseline deviates from defaults.
- **Evidence:** `FullExperiments.py:263-264` · paper: Appendix B.3, 'For all baseline detectors, we use their default configurations…'
- **Found in runs:** r09  (representative: r09#5)
- **Quoted at `FullExperiments.py:263-264`:**
```
elif mname == "GOAD":
    parameters = {"n_epoch": 25}
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[x]`

**Notes:**
checked GOAD's actual default (lironber/GOAD, train_ad_tabular.py). It is n_epoch=25, not 1. The 1 is just the copys placeholder. so 25 is the default and the paper's "we use defaults" holds. Even better would be if the author would also tune this method for each dataset, but it is defensible that they did not because of the unlabeled setup
---

### F14 · ContaminationStudies re-stacks already-StandardScaler'd train+test arrays and applies a fresh 50/50 split — double-scaling / split inconsistency

_category: Methodology / validity · topic: preprocessing / data splitting (contamination study)_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** The contamination study loads data via load_adbench_npz (which StandardScaler-fits on the original train normals and transforms the rest), then re-stacks the already-scaled train+test arrays into X_all and performs a brand-new 50/50 split without re-fitting the scaler on the new training subset.
- **Concern:** The features in the new train/test split were scaled with statistics computed from the original train-normal subset (a different partition), so the contamination-study test points are normalized using statistics partly derived from points that are now in the test set — a mild preprocessing-before-split irregularity (no label leakage).
- **Ask:** Authors: confirm whether the scaler should be re-fitted on each contamination-study training subset; quantify whether re-fitting changes the Figure 15 trends.
- **Evidence:** `ContaminationStudies.py:160-172` · paper: Appendix D.3 contamination ablation (Figure 15)
- **Found in runs:** r03  (representative: r03#6)
- **Quoted at `ContaminationStudies.py:160-172`:**
```
X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed) # By default, we use 0

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

X_normal = X_all[y_all == 0]
X_abnormal = X_all[y_all == 1]

X_train_normal, X_test_normal = train_test_split(X_normal, test_size=0.5, random_state=42, stratify=None)
X_train_abnormal_full, X_test_abnormal = train_test_split(X_abnormal, test_size=0.5, random_state=42, stratify=None)

X_test = np.vstack([X_test_normal, X_test_abnormal])
y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_abnormal))])
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
minor imprecision, it is a single scaling, not double scaling
---

### F15 · Figure 1 (vector-field visualization), Figure 6 (anomaly-mismatch boxplots validating Props 3-5), and the §C.3 representation-collapse tracking have no shipping code

_category: Missing code / data · topic: theory validation / result traceability_

**severity: low  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** Figure 1 (ring/two-moons/blobs vector-field visualization), Figure 6 (boxplots validating the anomaly-mismatch assumption of Props 3-5 over d in {2,5,10,15,20}), and the §C.3 representation-collapse tracking are described as empirical validations, but no script produces them (RobustnessStudy.py generates GMM data only for the PGD attack curves, not these figures).
- **Concern:** The empirical evidence backing the theoretical propositions and the conceptual illustration is not reproducible from the repo.
- **Ask:** Authors: please add the scripts for the synthetic illustration (Fig 1), the mismatch-validation boxplots (Fig 6), and the collapse analysis (§C.3).
- **Evidence:** `paper.pdf` · paper: Figure 1; Figure 6; Appendix C.3
- **Found in runs:** r04  (representative: r04#4)
- **Quoted at `paper.pdf`:**
```
Figure 6 shows boxplots
of the score distributions for normals and anomalies across d ∈{2, 5, 10, 15, 20}. In all cases, anomalous points exhibit consistently higher scores than normal points, with AUROC values exceeding
0.9 regardless of dimension.
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F16 · AggregateResults defaults to semi_only=True, dropping the 7 transductive models — the README's main-result command may not match the paper's figure population

_category: Paper–code mismatch · topic: evaluation consistency (README vs paper figure)_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** AggregateResults defaults to semi_only=True, which drops the 7 transductive models and includes the 7 force_inductive (_semisup) variants. The README's reproduction roadmap lists `python AggregateResults.py --semi_only` for the main Rank_ROC.pdf / Rank_PR.pdf. But the main-text Figure 2 leaderboard lists the transductive detectors (KNN, ABOD, LOF, PCA, COF, KPCA, INNE) and labels them 'Classical (Transductive)', i.e. it corresponds to the NON-semi_only (default-mode) aggregation; the _semisup variants only appear in App. E Figure 23.
- **Concern:** A user following the README's documented `--semi_only` command obtains the appendix inductive-setting ranking (Fig 23), not the headline Figure 2; the mapping from command to figure is ambiguous (the README does note removing the flag, but does not say which figure each mode yields).
- **Ask:** Authors: please state explicitly which command/flag reproduces Figure 2 vs Figure 23 (Figure 2 appears to require running AggregateResults WITHOUT --semi_only after run_main.sh).
- **Evidence:** `AggregateResults.py:102-103` · paper: Figure 2 (transductive detectors shown) vs Appendix E Figure 23 (_semisup variants)
- **Found in runs:** r06  (representative: r06#5)
- **Quoted at `AggregateResults.py:102-103`:**
```
argp.add_argument("--semi_only", action="store_true", help="If true, only semi-supervised models are considered.")
argp.set_defaults(semi_only=True)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
small reproducibility nuisance. The failure is silent and not documented correctly in the readme.
---

