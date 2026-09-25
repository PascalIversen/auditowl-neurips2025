# Code-repository audit — DiCoFlex (paper 2167)

## 1. Summary

DiCoFlex (NeurIPS 2025) proposes a model-agnostic conditional normalizing-flow
framework that generates diverse counterfactual explanations for tabular data in
a single forward pass, with inference-time control of sparsity (Lp-norm `p`) and
actionability (feature masks). The repository
(`code/ofurman__DiCoFlex/`, single squashed commit `7999981`, GitHub
`ofurman/DiCoFlex`) contains: the DiCoFlex method (conditional MAF over KNN-sampled
neighbors), five tabular dataset loaders (Adult, Bank, Default, GMC, Lending Club),
a discriminative MLP, generative flow models (MAF/NICE/RealNVP/KDE/CNF), a metrics
module, and a single driver `counterfactuals/dicoflex/train_generic_counterfactual.py`.

What I did. I read the paper (Tables 1–8, Figs 1–3, Appendices A–I) and the full
repo. I mapped every reported quantitative artefact to code, ran two read-only
deterministic checks under `_audit_code/`:
- `check_artifact_presence.py` — greps all 45 `.py` files for hypervolume, each
  baseline (DiCE/CCHVAE/ReViSE/TABCF/Wachter), German-credit, sensitivity-sweep,
  and model-selection drivers (output `_audit_code/out/artifact_presence.csv`).
- `check_metric_keys.py` — AST-extracts the keys returned by
  `CFMetrics.calc_all_metrics` (output `_audit_code/out/metric_keys.csv`).

Headline result: the repo computes validity, sparsity, proximity, and plausibility
(LOF / log-density / isolation-forest) for **DiCoFlex only**. It contains **no code
that computes Hypervolume** (the paper's diversity metric, present in every results
table), **no baseline-method code**, and **no drivers** for Tables 3, 5, 8 or the
runtime Figures 2/3. The bulk of the paper's numbers therefore have no producing
script in the repo. The flow training itself is sound and matches the described
algorithm; the gaps are about completeness/traceability and a metric definition
mismatch, not about a broken or invalid core method.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — Validity (all datasets/methods) | `counterfactuals/metrics/metrics.py:147` (`validity`) for DiCoFlex only | not run here | — | PARTIAL (DiCoFlex only; no baseline code) |
| Table 1 — Classif. prob. | not found | — | — | MISSING (no `prob`/confidence metric in `calc_all_metrics`) |
| Table 1 — Proximity cont. | `metrics.py:285` (`feature_distance`, cityblock) | not run | — | Present (DiCoFlex only) |
| Table 1 — Sparsity cat. / ϵ-sparsity cont. | `metrics.py:175` (`sparsity`) | single all-feature value | ✗ (no cat/num split, no relative ϵ) | MISMATCH |
| Table 1 — LOF (log scale) | `metrics.py:227` (`lof_scores`) | not run | — | Present (DiCoFlex only) |
| Table 1 — **Hypervolume** (diversity) | (none) | — | — | MISSING (no hypervolume code in repo) |
| Table 1 — baselines DiCE/CCHVAE/ReViSE/TABCF/Wachter (all columns) | (none) | — | — | MISSING (no baseline code) |
| Table 2 / Table 6 — actionability masks (Adult) | partial: masks built in driver, but mask-specific driver/Table not in repo | — | — | MISSING (no mask-experiment driver/Table) |
| Table 3 — sensitivity to p (0.01–2.0) | (none) | — | — | MISSING (no p-sweep driver) |
| Table 4 — dataset sizes (Adult 32,000 …) | `data/*/train.csv`; Adult code subsamples 2000 | Adult train file 32,561 rows | ✗ (code uses `.sample(2000)`) | MISMATCH |
| Table 5 — model selection NLL (MAF/NICE/RealNVP/KDE) | model classes exist; no NLL-comparison driver | — | — | MISSING (no comparison script) |
| Table 7 — std devs across CFs | (none) | — | — | MISSING |
| Table 8 — German Credit results | (none; `german_credit.csv` present but unused) | — | — | MISSING (no German experiment) |
| Fig. 2 / Fig. 3 — generation runtime | (none; no generation-timing harness) | — | — | MISSING |

Deterministic backing: `_audit_code/out/artifact_presence.csv` (0 hits for
hypervolume, every baseline, german, sensitivity sweep) and
`_audit_code/out/metric_keys.csv` (13 keys; no hypervolume/diversity/`prob`).

## 3. Findings

## missing

```yaml finding
id: hypervolume-not-computed
category: missing
topic: "result traceability / diversity metric"
title: "No code computes Hypervolume, the diversity metric in every results table"
severity: high
confidence: high
status: finding
file: counterfactuals/metrics/metrics.py
line_start: 352
line_end: 396
quote: |
        metrics = {
            "coverage": self.coverage(),
            "validity": self.validity(),
            "actionability": self.actionability(),
            "sparsity": self.sparsity(eps=self.sparsity_eps),
            # "target_distance": self.target_distance(),
            # "proximity_categorical_hamming": self.feature_distance(
            #     categorical_metric="hamming"
            # ),
            # "proximity_categorical_jaccard": self.feature_distance(
            #     categorical_metric="jaccard"
            # ),
            "proximity_continuous_manhattan": self.feature_distance(
                continuous_metric="cityblock"
            ),
            "proximity_continuous_euclidean": self.feature_distance(
                continuous_metric="euclidean"
            ),
claim: "CFMetrics.calc_all_metrics() is the only metric aggregator in the repo; an AST scan of its returned dict (check_metric_keys.py) shows 13 keys and none is hypervolume/diversity; a repo-wide grep (check_artifact_presence.py) finds 0 occurrences of hypervol/pymoo/pareto/nondomin."
concern: "Hypervolume (diversity) is reported for every method in Tables 1, 3, 6, 8 and is a headline claim ('outperforms ... in terms of ... diversity'), yet no script in the repo computes it, so the diversity numbers cannot be reproduced or checked."
resolution: "Provide the hypervolume computation (e.g. the pymoo/objective-space code) used to produce the 'Hypervol.' column in Tables 1/3/6/8."
cross_refs: ["baselines-and-tables-missing"]
check_script: _audit_code/check_metric_keys.py
paper_ref: "Table 1 (Hypervol. column); Section 4.2; Abstract"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baselines-and-tables-missing
category: missing
topic: "result traceability / baselines and experiment drivers"
title: "No baseline code and no drivers for Tables 3/5/7/8 or runtime Figs 2/3"
severity: high
confidence: high
status: finding
file: counterfactuals/dicoflex/train_generic_counterfactual.py
line_start: 363
line_end: 374
quote: |
    import argparse
    parser = argparse.ArgumentParser(description='Train multiclass counterfactual generative models')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    parser.add_argument('--adult', action='store_true', help='Run Adult multiclass example')
    parser.add_argument('--bank', action='store_true', help='Run Bank multiclass example')
    parser.add_argument('--gmc', action='store_true', help='Run GMC multiclass example')
    parser.add_argument('--lending', action='store_true', help='Run Lending Club multiclass example')
    parser.add_argument('--default', action='store_true', help='Run Default multiclass example')
    parser.add_argument('--save_dir', default='results', help='Save directory path')
    args = parser.parse_args()
claim: "The only experiment entrypoint trains/evaluates DiCoFlex on the five datasets; a repo-wide grep (check_artifact_presence.py) returns 0 hits for DiCE, CCHVAE, ReViSE, TABCF, Wachter, 'german', and any sensitivity-p sweep or NLL-comparison driver."
concern: "Roughly half of Table 1 (five baseline methods), Table 3 (p-sensitivity), Table 5 (model-selection NLL), Table 7 (std devs), Table 8 (German Credit), and Figs 2/3 (runtime) have no producing code, so these reported numbers and the comparative claims cannot be reproduced from the repo."
resolution: "Provide the baseline-method scripts (or the CARLA/library configs used) and the drivers that produced Tables 3, 5, 7, 8 and the runtime figures, run under the same split/metrics."
cross_refs: ["hypervolume-not-computed"]
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Table 1 (baselines), Tables 3/5/7/8, Figures 2/3"
tags: [reforms:2, reforms:4, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: deps-unpinned
category: missing
topic: "expected code completeness / dependencies"
title: "Dependencies unpinned; environment not reconstructible"
severity: medium
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 12
quote: |
  torchdiffeq
  alibi[tensorflow]
  numpy
  scipy
  cvxpy
  scikit-learn
  neptune
  hydra-core
  mlflow
  umap-learn
  nflows @ git+https://github.com/pfilo8/nflows.git
  UMNN
claim: "requirements.txt and requirements.in list package names with no version pins; the core flow library is a git fork (`nflows @ git+https://github.com/pfilo8/nflows.git`) pinned only to a moving branch (no commit)."
concern: "Without versions (and with a moving-fork dependency), the exact environment that produced the numbers cannot be rebuilt, and `OneHotEncoder(sparse=...)` in the dataset code is version-sensitive (renamed to sparse_output in recent sklearn)."
resolution: "Pin all versions (and the nflows fork to a commit hash), and provide the conf/ directory referenced by README."
cross_refs: []
paper_ref: "Appendix H (Computational Resources)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-conf-and-instructions-broken
category: missing
topic: "expected code completeness / reproduction instructions"
title: "README points to a non-existent conf/ dir and gives no per-table commands"
severity: low
confidence: high
status: finding
file: README.md
line_start: 46
line_end: 57
quote: |
  ```
  |── conf/                  # Configuration files
  |── data/                  # Datasets
  |── counterfactuals/       # Source code for the framework
  |   ├── datasets/          # Counterfactual methods
  |   ├── discriminative_models/  # Discriminative models for analysis
  |   ├── generative_models/      # Generative models for analysis
  |   ├── dicoflex/            # DiCoFlex method code
  |   └── metrics/           # Evaluation metrics
  |── README.md              # This document
  └── ...
  ```
claim: "README documents a `conf/` configuration directory and a single `--all` command, but no `conf/` exists in the repo and there is no results table or per-experiment command mapping outputs to the paper's tables."
concern: "A reviewer cannot map the run to any specific table/figure, and the documented config layer is absent."
resolution: "Add the conf/ directory (or remove the reference) and a README results table with exact commands per paper table/figure."
cross_refs: ["baselines-and-tables-missing"]
paper_ref: "README; NeurIPS checklist Q5 (open access to code)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-abs-syspath
category: bug
topic: "portability / hardcoded paths"
title: "Driver hardcodes two absolute machine-specific sys.path entries"
severity: low
confidence: high
status: finding
file: counterfactuals/dicoflex/train_generic_counterfactual.py
line_start: 5
line_end: 6
quote: |
  sys.path.append(r"C:\Users\marsz\Studia\GMUM\counters_base\counterfactuals")
  sys.path.append("/home/z1172691/counterfactuals")
claim: "The only experiment entrypoint appends two absolute author-machine paths to sys.path at import time."
concern: "These dead absolute paths are no-ops on any other machine (and signal the run depended on a local layout); harmless to execution but indicates the script was not cleaned for release."
resolution: "Remove the absolute sys.path hacks and rely on the installed package (setup.py is present)."
cross_refs: []
paper_ref: "n/a"
tags: [heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: sparsity-metric-mismatch
category: difference
topic: "evaluation consistency / sparsity metric"
title: "Code sparsity is one all-feature count, not the paper's cat/num split or relative ϵ-sparsity"
severity: medium
confidence: high
status: finding
file: counterfactuals/metrics/metrics.py
line_start: 175
line_end: 193
quote: |
    def sparsity(self, eps: float = 0.0) -> float:
        """
        Compute the sparsity metric.

        Args:
            eps (float, optional): Tolerance for considering features different. 
                For continuous features, differences smaller than eps will be considered equal.
                Defaults to 0.0 (exact comparison).

        Returns:
            float: Sparsity metric value.
        """
        if eps > 0.0:
            # For features with difference less than epsilon, consider them equal
            diff = np.abs(self.X_test - self.X_cf)
            return np.mean(diff > eps)
        else:
            # Traditional exact comparison
            return (self.X_test != self.X_cf).mean()
claim: "The implemented sparsity returns a single scalar over ALL features as mean(|x-x'| > eps); it does not separate categorical vs numerical features and uses an absolute threshold on the difference, not the relative change |x_d - x'_d|/|x_d| > 0.05 over continuous features only."
concern: "The paper (Eq. 29, 30; Table 1 columns 'Sparsity cat.' and 'ϵ-sparsity cont.') reports two separate, differently-defined sparsity quantities, so the repo's single all-feature metric cannot reproduce either reported column."
resolution: "Provide the cat/num-separated sparsity (Eq. 29) and the relative ϵ-sparsity (Eq. 30) functions actually used for Table 1, or clarify how the reported columns were computed."
cross_refs: ["hypervolume-not-computed"]
paper_ref: "Appendix C, Eqs. 29–30; Table 1"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: adult-subsample-2000-unseeded
category: difference
topic: "data / dataset size and determinism"
title: "Adult loader subsamples 2000 rows (unseeded); paper Table 4 says 32,000"
severity: medium
confidence: high
status: finding
file: counterfactuals/datasets/DCENF/adult.py
line_start: 80
line_end: 81
quote: |
        raw_data[target_column] = (raw_data[target_column] == " >50K").astype(int)
        raw_data = raw_data.sample(2000)
claim: "AdultDataset.preprocess subsamples 2000 rows via pandas `.sample(2000)` with no random_state, while the other loaders use full files; global seeding only sets np.random.seed(0) in the driver (pandas sample uses numpy's global RNG, but the result still differs from a 32,000-row run)."
concern: "Paper Table 4 lists 32,000 Adult samples; the code evaluates on a 2,000-row subsample, so the reported Adult numbers are not reproduced by this loader and the subsample is not pinned independently of execution order."
resolution: "Confirm the actual Adult sample size used for Table 1/2/3 and pin the subsample (random_state); reconcile with Table 4's 32,000."
cross_refs: []
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Table 4 (Adult Census = 32,000 samples)"
tags: [reforms:2, reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: training-epochs-mismatch
category: difference
topic: "training hyperparameters"
title: "Driver hardcodes 10 epochs / patience 50; paper states 1000 epochs / patience 300"
severity: low
confidence: high
status: finding
file: counterfactuals/dicoflex/train_generic_counterfactual.py
line_start: 191
line_end: 192
quote: |
        num_epochs=10,
        patience=50,
claim: "train_method trains the flow with num_epochs=10, patience=50 (and the disc/gen models at epochs=10 in prepare_dataset_and_models); Appendix E.1 states 'a maximum of 1000 epochs ... early stopping with a patience of 300 epochs.'"
concern: "The released configuration does not match the paper's stated training budget, so a default `--all` run would not reproduce the reported models; this is a setting in the script (not a methodological error)."
resolution: "Set the released defaults to the paper's 1000 epochs / patience 300, or document why the committed values differ."
cross_refs: []
paper_ref: "Appendix E.1 (Training Details)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A for a hard finding. The core procedure that *is* implemented is methodologically
sound for the stated task: the flow is trained by NLL on KNN-sampled, classifier-validated
neighbors (Eq. 4/6); the train/test split is honored at the dataset level (separate
train.csv/test.csv files, with the flow's own 80/20 split applied only over training-derived
batches in `get_counterfactual_dataloaders`); metrics are computed on `X_test`-derived
factuals against generated CFs; LOF and the plausibility threshold are fit on training data
only (`metrics.py:247`, `dicoflex/utils.py:74-78`). I found no leakage in the implemented
path. The headline reproducibility problems are absence of code (Section `missing`), not an
invalid implemented procedure. See the open question below on the diversity/validity
comparison, which I cannot resolve because the baseline and hypervolume code are absent.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Hypervolume + all baselines + Tables 3/5/7/8 + Figs 2/3 have no producing code; deps unpinned. |
| bug         | 1          | low          | Dead absolute sys.path entries in the only driver. |
| difference  | 3          | medium       | Sparsity metric definition, Adult 2000-row subsample, training-epoch defaults all differ from paper. |
| methodology | 0          | -            | Implemented flow-training/eval path is sound; no leakage found in the code that exists. |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing]** `hypervolume-not-computed` — the diversity metric reported in every
   results table has no implementation anywhere in the repo (high/high).
2. **[missing]** `baselines-and-tables-missing` — no code for any baseline
   (DiCE/CCHVAE/ReViSE/TABCF/Wachter) and no drivers for Tables 3/5/7/8 or runtime
   Figs 2/3; ~half of Table 1 is unreproducible (high/high).
3. **[difference]** `sparsity-metric-mismatch` — the single all-feature sparsity in
   `metrics.py` cannot reproduce the paper's separate categorical and relative
   ϵ-sparsity columns (medium/high).
4. **[difference]** `adult-subsample-2000-unseeded` — Adult loader uses 2,000 rows
   while Table 4 states 32,000 (medium/high).
5. **[missing]** `deps-unpinned` — unpinned requirements + moving-fork nflows make the
   environment non-reconstructible (medium/high).
6. **[difference]** `training-epochs-mismatch` — committed 10-epoch defaults vs paper's
   1000 (low/high).

### Items that genuinely look fine
- Train/test separation: datasets load distinct `train.csv`/`test.csv`; LOF and the
  plausibility threshold are fit on training data only (`metrics.py:247`,
  `dicoflex/utils.py:74-78`). No train→test leakage in the implemented path.
- The implemented DiCoFlex training matches Algorithm 1 / Eqs. 4 & 6: KNN neighbor
  sampling with classifier-probability filtering, conditional MAF trained by NLL,
  with `p` and mask passed as conditioning (`generic_counterfactual.py:645`).
- Validity, proximity (cityblock/euclidean), LOF, log-density, and isolation-forest
  metrics are implemented and consistent with their paper descriptions.
- Data files for all five datasets (plus German Credit) are shipped in `data/`.

### Open questions for the authors
- How were the Hypervolume diversity values and all baseline columns in Table 1
  produced (which library/scripts), given none are in the repo? (drives
  `hypervolume-not-computed`, `baselines-and-tables-missing`)
- Was Table 1 computed on the full Adult set or the 2,000-row subsample, and with
  which sparsity definition (Eq. 29/30 vs the all-feature `metrics.py:sparsity`)?
  (drives `adult-subsample-2000-unseeded`, `sparsity-metric-mismatch`)
