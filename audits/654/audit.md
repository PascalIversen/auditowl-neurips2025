# Audit — Unveiling Transformer Perception by Exploring Input Manifolds (NeurIPS 2025, #654)

## 1. Summary

The repo `alessiomarta__transformers_equivalence_classes` is the authors' official
implementation of two input-space exploration algorithms, SiMEC and SiMExp, built on a
Riemannian-geometry / pullback-metric framework. The core math (Jacobian, pullback metric,
eigendecomposition, exploration step) lives in `simec/simec/logics.py`. Driver scripts in
`experiments/` and `analysis/` run exploration/interpretation on ViT (MNIST, CIFAR10) and
BERT (WinoBias-MLM, MHS-classification) models; `experiments/group_results.py` collates raw
per-iteration outputs into `res/all_experiments_*.parquet/.npz`, and
`notebooks/plots_and_tables.ipynb` turns those into every figure and table in the paper. The
contribution is primarily methodological (an algorithm + theory), with a Section-4 empirical
demonstration rather than a benchmark; there are no train/test splits, no held-out evaluation,
and no learned-performance claim, so most leakage/baseline/split checklist items are N/A.

What I did: read the paper text and the full algorithm implementation; traced every Section-4
quantitative claim to the producing code; dumped and read all code cells of
`plots_and_tables.ipynb`; inspected the four shipped result CSVs in `tables/`. I wrote two
deterministic checks under `_audit_code/`:
`check_volume_pvalues.py` (re-reads `tables/volume_ratio.csv` and tests the paper's
"all p-values < 10⁻³" claim and the "order of 10¹" volume-ratio claim) and
`check_missing_artifacts.py` (file-existence of the `res/` result files the notebook needs, and
a repo-wide grep for any Wasserstein-distance computation). Outputs are in `_audit_code/out/`.

Headline results of the checks: (a) the result files driving all figures/tables are absent
(`res/` does not exist; 0 `.parquet` in repo); (b) **no Wasserstein-distance computation exists
anywhere in the repo**, although the paper reports Wasserstein medians of 0.0 / 0.049 — the
notebook computes an L1 (cityblock) distance instead; (c) the "Welch t-tests, all p < 10⁻³"
claim is contradicted by the shipped `volume_ratio.csv` (hatespeech, η=10 has p = 0.007), and
the test actually implemented is a one-sample t-test, not a Welch test.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2a orig/top-class probability curves (SiMEC/SiMExp, 4 datasets) | `notebooks/plots_and_tables.ipynb` cell 15 (`plot_embedding_init_pred_proba`/`plot_embedding_top_pred_proba`), driven by `res/all_experiments_*` | not runnable (inputs absent) | — | Code present, inputs MISSING |
| Fig. 2b baseline (gradient + Gaussian noise) probability curves | `baselines/vit_perturbation.py`, `baselines/bert_perturbation.py` (compute), notebook plots | not runnable (inputs absent) | — | Code present, inputs MISSING |
| Table 1 average class-ranking changes per 1000 iters | `notebooks/plots_and_tables.ipynb` cell 4 (`emb_ranking_changed`, pairwise inversions / 2) | not runnable (inputs absent) | — | Code present, inputs MISSING |
| Hypervolume ratio ρ_V "bigger by an order of 10¹" | `notebooks/plots_and_tables.ipynb` cells 34/39 → `tables/volume_ratio.csv` | ratios 2.44–158.8 (incl. order-10²) | partial | Verified (computation present; "order 10¹" is loose) |
| "Welch t-tests on ρ_V; all p-values < 10⁻³" | cell 39 (`ttest_1samp(... popmean=1, alternative="greater")`) → `tables/volume_ratio.csv` | one-sample t-test; one p = 0.007 | ✗ | MISMATCH (see welch-test-mislabeled, volume-pvalue-mismatch) |
| Per-patch/token mean diff: SiMEC 2.219·10⁻³, SiMExp 83.028·10⁻³ | cell 23 → `tables/pixel_diff.csv` (per-config means) | per-config values present | not checkable | Computation present; aggregate not directly emitted |
| Wasserstein distance (p=1): SiMEC median 0.0, SiMExp median 0.049 | (none — repo computes L1/cityblock "decoding_difference" in cells 18/21 → `tables/decoding_difference.csv`) | — | — | MISSING (metric not computed; see wasserstein-not-computed) |
| "Catch-up": 10.9% realign, after 290.65 iters | `notebooks/plots_and_tables.ipynb` cell 27 → `tables/catchup.csv` (`mean_catch`, `mean_iter_catch`) | per-config values present | not checkable | Computation present; paper aggregate not directly emitted |
| Pearson corr 0.32 (avg p 0.08) emb vs interpretation top-proba | cell 27 (`pearsonr`) → `tables/catchup.csv` (`mean_corr_stat`, `mean_corr_pval`) | per-config values present | not checkable | Computation present |
| Per-patch/token time: CIFAR 0.126s, MNIST 0.050s, WinoBias 0.300s, MHS 0.310s | cell 8 (`results.groupby(...)["time"].agg(mean,std)`); timing recorded in `simec/simec/logics.py:418-420` | not runnable (inputs absent) | — | Code present, inputs MISSING |
| Algorithm 1/2 step δ = η·√(min|λ|/max|λ|) | `simec/simec/logics.py:370-374` | computes η/√(max/min) = η·√(min/max) | ✓ | Verified |

## 3. Findings

## missing

```yaml finding
id: wasserstein-not-computed
category: missing
topic: "result traceability / evaluation metric"
title: "Reported Wasserstein distances have no producing code; repo computes L1 instead"
severity: medium
confidence: high
status: finding
file: _audit_code/out/missing_artifacts.txt
line_start: 10
line_end: 11
quote: |
  Grep for 'wasserstein' anywhere in repo (case-insensitive):
    hits (excluding .git): 0
claim: "The paper reports an average Wasserstein distance (p=1) between embedding and interpretation prediction distributions (SiMEC median 0.0, SiMExp median 0.049), but no Wasserstein computation exists in the repo; the notebook only computes an L1/cityblock distance (`safe_cityblock`, cells 18/21) saved to tables/decoding_difference.csv."
concern: "The specific metric and numbers reported for objective (ii) (embedding-vs-interpretation consistency) are not reproducible from the released code, and the released code computes a different distance than the one the paper names."
resolution: "Authors: provide the script that computes the Wasserstein (p=1) distances quoted in Section 4, or confirm whether the L1 'decoding_difference' in the notebook is what was actually reported under the name Wasserstein."
cross_refs: ["res-results-missing"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 4, 'Using interpretation outputs as alternative input data' (median Wasserstein 0.0 / 0.049)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: res-results-missing
category: missing
topic: "result traceability / intermediate outputs"
title: "res/ result files that drive all figures and tables are not shipped"
severity: low
confidence: high
status: finding
file: _audit_code/out/missing_artifacts.txt
line_start: 1
line_end: 4
quote: |
  Result files the notebook (notebooks/plots_and_tables.ipynb) reads:
    MISSING: res/all_experiments_results.parquet
    MISSING: res/all_experiments_embeddings.npz
    MISSING: res/all_experiments_embeddings_input.npz
claim: "notebooks/plots_and_tables.ipynb reads res/all_experiments_results.parquet and two .npz files (load_data/load_plot_data) to produce every figure and table, but none of these files (nor the res/ directory) are present in the repo; no .parquet ships at all."
concern: "All reported figures/tables can only be regenerated by re-running the full train -> preprocess -> explore -> group_results pipeline (which also needs GPU and HF model downloads), so none of the headline numbers are directly verifiable from shipped artefacts."
resolution: "Authors: ship the all_experiments_* result bundle (or a small sample) so the notebook's figures/tables can be regenerated without the full re-run, or document the exact commands and runtime to regenerate res/."
cross_refs: ["wasserstein-not-computed"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 4 (Figure 2, Table 1, all reported statistics)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: vit-weights-not-shipped
category: missing
topic: "expected code completeness / pretrained models"
title: "Trained ViT checkpoints absent; only config.json + metrics.json shipped"
severity: low
confidence: high
status: finding
file: models/cifar10_experiment/metrics.json
line_start: 1
line_end: 1
quote: |
  {
claim: "models/cifar10_experiment/ and models/mnist_experiment/ contain only config.json and metrics.json (training accuracy curves); no model weight file (.pt/.pth/.safetensors) is present anywhere in the repo, so the exact ViTs used must be retrained via experiments/models/train_vit.py."
concern: "Exact reproduction of the reported numbers requires retraining the ViTs (nondeterministic), and the BERT models are pulled from external HF hubs; no checkpoint is pinned."
resolution: "Authors: release the trained ViT checkpoints (or pin exact seeds/commands), and pin the exact HF model revisions used for BERT."
cross_refs: []
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 4 (one ViT trained per image dataset; pretrained BERT for text)"
tags: [reforms:4, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No bugs found that the code's own intent contradicts. The stale comment
`#attention masks missing!!` in `simec/simec/logics.py:34` is misleading, but the attention
mask is in fact threaded through (`bert_exploration.py:225-247` builds `extended_attention_mask`
and passes it into `explore` -> `pullback` -> `jacobian` -> `OutputOnlyModel.forward`), so this
is not a defect.

## difference

```yaml finding
id: volume-pvalue-mismatch
category: difference
topic: "statistical integrity"
title: "Paper claims all volume-ratio p-values < 1e-3, but shipped table has p = 0.007"
severity: medium
confidence: high
status: finding
file: tables/volume_ratio.csv
line_start: 5
line_end: 5
quote: |
  hatespeech,10.0,2.44,0.007
claim: "The repo's own tables/volume_ratio.csv (produced by notebook cell 39) reports the hatespeech, delta=10 volume-ratio test with p = 0.007, whereas the paper states 'all p-values resulted lower than 10^-3'."
concern: "One of the eight reported tests does not meet the significance threshold the paper claims for all of them, so the blanket significance statement is not supported by the released results."
resolution: "Authors: correct the claim to reflect the actual p-values (one is 7e-3, not < 1e-3), or clarify whether a different test/aggregation produced the reported < 1e-3."
cross_refs: ["welch-test-mislabeled"]
check_script: _audit_code/check_volume_pvalues.py
paper_ref: "Section 4: 'We validated these results by performing Welch t-tests on rho_V: all p-values resulted lower than 10^-3.'"
tags: [stats:statcheck, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: welch-test-mislabeled
category: difference
topic: "statistical integrity"
title: "Paper says 'Welch t-tests'; code runs a one-sample t-test (ttest_1samp)"
severity: low
confidence: high
status: finding
file: notebooks/plots_and_tables.ipynb
line_start: 42375
line_end: 42377
quote: |
      "    volume_ratio_test=(\"svd_volume_ratio\", lambda x: ttest_1samp(x, popmean = 1, alternative = \"greater\").statistic),\n",
      "    volume_ratio_test_pvalue=(\"svd_volume_ratio\", lambda x: ttest_1samp(x, popmean = 1, alternative = \"greater\").pvalue)\n",
claim: "The volume-ratio significance test is implemented as a one-sample t-test of the per-input SiMExp/SiMEC volume ratios against popmean=1 (scipy ttest_1samp), not a Welch (two-sample, unequal-variance) t-test as the paper states."
concern: "The named test (Welch) and the implemented test (one-sample against a fixed mean of 1) are different procedures; the one-sample test is itself a valid way to test ratio>1, so the issue is the paper's description, not the validity of the code."
resolution: "Authors: rename the test in the paper to a one-sample t-test (H0: mean ratio <= 1), or, if a Welch test was intended, provide that implementation and its p-values."
cross_refs: ["volume-pvalue-mismatch"]
paper_ref: "Section 4: 'performing Welch t-tests on rho_V'"
tags: [stats:statcheck, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodological-validity findings. This is a method/theory paper: there is no
train/test split, no held-out performance claim, and no baseline-beating accuracy claim, so the
usual leakage / split / metric-fit / baseline-tuning checks are structurally N/A. The Section-4
experiments are descriptive demonstrations of algorithm behaviour, and the one-sample t-test on
volume ratios (see welch-test-mislabeled) is itself a valid test for the ratio>1 hypothesis.
Scope-filter notes:
- Data splitting / sample independence / target leakage: N/A (no predictive task with held-out
  evaluation; the models are used only to compute Jacobians).
- Pretraining contamination: N/A (pretrained BERT is used as the object of study, not to
  produce a benchmarked metric on a test set it might have seen).
- Temporal integrity: N/A (no time dimension).
- Baselines: a baseline IS present and compared (gradient + Gaussian-noise perturbation,
  Fig. 2b, `baselines/`), which is appropriate for the demonstrative claims.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 3          | medium       | Wasserstein metric never computed; res/ result bundle & ViT weights absent |
| bug         | 0          | -            | None (stale "attention masks missing" comment is not a real defect)    |
| difference  | 2          | medium       | "all p<1e-3" contradicted by shipped table (p=0.007); "Welch" test is actually one-sample |
| methodology | 0          | -            | Method/theory paper; no split/leakage/metric-fit issues in scope        |

## Top take-aways (<= 6, by severity x confidence)

1. **[missing] wasserstein-not-computed** (med/high): the paper's reported Wasserstein (p=1)
   distances (median 0.0 / 0.049) have no producing code anywhere in the repo; the notebook
   computes an L1/cityblock distance instead.
2. **[difference] volume-pvalue-mismatch** (med/high): the repo's own `volume_ratio.csv` shows
   p = 0.007 for one of the eight volume-ratio tests, contradicting the paper's "all p-values
   resulted lower than 10⁻³".
3. **[difference] welch-test-mislabeled** (low/high): the test described as a "Welch t-test" is
   implemented as a one-sample t-test against popmean=1 (`ttest_1samp`).
4. **[missing] res-results-missing** (low/high): the `res/all_experiments_*` parquet/npz bundle
   that drives every figure and table is not shipped, so headline numbers require a full GPU
   re-run to regenerate.
5. **[missing] vit-weights-not-shipped** (low/high): no trained ViT checkpoint is included
   (only config + metrics); models must be retrained.

## Items that genuinely look fine

- The exploration step length matches Algorithms 1/2: `simec/simec/logics.py:370-374` computes
  η·√(min|λ|/max|λ|), consistent with line 7 of both pseudocodes.
- SiMEC vs SiMExp eigenvector selection (zero vs non-zero eigenvalues) is correctly branched on
  `same_equivalence_class` in `logics.py:342-348`.
- A baseline (gradient + orthogonal Gaussian noise) is implemented and compared (Fig. 2b,
  `baselines/vit_perturbation.py`, `baselines/bert_perturbation.py`).
- BERT attention masks ARE threaded through to the Jacobian (`bert_exploration.py:225-247` ->
  `explore`/`pullback`/`jacobian`); the `#attention masks missing!!` comment is stale, not a bug.
- Dependencies are fully pinned (`requirements.txt`, exact versions) and the package is
  installable (`pyproject.toml`, `simec/setup.py`).
- Class-ranking-change counting divides pairwise inversions by 2 to avoid double counting
  (notebook cell 4), matching the Table 1 definition.

## Open questions for the authors

- Was the metric reported as "Wasserstein distance (p=1)" actually the L1/cityblock
  `decoding_difference` in the notebook, or is there an off-repo script that computes a true
  Wasserstein distance? (drives wasserstein-not-computed severity)
- Do the paper aggregates (per-patch diff 2.219e-3 / 83.028e-3; catch-up 10.9% / 290.65 iters;
  Pearson 0.32, p 0.08) come from a specific aggregation over the per-config rows in
  `tables/pixel_diff.csv` / `tables/catchup.csv`, and over which configurations?
