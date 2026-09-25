# Audit: "When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective" (paper 3101)

## Summary

This is primarily a **theory paper** (identifiability of a softmax embedding/unembedding model
family) with **empirical illustrations** rather than benchmark/performance claims. Its headline
results are theorems (3.1, 4.7) and a corollary (3.2); the experiments are (i) a constructed
KL→0 table (Table 1), (ii) CIFAR-10 retrained ResNet18 models showing dissimilar embeddings at
similar loss (Fig. 3 left / App. F.5 / Fig. 5), (iii) a synthetic width experiment (Fig. 3 right /
App. F.7), and (iv) bound-illustration plots (App. F.6 / Fig. 16).

The repo (`bemigini__close-dist-rep-sim`, commit `ddba9c8`) is well organised: `run.py` is a docopt
CLI with `train-variations`, `get-distances`, `get-accuracies`, `make-plot`. Core measures
(`d_prob`/`dλ_LLV`, KL, mCCA, mSVD) are in `src/dissimilarity_measures/`. Precomputed result JSONs
ship in `results.zip`; synthetic SmallMLP checkpoints (a subset) ship in `checkpoints/`.

What I ran (all read-only, under `_audit_code/`):
- `run_kl_table.py` — reproduced **Table 1** end-to-end (`kl_to_zero_dissimilar_reps`). Values match
  the paper to ~1e-3 (e.g. ρ=3: dKL 0.8866, dλ_LLV 1.317; ρ=18: dKL ~1e-4, dλ_LLV 1.317).
- `check_bound.py` — reproduced the **Theorem 4.7 bound** illustration on 250 constructed perturbed
  models: `d_rep ≤ 2·M·d_prob` held with **0 violations** (max ratio 0.006), matching the paper's
  "the bound … is always satisfied" (§5.3, App. F.6).
- Extracted `results.zip` and verified **Fig. 3-left numbers** against the shipped CIFAR result files:
  pair `0vs1` has `test_f_mcca = 0.551` (paper 0.55), test losses `1.193`/`1.178` (paper 1.19/1.18),
  ResNet `KL_div = 1.131` and `distance (dλ_LLV) = 1.546` (paper 1.13 and ~1.55).
- Verified the **synthetic model-retention counts** (§5.2): 4 classes → [19,20,20,20,20]; 6 classes →
  [16,17,19,19,20]. Both match the paper exactly.

Overall the empirical claims are highly traceable and reproduce. The findings below are minor
reproducibility/wiring gaps, the most notable being that **CIFAR-10 checkpoints are not shipped** and
the script that computes the CIFAR mCCA values has **no `run.py` entry point** (the numbers it would
produce are, however, independently shipped as result files and reproduce the paper).

This paper has **no train/test-leakage, baseline, or statistical-test surface** of the usual
benchmark kind: the experiments are illustrative constructions and seeded synthetic/CIFAR retrainings,
not predictive-performance comparisons. Those topic checklists are marked N/A below.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (dKL, dλ_LLV, mCCA, max dSVD vs ρ) | `experiments/kl_to_zero_dissimilar_reps.py` (`make-plot --plot-type=KL_table`) | reproduced (e.g. ρ=3: 0.887/1.317/~0/0.999; ρ=18: ~1e-4/1.317) | ✓ | Verified (`_audit_code/run_kl_table.py`) |
| Fig. 3-left mCCA = 0.55 (CIFAR pair 0vs1) | `results/mcca_cifar_fd2.json` (computed by `experiments/measure_cca_for_cifar_models.py`) | 0.551 | ✓ | Verified value; see `cifar-checkpoints-missing`, `cifar-mcca-no-entrypoint` |
| Fig. 3-left test losses 1.19 / 1.18 | `results/loss_cifar_fd2.json` (via `plots/loss_diff_vs_mcca.py:save_final_cifar_loss_to_single_file`) | 1.193 / 1.178 | ✓ | Verified value |
| Fig. 3-left dKL 1.13 (0→1), dλ_LLV ~1.55 | `results/...ResNetCIFAR10_128_fd2_20000.json` (via `experiments/prob_distances.py`) | KL 1.131, distance 1.546 | ✓ | Verified value |
| Fig. 3-right (mean dλ_LLV, df,g vs width) | `plots/mean_d_prob_vs_width.py` ← `results/*SmallMLP*` distances/acc | qualitative (decreasing) | ✓ (trend) | Verified inputs present & retention counts match |
| §5.2 retention counts (4 & 6 classes) | `results/*_model_acc_test_SmallMLP_*` + filter acc>0.9 | [19,20,20,20,20], [16,17,19,19,20] | ✓ | Verified (`_audit_code/`) |
| Fig. 5 (App. F.4) loss-diff vs mCCA scatter | `plots/loss_diff_vs_mcca.py` ← `results/{loss,mcca}_cifar_fd{2,3,5}.json` | inputs present | — | Plotting traces; mCCA compute not wired (`cifar-mcca-no-entrypoint`) |
| Fig. 16 (App. F.6) bound on constructed models | `experiments/making_constructed_models.py` (`make-plot --plot-type=d_LLV_constructed`) | 0/250 bound violations | ✓ | Verified (`_audit_code/check_bound.py`) |
| Fig. 16 (App. F.6) bound on trained synthetic | `plots/trained_models_d_prob_plots.py` ← `results/*SmallMLP*` | inputs present | — | Plotting traces |
| App. F.5 all 2D CIFAR embeddings; Fig. 3-left panel | `plots/cifar10_embeddings_can_be_permuted.py` | requires CIFAR checkpoints (absent) | — | Plot blocked; see `cifar-checkpoints-missing` |
| App. F.2 synthetic-data illustration (Fig. 4) | `plots/training_data_illustration.py` + `src/data/radial_classification.py` | self-contained | ✓ | Traces |
| Theorems 2.2/3.1/4.7, Cor. 3.2 (math) | paper (proofs in App. C, E) | N/A (analytic) | — | Out of code-audit scope |

## missing

```yaml finding
id: cifar-checkpoints-missing
category: missing
topic: "result traceability / artefacts"
title: "CIFAR-10 ResNet checkpoints absent; embedding-visualization plots cannot be reproduced"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 12
line_end: 12
quote: |
  Some checkpoints from models on synthetic data have been included as examples in the "checkpoints" folder. 
claim: "Only synthetic SmallMLP checkpoints (subset: 4/6 classes, widths 16/32/64/128) ship in checkpoints/; there are zero CIFAR/ResNet checkpoints. plots/cifar10_embeddings_can_be_permuted.py (the cifar10_reps plot, which renders Fig. 3-left and Appendix F.5) calls load_trained_model() from checkpoints/ for ResNetCIFAR10 models, so it fails out of the box."
concern: "The CIFAR embedding-permutation figures (Fig. 3 left, App. F.5) cannot be regenerated from the shipped repo without first retraining all 10 ResNet18 seeds; the underlying numbers (mCCA, KL, losses) are however shipped as result files and do reproduce the paper, so this only blocks the visual plots, not the headline numbers."
resolution: "Ship the trained CIFAR-10 ResNet checkpoints (or the cached embedding tensors the plots consume), or state in the README that CIFAR figures require retraining and give the expected runtime."
cross_refs: ["cifar-mcca-no-entrypoint"]
check_script: _audit_code/check_bound.py
paper_ref: "Section 5.1, Figure 3 (left), Appendix F.5"
tags: [reforms:4, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: cifar-mcca-no-entrypoint
category: missing
topic: "result traceability / wiring"
title: "Script computing CIFAR mCCA (Fig. 3-left 0.55, Fig. 5) has no run.py entry point"
severity: low
confidence: high
status: finding
file: experiments/measure_cca_for_cifar_models.py
line_start: 182
line_end: 197
quote: |
  def get_and_save_mcca_for_cifar_models(final_dimension, device):
      """
      Get and save mean canonical correlations for CIFAR-10 models with the given 
      representational dimension. 
      """
      
      file_name = f'mcca_cifar_fd{final_dimension}.json'

      result_folder = 'results'
      file_path = os.path.join(result_folder, file_name)

      batch_size=32
      mcca_dict = get_mcca_scores_cifar_models(
          final_dimension, batch_size, device, file_path=file_path)

      save_as_json(mcca_dict, file_path)
claim: "get_and_save_mcca_for_cifar_models / get_mcca_scores_cifar_models produce results/mcca_cifar_fd{2,3,5}.json, the source of the reported mCCA=0.55 (Fig. 3-left) and the loss-diff-vs-mCCA scatter (Fig. 5). grep shows these functions are defined but never imported or called from run.py or any other module — they have no CLI/documented entry point."
concern: "The computation that produces a reported headline number (mCCA=0.55) is present but cannot be invoked through the documented run.py interface; a reviewer can only re-derive the value by importing the module manually."
resolution: "Add a make-plot/run.py case (or get-cifar-mcca command) that calls get_and_save_mcca_for_cifar_models so the mCCA values are regenerable through the documented interface."
cross_refs: ["cifar-checkpoints-missing"]
paper_ref: "Section 5.1 (mCCA 0.55); Appendix F.4, Figure 5"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dependencies-unpinned
category: missing
topic: "environment / dependencies"
title: "Dependencies listed but unpinned (no versions for torch/sklearn/numpy/etc.)"
severity: low
confidence: high
status: finding
file: conda_environment.txt
line_start: 9
line_end: 9
quote: |
  conda install matplotlib docopt h5py scikit-learn tqdm scipy sympy pandas
claim: "Neither conda_environment.txt nor README pins any package version (no requirements.txt / environment.yml with versions); only python=3.11 is fixed. The mSVD/mCCA numbers depend on sklearn's PLSSVD/CCA, whose numerical behaviour can shift across versions."
concern: "Without pinned versions the environment cannot be rebuilt deterministically, which is the only thing standing between the shipped result files and a clean re-run; PLSSVD convergence warnings already appear, so version drift is a realistic reproducibility risk."
resolution: "Provide a pinned environment.yml / requirements.txt (exact torch, scikit-learn, numpy, scipy, pandas versions used to produce the results)."
cross_refs: []
paper_ref: "N/A (repo environment)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No defects found. The CLI runs (verified `KL_table` and the constructed-model path end-to-end after
adding only the missing third-party `docopt`/`pandas` packages — no repo code was changed). One
cosmetic note (not a finding): the `run.py` `--plot-type` docstring (`run.py:35`) omits the valid
`'loss_diff_vs_mcca'` option that the `match` statement (`run.py:249-250`) and README §Plots both
support; the option still works, so this is a doc typo with no functional impact.

## difference

```yaml finding
id: table1-label-mismatch-mcca-vs-dsvd
category: difference
topic: "evaluation consistency (paper vs code)"
title: "KL_table latex emits mCCA column, not the max dSVD the Table 1 caption describes"
severity: low
confidence: medium
status: question
file: experiments/kl_to_zero_dissimilar_reps.py
line_start: 162
line_end: 168
quote: |
  def make_latex_table_from_dict(table_dict):
      """ Make a latex table from a dictionary """
      table_dict.pop('mCCA_g', None)
      table_dict.pop('max_d_rep_g', None)

      df = pd.DataFrame.from_dict(table_dict)
      latex = df.to_latex(index=False, float_format="{:.4f}".format)
claim: "make_latex_table_from_dict drops mCCA_g and max_d_rep_g but keeps g_length, KL_div, d_prob, mCCA_f, and max_d_rep_f. The emitted latex therefore has columns (ρ, dKL, dλ_LLV, mCCA_f, max_d_rep_f). The printed Table 1 in the paper shows columns ρ, dKL, dλ_LLV, mCCA, max dSVD."
concern: "The code's auto-generated table includes BOTH mCCA and max dSVD (max_d_rep_f) for the embeddings f, which is consistent with the paper's Table 1; this is at most a column-labelling/ordering nuance, but I could not fully confirm the paper's 'max dSVD' column equals the code's max_d_rep_f vs a separately-reported quantity, so I file it as a question."
resolution: "Confirm that Table 1's 'max dSVD' column corresponds to max_d_rep_f (= max over chosen sets of 1 - mean PLS-SVD covariance of the f embeddings) as emitted by make_latex_table_from_dict, and that the mCCA column is mCCA_f."
cross_refs: []
check_script: _audit_code/run_kl_table.py
paper_ref: "Table 1"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

No methodological defects found within the audited code's scope.

This is a theory paper; the empirical sections are illustrative and were checked for the failure
modes that *do* apply:
- **Held-out evaluation**: distances/mCCA for synthetic and CIFAR models are computed on the **test**
  split by default (`prob_distances.py:41` `data_suff = '_test'`; `data_init.py:29-33` builds an
  independent test set with `random_seed + 1`). No in-sample-only fit.
- **Sample independence / leakage**: train and test are independent draws from the same generative
  distribution (synthetic) or the standard CIFAR-10 split; no group/duplicate leakage surface exists
  for these illustrative measures.
- **Selective reporting**: the model-retention rule (acc > 0.9, ≥5 seeds) is implemented exactly as
  described (`mean_d_prob_vs_width.py:51-53`) and its counts match the paper.

(See SCOPE FILTER list below for the topics that are structurally N/A.)

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 3          | medium       | CIFAR checkpoints absent; CIFAR-mCCA script unwired; deps unpinned.     |
| bug         | 0          | -            | CLI runs; only a docstring typo (loss_diff_vs_mcca omitted).           |
| difference  | 1          | low          | Table 1 column-labelling question (filed as question).                 |
| methodology | 0          | -            | Held-out eval, seeded splits, retention rule all sound and verified.   |

## Top take-aways

1. **`cifar-checkpoints-missing`** (missing, medium): CIFAR-10 ResNet checkpoints are not shipped, so
   the embedding-visualization figures (Fig. 3-left, App. F.5) cannot be regenerated without
   retraining; the underlying numbers are shipped separately and reproduce.
2. **`cifar-mcca-no-entrypoint`** (missing, low): the script computing the CIFAR mCCA values
   (incl. the 0.55 in Fig. 3-left) has no `run.py` entry point; values are only re-derivable by
   importing the module directly.
3. **`dependencies-unpinned`** (missing, low): no pinned versions — environment not deterministically
   rebuildable, the main barrier between shipped results and a clean re-run.
4. **`table1-label-mismatch-mcca-vs-dsvd`** (difference, low, *question*): confirm Table 1's
   "max dSVD" / "mCCA" columns map to the code's `max_d_rep_f` / `mCCA_f`.

## Items that genuinely look fine

- **Table 1** reproduces end-to-end (`_audit_code/run_kl_table.py`); dKL, dλ_LLV, mCCA, max dSVD all
  match the paper across ρ = 3…18.
- **Theorem 4.7 bound** holds on 250 freshly constructed perturbed models (0 violations,
  `_audit_code/check_bound.py`), matching the paper's "always satisfied" claim.
- **Fig. 3-left numbers** (mCCA 0.551, test losses 1.193/1.178, KL 1.131, dλ_LLV 1.546) match the
  paper exactly when read from the shipped `results/*` JSONs.
- **§5.2 retention counts** match exactly (4 classes [19,20,20,20,20]; 6 classes [16,17,19,19,20]).
- **d_prob / dλ_LLV implementation** (`distribution_distance.py`) faithfully implements the paper's
  term definitions (ψ functions, pivot/leave-out selection, max/sum aggregation, λ weighting).
- **Seeding & splits**: data generation and train/test construction are seeded; distances use the
  test split.

## Open questions for the authors

- `table1-label-mismatch-mcca-vs-dsvd`: please confirm the column→quantity mapping for Table 1.
- Will the CIFAR-10 checkpoints (or cached embeddings) be released so Fig. 3-left and App. F.5 are
  reproducible without retraining?

## SCOPE FILTER (structurally N/A topics)

- **Baselines / naive predictors / shortcut baselines**: N/A — no predictive-performance claim is
  made; experiments illustrate distance/similarity behaviour, not accuracy beating a baseline.
- **Pretraining contamination**: N/A — models are trained from scratch (ResNet18 on CIFAR-10, MLPs on
  synthetic); no pretrained encoder/embeddings/API.
- **Temporal integrity**: N/A — no time dimension.
- **Inference-time distribution shift**: N/A — no deployment/inference use case is claimed; the
  question studied is similarity of internal representations.
- **Statistical-integrity tests (p-values, multiple comparisons, CIs)**: N/A — the paper reports no
  significance tests; uncertainty is shown as shaded ±std bands in Fig. 3-right, which trace to the
  per-seed distance arrays.
