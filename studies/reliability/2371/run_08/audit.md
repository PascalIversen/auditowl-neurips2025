# Audit — Paper 2371: "Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation" (SDForger)

## Summary

The workspace contains two author code trees. `code/IBM__fms-dgt/` is the public
GitHub release, which holds only the SDForger *generation* databuilder
(`fms_dgt/public/databuilders/time_series/`) — no evaluation, metrics, or
baselines. `code/SDForger__neurips_supplemental/` is the NeurIPS supplemental
ZIP and is the intended full reproduction package: it contains the SDForger
generation pipeline (`utils/augmentation/`), the similarity metrics
(`utils/evaluation/`), the TTM utility evaluation, the runner scripts
(`sources/`), pinned conda environments, and the 12 benchmark datasets
(`data/*.csv`). I read the paper text (`paper_text.txt`, citing `paper.pdf`),
mapped every numbered Table/Figure metric to its computing function, and ran
three read-only checks under `_audit_code/`:
`check_traceability.py` (which paper artefacts have computing code),
`check_seed_override.py` (whether the config `seed` reaches the LLM), and an
inspection of `notebook/conditional_generation.ipynb`. I did not execute the
training/generation pipeline (it requires GPUs, GPT-2 weights, and the external
`tsfm_public` package).

Headline findings: all of SDForger's *own* metric values are traceable to code,
but (1) none of the five baseline generators (TimeVAE, TimeVQVAE, RTSGAN,
SDEGAN, LS4) that fill the comparison columns of Tables 1 and 2 are in the repo,
nor are their precomputed outputs, so the "outperforms baselines" claim is not
reproducible from these artefacts; and (2) the Section-6 "0.81 accuracy" kNN
classifier is absent.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — SDF-ICA/FPC: MDD | `utils/evaluation/feature_based_measures.py:88` (`calculate_mdd`) | not run | — | Code present (not executed) |
| Table 1 — ACD | `feature_based_measures.py:155` (`calculate_acd`) | not run | — | Code present |
| Table 1 — SD | `feature_based_measures.py:183` (`calculate_sd`) | not run | — | Code present |
| Table 1 — KD | `feature_based_measures.py:215` (`calculate_kd`) | not run | — | Code present |
| Table 1 — ED | `utils/evaluation/distance_based_measures.py:13` (`calculate_ed`) | not run | — | Code present |
| Table 1 — DTW | `distance_based_measures.py:28` (`calculate_dtw`) | not run | — | Code present |
| Table 1 — SHR | `utils/evaluation/shapelet_based_measures.py:6` | not run | — | Code present |
| Table 1 — TimeVAE/TimeVQVAE/RtsGAN/SdeGAN/LS4 rows | (none) | — | — | **MISSING (no baseline generator/results)** |
| Table 1 — "Norm. Avg." + "Rank" columns | (none) | — | — | **MISSING (no aggregation/ranking code)** |
| Table 2 — SDForger RMSE/MASE/WQL | `utils/evaluation/utils_ttm.py:243-259, 301` | not run | — | Code present |
| Table 2 — baseline-generated rows + "Avg. Rank" | (none) | — | — | **MISSING (no baseline generators / ranking)** |
| §6 / Fig. 2 — "accuracy of 0.81" (longitudinal kNN classifier) | (none) | — | — | **MISSING (no classifier code)** |
| Fig. 2 — conditional generation plot | `notebook/conditional_generation.ipynb` | not run | — | Plotting code present |
| Appendix Tables D.10–D.14 (per-dataset similarity) | same metric fns as Table 1 (SDForger only) | not run | — | SDForger present; baselines MISSING |

## missing

```yaml finding
id: baseline-generators-missing
category: missing
topic: "result traceability / baselines"
title: "No baseline generators (TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4) in repo; Tables 1 & 2 comparisons unreproducible"
severity: high
confidence: high
status: finding
file: out/traceability.csv
csv_row: 10
quote: |
  10,Baseline generator TimeVAE,MISSING
  11,Baseline generator TimeVQVAE,MISSING
  12,Baseline generator RTSGAN,MISSING
  13,Baseline generator SDEGAN,MISSING
  14,Baseline generator LS4,MISSING
claim: "The supplemental package contains SDForger's generation and the shared evaluation metrics, but no code (and no precomputed outputs) for any of the five baseline generators compared against in Tables 1 and 2."
concern: "The paper's central comparative claim ('SDForger outperforms existing generative models', Abstract; Table 1 bold/rank, Table 2 Avg. Rank) cannot be reproduced or checked from this repo because the baseline rows have neither generating code nor stored results."
resolution: "Authors: please add the baseline generation scripts (or their precomputed synthetic-data / result files) and the configuration used, so the baseline rows of Tables 1 and 2 and the average-rank columns can be reproduced."
cross_refs: ["table1-aggregation-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "Abstract; Table 1; Table 2; §4 'Baselines'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: section6-classifier-missing
category: missing
topic: "result traceability"
title: "Section 6 '0.81 accuracy' kNN classifier code is absent"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  For instance, using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
  on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2).
claim: "Section 6 reports an accuracy of 0.81 from a longitudinal kNN classifier (scikit-fda), but no script, notebook cell, or function in the supplemental package computes any classifier accuracy; the only Section-6 artefact present (`notebook/conditional_generation.ipynb`) does conditional generation and plotting (Figure 2) but not classification."
concern: "The headline number quantifying text-conditioned generation quality (0.81) has no computing code in the repo, so it cannot be reproduced or verified."
resolution: "Authors: please add the classifier evaluation script (train/test setup, label definition, scikit-fda kNN configuration) that produces the 0.81 accuracy."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "Section 6 'Shaping time series with language'; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: table1-aggregation-missing
category: missing
topic: "result traceability / aggregation"
title: "No code computes Table 1 'Norm. Avg.' / 'Rank' or Table 2 'Avg. Rank' columns"
severity: medium
confidence: high
status: finding
file: out/traceability.csv
csv_row: 10
quote: |
  10,Baseline generator TimeVAE,MISSING
claim: "The repo computes per-(metric, dataset) raw scores for SDForger only; there is no script that normalises scores into [0-1], averages them per metric group, or computes the average-rank columns reported in Tables 1 and 2."
concern: "The normalized-average and rank columns are headline summaries used to argue 'balanced' best-overall performance, but the aggregation/ranking computation (which also requires the missing baselines) is not in the repo."
resolution: "Authors: please add the normalization and ranking aggregation script that turns raw per-method metric scores into the 'Norm. Avg.', 'Rank', and 'Avg. Rank' columns."
cross_refs: ["baseline-generators-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "Table 1 (Norm. Avg., Rank); Table 2 (Avg. Rank)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ttm-dependency-unpinned
category: missing
topic: "dependencies / environment"
title: "tsfm_public (granite-tsfm) required for Table 2 is not pinned in the env files"
severity: low
confidence: high
status: finding
file: README.md
line_start: 39
line_end: 41
quote: |
  git clone "https://github.com/ibm-granite/granite-tsfm.git" 
  cd granite-tsfm
  pip install ".[notebooks]"
claim: "The TTM utility evaluation (`utils/evaluation/utils_ttm.py`) imports `tsfm_public`, but neither conda env file (`sdforgerpy310cuda.yaml`, `sdforgerpy310mps.yaml`) lists it; the README instructs cloning `granite-tsfm` from `main` with no pinned commit/tag."
concern: "Because the external dependency that produces Table 2 is unpinned to a specific version, the forecasting utility results may not be exactly reproducible if the upstream `tsfm_public`/TTM weights change."
resolution: "Authors: pin `tsfm_public`/`granite-tsfm` to a specific commit or release and add it (and the TTM model revision) to the environment specification."
cross_refs: []
paper_ref: "Table 2 utility evaluation; §4 utility metrics (TTM)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: bf16-hardcoded-breaks-mps-path
category: bug
topic: "reproducibility / training config"
title: "bf16=True is hardcoded in SDForger.fit, conflicting with the documented MPS run path"
severity: low
confidence: medium
status: finding
file: utils/augmentation/sdforger.py
line_start: 287
line_end: 306
quote: |
            training_args = TrainingArguments(
                self.output_dir,
                **self.train_args,
                adam_epsilon=1e-04,
                logging_strategy="steps",
                logging_steps=10,
                # weight_decay=0.0001,
                # optim='adafactor',
                # max_grad_norm=1,
                # max_grad_norm=5,
                evaluation_strategy="steps",
                eval_steps=5,
                save_strategy="steps",
                save_steps=100,
                load_best_model_at_end=True,
                # metric_for_best_model="loss",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                bf16=True,
            )
claim: "`bf16=True` is passed unconditionally to HF `TrainingArguments`, while on darwin `self.train_args` also sets `use_mps_device=True` (lines 47-50); the README documents a local MPS run path (`sdforgerpy310mps.yaml`)."
concern: "bf16 mixed-precision training is not supported on Apple MPS / CPU in this configuration, so the documented MPS reproduction command is likely to raise an error before training; it works on the A100/CUDA path used for the paper but not on the alternative path the README advertises."
resolution: "Authors: gate `bf16` on CUDA availability (e.g. set it from device, as in the commented-out line 50) so the documented MPS/CPU path runs."
cross_refs: []
paper_ref: "README 'Setup' (MPS); Parameter settings §4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: sdforger-seed-hardcoded
category: bug
topic: "reproducibility / seeding"
title: "Config `seed` is not propagated to the SDForger LLM; its RNG is hardcoded to default 42"
severity: low
confidence: high
status: finding
file: utils/augmentation/sdforger.py
line_start: 86
line_end: 87
quote: |
        self.seed = kwargs['seed'] if 'seed' in kwargs else 42
        self.set_seed()
claim: "`sdforger_augmentation` constructs `SDForger(model_path=llm, text_template=..., float_type=...)` (sdforger_augmentation.py:224) without a `seed` kwarg, so `SDForger.__init__` falls back to `seed=42` and calls `self.set_seed()`, re-seeding random/numpy/torch to 42 for the LLM fine-tuning split, shuffling, and generation — regardless of the `seed` value in the config (e.g. `seed: 54`)."
concern: "Changing the documented `seed` config field does not change the LLM fine-tuning/generation randomness (only the FPCA/ICA embedding and preprocessing, which were seeded earlier), so the seed parameter is misleading and runs labelled with different seeds share the identical LLM RNG state."
resolution: "Authors: forward the config `seed` into the `SDForger(...)` constructor (e.g. `SDForger(..., seed=SEED)`), or document that the LLM RNG is fixed at 42."
cross_refs: []
check_script: _audit_code/check_seed_override.py
paper_ref: "config.yaml `seed`; Parameter settings §4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

No `difference`-only findings. The implemented similarity metrics (ED paired
index-wise between an original series and "its generated" series) match the
paper's Appendix B definitions and the TSGBench convention they cite (Ang et
al., 2023). The config defaults differ from paper settings (e.g.
`config.yaml` ships `sdforger_embedding_dim: 5` while Table 1 uses k=3) but the
CLI/config fully supports the paper's setting, so per the prompt's exclusion
rule this is not a finding.

## methodology

No `methodology` findings. The TTM utility evaluation uses a chronologically
forward train/val/test split (`utils_preprocess_data.py:339-357`: contiguous
slices `start → end_train → end_val → end_test`, with the StandardScaler fit on
train and applied to val/test), and the test set is the most recent slice —
appropriate for forecasting. The similarity evaluation measures fidelity of
synthetic data to the *training* windows, which is the standard goal for
generative-model similarity (matching the cited TSGBench protocol), not a
leakage flaw.

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 4          | high         | Baselines, §6 classifier, aggregation, unpinned TTM dep absent |
| bug         | 2          | low          | Hardcoded bf16 breaks MPS path; config seed not wired to LLM |
| difference  | 0          | -            | Metrics & config match paper / are CLI-configurable        |
| methodology | 0          | -            | Forward TTM split + train-fidelity similarity are sound     |

## Top take-aways

1. **[missing, high]** No baseline generators (TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4) or their outputs are in the repo, so the baseline columns and "outperforms baselines" claim in Tables 1 & 2 are not reproducible (`baseline-generators-missing`).
2. **[missing, medium]** The Section-6 "0.81 accuracy" kNN classifier has no computing code (`section6-classifier-missing`).
3. **[missing, medium]** No code computes Table 1 "Norm. Avg."/"Rank" or Table 2 "Avg. Rank" aggregations (`table1-aggregation-missing`).
4. **[bug, low]** `bf16=True` is hardcoded, breaking the documented MPS reproduction path (`bf16-hardcoded-breaks-mps-path`).
5. **[bug, low]** The config `seed` is silently ignored by the SDForger LLM, which always uses seed 42 (`sdforger-seed-hardcoded`).
6. **[missing, low]** `tsfm_public`/granite-tsfm (needed for Table 2) is unpinned and absent from the env files (`ttm-dependency-unpinned`).

## Items that genuinely look fine

- All seven similarity metrics and the three TTM utility metrics for SDForger have explicit, locatable computing functions (`_audit_code/check_traceability.py`).
- Dependencies for the core SDForger pipeline are pinned in both conda env files (torch, transformers==4.46.3, scikit-learn==1.6.1, dtaidistance==2.3.13).
- The 12 benchmark datasets are all shipped under `data/*.csv`.
- `utils/generals.py:set_seed` seeds random/numpy/torch/HF/cuda for the embedding and preprocessing stages.
- The TTM forecasting split is chronologically forward with train-only scaler fitting (no obvious temporal leakage).

## Open questions for the authors

- Were the baseline synthetic datasets generated with a separate (unreleased) pipeline, and can the precomputed baseline outputs or generation scripts be shared so Tables 1–2 can be reproduced end-to-end?
- What were the exact train/test construction and label definition for the Section-6 longitudinal kNN classifier that yields accuracy 0.81?
