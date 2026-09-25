# Code audit — "How Patterns Dictate Learnability in Sequential Data" (paper 713)

## 1. Summary

The repo (`EkMeasurable__Learnability_Ipred`, single commit `3e251eb "Add files via upload"`)
contains the code for an *empirical*, synthetic-data-only paper that introduces a
predictive-information estimator `Î_pred(k,k')`, a learning curve `Λ̂(k)`, and a
minimal-achievable-risk estimator `R̂∞(Q*)`. It has three experiments:
- **Exp 1 / Fig 1** — neural variational estimation of `I_pred` on Gaussian processes
  (`experiments/experiment_1.py`, `utils/estimators.py`, `utils/critics.py`, `utils/lower_bounds.py`).
- **Exp 2 / Fig 2** — AR(p) learning curves (same driver + notebook `nb_experiment_1.ipynb`,
  which recomputes the theoretical curve itself with `sklearn.Ridge`).
- **Exp 3 / Tables 1, 2, 5–8** — Ising-spin chains, EvoRate, `Λ̂(k)`, `R̂∞(Q*)`, model losses
  (`experiments/experiment_2.py`, `utils/process/data_spins.py`, `nb_experiment_2.ipynb`).

What I did: read every source file; ran read-only scripts under `_audit_code/` against the
committed result JSONs (`results_data/spin_xps/*.json`, `results_data/ar_sequence_estimation_*/`)
to (a) compare the saved spin learning-curve / EvoRate / loss values to paper Tables 1 and 6,
(b) check the M-invariance of the reported `Λ̂(k)`, and (c) check the units (bits vs nats) of the
entropy code. I did **not** retrain anything. There is no leakage / held-out-test concern in the
usual sense because all data is synthetic and the "test" quantity is an in-sample
mutual-information / cross-entropy estimate; the substantive concerns are traceability,
unit consistency, reproducibility infrastructure, and a paper↔code dimension mismatch.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig 1 `Î_pred(5,10)` per estimator (NWJ/SMILE/InfoNCE/DV/TUBA) | `utils/lower_bounds.py`, `utils/critics.py`, `experiments/experiment_1.py` | estimators present; only `ConcatCritic_SMILE` enabled in config | partial | Code present, only 1 of 5 bounds enabled; no saved Fig-1 outputs |
| Fig 2 `Λ̂(k)` for AR p=5,10 | `nb_experiment_1.ipynb` (diff of `summary_results.csv`) + saved CSVs | CSV reproduces `final_estimate` per k | ✓ (qualitatively) | Verified curve source present |
| Fig 2 theoretical `Λ̃(k)` | `nb_experiment_1.ipynb` `estimate_conditional_entropy_vector` | recomputed in-notebook | ✓ | Verified |
| §5.2 AR process is in **R³** | `configs/ar_sequence_config.yaml` `dimensions:[5]`, saved `dim5` npy | dim = 5 | ✗ | MISMATCH (dim 5 vs paper 3) — see `ar-dim-mismatch` |
| Table 1 / Table 6 `EvoRate(k)` (M=100k) | `data_spins.compute_word_entropy` (log2/bits) + saved JSON | JSON EvoRate(10)=0.3214; paper 0.2861 | ✗ | MISMATCH — see `entropy-bits-not-nats`, `spin-results-not-reproducing` |
| Tables 1,5–8 `Λ̂(k)` (all M) | `experiment_2.py` `np.diff(predictive_info_1)` + saved JSON | saved ulc(M=100k)=[0.401,0.254,0.189,…]; paper Λ̂=[0.321,0.206,0.151,…], ~identical across all M | ✗ | MISMATCH — see `lambda-table-not-reproduced` |
| Table 1 / 5–8 `R̂k(Q)` model losses | `experiment_2.py` (loss source ambiguous) / `nb_experiment_2.ipynb` (`mean(train_losses[-100:])`) | notebook reports **train** loss as `R^k`; paper calls it empirical risk | ✗ | see `spin-train-loss-as-risk` |
| Table 1 `R̂∞(Q*) = min_k {R̂k−Λ̂(k)}` | derived in paper text from the two columns above | inherits both mismatches above | ✗ | MISSING/derived, not computed by any script |
| Table 2 optimal orders k* | (no script computes `argmin_k {R̂k−Λ̂}` and reports it) | — | — | MISSING (no traceable script) |
| Appendix B.5 / Table 4 estimator comparison (Tpast=30,Tfut=40) | `KernelBasedGenerator` present in `process.py` | generator present; no saved outputs/driver for Table 4 | — | partial |
| Dependency spec (versions) | (none) | — | — | MISSING — see `no-dependency-pinning` |

## 3. Findings

## missing

```yaml finding
id: lambda-table-not-reproduced
category: missing
topic: "result traceability / learning curve"
title: "Reported Λ̂(k) (Tables 1,5–8) not reproduced by the repo's spin learning-curve code"
severity: high
confidence: medium
status: finding
file: experiments/experiment_2.py
line_start: 269
line_end: 270
quote: |
    word_lengths, entropy_empirical, evoRate_empirical, predictive_info_1 = computing_entropy(force_recompute=False)
    universal_learning_curve = np.diff(predictive_info_1).tolist()
claim: "The Ising-spin learning curve Λ̂(k) is computed in the repo as np.diff(predictive_info_1), where predictive_info_1 = S(k) - k*l0 with l0 estimated by a linregress slope on log2 block entropies (experiment_2.py:50-55). The committed result for M=100,000 (results_data/spin_xps/results_exp_100000.json) gives Λ̂=[0.401,0.254,0.189,...], whereas the paper's Tables 5-8 report Λ̂(k)=[~0.321,~0.206,~0.151,...] that are nearly identical for ALL four block sizes M (differences < 0.003 across M=10k/100k/1M; see _audit_code/out/spin_table_compare.txt and check below)."
concern: "The paper's Λ̂(k) is essentially block-size-invariant and matches a smooth ~p/2k analytic decay, which cannot be the noisy np.diff of an empirical predictive-information curve computed on different M-block datasets; the headline R̂∞(Q*) and dimΘ estimates depend on these Λ̂ values, so the number-producing pipeline for the main table is not the code in this repo."
resolution: "Authors: provide the exact script that produced the Λ̂(k) column of Tables 1 and 5-8; confirm whether it is the empirical np.diff(predictive_info_1) of experiment_2.py or an analytic/Ridge-based curve, and explain why the reported values are nearly identical across all block sizes M."
cross_refs: ["spin-results-not-reproducing", "entropy-bits-not-nats"]
check_script: _audit_code/check_spin_table.py
paper_ref: "Tables 1, 5, 6, 7, 8"
tags: [reforms:2, forensics:hidden-iteration]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-pinning
category: missing
topic: "reproducibility / environment"
title: "No dependency specification with versions (only an unpinned package list in README)"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 5
line_end: 18
quote: |
  Make sure you have the following installed:

  - PyTorch 
  - Python3
  - Numpy
  - pandas 
  - matplotlib
  - json
  - tqdm
  - yaml
  - pathlib
  - datetime
  - seaborn
  - sklearn
claim: "The repo has no requirements.txt / environment.yml / pyproject.toml / setup.py; the only dependency information is this unpinned bullet list in the README (verified: ls of the repo root finds none of those files)."
concern: "Without pinned versions the environment cannot be rebuilt deterministically, and the list mixes stdlib modules (json, datetime, pathlib) with PyPI packages, giving no reproducible install path."
resolution: "Authors: add a requirements.txt / environment.yml with pinned versions (at minimum torch, numpy, scipy, scikit-learn, pandas)."
cross_refs: []
paper_ref: "NeurIPS checklist Q5 (open access to code)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: table2-orders-no-script
category: missing
topic: "result traceability"
title: "Table 2 estimated optimal regression orders k* not produced by any script"
severity: low
confidence: medium
status: question
file: experiments/experiment_2.py
line_start: 304
line_end: 311
quote: |
            model_results[f"k_{k}"] = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "theoretical_optimal": val_losses[-1] - universal_learning_curve[k-1] if k-1 < len(universal_learning_curve) else None
            }
claim: "experiment_2.py saves per-k losses and a 'theoretical_optimal' field, but no code in the repo computes argmin_k {R̂k(Q)-Λ̂(k)} and reports it as the optimal order k* of Table 2; the values (e.g. k*=18,16 for M=1M) are only in the paper."
concern: "Table 2's k* values are not traceable to a computation in the repo, so they cannot be re-derived or checked."
resolution: "Authors: point to the script/cell that computes the argmin reported as k* in Table 2."
cross_refs: ["lambda-table-not-reproduced"]
paper_ref: "Table 2"
tags: [reforms:2]
```

## bug

```yaml finding
id: hardcoded-project-path
category: bug
topic: "runnability / paths"
title: "Hardcoded PROJECT_PATH './Data_Pattern_Learnability' does not match repo dir; scripts fail unless edited"
severity: medium
confidence: high
status: finding
file: experiments/experiment_1.py
line_start: 2
line_end: 3
quote: |
  PROJECT_PATH = "./Data_Pattern_Learnability"  # Absolute path to the project directory
  sys.path.append(PROJECT_PATH)  # Add the project path to sys.path
claim: "experiment_1.py, experiment_2.py and utils/estimators.py all hardcode PROJECT_PATH='./Data_Pattern_Learnability' and build every input/output path (configs, results_data, sys.path) from it; the cloned repo directory is 'Learnability_Ipred', so all reads/writes target a non-existent 'Data_Pattern_Learnability' folder. The README explicitly instructs the user to edit this path before running."
concern: "Out of the box the scripts load no config and write to the wrong location (or crash on missing files); reproduction requires undocumented manual path edits, and the value left in the file is a relative placeholder, not even an absolute path."
resolution: "Authors: derive PROJECT_PATH from __file__ / pathlib so the code runs from a fresh clone without manual edits, or document the exact required folder name."
cross_refs: []
paper_ref: "README 'Initial Setup'"
tags: [reforms:2, lones:stage-7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: entropy-bits-not-nats
category: difference
topic: "units / metric"
title: "Entropy code uses log2 (bits) but paper states all info-theoretic results are in nats"
severity: medium
confidence: high
status: finding
file: utils/process/data_spins.py
line_start: 216
line_end: 221
quote: |
    total = sum(freq.values())
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * np.log2(p)
    return entropy
claim: "compute_word_entropy (the function feeding EvoRate, predictive_info_1 and Λ̂ for the Ising experiment) computes entropy in bits via np.log2, and EvoRate/Λ̂ are built from it (experiment_2.py:42-55). The paper's Appendix C states 'all information-theoretic results and measures ... are reported in nats rather than in bits' using the natural logarithm, and the model losses in the same Tables are cross-entropy in nats (torch.nn.CrossEntropyLoss, experiment_2.py:182)."
concern: "EvoRate/Λ̂ (bits) and the model risks R̂k (nats) are placed in the same tables and combined as R̂k − Λ̂; mixing log bases makes the subtraction and the reported nat-valued quantities unit-inconsistent unless a 1/ln2 conversion is applied somewhere, which is not visible in the repo."
resolution: "Authors: confirm in which base the table values are reported and where (if anywhere) bits→nats conversion happens before combining Λ̂ with the cross-entropy losses."
cross_refs: ["spin-results-not-reproducing", "lambda-table-not-reproduced"]
check_script: _audit_code/check_spin_table.py
paper_ref: "Appendix C 'Note on the Use of Nats'; Tables 1, 5-8"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ar-dim-mismatch
category: difference
topic: "experimental setting"
title: "AR experiment uses dimension 5, paper §5.2 states the process is in R³"
severity: low
confidence: high
status: finding
file: configs/ar_sequence_config.yaml
line_start: 18
line_end: 21
quote: |
    params:
      p: [5]
      rho: [0.9]
      dimensions: [5]
claim: "The committed AR config and the saved results (filenames sequence_autoregressive_dim5_*, summary_results.csv with dim=5) use dimension d=5, but the paper §5.2 states the AR process is 'a stationary vector autoregressive process {Xt} ⊂ R³ of order p ∈ {5,10}'."
concern: "The dimension actually used to produce Fig 2 (d=5) differs from the d=3 stated in the paper; either the figure was made with a different dim than reported or the saved artefacts are from a different configuration."
resolution: "Authors: confirm the dimension used for Fig 2 and reconcile the d=3 in the text with d=5 in the config/saved outputs."
cross_refs: []
paper_ref: "Section 5.2 (R³); configs/ar_sequence_config.yaml"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: spin-train-loss-as-risk
category: difference
topic: "evaluation / reported metric"
title: "Spin-table risk R̂k(Q) is reported from training loss in the results notebook"
severity: medium
confidence: medium
status: finding
file: experiments/notebooks/nb_experiment_2.ipynb
line_start: 51
line_end: 53
quote: |
        train_losses = model_data["train_losses"]
        
        # Compute the mean of the last 100 epochs
        final_loss = np.mean(train_losses[-100:])
claim: "The notebook that builds the spin results table sets final_loss = mean of the last 100 TRAIN losses and labels it the model risk R^k(Q) (rendered column 'Final Loss (mean last 100)', compared with R̂∞ and EvoRate). experiment_2.py also stores final_train_loss/final_val_loss but uses val_losses[-1] for its own 'theoretical_optimal'. So the per-k risk reported is the in-sample training loss, while the paper presents R̂k(Q) as an empirical risk and contrasts it with the optimal achievable risk."
concern: "Using the training loss (rather than the held-out 20% validation loss the code also computes) as the reported empirical risk biases R̂k(Q) downward and makes the gap to R̂∞(Q*) and the model-adequacy ratio R̂k/R̂∞ unreliable; the choice between train and val loss is inconsistent across the two files."
resolution: "Authors: confirm whether Tables 1/5-8 use training or validation loss for R̂k(Q); if training, justify using in-sample loss as the empirical risk, and reconcile with experiment_2.py which uses validation loss for theoretical_optimal."
cross_refs: ["spin-results-not-reproducing"]
check_script: _audit_code/check_spin_table.py
paper_ref: "Tables 1, 5-8 (R̂k(Q) columns)"
tags: [reforms:4, reforms:6]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: spin-patience-mismatch
category: difference
topic: "training hyperparameters"
title: "Early-stopping patience is 50 in code, 10 in the paper"
severity: low
confidence: high
status: finding
file: experiments/experiment_2.py
line_start: 296
line_end: 302
quote: |
            train_losses, val_losses, best_epoch = training_a_predictive_model(
                k=k,
                model_type=model_type,
                max_epochs=1000,
                patience=50,
                num_batches_per_epoch=50
            )
claim: "main() trains the spin models with patience=50, but the paper (Appendix C.1 and the §5.3 training setup) states early stopping with 'patience of 10 epochs'."
concern: "A patience mismatch (50 vs 10) changes how long models train and therefore the reported R̂k(Q) losses that drive Table 1; the run actually used differs from the documented protocol."
resolution: "Authors: confirm the patience used for the reported numbers and align code or text."
cross_refs: ["spin-train-loss-as-risk"]
paper_ref: "Appendix C.1 / §5.3 (patience 10)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: only-one-estimator-enabled
category: difference
topic: "experiment coverage / config"
title: "Fig-1 reports 5 MI estimators but the shipped config enables only ConcatCritic+SMILE"
severity: low
confidence: high
status: finding
file: configs/ar_sequence_config.yaml
line_start: 57
line_end: 66
quote: |
  estimators:
    - name: "ConcatCritic_SMILE"
      enabled: true
      estimator_type: "smile"
      critic_type: "ConcatCritic"
      critic_params:
        hidden_dim: 256
        layers: 2
        activation: "relu" 
claim: "Fig 1 / Table 4 report Î_pred for SMILE, NWJ, InfoNCE, DV and TUBA across many kernels, but the committed config enables a single estimator (ConcatCritic_SMILE) on a single AR kernel; the other bounds exist in utils/lower_bounds.py but no config/driver in the repo runs the full Fig-1/Table-4 sweep."
concern: "The configuration that produced the multi-estimator, multi-kernel Fig 1 and Table 4 is absent, so those panels are not directly reproducible from the shipped config."
resolution: "Authors: provide the config(s) that enumerate all estimators and kernels used for Fig 1 and Table 4."
cross_refs: []
paper_ref: "Figure 1; Appendix Table 4"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: spin-results-not-reproducing
category: methodology
topic: "reproducibility / no seeding"
title: "Saved spin results do not reproduce paper Table 6 and no randomness is seeded"
severity: medium
confidence: medium
status: question
file: experiments/experiment_2.py
line_start: 158
line_end: 166
quote: |
    data_generator = GeneratingData(chain_length=10**9)
    chain_1_bin = data_generator.generate_chain_2(block_size=block_size)
    
    # Create datasets
    full_dataset = SpinDataset(chain_1_bin, k)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
claim: "The committed M=100,000 spin results (results_data/spin_xps/results_exp_100000.json) give per-k LSTM losses (e.g. k=2: train≈0.637, val[-1]≈0.624) and EvoRate(10)=0.3214 that do not match paper Table 6 (k=2 R_LSTM=0.5789; EvoRate(10)=0.2861); see _audit_code/out/spin_table_compare.txt. No seed is set for numpy (chain generation), torch (model init) or random_split, and the chain is regenerated (chain_length=10**9) on every run."
concern: "With no seeding and a freshly generated 10**9-long chain per run, the reported Table 1/6 numbers are not reproducible, and the committed JSON (the only artefact shipped) does not match the published table, so the exact numbers cannot be verified."
resolution: "Authors: seed all RNGs and ship the exact result JSON (or the seed + chain) that produced Tables 1, 2, 5-8; clarify why the committed results_exp_100000.json differs from Table 6."
cross_refs: ["lambda-table-not-reproduced", "entropy-bits-not-nats", "spin-train-loss-as-risk"]
check_script: _audit_code/check_spin_table.py
paper_ref: "Tables 1, 6"
tags: [reforms:2, reforms:7, heil:bronze]
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 3          | high         | Headline Λ̂(k) table not produced by repo code; no pinned deps; k* untraced |
| bug         | 1          | medium       | Hardcoded relative PROJECT_PATH breaks a fresh clone                    |
| difference  | 6          | medium       | bits vs nats; train-vs-val loss as risk; dim 5 vs 3; patience 50 vs 10  |
| methodology | 1          | medium       | Saved spin results don't match Table 6; no seeding                      |

## 5. Closing lists

**Top take-aways** (ranked):
1. `lambda-table-not-reproduced` (missing, high) — the reported `Λ̂(k)` is ~identical across all block sizes M and does not equal the repo's `np.diff(predictive_info_1)` (M=100k → 0.401,0.254,… vs paper 0.321,0.206,…); the main-table learning curve is not produced by the shipped spin code.
2. `entropy-bits-not-nats` (difference, medium) — entropy/EvoRate/Λ̂ are computed in bits (`np.log2`) while the paper says nats and combines them with nat-valued cross-entropy losses.
3. `spin-train-loss-as-risk` (difference, medium) — the results notebook reports the **training** loss (mean of last 100 epochs) as the empirical risk `R̂k(Q)`.
4. `spin-results-not-reproducing` (methodology, medium) — committed spin JSON does not match Table 6 and nothing is seeded.
5. `hardcoded-project-path` (bug, medium) — `PROJECT_PATH="./Data_Pattern_Learnability"` does not match the repo folder; scripts fail without manual edits.
6. `no-dependency-pinning` (missing, medium) — no requirements/environment file at all.

**Items that genuinely look fine** (actively checked):
- `AutoRegressiveGenerator.generate_long_array` (`process.py:45-57`) faithfully implements Eq. 9 (coeffs `rho/p` over the last `p` states, noise scale `sqrt(1-rho²)`).
- The active `smile_lower_bound` imported in `estimators.py` is `utils.lower_bounds.smile_lower_bound`, which correctly returns `js + (dv-js).detach()` (the stable SMILE objective); the divergent `dv`-only copy in `utils/utils.py` is not used.
- EvoRate is computed as `S(k)+S(1)−S(k+1)` (`experiment_2.py:42-47`), a valid `I(past-k ; next-symbol)` mutual-information definition.
- The Fig-2 theoretical curve in `nb_experiment_1.ipynb` (`estimate_conditional_entropy_vector` via Ridge + Gaussian conditional-entropy formula, and `calculate_l_0_ar`) matches Appendix C's described `l(k)`/`l0` estimation.
- No train/test leakage in the conventional sense: data is synthetic and the quantities of interest are in-sample information estimates (the train-vs-val concern is filed separately as `spin-train-loss-as-risk`).

**Open questions for the authors:**
- Which exact script produced the `Λ̂(k)` column of Tables 1/5–8, and why is it block-size-invariant? (`lambda-table-not-reproduced`)
- Are the table values in bits or nats, and where does any bits→nats conversion occur? (`entropy-bits-not-nats`)
- Is `R̂k(Q)` the training or validation loss, and what patience/seed produced the shipped results? (`spin-train-loss-as-risk`, `spin-patience-mismatch`, `spin-results-not-reproducing`)
- Is the AR process dimension 3 (paper) or 5 (config/saved outputs)? (`ar-dim-mismatch`)
