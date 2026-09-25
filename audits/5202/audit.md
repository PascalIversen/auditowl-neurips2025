# Code-repository audit — GABAR: Graph Neural Network Based Action Ranking for Planning (NeurIPS 2025, #5202)

## 1. Summary

The repo (`code/Learning-for-Seq-Decision-Making__GABAR-...`) is the author code for GABAR,
a GNN+GRU action-ranking policy for classical planning. It is a research codebase built on
top of a PLOI/PDDLGym scaffold: `main.py` is the single entrypoint (the GABAR method is
selected with `--method ltp` and `--ablation main`), `ploi/datautils_ltp.py` builds the
action-centric graphs, `ploi/modelutils_ltp.py` / `ploi/ablations.py` hold the model and its
ablations (`no_cd`, `no_ag`, `val`, `--use-global-node False`), `ploi/traineval.py` trains it
(cross-entropy on action + object scores, matching the paper's `L = L_action + L_objects`),
and `ploi/run_planner_with_ltp_v2.py` + `ploi/test_utils.py` run the learned policy on test
problems and compute Coverage and the Plan-Quality-Ratio (PQR). Baselines GPL/ASNets/GRAPL
live under `ploi/baselines/exp_{1,2,3}`. The PQR denominator (Fast-Downward `fd-lama-first`
plan lengths) is cached as JSON under `cache/results/planner_data/*_non_optimal.json`.

I read the paper (PDF + text extraction), the README, `main.py`, the data pipeline, the
train/eval functions, the checkpoint manager, the test harness, and the runner scripts. I ran
three read-only checks under `_audit_code/` (outputs in `_audit_code/out/`): a repo-completeness
scan (`check_repo_completeness.py`), a model-selection / hyperparameter-sweep scan
(`check_selection_on_test.py`), and an HTTP probe of the dataset URL in the README. The PQR
metric (`compute_combined_metrics`) and the ablation wiring are faithful to the paper; the main
concerns are (a) the dataset that defines train/val/test instances is not in the repo, (b) the
test harness reports the checkpoint with the best **test-set** coverage rather than the
validation-loss checkpoint the paper claims, and (c) an undisclosed hyperparameter sweep.

## 2. Traceability table

Coverage/PQR numbers are produced by re-running `main.py` per domain; the values are not stored
in the repo, so "Computed value" is "(not stored; recomputable iff dataset present)". The
underlying *computation* exists and is faithful for GABAR and its ablations/baselines; the
blocker is the missing dataset (see `missing-pddl-problems`).

| Paper artefact | Repo location (computes value) | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 GABAR Coverage (all domains/diff) | `ploi/test_utils.py:154` (`compute_metrics`) via `run_planner_with_ltp_v2.py` | recomputable iff dataset present | — | Code present; data MISSING |
| Table 2 GABAR PQR `P` | `ploi/test_utils.py:186` (`compute_combined_metrics`, `total_plan_len_non_opt/total_plan_len_learned`) | recomputable iff dataset present | ✓ (formula matches §4.1 PQR def) | Code present; data MISSING |
| Table 2 GPL/ASNets/GRAPL baselines | `ploi/baselines/exp_{1,2,3}/` + `run_planner_with_ltp_v2.py` | recomputable iff dataset+weights | — | Code present; data MISSING |
| Table 2/4 ablations GABAR-ACT/-CD/-RANK/-G/-ACT_CD | `ploi/ablations.py` (`GNN_non_CD`,`GNN_non_AG_CD`,`GNN_Val`), `--use-global-node` | recomputable iff dataset present | — | Code present; data MISSING |
| PQR denominator (FD fd-lama-first plan lengths) | `cache/results/planner_data/*_non_optimal.json` + `run_planner_with_ltp_v2.py:793` | cached (e.g. 411 entries for blocks) | — | Verified present |
| Table 5 GABAR vs OpenAI-O3 / Gemini-2.5-Pro (App. A.3) | (none) | — | — | MISSING (no LLM-comparison code) |
| Reported checkpoint = "lowest validation loss" (§3.4) | `ploi/test_utils.py:356` (`log_model_metrics`) selects max **test** success | n/a | ✗ (selects by test coverage) | MISMATCH → `methodology` |
| Training config "lr 0.0005, 9 GNN rounds, bs 16, hidden 64" single run (§3.4) | `train_test_scripts/ltp_all_run.sh:57-61,175` sweeps n_heads∈{1,2,4,8}, attn-drop∈{0.1,0.2} | n/a | ✗ (undisclosed sweep) | MISMATCH → `difference` |

## 3. Findings

## missing

```yaml finding
id: missing-pddl-problems
category: missing
topic: "data availability"
title: "Train/val/test PDDL problems absent from repo; only an inaccessible anonymized URL"
severity: high
confidence: high
status: finding
file: README.md
line_start: 23
line_end: 26
quote: |
  ### Additional Requirements

  For use with pddlgym, we require our fork of [pddlgym](https://anonymous.4open.science/r/pddlgym-0F03/
    ), which houses our custom domains and problems.
claim: "All training, validation, and test PDDL instances live in a separate custom pddlgym fork; the repo itself contains no .pddl domain or problem files (check_repo_completeness.py: pddl_problem_files_in_repo=0), and the only pointer is an anonymous.4open.science link that returns HTTP 401."
concern: "Every headline number (Tables 2/4/5/6) is computed over these instances, but the dataset defining train/val/test is not in the repo and the linked source is unreachable, so none of the results can be reproduced."
resolution: "Authors: provide the custom pddlgym domains/problems (or the supplementary dataset) at a resolvable, non-anonymized location and pin which problem indices form each easy/medium/hard test subset."
cross_refs: []
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Appendix A.2 footnote 'Dataset submitted as supplementary material'; Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-dependency-spec
category: missing
topic: "reproducibility / dependencies"
title: "No dependency-specification file; deps only as prose with several unpinned packages"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 3
line_end: 21
quote: |
  pip install numpy==1.26.0
  pip install protobuf==3.20.0
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  pip install torch-geometric==2.6.1
  pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
  pip install tensorboard
  pip install torchviz==0.0.2
  pip install pytorch-lightning==2.0.1
  pip install wandb
  pip install icecream
  pip install pymimir==0.9.71
  pip install pyperplan==2.1
  pip install pandas
claim: "There is no requirements.txt / setup.py / environment.yml / pyproject.toml in the repo (check_repo_completeness.py: dependency_spec_file=0); dependencies are listed only as README pip commands, and torch/torchvision/torchaudio, tensorboard, wandb, icecream, pandas, termcolor, tarski, and clingo are unpinned."
concern: "Without a pinned, machine-readable environment the build is not deterministically reproducible (e.g. an unpinned torch against the pinned torch-geometric 2.6.1 / torch-2.6.0 wheels can break)."
resolution: "Authors: add a requirements.txt or environment.yml with fully pinned versions for every dependency, including torch."
cross_refs: []
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Appendix A.2 (Code link)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-llm-comparison-code
category: missing
topic: "result traceability"
title: "No code for the LLM comparison (Table 5 / Appendix A.3, OpenAI-O3 & Gemini-2.5-Pro)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  To evaluate the relative capabilities of our approach against state-of-the-art language models, we
  conducted experiments using OpenAI's O3 model and Gemini-2.5-Pro (both released in 2025).
  Following the methodology of [28], we adopted their One-Shot prompting technique to generate
  prompts for our planning problems.
claim: "Appendix A.3 / Table 5 report a GABAR-vs-LLM comparison, but no script in the repo generates the one-shot prompts, queries OpenAI-O3 or Gemini-2.5-Pro, or validates the returned plans (grep for openai/gemini/gpt/one_shot/prompt across all .py returns nothing outside baselines)."
concern: "The OpenAI-O3 and Gemini-2.5-Pro Coverage/PQR columns in Table 5 cannot be reproduced or audited; commercial-API nondeterminism makes this comparison unreproducible without the prompting/validation code and run logs."
resolution: "Authors: release the prompt-generation, model-query, and plan-validation scripts plus the raw model responses used for Table 5, and state model snapshot dates and decoding parameters."
cross_refs: []
paper_ref: "Appendix A.3, Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-trained-weights
category: missing
topic: "reproducibility / artefacts"
title: "No trained model weights or cached training-data shipped"
severity: low
confidence: high
status: finding
file: README.md
line_start: 105
line_end: 109
quote: |
  #### Testing a Pre-trained Model

  ```sh
  python main.py --method ltp --domain manyblocks_ipcc_big --all-problems --num-test-problems 10 --mode test --max-plan-length 500 --run-learned-model True --use-global-node True
  ```
claim: "The README documents a 'Testing a Pre-trained Model' workflow, but no .pt/.pth/.ckpt weights and no training-data .pkl caches exist in the repo (check_repo_completeness.py: trained_weight_files=0, training_data_pickles=0); only the cache/results/.gitkeep skeleton ships."
concern: "Reproducing any number requires retraining from scratch, which is itself blocked by the missing dataset (see missing-pddl-problems) and the undisclosed sweep (see undisclosed-hparam-sweep)."
resolution: "Authors: ship the per-domain best-model checkpoints (and the model_tracking.json) used to produce Table 2."
cross_refs: ["missing-pddl-problems", "undisclosed-hparam-sweep"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "README 'Testing a Pre-trained Model'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone runtime bug affecting a reported number was confirmed. (`main.py` and the data
pipeline contain abundant commented-out dead code and the model-name space is `ltp*` rather than
`gabar`, but the documented `--method ltp --ablation main` path is internally consistent.)

## difference

```yaml finding
id: undisclosed-hparam-sweep
category: difference
topic: "hyperparameter tuning"
title: "Runner sweeps n_heads and attention-dropout; paper describes a single fixed config"
severity: medium
confidence: high
status: finding
file: train_test_scripts/ltp_all_run.sh
line_start: 57
line_end: 61
quote: |
  METHOD="ltp"
  heads=(1 2 4 8)
  lrs=(0.0005)
  decays=(0.000)
  other_drops=(0)
  gnn_rounds=(9)
claim: "The canonical runner trains+tests GABAR across n_heads in {1,2,4,8} and (for ablation 'main', line 175) attention-dropout in {0.1,0.2}, logging each run to a separate file; analyse_domain_results.py then aggregates all logs into a CSV (check_selection_on_test.py confirms heads_swept='1 2 4 8', attn_drops_for_main='0.1 0.2')."
concern: "The paper's Training Procedure (§3.4) specifies a single fixed configuration (lr 0.0005, 9 GNN rounds, batch 16, hidden 64) and never mentions tuning the number of attention heads or attention dropout, so the reported numbers may be the best of an undisclosed sweep."
resolution: "Authors: state the n_heads and attention-dropout actually used per domain and how they were selected; if a sweep was run, report the selection criterion and the full grid."
cross_refs: ["model-selection-on-test-coverage"]
check_script: _audit_code/check_selection_on_test.py
paper_ref: "Section 3.4 'Training Procedure'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: model-selection-on-test-coverage
category: methodology
topic: "model selection / test-set leakage"
title: "Reported checkpoint chosen by best TEST-set coverage, not validation loss as paper claims"
severity: high
confidence: medium
status: finding
file: ploi/test_utils.py
line_start: 391
line_end: 395
quote: |
                    # Track best model for this planner type
                    if metrics.success_rate_with_monitor > best_results[planner_type]["success_rate"]:
                        best_results[planner_type]["success_rate"] = metrics.success_rate_with_monitor
                        best_results[planner_type]["model_type"] = model_type
                        best_results[planner_type]["epoch"] = epoch
claim: "main.py evaluates up to 6 checkpoints per run on the test set (num_models_to_test=2 for each of all_model_types=['validation','training','combined']) and log_model_metrics then reports, as 'Best Model', the checkpoint with the highest success_rate_with_monitor, which is the test-set coverage (check_selection_on_test.py confirms the active definition selects by 'success_rate_with_monitor (TEST coverage)')."
concern: "Selecting which checkpoint/run to report by its score on the test set is test-set leakage into model selection and contradicts the paper's stated criterion 'we select the model checkpoint that achieves the lowest loss on the validation set', biasing the reported Coverage/PQR upward."
resolution: "Authors: confirm whether the Table 2 numbers come from the validation-loss checkpoint or from this best-by-test-coverage selection; if the latter, re-report using only the validation-selected checkpoint (and validation-selected hyperparameters)."
cross_refs: ["undisclosed-hparam-sweep"]
check_script: _audit_code/check_selection_on_test.py
paper_ref: "Section 3.4: 'we select the model checkpoint that achieves the lowest loss on the validation set for evaluation'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Dataset (PDDL problems) not in repo + dead anon URL; no deps file; no LLM-comparison code; no weights. |
| bug         | 0          | -            | No reported-number-affecting runtime bug confirmed on the documented `--method ltp --ablation main` path. |
| difference  | 1          | medium       | Undisclosed n_heads / attention-dropout sweep; paper describes a single fixed config. |
| methodology | 1          | high         | Reported checkpoint selected by best test-set coverage, not the claimed validation-loss criterion. |

### Top take-aways (≤6, ranked by severity × confidence)
1. `[missing] missing-pddl-problems` (high/high) — train/val/test PDDL instances are not in the repo; the only source is an anonymized fork URL returning HTTP 401, so no reported number is reproducible.
2. `[methodology] model-selection-on-test-coverage` (high/medium) — the harness reports the checkpoint with the best **test** coverage, contradicting the paper's "lowest validation loss" claim; this can inflate Coverage/PQR.
3. `[difference] undisclosed-hparam-sweep` (medium/high) — the runner sweeps n_heads∈{1,2,4,8} and attn-dropout∈{0.1,0.2}, never disclosed in the paper's single-config training description.
4. `[missing] missing-llm-comparison-code` (medium/high) — no code/logs for the GABAR-vs-O3/Gemini comparison in Table 5.
5. `[missing] missing-dependency-spec` (medium/high) — no pinned environment file; torch and others unpinned.
6. `[missing] missing-trained-weights` (low/high) — no checkpoints shipped; "test pre-trained model" workflow is unrunnable as-is.

### Items that genuinely look fine
- PQR computation (`ploi/test_utils.py:186`, `compute_combined_metrics`) matches the paper's definition: FD plan length / learned plan length, averaged only over problems solved by **both** methods.
- Training loss (`ploi/traineval.py:507-508`) is cross-entropy on action scores plus cross-entropy on object scores, faithful to the paper's `L = L_action + L_objects`.
- Ablations described in the paper (GABAR-ACT, GABAR-CD, GABAR-RANK, GABAR-G, GABAR-ACT_CD) are all wired in the code (`ploi/ablations.py`, `--use-global-node`, `--ablation`), so the ablation study is reproducible given the dataset.
- The PQR reference (FD `fd-lama-first` plan lengths) is cached and present for all nine domains under `cache/results/planner_data/` and is recomputed live when a problem is missing (`run_planner_with_ltp_v2.py:793`).

### Open questions for the authors
- Which checkpoint-selection metric and which n_heads / attention-dropout actually produced the Table 2/4 numbers? (Bears on `model-selection-on-test-coverage` and `undisclosed-hparam-sweep`.)
- The paper states "Each test subset has 100 problems" (300/domain), but the runner's default `--num-test-problems` per domain (e.g. blocks=200, miconic=119, rovers=54) does not obviously map to 3×100; how are the easy/medium/hard subsets and their per-subset counts defined relative to `problems_per_division`?
