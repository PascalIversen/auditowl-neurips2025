# Audit — MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees (paper 1629)

## 1. Summary

MESS+ is a stochastic-optimization LLM router: for each request it either *explores*
(query all models in the zoo, train an online ModernBERT request-satisfaction
predictor on the binary correctness labels, return the largest model's output) or
*exploits* (predict per-model satisfaction probabilities and pick the model that
minimizes `V·E + Q·(α − ŝ)`, with a virtual queue `Q` enforcing the SLA target `α`).
The headline empirical claim (abstract, Conclusion) is "≈2× cost savings vs. existing
routing techniques" while satisfying per-benchmark SLA targets, evaluated on 8
LM-Eval-Harness benchmarks with a Llama 3 zoo (Table 2) and a Qwen zoo (Table 5).

The repo (`laminair/mess-plus`, single commit `9a55bc6`) contains:
- the algorithm core (`algorithm/exploration.py`, `exploitation.py`, `update_q.py`,
  `utils/mess_plus/bernoulli_sampler.py`);
- two drivers: `main.py` (real inference + data capture via LM-Eval/vLLM) and
  `simulator.py` (replays pre-captured per-model CSV outputs — per the README this is
  what produces the paper results "for fast iteration");
- the online predictor (`classifier/model.py`);
- the "educated guessing" baseline and post-processing of RouteLLM/RouterDC logs
  (`run_baselines.ipynb`); and
- W&B-log → LaTeX-table/figure notebooks (`evaluations/*.ipynb`,
  `evaluations/utils/{wandb_loader,tables}.py`, `plots/`).

What I did: read the paper (main text + appendices A–C), the README, all algorithm
and driver code, the classifier, the config files, and the evaluation/baseline
notebooks. I ran three read-only deterministic checks under `_audit_code/`
(`check_repo_artifacts.py`, `check_main_chosen_model_bug.py`) confirming (a) no
captured inference data / predictor checkpoints / W&B logs are shipped, (b) the
Llama configs declare a 4-model zoo while the paper describes 3, and (c) a
read-before-assignment bug for the logged "chosen model" in `main.py`. I could not
execute the pipeline end-to-end (requires multi-GPU vLLM, the gitignored captured
data, and the authors' private W&B projects), so quantitative reproduction of Table 2
numbers was not possible; this absence is itself the dominant finding.

## 2. Traceability table

The paper's quantitative artefacts are computed by the `simulator.py`/`main.py`
run loop, logged to W&B, then aggregated to tables/figures by
`evaluations/utils/tables.py` + the `experiments_*.ipynb` notebooks. The aggregation
code is present, but every value depends on (i) per-model inference CSVs and
(ii) W&B run histories that are **gitignored and absent** (`_audit_code/out/repo_artifacts.txt`).
"Repo location" below names the code that *would* compute the value; "Reproducible"
flags whether the inputs to that code are in the repo.

| Paper artefact | Repo location (computes value) | Reproducible from repo? | Status |
|---|---|---|---|
| Table 2 — MESS+ operating cost (MJ) per benchmark | `simulator.py:168,202` (sum/select energy) → W&B → `evaluations/utils/tables.py:18` (`mess_plus/energy`, aggfunc `sum`) | No — needs captured CSVs + W&B logs | MISSING inputs (see `missing-inference-data`) |
| Table 2 — MESS+ request satisfaction (%) | `simulator.py:166/206,231` (`avg_accuracy`) → `tables.py:18` | No | MISSING inputs |
| Table 2 — model call ratio (L70B/L8B/L1B) | `simulator.py:223-242` (online), `main.py:488-506` (capture) | No | MISSING inputs; `main.py` ratio is buggy (see `main-chosen-model-stale`) |
| Table 2 — single-model rows (1B/8B/70B) | `main.py` exploration capture; `run_baselines.ipynb` | No | MISSING inputs |
| Table 2 — Educated Guessing baseline | `run_baselines.ipynb` cells 1-3 (`calculate_probabilities`, `choose_llm_from_zoo`) | No (needs captured data) | code present, inputs MISSING |
| Table 2 — RouteLLM baseline | `run_baselines.ipynb` cell 4-5 (reads W&B `tum-i13/routellm-sweepv2` logs) | No — router code not in repo; logs private | MISSING (see `baselines-not-in-repo`) |
| Table 2 — RouterDC baseline | `run_baselines.ipynb` cell 6-7 (reads W&B `tum-i13/routerdc-sweep` logs) | No — router code not in repo; logs private | MISSING (see `baselines-not-in-repo`) |
| Abstract/Conclusion — "≈2× cost savings" | derived from Table 2 cost column | No | MISSING inputs |
| Fig. 2 — Winogrande Q/cost dynamics; SLA@step 740/803/994 | `simulator.py:228-234` logs → `experiments_llama3.ipynb` | No | MISSING inputs |
| Fig. 3 / Fig. C.1 — predictor training loss & exploration cost vs c | `classifier/model.py` loss → `simulator.py:154` → notebooks | No | MISSING inputs |
| Table 3 — routing/prediction overhead (J, %) | `main.py:692-708` (classifier energy window) | No | MISSING inputs |
| Table 4 — sparse-feedback (20% vs 100%) | `simulator.py:130-131` (`feedback_sparsity`) + `update_q.py` | No | MISSING inputs |
| Table 5 — Qwen larger-zoo results | `config/qwen2/*` + simulator | No | MISSING inputs |
| Tables 8/9 — V-variation results | `simulator.py` `v_values` sweep | No | MISSING inputs |
| Table 6 — predictor HP per benchmark | `classifier/sweep_config.yaml`, configs | partial (sweep config present; sweep logs absent) | partial |
| Theorems 1–2 (proofs) | Appendix A (math only) | N/A (no code expected) | N/A |

Net: no quantitative artefact in the paper can be reproduced from the repository as
shipped, because the captured per-model inference data, the trained predictor
checkpoints, and the W&B run logs that feed every table and figure are not included.

## 3. Findings

## missing

```yaml finding
id: missing-inference-data
category: missing
topic: "result traceability / data availability"
title: "Captured inference data, predictor checkpoints, and W&B logs absent; no table/figure reproducible"
severity: high
confidence: high
status: finding
file: code/laminair__mess-plus/.gitignore
line_start: 8
line_end: 18
quote: |
  data/
  datasets/
  legacy/figures/
  classifier/checkpoints/
  /evaluations/pivot_sample_data

  *.zip
  *.sqsh
  *.out
  *.csv
  *.ckpt
  *.log
claim: "simulator.py (the README's path for reproducing paper results) requires --dataset-path pointing to per-model inference CSVs (read_files_from_folder), and every table/figure is aggregated from W&B run histories via evaluations/utils/wandb_loader.py + tables.py; .gitignore excludes data/, *.csv, classifier/checkpoints/, *.ckpt and wandb/, and the repo ships none of them (0 CSVs, 0 checkpoints, 0 data files — _audit_code/out/repo_artifacts.txt)."
concern: "Not a single quantitative result (Table 2 costs/satisfaction/ratios, the headline 2x claim, Figs 2/3, Tables 3-5/8/9) can be reproduced from the repository because the inputs to the computing code are absent and the W&B projects are private."
resolution: "Authors: release the captured per-model inference CSVs (one per benchmark/zoo), the trained predictor checkpoints, and either the W&B export or a script that regenerates the tables from the CSVs end-to-end."
cross_refs: ["baselines-not-in-repo"]
check_script: _audit_code/check_repo_artifacts.py
paper_ref: "Table 2; README 'Running experiments'"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "baselines"
title: "RouteLLM / RouterDC routing code not in repo; only their private W&B logs are post-processed"
severity: medium
confidence: high
status: finding
file: code/laminair__mess-plus/run_baselines.ipynb
line_start: 1
line_end: 1
quote: |
  # Comparison with RouteLLM
  DATA_DIR = f"{PROJECT_ROOT_PATH}/data/routellm_raw"
  routellm_logs = download_log_data(
      entity="tum-i13",
      project_name="routellm-sweepv2",
      save_dir=DATA_DIR,
      batch_size=50
  )
claim: "The two competitor baselines (RouteLLM, RouterDC) are reproduced by downloading the authors' private W&B run histories (tum-i13/routellm-sweepv2, tum-i13/routerdc-sweep) that already contain the per-request model_choice; the router models themselves (RouteLLM BERT controller, RouterDC deberta-v3 contrastive router) are not trained or invoked anywhere in the repo (grep: no 'Controller', 'deberta', 'contrastive')."
concern: "The central comparison underpinning the '2x cheaper than existing routing' headline cannot be re-run; the baseline numbers are taken from inaccessible private logs rather than runnable baseline code."
resolution: "Authors: include the RouteLLM/RouterDC training+routing scripts (or configs) used, and ship the per-request model-choice logs, so the comparison can be reproduced independently."
cross_refs: ["missing-inference-data", "baseline-notebook-stale-paths"]
check_script: _audit_code/check_repo_artifacts.py
paper_ref: "Table 2; Appendix B 'RouteLLM' / 'RouterDC'"
tags: [reforms:5, reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: main-chosen-model-stale
category: bug
topic: "model-call accounting"
title: "main.py logs/records stale `chosen_model_id` on every exploitation step"
severity: medium
confidence: high
status: finding
file: code/laminair__mess-plus/main.py
line_start: 466
line_end: 472
quote: |
                        INFERENCE_TIME_LIST.append(step_time)
                        ENERGY_PER_MODEL[model_category_chosen].append(step_energy)
                        MODEL_CHOSEN_LIST.append(chosen_model_id)
                        ACCURACY_LIST.append(target_metric)

                        monitoring_dict[f"mess_plus/energy"] = step_energy
                        monitoring_dict[f"mess_plus/chosen_model"] = chosen_model_id
claim: "In the exploitation branch the selected model is `model_category_chosen` (returned by exploit()), but the code appends/logs `chosen_model_id`, a variable only ever assigned in the exploration branch (main.py:417 `chosen_model_id = len(label_cols) - 1`, i.e. the largest model). It is never set in the exploit branch (verified: zero assignments to chosen_model_id in lines 433-474, _audit_code/out/main_chosen_model_bug.txt)."
concern: "When main.py is used to produce model-call ratios, every exploitation step is recorded as the largest model (stale value) instead of the model actually chosen, biasing the model-call-ratio statistics toward L70B (and raising NameError if the first non-explore step is an exploit)."
resolution: "Use the index of `model_category_chosen` (e.g. `self.labels.index(model_category_chosen)`, as simulator.py:198 correctly does) when appending to MODEL_CHOSEN_LIST and logging mess_plus/chosen_model in the exploit branch."
cross_refs: []
check_script: _audit_code/check_main_chosen_model_bug.py
paper_ref: "Table 2 'Model Call Ratio' columns"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baseline-notebook-stale-paths
category: bug
topic: "baseline reproduction"
title: "run_baselines.ipynb reads config/online/*.yaml and CONFIG['algorithm'] that do not exist"
severity: low
confidence: high
status: finding
file: code/laminair__mess-plus/run_baselines.ipynb
line_start: 1
line_end: 1
quote: |
      config_path = Path(f"{PROJECT_ROOT_PATH}/config/online/{benchmark}.yaml")
      NUM_PRETRAINING_STEPS = 0

      with config_path.open("r") as f:
          CONFIG = yaml.safe_load(f)

      algorithm_config = CONFIG["algorithm"]
claim: "The RouteLLM/RouterDC baseline cells open `config/online/<benchmark>.yaml` and read `CONFIG['algorithm']`, but the repo has no `config/online/` (or `config/qwen2/online/`) directory and no config file with a top-level `algorithm:` key — configs use `empirical:`/`simulated:` (verified _audit_code/out/repo_artifacts.txt)."
concern: "These notebook cells cannot run as shipped (FileNotFoundError / KeyError), so the baseline post-processing is not executable even if the W&B logs were available."
resolution: "Update the notebook to the current config layout (config/<family>/<benchmark>.yaml, keys empirical/simulated) or add the referenced files."
cross_refs: ["baselines-not-in-repo"]
check_script: _audit_code/check_repo_artifacts.py
paper_ref: "Appendix B baselines"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: zoo-size-config-mismatch
category: difference
topic: "experimental setup"
title: "Llama configs declare a 4-model zoo (adds 3.2 3B); paper Table 2 reports a 3-model zoo"
severity: medium
confidence: high
status: finding
file: code/laminair__mess-plus/config/llama3/arc_challenge.yaml
line_start: 11
line_end: 16
quote: |
    meta-llama/Llama-3.2-3B-Instruct:
      category: "small"
      gpu_indices: [0]
      max_seq_len: 2048
      gpu_memory_utilization: 0.15
      quantization: null
claim: "Every config/llama3/*.yaml lists four models — Llama 3.2 1B (xsmall), 3.2 3B (small), 3.1 8B (medium), 3.3 70B (large) — whereas the paper's Appendix B 'LLM Zoo' and Table 2 describe a three-model zoo (1B/8B/70B), with model-call ratios reported only for L70B/L8B/L1B (verified across all 8 llama3 configs, _audit_code/out/repo_artifacts.txt)."
concern: "The shipped Llama configs do not match the zoo composition used for the reported Table 2 numbers, so a user running main.py with these configs would evaluate a different (4-model) system than the paper reports; both setups are individually valid routing problems."
resolution: "Authors: confirm whether Table 2 used 3 or 4 Llama models, and ship configs matching the reported zoo (or clarify that the 3B 'small' model was excluded for the main results)."
cross_refs: []
check_script: _audit_code/check_repo_artifacts.py
paper_ref: "Appendix B 'LLM Zoo'; Table 2 column headers (L70B/L8B/L1B)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: predictor-no-disjoint-eval
category: methodology
topic: "evaluation design"
title: "Request-satisfaction predictor never evaluated on data held out from its training stream"
severity: low
confidence: medium
status: question
file: code/laminair__mess-plus/simulator.py
line_start: 63
line_end: 65
quote: |
          for seed in self.algorithm_config["seed"]:
              set_all_seeds(seed)
              self.dataset = self.dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
claim: "The online predictor is trained on exploration requests and used to route subsequent exploitation requests from the same shuffled stream; reported request-satisfaction/cost are accumulated over that same stream, with no separate held-out test split. This is the intended online formulation (predict-then-act on each arriving request), so a held-out split is arguably not required."
concern: "Because evaluation and predictor training share one shuffled stream, the reported per-benchmark satisfaction is an in-stream online estimate, not an out-of-distribution generalization measure; whether this matters depends on interpreting the claim as online performance vs. generalization."
resolution: "Authors: clarify that results are online cumulative metrics over the request stream (not held-out generalization), and consider reporting predictor accuracy on requests excluded from its own training to bound any optimism."
cross_refs: []
paper_ref: "Section 2.3; Section 4.1"
tags: [leakage:L1.1, reforms:6]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 2          | high         | No captured data/checkpoints/W&B logs; baseline router code absent.    |
| bug         | 2          | medium       | `main.py` records stale chosen-model; baseline notebook paths broken.  |
| difference  | 1          | medium       | Llama configs are a 4-model zoo; paper reports 3 models.               |
| methodology | 1          | low          | Predictor evaluated on its own (online) stream; filed as a question.   |

### Top take-aways (≤6, by severity × confidence)
1. **[missing]** `missing-inference-data` — every reported number depends on captured inference CSVs / predictor checkpoints / W&B logs that are gitignored and absent; nothing in the paper is reproducible from the repo (high/high).
2. **[missing]** `baselines-not-in-repo` — RouteLLM/RouterDC routing code is not in the repo; only their private W&B model-choice logs are post-processed, so the "2× cheaper than existing routers" comparison is not reproducible (medium/high).
3. **[bug]** `main-chosen-model-stale` — `main.py` logs/records the largest model on every exploitation step (stale `chosen_model_id`), biasing model-call-ratio accounting (medium/high). (Note: the paper's main tables use `simulator.py`, which is correct.)
4. **[difference]** `zoo-size-config-mismatch` — shipped Llama configs declare 4 models (incl. 3.2 3B) vs. the 3-model zoo in the paper (medium/high).
5. **[bug]** `baseline-notebook-stale-paths` — baseline notebook references nonexistent `config/online/*.yaml` and `CONFIG['algorithm']` (low/high).
6. **[methodology/question]** `predictor-no-disjoint-eval` — online metrics over a single shared stream; flagged for clarification (low/medium).

### Items that genuinely look fine
- Algorithm core matches the paper: Bernoulli exploration prob `min(1, c·t^-1/4)` (`bernoulli_sampler.py:7-9` ↔ Alg. 1 line 3); virtual-queue update `Q=max(0,Q+α−s)` (`update_q.py:4` ↔ Eq. 2); exploitation objective `argmin V·E + Q·(α−ŝ)` (`exploitation.py:29-31` ↔ Eq. 3a).
- During exploration the energy cost sums **all** models' energy (`simulator.py:168`, `main.py:409`), correctly charging the full zoo-query cost as the paper's footnote 4 specifies.
- The predictor matches the appendix description (frozen ModernBERT, linear→LayerNorm→ReLU→dropout→linear, sigmoid, BCE; `classifier/model.py:62-98,175`); BERT is frozen so only the head trains via SGD (`model.py:479-493`), as stated.
- `lm_eval` is pinned to an exact git commit (`requirements.txt:107`), aiding environment reproducibility.
- Online predictor training (`online_learn=True`) does no early stopping / no validation-based checkpoint selection, so no test-set selection leakage in the exploitation path (`model.py:153,166,250`).

### Open questions for the authors
- Did Table 2's Llama results use 3 or 4 models? (`zoo-size-config-mismatch`)
- Can you release the captured per-benchmark inference CSVs and predictor checkpoints, plus a script that regenerates Table 2 from them without the private W&B projects? (`missing-inference-data`)
- Where is the RouteLLM/RouterDC routing code, and can the per-request model-choice logs be shared? (`baselines-not-in-repo`)
- Are the reported satisfaction/cost figures online cumulative metrics over the request stream rather than held-out generalization? (`predictor-no-disjoint-eval`)
