# Code-repository audit — RF-Agent (NeurIPS 2025, paper 5208)

## 1. Summary

The audited artefact is `code/deng-ai-lab__RF-Agent/`, the authors' official
implementation of *RF-Agent: Automated Reward Function Design via Language Agent
Tree Search*. RF-Agent uses an LLM (GPT-4o / GPT-4o-mini) inside an MCTS loop to
write Python dense reward functions for IsaacGym (7 tasks) and Bi-DexHands (10
tasks); each candidate reward is scored by training a PPO policy and reading the
task evaluation metric from TensorBoard. The repo bundles a modified
`isaacgymenvs/` (NVIDIA IsaacGym tasks + the 10 Bi-DexHands tasks, with assets)
and `rl_games/`; these plus the `RF_Agent/` driver are the author code. The two
other cloned repos (`RishiHazra__Revolve`, `isaac-sim__IsaacGymEnvs`) are
baseline/dependency context and were not audited for internals.

What I did: read the paper (PDF + text extraction), the four drivers
(`rfagent.py`, `test.py`, `test_human.py`, baseline scripts), the MCTS core
(`rf_agent_algo/rfagent_algo.py`, `utils.py`), the configs, the bundled task
registry, and the provided reward-function files. I ran one deterministic check
(`_audit_code/check_traceability_artifacts.py`) confirming reward-function
coverage, the evaluation metric reduction, the seed configuration, and the
absence of any aggregation/figure script or stored result logs. I did NOT run
the RL pipeline (requires NVIDIA IsaacGym Preview 4, GPUs, and paid OpenAI API
calls).

Headline reading: the repo is unusually complete for an LLM-agent paper — it
ships the proposed method, both LLM baselines, sparse/human baseline harnesses,
the bundled environments with assets, and the best reward functions for all 17
tasks for all three LLM methods. The evaluation protocol is applied
symmetrically across methods and baselines, so the *comparison* looks fair. The
findings below are about (a) reproduction completeness — no script computes the
reported aggregate numbers or figures and no result logs are shipped, so every
reported value requires expensive nondeterministic retraining; (b) a minor
paper↔code seed-disjointness discrepancy; and (c) unpinned/partly-wrong
dependency spec.

## 2. Result-traceability table

"Computed value" = whether a script in the repo *computes* the number (not just
trains). All reported metrics require running `train_with_seed.py` (IsaacGym +
GPU + OpenAI key); none are reproducible from shipped artefacts alone.

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Table 1: per-task IsaacGym scores (Ours, 4o-mini & 4o) | `RF_Agent/test.py` + `reward_functions/isaac/RFAgent/*.py` | re-trains & reads `max(consecutive_successes)` per seed; mean/std over 5 seeds | not checkable (needs GPU+sim) | Re-derivable only |
| Table 1: Eureka / Revolve rows | `RF_Agent/test.py` + `reward_functions/isaac/{Eureka,Revolve}/*.py` | same harness, same metric | not checkable | Re-derivable only |
| Table 1: Sparse rows | `RF_Agent/envs/isaac_sparse/*` + `test.py` | sparse-reward env variants present | not checkable | Re-derivable only |
| Table 1: Human rows | `RF_Agent/test_human.py` (uses benchmark `train.py` GT reward) | re-trains GT reward | not checkable | Re-derivable only |
| Table 1: **"Avg norm score"** column (per-task `(Method−Sparse)/(Human−Sparse)`, averaged) | (none) | NO — no normalization/aggregation script exists | — | MISSING (aggregation) |
| Fig. 3: Bi-DexHands Expert-Easy/Hard success rates | `reward_functions/bidex/*/*.py` + `test.py` | re-trains per task; no bar-chart/aggregation script | not checkable | Re-derivable; no plot/agg code |
| Fig. 4: success-rate vs training-step curves | (none) | NO producing/plotting script for these curves | — | MISSING (figure) |
| Fig. 5: avg-max score vs sampling count | (none) | NO producing/plotting script | — | MISSING (figure) |
| Fig. 6: ablations (search method / action types / reasoning) | (none) | NO ablation harness or aggregation script in repo | — | MISSING (ablation code) |
| Tables 4–5: token / cost numbers (Ant, Humanoid) | (none) | NO token-accounting/export script | — | MISSING |
| Appendix F.2/F.3 per-task max & optimization examples | `reward_functions/*` (functions only) | partial; no result tables | — | Re-derivable only |

Deterministic backing: `_audit_code/out/traceability_artifacts.csv`
(normalization-script hits = `[]`; stored result logs = `[]`; the only `savefig`
calls are the baselines' own per-search summary plots).

## 3. Findings

## missing

```yaml finding
id: no-aggregation-or-figure-scripts
category: missing
topic: "result traceability"
title: "No script computes Table 1 Avg-norm column, Figs 3-6, or cost tables; no result logs shipped"
severity: medium
confidence: high
status: finding
file: RF_Agent/test.py
line_start: 60
line_end: 77
quote: |
    cur_reward_code_final_successes = []
    for i, rl_run in enumerate(cur_eval_runs):
        rl_run.communicate()
        seed = i
        rl_filepath = f"reward_code_cur_eval{seed}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read()
        lines = stdout_str.split('\n')
        for k, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break
        tensorboard_logdir = line.split(':')[-1].strip()
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        cur_reward_code_final_successes.append(max_success)

    logging.info(f"Current Reward Code Final Success Mean: {np.mean(cur_reward_code_final_successes)}, "
                 f"Std: {np.std(cur_reward_code_final_successes)}, Raw: {cur_reward_code_final_successes}")
claim: "test.py logs per-task mean/std of max success over 5 seeds, but the repo contains no script that computes the human-normalized 'Avg norm score' (Table 1), aggregates Bi-DexHands groups (Fig 3), produces the training-curve / sampling-count figures (Figs 4,5), runs/aggregates the ablations (Fig 6), or accounts tokens/cost (Tables 4,5); no result logs (json/csv/npy) are shipped."
concern: "Every headline aggregate and every figure must be regenerated by expensive, nondeterministic retraining under closed-source IsaacGym; the reported numbers cannot be re-derived or cross-checked from any artefact in the repo."
resolution: "Provide the normalization/aggregation and plotting scripts (and/or the raw per-seed result logs) used to produce Table 1's Avg-norm column and Figs 3-6 and Tables 4-5."
cross_refs: []
check_script: _audit_code/check_traceability_artifacts.py
paper_ref: "Table 1 (Avg norm score), Figs 3-6, Tables 4-5"
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-ablation-harness
category: missing
topic: "ablations"
title: "Ablation variants (DFS/BFS/Greedy search, per-action removal, no self-verify/thought-align) not in repo"
severity: medium
confidence: high
status: finding
file: RF_Agent/rf_agent_algo/rfagent_algo.py
line_start: 122
line_end: 124
quote: |
        # param about "actions"
        self.action_list = ['0i', '1m', '2m', '3e', '4r', '5d']
        self.action_weight_num = [0, 2, 2, 2, 1, 1] # weight_num大于1可以用bs
claim: "The MCTS core hardcodes the full action set and UCT selection; there are no switches/configs or alternate drivers to run the DFS/BFS/Greedy search ablations, the per-action-type removal ablations, or the no-self-verify / no-thought-align ablations reported in Fig 6."
concern: "Fig 6's ablation conclusions ('each action holds its value', 'reasoning paradigm helps') cannot be reproduced because the ablated configurations are absent from the code."
resolution: "Provide the ablation drivers/flags used for Fig 6 (search-method swap, action-subset, and reasoning-component removal)."
cross_refs: ["no-aggregation-or-figure-scripts"]
paper_ref: "Fig 6 (Ablation Studies, §5.5)"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unpinned-and-incorrect-deps
category: missing
topic: "dependencies / environment"
title: "setup.py pins numpy==1.20.0 (incompatible with torch<=2.0) and omits the IsaacGym/Bi-DexHands runtime"
severity: low
confidence: high
status: finding
file: setup.py
line_start: 4
line_end: 21
quote: |
  INSTALL_REQUIRES = [
      "charset-normalizer",
      "matplotlib",
      "openai==0.28.0",
      'torch<=2.0.0',
      'numpy==1.20.0',
      'ray>=1.1.0',
      'tensorboard>=1.14.0',
      'tensorboardX>=1.6',
      'setproctitle',
      'psutil',
      'pyyaml',
      "gym==0.23.1",
      "omegaconf",
      "termcolor",
      "hydra-core>=1.1",
      "pyvirtualdisplay",
  ]
claim: "The dependency list pins numpy==1.20.0 and torch<=2.0.0 (numpy 1.20 predates and is incompatible with torch 2.0's expected numpy ABI), lists gpustat usage in code but not as a dependency, and does not pin the OpenAI model snapshots; the closed-source IsaacGym Preview 4 runtime is required but documented only in the README, not specifiable via pip."
concern: "The environment cannot be rebuilt as specified; numpy 1.20 with torch 2.0 will not import cleanly, and `misc.py`/`utils.py` shell out to `gpustat` which is unlisted."
resolution: "Pin a mutually compatible numpy/torch pair, add `gpustat`, and pin the GPT-4o/4o-mini snapshots actually used (paper cites GPT-4o-mini-0718 / GPT-4o-0806)."
cross_refs: []
check_script: _audit_code/check_traceability_artifacts.py
paper_ref: "§5.3 (model snapshots), README Installation"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No technical bugs found that would prevent the intended pipeline from running.
The `gpustat`-based GPU selection, the parallel simulation loop, the OpenAI
0.28 `ChatCompletion` calls, and the TensorBoard parsing are internally
consistent. (The `numpy`/`gpustat` packaging problems are routed to `missing`
above, since they are environment-specification gaps rather than logic defects.)

## difference

```yaml finding
id: eval-seeds-not-disjoint-from-search
category: difference
topic: "evaluation protocol / seeding"
title: "Paper says eval used 5 seeds different from search, but eval seeds 0-4 include the search seed 0"
severity: low
confidence: high
status: finding
file: RF_Agent/test.py
line_start: 41
line_end: 54
quote: |
    for i in range(cfg.num_eval):
        set_freest_gpu()
        env = os.environ.copy()
        seed = i
        rl_filepath = f"reward_code_cur_eval{seed}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train_with_seed.py',
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}',
                                        f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}',
                                        f'capture_video={cfg.capture_video}',
                                        'force_render=False', f'seed={seed}',
claim: "Final evaluation trains with seeds 0..num_eval-1 (i.e. 0-4). RF-Agent's search trains every candidate at cfg.train_seed=0 (config_rf_agent.yaml:30; passed via rfagent.run(train_seed=cfg.train_seed) and used as the simulation seed), so evaluation seed 0 coincides with the search seed."
concern: "The appendix states evaluation used '5 seeds different from those during the search'; in code one of the five evaluation seeds (0) is the search seed, giving a mild optimistic bias on absolute scores (applied symmetrically across methods, so it does not change the method-vs-baseline ranking)."
resolution: "Either offset evaluation seeds (e.g. range(100,105)) to make them disjoint from the search seed, or correct the appendix claim of disjoint seeds."
cross_refs: []
check_script: _audit_code/check_traceability_artifacts.py
paper_ref: "Appendix C, 'Other evaluation details' (5 seeds different from those during the search)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: best-checkpoint-selection-by-eval-metric
category: methodology
topic: "evaluation / checkpoint selection"
title: "Reported score is the max over training checkpoints of the eval metric itself (best-checkpoint selection)"
severity: low
confidence: high
status: finding
file: RF_Agent/test.py
line_start: 73
line_end: 73
quote: |
        max_success = max(tensorboard_logs['consecutive_successes'])
claim: "For each evaluation run, the reported per-task score is the maximum value of the task evaluation metric ('consecutive_successes', which in the IsaacGym task code is overloaded to be the paper's metric F) over all logged checkpoints during training, then averaged over 5 seeds."
concern: "Selecting the best checkpoint by the same metric being reported can optimistically bias absolute scores versus reporting the final-checkpoint or held-out value; it is, however, applied identically to RF-Agent, Eureka, Revolve, Sparse and Human, matching the Eureka protocol the paper follows, so it does not bias the cross-method comparison."
resolution: "Confirm this 'max over checkpoints' is the intended 'average maximum evaluation score at each policy checkpoint' (§5.3) and that it is applied identically to all methods (it is, per baseline_eureka.py:394 and baseline_revolve.py:516)."
cross_refs: ["eval-seeds-not-disjoint-from-search"]
paper_ref: "§5.3 Training Setup; Table 1; Fig 3"
tags: [reforms:4, lones:stage-4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | medium       | No aggregation/figure/ablation scripts or result logs; deps unpinned/wrong. |
| bug         | 0          | -            | No blocking logic defects found in the audited drivers/core. |
| difference  | 1          | low          | Eval seeds 0-4 overlap the search seed 0, contra "disjoint seeds" claim. |
| methodology | 1          | low          | Best-checkpoint selection by the eval metric, but applied symmetrically. |

### Top take-aways (≤6, ranked by severity × confidence)

1. **[missing] No script computes the reported aggregates or figures, and no
   result logs are shipped** (`no-aggregation-or-figure-scripts`): Table 1's
   Avg-norm column, Figs 3-6, and the cost tables can only be regenerated by
   expensive nondeterministic retraining under closed IsaacGym — none are
   re-derivable from repo artefacts. (medium / high)
2. **[missing] Ablation configurations (Fig 6) are absent from the code**
   (`no-ablation-harness`): the DFS/BFS/Greedy search swaps, per-action removal,
   and no-self-verify / no-thought-align variants have no driver or flag.
   (medium / high)
3. **[missing] Dependency spec is unbuildable** (`unpinned-and-incorrect-deps`):
   numpy 1.20.0 vs torch 2.0 incompatibility, unlisted `gpustat`, unpinned model
   snapshots. (low / high)
4. **[difference] Evaluation seeds are not disjoint from the search seed**
   (`eval-seeds-not-disjoint-from-search`): seed 0 is used both in search and in
   the 5-seed evaluation, contradicting the appendix. (low / high)
5. **[methodology] Reported score is the per-run max over checkpoints of the
   eval metric** (`best-checkpoint-selection-by-eval-metric`): mild optimistic
   bias, but symmetric across all methods/baselines. (low / high)

### Items that genuinely look fine

- **Reward-function coverage**: all 7 IsaacGym tasks × 2 models (14 files) and
  all 10 Bi-DexHands tasks ship for RF-Agent, and identically for the Eureka and
  Revolve baselines (`reward_functions/`), so a reviewer with IsaacGym can re-run
  every per-task number.
- **Symmetric evaluation harness**: `test.py`, `baseline_eureka.py:394`, and
  `baseline_revolve.py:516` use the same metric (`max consecutive_successes`),
  the same 5 seeds (0-4), and the same `test_max_iterations`, so the
  method-vs-baseline comparison is apples-to-apples.
- **Bundled environments**: the modified `isaacgymenvs/` includes and registers
  the 10 Bi-DexHands tasks plus assets (`tasks/__init__.py`, `cfg/task/*.yaml`),
  so the complex-task experiments are not gated behind an external download in
  this version (despite the supplemental `RF-Agent Instruction.md` note).
- **Closed-dependency disclosure**: dependence on NVIDIA IsaacGym Preview 4 is a
  legitimate non-self-contained reason (free registration-gated simulator, not
  the paper's contribution) and is documented in the README.

### Open questions for the authors

- Were Table 1 / Fig 3-6 / Table 4-5 generated by off-repo scripts? If so,
  please add them (or the raw per-seed logs) — currently nothing in the repo
  computes those aggregates.
- The appendix states evaluation used "5 seeds different from those during the
  search," but code uses seeds 0-4 for both. Which is correct?
- Please confirm GPT-4o-mini-0718 / GPT-4o-0806 are the exact snapshots; the
  configs and `setup.py` only say `gpt-4o-mini` and pin `openai==0.28.0`.
