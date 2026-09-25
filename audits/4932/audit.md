# Code-Repository Audit — CenRA (paper 4932)

## 1. Summary

The repository `mahaozhe__CenRA` (commit `7353468`, 2025-03-15) implements the
**Centralized Reward Agent (CenRA)** multi-task RL framework: a centralized
reward agent (`CenRA_dis` / `CenRA_con` in `CenRA/Algorithms.py`) that learns a
shaped "knowledge reward" shared across several distributed policy agents
(`DQNAgent` / `SACAgent` in `CenRA/Agents.py`), plus custom environments under
`RLEnvs/`. The repo provides three training entry points: `run-2dmaze.py`,
`run-3dpickup.py`, `run-mujococar.py`.

I read all Python sources (`CenRA/`, `RLEnvs/`, the three `run-*.py` scripts),
the README, and `requirements.txt`, and cross-checked them against the paper
(Algorithm 1, Section 4, Section 5, Tables 1–3, Figures 3/4/7, Appendix B). I
wrote one verification script, `_audit_code/check_mujococar_arg.py`, which uses
AST to confirm that `run-mujococar.py` references a CLI argument
(`ra_buffer_size`) that is never declared.

Headline conclusions about scope: the repo is a **training-only** artefact for
three of the four reported domains. It contains **no Meta-World code**
(the headline ML10/ML50 benchmark), **no baseline implementations** (the 9
baselines compared against in Tables 1–2), and **no evaluation script** that
computes the "tested over 100 episodes" returns reported in every results
table. The information-synchronization ablation (Table 3) cannot be reproduced
because the balance factor α is hard-coded to 0.5.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|----------------|---------------|----------------|---------------|--------|
| Table 1, ML10-sparse / ML50-sparse columns (all algos) | (none — no Meta-World env, no training/eval code) | — | — | MISSING |
| Table 1, 2DMaze/3DPickup/MujocoCar — CenRA row | `run-2dmaze.py`, `run-3dpickup.py`, `run-mujococar.py` (training only; no eval over 100 episodes) | — | — | PARTIAL (training only; MujocoCar crashes — see `mujococar-undefined-ra-buffer-size`) |
| Table 1, all baseline rows (DQN/SAC, ReLara, TD-MPC2, CMTA, PiCor, MCAL, PaCo, SC, SoftModule) | (none) | — | — | MISSING (no baseline code) |
| Table 2, knowledge-transfer rows (CenRA w/ & w/o learning, ReLara, DQN/SAC) | (commented `# for the new task` env lines only; no transfer/eval driver) | — | — | MISSING |
| Table 3, α ablation (α=0.25/0.5/0.75, w/o wsim, w/o wper, w/o both) | `CenRA/Algorithms.py:189,395` (α hard-coded 0.5) | — | — | MISSING (no α control / no ablation harness) |
| Figure 3, learning curves (2DMaze/3DPickup/MujocoCar) | training scripts write TensorBoard scalars (`Agents.py:165,337`); no plotting/aggregation script | — | — | PARTIAL (curves logged, no figure code) |
| Figure 4a/4b, transfer curves + reward-direction viz | (none) | — | — | MISSING |
| Figure 7 / Appendix C, knowledge-reward direction maps | (none) | — | — | MISSING |

## 3. Findings

## missing

```yaml finding
id: metaworld-benchmark-absent
category: missing
topic: "result traceability / repository completeness"
title: "Meta-World ML10/ML50 headline benchmark has no code in the repo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 32
line_end: 36
quote: |
  All available environments with corresponding `<Environment>` are listed below:
  * *2DMaze* environment: `2dmaze`, [running script](./run-2dmaze.py).
  * *3DPickup* environment: `3dpickup`, [running script](./run-3dpickup.py).
  * *MujocoCar* environment: `mujococar`, [running script](./run-mujococar.py).
claim: "The repo provides run scripts only for 2DMaze, 3DPickup, and MujocoCar; there is no Meta-World environment, run script, or task configuration anywhere in the repository (grep for 'metaworld/ML10/ML50' returns no source hits)."
concern: "Meta-World ML10-sparse and ML50-sparse are the headline benchmark (leading columns of Tables 1 and 2 and the abstract's claim of validation on 'the representative Meta-World benchmark'), yet none of the code, environments, or task splits needed to produce those numbers is present, so those results cannot be reproduced from this repo."
resolution: "Authors: please add the Meta-World sparse-reward environment wrappers, the ML10/ML50 task lists (10/45 train + 5 test tasks), and the run scripts used to produce the Meta-World columns of Tables 1–2."
cross_refs: ["baselines-not-implemented", "eval-script-missing"]
paper_ref: "Tables 1–2 (ML10-sparse / ML50-sparse columns); Abstract"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baselines-not-implemented
category: missing
topic: "baselines"
title: "None of the 9 compared baselines are implemented in the repo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 57
line_end: 59
quote: |
  ## Comparative Evaluation

  The comparison of CenRA with several baselines, including the backbone algorithms [DQN](https://www.nature.com/articles/nature14236) (Mnih et al. 2015) for discrete control and [SAC](https://proceedings.mlr.press/v80/haarnoja18b) (Haarnojaet al. 2018) for continuous control, [ReLara](https://proceedings.mlr.press/v235/ma24l.html) (Ma et al. 2024), [PiCor](https://ojs.aaai.org/index.php/AAAI/article/view/25825) (Bai et al. 2023) and [MCAL](https://openreview.net/forum?id=rJvY_5OzoI) (Mysore et al. 2022).
claim: "The repo contains only the CenRA framework (CenRA_dis/CenRA_con plus DQNAgent/SACAgent as its own backbones); it does not contain runnable implementations of ReLara, TD-MPC2, CMTA, PiCor, MCAL, PaCo, SC, or SoftModule, which Tables 1–2 report numbers for."
concern: "Every baseline column in Tables 1 and 2 (including the central claim 'CenRA consistently outperforms all baselines') is unverifiable from this repo because no baseline training/evaluation code is provided, and the README only links to external papers rather than the code used."
resolution: "Authors: please provide the baseline implementations (or exact forks/commit hashes of the CleanRL/official codebases) and the scripts used to run them under the identical sparse-reward tasks."
cross_refs: ["metaworld-benchmark-absent", "eval-script-missing"]
paper_ref: "Tables 1–2 (all baseline rows); Section 5.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eval-script-missing
category: missing
topic: "result traceability / evaluation"
title: "No evaluation script computes the '100-episode' returns in Tables 1–3"
severity: high
confidence: high
status: finding
file: code/mahaozhe__CenRA/CenRA/Algorithms.py
line_start: 252
line_end: 258
quote: |
  def save(self, indicator="best"):
        torch.save(self.actor.state_dict(),
                   os.path.join(self.save_folder, f"ra-actor-{self.exp_name}-{indicator}-{self.seed}.pth"))
        torch.save(self.qf_1.state_dict(),
                   os.path.join(self.save_folder, f"ra-qf_1-{self.exp_name}-{indicator}-{self.seed}.pth"))
        torch.save(self.qf_2.state_dict(),
                   os.path.join(self.save_folder, f"ra-qf_2-{self.exp_name}-{indicator}-{self.seed}.pth"))
claim: "The training drivers only `learn(...)` then `save(...)` model weights; there is no function or script that loads a trained agent and evaluates it over 100 episodes to produce the mean ± standard-error returns reported in Tables 1, 2, and 3."
concern: "Tables 1–3 report returns 'tested over 100 episodes' with standard errors over '10 different seeds', but no code performs this held-out evaluation or aggregates seeds/standard errors, so the reported numbers cannot be regenerated from the repo."
resolution: "Authors: please add the evaluation harness (deterministic 100-episode rollout, per-seed aggregation, standard-error computation) that produced Tables 1–3."
cross_refs: ["metaworld-benchmark-absent", "baselines-not-implemented"]
paper_ref: "Tables 1–3 captions ('tested over 100 episodes ... mean ± standard error')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: alpha-ablation-not-controllable
category: missing
topic: "ablations / hyperparameters"
title: "Table 3 sampling-weight ablation (α and weight on/off) not implemented"
severity: medium
confidence: high
status: finding
file: code/mahaozhe__CenRA/CenRA/Algorithms.py
line_start: 182
line_end: 191
quote: |
        # calculate the return weights: weights = softmax(1 / tail_returns)
        return_weights = np.exp(1 / tail_returns) / np.sum(np.exp(1 / tail_returns))
        # calculate the feature weights: weights = softmax(1 / sim_i), sim_i = c dot f_i / sqrt(dim(c)), c = mean(f)
        center_feature = np.mean(tail_features, axis=0)
        feature_sim = center_feature @ tail_features.T / math.sqrt(len(center_feature))
        feature_weights = np.exp(1 / feature_sim) / np.sum(np.exp(1 / feature_sim))

        weights = 0.5 * (return_weights + feature_weights)

        pa_batch_size = (self.batch_size * weights).astype(int)
claim: "The balance factor between the performance weight (return_weights) and the similarity weight (feature_weights) is hard-coded to 0.5; there is no α argument, no code path for α=0.25/0.75, and no way to disable either weight (w/o wsim => α=0; w/o wper => α=1; w/o both => uniform)."
concern: "Table 3 reports an ablation over α∈{0,0.25,0.5,0.75,1} and over removing each sampling weight, but the only configuration the code can run is α=0.5 with both weights on, so the ablation rows cannot be reproduced from the repo."
resolution: "Authors: please expose the α balance factor (and switches for w/o wsim / w/o wper / w/o both) as configurable parameters, matching the variants in Table 3."
cross_refs: []
paper_ref: "Table 3; Section 5.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: knowledge-transfer-driver-missing
category: missing
topic: "result traceability / knowledge transfer"
title: "No driver for the new-task transfer experiments (Table 2, Fig. 4)"
severity: medium
confidence: high
status: finding
file: code/mahaozhe__CenRA/run-2dmaze.py
line_start: 69
line_end: 70
quote: |
    env_ids = ["MiniGrid-Type1", "MiniGrid-Type2", "MiniGrid-Type3", "MiniGrid-Type4"]
    # env_ids = ["MiniGrid-Type5"]  # for the new task
claim: "The 'new task' transfer setting exists only as a commented-out single-task env line; there is no script that loads a pre-trained CRA, initialises a fresh policy agent on the held-out task, and runs the 'CenRA w/ learning' vs 'CenRA w/o learning' (frozen-CRA) conditions reported in Table 2 / Figure 4a."
concern: "Section 5.2 / Table 2 / Figure 4 are central to the paper's transferability claim, but reproducing them requires loading a saved CRA into a new run with the CRA optionally frozen — a code path that does not exist (the training algorithms always optimise the CRA and never load it from disk)."
resolution: "Authors: please add the transfer-experiment script that loads the trained CRA checkpoint and runs the frozen-CRA ('w/o learning') and continued-learning ('w/ learning') conditions on the held-out tasks."
cross_refs: ["eval-script-missing"]
paper_ref: "Table 2; Figure 4; Section 5.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: mujococar-undefined-ra-buffer-size
category: bug
topic: "runnability / CLI arguments"
title: "run-mujococar.py crashes: references undefined args.ra_buffer_size"
severity: high
confidence: high
status: finding
file: code/mahaozhe__CenRA/run-mujococar.py
line_start: 106
line_end: 111
quote: |
    agent = CenRA_con(policy_agents=policy_agents, sample_env=envs[0], actor_class=RAActorVectorObs,
                      critic_class=RAQNetVectorObs, buffer_size=args.ra_buffer_size * len(envs),
                      batch_size=args.ra_batch_size, policy_lr=args.ra_actor_lr, q_lr=args.ra_critic_lr,
                      alpha_lr=args.ra_alpha_lr, policy_frequency=args.ra_policy_frequency,
                      alpha=args.ra_alpha, alpha_autotune=args.ra_alpha_autotune,
                      suggested_reward_scale=args.suggested_reward_scale, lamb=args.lamb)
claim: "Line 107 reads args.ra_buffer_size, but the argument parser in run-mujococar.py only declares --pa-buffer-size (no --ra-buffer-size). At runtime this raises AttributeError: 'Namespace' object has no attribute 'ra_buffer_size'. Verified by AST in _audit_code/check_mujococar_arg.py (ra_buffer_size is the only args.* access with no matching add_argument, and only in run-mujococar.py)."
concern: "The MujocoCar experiment (a full column of Tables 1–3 and a Figure 3 panel) cannot be launched as shipped; the entry point fails before training begins."
resolution: "Authors: add a --ra-buffer-size argument (or change line 107 to args.pa_buffer_size) and confirm which buffer size was used for the reported MujocoCar runs."
cross_refs: []
check_script: _audit_code/check_mujococar_arg.py
paper_ref: "Tables 1–3 MujocoCar column; Figure 3 (MujocoCar)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: cra-critic-objective-vs-paper
category: difference
topic: "evaluation consistency (paper vs code)"
title: "CRA optimised with twin-Q SAC objective, not the single-V actor-critic of Eq. 1–2"
severity: low
confidence: medium
status: finding
file: code/mahaozhe__CenRA/CenRA/Algorithms.py
line_start: 205
line_end: 217
quote: |
        with torch.no_grad():
            next_shaped_reward, next_state_log_pi, _ = self.actor.get_action(data.next_observations, data.next_actions)
            qf_1_next_target = self.qf_1_target(data.next_observations, data.next_actions, next_shaped_reward)
            qf_2_next_target = self.qf_2_target(data.next_observations, data.next_actions, next_shaped_reward)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(
                -1)

        qf_1_a_values = self.qf_1(data.observations, data.actions, data.shaped_rewards).view(-1)
        qf_2_a_values = self.qf_2(data.observations, data.actions, data.shaped_rewards).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
claim: "The CRA is trained as a SAC agent with two soft Q-functions (twin critics, entropy term -alpha*log_pi, min over qf_1/qf_2, soft target updates, action = shaped reward). The paper's Eq. (1)–(2) instead describe a single state-value critic V_rwd trained on the TD error delta_t = r_env + gamma*V(s') - V(s) and an actor updated by log pi(r_knw|s)*delta_t (a one-step actor-critic, no twin Q, no entropy term)."
concern: "The CRA learning rule actually implemented (twin-Q soft actor-critic) differs from the actor-critic equations stated in the methodology; both are valid RL objectives, but a reader reproducing Eq. (1)–(2) literally would build a different algorithm than the one that produced the results."
resolution: "Authors: please reconcile Eq. (1)–(2) with the SAC implementation (e.g., update the equations to the actual twin-Q soft actor-critic, or clarify that the CRA uses SAC with the listed entropy/temperature terms)."
cross_refs: []
paper_ref: "Section 4.1.2, Eq. (1)–(2)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | Meta-World benchmark, all baselines, eval harness, α-ablation, and transfer driver absent. |
| bug         | 1          | high         | run-mujococar.py crashes on undefined args.ra_buffer_size. |
| difference  | 1          | low          | CRA implemented as twin-Q SAC; paper states single-V actor-critic (Eq. 1–2). |
| methodology | 0          | -            | No invalid-procedure finding in the present (training-only) code; evaluation code is absent rather than wrong. |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing]** Meta-World ML10/ML50 — the headline benchmark — has no environment, task split, or run script in the repo (`metaworld-benchmark-absent`).
2. **[missing]** None of the 9 baselines (ReLara, TD-MPC2, CMTA, PiCor, MCAL, PaCo, SC, SoftModule, DQN/SAC) are implemented; every baseline column of Tables 1–2 is unverifiable (`baselines-not-implemented`).
3. **[missing]** No evaluation script computes the "100-episode" mean ± standard-error returns reported in all results tables (`eval-script-missing`).
4. **[bug]** `run-mujococar.py` references undefined `args.ra_buffer_size` and crashes at startup, so the MujocoCar runs cannot be launched as shipped (`mujococar-undefined-ra-buffer-size`).
5. **[missing]** Table 3's sampling-weight ablation is not reproducible: α is hard-coded to 0.5 with no switch for the other α values or for disabling either weight (`alpha-ablation-not-controllable`).
6. **[missing]** The new-task transfer experiments (Table 2 / Figure 4) have no driver — frozen-CRA / continued-learning conditions and CRA checkpoint loading are absent (`knowledge-transfer-driver-missing`).

### Items that genuinely look fine
- The sampling weights `softmax(1/Rtail)` and `softmax(1/sim)` in `Algorithms.py:183-187` faithfully implement the wper / wsim formulas in Section 4.2.
- Latent task features are taken post-ReLU (`Networks.py:53-57,92-96` return the output of a CNN whose last conv block is ReLU-then-Flatten), matching the paper's stated non-negativity argument for the centroid c.
- The DQN/SAC policy agents add the knowledge reward as `r_env + lamb * r_knw` (`Agents.py:183,358`), matching Eq. (3) with λ=0.5 (Appendix B).
- Random seeds for Python/NumPy/PyTorch and cuDNN determinism are set in every agent constructor (`Agents.py:61-64`, `Algorithms.py:62-65`).
- `requirements.txt` pins the core scientific dependencies (numpy, gymnasium, minigrid, miniworld, mujoco, stable-baselines3) to specific versions.

### Open questions for the authors
- Was the MujocoCar `ra_buffer_size` intended to equal `pa_buffer_size`, and what buffer size produced the reported MujocoCar numbers? (`mujococar-undefined-ra-buffer-size`)
- Should Eq. (1)–(2) be read as a SAC objective (twin Q, entropy temperature), or was a single-V actor-critic actually used for some runs? (`cra-critic-objective-vs-paper`)
- Where are the Meta-World, baseline, evaluation, ablation, and transfer scripts — are they intended for a later public release beyond this supplementary snapshot?
