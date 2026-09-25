# Code Audit — A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning (paper 4140)

## 1. Summary

The repository (`iamlilAJ__Pre-Strategy-Intervention`, a fork of [JaxMARL](https://github.com/FLAIROx/JaxMARL)) contains the JAX implementation of Pre-Strategy Intervention (PSI). It provides training entrypoints for the algorithms in the paper (IQL/VDN/QMIX for MPE; IPPO/MAPPO/PQN-IQL/PQN-VDN for Hanabi), Hydra configs for the three reported conditions (PSI "our method", `base_marl_*` Base MARL, `intrinsic_reward_*` ablation), and the modified environments (`AugmentedMPE`, modified Hanabi) that emit the intrinsic ("additional desired outcome") rewards. Dependencies are pinned in `pyproject.toml`; a README gives exact reproduction commands.

The paper's quantitative results are training curves (Figure 5, and Figures 8/9 in the appendix): extrinsic return (primary task) and intrinsic return (additional-outcome reachability), "from 5 random seeds … with 95% confidence intervals." There are no point-estimate tables in the main results; the only tables are hyperparameter tables (Tables 2–7).

I read the training scripts (`baselines/QLearning/iql_pre.py` and siblings), the reward/observation wrappers (`jaxmarl/wrappers/baselines.py`), the augmented MPE and Hanabi reward implementations (`jaxmarl/environments/mpe/augmented_mpe.py`, `jaxmarl/environments/hanabi/hanabi_game.py`), and all Hydra configs. I wrote one read-only check, `_audit_code/check_hparam_symmetry.py`, which parses the PSI vs `base_marl_*` configs and confirms they differ in multiple hyperparameters across every algorithm.

Three substantive findings emerged: (1) the LIIR and LAIES baselines used in Figure 5d are entirely absent from the repo; (2) the Base MARL baselines (and the Intrinsic-Reward ablation) use *different* hyperparameters than PSI, contradicting the paper's explicit "same hyperparameters" claim and constituting undisclosed asymmetric tuning; (3) the Hanabi "5 Save" intrinsic-reward penalty in code is −2, whereas paper Algorithm 2 specifies −1 (plus a narrower trigger condition). The extrinsic/intrinsic return logging is correct (the primary-task curve measures the true task reward, not the shaped reward).

## 2. Traceability table

Results are training curves; the metric-computing code is the training loop + reward wrappers (`PrePolicyWrapper`/`PrePolicyHanabiWrapper` compute extrinsic & intrinsic returns; `get_greedy_metrics` evaluates them). No script in the repo aggregates seeds or renders the figures with CIs (off-repo from wandb logs); per Rule G plotting code is not required, so figure panels whose underlying metrics are computed by the training code are marked Computed.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig 5a (MPE) extrinsic/intrinsic returns, PSI/IntrinsicReward/BaseMARL × IQL,VDN | `baselines/QLearning/iql_pre.py`, `vdn_pre.py`; `jaxmarl/wrappers/baselines.py:159-210` | curve (off-repo agg) | n/a (curve) | Computed (training code present) |
| Fig 5b (Hanabi) extrinsic/intrinsic, PSI/IntrinsicReward/BaseMARL × IPPO,MAPPO | `baselines/IPPO/ippo_pre.py`, `baselines/MAPPO/mappo_pre.py`; `hanabi_game.py` | curve | n/a | Computed |
| Fig 5c Hanabi PSI vs GPSI | configs `intervene_two_agents=True`; `baselines.py:258-274` | curve | n/a | Computed (GPSI = `intervene_two_agents`) |
| Fig 5d PSI vs **LIIR** (Hanabi) and **LAIES** (MPE) | (none) | — | — | **MISSING** (no LIIR/LAIES code) |
| MPE Scenario 1 intrinsic reward (fixed landmark idx 0) | `augmented_mpe.py:703-716` | matches | ✓ | Verified |
| MPE Scenario 2 intrinsic reward (farthest from teammates 1,2) | `augmented_mpe.py:718-747` | matches | ✓ | Verified |
| Hanabi "5 Save" reward (Alg 2; paper penalty −1) | `hanabi_game.py:486-492` | penalty −2 | ✗ | MISMATCH (see hanabi-5save-penalty) |
| Hanabi "The Chop" reward (Alg 3) | `hanabi_game.py:347-355` | all-hinted −1, non-chop −2 | ✓ (magnitudes) | Verified (mostly; see notes) |
| "5 random seeds, 95% CI" | `iql_pre.py:608` (`NUM_SEEDS` split); config default `NUM_SEEDS:1` | seeds run = config value | partial | config default 1, not 5 (CLI-overridable) |
| Hyperparameter Tables 2–7 (PSI values) | configs `iql.yaml` etc. | several defaults differ from tables | partial | see hparam notes (CLI-overridable) |

## 3. Findings

## missing

```yaml finding
id: liir-laies-baselines-absent
category: missing
topic: "baselines"
title: "LIIR and LAIES baselines (Figure 5d, Appendix H.2) absent from repo"
severity: high
confidence: high
status: finding
file: baselines/QLearning/config/config.yaml
line_start: 1
line_end: 1
quote: |
  "NUM_SEEDS": 1
claim: "A repo-wide search (git grep -in 'liir|laies|diligence') returns no matches anywhere in the tracked code, configs, or environments; only Base MARL, Intrinsic Reward, GPSI (intervene_two_agents) and PSI conditions are implemented."
concern: "Figure 5d and Section 5.2 ('our PSI can outperform both LIIR and LAIES') are a headline comparison against the global-intervention methods LIIR (Hanabi) and LAIES-IDI (MPE), described in detail in Appendix H.2, yet no code implements either baseline, so that result cannot be reproduced."
resolution: "Authors: please add the LIIR and LAIES (IDI) baseline implementations / configs used to produce Figure 5d, or point to where they live."
cross_refs: []
check_script: _audit_code/check_hparam_symmetry.py
paper_ref: "Figure 5d; Appendix H.2 'Global Intervention'"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

N/A — no run-blocking or wired-wrong-against-own-intent defects identified in the audited paths. (The training scripts, reward wrappers, and environments are internally consistent and the README commands map onto existing entrypoints/configs.)

## difference

```yaml finding
id: hanabi-5save-penalty
category: difference
topic: "intrinsic reward (Hanabi 5 Save)"
title: "Hanabi '5 Save' penalty is -2 in code but -1 in paper Algorithm 2; trigger also narrower"
severity: medium
confidence: high
status: finding
file: jaxmarl/environments/hanabi/hanabi_game.py
line_start: 480
line_end: 492
quote: |
            # 2) check if a rank-hint was given but not rank=5
            did_not_hint_5 = jnp.logical_and(
                is_rank_hint,
                hint_idx != (self.num_ranks - 1)  # rank != 4 for 5 in 0-based
            )

            missed_5_opportunity = jnp.logical_and(target_has_unhinted_5, did_not_hint_5)

            # If missed 5, penalty -2; else 0
            penalty_for_missing_5_save = jnp.where(missed_5_opportunity, -2, 0)

            # sum up any other hint-based intrinsic logic if you have it
            intrinsic_reward = intrinsic_reward + penalty_for_missing_5_save
claim: "The implemented '5 Save' intrinsic reward applies a penalty of -2 when the targeted agent gives a rank hint that is not rank 5 while a teammate holds an unhinted 5; a color hint is never penalized."
concern: "Paper Algorithm 2 specifies the penalty as r5save = -1 and triggers it for any hint action that 'does NOT hint rank 5 or the color of an unhinted 5' (i.e. also penalizing non-relevant color hints), so both the magnitude and the trigger condition of the optimized/plotted intrinsic reward differ from the paper; the code's version is itself a valid shaping signal."
resolution: "Authors: confirm which definition (penalty -1 vs -2; rank-only vs rank-or-color trigger) produced the Hanabi 5-Save results, and reconcile Algorithm 2 with the code."
cross_refs: []
paper_ref: "Algorithm 2 ('5 Save'), lines 6-7"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: num-seeds-default-one
category: difference
topic: "reproducibility / seeds"
title: "Default NUM_SEEDS=1 in shipped configs vs paper's 5 seeds with 95% CIs"
severity: low
confidence: high
status: finding
file: baselines/QLearning/config/config.yaml
line_start: 1
line_end: 1
quote: |
  "NUM_SEEDS": 1
claim: "All shipped configs default to NUM_SEEDS=1 (also set in the three MAPPO alg configs); iql_pre.py:608 splits the PRNG into NUM_SEEDS keys and vmaps training, so the default run produces a single seed."
concern: "The paper reports 'Results from 5 random seeds ... with 95% confidence intervals', but the repo defaults to 1 seed; reproducing the reported CIs requires overriding NUM_SEEDS=5 (supported on the CLI) and is not documented in the README."
resolution: "Authors: set the default NUM_SEEDS to 5 or document the exact override used so the reported confidence intervals can be reproduced."
cross_refs: []
paper_ref: "Section 5.1: 'Results from 5 random seeds are reported as means with 95% confidence intervals.'"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: baseline-asymmetric-hparams
category: methodology
topic: "baselines / fairness of comparison"
title: "Base MARL & ablation baselines use different hyperparameters than PSI, contradicting paper's 'same hyperparameters' claim"
severity: high
confidence: medium
status: finding
file: baselines/QLearning/config/alg/base_marl_iql.yaml
line_start: 15
line_end: 22
quote: |
  "EPS_START": 1.0
  "EPS_FINISH": 0.05
  "EPS_DECAY": 0.1 # percentage of updates
  "MAX_GRAD_NORM": 25
  "TARGET_UPDATE_INTERVAL": 200
  "TAU": 1.
  "NUM_EPOCHS": 1
  "LR": 0.005
claim: "For every algorithm, the base_marl_* (Base MARL) config differs from the PSI config in multiple hyperparameters (see _audit_code/out/hparam_symmetry.json): e.g. IQL PSI LR=0.0035 vs base 0.005, eps 0.8->0.1 vs 1.0->0.05, grad-norm 5 vs 25, SEED 0 vs 30; MAPPO PSI LR=0.0065 vs base 0.0005; IPPO PSI ENT_COEF=0.02 vs 0.01. The intrinsic_reward_* ablation configs likewise differ from PSI in eps schedule, grad-norm and LR, not only the pre-policy module."
concern: "The paper states 'All Base MARL variants use the same network architecture and hyperparameters as our method' and frames the Intrinsic-Reward ablation as removing only the pre-policy module, but the code gives the baselines and the ablation separately-chosen (and in MAPPO's case ~13x smaller) learning rates, exploration schedules, gradient clipping and a different random seed, so the reported PSI-vs-baseline gains conflate the method with undisclosed asymmetric tuning."
resolution: "Authors: justify why each baseline/ablation uses different hyperparameters than PSI, or re-run the comparison with matched hyperparameters (and the same seeds) as the paper claims; clarify whether the baseline configs were tuned."
cross_refs: ["num-seeds-default-one"]
check_script: _audit_code/check_hparam_symmetry.py
paper_ref: "Appendix H.2 'Base MARL'; Section 5.1 'Baselines'"
tags: [reforms:5, whalen:pitfall-2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 1          | high         | LIIR/LAIES baselines (Figure 5d) absent from repo.                     |
| bug         | 0          | -            | No run-blocking / wired-wrong defects found in audited paths.          |
| difference  | 2          | medium       | Hanabi 5-Save penalty -2 vs paper -1; default NUM_SEEDS=1 vs 5.        |
| methodology | 1          | high         | Baselines & ablation use different hyperparameters than PSI (paper claims identical). |

## 5. Closing lists

**Top take-aways**
1. (`methodology`, high/med) Base MARL baselines and the Intrinsic-Reward ablation are given different hyperparameters (LR, eps schedule, grad-norm, seed) than PSI, contradicting the paper's "same hyperparameters" claim — the reported gains may partly reflect asymmetric tuning. [`baseline-asymmetric-hparams`]
2. (`missing`, high) LIIR and LAIES baselines used in the headline Figure 5d / Section 5.2 comparison are entirely absent from the repository. [`liir-laies-baselines-absent`]
3. (`difference`, medium) The Hanabi "5 Save" intrinsic-reward penalty is −2 in code vs −1 in paper Algorithm 2, with a narrower (rank-hint-only) trigger condition — this is both optimized and plotted. [`hanabi-5save-penalty`]
4. (`difference`, low) Shipped configs default to `NUM_SEEDS=1`, while the paper reports 5 seeds with 95% CIs. [`num-seeds-default-one`]

**Items that genuinely look fine**
- Extrinsic vs intrinsic return logging: `PrePolicyWrapper` (`baselines.py:159-210`) logs extrinsic (true task) returns separately from the intrinsic-augmented *training* reward, so the "primary task completion" curves are not contaminated by reward shaping.
- MPE Scenario 1 (fixed landmark index 0) and Scenario 2 (farthest landmark from teammates 1 & 2) intrinsic rewards (`augmented_mpe.py:703-747`) match Appendix H.4 exactly.
- GPSI is implemented as `intervene_two_agents=True` (`baselines.py:258-274`), matching the paper's global-intervention version of PSI.
- Dependencies are fully pinned (`pyproject.toml`); the README gives concrete reproduction commands that map onto existing entrypoints/configs.
- Single-targeted-agent intervention is consistently `agent_0` across MPE and Hanabi reward code.

**Open questions for the authors**
- Were the `base_marl_*` configs separately tuned, or are they intended to be upstream JaxMARL defaults? This determines whether finding `baseline-asymmetric-hparams` is asymmetric tuning or merely a documentation mismatch.
- Which "5 Save" definition (−1 vs −2, rank-only vs rank-or-color trigger) generated the Hanabi convention curves?
- Where are the LIIR / LAIES implementations and the figure-generation (seed-aggregation + CI) scripts?
