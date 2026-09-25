# Audit — Learning to Better Search with Language Models via Guided Reinforced Self-Training (NeurIPS 2025, #5067)

## 1. Summary

The repo (`snu-mllab/guided-rest`) is a fork of the **verl** RL framework. The
paper's own contribution lives in two recipe folders, `recipe/countdown/` and
`recipe/code_repair/`, plus two custom additions to verl's PPO core
(`hgae` advantage estimator and `hppo` policy loss, implementing the
"operation-level MDP" of §3.3). Each recipe provides the full pipeline:
`download_data.py`, `download_model.py`, generation (`main_gen.py`,
implementing the subgoal-augmentation / guided data generation), data prep
(`main_data.py`), training driver scripts (`run_sft_*.sh`, `run_rl.sh`), and
pass@k evaluation (`main_eval.py`).

What I did:
- Read the paper (`paper.pdf` / `paper_text.txt`) and mapped every figure/table
  to repo code.
- Read the core method code: `recipe/countdown/core_algos.py` (subgoal
  augmentation = Algorithm 1/2), `main_gen.py`, `reward_function.py`,
  `utils.py`, `main_eval.py`; `recipe/code_repair/main_gen.py` (episode-level
  variant = Algorithm 3), `rewards/code_reward.py`; and the verl additions
  `verl/trainer/ppo/core_algos.py:265-295` (hgae), `:1008` (hppo).
- Verified data availability: all four HuggingFace datasets referenced by the
  download scripts (`symoon11/countdown-sft`, `countdown-rl`,
  `code-repair-rl-pi`, `code-repair-rl-cc`, `code-repair-rl-cf`) return HTTP
  200 and have split sizes matching the paper (train 200K, valid 1K,
  test_seen/unseen 10K each; CC 165, CF 408).
- Ran `_audit_code/check_passk.py` (the pass@k mean matches the Chen et al.
  unbiased estimator exactly), `_audit_code/check_seen_unseen.py` (test_unseen
  targets are fully disjoint from train targets; test_seen fully overlap —
  the seen/unseen generalization split is sound).

The core algorithm, data-generation logic, training scripts, evaluation, and
the operation-level PPO are all present and look faithful. The main gaps are
**missing baseline / ablation code** (SoS, BC, the token-level-MDP ablation in
Fig 5, and the dense-reward ablation in Fig 6 / §5.5), one **path-typo bug** in
the code-repair model download, and one **paper↔code difference** in the
code-repair SFT loss masking.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 3 — Guided-ReST greedy accuracy vs budget (Countdown) | `recipe/countdown/{main_gen,main_eval}.py` + `scripts/llama_3.2_1b/guided_rest/*` | recomputable | — (not run, GPU) | Code present |
| Fig. 3 — **SoS** curve | no trace-generation script; SoS data on HF (`countdown-sft`) | — | — | PARTIAL (data only, gen code MISSING) |
| Fig. 3 — **BC** curve (abstract "<40%") | (none) — no BC data, no BC script | — | — | MISSING |
| Fig. 3 / Table 2 — **ReST** | `scripts/llama_3.2_1b/rest/*` + `run_gen.sh num_iters=0` | recomputable | — | Code present |
| Table 2 — pass@k Countdown (SoS/ReST/Guided-ReST) | `main_eval.py:27-72` | estimator verified (`_audit_code/check_passk.py`) | ✓ (formula) | Code present |
| Fig. 4 — ReST+PPO vs PPO | `rest/run_rl.sh`, `base/run_rl.sh` | recomputable | — | Code present |
| Fig. 5 — **token-level MDP** ablation | (none) — all 3 rl scripts use `hgae`/`hppo` (operation-level) | — | — | MISSING (no GAE/token-PPO script) |
| Fig. 6 / §5.5 — **dense (subgoal) reward** ablation | (none) — no subgoal-reward code anywhere | — | — | MISSING |
| Table 1 — preliminary partial-solution accuracy/CE | (none) | — | — | MISSING |
| Table 3 — pass@k code self-repair (CC/CF) | `recipe/code_repair/{main_gen,main_eval}.py`, `rewards/code_reward.py` | recomputable | — | Code present |
| §5.1.2 — "loss only on last response" (code) | `verl/utils/dataset/multiturn_sft_dataset.py:283-330` masks loss=1 on **all** assistant turns | conflicts | ✗ | DIFFERENCE |
| §3.3 — operation-level PPO (hgae/hppo) | `verl/trainer/ppo/core_algos.py:265-295`, `:1008` | present | ✓ | Code present |
| Setup — base model download (code-repair) | `recipe/code_repair/download_model.py:23` writes `qwen.2.5_7b/` but scripts read `qwen2.5_7b/` | path mismatch | ✗ | BUG |

## 3. Findings

## missing

```yaml finding
id: bc-baseline-absent
category: missing
topic: "baselines"
title: "BC baseline (Fig. 3) has neither training code nor data in the repo"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 26
line_end: 90
quote: |
  ## Countdown

  1. Download the SFT and RL datasets
claim: "The Countdown section of the README and all recipe scripts provide only SoS (base_sft), ReST, and Guided-ReST pipelines; there is no behavior-cloning (BC) data file on HuggingFace and no BC training/eval script, yet Fig. 3 and the abstract report a BC curve (\"BC achieves less than 40% accuracy\")."
concern: "A reported baseline that establishes the lower bound of the comparison cannot be reproduced because neither the BC training data (optimal-solution-only traces) nor a BC training script is provided."
resolution: "Authors: please add the BC data-preparation script (converting extra_info.solution into solution-only SFT targets) and the BC training command."
cross_refs: ["sos-generation-absent"]
paper_ref: "Figure 3; abstract; §5.2"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: token-level-mdp-ablation-absent
category: missing
topic: "ablations"
title: "Token-level MDP ablation (Fig. 5) has no corresponding script"
severity: medium
confidence: high
status: finding
file: recipe/countdown/scripts/llama_3.2_1b/guided_rest/run_rl.sh
line_start: 6
line_end: 9
quote: |
    algorithm.adv_estimator=hgae \
    data.train_files=${train_path} \
    data.val_files=${valid_path} \
    data.train_batch_size=1024 \
claim: "All three RL drivers (base/run_rl.sh, rest/run_rl.sh, guided_rest/run_rl.sh) hardcode the operation-level formulation via `algorithm.adv_estimator=hgae` and `actor_rollout_ref.actor.policy_loss.loss_mode=hppo`; no script runs standard token-level PPO (e.g. `adv_estimator=gae`, default loss)."
concern: "Figure 5 compares the proposed operation-level MDP against a token-level MDP and reports a 83%→87% gain, but the token-level configuration that produces the 83% baseline curve is not provided, so the ablation cannot be reproduced."
resolution: "Authors: please add the token-level-MDP RL script (the exact adv_estimator/loss_mode/config used for the \"Ours (token-level)\" curve in Fig. 5)."
cross_refs: []
paper_ref: "Figure 5; §5.4"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dense-subgoal-reward-ablation-absent
category: missing
topic: "ablations"
title: "Dense subgoal-reward RL ablation (Fig. 6 / §5.5) has no code"
severity: medium
confidence: high
status: finding
file: recipe/countdown/reward_function.py
line_start: 20
line_end: 27
quote: |
  def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any]) -> float:
      target = extra_info["target"]
      nums = extra_info["nums"]
      try:
          is_correct = grade_search_path(solution_str, target, nums)
          return 1.0 if is_correct else 0.0
      except:
          return 0.0
claim: "The only reward function is the binary outcome reward (1 for success, 0 otherwise); a grep across the repo for the subgoal reward (value 0.25, \"subgoal\", \"dense\") returns nothing in recipe/countdown."
concern: "§5.5 / Fig. 6 report a \"PPO (dense)\" experiment using a subgoal-based reward R_subgoal=0.25 combined with the outcome reward, but no code computes this dense reward, so the ablation cannot be reproduced."
resolution: "Authors: please add the subgoal/dense reward function and the RL script used for the \"PPO (dense)\" curve in Fig. 6."
cross_refs: []
check_script: _audit_code/out/passk.txt
paper_ref: "Figure 6; §5.5"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: sos-generation-absent
category: missing
topic: "baselines / data generation"
title: "SoS heuristic DFS/BFS trace-generation code not in repo (data only on HF)"
severity: low
confidence: high
status: finding
file: recipe/countdown/download_data.py
line_start: 18
line_end: 25
quote: |
  def main():
      # download sft dataset
      sft_train_dataset = load_dataset("symoon11/countdown-sft", split="train")
      sft_valid_dataset = load_dataset("symoon11/countdown-sft", split="validation")

      # save sft dataset
      sft_train_dataset.to_parquet(f"data/countdown/sft/train.parquet")
      sft_valid_dataset.to_parquet(f"data/countdown/sft/valid.parquet")
claim: "The SoS base-SFT search traces are obtained only by downloading the pre-built `symoon11/countdown-sft` dataset; the symbolic heuristic-guided DFS/BFS generator that produced these traces (paper §5.1.1, \"we generate search traces using heuristic-guided DFS and BFS over 500K training examples\") is not present in the repo."
concern: "The SoS baseline's data-generation procedure is not reproducible from the code; only the resulting dataset is distributed, so the trace-construction step cannot be inspected or re-run."
resolution: "Authors: please add (or link) the DFS/BFS search-trace generator used to build the countdown-sft dataset, or confirm it is intentionally distributed as data only."
cross_refs: ["bc-baseline-absent"]
paper_ref: "§5.1.1"
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: table1-preliminary-experiment-absent
category: missing
topic: "result traceability"
title: "Table 1 preliminary partial-solution experiment has no code"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  "Table 1: Accuracy and cross-entropy loss of search traces from partial solutions of different lengths."
claim: "Table 1 (accuracy and cross-entropy loss of traces seeded from partial optimal solutions of length 0/1/2/3, over 10K examples) is a motivating experiment, but no script in the repo seeds generation from fixed-length partial solutions or computes the cross-entropy loss reported there."
concern: "The motivating numbers in Table 1 cannot be reproduced from the released code."
resolution: "Authors: please add the script that prepends partial solutions of length t and measures continuation accuracy and CE loss (Table 1)."
cross_refs: []
paper_ref: "Table 1; §3.1"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: coderepair-model-download-path-typo
category: bug
topic: "reproducibility / paths"
title: "code-repair download_model.py writes to qwen.2.5_7b/ but scripts read qwen2.5_7b/"
severity: low
confidence: high
status: finding
file: recipe/code_repair/download_model.py
line_start: 22
line_end: 27
quote: |
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16)
    model.save_pretrained("checkpoints/code_repair/qwen.2.5_7b/huggingface")

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokenizer.save_pretrained("checkpoints/code_repair/qwen.2.5_7b/huggingface")
claim: "download_model.py saves the base model/tokenizer to `checkpoints/code_repair/qwen.2.5_7b/huggingface` (with a dot in `qwen.2.5_7b`), whereas the README (`model_name=qwen2.5_7b`) and every training/gen script read `checkpoints/code_repair/qwen2.5_7b/huggingface` (no dot, e.g. run_sft_1.sh:5 `model_path=checkpoints/code_repair/qwen2.5_7b/huggingface`)."
concern: "Following the README verbatim downloads the base model into a directory that the subsequent generation and SFT scripts never look in, so the code-repair pipeline fails to find the base model out of the box."
resolution: "Rename the save path to `checkpoints/code_repair/qwen2.5_7b/huggingface` (remove the dot) so it matches the scripts and README."
cross_refs: []
paper_ref: "README 'Code self-repair' step 2"
tags: [lones:stage-5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: coderepair-sft-loss-all-turns
category: difference
topic: "training objective"
title: "Code-repair SFT computes loss on all assistant turns, paper says only the last"
severity: medium
confidence: medium
status: finding
file: verl/utils/dataset/multiturn_sft_dataset.py
line_start: 283
line_end: 296
quote: |
        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages,
                    i,
                    i + 1,
                    is_assistant=True,
                    continue_final_message=continue_final_message and i == len(messages) - 1,
                    enable_thinking=enable_thinking,
                    tools=tools,
                )
claim: "verl's multi-turn SFT dataset assigns loss_mask=1 to every assistant message (the loop tokenizes each assistant turn with is_assistant=True; per-message override at lines 319-326 is only used if the message dict carries a `loss_mask` key). recipe/code_repair/main_data.py builds `messages = prompt + response` with no per-message loss_mask, and main_gen.py creates assistant messages without one, so all assistant turns in a multi-turn episode are trained on."
concern: "The paper (§5.1.2) states \"We compute the loss only on the last model response, following the practice of Snell et al.\", but the released code-repair SFT trains on every assistant turn of the episode, a different training objective than reported."
resolution: "Authors: confirm whether the code-repair SFT data carries a per-turn loss_mask (set to 0 on non-final assistant turns); if not, reconcile the code with the \"loss only on last response\" claim."
cross_refs: []
paper_ref: "§5.1.2"
tags: [reforms:6, lones:stage-4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The split (seen/unseen target disjointness) was
verified sound (`_audit_code/check_seen_unseen.py`); the pass@k estimator is the
standard unbiased Chen-et-al. estimator (`_audit_code/check_passk.py`);
evaluation reward uses the private test cases while the public-test subset is
only used for the in-episode revision feedback (no test leakage into the
score); each Guided-ReST/ReST SFT iteration re-initializes from the reference
model, matching Algorithm 2.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 5          | medium       | BC, token-level-MDP & dense-reward ablations, SoS gen, Table 1 |
| bug         | 1          | low          | code-repair base-model download path typo (`qwen.2.5_7b`)   |
| difference  | 1          | medium       | code-repair SFT loss on all turns vs "only last" claim      |
| methodology | 0          | -            | split, metric, and reward checks all passed                 |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **(missing)** `token-level-mdp-ablation-absent` — the token-level-MDP RL
   config behind Fig. 5's 83% baseline is not in the repo (all RL scripts hardcode operation-level hgae/hppo).
2. **(missing)** `dense-subgoal-reward-ablation-absent` — the dense subgoal
   reward (Fig. 6 / §5.5) is implemented nowhere; only a binary outcome reward exists.
3. **(missing)** `bc-baseline-absent` — the BC baseline (Fig. 3, abstract) has
   neither data nor a training script.
4. **(difference)** `coderepair-sft-loss-all-turns` — code-repair SFT trains on
   all assistant turns; paper claims loss only on the last response.
5. **(missing)** `sos-generation-absent` — SoS DFS/BFS trace generator absent
   (the resulting dataset is downloadable, so impact is limited).
6. **(bug)** `coderepair-model-download-path-typo` — base model is saved to a
   directory (`qwen.2.5_7b`) the scripts never read; trivial but blocks out-of-box runs.

### Items that genuinely look fine
- Core subgoal-augmentation algorithm (`core_algos.py:augment_search_path` /
  `get_subgoal_nodes`) faithfully implements Algorithm 1/2.
- Seen/unseen target split is leakage-free: unseen targets disjoint from train,
  seen targets overlap (`_audit_code/check_seen_unseen.py`).
- pass@k mean is the standard unbiased estimator (`_audit_code/check_passk.py`,
  exact match for n∈{32,128}).
- Code-repair scoring uses **private** tests; public tests only drive the
  in-episode revision feedback — no test leakage into the reported metric.
- All four/five HuggingFace datasets resolve (HTTP 200) with split sizes
  exactly matching the paper (200K/1K/10K/10K; CC 165, CF 408).
- Operation-level PPO (hgae/hppo) is present in verl and matches §3.3 (MC
  return, no KL penalty, operation-span aggregation skipping observation tokens).
- No hardcoded absolute paths; relative checkpoint dirs throughout.

### Open questions for the authors
- Does the released code-repair SFT data carry a per-turn `loss_mask` field
  (so verl masks non-final assistant turns), or is loss really applied to all
  turns? (drives `coderepair-sft-loss-all-turns` severity)
- Are the SoS/BC/ablation generation scripts intentionally omitted (data-only
  release) or were they used off-repo?
