# Audit — ARGRE (Detoxifying LLMs via Autoregressive Reward Guided Representation Editing)

## Summary

The audited artefact is the author repository `code/xiaoyisong__ARGRE/` (NeurIPS 2025 paper #2188).
It implements the ARGRE test-time detoxification pipeline: (1) collect last-layer hidden states for
toxic/non-toxic pairs and extract a PCA "non-toxic direction" (`evaluation/collect_hidden/`),
(2) interpolate trajectories and train a 2-layer-MLP autoregressive reward model
(`evaluation/train_hidden/`), and (3) run reward-guided two-step representation editing at inference
on RealToxicityPrompts, scoring toxicity with Detoxify and perplexity/zero-shot accuracy with the LM
harness (`evaluation/baselines/run_argre.py`, `argre_infer.py`, `utils/evaluate_model.py`). The
custom-subclassed HF models in `reward_model/` implement the steering+gradient-ascent edit. The
second cloned repo `code/unitaryai__detoxify/` is the third-party Detoxify classifier (paper ref [68])
used only as the toxicity metric; it is not the paper's contribution and is not audited internally.

What I did: read every Python file in the author repo and the methodology/experiments/appendix of
`paper.pdf`; mapped paper artefacts to code; ran deterministic checks under `_audit_code/`
(`check_missing_artifacts.py` → `out/missing_artifacts.json`) verifying that the training data splits,
trained weights, several imported dependencies, and the Detoxify checkpoint file are absent; and
retrieved the dataset Google-Drive link. I confirmed the inference editing step matches Eqn 9/10 and
the reward loss matches Eqn 7, and that the toxicity/wiki evaluation sets shipped in `data/evaluation/`
match the paper's reported sizes (1,199 challenge prompts; 2,064 WikiText-2 dev samples).

The core method code is present and faithful to the paper's equations. The principal problems are
reproducibility gaps: the pairwise toxicity training data and trained reward models are not in the repo
(only an off-repo Drive link), several runtime dependencies are unlisted, the Detoxify metric is wired
to a nonexistent placeholder checkpoint path, and the two auxiliary experiments in Section 5
(stereotype recognition, jailbreak mitigation) have no code.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Tab. 1 Toxic/PPLg for ARGRE across 8 LLMs (headline −62.21%) | `evaluation/baselines/argre_infer.py` + `utils/evaluate_model.py` (`evaluate_model_regressive`, `toxicity_over_dataset`) | not runnable (needs training data + weights + Detoxify ckpt) | — | UNVERIFIABLE (depends on missing data/weights/ckpt) |
| Tab. 1 "Orig" / "banned" baseline rows | `evaluation/baselines/orig.py` (Orig); banned baseline (none) | — | — | Orig: present; banned: MISSING (external dict [82]) |
| Tab. 1 ProFS / Re-Control / RAD / GenARM / DPO rows | (none — paper uses external GitHub repos) | — | — | OUT OF SCOPE (declared external, App. A.1.3) |
| Tab. 2 Efficiency (128-token timing on LLaMA-30B) | (no timing script) | — | — | MISSING (no inference-timing harness) |
| Tab. 3 Capability PPLw / ACC | `utils/evaluate_model.py:89-194` (perplexity), `evaluate_ability:15-52` (LM harness) | not runnable | — | UNVERIFIABLE (deps + weights) |
| Fig. 3 toxicity vs #annotations (100..2000) | train+infer scripts via `--dp_nums` / `--data_nums` CLI | not runnable | — | SUPPORTED-BY-ARGS (re-run with arg) |
| Fig. 4 toxicity vs Nin (0..15) | train+infer via `--num_interpolations` CLI | not runnable | — | SUPPORTED-BY-ARGS |
| Tab. 4 step-size sweep (η 0..1) | infer via `--guide_lr` CLI | not runnable | — | SUPPORTED-BY-ARGS |
| §4.3 trajectory analysis (scale 0..1 toxicity) | (no scaled-interpolation eval script) | — | — | MISSING |
| Tab. 5 instruction-tuned LLMs | (no Mistral-Instruct / LLaMA-2-Chat config) | — | — | MISSING (no config/identifier) |
| Tab. 6 cross-model generalizability | `argre_infer.py` (swap `--score_model_path`) | not runnable | — | SUPPORTED-BY-ARGS |
| Tab. 7 Stereotype Recognition / Jailbreak Mitigation | (none) | — | — | MISSING (no code) |
| Non-toxic direction d+ (Eqn 4–5, PCA) | `collect_hidden/extract_direction.py`, `pca.py` | logic present | ✓ | Verified (code matches Eqn 4 + PCA first component) |
| Reward loss (Eqn 7) | `train_hidden/rm_trainer.py:196-228` (`rm_loss`, sigmoid) | logic present | ✓ | Verified |
| Two-step edit (Eqn 9 steer + Eqn 10 grad-ascent) | `reward_model/llama_rm.py:418-462` | logic present | ✓ | Verified (β=1 at inference per §4.1) |

## missing

```yaml finding
id: toxicity-pairwise-data-absent
category: missing
topic: "data availability"
title: "Pairwise toxicity training data (split_0/split_1.jsonl) not in repo; only a Drive link"
severity: high
confidence: high
status: finding
file: evaluation/collect_hidden/collect_hidden.py
line_start: 47
line_end: 52
quote: |
      filedir = "./ARGRE/data/toxicity_pairwise"
      if split == "train":
          filepath = os.path.join(filedir, 'split_0.jsonl')
          default_num_dps = args.data_nums
      else:
          filepath = os.path.join(filedir, 'split_1.jsonl')
          default_num_dps = 500
claim: "The hidden-state collection step reads split_0.jsonl (train) and split_1.jsonl (eval) from data/toxicity_pairwise/, but the repo ships only down.txt (a Google-Drive link to toxicity_pairwise.zip); the JSONL files are absent."
concern: "Every reported number depends on this dataset to build the non-toxic direction, trajectories, and reward model; without the files the entire pipeline cannot run, and the only source is an off-repo Drive link that prompts for sign-in when fetched."
resolution: "Authors: include the split_0/split_1.jsonl files (or a working fetch script with a public, unauthenticated link) and confirm the exact subset of the 24,576-example dataset used."
cross_refs: ["missing-trained-reward-weights"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "App. A.1.1 (Toxicity Annotations); README Setup"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-deps-not-in-req
category: missing
topic: "dependencies / environment"
title: "lm_eval, wandb, datasets, peft, numpy imported but absent from req.txt"
severity: medium
confidence: high
status: finding
file: req.txt
line_start: 1
line_end: 10
quote: |
  accelerate==0.29.2
  detoxify==0.5.2
  huggingface-hub==0.29.3
  scikit-learn==1.4.2
  scipy==1.13.0
  torch==2.2.2
  torchaudio==2.2.2
  torchvision==0.17.2
  tqdm==4.66.2
  transformers==4.41.0
claim: "req.txt lists 10 packages but five third-party modules that the code imports — lm_eval (capability eval), wandb (reward training), datasets (hidden-state collection), peft (model_utils), and numpy — are not listed."
concern: "The environment cannot be rebuilt from req.txt as specified; the capability/training/collection scripts will fail on import, and unpinned lm_eval in particular can change harness task definitions and thus the reported ACC."
resolution: "Authors: add lm_eval, wandb, datasets, peft, and numpy (pinned) to req.txt, or provide a complete lockfile/environment.yml."
cross_refs: []
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "README Setup ('pip install req.txt')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stereotype-jailbreak-code-absent
category: missing
topic: "result traceability / extensions"
title: "No code for Section 5 stereotype-recognition and jailbreak-mitigation experiments (Table 7)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "we perform 2-fold cross-validation on the 654 samples using Mistral 7B and report the average accuracy"
claim: "The paper reports a stereotype-recognition task (TrustLLM, 654 samples, 2-fold CV) and a jailbreak-mitigation task (JailbreakTrigger, 700 prompts, RtA rate) with ARGRE results in Table 7, but the repo contains no script, data loader, config, or metric for either task (grep for stereotype/jailbreak/TrustLLM/RtA/refus across all .py files returns nothing)."
concern: "Two of the paper's generalizability claims (Table 7) have no reproducible artefact in the repo."
resolution: "Authors: add the stereotype-recognition and jailbreak-mitigation data loaders, prompts, and evaluation/scoring code, or state that they are intentionally out of the released scope."
cross_refs: []
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 5, Table 7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-trained-reward-weights
category: missing
topic: "trained weights / intermediate outputs"
title: "No trained reward-model checkpoints or exp/ outputs shipped; inference loads best_score_model.pth that is not present"
severity: medium
confidence: high
status: finding
file: evaluation/baselines/run_argre.py
line_start: 84
line_end: 96
quote: |
        hidden_states_root="./ARGRE/evaluation/exp/collect_hidden_states"
        model_dir = "./ARGRE/evaluation/exp/train_hidden_states"

        hidden_dir = f"{model_name}_toxicity_2000"
        hidden_dir = os.path.join(hidden_states_root, hidden_dir)
        tasks = []

        for num_interpolations in num_interpolations_list:
            model_name_suffix = f"dp_{dp_nums}_interp{num_interpolations}"
            traget_dir = os.path.join(model_dir, f'{model_name}_{model_name_suffix}')
            score_path = os.path.join(traget_dir, f"checkpoints/best_score_model.pth")

            if not os.path.exists(score_path):
                continue
claim: "Inference loads trained reward-model weights from exp/train_hidden_states/.../checkpoints/best_score_model.pth and cached hidden states from exp/collect_hidden_states/, but no exp/ directory exists in the repo; run_argre.py silently `continue`s when the checkpoint is absent, producing no output."
concern: "The eight per-model reward checkpoints and cached hidden states needed to reproduce Tables 1/3 without re-running the full collection+training pipeline are not provided; combined with the missing training data, no headline number can be regenerated as shipped."
resolution: "Authors: release the trained reward-model checkpoints (or, given they retrain cheaply, the cached collected hidden states plus the training data) so Tables 1/3 can be reproduced."
cross_refs: ["toxicity-pairwise-data-absent"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Tables 1, 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: detoxify-checkpoint-placeholder-path
category: bug
topic: "evaluation metric wiring"
title: "Toxicity metric loads Detoxify from a nonexistent placeholder path 'todo/toxic_original-c1212f89.ckpt'"
severity: medium
confidence: high
status: finding
file: evaluation/utils/evaluate_model.py
line_start: 207
line_end: 207
quote: |
      toxicity_scorer = Detoxify('original', device=model.device, checkpoint='todo/toxic_original-c1212f89.ckpt')
claim: "The toxicity scorer is constructed with an explicit checkpoint= argument pointing to the relative path 'todo/toxic_original-c1212f89.ckpt', a placeholder that does not exist in the repo (verified by check_missing_artifacts.py); the explicit argument overrides Detoxify's default auto-download."
concern: "The toxicity number that drives every headline result (Tables 1/4/5/7, Figs 2-4) cannot be computed as written: Detoxify will fail to load the checkpoint from this dead path."
resolution: "Authors: replace 'todo/...' with the real checkpoint path, or drop the checkpoint= argument so Detoxify('original') auto-downloads its weights, and document the exact Detoxify model/version used."
cross_refs: []
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "App. A.1.1; ref [68] Detoxify"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: toxicity-eval-generation-length
category: difference
topic: "evaluation consistency"
title: "Toxicity is scored on only 10 generated tokens, not stated in the paper"
severity: low
confidence: low
status: question
file: evaluation/utils/evaluate_model.py
line_start: 212
line_end: 213
quote: |
          response = model.generate(input_ids=input_ids, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)   
          response = response[0, len(input_ids[0]):].tolist()  
claim: "Detoxify toxicity is computed on continuations of max_new_tokens=10, whereas the efficiency benchmark (Tab. 2) times 128-token generations and the paper does not state the toxicity-evaluation generation length; the ProFS/detox-edit protocol the paper follows typically scores ~20-token continuations."
concern: "The toxicity scores depend on continuation length, so a 10-token cap (if it diverges from the protocol the baselines were run under) could make the cross-method comparison length-inconsistent; ambiguous because the paper omits the number."
resolution: "Authors: state the generation length used for toxicity scoring and confirm all methods (including the externally-run baselines) used the same length."
cross_refs: []
paper_ref: "§4.1 (Toxicity); Tab. 2 (128 tokens)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

N/A — the implemented procedure is methodologically sound where present. The non-toxic direction is
fit on the training-split pairs only (PCA in `extract_direction.py`), the average non-toxic reward
`r+_mean` used for steering is computed from the training hidden states (`argre_infer.py:87,118`, loaded
with `prefix='train'`), and the reward-model "best" checkpoint is selected on the pairwise reward eval
split (`split_1.jsonl`), which is distinct from the RealToxicityPrompts toxicity test set, so checkpoint
selection does not touch the reported toxicity metric. No train/test leakage into the headline metric
was found.

## Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Training data, trained weights, several deps, and the two Section-5 experiments are absent. |
| bug         | 1          | medium       | Toxicity metric points at a nonexistent placeholder Detoxify checkpoint path. |
| difference  | 1          | low          | Toxicity scored on 10 generated tokens; length unstated in paper (question). |
| methodology | 0          | -            | Direction/reward/edit logic faithful; no leakage into headline metric found. |

## Top take-aways

1. (missing, high) `toxicity-pairwise-data-absent` — the pairwise toxicity training data is not in the
   repo; only an off-repo Google-Drive link (`toxicity_pairwise.zip`) that prompts for sign-in. Every
   reported number depends on it.
2. (missing, medium) `missing-trained-reward-weights` — no `exp/` outputs or reward checkpoints, so
   Tables 1/3 cannot be regenerated without first rebuilding the full collection+training pipeline
   (which itself needs the missing data).
3. (bug, medium) `detoxify-checkpoint-placeholder-path` — the headline toxicity metric is wired to
   `checkpoint='todo/toxic_original-c1212f89.ckpt'`, a dead placeholder path.
4. (missing, medium) `missing-deps-not-in-req` — `lm_eval`, `wandb`, `datasets`, `peft`, `numpy` are
   imported but not in `req.txt`; the environment cannot be rebuilt as instructed.
5. (missing, medium) `stereotype-jailbreak-code-absent` — Section 5 / Table 7 experiments have no code
   in the repo.
6. (difference, low) `toxicity-eval-generation-length` — toxicity scored on 10 generated tokens; the
   paper never states the length (filed as a question pending clarification).

## Items that genuinely look fine

- Non-toxic direction extraction (`extract_direction.py` + `pca.py`) implements Eqn 4 + first PCA
  component on the difference vectors, fit on training pairs only.
- Reward-model loss (`rm_trainer.py:196-228`) is the standard Bradley-Terry sigmoid objective of Eqn 7;
  masking uses the attention mask correctly (`get_batch_logps`).
- Two-step edit in `reward_model/*_rm.py::optimize_hidden_states_with_reward` matches Eqn 9 (gated
  directional steering by `pos_mean − reward`) and Eqn 10 (SGD gradient ascent on the reward for
  `guide_epochs` iterations); identical across the llama/mistral/opt/gpt2 variants.
- `r+_mean` is computed from training hidden states, and reward-model checkpoint selection uses the
  pairwise reward-eval split, not the toxicity test set — no leakage into the headline metric.
- Shipped evaluation sets match the paper: 1,199 RealToxicityPrompts challenge prompts and 2,064
  WikiText-2 dev samples (`data/evaluation/`).
- Baseline methods (ProFS, Re-Control, GenARM, RAD, DPO, banned) are explicitly run from their own
  GitHub repos / external dictionary (App. A.1.3), so their absence from this repo is declared, not a gap.
- Ablation sweeps (Figs 3/4, Tab. 4, Tab. 6) are reproducible via existing CLI args
  (`--data_nums`, `--num_interpolations`, `--guide_lr`, `--score_model_path`) once data/weights exist.

## Open questions for the authors

- Is the `toxicity_pairwise.zip` Drive link intended to be public (it currently prompts for Google
  sign-in)? Which subset / random seed of the 24,576-example dataset produces `split_0`/`split_1`?
- What generation length was used for Detoxify toxicity scoring, and was it identical for the externally
  run baselines?
- Will trained reward-model checkpoints (or cached hidden states) be released to allow reproducing
  Tables 1/3 without re-running collection+training on eight LLMs?
