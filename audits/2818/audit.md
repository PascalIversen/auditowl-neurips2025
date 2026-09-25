# Audit — Scaling Up Parameter Generation: A Recurrent Diffusion Approach (RPG)

## 1. Summary

The repository (`NUS-HPC-AI-Lab/Recurrent-Parameter-Generation`, single commit
`65da1eb`, shallow clone, no submission tag) implements RPG: a generator that
tokenizes a network's weights, runs a recurrent model (Mamba by default) to
produce per-token "prototypes", and a 1-D-conv diffusion model that denoises
each token into final parameters. Per-architecture "dataset" folders
(`dataset/<arch>/{train.py,test.py,model.py}`) finetune a `timm` model and save
50 checkpoints; `workspace/main/*` and `workspace/example/*` train RPG on those
checkpoints; `workspace/evaluate/{generate,evaluate}.py` generate weights and
run the per-arch `test.py` to report accuracy. Ablation drivers exist for the
recurrent-model, position-embedding, tokenization, token-size, AR, and
recurrent-structure studies. The unseen-task experiment lives in
`workspace/condition/generalization.py` + `dataset/condition_classinput_vittiny`.

I read the paper (PDF + text extraction) and the core code (model, dataset,
train/eval/generate, ablation and condition drivers). I wrote two read-only
checks under `_audit_code/`:
- `check_repo_facts.py` → `out/repo_facts.json`: hardcoded `/path/to`
  placeholders, requirements completeness/pinning, stubbed "coming soon" test
  commands, and the unseen-task split.
- `check_attention_mask.py` → `out/attention_mask.json`: shows the AR-ablation
  causal mask leaks future tokens.

I could not execute the pipeline (needs GPUs, `mamba-ssm`, ImageNet, and the
68 GB HF checkpoint set), so accuracy values in the traceability table are
"not reproduced here"; I verified instead which script *computes* each number
and whether the procedure is sound. Trained RPG weights and the ViTTiny1022
condition dataset are published on Hugging Face (both URLs resolve).

Central reading of the method (informs the methodology findings): for the main
tables (1–3, 10) RPG is trained on 50 checkpoints of *the same* finetuned model
on *the same* task and then "generates" weights for that same task — i.e. it
reproduces the distribution of its own training checkpoints; there is no
held-out task in those experiments (this is acknowledged by the paper, which
frames Section 4 as the generalization test). In Section 4 the task condition
is injected only into the last 8 tokens of the sequence (the classifier head),
so the generated backbone is task-agnostic.

## 2. Traceability table

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Tab.1 ImageNet acc, 7 archs (best/avg/min) | `dataset/imagenet_*/test.py` via `workspace/evaluate/evaluate.py` + `workspace/main/*` | Yes (downstream acc on standard val) | not reproduced here | Verified-present |
| Tab.2 ADE20K mIoU/mAcc, COCO mAP | (no runnable eval; `register.py:58-67` test_command = "coming soon"; `dataset/downtask_{detection,segmentation}/test.sh` use `/path/to/...`; no `train.py` to build checkpoints) | No | — | MISSING |
| Tab.3 DoRA LLaMA-7B 7 sub-tasks (r4/r64) | `dataset/downtask_dora_r4|r16|r64/{train,test}.py` (shells out to external DoRA repo) | Yes (external repo) | not reproduced here | Verified-present (needs external DoRA) |
| Tab.4a −recurrent model "fail" | `workspace/ablation/no_relation/no_relation.py` (GMLP) | Yes | not reproduced | Verified-present |
| Tab.4b position-embedding manners | `workspace/ablation/trainable_pe/*` + `pe_granularity` in `dataset/dataset.py:230-262` | Yes | not reproduced | Verified-present |
| Tab.4c tokenization (flatten/channel/layer) | `granularity` switch in `dataset/dataset.py:26-93` | Yes | not reproduced | Verified-present |
| Tab.5 token size 1024–16384 | `workspace/ablation/token_dim/*` | Yes | not reproduced | Verified-present |
| Tab.6 AR vs recurrent (transformer) | `workspace/ablation/rm_structure/transformer.py` + `model/transformer.py` | Yes, but causal mask is broken (see `causal-mask-leaks-future`) | not reproduced | Verified-present (buggy mask) |
| Tab.7 LSTM/Trans./Mamba | `model/{lstm,transformer,mamba}.py` + `workspace/ablation/rm_structure/*` | Yes | not reproduced | Verified-present |
| Tab.8/Tab.16 RPG & generated param counts | `workspace/main/*` + `model/__init__.py`, `model/mamba.py` | Implicit (config-derived) | not reproduced | Verified-present |
| Tab.9/Tab.23 memory & time | `workspace/evaluate/{memory,efficiency}.py` | Yes | not reproduced | Verified-present |
| Tab.10 SKDE30/p-diff/D2NWG/SANE vs RPG | `workspace/compare/ours_*` and `pdiff_*`; **no** SKDE/D2NWG/SANE code | Partial (only p-diff + ours) | not reproduced | Partial (external baselines not in repo) |
| Tab.11/Tab.19 unseen-task acc (ViT-Tiny CIFAR-10) | `workspace/condition/generalization.py` + `dataset/condition_classinput_vittiny` | Yes (but head-only conditioning, see `condition-only-head-tokens`) | not reproduced | Verified-present |
| Tab.17 permutation-state effect | `workspace/condition/{no_,}permutation_*.py` | Yes | not reproduced | Verified-present |
| Tab.18 LMC | (no dedicated LMC script found) | No | — | MISSING (low importance) |
| Tab.22 LLM-prompt → embedding | `workspace/classinput/qwen25llm.py` | Yes | not reproduced | Verified-present |
| Tab.26 with/without diffusion | (toggle not located as a driver script) | Unclear | — | not located |

## 3. Findings

## missing

```yaml finding
id: ade20k-coco-eval-missing
category: missing
topic: "result traceability / evaluation code"
title: "Table 2 (ADE20K/COCO) has no runnable evaluation or checkpoint-collection code"
severity: high
confidence: high
status: finding
file: dataset/register.py
line_start: 55
line_end: 67
quote: |
  class CocoDetection(BaseDataset):
      data_path = "./dataset/downtask_detection/checkpoint"
      generated_path = "./dataset/downtask_detection/generated/generated_model.pth"
      test_command = "echo \"Code for testing is coming soon!\n\""
      # test_command = "bash ./dataset/downtask_detection/test.sh " + \
      #                "./dataset/downtask_detection/generated/generated_model.pth"
  
  class ADE20KSegmentation(BaseDataset):
      data_path = "./dataset/downtask_segmentation/checkpoint"
      generated_path = "./dataset/downtask_segmentation/generated/generated_model.pth"
      test_command = "echo \"Code for testing is coming soon!\n\""
      # test_command = "bash ./dataset/downtask_segmentation/test.sh " + \
      #                "./dataset/downtask_segmentation/generated/generated_model.pth"
claim: "The active test commands for the COCO and ADE20K tasks are stub `echo \"...coming soon!\"` strings; the commented-out real commands call `dataset/downtask_{detection,segmentation}/test.sh`, which themselves reference non-existent external dirs via hardcoded `/path/to/...` placeholders. There are also no `train.py` scripts in those two folders to build the checkpoints."
concern: "Every value in Table 2 (mIoU 47.1, mAcc 57.5, mAP Bbox 44.5, mAP Seg 39.6) is unreproducible from the repository: neither the checkpoint-collection nor the evaluation is present and runnable."
resolution: "Authors: please add the segmentation/detection checkpoint-collection scripts and a self-contained (or clearly documented external) evaluation, replacing the `/path/to/...` placeholders and the `coming soon` stubs."
cross_refs: ["hardcoded-paths-downtask"]
check_script: _audit_code/check_repo_facts.py
paper_ref: "Table 2; Section 3.2 'On ADE20K and COCO'"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: requirements-incomplete-unpinned
category: missing
topic: "dependencies / reproducibility"
title: "requirements.txt omits core deps (numpy, pandas, matplotlib, tqdm) and pins nothing"
severity: medium
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 9
quote: |
  timm
  wandb
  einops
  seaborn
  openpyxl
  bitsandbytes
  transformers
  accelerate
  bypy
claim: "requirements.txt lists 9 packages, none version-pinned. Code additionally imports numpy, pandas, matplotlib, and tqdm (none listed anywhere) plus torch/torchvision/mamba-ssm (covered only by the README conda commands). `check_repo_facts.py` enumerates imports vs the file."
concern: "A clean environment built from requirements.txt fails at import (e.g. `import numpy`, `import pandas`, `import matplotlib` in `workspace/evaluate/evaluate.py`), and the absence of any version pins makes the environment non-reconstructible."
resolution: "Authors: add numpy, pandas, matplotlib, tqdm (and scipy if used) and pin versions, or document them in the README install steps."
cross_refs: []
check_script: _audit_code/check_repo_facts.py
paper_ref: "Appendix B (training recipe)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: causal-mask-leaks-future
category: bug
topic: "attention masking (AR ablation)"
title: "Causal mask fills masked logits with 1e-8 (not -inf), so AR ablation leaks future tokens"
severity: low
confidence: high
status: finding
file: model/transformer.py
line_start: 45
line_end: 48
quote: |
          attn = q @ k.transpose(-1, -2)
          if self.use_mask:
              attn = torch.where(self.mask, attn, 1e-8)
          out = self.softmax(attn) @ v
claim: "Masked attention positions are set to the constant 1e-8 before softmax instead of -inf. The AR ablation (`workspace/ablation/rm_structure/transformer.py:124-125`) builds a triangular causal mask and passes it via `config['model_config']['mask']`, so this path is exercised. `check_attention_mask.py` shows token 0 still places ~0.97 of its softmax weight on future tokens (vs 0.0 with a correct mask)."
concern: "The 'causal' / auto-regressive transformer used for Table 6 does not actually prevent attending to future tokens, so the AR-vs-recurrent comparison is confounded by a masking defect (note the main results use Mamba and are unaffected)."
resolution: "Authors: use `attn.masked_fill(~mask, float('-inf'))` (and apply the `self.scale` factor) and re-run the AR ablation, or confirm the Table 6 'fail' result is robust to a correct causal mask."
cross_refs: []
check_script: _audit_code/check_attention_mask.py
paper_ref: "Table 6; Appendix C.9"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: generalization-scheduler-double-step
category: bug
topic: "lr scheduler"
title: "generalization.py steps the scheduler twice per iteration on the main process"
severity: low
confidence: medium
status: finding
file: workspace/condition/generalization.py
line_start: 168
line_end: 171
quote: |
          accelerator.backward(loss)
          scheduler.step(batch_idx)
          if accelerator.is_main_process:
              scheduler.step()
claim: "The scheduler is stepped with `scheduler.step(batch_idx)` and then, on the main process only, stepped again with `scheduler.step()`, advancing the cosine LR schedule at roughly double the intended rate (and inconsistently between main and non-main processes). `optimizer.step()` is also absent from this loop."
concern: "The learning-rate schedule for the unseen-task experiment does not match the documented cosine-over-120k-steps recipe, and the optimizer step appears omitted, so the script as committed does not implement the described training; the committed commit message is 'fix schuler'."
resolution: "Authors: confirm whether `optimizer.step()` and a single scheduler step were intended; provide the exact training loop used to produce the Section 4 results."
cross_refs: []
paper_ref: "Section 4; Table 15 (cosine schedule, 120k steps)"
tags: [forensics:git-archaeology, lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: readme-generalization-script-commented-out
category: difference
topic: "reproduction instructions"
title: "README launches a generalization script whose training loop is fully commented out"
severity: low
confidence: high
status: finding
file: workspace/classinput/generalization.py
line_start: 219
line_end: 223
quote: |
  # if __name__ == '__main__':
  #     train()
  #     del train_loader  # deal problems by dataloader
  #     print("Finished Training!")
  #     exit(0)
claim: "`workspace/classinput/generalization.py` has its optimizer, accelerator, training loop, and `__main__` block all commented out, so it trains nothing. The runnable Section-4 trainer is the differently-located `workspace/condition/generalization.py`, while the README quick-start points users to `./condition/generalization.py` and Section-4 wiring is split across both files."
concern: "Following the repo as organized (the `classinput/generalization.py` driver) produces no training; the correct entry point exists but the duplicated, commented-out file is a reproduction hazard rather than a result-invalidating defect."
resolution: "Authors: remove or clearly mark the dead `workspace/classinput/generalization.py`, and make the README point unambiguously to the working Section-4 driver."
cross_refs: ["generalization-scheduler-double-step"]
paper_ref: "Section 4; README 'Reproduce Section 4'"
tags: [lones:stage-5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-paths-downtask
category: difference
topic: "hardcoded paths"
title: "Detection/segmentation shells contain /path/to placeholders and out-of-repo dependencies"
severity: low
confidence: high
status: finding
file: dataset/downtask_segmentation/test.sh
line_start: 3
line_end: 9
quote: |
  source /path/to/miniconda3/bin/activate /path/to/miniconda3/envs/environment

  python ./convert.py "$1"

  PYTHONPATH=/path/to/Segmentation:$PYTHONPATH \
      python /path/to/Segmentation/tools/test.py \
      /path/to/Segmentation/configs/beit/upernet/our_vit.py \
claim: "The segmentation (and analogous detection) evaluation scripts hardcode `/path/to/...` conda and external-codebase paths that do not exist in the repo; `check_repo_facts.py` lists all six `/path/to` occurrences."
concern: "These scripts cannot run as shipped; combined with the disabled `register.py` test commands, they are why Table 2 is unreproducible (cross-ref the owning `missing` finding)."
resolution: "Authors: parameterize these paths via config and document the required external Segmentation/Detection codebases and versions."
cross_refs: ["ade20k-coco-eval-missing"]
check_script: _audit_code/check_repo_facts.py
paper_ref: "Table 2"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: condition-only-head-tokens
category: methodology
topic: "unseen-task generalization design"
title: "Unseen-task condition is injected only into the last 8 tokens (the classifier head)"
severity: high
confidence: high
status: finding
file: model/__init__.py
line_start: 176
line_end: 191
quote: |
  to_condition_gate = torch.zeros(size=(1, sequence_length, 1))
  to_condition_gate[:, -8:, :] = 1.
  self.register_buffer("to_condition_gate", to_condition_gate)
  ...
  def _to_condition(self, x):
      assert len(x.shape) == 3
      x = self.to_condition_linear(x)
      x = x * self.to_condition_gate
      return x
claim: "In `ClassConditionMambaDiffusion` (the model used by `workspace/condition/generalization.py`) the per-task binary embedding is multiplied by a gate that is 1 only on the last 8 tokens of the sequence and 0 everywhere else, so the task condition can only modulate the final tokens — which correspond to the classifier head — while the backbone tokens are generated identically (position-embedding + a randomly-sampled permutation state) regardless of the task."
concern: "The headline 'generalizes to unseen tasks' claim (Tab.11/19) reduces to generating a task-specific 2-way head on top of an essentially task-agnostic backbone; high unseen-task accuracy is then largely attributable to the shared, well-trained backbone rather than to genuine parameter generalization, which the paper does not disentangle (no ablation generating the head only, or random-head baseline)."
resolution: "Authors: report a baseline that keeps the generated/average backbone but uses a randomly-initialized or trivially-trained head, and clarify whether the backbone meaningfully varies with the task condition; quantify how much of the unseen-task accuracy comes from the head vs the backbone."
cross_refs: ["main-experiments-reproduce-training-checkpoints"]
paper_ref: "Section 4.2; Tables 11, 19; Figure 4"
tags: [whalen:pitfall-2, reforms:6]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: main-experiments-reproduce-training-checkpoints
category: methodology
topic: "evaluation validity / claim scope"
title: "Main tables compare against the same checkpoints RPG was trained to reproduce"
severity: medium
confidence: high
status: finding
file: workspace/example/cifar10_resnet18.py
line_start: 98
line_end: 119
quote: |
  train_set = config["dataset"](dim_per_token=config["dim_per_token"])
  print("Dataset length:", train_set.real_length)
  print("input shape:", train_set[0][0].shape)
  if config["model_config"]["num_permutation"] == "auto":
      config["model_config"]["num_permutation"] = train_set.max_permutation_state
  if config["model_config"]["condition_dim"] == "auto":
      config["model_config"]["condition_dim"] = config["model_config"]["d_model"]
  if config["model_config"]["model_dim"] == "auto":
      config["model_config"]["model_dim"] = config["dim_per_token"]
  if config["sequence_length"] == "auto":
      config["sequence_length"] = train_set.sequence_length
      print(f"sequence length: {config['sequence_length']}")
  else:  # set fixed sequence_length
      assert train_set.sequence_length == config["sequence_length"], f"sequence_length={train_set.sequence_length}"
  train_loader = DataLoader(
claim: "For the main experiments the RPG training set is exactly the set of finetuned checkpoints of the target architecture/task (no condition input — `d_condition=1`, condition is zeros; generation varies only a randomly-sampled permutation-state embedding and diffusion noise, see `model/__init__.py:39-45`). The reported best/avg/min in Tables 1/2/3/10 are downstream accuracies of weights drawn from this same training distribution, on the same task's standard test set."
concern: "Tables 1–3/10 demonstrate that RPG can *reproduce* the performance of the checkpoints it was trained on, not that it generalizes; reported similarity ~0.84 (Tab.25/26) indicates the generated weights are close to the training ensemble, so 'on par with fully trained networks' is a memorization/reproduction result and should be scoped as such (the genuine generalization test is Section 4 — see `condition-only-head-tokens`)."
resolution: "Authors: state explicitly that Tables 1–3 measure in-distribution reproduction of the training checkpoints (no held-out tasks), and consider a held-out-checkpoint or held-out-seed evaluation to substantiate any generalization claim for these settings."
cross_refs: ["condition-only-head-tokens"]
paper_ref: "Section 3.2; Tables 1, 2, 3, 10; Appendix C.6/C.7"
tags: [whalen:pitfall-2, leakage:L1.1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 2          | high         | Table 2 (ADE20K/COCO) eval+checkpoint code absent/stubbed; requirements incomplete & unpinned. |
| bug         | 2          | low          | AR-ablation causal mask uses 1e-8 not -inf (leaks future); generalization.py double-steps scheduler / no optimizer.step. |
| difference  | 2          | low          | README points to a fully commented-out generalization driver; hardcoded `/path/to` in downtask shells. |
| methodology | 2          | high         | Unseen-task condition only modulates the head tokens; main tables reproduce the training checkpoints (no held-out tasks). |

## 5. Closing lists

### Top take-aways (≤6, ranked)
1. **[methodology, high]** `condition-only-head-tokens` — the unseen-task
   condition is gated to the last 8 tokens (the classifier head); the generated
   backbone is task-agnostic, so the headline "generalizes to unseen tasks"
   claim is largely a generated-head result on a shared backbone.
2. **[missing, high]** `ade20k-coco-eval-missing` — Table 2's segmentation /
   detection numbers have no runnable evaluation (stub "coming soon" commands,
   `/path/to` placeholders, no checkpoint-collection scripts).
3. **[methodology, medium]** `main-experiments-reproduce-training-checkpoints` —
   Tables 1–3/10 compare against the very checkpoints RPG was trained to
   reproduce; this is in-distribution reproduction, not generalization.
4. **[missing, medium]** `requirements-incomplete-unpinned` — numpy/pandas/
   matplotlib/tqdm unlisted and nothing pinned; environment not reconstructible
   from `requirements.txt`.
5. **[bug, low]** `causal-mask-leaks-future` — the AR-ablation causal mask is
   ineffective (fills with 1e-8 instead of -inf), confounding Table 6.
6. **[difference, low]** `readme-generalization-script-commented-out` — README
   points at a driver whose entire training loop is commented out.

### Items that genuinely look fine
- The recurrent-model, position-embedding, tokenization, token-size, AR, and
  recurrent-structure ablations (Tables 4–7) all have corresponding driver code.
- Published artefacts resolve: the HF model repo and the 68 GB `ViTTiny1022`
  condition dataset both exist (matching the README's "download about 68 GB").
- For ViT/ResNet/ConvNeXt the full `model.state_dict()` is saved and generated
  (e.g. `dataset/imagenet_vittiny/model.py`), supporting "entire model"; BN
  buffers / scalar params are copied verbatim through `postprocess`, which is a
  reasonable design choice rather than a defect.
- Downstream evaluation uses the standard timm/ImageNet (and CIFAR-10) test
  loaders with `shuffle=False`; no test-set tampering in the accuracy path.
- The unseen-task split (`split.sh`) holds out 20 distinct task embeddings and
  RPG is trained only on the remaining tasks (verified disjoint by
  `check_repo_facts.py`); the test labels are derived correctly from the task id.

### Open questions for the authors
- Table 18 (LMC) and Table 26 (with/without diffusion) toggles were not located
  as standalone driver scripts — were these produced by ad-hoc edits? (filed as
  table rows, not findings, per the no-extrapolation rule.)
- Table 10's SKDE30 / D2NWG / SANE comparison numbers: only p-diff and the
  "ours" baselines have code in `workspace/compare/`; how were the other three
  baselines obtained (cited vs reproduced)?
- For `generalization-scheduler-double-step`: was `optimizer.step()` actually
  present in the run that produced the Section-4 numbers?
