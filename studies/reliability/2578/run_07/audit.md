# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (#2578)

## 1. Summary

The repository `code/zhengchen1999__DOVE/` is the authors' official DOVE code (README,
citation, abstract, and asset figures all match the NeurIPS 2025 paper). DOVE is a
one-step VSR model obtained by SFT-finetuning CogVideoX1.5-5B; the contributions are
(a) a two-stage "latent–pixel" training strategy and (b) a curated HQ-VSR dataset built
by a four-step video processing pipeline.

I read the paper (`paper.pdf` + `paper_text.txt`) and the full code tree. I verified:
the inference pipeline (`inference_script.py`), the metric scripts (`eval_metrics.py`,
`finetune/scripts/eval_{metrics,dover,vbench,ewarp}.py`, `finetune/utils/metric_utils.py`),
the training entry points (`finetune/train.py`, `finetune/trainer.py`,
`finetune/train_ddp_one_s1.sh`, `finetune/train_ddp_one_s2.sh`), the Stage-2 loss
(`finetune/models/dove/lora_one_s2_trainer.py`), the degradation pipeline
(`finetune/datasets/degradation.py`), `requirements.txt`, and the checkpoint-prep helper
(`finetune/scripts/prepare_sft_ckpt.py`). I ran three read-only checks under
`_audit_code/` (results in `_audit_code/out/`):
`check_missing_metric_deps.py`, `check_dover_offbyone.py`, `check_inference_sh_gt.py`.

Headline result: training code, the regression+DISTS+frame-diff losses, φ, learning
rates, step counts, and degradation pipeline all match the paper. The reproduction gaps
are concentrated in the **evaluation metrics**: three of the paper's reported metrics
(FasterVQA, DOVER, E*_warp) have no working in-repo computation, and the driver shell
script for Table 2 passes the wrong ground-truth folder for every non-UDM10 dataset.

## 2. Traceability table (Rule G)

Paper artefacts vs. the code that *computes* them. Pretrained weights / HQ-VSR / test
sets are hosted on Google Drive (external; not verifiable offline in this sandbox).

| Paper artefact (Tab. 2 / Tab. 1 / Tab. 3 / Fig. 1) | Repo location | Computes value? | Status |
|---|---|---|---|
| PSNR, SSIM (fidelity) | `eval_metrics.py` (pyiqa) | yes | Present |
| LPIPS, DISTS, CLIP-IQA (perceptual) | `eval_metrics.py` (pyiqa) | yes | Present |
| **FasterVQA** (Tab. 2 all datasets, ablations, Fig. 1) | (none) | no | MISSING (no code at all; README "TODO") |
| **DOVER** (Tab. 2 all datasets, Tab. 3, Fig. 1) | `finetune/scripts/eval_dover.py` | imports `DOVER.*` not in repo; off-by-one name map | MISSING dep + BUG |
| **E\*_warp** (Tab. 2 all datasets) | `finetune/scripts/eval_ewarp.py` | imports `ewarp.Ewarp` not in repo; wrong RAFT paths | MISSING dep + BUG |
| VBench (paper does not report; helper present) | `finetune/scripts/eval_vbench.py` | imports `evaluate.calculate_final` not in repo | MISSING dep (not paper-reported) |
| Tab. 2 reproduction driver | `inference.sh` | uses UDM10 GT for SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ | BUG (wrong GT) |
| Running time / 28× speed-up (Tab. 3, Fig. 1) | (none) | no timing script | Not computed (timing is measured, lower priority) |
| HQ-VSR dataset (2,055 videos) construction | (none) | pipeline code not released | MISSING (README "TODO: Release the video processing pipeline") |
| Stage-1 latent MSE loss (Eq. 3) | `finetune/models/dove/lora_one_s1_trainer.py` | yes | Present |
| Stage-2 MSE+DISTS+frame loss (Eq. 4/6/7), φ=0.8, λ1=λ2=1 | `lora_one_s2_trainer.py` + `train_ddp_one_s2.sh` | yes | Present, matches paper |

## 3. Findings

## missing

```yaml finding
id: fastervqa-no-code
category: missing
topic: "result traceability / evaluation metrics"
title: "FasterVQA metric reported throughout the paper has no computation code in the repo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 269
line_end: 269
quote: |
  > **TODO:** Add metric computation scripts for FasterVQA, DOVER, and $E^*_{warp}$.
claim: "FasterVQA is reported for DOVE and all baselines in Tab. 2 (every dataset), in the ablation-adjacent discussion, and as a headline axis in Fig. 1, yet no script, function, or import in the repo computes FasterVQA — a repo-wide search (`_audit_code/check_missing_metric_deps.py`) finds zero Python references; the README explicitly lists it under TODO."
concern: "A headline quantitative metric in the main comparison table cannot be reproduced from the released code, so the FasterVQA rows in Tab. 2 and the FasterVQA axis in Fig. 1 are untraceable."
resolution: "Authors: please release the FasterVQA evaluation script (model checkpoint, preprocessing, and the exact command), or state the external tool/version used."
cross_refs: ["dover-ewarp-missing-deps"]
check_script: _audit_code/check_missing_metric_deps.py
paper_ref: "Table 2 (FasterVQA rows), Figure 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-ewarp-missing-deps
category: missing
topic: "result traceability / evaluation metrics"
title: "DOVER and E*_warp eval scripts import packages/files absent from the repo"
severity: high
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 150
line_end: 150
quote: |
    from DOVER.evaluate_a_set_of_videos import evaluate_set as dover
claim: "eval_dover.py imports `DOVER.evaluate_a_set_of_videos` and eval_ewarp.py imports `ewarp.Ewarp` (line 149); neither the `DOVER` package, the `ewarp.py` module, nor a `VBench` package exists anywhere in the repo (verified by `_audit_code/check_missing_metric_deps.py`: all four directory/file checks return False, and no `ewarp.py` exists in the tree)."
concern: "DOVER (Tab. 2 all datasets, Tab. 3 'Performance' column, Fig. 1) and E*_warp (Tab. 2 temporal-consistency rows) cannot be computed from the released code because the metric backends are not provided and are unlisted as dependencies; the README marks these scripts as TODO."
resolution: "Authors: vendor or pin the exact DOVER and RAFT-based Ewarp implementations (repo/commit) and provide the `ewarp.py`/`DOVER` modules, or document the external repos and versions used."
cross_refs: ["fastervqa-no-code", "ewarp-broken-paths"]
check_script: _audit_code/check_missing_metric_deps.py
paper_ref: "Table 2 (DOVER, E*_warp rows), Table 3, Figure 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hqvsr-pipeline-not-released
category: missing
topic: "dataset construction"
title: "HQ-VSR video processing pipeline (a stated contribution) is not in the repo"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 80
line_end: 80
quote: |
  - [ ] Release the video processing pipeline.
claim: "The paper's second contribution is the four-step video processing pipeline (Sec. 3.3, Fig. 3, Eq. 8) used to build HQ-VSR; the README's TODO list confirms the pipeline code is not released (only the final 2,055-video dataset is linked on Google Drive), and no metadata/scene/quality/motion-cropping script exists in the repo (grep over `finetune/` finds the degradation pipeline but no S1–S4 filtering/optical-flow motion-area code)."
concern: "A claimed methodological contribution (the curation pipeline, including the motion-area bounding-box algorithm in Eq. 8) cannot be reproduced or inspected, and Tab. 1d's '+Filter'/'+Motion' ablation cannot be re-run."
resolution: "Authors: release the metadata/scene/quality/motion-processing scripts and their thresholds (Aesthetic>6.5, CLIP-IQA>0.4, FasterVQA>0.6, DOVER>0.7, τ, padding p)."
cross_refs: []
paper_ref: "Section 3.3, Figure 3, Table 1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: requirements-missing-core-deps
category: missing
topic: "dependencies / environment"
title: "requirements.txt omits diffusers, pyiqa, and safetensors required by the scripts"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 20
quote: |
  accelerate>=1.1.1
  transformers>=4.46.2
  numpy==1.26.0
  torch>=2.5.0
  torchvision>=0.20.0
  sentencepiece>=0.2.0
  SwissArmyTransformer>=0.4.12
  gradio>=5.5.0
  imageio>=2.35.1
  imageio-ffmpeg>=0.5.1
  openai>=1.54.0
  moviepy>=2.0.0
  scikit-video>=1.1.11
  pydantic>=2.10.3
  wandb
  peft
  opencv-python
  decord
  av
  torchdiffeq
claim: "`inference_script.py` imports `diffusers`, `safetensors`, and `pyiqa`; `eval_metrics.py` imports `pyiqa`; none of `diffusers`, `safetensors`, or `pyiqa` is listed (pinned or unpinned) in requirements.txt — they are only mentioned as separate ad-hoc `pip install` lines in the README."
concern: "`pip install -r requirements.txt` alone does not produce a runnable environment, and the unpinned `diffusers` version is critical because the model relies on version-specific diffusers internals (CogVideoXDPMScheduler.get_velocity, get_3d_rotary_pos_embed signatures)."
resolution: "Authors: add diffusers (pinned), safetensors, and pyiqa to requirements.txt and pin the diffusers version used."
cross_refs: []
paper_ref: "N/A"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: inference-sh-wrong-gt
category: bug
topic: "evaluation driver / Table 2 reproduction"
title: "inference.sh evaluates every dataset against UDM10's ground truth"
severity: high
confidence: high
status: finding
file: inference.sh
line_start: 22
line_end: 25
quote: |
  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/SPMCS \
      --metrics psnr,ssim,lpips,dists,clipiqa
claim: "The provided reproduction script `inference.sh` hard-codes `--gt datasets/test/UDM10/GT` for SPMCS, YouHQ40, RealVSR, and MVSR4x (and VideoLQ) instead of each dataset's own GT folder (verified: 5 of 6 eval_metrics calls mismatch, `_audit_code/check_inference_sh_gt.py`). eval_metrics.py matches predictions to GT by basename and silently skips predictions with no matching GT name (eval_metrics.py:174-176)."
concern: "Running the shipped script reproduces the full-reference metrics (PSNR/SSIM/LPIPS/DISTS) for SPMCS/YouHQ40/RealVSR/MVSR4x against the wrong (UDM10) GT — names will not match, so those FR metrics are skipped or computed against unrelated frames, so the script as-is does not reproduce the Tab. 2 FR numbers for those datasets."
resolution: "Authors: fix each block's `--gt` to point at the corresponding `datasets/test/<NAME>/GT`."
cross_refs: []
check_script: _audit_code/check_inference_sh_gt.py
paper_ref: "Table 2 (SPMCS, YouHQ40, RealVSR, MVSR4x rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-score-name-misalign
category: bug
topic: "evaluation metrics / DOVER"
title: "eval_dover.py assigns each video the previous video's DOVER score (off-by-one)"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
      for i, name in enumerate(pred_names):
          results[name] = dover_results[i-1]
claim: "DOVER scores are mapped to video names with index `i-1`, so the first sorted video name receives `dover_results[-1]` (the last score), the second receives `dover_results[0]`, etc. — a cyclic shift verified in `_audit_code/check_dover_offbyone.py` (every name gets a wrong score)."
concern: "All per-sample DOVER values in the output JSON are attached to the wrong video; the reported per-dataset average is unaffected only if `dover()` returns scores in the same sorted order (then the shift is a permutation), but any per-clip analysis or non-sorted return order is corrupted."
resolution: "Authors: change to `results[name] = dover_results[i]` and confirm `evaluate_set` returns scores aligned to the sorted input order."
cross_refs: ["dover-ewarp-missing-deps"]
check_script: _audit_code/check_dover_offbyone.py
paper_ref: "Table 2 (DOVER rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ewarp-broken-paths
category: bug
topic: "evaluation metrics / E*_warp"
title: "eval_ewarp.py chdir's to a nonexistent RAFT dir and defaults to a wrong model path"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_ewarp.py
line_start: 146
line_end: 152
quote: |
    original_dir = os.getcwd()
    raft_dir = os.path.join(original_dir, "RAFT")
    os.chdir(raft_dir)
    from ewarp import Ewarp as Ewarp
    results, avg_score = Ewarp(args)
    os.chdir(original_dir)
claim: "The script `os.chdir`'s to its own dir (`finetune/scripts`) at line 7, then builds `raft_dir = <finetune/scripts>/RAFT` and chdir's into it before `from ewarp import Ewarp`; but RAFT actually lives at `finetune/utils/RAFT/` and contains no `ewarp.py` (verified `_audit_code/check_missing_metric_deps.py`). The default `--model finetune/scripts/models/raft-things.pth` (line 190) also does not exist; the checkpoint is at `finetune/utils/RAFT/raft-things.pth`."
concern: "Even if the missing `ewarp.py` were supplied, the script would crash on the chdir to a nonexistent `finetune/scripts/RAFT` and on the wrong default RAFT checkpoint path, so the E*_warp temporal-consistency numbers (Tab. 2) are not reproducible as shipped."
resolution: "Authors: provide `ewarp.py`, point `raft_dir` and the default `--model` at `finetune/utils/RAFT/`, and verify the script runs end-to-end."
cross_refs: ["dover-ewarp-missing-deps"]
check_script: _audit_code/check_missing_metric_deps.py
paper_ref: "Table 2 (E*_warp rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: undocumented-extra-loss-options
category: difference
topic: "training loss / paper omission"
title: "Code contains edge-aware DISTS, LPIPS, and GAN/optical-flow loss options not in the paper"
severity: low
confidence: medium
status: question
file: finetune/models/dove/lora_one_s2_trainer.py
line_start: 245
line_end: 264
quote: |
            if self.args.ea_dists_weight > 0:
                dists_loss = self.dists_loss(pred_frame, gt_frame)
                edge_loss = self.dists_loss(
                    self.edge_detection_model(pred_frame), 
                    self.edge_detection_model(gt_frame)
                )
                perceptual_loss = perceptual_loss + dists_loss + edge_loss
            elif self.args.dists_weight > 0:
                dists_loss = self.dists_loss(pred_frame, gt_frame)
                perceptual_loss = perceptual_loss + dists_loss
            elif self.args.ea_lpips_weight > 0:
                lpips_loss = self.lpips_loss(pred_frame, gt_frame)
                edge_loss = self.lpips_loss(
                    self.edge_detection_model(pred_frame), 
                    self.edge_detection_model(gt_frame)
                )
                perceptual_loss = perceptual_loss + lpips_loss + edge_loss
            elif self.args.lpips_weight > 0:
                lpips_loss = self.lpips_loss(pred_frame, gt_frame)
                perceptual_loss = perceptual_loss + lpips_loss
claim: "The Stage-2 trainer exposes loss variants the paper never mentions: an edge-aware DISTS (`ea_dists_weight`, using a Sobel `EdgeDetectionModel`), LPIPS and edge-aware LPIPS, and GAN-related args (`gen_cls_loss_weight`, `diffusion_gan_max_timestep`) plus optical-flow flags in args.py. The shipped `train_ddp_one_s2.sh` sets only `--dists_weight 1.0` and `--frame_diff_weight 1.0`, which matches the paper's Eq. (4)/(7)."
concern: "These are unused-by-default alternative loss paths; they do not change the released training recipe, but their presence is undocumented in the paper and could matter if reviewers assume the released defaults are the only configuration used."
resolution: "Authors: confirm the paper results use only plain DISTS + frame-diff (the shell defaults) and that edge-aware/LPIPS/GAN paths were exploratory."
cross_refs: []
paper_ref: "Section 3.2, Eq. (4), Eq. (7)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

```yaml finding
id: udm10-validation-and-benchmark
category: methodology
topic: "evaluation / model selection"
title: "UDM10 serves as both the training-time validation set and a reported benchmark/ablation set"
severity: low
confidence: low
status: question
file: finetune/train_ddp_one_s1.sh
line_start: 62
line_end: 73
quote: |
  VALIDATION_ARGS=(
      --do_validation true  # ["true", "false"]
      --validation_dir "../datasets/test/UDM10"
      --validation_steps 500  # should be multiple of checkpointing_steps
      --validation_videos "LQ-Video.txt"
      --validation_ref_videos "GT-Video.txt"
      # --validation_prompts "prompts.txt"
      --gen_fps 8
      --raw_test true
      --num_inference_steps 1
      --eval_metric_list "psnr,ssim,lpips,dists,clipiqa"  # ["psnr", "ssim", "lpips", "dists", "clipiqa", "musiq", "maniqa", 'niqe']
  )
claim: "Both training shells set the validation set to `../datasets/test/UDM10`, the same UDM10 used as a reported benchmark in Tab. 2 and as the sole evaluation set for all ablations in Tab. 1; validation metrics are logged during training. Checkpoints are saved at fixed `checkpointing_steps` (trainer.py:1002-1014), not chosen by best validation metric, so there is no automated test-loss leakage."
concern: "Because UDM10 is monitored during training and is also the ablation/benchmark set, manual design or checkpoint choices could be informed by UDM10 performance; with only fixed-interval checkpointing this is a soft concern rather than a demonstrated leak."
resolution: "Authors: confirm UDM10 was not used to pick checkpoints/hyperparameters, or report the held-out validation set used for selection."
cross_refs: []
paper_ref: "Table 1 (ablations on UDM10), Table 2 (UDM10 benchmark)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 4          | high         | FasterVQA has no code; DOVER/E*_warp/VBench backends absent; HQ-VSR pipeline unreleased; core deps unlisted |
| bug         | 3          | high         | inference.sh uses UDM10 GT for all datasets; DOVER off-by-one name map; eval_ewarp wrong RAFT/model paths |
| difference  | 1          | low          | Undocumented edge-aware/LPIPS/GAN loss options (unused by default) — question |
| methodology | 1          | low          | UDM10 is both validation and reported benchmark/ablation set — question |

## 5. Closing lists

**Top take-aways (≤6, severity × confidence):**
- [missing] FasterVQA — a Tab. 2/Fig. 1 headline metric — has zero computation code in the repo (`fastervqa-no-code`).
- [missing] DOVER and E*_warp eval scripts import packages/files (`DOVER`, `ewarp.py`) that are not in the repo (`dover-ewarp-missing-deps`).
- [bug] `inference.sh` evaluates SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ against UDM10's GT, so the shipped Tab. 2 driver does not reproduce those FR numbers (`inference-sh-wrong-gt`).
- [bug] `eval_dover.py` maps each video to the previous video's DOVER score (off-by-one) (`dover-score-name-misalign`).
- [bug] `eval_ewarp.py` chdir's into a nonexistent RAFT directory and defaults to a wrong RAFT checkpoint path (`ewarp-broken-paths`).
- [missing] The HQ-VSR video processing pipeline (a stated contribution, Sec. 3.3) is not released (`hqvsr-pipeline-not-released`).

**Items that genuinely look fine:**
- Stage-2 loss exactly implements MSE + DISTS + frame-difference (Eq. 4/6/7); shell defaults φ=0.8, λ1=λ2=1, lr 5e-6, 500 steps match Sec. 4.1.
- Stage-1 config (10,000 steps, lr 2e-5, 25×320×640, batch 2 ×4 GPUs = total 8) matches Sec. 4.1.
- The `rgb_to_y` luma coefficients (0.257/0.504/0.098 + 0.0625) are the standard BT.601 Y' used in SR PSNR/SSIM — correct.
- `prepare_sft_ckpt.py` produces `ckpt-10000-sft`, exactly the path `train_ddp_one_s2.sh` consumes — consistent.
- PSNR/SSIM/LPIPS/DISTS/CLIP-IQA are computed via pyiqa in `eval_metrics.py` and match the metrics named in Sec. 4.1.
- The RealBasicVSR/Real-ESRGAN-style degradation pipeline (`degradation.py`) is present and complete.

**Open questions for the authors:**
- Were UDM10 validation logs ever used to select checkpoints/hyperparameters? (`udm10-validation-and-benchmark`)
- Do the paper's results use only the default DISTS+frame-diff loss, with the edge-aware/LPIPS/GAN paths being exploratory? (`undocumented-extra-loss-options`)
- Note: HQ-VSR, test sets, and pretrained weights are hosted on external Google Drive links that cannot be verified offline in this sandbox; their existence/integrity is out of scope here.
