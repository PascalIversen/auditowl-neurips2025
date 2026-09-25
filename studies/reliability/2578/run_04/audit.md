# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (paper 2578)

## 1. Summary

The repo (`code/zhengchen1999__DOVE/`) is the authors' official DOVE code (README title,
author list, and NeurIPS-2025 citation all match `paper.pdf`). It contains a one-step VSR
inference script (`inference_script.py`) built on CogVideoX1.5-5B, a metric script
(`eval_metrics.py`), an inference shell driver (`inference.sh`), and a `finetune/` package
with the two-stage training code (Stage-1 latent MSE, Stage-2 pixel MSE+DISTS+frame-diff),
the RealBasicVSR/RealESRGAN degradation configs, and several `finetune/scripts/eval_*.py`
metric drivers. Pretrained weights and the HQ-VSR / test datasets are hosted on Google Drive
(not in the repo).

What I did:
- Read the paper (PDF + `paper_text.txt`) to enumerate the headline artefacts: Table 2
  (8 metrics × 7 datasets), Table 1 (4 ablation sub-tables), Table 3 (running time / DOVER),
  Fig. 1 (DOVER / FasterVQA / Time / E*warp on VideoLQ), and the §4.1 training settings.
- Read every Python/shell file in the repo and matched the training code (`lora_one_s1_trainer.py`,
  `lora_one_s2_trainer.py`) against the loss equations (3)–(7) and §4.1 hyperparameters.
- Wrote `_audit_code/check_eval_scripts.py` (deterministic: file/module existence, import
  targets, and a parse of every `--gt`/`--pred` pair in `inference.sh`). Output in
  `_audit_code/out/eval_scripts_check.csv`. I could not *run* the model (no GPU / no weights /
  no datasets in the sandbox), so all metric-value matches are traced structurally, not numerically.

The training code is faithful to the paper's method and reads as methodologically sound (latent
MSE in Stage-1; MSE+DISTS+frame-difference in Stage-2; φ=0.8, λ1=λ2=1; standard real-world
degradation). The defects are concentrated in the **evaluation/reproduction path**: three of the
eight metrics reported in Table 2 (FasterVQA, DOVER, E*warp) have no working compute path in the
repo, and the provided `inference.sh` driver points every dataset's metric step at the UDM10 GT.

## 2. Traceability table

Datasets: U=UDM10, SP=SPMCS, Y=YouHQ40, RV=RealVSR, MV=MVSR4x, VL=VideoLQ.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Tab. 2 PSNR / SSIM (U,SP,Y,RV,MV) | `eval_metrics.py` (pyiqa psnr/ssim) | not run (no GPU/weights) | — | Script present; numeric match not verifiable here |
| Tab. 2 LPIPS / DISTS / CLIP-IQA (all) | `eval_metrics.py` (pyiqa lpips/dists/clipiqa) | not run | — | Script present; numeric match not verifiable here |
| Tab. 2 FasterVQA (all 7 datasets) | **(none)** | — | — | **MISSING** — no script computes FasterVQA (see `fastervqa-no-compute-script`) |
| Tab. 2 DOVER (all 7 datasets) | `finetune/scripts/eval_dover.py` | — | — | **MISSING dependency** — imports absent `DOVER` package (see `dover-package-not-in-repo`); also off-by-one (see `dover-result-index-off-by-one`) |
| Tab. 2 E*warp (all 7 datasets) | `finetune/scripts/eval_ewarp.py` | — | — | **MISSING module** — imports absent `ewarp.py`/`Ewarp` (see `ewarp-module-missing`) |
| Fig. 1 DOVER/FasterVQA/E*warp on VideoLQ | same as Tab.2 rows above | — | — | Same MISSING status as Tab. 2 |
| Tab. 3 DOVER on VideoLQ | `finetune/scripts/eval_dover.py` | — | — | MISSING dependency (cross-ref `dover-package-not-in-repo`) |
| Tab. 3 running time (s) | (none) | — | — | No timing harness in repo (not separately flagged; benchmark-config dependent) |
| `inference.sh` reproduction of Tab. 2 | `inference.sh` | — | — | **BUG** — every dataset's eval uses `--gt datasets/test/UDM10/GT` (see `inference-sh-wrong-gt-dir`) |
| Tab. 1a–d ablations | training code + `eval_metrics.py` | not run | — | Training code present & faithful; ablation eval reuses same metric path (PSNR/LPIPS/CLIP-IQA/DOVER → DOVER again MISSING) |
| §4.1 Stage-1: 25×320×640, 10k it, lr 2e-5 | `finetune/train_ddp_one_s1.sh` | matches | ✓ | Verified (res/steps/lr match) |
| §4.1 Stage-2: φ=0.8, lr 5e-6, 500 it, λ1=λ2=1 | `finetune/train_ddp_one_s2.sh` | matches | ✓ | Verified (`image_ratio 0.8`, `dists_weight 1.0`, `frame_diff_weight 1.0`) |
| §4.1 "4 A800 GPUs, total batch size 8" | `finetune/accelerate_config.yaml` + shell | 2 GPUs × bs 2 = 4 | ✗ | **DIFFERENCE** (low) — shipped config is 2 GPUs / effective batch 4 (see `released-config-2gpu-batch4`) |
| §3.1 single step, t=399 | `inference_script.py` / trainers (`sr_noise_step=399`) | matches | ✓ | Verified |

## 3. Findings

## missing

```yaml finding
id: fastervqa-no-compute-script
category: missing
topic: "result traceability / evaluation metrics"
title: "No script computes FasterVQA, a metric reported for all 7 datasets in Table 2"
severity: high
confidence: high
status: finding
file: README.md
line_start: 269
line_end: 269
quote: |
  > **TODO:** Add metric computation scripts for FasterVQA, DOVER, and $E^*_{warp}$.
claim: "FasterVQA is reported in Table 2 (every dataset) and Fig. 1, but no Python file in the repo computes it; the README itself lists FasterVQA computation as an unfinished TODO. A repo-wide search for 'faster' in .py files returns no metric code (only this README line)."
concern: "A headline metric appearing in every row of the main quantitative table has no computation path in the released code, so those values cannot be reproduced from the repo."
resolution: "Authors: release the FasterVQA evaluation script (model weights, preprocessing, and call site) used to produce the Table 2 / Fig. 1 numbers."
cross_refs: ["dover-package-not-in-repo", "ewarp-module-missing"]
check_script: _audit_code/check_eval_scripts.py
paper_ref: "Table 2 (FasterVQA rows); Fig. 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ewarp-module-missing
category: missing
topic: "result traceability / temporal-consistency metric"
title: "E*warp script imports an 'ewarp' module that does not exist in the repo"
severity: high
confidence: high
status: finding
file: finetune/scripts/eval_ewarp.py
line_start: 146
line_end: 150
quote: |
      original_dir = os.getcwd()
      raft_dir = os.path.join(original_dir, "RAFT")
      os.chdir(raft_dir)
      from ewarp import Ewarp as Ewarp
      results, avg_score = Ewarp(args)
claim: "eval_ewarp.py chdir's to ./RAFT (i.e. finetune/scripts/RAFT) and does `from ewarp import Ewarp`, but no `ewarp.py` exists anywhere in the repo, the `finetune/scripts/RAFT` directory does not exist (RAFT lives at finetune/utils/RAFT), and the default `--model finetune/scripts/models/raft-things.pth` path is also absent (the checkpoint is at finetune/utils/RAFT/raft-things.pth)."
concern: "E*warp is reported for every dataset in Table 2 and in Fig. 1, but the only script that would compute it cannot run because the module that actually computes the value (Ewarp) is missing and the RAFT directory/model paths it assumes do not exist."
resolution: "Authors: add the `ewarp.py` (Ewarp implementation) and fix the RAFT directory / checkpoint paths, so E*warp in Table 2 / Fig. 1 is reproducible."
cross_refs: ["fastervqa-no-compute-script"]
check_script: _audit_code/check_eval_scripts.py
paper_ref: "Table 2 (E*warp rows); Fig. 1; §4.1 (E*warp definition)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-package-not-in-repo
category: missing
topic: "result traceability / video-quality metric"
title: "DOVER metric depends on an external 'DOVER' package absent from repo and requirements"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 150
line_end: 154
quote: |
      from DOVER.evaluate_a_set_of_videos import evaluate_set as dover

      input_path = os.path.abspath(input_path)

      dover_results = dover(input_path, device=device)
claim: "eval_dover.py imports `DOVER.evaluate_a_set_of_videos`, but there is no `DOVER` package/directory in the repo and `DOVER` is not listed in requirements.txt; the import will fail at runtime."
concern: "DOVER is a headline VQA metric in Table 2 (all datasets), Table 3, and Fig. 1, yet the script that would compute it cannot import its core dependency from the released code, so the DOVER values are not reproducible from the repo alone."
resolution: "Authors: vendor or pin the exact DOVER implementation (and its weights/config) used, or add install/setup instructions to requirements/README so DOVER eval is runnable."
cross_refs: ["fastervqa-no-compute-script", "dover-result-index-off-by-one"]
check_script: _audit_code/check_eval_scripts.py
paper_ref: "Table 2 (DOVER rows); Table 3; Fig. 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: inference-sh-wrong-gt-dir
category: bug
topic: "evaluation reproduction driver"
title: "inference.sh evaluates every dataset against UDM10 ground truth"
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
claim: "The reproduction driver inference.sh hardcodes `--gt datasets/test/UDM10/GT` in the eval step for SPMCS, YouHQ40, RealVSR, MVSR4x, and VideoLQ (only the UDM10 block uses the matching GT). The shown SPMCS block evaluates SPMCS predictions against UDM10 ground truth."
concern: "Anyone running the provided script to reproduce Table 2 computes full-reference metrics (PSNR/SSIM/LPIPS/DISTS) for 5 of 6 datasets against the wrong ground truth; eval_metrics.py skips predictions whose filenames are absent from the UDM10 GT folder (eval_metrics.py:174), so those datasets' fidelity numbers come out empty or against mismatched clips."
resolution: "Fix each eval block to use the dataset's own GT (e.g. `--gt datasets/test/SPMCS/GT`); confirm whether the Table 2 numbers were produced with corrected paths."
cross_refs: []
check_script: _audit_code/check_eval_scripts.py
paper_ref: "Table 2; README 'Testing' section"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-result-index-off-by-one
category: bug
topic: "evaluation metric indexing"
title: "eval_dover.py assigns DOVER scores with an off-by-one index (i-1)"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
      for i, name in enumerate(pred_names):
          results[name] = dover_results[i-1]
claim: "Per-video DOVER scores are mapped to filenames with `dover_results[i-1]`, so the first sorted name (i=0) receives `dover_results[-1]` (the last video's score) and every name is shifted by one relative to `dover_results`."
concern: "Per-sample DOVER values are misattributed by one position; this does not change the mean (a cyclic permutation), so the averaged Table-2/Table-3 DOVER number is unaffected, but per-sample DOVER outputs are wrong and any per-clip analysis would be misleading."
resolution: "Use `results[name] = dover_results[i]` and confirm whether `evaluate_set` returns scores in the same sorted order as `pred_names`."
cross_refs: ["dover-package-not-in-repo"]
check_script: _audit_code/check_eval_scripts.py
paper_ref: "Table 2 (DOVER); Table 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: released-config-2gpu-batch4
category: difference
topic: "training configuration"
title: "Shipped accelerate config uses 2 GPUs / effective batch 4, vs paper's 4 GPUs / batch 8"
severity: low
confidence: high
status: finding
file: finetune/accelerate_config.yaml
line_start: 3
line_end: 4
quote: |
  gpu_ids: "0, 1"
  num_processes: 2  # should be the same as the number of GPUs
claim: "The released accelerate config launches 2 processes (2 GPUs) and both train shell scripts set `--batch_size 2 --gradient_accumulation_steps 1`, giving a global batch size of 4. The paper (§4.1) states training used '4 NVIDIA A800-80G GPUs with the total batch size 8'."
concern: "The shipped configuration does not match the paper's reported training setup (batch size 4 vs 8 / 2 GPUs vs 4); reproducing with the default files would not match the paper's effective batch size, which can affect the trained weights."
resolution: "Authors: confirm the GPU count / batch configuration used for the released checkpoints, or update accelerate_config.yaml / the shell scripts to the paper's 4-GPU, batch-8 setting."
cross_refs: []
paper_ref: "§4.1 Implementation Details ('4 NVIDIA A800-80G GPUs with the total batch size 8')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The training procedure the code actually implements is sound and
faithful to the paper: Stage-1 minimises MSE between predicted and HR latents in latent space
(`lora_one_s1_trainer.py:206-207`, matching Eq. 3); Stage-2 applies MSE + DISTS perceptual +
frame-difference L1 with φ=0.8 and λ1=λ2=1 (`lora_one_s2_trainer.py:238-290`, matching Eqs.
4–7); the degradation pipeline is a standard two-stage RealBasicVSR/RealESRGAN scheme. VSR has no
classification-style train/test-leakage surface, and evaluation is on standard held-out public
benchmarks; no leakage, label-construction, or baseline-fairness issue was identified.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | FasterVQA / E*warp / DOVER (3 of 8 Table-2 metrics) have no working compute path in the repo |
| bug         | 2          | high         | inference.sh uses UDM10 GT for all datasets; eval_dover index off-by-one |
| difference  | 1          | low          | shipped accelerate config = 2 GPUs / batch 4 vs paper's 4 GPUs / batch 8 |
| methodology | 0          | -            | training method is faithful and sound; no leakage/baseline issues |

## 5. Closing lists

**Top take-aways** (≤6, ranked by severity × confidence):
1. `[missing]` FasterVQA has no computation script at all — reported in every Table 2 row + Fig. 1 (`fastervqa-no-compute-script`).
2. `[missing]` E*warp script imports a non-existent `ewarp`/`Ewarp` module (RAFT dir/model paths also wrong) (`ewarp-module-missing`).
3. `[bug]` `inference.sh` evaluates SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ against UDM10 GT (`inference-sh-wrong-gt-dir`).
4. `[missing]` DOVER eval needs an external `DOVER` package absent from repo and requirements (`dover-package-not-in-repo`).
5. `[bug]` `eval_dover.py` misattributes per-sample DOVER scores via `dover_results[i-1]` (mean unaffected) (`dover-result-index-off-by-one`).
6. `[difference]` shipped accelerate config = 2 GPUs / batch 4 vs paper's 4 GPUs / batch 8 (`released-config-2gpu-batch4`).

**Items that genuinely look fine** (actively checked):
- Stage-1 latent MSE loss matches Eq. (3) (`lora_one_s1_trainer.py:206-207`); even with `--is_latent false` the loss is still computed in latent space by encoding LR+HR.
- Stage-2 loss = MSE + DISTS + frame-difference L1 matches Eqs. (4)–(7); frame-diff = L1 of temporal forward differences (`lora_one_s2_trainer.py:283-285`).
- Stage-2 shell hyperparameters match §4.1: `image_ratio 0.8`, `dists_weight 1.0`, `frame_diff_weight 1.0`, lr 5e-6, 500 steps.
- Stage-1 shell hyperparameters match §4.1: 25×320×640, 10000 steps, lr 2e-5, sr_noise_step 399.
- `prepare_sft_ckpt.py` output naming (`/ckpt-...-sft`) is consistent with the Stage-2 `--model_path checkpoint/DOVE-s1/ckpt-10000-sft` (no path inconsistency).
- PSNR/SSIM via `eval_metrics.py` use the standard pyiqa metrics; the script supports Y-channel/border-crop options (defaults to RGB, no crop) — under-specified vs paper but both are valid (not flagged).
- Single-step inference at t=399 (`sr_noise_step` default 399) matches §3.1; the `decode_latents → *0.5+0.5` output range handling is consistent.
- RealBasicVSR/RealESRGAN two-stage degradation config present and consistent with §4.1.

**Open questions for the authors**:
- Which exact FasterVQA / DOVER / E*warp implementations (and weights) produced the Table 2 / Fig. 1 numbers, and can they be released or pinned?
- Were the Table 2 fidelity numbers produced with corrected per-dataset GT paths (i.e. is `inference.sh`'s UDM10-GT-for-all a copy-paste slip in the published driver only)?
- Was the released checkpoint trained with the paper's 4-GPU / batch-8 setup, or with the shipped 2-GPU / batch-4 config?
