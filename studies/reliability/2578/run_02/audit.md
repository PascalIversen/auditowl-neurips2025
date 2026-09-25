# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (paper 2578)

## Summary

The repo (`code/zhengchen1999__DOVE/`) is the genuine author code for the NeurIPS-2025
paper DOVE. Its README matches the paper title, abstract, figures (Strategy, Pipeline,
Quantitative/Qualitative panels reproduce Tab. 2 / Fig. 4), and citation. It contains:
inference (`inference_script.py`, `inference.sh`), full-reference / IQA metric computation
(`eval_metrics.py` via `pyiqa`), a two-stage training harness (`finetune/train.py`,
`finetune/models/dove/*_s1_*` and `*_s2_*` trainers, `train_ddp_one_s1.sh`,
`train_ddp_one_s2.sh`), degradation configs, RAFT weights, and a `requirements.txt`.

What I did. I read the paper (PDF + text extraction) and mapped every quantitative
artefact (Tab. 1a–d, Tab. 2, Tab. 3, Fig. 1) to repo code. I verified the training losses
against the paper's equations (Stage-1 latent MSE = Eq. 3; Stage-2 = MSE + DISTS + frame-diff
= Eq. 4/7), the timestep choice (t=399), the empty-prompt design, the φ=0.8 image ratio, and
the RealBasicVSR/RealESRGAN degradation. I then wrote four deterministic checks under
`_audit_code/`: (1) existence of the modules the VQA/E*_warp eval scripts import
(`check_missing_eval_deps.py`); (2) the DOVER per-sample off-by-one
(`check_dover_offbyone.py`); (3) the `--gt` path each `inference.sh` eval call uses
(`check_inference_sh_gt.py`); (4) the padding-removal vs `--upscale` arithmetic
(`check_pad_upscale_mismatch.py`). All outputs are in `_audit_code/out/`.

Headline reproducibility picture. The training/loss code is faithful to the method.
However, two of the paper's contributions/results are not fully reproducible from the repo:
(a) the **video processing pipeline** that constructs HQ-VSR (Sec. 3.3 / Fig. 3 / Tab. 1c–1d)
is entirely absent (README marks it TODO); (b) three of the eight reported metrics
(FasterVQA, DOVER, E*_warp) have no working computation path — FasterVQA has no script at all,
and the provided `eval_dover.py` / `eval_ewarp.py` import packages/modules and a directory
that do not exist in the repo. PSNR/SSIM/LPIPS/DISTS/CLIP-IQA are computed by `eval_metrics.py`
and are runnable given the released weights + datasets.

## Traceability table

| Paper artefact | Repo location | Computed? | Status |
|---|---|---|---|
| Tab. 2 PSNR/SSIM/LPIPS/DISTS/CLIP-IQA (DOVE col, all datasets) | `eval_metrics.py` (pyiqa) + `inference_script.py` | yes (needs released weights/data) | Present; runnable |
| Tab. 2 / Fig. 1 FasterVQA | (none) | no | MISSING (no FasterVQA script; README TODO L269) |
| Tab. 2 / Tab. 3 / Fig. 1 DOVER | `finetune/scripts/eval_dover.py` | broken | BUG: imports absent `DOVER` pkg; per-sample off-by-one |
| Tab. 2 E*_warp | `finetune/scripts/eval_ewarp.py` | broken | BUG: chdir to non-existent `scripts/RAFT`; imports absent `ewarp` |
| Tab. 1a ablation (S1 / S2-I / S2-I/V) | `train.py` + s1/s2 trainers | partial | needs full retrain; no driver enumerating the 3 variants |
| Tab. 1b ablation (image ratio φ) | s2 trainer `--image_ratio` | partial | sweep not scripted; manual reruns |
| Tab. 1c ablation (training dataset: YouHQ/OpenVid/HQ-VSR) | (none for HQ-VSR construction) | no | MISSING (HQ-VSR pipeline absent) |
| Tab. 1d ablation (+Filter / +Motion pipeline) | (none) | no | MISSING (pipeline code absent) |
| Tab. 3 running time (28× speedup) | (no timing harness) | no | not scripted; numbers not reproducible from repo |
| inference.sh non-UDM10 eval rows | `inference.sh` L23/35/48/61/73 | broken | BUG: `--gt` hardcoded to UDM10 for every dataset |

## missing

```yaml finding
id: video-pipeline-absent
category: missing
topic: "result traceability / dataset construction"
title: "HQ-VSR video processing pipeline (Sec 3.3, Fig 3, Tab 1c/1d) not in repo"
severity: high
confidence: high
status: finding
file: code/zhengchen1999__DOVE/README.md
line_start: 80
line_end: 80
quote: |
  - [ ] Release the video processing pipeline.
claim: "The four-step pipeline (metadata/scene/quality filtering + optical-flow motion-area detection and cropping, Eq. 8) that builds the 2,055-video HQ-VSR dataset is not present anywhere in the repo; grep for motion_mask/bounding_box/aesthetic/scenedetect/optical-flow-score returns nothing, and the README TODO marks it unreleased."
concern: "HQ-VSR construction is the paper's second headline contribution and underpins Tab. 1c (dataset ablation) and Tab. 1d (+Filter/+Motion ablation); without the pipeline these ablation results cannot be reproduced and the dataset cannot be regenerated."
resolution: "Release the pipeline scripts (metadata/scene/quality filtering, motion-area detection/cropping) or the exact configs and thresholds (Aesthetic>6.5, CLIP-IQA>0.4, FasterVQA>0.6, DOVER>0.7, τ, padding p) used to produce HQ-VSR."
cross_refs: ["§3.3", "Tab. 1c", "Tab. 1d"]
check_script: _audit_code/check_missing_eval_deps.py
paper_ref: "Section 3.3 (Video Processing Pipeline); Table 1c/1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fastervqa-script-missing
category: missing
topic: "result traceability / metrics"
title: "No FasterVQA computation script for Tab. 2 / Fig. 1 FasterVQA column"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/README.md
line_start: 269
line_end: 269
quote: |
  > **TODO:** Add metric computation scripts for FasterVQA, DOVER, and $E^*_{warp}$.
claim: "FasterVQA is reported in Tab. 2 and Fig. 1, but no script computes it: a repo-wide grep for 'fastervqa'/'faster_vqa' over all .py files returns no hits (see _audit_code/out/missing_eval_deps.json: fastervqa_script_present=false)."
concern: "A reported metric column has no traceable computation in the repo, so those FasterVQA numbers cannot be reproduced from the released code; the authors acknowledge this in the README TODO."
resolution: "Add the FasterVQA evaluation script (or document the exact external repo/commit and command used)."
cross_refs: ["Tab. 2", "Fig. 1"]
check_script: _audit_code/check_missing_eval_deps.py
paper_ref: "Table 2 (FasterVQA rows); Figure 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: dover-eval-imports-absent-package
category: bug
topic: "metrics / DOVER evaluation"
title: "eval_dover.py imports a DOVER package that does not exist in the repo"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 150
line_end: 150
quote: |
    from DOVER.evaluate_a_set_of_videos import evaluate_set as dover
claim: "After os.chdir(script_directory) (the script's own dir, finetune/scripts/), the script imports DOVER.evaluate_a_set_of_videos, but no 'DOVER' package/directory exists anywhere in the repo (see _audit_code/out/missing_eval_deps.json: DOVER_pkg_anywhere=false)."
concern: "The DOVER metric reported throughout Tab. 2/Tab. 3/Fig. 1 cannot be computed with the shipped code as-is; the script raises ModuleNotFoundError on import."
resolution: "Vendor the DOVER package, add it as a pinned dependency with the expected import path, or document the external DOVER repo/commit and how to place it on sys.path."
cross_refs: ["Tab. 2", "Tab. 3", "fastervqa-script-missing"]
check_script: _audit_code/check_missing_eval_deps.py
paper_ref: "Table 2 (DOVER rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ewarp-eval-wrong-raft-path-and-missing-module
category: bug
topic: "metrics / temporal consistency E*_warp"
title: "eval_ewarp.py chdir's to non-existent scripts/RAFT and imports absent ewarp"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_ewarp.py
line_start: 147
line_end: 149
quote: |
    raft_dir = os.path.join(original_dir, "RAFT")
    os.chdir(raft_dir)
    from ewarp import Ewarp as Ewarp
claim: "original_dir is finetune/scripts/ (set by os.chdir(script_directory) at the top), so raft_dir resolves to finetune/scripts/RAFT, which does not exist (RAFT lives at finetune/utils/RAFT); furthermore no 'ewarp.py' exists anywhere in the repo (see _audit_code/out/missing_eval_deps.json: RAFT_under_scripts=false, RAFT_under_utils=true, ewarp_module_anywhere=false)."
concern: "The E*_warp temporal-consistency metric reported in Tab. 2 cannot be computed with the shipped code; the chdir raises FileNotFoundError and the import would raise ModuleNotFoundError."
resolution: "Point raft_dir at finetune/utils/RAFT (or the correct location) and add the missing ewarp.py implementation, or document the external source for both."
cross_refs: ["Tab. 2"]
check_script: _audit_code/check_missing_eval_deps.py
paper_ref: "Table 2 (E*_warp rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: inference-sh-gt-hardcoded-udm10
category: bug
topic: "evaluation harness"
title: "inference.sh evaluates every dataset against UDM10 ground truth"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/inference.sh
line_start: 22
line_end: 25
quote: |
  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/SPMCS \
      --metrics psnr,ssim,lpips,dists,clipiqa
claim: "All six eval_metrics.py calls pass --gt datasets/test/UDM10/GT while --pred varies (SPMCS, YouHQ40, RealVSR, MVSR4x, VideoLQ); 5/6 calls pair non-UDM10 predictions with UDM10 GT (see _audit_code/out/inference_sh_gt.json)."
concern: "eval_metrics.py skips any prediction whose basename has no matching GT filename (process(): 'if has_gt and name_hr not in gt_files: continue'), so running inference.sh as written would compute full-reference metrics for the wrong/empty GT set on 5 of 6 datasets, not reproducing the Tab. 2 fidelity columns."
resolution: "Set --gt to the matching dataset's GT folder in each block (e.g. datasets/test/SPMCS/GT)."
cross_refs: ["Tab. 2"]
check_script: _audit_code/check_inference_sh_gt.py
paper_ref: "Table 2 (per-dataset PSNR/SSIM/LPIPS/DISTS)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-per-sample-offbyone
category: bug
topic: "metrics / DOVER per-sample mapping"
title: "eval_dover.py maps DOVER scores to names with an off-by-one (i-1)"
severity: low
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
    for i, name in enumerate(pred_names):
        results[name] = dover_results[i-1]
claim: "The loop assigns dover_results[i-1] to pred_names[i]; for i=0 this is dover_results[-1] (the last score), shifting every per-sample DOVER score by one name (see _audit_code/out/dover_offbyone.json: 4/4 mismatched, first_name_gets_last_score=true)."
concern: "Per-sample DOVER values written to metrics_dover.json are attributed to the wrong videos; the overall average (np.mean over results.values(), L164) is unaffected since it is the same multiset, so reported Tab. 2 DOVER averages are not corrupted by this alone."
resolution: "Index with dover_results[i] instead of dover_results[i-1]; confirm len(dover_results)==len(pred_names) and the ordering returned by evaluate_set matches sorted(pred_names)."
cross_refs: ["dover-eval-imports-absent-package"]
check_script: _audit_code/check_dover_offbyone.py
paper_ref: "Table 2 (DOVER rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pad-removal-hardcoded-x4
category: bug
topic: "inference / spatial padding"
title: "SR-output padding removed as pad*4 regardless of --upscale (breaks upscale=1)"
severity: low
confidence: medium
status: finding
file: code/zhengchen1999__DOVE/inference_script.py
line_start: 731
line_end: 731
quote: |
            video_generate = remove_padding_and_extra_frames(video_generate, pad_f, pad_h*4, pad_w*4)
claim: "pad_h/pad_w are LR-pixel pads making H/W multiples of 16 (lines 228-229); the LR is then upscaled by args.upscale (line 672), so the SR padding region is pad*args.upscale, but removal hardcodes pad*4. inference.sh runs RealVSR and MVSR4x with --upscale 1 (lines 45, 58), where for non-multiple-of-16 input dims the code over-crops by 3*pad pixels (see _audit_code/out/pad_upscale_mismatch.json)."
concern: "For RealVSR/MVSR4x (upscale=1) with input width/height not divisible by 16, the SR output is over-cropped on the right/bottom edge, slightly shifting the evaluated region and the Tab. 2 fidelity metrics for those two real-world datasets."
resolution: "Use pad_h*args.upscale and pad_w*args.upscale in remove_padding_and_extra_frames; confirm whether RealVSR/MVSR4x inputs are multiples of 16 (if always, impact is nil)."
cross_refs: ["Tab. 2"]
check_script: _audit_code/check_pad_upscale_mismatch.py
paper_ref: "Table 2 (RealVSR, MVSR4x rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: div2k-image-count-900-vs-800
category: difference
topic: "training data description"
title: "Paper says DIV2K = 900 images; README/standard DIV2K_train_HR = 800"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  The image dataset is DIV2K [3], with 900 images, which follows the RealESRGAN [38] degradation process.
claim: "The paper (Sec 4.1) states 900 DIV2K images; the repo README lists DIV2K-HR with 800 images and links DIV2K_train_HR.zip (the standard DIV2K training split is exactly 800; 900 = 800 train + 100 val)."
concern: "Minor inconsistency in the reported image-dataset size between paper and released code; does not affect headline conclusions but the exact training-set composition for Stage-2 is ambiguous."
resolution: "Clarify whether 800 or 900 DIV2K images were used for Stage-2 (and whether the 100 DIV2K validation images were included)."
cross_refs: ["§4.1"]
paper_ref: "Section 4.1 (Datasets)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The implemented procedure is methodologically sound: Stage-1 latent
MSE (Eq. 3), Stage-2 pixel MSE + DISTS + L1 frame-difference (Eq. 4/7), t=399 single-step
denoising (Eq. 2), φ=0.8 mixed image/video sampling, and RealBasicVSR/RealESRGAN degradation
all match the paper. Evaluation uses standard FR/NR-IQA via pyiqa on a held-out set of
established benchmarks (UDM10/SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ) distinct from the training
sources; there is no train/test leakage concern for an SR task evaluated on separate benchmark
datasets. Baselines (RealESRGAN, ResShift, RealBasicVSR, UAV, MGLD-VSR, VEnhancer, STAR) are
prior published methods.

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 2          | high         | HQ-VSR video pipeline (Tab 1c/1d, contribution) and FasterVQA script absent |
| bug         | 4          | medium       | DOVER/E*_warp eval scripts import absent code; inference.sh GT hardcoded to UDM10 |
| difference  | 1          | low          | DIV2K image count 900 (paper) vs 800 (README)                          |
| methodology | 0          | -            | Training/eval procedure is sound and faithful to the paper             |

## Top take-aways

1. **[missing]** The HQ-VSR video processing pipeline (Sec 3.3 / Fig 3 / Tab 1c–1d), a headline
   contribution, is entirely absent from the repo (README TODO). `video-pipeline-absent` (high).
2. **[bug]** `eval_dover.py` imports a `DOVER` package that does not exist in the repo, so the
   DOVER metric in Tab. 2/3/Fig. 1 is not computable as shipped. `dover-eval-imports-absent-package` (medium).
3. **[bug]** `eval_ewarp.py` chdir's to a non-existent `finetune/scripts/RAFT` and imports an
   absent `ewarp` module, so E*_warp (Tab. 2) is not computable as shipped.
   `ewarp-eval-wrong-raft-path-and-missing-module` (medium).
4. **[bug]** `inference.sh` passes `--gt datasets/test/UDM10/GT` for all six datasets, so running
   it would evaluate SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ predictions against UDM10 GT and skip
   the unmatched samples. `inference-sh-gt-hardcoded-udm10` (medium).
5. **[missing]** No FasterVQA computation script anywhere in the repo (acknowledged TODO).
   `fastervqa-script-missing` (medium).
6. **[bug]** `inference_script.py` removes padding as `pad*4` regardless of `--upscale`,
   over-cropping RealVSR/MVSR4x (run at `--upscale 1`) when input dims are not multiples of 16.
   `pad-removal-hardcoded-x4` (low).

## Items that genuinely look fine

- Stage-1 latent MSE loss (`lora_one_s1_trainer.py:207`) matches Eq. 3.
- Stage-2 loss = MSE + DISTS + L1 frame-difference (`lora_one_s2_trainer.py:238,253,285,290`)
  matches Eq. 4/7; weights dists=1.0, frame_diff=1.0 (λ1=λ2=1) set in `train_ddp_one_s2.sh:93-94`.
- Single-step denoising at t=399 via v-prediction `get_velocity` (`inference_script.py:459-493`,
  `sr_noise_step` default 399 in `schemas/args.py:95`) matches Eq. 2 and the paper's t=399 choice.
- Empty-prompt pre-encoding optimization (`inference_script.py:580-590`) matches the README/paper.
- φ=0.8 mixed image/video sampling (`lora_one_s2_trainer.py:125`, `train_ddp_one_s2.sh:33`)
  matches Tab. 1b's chosen ratio.
- RealBasicVSR two-stage degradation incl. video compression (`configs/degradation.yaml`)
  matches the paper's stated degradation pipeline.
- PSNR/SSIM/LPIPS/DISTS/CLIP-IQA computation via pyiqa (`eval_metrics.py`) is present and runnable.
- Hardware/optimizer details (4 GPUs, total batch 8, AdamW, lr 2e-5 / 5e-6, 10k / 500 steps)
  in the train shells match Sec 4.1.

## Open questions for the authors

- Are the ordering/length guarantees of `evaluate_set` in `eval_dover.py` such that the per-sample
  DOVER mapping was wrong in the released numbers, or were Tab. 2 DOVER values computed off-repo?
- Were RealVSR/MVSR4x inputs always multiples of 16 (which would make `pad-removal-hardcoded-x4`
  benign), or could the over-crop have shifted those two datasets' fidelity metrics?
- Which DIV2K count (800 vs 900) and which split were actually used for Stage-2 training?
