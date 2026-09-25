# Code-repository audit — DOVE: Efficient One-Step Diffusion Model for Real-World VSR (NeurIPS 2025, #2578)

## 1. Summary

The repo `code/zhengchen1999__DOVE/` is the authors' own DOVE code (README title, author list,
and BibTeX match the paper). It contains: training code for the two-stage latent–pixel strategy
(`finetune/train.py`, `finetune/trainer.py`, `finetune/models/dove/*`), the RealBasicVSR-style
degradation dataset (`finetune/datasets/*`), the one-step inference driver (`inference_script.py`,
`inference.sh`), a full-reference / no-reference metric script (`eval_metrics.py`, computing
PSNR/SSIM/LPIPS/DISTS/CLIP-IQA via `pyiqa`), and three video-quality metric wrappers
(`finetune/scripts/eval_dover.py`, `eval_ewarp.py`, `eval_vbench.py`). Pretrained weights and the
HQ-VSR dataset are hosted off-repo (Google Drive / Baidu, linked in README).

What I did: read the paper (PDF + text extraction) and every code file relevant to producing the
reported numbers; compared training config in `train_ddp_one_s1.sh` / `train_ddp_one_s2.sh` against
Section 4.1; traced each Table 1 / Table 2 / Table 3 metric to the code that computes it; and ran
three deterministic checks under `_audit_code/` (`check_inference_sh_gt.py`,
`check_missing_metric_deps.py`, `check_dover_offbyone.py`). The core training/inference math
(v-prediction one-step denoise at t=399, MSE latent loss, MSE+DISTS+frame-difference pixel loss,
φ=0.8 image ratio, λ1=λ2=1) faithfully matches the paper. The main problems are on the *evaluation*
side: three of the eight reported metrics (FasterVQA, DOVER, Ewarp) have no runnable computation in
the repo, the released `inference.sh` feeds the wrong ground truth to five of six datasets, and the
video-processing pipeline that builds the headline HQ-VSR dataset (Sec 3.3 / Fig 3, a stated
contribution) is absent.

## 2. Result-traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Tab.2 PSNR / SSIM | `eval_metrics.py:109-152` (pyiqa psnr/ssim) | yes | n/a (no weights run) | Code present |
| Tab.2 LPIPS / DISTS / CLIP-IQA | `eval_metrics.py:109-152` (pyiqa) | yes | n/a | Code present |
| Tab.2 / Tab.1 **FasterVQA** | (none) | **no** | — | MISSING (no FasterVQA code anywhere; README TODO) |
| Tab.2 / Tab.1 / Tab.3 **DOVER** | `finetune/scripts/eval_dover.py:150` imports `DOVER` pkg | **not runnable** | — | MISSING dep (`DOVER` package absent + per-sample off-by-one) |
| Tab.2 **E\*warp** | `finetune/scripts/eval_ewarp.py:147-150` imports `ewarp.Ewarp` from `RAFT/` | **not runnable** | — | MISSING module/dir (`ewarp.py`, `finetune/scripts/RAFT/`, model ckpt) |
| Tab.2 numbers for SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ via `inference.sh` | `inference.sh:22-75` | runs but wrong GT | — | BUG (GT hardcoded to UDM10) |
| Tab.1a/b/c/d ablations (PSNR/LPIPS/CLIP-IQA/DOVER on UDM10) | training shells + eval scripts | partial | n/a | DOVER column not runnable (see above) |
| Tab.1c/d **HQ-VSR construction** (+Filter, +Motion) | (none) | **no** | — | MISSING (video processing pipeline not released) |
| Tab.3 running time (s) | (none — no timing harness) | no | — | Not reproducible from repo (no timing script) |
| Stage-1/2 training config (10k/500 iters, lr, φ, λ) | `finetune/train_ddp_one_s1.sh`, `train_ddp_one_s2.sh`, `lora_one_s{1,2}_trainer.py` | yes | matches §4.1 | Verified faithful |

## 3. Findings

## missing

```yaml finding
id: fastervqa-no-code
category: missing
topic: "result traceability / evaluation metrics"
title: "FasterVQA metric (every Table 1/2 row) has no computation code in the repo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 269
line_end: 269
quote: |
  > **TODO:** Add metric computation scripts for FasterVQA, DOVER, and $E^*_{warp}$.
claim: "FasterVQA is reported for DOVE and all baselines in Table 1 (ablations) and Table 2 (every dataset), but no script, function, or dependency in the repo computes FasterVQA; the README explicitly lists it as a TODO, and a tree-wide search finds 'fastervqa' only in README.md (see _audit_code/out/missing_metric_deps.json)."
concern: "A headline perceptual metric in every comparison table cannot be reproduced from the released code, so the FasterVQA numbers in Tab.1/Tab.2 are not traceable to any computation."
resolution: "Authors: release the FasterVQA evaluation script (model weights, preprocessing, and the exact invocation), or clarify which external tool/version produced the FasterVQA columns."
cross_refs: ["dover-dep-missing", "ewarp-module-missing"]
check_script: _audit_code/check_missing_metric_deps.py
paper_ref: "Table 1, Table 2 (FasterVQA rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-dep-missing
category: missing
topic: "result traceability / evaluation metrics"
title: "DOVER evaluation imports a DOVER package that is not in the repo or requirements"
severity: high
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 150
line_end: 154
quote: |
      from DOVER.evaluate_a_set_of_videos import evaluate_set as dover

      input_path = os.path.abspath(input_path)

      dover_results = dover(input_path, device=device)
claim: "eval_dover.py computes the DOVER metric (used in Tab.1, Tab.2 and Tab.3) by importing `DOVER.evaluate_a_set_of_videos.evaluate_set`, but no `DOVER` package/directory exists in the repo and `DOVER` is not in requirements.txt (see _audit_code/out/missing_metric_deps.json: DOVER_package_dir=false)."
concern: "The DOVER column appears in Tab.1, the entire Tab.2 DOVER row, and the Tab.3 'Performance' column, yet the metric is uncomputable from the repo as shipped because its core dependency is absent and unpinned."
resolution: "Authors: vendor or pin the exact DOVER package (commit + weights) and document its installation, so the DOVER scores can be reproduced."
cross_refs: ["fastervqa-no-code", "dover-per-sample-offbyone"]
check_script: _audit_code/check_missing_metric_deps.py
paper_ref: "Table 2 DOVER row; Table 3 Performance column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ewarp-module-missing
category: missing
topic: "result traceability / evaluation metrics"
title: "E*warp script imports an 'ewarp' module and RAFT/ directory that do not exist"
severity: medium
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
claim: "eval_ewarp.py computes the temporal-consistency metric E*warp (Tab.2) by chdir-ing into `<scripts>/RAFT` and importing `ewarp.Ewarp`, but the repo has no `finetune/scripts/RAFT/` directory and no `ewarp.py` defining `Ewarp` (grep 'def Ewarp' returns nothing; only `finetune/utils/RAFT/` exists). The default RAFT weight path `finetune/scripts/models/raft-things.pth` (line 190) is also absent. See _audit_code/out/missing_metric_deps.json."
concern: "The E*warp row in Table 2 cannot be produced from the repo: the script's RAFT directory, the `ewarp` module, and the checkpoint path it points to are all missing."
resolution: "Authors: release the `ewarp.py`/`Ewarp` implementation and the `RAFT/` folder (with `raft-things.pth`) under `finetune/scripts/`, or fix the import to point at the existing `finetune/utils/RAFT/`."
cross_refs: ["fastervqa-no-code", "dover-dep-missing"]
check_script: _audit_code/check_missing_metric_deps.py
paper_ref: "Table 2 E*warp row"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hqvsr-pipeline-missing
category: missing
topic: "data construction / reproducibility"
title: "HQ-VSR video-processing pipeline (Sec 3.3, a stated contribution) is not in the repo"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 80
line_end: 80
quote: |
  - [ ] Release the video processing pipeline.
claim: "Section 3.3 / Fig. 3 present a four-step pipeline (metadata filter, scene filter, quality filter with CLIP-IQA/FasterVQA/DOVER thresholds, motion-area detection/cropping) used to build the 2,055-video HQ-VSR dataset, and Table 1d ablates +Filter / +Motion. No code in the repo implements scene detection, the quality-metric thresholding, or the motion-mask/bounding-box cropping of Eq.(8); the README lists this pipeline as an unchecked TODO (see _audit_code/out/missing_metric_deps.json for the metric-script searches)."
concern: "HQ-VSR construction is one of the paper's three contributions and underlies the Tab.1c/Tab.1d dataset ablations, but the pipeline that produces it is absent, so the dataset and the +Filter/+Motion ablation cannot be reproduced (only the final dataset blob is downloadable)."
resolution: "Authors: release the video-processing pipeline scripts (scene split, quality thresholds, motion-area detection per Eq.(8)) so HQ-VSR and the Tab.1d ablation can be reconstructed."
cross_refs: []
paper_ref: "Section 3.3, Figure 3, Table 1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: timing-harness-missing
category: missing
topic: "result traceability / efficiency claim"
title: "No timing/efficiency script for the headline 28x speed-up (Table 3, Fig 1)"
severity: low
confidence: medium
status: question
file: paper.pdf
quote: |
  "Our method is approximately 28× faster than MGLD-VSR [50]. ... For fairness, all methods are measured running time on the same A100 GPU, generating a 33-frame 720×1280 video."
claim: "Table 3 / Fig. 1 report per-method running times (e.g., DOVE 14.90 s vs MGLD-VSR 425.23 s) measured on one A100 with a 33-frame 720x1280 video, but the repo contains no timing harness that measures or logs inference latency under those conditions."
concern: "The central efficiency claim (28x speed-up) has no accompanying measurement script, so the reported times are not reproducible from the repo."
resolution: "Authors: provide the timing script (warm-up, frame count, resolution, device, and how baseline times were obtained) used for Table 3 / Fig. 1."
cross_refs: []
paper_ref: "Table 3; Figure 1 (Time)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: inference-sh-gt-mismatch
category: bug
topic: "evaluation driver"
title: "inference.sh evaluates 5 of 6 datasets against UDM10's ground truth"
severity: high
confidence: high
status: finding
file: inference.sh
line_start: 22
line_end: 50
quote: |
  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/SPMCS \
      --metrics psnr,ssim,lpips,dists,clipiqa

  # YouHQ40
  python inference_script.py \
      --input_dir datasets/test/YouHQ40/LQ-Video \
      --model_path pretrained_models/DOVE \
      --output_path results/DOVE/YouHQ40 \
      --is_vae_st \

  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/YouHQ40 \
      --metrics psnr,ssim,lpips,dists,clipiqa

  # RealVSR
  python inference_script.py \
      --input_dir datasets/test/RealVSR/LQ-Video \
      --model_path pretrained_models/DOVE \
      --output_path results/DOVE/RealVSR \
      --is_vae_st \
      --upscale 1 \

  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/RealVSR \
      --metrics psnr,ssim,lpips,dists,clipiqa
claim: "For SPMCS, YouHQ40, RealVSR, MVSR4x and VideoLQ the eval step is hardcoded to `--gt datasets/test/UDM10/GT` instead of the matching dataset's GT (5/6 blocks; only the first UDM10 block is correct). Verified by parsing inference.sh: see _audit_code/out/inference_sh_gt.json (n_mismatch=5)."
concern: "As shipped, the full-reference Tab.2 numbers (PSNR/SSIM/LPIPS/DISTS) for five datasets would be computed against UDM10 ground truth; since eval_metrics.py matches predictions to GT by filename (`eval_metrics.py:174`), non-matching names are silently skipped and the reproduction script cannot regenerate Tab.2 for those datasets."
resolution: "Authors: fix the `--gt` path in each block to the corresponding `datasets/test/<NAME>/GT`, and confirm the published Tab.2 numbers were computed against the correct per-dataset GT."
cross_refs: []
check_script: _audit_code/check_inference_sh_gt.py
paper_ref: "Table 2 (SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-per-sample-offbyone
category: bug
topic: "evaluation metrics"
title: "eval_dover.py shifts per-video DOVER scores by one (results[name]=dover_results[i-1])"
severity: low
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 156
line_end: 159
quote: |
      results = {}

      for i, name in enumerate(pred_names):
          results[name] = dover_results[i-1]
claim: "Per-video DOVER scores are assigned with `dover_results[i-1]`, so the first sorted name gets the LAST video's score and every name is mapped to the previous video's score. The reported average is unaffected because indices {-1,0,...,count-2} still cover all videos (verified in _audit_code/out/dover_offbyone.json: mean_unchanged=true, per_sample_matches_correct=false)."
concern: "Any per-clip DOVER analysis or per-sample JSON written by this script is mislabeled; only the headline mean DOVER is unaffected, so this does not by itself invalidate Tab.2/Tab.3 DOVER averages."
resolution: "Change `dover_results[i-1]` to `dover_results[i]` so per-sample scores align with names."
cross_refs: ["dover-dep-missing"]
check_script: _audit_code/check_dover_offbyone.py
paper_ref: "Table 2 DOVER row"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: requirements-missing-pyiqa
category: difference
topic: "dependencies / environment"
title: "pyiqa (the metric backend) is required by README but absent from requirements.txt"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 21
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
claim: "All metric computation (`eval_metrics.py`, `inference_script.py`, training perceptual loss in `trainer.py:424-431`) depends on `pyiqa`, but `pyiqa` is not listed in requirements.txt; it is only installed via a separate `pip install pyiqa` line in the README (line 99). The version is unpinned, so the exact PSNR/SSIM/LPIPS/DISTS/CLIP-IQA implementations are not reproducible from requirements.txt alone (see _audit_code/out/missing_metric_deps.json: pyiqa_in_requirements=false)."
concern: "Reproducing the reported full-reference and CLIP-IQA numbers requires an unpinned external metric library not captured in the dependency spec, which can change metric definitions across versions."
resolution: "Authors: add `pyiqa` (pinned to the version used) to requirements.txt."
cross_refs: []
check_script: _audit_code/check_missing_metric_deps.py
paper_ref: "Evaluation Metrics (§4.1)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: udm10-validation-and-ablation-overlap
category: methodology
topic: "model selection / evaluation"
title: "UDM10 is both the in-training validation monitor and the ablation benchmark"
severity: low
confidence: low
status: question
file: finetune/train_ddp_one_s1.sh
line_start: 64
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
      --eval_metric_list "psnr,ssim,lpips,dists,clipiqa"
claim: "Both training stages run validation every 500 steps on `../datasets/test/UDM10` (the same set used as the Table 1 ablation benchmark, paper §4.2: 'We evaluate all models on UDM10'). Checkpoints are saved at fixed step intervals (`trainer.py:1002-1014`), not selected by validation metric, so the validation here is logging-only and does not automatically pick the reported checkpoint."
concern: "Because UDM10 metrics are monitored live during training and UDM10 is also the ablation test set, manual design/iteration decisions (e.g., 10k/500-iteration budget, φ=0.8) could be informed by UDM10 performance; there is no separate held-out validation split distinct from the ablation benchmark."
resolution: "Authors: confirm whether any architecture/hyperparameter/iteration-count choice was made by observing UDM10 validation metrics, and report ablations on a validation set disjoint from the UDM10 evaluation set."
cross_refs: []
paper_ref: "Section 4.2 (ablations on UDM10)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | FasterVQA/DOVER/Ewarp metric code + HQ-VSR pipeline + timing harness not in repo |
| bug         | 2          | high         | inference.sh uses UDM10 GT for 5/6 datasets; DOVER per-sample off-by-one (mean ok) |
| difference  | 1          | low          | pyiqa (metric backend) missing from requirements.txt |
| methodology | 1          | low          | UDM10 is both live validation monitor and ablation benchmark (logging-only; question) |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `fastervqa-no-code`** — FasterVQA (every Tab.1/Tab.2 row) has zero computation code; README lists it as TODO. (high/high)
2. **[bug] `inference-sh-gt-mismatch`** — the reproduction script `inference.sh` evaluates SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ against UDM10's ground truth. (high/high)
3. **[missing] `dover-dep-missing`** — DOVER metric (Tab.1/2/3) imports a `DOVER` package absent from repo and unpinned. (high/high)
4. **[missing] `ewarp-module-missing`** — E*warp (Tab.2) imports an `ewarp` module and `RAFT/` dir + checkpoint that do not exist. (medium/high)
5. **[missing] `hqvsr-pipeline-missing`** — the HQ-VSR construction pipeline (Sec 3.3, a contribution; Tab.1d ablation) is not released. (medium/high)
6. **[difference] `requirements-missing-pyiqa`** — pyiqa, the backend for every reported metric and the training perceptual loss, is not in requirements.txt. (low/high)

### Items that genuinely look fine
- One-step inference math: v-prediction denoise at t=399 with `scheduler.get_velocity` (`inference_script.py:459-503`, `lora_one_s1_trainer.py:167-209`) matches Eq.(2).
- Stage-1 latent MSE loss (`lora_one_s1_trainer.py:207`) and Stage-2 MSE + DISTS + frame-difference loss (`lora_one_s2_trainer.py:238-290`) match Eqs.(3),(4),(6),(7).
- Training config matches §4.1: stage-1 10k iters @ lr 2e-5 (`train_ddp_one_s1.sh`), stage-2 500 iters @ lr 5e-6 with φ=0.8, λ1=λ2=1 (`train_ddp_one_s2.sh`, args defaults in `schemas/args.py`).
- Train (HQ-VSR/DIV2K) and test (UDM10/SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ) sets are disjoint datasets; per-sample synthetic degradation (`finetune/datasets/real_sr_dataset.py`) follows RealBasicVSR — no train/test sample leakage.
- PSNR/SSIM/LPIPS/DISTS/CLIP-IQA are computed by a present, correct `pyiqa`-based script (`eval_metrics.py`); optional Y-channel conversion uses standard BT.601 limited-range coefficients (`eval_metrics.py:20-24`).

### Open questions for the authors
- `timing-harness-missing`: which script measured the Table 3 / Fig. 1 running times and how were the baseline times obtained?
- `udm10-validation-and-ablation-overlap`: were any design choices (iteration budget, φ) selected on UDM10 validation, given UDM10 is also the ablation benchmark?
- For `inference-sh-gt-mismatch`: were the published Tab.2 numbers computed with the correct per-dataset GT (i.e., is the bug only in the released convenience script)?
