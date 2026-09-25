# Code-repository audit — DOVE (NeurIPS 2025, paper #2578)

## 1. Summary

The repository `code/zhengchen1999__DOVE/` is the authors' code for *DOVE: Efficient
One-Step Diffusion Model for Real-World Video Super-Resolution*. Its README, citation
block, and assets match the paper, so it is the author code. DOVE fine-tunes CogVideoX1.5-5B
for one-step VSR using a two-stage latent→pixel strategy. The repo contains: an inference
entrypoint (`inference_script.py`, `inference.sh`), a full-reference metric script
(`eval_metrics.py`), a training stack under `finetune/` (two-stage SFT/LoRA trainers,
RealBasicVSR/RealESRGAN degradation configs, dataset code, RAFT optical-flow weights), and
helper eval scripts (`finetune/scripts/eval_{dover,ewarp,vbench,metrics}.py`).

What I did (read-only):
- Read the paper (PDF + text), README, requirements, training shell scripts, args schema,
  both stage trainers, the inference script, and all eval scripts.
- Confirmed the implementation matches the paper's headline hyperparameters
  (t=399; AdamW β1/β2/β3 = 0.9/0.95/0.98; stage-1 lr 2e-5 / 10k steps / 25×320×640;
  stage-2 lr 5e-6 / 500 steps / image-ratio φ=0.8; λ1=λ2=1; empty-prompt; ×4 bilinear).
- Verified the loss code implements Eq. 3 (latent MSE), Eq. 4 (pixel MSE + DISTS), and
  Eq. 6/7 (frame-difference L1 loss).
- Ran `_audit_code/check_missing_metric_code.py`: a deterministic, read-only audit of which
  Table-2 metrics have working computation code and whether the eval-script imports/paths
  resolve. Output: `_audit_code/out/missing_metric_code.csv`.

Headline findings: the **video-processing pipeline (Sec 3.3 / Fig 3 / Table 1d)** has no code
(README TODO confirms it is unreleased); the **E\*warp** metric script imports a module
(`ewarp`) that does not exist; **FasterVQA** has no computation code at all; **DOVER** is only
reachable via an un-vendored external package and contains a per-sample off-by-one. The IQA
fidelity/perceptual metrics (PSNR/SSIM/LPIPS/DISTS/CLIP-IQA) are fully traceable.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Tab 2 PSNR / SSIM / LPIPS / DISTS / CLIP-IQA (all 6 datasets) | `eval_metrics.py` (pyiqa) | yes | n/a (no weights run) | Verified-present |
| Tab 2 / Tab 1 / Tab 3 / Fig 1 **DOVER** | `finetune/scripts/eval_dover.py:150` (imports external `DOVER`) | only via un-vendored pkg; per-sample i-1 mislabel | n/a | PARTIAL / BUG |
| Tab 2 / Fig 1 **FasterVQA** | (none — README TODO line 269) | no | — | MISSING |
| Tab 2 / Fig 1 **E\*warp** | `finetune/scripts/eval_ewarp.py:149` (`from ewarp import Ewarp`) | no — module absent | — | MISSING (broken import) |
| Sec 3.3 / Fig 3 / Tab 1d video-processing pipeline (filter + motion-area crop) | (none — README TODO line 80) | no | — | MISSING |
| Tab 1a/b/c/d ablation drivers (UDM10) | trainer flags exist; no per-row harness | inference+eval re-runnable manually | — | Partial |
| Tab 3 timings (14.90 s; "28× faster") | `inference_script.py` (no timing harness) | not scripted | — | Not-scripted (runtime) |
| Stage-1 loss Eq. 3 (latent MSE) | `finetune/models/dove/lora_one_s1_trainer.py:207` | yes | ✓ | Verified |
| Stage-2 loss Eq. 4 (MSE+DISTS) | `finetune/models/dove/lora_one_s2_trainer.py:238-277` | yes | ✓ | Verified |
| Stage-2 frame loss Eq. 6/7 | `finetune/models/dove/lora_one_s2_trainer.py:283-286` | yes | ✓ | Verified |
| t=399 one-step denoise (Eq. 2) | `inference_script.py:535`, `:459-493` | yes | ✓ | Verified |
| DIV2K train images = 900 | README:120 says 800 (`DIV2K_train_HR.zip`) | n/a | ✗ | MISMATCH |
| HQ-VSR 2,055 videos | README:119 (dataset link) | dataset released | ✓ | Verified-present |

## 3. Findings

## missing

```yaml finding
id: video-pipeline-code-absent
category: missing
topic: "data construction / ablation reproducibility"
title: "Sec 3.3 video-processing pipeline (and Table 1d ablation) has no code"
severity: high
confidence: high
status: finding
file: code/zhengchen1999__DOVE/README.md
line_start: 80
line_end: 80
quote: |
  - [ ] Release the video processing pipeline.
claim: "The four-step pipeline of Sec 3.3 / Fig 3 (metadata filter, scene filter, quality filter via CLIP-IQA/FasterVQA/DOVER, and optical-flow motion-area bounding-box cropping, Eq. 8) that constructs the headline HQ-VSR dataset is not in the repo; an AST/grep over all .py files finds no scene-detection, motion-mask, motion-area, bounding-box, or aesthetic-filter code (see check)."
concern: "HQ-VSR is a central contribution and the source of the dataset/pipeline ablations in Table 1c-1d (+Filter, +Motion rows); without the pipeline code those ablation values and the dataset-curation claim cannot be reproduced or verified."
resolution: "Release the video-processing pipeline code (steps S1-S4 incl. Eq. 8 motion-area cropping) and the per-step configuration referenced as 'in the supplementary material'."
cross_refs: ["fastervqa-no-computation-code"]
check_script: _audit_code/check_missing_metric_code.py
paper_ref: "Sec 3.3, Fig 3, Table 1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fastervqa-no-computation-code
category: missing
topic: "result traceability / metrics"
title: "FasterVQA metric (Tab 2, Fig 1) has no computation code in the repo"
severity: high
confidence: high
status: finding
file: code/zhengchen1999__DOVE/README.md
line_start: 269
line_end: 269
quote: |
  > **TODO:** Add metric computation scripts for FasterVQA, DOVER, and $E^*_{warp}$.
claim: "A grep over every .py file for 'fastervqa'/'faster_vqa' finds zero matches except this README TODO; eval_metrics.py only computes psnr/ssim/lpips/dists/clipiqa, and no script computes FasterVQA."
concern: "FasterVQA is reported for DOVE and every baseline in Table 2 and in Fig 1; with no code producing it, those reported FasterVQA numbers are not reproducible from the repo."
resolution: "Release the FasterVQA evaluation script (and pinned dependency/weights) used to produce the Table 2 / Fig 1 FasterVQA column."
cross_refs: ["ewarp-import-broken", "dover-external-and-offbyone"]
check_script: _audit_code/check_missing_metric_code.py
paper_ref: "Table 2 (FasterVQA rows), Figure 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-external-and-offbyone
category: missing
topic: "result traceability / metrics"
title: "DOVER eval depends on an un-vendored external package (no dependency listed)"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 150
line_end: 154
quote: |
    from DOVER.evaluate_a_set_of_videos import evaluate_set as dover

    input_path = os.path.abspath(input_path)

    dover_results = dover(input_path, device=device)
claim: "eval_dover.py imports the external `DOVER` package, which is not vendored in the repo (no `DOVER/` directory) and is not listed in requirements.txt; DOVER's version/weights are unspecified."
concern: "DOVER scores appear in Tables 1, 2, 3 and Fig 1 (it is the headline VideoLQ 'Performance' column of Table 3); the metric cannot be reproduced from the repo without an unspecified external install."
resolution: "Vendor or pin the exact DOVER commit + weights used, and add it to requirements; document the install."
cross_refs: ["dover-per-sample-offbyone"]
check_script: _audit_code/check_missing_metric_code.py
paper_ref: "Table 2 / Table 3 (DOVER columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: diffusers-pyiqa-unpinned
category: missing
topic: "dependencies / environment"
title: "diffusers and pyiqa absent from requirements.txt (env not rebuildable)"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/requirements.txt
line_start: 1
line_end: 20
quote: |
  accelerate>=1.1.1
  transformers>=4.46.2
  numpy==1.26.0
  torch>=2.5.0
  torchvision>=0.20.0
claim: "requirements.txt does not list `diffusers` (the package providing CogVideoXPipeline / CogVideoXDPMScheduler / get_3d_rotary_pos_embed that the training and inference code import) nor `pyiqa` (which computes every reported IQA metric); both are only mentioned as ad-hoc `pip install` lines in the README with no version pin."
concern: "The diffusers API used (rotary-embedding helper, DPM scheduler get_velocity) is version-sensitive, so an unpinned/missing diffusers makes the reported numbers non-rebuildable and risks silent API drift."
resolution: "Add diffusers and pyiqa to requirements.txt with the exact versions used for the paper results."
cross_refs: []
check_script: _audit_code/check_missing_metric_code.py
paper_ref: "Sec 4.1 Implementation Details"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: ewarp-import-broken
category: bug
topic: "metrics / temporal consistency"
title: "eval_ewarp.py imports nonexistent `ewarp` module and chdir's into missing RAFT dir"
severity: high
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_ewarp.py
line_start: 146
line_end: 150
quote: |
    original_dir = os.getcwd()
    raft_dir = os.path.join(original_dir, "RAFT")
    os.chdir(raft_dir)
    from ewarp import Ewarp as Ewarp
    results, avg_score = Ewarp(args)
claim: "After os.chdir(script_dir) at module top (line 6-7), this code chdir's into 'RAFT' under finetune/scripts/ — which does not exist (RAFT lives at finetune/utils/RAFT) — then does `from ewarp import Ewarp`; a repo-wide search finds no file `ewarp.py` and no `def Ewarp` anywhere (see check). The default --model path `finetune/scripts/models/raft-things.pth` also does not exist."
concern: "The E*warp (flow warping error) script cannot run, so the E*warp column of Table 2 (reported for DOVE and every baseline) and the temporal-consistency claims (Fig 5) are not reproducible from the released code."
resolution: "Add the missing `ewarp.py` (the `Ewarp` implementation), fix the RAFT directory path and the default RAFT checkpoint path."
cross_refs: ["fastervqa-no-computation-code", "vbench-missing-dir"]
check_script: _audit_code/check_missing_metric_code.py
paper_ref: "Table 2 (E*warp rows), Figure 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: vbench-missing-dir
category: bug
topic: "metrics"
title: "eval_vbench.py chdir's into a nonexistent finetune/scripts/VBench directory"
severity: low
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_vbench.py
line_start: 145
line_end: 150
quote: |
    original_dir = os.getcwd()
    vbench_dir = os.path.join(original_dir, "VBench")
    os.chdir(vbench_dir)
    print(f"Changed directory to: {vbench_dir}")
    from evaluate import calculate_final as Vbench
    results, avg_score, dim_results, dim_avg = Vbench(input_path)
claim: "eval_vbench.py changes into 'VBench' under finetune/scripts/ (absent in repo, see check) and imports `from evaluate import calculate_final`; the VBench package is neither vendored nor in requirements.txt."
concern: "The script crashes immediately; however VBench is not a metric reported in the main paper's Table 2, so impact on the headline numbers is low."
resolution: "Remove this helper or vendor/pin VBench and fix the path if any reported number depends on it."
cross_refs: ["ewarp-import-broken"]
check_script: _audit_code/check_missing_metric_code.py
paper_ref: "N/A (VBench not in main Table 2)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-per-sample-offbyone
category: bug
topic: "metrics"
title: "DOVER per-sample scores misaligned by one (results[name]=dover_results[i-1])"
severity: low
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
    for i, name in enumerate(pred_names):
        results[name] = dover_results[i-1]
claim: "The loop assigns DOVER score index i-1 to the i-th sorted video name, so for i=0 the first video receives the LAST video's score and every per-sample DOVER score is shifted by one (verified by simulation in the check script)."
concern: "Per-video DOVER scores written to metrics_dover.json are mislabeled; the reported aggregate DOVER (an unweighted mean over all videos) is unaffected by the permutation, so the Table-2/Table-3 averages stand, but any per-sample DOVER inspection is wrong."
resolution: "Change to `results[name] = dover_results[i]`."
cross_refs: ["dover-external-and-offbyone"]
check_script: _audit_code/check_missing_metric_code.py
paper_ref: "Table 2 (DOVER), Table 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: inference-sh-wrong-gt
category: bug
topic: "evaluation harness"
title: "inference.sh passes UDM10 GT for every dataset's full-reference eval"
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
claim: "For SPMCS, YouHQ40, RealVSR, and MVSR4x the script evaluates predictions against `datasets/test/UDM10/GT` instead of each dataset's own GT (copy-paste error); eval_metrics.py matches files by basename, so with mismatched filenames it prints 'Skipping ... no matching GT file' and reports count=0."
concern: "As written, the released reproduction harness cannot reproduce the full-reference Table-2 metrics for any dataset except UDM10; a user running inference.sh verbatim gets no/zero scores for SPMCS/YouHQ40/RealVSR/MVSR4x."
resolution: "Fix the `--gt` path in each block to the matching dataset (e.g. datasets/test/SPMCS/GT)."
cross_refs: []
paper_ref: "Table 2 (per-dataset full-reference metrics)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: div2k-image-count-mismatch
category: difference
topic: "training data description"
title: "Paper says DIV2K 900 images; README/link give the standard 800-image set"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  "The image dataset is DIV2K [3], with 900 images, which follows the RealESRGAN [38] degradation process."
claim: "Sec 4.1 states the image training set is DIV2K with 900 images, but the repo README (line 119-120) lists DIV2K-HR as 800 images and links the official `DIV2K_train_HR.zip`, which contains exactly 800 HR training images."
concern: "The image-data count in the paper does not match the dataset the code actually uses (the standard DIV2K train split is 800), a minor description inconsistency in a small part of the training data."
resolution: "Clarify whether 900 images were used (and from which DIV2K split), or correct the paper to 800."
cross_refs: []
paper_ref: "Sec 4.1 Datasets; README Train Datasets table"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: realvsr-mvsr-upscale1
category: difference
topic: "evaluation settings"
title: "inference.sh uses --upscale 1 for RealVSR/MVSR4x while paper states ×4 for all"
severity: low
confidence: medium
status: question
file: code/zhengchen1999__DOVE/inference.sh
line_start: 40
line_end: 46
quote: |
  python inference_script.py \
      --input_dir datasets/test/RealVSR/LQ-Video \
      --model_path pretrained_models/DOVE \
      --output_path results/DOVE/RealVSR \
      --is_vae_st \
      --upscale 1 \
claim: "For RealVSR and MVSR4x the inference script is invoked with --upscale 1 (no bilinear pre-upscale), whereas Sec 4.1 says 'All experiments are conducted with a scaling factor ×4.'"
concern: "If the LQ inputs for these phone-captured real-world datasets are already at target resolution, upscale=1 is the correct operational choice and the ×4 statement is a coarse description; flagged because the code and the blanket ×4 statement differ."
resolution: "Confirm that RealVSR/MVSR4x LQ-HQ pairs are stored at the same spatial resolution (so the effective SR factor is realized in the data, not the bilinear pre-upscale), and reconcile with the ×4 statement."
cross_refs: []
paper_ref: "Sec 4.1 ('scaling factor ×4')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

No methodology finding rises to the level of a defect. One item I actively checked and judged
sound: the training-time validation runs on the **UDM10 test set** (`--validation_dir
.../UDM10`, the same set used for all Table-1 ablations), but `__maybe_save_checkpoint`
(`finetune/trainer.py:1002-1014`) saves checkpoints purely periodically and keeps the last N
(`checkpointing_limit`); there is no best-by-validation selection, and the README directs use
of the fixed final checkpoints (checkpoint-10000 / checkpoint-500). UDM10 metrics are therefore
*observed* during training but do not *influence* the released model, so this is monitoring, not
test-set leakage (the prompt's "if the validation loss were a hidden NaN, would the reported
metric change?" test → no). Recorded here for transparency rather than as a finding.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Sec 3.3 pipeline + FasterVQA code absent; DOVER/diffusers/pyiqa deps unspecified |
| bug         | 4          | high         | E*warp script imports nonexistent module; VBench dir missing; DOVER off-by-one; inference.sh wrong GT |
| difference  | 2          | low          | DIV2K 900-vs-800; RealVSR/MVSR4x upscale=1 vs "×4" (question) |
| methodology | 0          | -            | UDM10 used for monitoring only, not checkpoint selection — looks fine |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing] `video-pipeline-code-absent`** — the Sec 3.3 / Fig 3 video-processing pipeline
   (incl. Eq. 8 motion-area cropping) that builds HQ-VSR and produces Table 1d is not released
   (README TODO confirms). High / high.
2. **[bug] `ewarp-import-broken`** — `eval_ewarp.py` imports a nonexistent `ewarp` module and a
   missing RAFT dir; the E\*warp column of Table 2 and the temporal-consistency claim are not
   reproducible. High / high.
3. **[missing] `fastervqa-no-computation-code`** — no code computes FasterVQA (Table 2, Fig 1);
   only a README TODO. High / high.
4. **[bug] `inference-sh-wrong-gt`** — the reproduction script feeds UDM10 GT to every dataset's
   full-reference eval, so SPMCS/YouHQ40/RealVSR/MVSR4x score nothing as shipped. Medium / high.
5. **[missing] `dover-external-and-offbyone`** — DOVER metric reachable only via an un-vendored,
   unpinned external package (Tables 1/2/3, Fig 1). Medium / high.
6. **[missing] `diffusers-pyiqa-unpinned`** — diffusers and pyiqa missing from requirements.txt;
   environment not rebuildable from the repo. Medium / high.

### Items that genuinely look fine
- Stage-1 latent MSE (Eq. 3), stage-2 MSE+DISTS (Eq. 4), and frame-difference L1 loss (Eq. 6/7)
  are implemented as described (`lora_one_s1_trainer.py:207`, `lora_one_s2_trainer.py:238-286`).
- Headline hyperparameters match the paper: t=399, AdamW β1/β2/β3 = 0.9/0.95/0.98, stage-1 lr
  2e-5 / 10k steps / 25×320×640, stage-2 lr 5e-6 / 500 steps / φ=0.8, λ1=λ2=1, empty prompt,
  ×4 bilinear, total batch size 8 (batch 2 × 4 GPUs).
- The full-reference + CLIP-IQA metrics (PSNR/SSIM/LPIPS/DISTS/CLIP-IQA) are computed by
  `eval_metrics.py` via pyiqa and are fully traceable.
- Checkpointing is periodic with no best-by-test selection; UDM10 validation is monitoring only.
- RealBasicVSR/RealESRGAN degradation configs are present (`configs/degradation*.yaml`).

### Open questions for the authors
- `realvsr-mvsr-upscale1`: are RealVSR/MVSR4x LQ inputs stored at HR resolution (making
  `--upscale 1` correct), and how does that reconcile with the blanket "scaling factor ×4"?
- Which exact DOVER / FasterVQA / E\*warp implementations and weights produced Table 2, and can
  they be pinned/vendored?
- Was DIV2K trained on 800 or 900 images, and from which split?
