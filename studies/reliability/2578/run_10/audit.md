# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (NeurIPS 2025)

## 1. Summary

The repo (`code/zhengchen1999__DOVE/`, owner matches first author) is the author
code for DOVE, a one-step diffusion VSR model fine-tuned from CogVideoX1.5-5B. It
contains: an inference entrypoint (`inference_script.py`) + reproduction shell
(`inference.sh`); a full-reference IQA evaluator (`eval_metrics.py`, also duplicated
at `finetune/scripts/eval_metrics.py`); the two-stage training code under
`finetune/` (latent-space Stage-1 and pixel-space Stage-2 trainers, degradation
pipeline, dataset loaders, RAFT optical-flow utilities); train shell scripts; and a
requirements file. Pretrained weights and all datasets are hosted off-repo (Google
Drive / Baidu / HuggingFace links in the README); only 7 demo `.mp4`s and an empty
prompt embedding are committed.

I read the paper (Tables 1–3, Fig. 1, method §3, settings §4.1) and walked the code.
I verified the loss functions against Eqs. (3)–(7), the train hyperparameters against
§4.1, and the evaluation pipeline against the metric list in §4.1. I wrote three
deterministic checks under `_audit_code/` (run read-only): `check_inference_sh_gt.py`
(parses `inference.sh`), `check_eval_deps.py` (file-existence of metric-script
dependencies), and `check_dover_offbyone.py` (reproduces the per-video indexing in
`eval_dover.py`). Outputs are in `_audit_code/out/`.

The training/inference core matches the paper. The defects concentrate in the
evaluation harness: (a) the committed reproduction script `inference.sh` evaluates
every dataset against UDM10's ground truth; (b) the scripts that compute the paper's
VQA (FasterVQA, DOVER) and temporal-consistency (E*warp) columns are present but
import absent modules and `chdir` to non-existent paths, so they cannot run as
shipped; (c) `eval_dover.py` misassigns per-video scores by one index (average
unaffected); (d) `pyiqa`, required by the evaluator, is not in `requirements.txt`.

I could not execute the model (no GPU, no multi-GB weights/datasets in the sandbox),
so the numeric values in Table 2 cannot be re-derived here; findings are based on
static reading + deterministic file/AST/string checks.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Tab. 2: PSNR/SSIM/LPIPS/DISTS/CLIP-IQA (per dataset) | `eval_metrics.py` via `inference_script.py` | Yes (FR-IQA via pyiqa) | not runnable in sandbox | Code present; reproduction script mis-wires GT (see eval-gt-hardcoded-udm10) |
| Tab. 2 / Tab. 1 / Tab. 3: DOVER ↑ | `finetune/scripts/eval_dover.py` | Imports absent `DOVER.*` pkg; per-video off-by-one | n/a | MISSING dep (eval-vqa-temporal-missing-deps) + bug (dover-per-video-off-by-one) |
| Tab. 2: FasterVQA ↑ | `finetune/scripts/eval_vbench.py` | chdir to absent `VBench/`; imports absent `evaluate` | n/a | MISSING dep (eval-vqa-temporal-missing-deps) |
| Tab. 2: E*warp ↓ (temporal) | `finetune/scripts/eval_ewarp.py` | chdir to absent `finetune/scripts/RAFT/`; imports absent `ewarp` | n/a | MISSING dep / broken paths (eval-vqa-temporal-missing-deps) |
| Tab. 1a–d ablations (PSNR/LPIPS/CLIP-IQA/DOVER on UDM10) | training scripts + `eval_metrics`/`eval_dover` | Partly (no per-ablation configs/driver committed) | n/a | No ablation driver; DOVER path broken |
| Tab. 3: Time (s), Step | (none) | No timing/benchmark script in repo | n/a | MISSING (timing-script-missing) |
| Fig. 1: 28× speed-up / DOVER on VideoLQ | (none for timing; DOVER script broken) | No | n/a | MISSING timing |
| §4.1 training: lr 2e-5/5e-6, 10k/500 steps, φ=0.8, λ1=λ2=1, 320×640 | `finetune/train_ddp_one_s1.sh`, `train_ddp_one_s2.sh` | Yes | ✓ matches | Verified |
| Eq. (3) Stage-1 latent MSE | `finetune/models/dove/lora_one_s1_trainer.py:207` | Yes | ✓ | Verified |
| Eq. (4)/(7) Stage-2 MSE+DISTS, Eq. (6) frame-diff L1 | `finetune/models/dove/lora_one_s2_trainer.py:238,253,285` | Yes | ✓ | Verified |
| t=399 starting timestep | `inference_script.py:535`, trainers `sr_noise_step=399` | Yes | ✓ | Verified |

## 3. Findings

## missing

```yaml finding
id: eval-vqa-temporal-missing-deps
category: missing
topic: "result traceability / evaluation"
title: "DOVER, FasterVQA and E*warp metric scripts import absent modules and chdir to non-existent paths"
severity: high
confidence: high
status: finding
file: finetune/scripts/eval_ewarp.py
line_start: 19
line_end: 20
quote: |
  sys.path.append(os.path.join(os.getcwd(), "RAFT"))
  sys.path.append(os.path.join(os.getcwd(), "RAFT/core"))
claim: "The three scripts that compute the paper's VQA and temporal columns each depend on resources absent from the repo: eval_ewarp.py chdir's into finetune/scripts/RAFT (absent) and does `from ewarp import Ewarp` (no ewarp.py anywhere; the actual RAFT tree is at finetune/utils/RAFT and its checkpoint is finetune/utils/RAFT/raft-things.pth, not the default finetune/scripts/models/raft-things.pth); eval_dover.py does `from DOVER.evaluate_a_set_of_videos import evaluate_set` (no DOVER package in repo); eval_vbench.py chdir's into finetune/scripts/VBench (absent) and does `from evaluate import calculate_final`."
concern: "Every DOVER (Tab. 1, Tab. 2, Tab. 3, Fig. 1), FasterVQA (Tab. 2) and E*warp (Tab. 2) value in the paper is produced by code that cannot run as shipped, so those reported numbers are not reproducible from the repo; the README itself lists these as a 'TODO: Add metric computation scripts for FasterVQA, DOVER, and E*warp'."
resolution: "Provide the DOVER and VBench/FasterVQA packages (or pinned install + correct sys.path), the ewarp module, and fix the RAFT directory/checkpoint paths, with exact commands; or state these metrics were computed with the official upstream tools and pin their versions."
cross_refs: ["dover-per-video-off-by-one", "eval-gt-hardcoded-udm10"]
check_script: _audit_code/check_eval_deps.py
paper_ref: "Table 2 (DOVER/FasterVQA/E*warp columns); Table 3; Fig. 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: timing-script-missing
category: missing
topic: "result traceability / efficiency claim"
title: "No timing/efficiency script for the 28x speed-up and Table 3 running times"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  For fairness, all methods are measured running time on the same A100 GPU, generating a 33-frame 720×1280 video. Our method is approximately 28× faster than MGLD-VSR [50].
claim: "The headline efficiency claim (Fig. 1 / Table 3: Time = 14.90 s for DOVE, up to 28x faster than MGLD-VSR) is a measured quantity, but no benchmark/timing harness exists in the repo (no script measures per-method seconds for a 33-frame 720x1280 video). inference_script.py prints no timing."
concern: "The central efficiency contribution cannot be reproduced or independently checked from the repo because the measurement code is absent."
resolution: "Add the timing script (and the configs/commands used for each compared method) that produced Table 3 and Fig. 1, including how warm-up/IO were handled."
cross_refs: []
paper_ref: "Table 3; Fig. 1; §4.3 Running Time Comparisons"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pyiqa-missing-from-requirements
category: missing
topic: "dependencies / environment"
title: "pyiqa (required by the evaluator) is not in requirements.txt"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 5
quote: |
  accelerate>=1.1.1
  transformers>=4.46.2
  numpy==1.26.0
  torch>=2.5.0
  torchvision>=0.20.0
claim: "eval_metrics.py and inference_script.py both `import pyiqa` and call pyiqa.create_metric for PSNR/SSIM/LPIPS/DISTS/CLIP-IQA, but pyiqa is absent from requirements.txt; it appears only as a separate `pip install pyiqa` line buried in the README Dependencies block."
concern: "A fresh `pip install -r requirements.txt` environment cannot run the evaluation, so the dependency spec is incomplete and unpinned for the metric library that defines the reported IQA values."
resolution: "Add pyiqa (pinned) and any other run-only deps (e.g. pyiqa's model weights source) to requirements.txt."
cross_refs: ["eval-gt-hardcoded-udm10"]
check_script: _audit_code/check_eval_deps.py
paper_ref: "§4.1 Evaluation Metrics"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: eval-gt-hardcoded-udm10
category: bug
topic: "evaluation harness"
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
claim: "The committed reproduction script passes `--gt datasets/test/UDM10/GT` for ALL six eval calls; only the UDM10 call uses the matching GT. SPMCS, YouHQ40, RealVSR, MVSR4x and VideoLQ predictions are all evaluated against UDM10's GT folder (5 of 6 calls; see check output)."
concern: "Because eval_metrics.py matches predictions to GT by filename and skips non-matching names (eval_metrics.py:174), running inference.sh as written silently skips every non-UDM10 clip (mismatched names) or, where names collide, compares against the wrong reference — so the script does NOT reproduce the per-dataset Table 2 numbers for 5 of the 6 datasets."
resolution: "Point each `--gt` at the dataset's own GT folder (e.g. datasets/test/SPMCS/GT, datasets/test/YouHQ40/GT, ...)."
cross_refs: ["eval-vqa-temporal-missing-deps"]
check_script: _audit_code/check_inference_sh_gt.py
paper_ref: "Table 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-per-video-off-by-one
category: bug
topic: "evaluation / indexing"
title: "eval_dover.py assigns each video the previous video's DOVER score (off-by-one)"
severity: low
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
  for i, name in enumerate(pred_names):
        results[name] = dover_results[i-1]
claim: "Per-video DOVER scores are written with `dover_results[i-1]`, so name at index i receives the score of index i-1; the first name (i=0) receives `dover_results[-1]`, i.e. the LAST video's score. The reproduction (check_dover_offbyone.py) shows the resulting dict is shifted by one."
concern: "Any per-video / qualitative-selection use of these DOVER scores is wrong; the reported AVERAGE DOVER is unaffected (the values are merely permuted), so this does not by itself change Table 2's averages — hence low severity — but it indicates the per-sample export is unreliable."
resolution: "Change to `results[name] = dover_results[i]` and confirm dover_results is ordered to match sorted(pred_names)."
cross_refs: ["eval-vqa-temporal-missing-deps"]
check_script: _audit_code/check_dover_offbyone.py
paper_ref: "Table 2 DOVER column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: psnr-rgb-not-y-channel
category: difference
topic: "metric definition"
title: "PSNR/SSIM computed on RGB by default, not the Y channel typical for VSR"
severity: low
confidence: low
status: question
file: eval_metrics.py
line_start: 245
line_end: 246
quote: |
      parser.add_argument('--crop', type=int, default=0, help='Crop border size for PSNR/SSIM')
      parser.add_argument('--test_y_channel', action='store_true', help='Use Y channel for PSNR/SSIM')
claim: "PSNR/SSIM are computed on full RGB unless `--test_y_channel` and `--crop` are passed; neither inference.sh nor the README example passes them, so by default PSNR/SSIM use RGB with no border crop. Many VSR papers report Y-channel PSNR/SSIM."
concern: "If the paper's Table 2 PSNR/SSIM were computed on Y channel (or with a border crop) but the shipped command uses RGB/no-crop, the reproduced fidelity numbers would differ; the paper does not state the channel/crop convention, so this is ambiguous."
resolution: "Authors: state whether Table 2 PSNR/SSIM are RGB or Y-channel and whether any border crop was applied, and ship the matching eval command."
cross_refs: []
paper_ref: "§4.1 Evaluation Metrics (PSNR, SSIM)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

No methodology findings. The training procedure (latent-space Stage-1 MSE, pixel-space
Stage-2 MSE+DISTS+frame-difference), the t=399 one-step formulation, the
RealBasicVSR/RealESRGAN-style degradation pipeline, and the hyperparameters all match
the paper. Train and test datasets come from disjoint sources (HQ-VSR/OpenVid-1M +
DIV2K for training; UDM10/SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ for testing), so there is
no train/test leakage in the classic sense. N/A: pretraining-contamination of the test
sets is not separately bounded, but the contribution is a generative restorer evaluated
on standard external benchmarks, so this is not a leakage concern of the kind the
checklist targets.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | VQA/temporal metric scripts depend on absent pkgs; no timing harness; pyiqa unpinned |
| bug         | 2          | high         | inference.sh uses UDM10 GT for all datasets; eval_dover per-video off-by-one (avg unaffected) |
| difference  | 1          | low          | PSNR/SSIM default to RGB, channel/crop convention unstated (question) |
| methodology | 0          | -            | Training/inference faithfully implement the paper; no leakage |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[bug, high/high]** `eval-gt-hardcoded-udm10` — the shipped reproduction script
   `inference.sh` evaluates SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ against UDM10's GT, so
   it does not reproduce 5 of 6 Table 2 dataset rows out of the box.
2. **[missing, high/high]** `eval-vqa-temporal-missing-deps` — DOVER, FasterVQA and
   E*warp scripts import absent modules / chdir to non-existent paths; every VQA and
   temporal-consistency number in Tables 1–3 / Fig. 1 is not reproducible as shipped.
3. **[missing, medium/high]** `timing-script-missing` — no benchmark code for the
   headline 28× speed-up / Table 3 running times.
4. **[missing, low/high]** `pyiqa-missing-from-requirements` — the metric library is
   absent from `requirements.txt`.
5. **[bug, low/high]** `dover-per-video-off-by-one` — per-video DOVER scores shifted by
   one index; reported averages unaffected.

### Items that genuinely look fine
- Stage-1 latent-space MSE loss (`lora_one_s1_trainer.py:207`) matches Eq. (3).
- Stage-2 loss = MSE + DISTS perceptual + frame-difference L1
  (`lora_one_s2_trainer.py:238/253/285`) matches Eqs. (4),(6),(7) with λ1=λ2=1.
- Train hyperparameters in `train_ddp_one_s1.sh`/`train_ddp_one_s2.sh` (lr 2e-5/5e-6,
  10k/500 steps, image_ratio 0.8, 320×640, batch 2×4GPU=8) match §4.1.
- One-step t=399 v-prediction formulation matches Eq. (2) and §3.1.
- Empty-prompt embedding file name = sha256("") = e3b0c4…, consistent with the README's
  empty-prompt pre-encoding optimisation.
- Degradation config is a standard RealBasicVSR second-order pipeline, matching §4.1.
- Train/test datasets are from disjoint sources (no leakage).

### Open questions for the authors
- `psnr-rgb-not-y-channel`: were Table 2 PSNR/SSIM computed on RGB or Y channel, and
  with what (if any) border crop? The shipped command defaults to RGB / no crop.
- Were the DOVER/FasterVQA/E*warp numbers produced by the in-repo scripts (currently
  non-runnable) or by the official upstream tools? If the latter, please pin versions.
