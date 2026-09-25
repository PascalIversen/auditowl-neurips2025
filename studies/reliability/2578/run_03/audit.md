# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (#2578)

## 1. Summary

The cloned repo `code/zhengchen1999__DOVE/` is the authors' own code (README header,
abstract, citation, and figure names all match paper #2578). It is an empirical
deep-learning paper: DOVE fine-tunes CogVideoX1.5-5B into a one-step video
super-resolution model. The repo contains a complete *training* pipeline
(`finetune/`: two-stage trainer, RealBasicVSR-style degradation, dataset filelist
builder, DeepSpeed/accelerate configs, prompt-embedding cache), an *inference* entry
point (`inference_script.py` + `inference.sh`), and a *metric* script
(`eval_metrics.py`). Pretrained weights and the HQ-VSR dataset are distributed via
external Google Drive links (legitimate: a 5B model trained on 4×A100 cannot be
shipped in-repo).

What I did: read the paper (PDF + extraction) and every Python/shell/yaml file in the
repo; matched the training hyper-parameters in `train_ddp_one_s1.sh` /
`train_ddp_one_s2.sh` / `schemas/args.py` against §4.1 (all consistent: lr 2e-5/5e-6,
seed 42, 25×320×640, 10k/500 steps, AdamW β=0.9/0.95/0.98, φ=0.8, λ1=λ2=1,
sr_noise_step=399); confirmed the loss in `lora_one_s2_trainer.py` implements
Eq. (3)–(7); verified the empty-prompt embedding file is the SHA-256 of the empty
string (consistent with the "empty prompt" design). I wrote two read-only checks in
`_audit_code/` to confirm (a) which Table-2 metrics actually have a runnable
computation script and (b) that `inference.sh` passes the wrong `--gt` path for 5 of
6 evaluation blocks. The main finding is a **result-traceability gap**: three of the
metrics reported in Table 2 / Fig. 1 (FasterVQA, DOVER, E*_warp) and the running-time
/ "28× speed-up" claim (Tab. 3, Fig. 1) cannot be reproduced from the released code —
the FasterVQA/E*_warp/timing scripts are absent or broken and DOVER depends on an
unbundled package. The README's own TODO acknowledges the metric scripts are not yet
added.

## 2. Traceability table

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Tab. 2 PSNR, SSIM, LPIPS, DISTS, CLIP-IQA (8 datasets) | `eval_metrics.py` (pyiqa) | yes (pyiqa) | n/a (needs weights+data to run) | Traceable |
| Tab. 2 **FasterVQA** column (all datasets) + Fig. 1 | (none) | no | — | MISSING (no script; README TODO) |
| Tab. 2 **DOVER** column + Tab. 3 "Performance" + Fig. 1 | `finetune/scripts/eval_dover.py` | no (imports unbundled `DOVER` pkg) | — | MISSING dependency (+ off-by-one bug) |
| Tab. 2 **E*_warp** column + Fig. 1 | `finetune/scripts/eval_ewarp.py` | no (imports nonexistent `ewarp`; bad RAFT path) | — | BROKEN (bug) |
| Tab. 3 running time (s) + "~28× faster" / Fig. 1 Time | (none) | no | — | MISSING (no timing harness) |
| Tab. 1a–d ablations (S1/S2, φ, dataset, pipeline) | training args (image_ratio, data_root, degradation cfg) | configurable, no dedicated driver | — | Partially traceable |
| §4.1 "DIV2K … 900 images" | README "DIV2K-HR 800" + official 800-img link | n/a | ✗ (800 vs 900) | MISMATCH (low) |
| Empty-prompt embedding (bypasses text encoder) | `pretrained_models/prompt_embeddings/e3b0c442….safetensors` | yes; hash = sha256("") | ✓ | Verified |
| Reproduction script `inference.sh` (FR metrics, 5 datasets) | `inference.sh` (`--gt datasets/test/UDM10/GT`) | runs but wrong GT for 5/6 blocks | ✗ | BUG |

## 3. Findings

## missing

```yaml finding
id: fastervqa-no-compute-script
category: missing
topic: "result traceability / metrics"
title: "FasterVQA column (Tab. 2, Fig. 1) has no computation script in the repo"
severity: high
confidence: high
status: finding
file: code/zhengchen1999__DOVE/README.md
line_start: 269
line_end: 269
quote: |
  > **TODO:** Add metric computation scripts for FasterVQA, DOVER, and $E^*_{warp}$.
claim: "FasterVQA is reported for every dataset in Table 2 and is one of the two headline bars in Fig. 1, but the repo contains no script that computes it: eval_metrics.py only computes psnr/ssim/lpips/dists/clipiqa via pyiqa, and a repo-wide search finds 'fastervqa' only inside this README TODO line."
concern: "A headline metric used in the main quantitative table and the teaser figure cannot be reproduced from the released code."
resolution: "Authors: release the FasterVQA computation script (model checkpoint, preprocessing, and exact invocation), or state which external tool/version produced the Table 2 FasterVQA numbers."
cross_refs: ["dover-external-dependency-missing", "ewarp-broken-import"]
check_script: _audit_code/check_metric_scripts.py
paper_ref: "Table 2 FasterVQA row; Figure 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-external-dependency-missing
category: missing
topic: "result traceability / metrics"
title: "DOVER metric script imports unbundled DOVER package (not in repo or requirements)"
severity: high
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 150
line_end: 154
quote: |
      from DOVER.evaluate_a_set_of_videos import evaluate_set as dover

      input_path = os.path.abspath(input_path)

      dover_results = dover(input_path, device=device)
claim: "eval_dover.py relies on `from DOVER.evaluate_a_set_of_videos import evaluate_set`, but no `DOVER` package/dir exists anywhere in the repo and `dover` is not listed in requirements.txt; there is no fetch/setup instruction for it."
concern: "DOVER is a headline metric (Table 2, Table 3 'Performance' column, Fig. 1); without the bundled or pinned DOVER package the script cannot run, so the reported DOVER numbers are not reproducible from the release."
resolution: "Authors: vendor the DOVER code (or pin its version + weights and document the install), and provide the exact command used to produce the Table 2/3 DOVER values."
cross_refs: ["dover-eval-off-by-one", "fastervqa-no-compute-script"]
check_script: _audit_code/check_metric_scripts.py
paper_ref: "Table 2 DOVER row; Table 3 Performance column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: running-time-no-harness
category: missing
topic: "result traceability / efficiency claim"
title: "No timing harness for Tab. 3 running times / '~28× faster' headline claim"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  For fairness, all methods are measured running time on the same A100 GPU, generating a
  33-frame 720×1280 video. Our method is approximately 28× faster than MGLD-VSR [50].
claim: "Table 3 and Figure 1 report per-method wall-clock running times (e.g. DOVE 14.90 s vs MGLD-VSR 425.23 s) and a '~28× faster' headline, but the repo contains no benchmarking/timing script that measures inference time on a 33-frame 720×1280 video; inference_script.py only runs SR and never times or logs latency."
concern: "The central efficiency claim (28× speed-up) — the paper's main selling point — has no accompanying measurement code, so the timing comparison cannot be independently reproduced or audited for fairness."
resolution: "Authors: release the timing script (warm-up, repetitions, what is included/excluded, e.g. VAE tiling, model load) used for Table 3, or document the exact measurement protocol."
cross_refs: []
paper_ref: "Table 3; Figure 1 (Time panel); Abstract '28×'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pyiqa-diffusers-not-in-requirements
category: missing
topic: "dependencies / environment"
title: "requirements.txt omits pyiqa and diffusers (needed to run inference + metrics)"
severity: low
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
claim: "`pyiqa` (imported by eval_metrics.py and inference_script.py) and `diffusers` (imported throughout) are not in requirements.txt; neither is version-pinned. diffusers in particular is unpinned despite the code using model-specific APIs (CogVideoXPipeline, CogVideoXDPMScheduler.get_velocity)."
concern: "The environment cannot be rebuilt from requirements.txt alone, and an unpinned diffusers risks API drift in get_velocity / pipeline signatures."
resolution: "Add pyiqa and a pinned diffusers (and its pyiqa version) to requirements.txt; the README mentions installing them separately but the pin is still missing."
cross_refs: []
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: ewarp-broken-import
category: bug
topic: "result traceability / metrics"
title: "eval_ewarp.py imports nonexistent `ewarp` module and chdirs into a missing RAFT dir"
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
claim: "After chdir into script-dir, eval_ewarp.py changes directory to `<finetune/scripts>/RAFT` and does `from ewarp import Ewarp`; but there is no `finetune/scripts/RAFT` directory (RAFT lives at finetune/utils/RAFT) and there is no `ewarp.py` module anywhere in the repo."
concern: "E*_warp (temporal-consistency metric) is reported for every dataset in Table 2 and in Fig. 1, but its script crashes immediately on import/chdir, so the reported E*_warp values are not reproducible from the release; README TODO confirms the script was not finished."
resolution: "Authors: add the missing `ewarp.py` (and ensure the RAFT path resolves to finetune/utils/RAFT), or document the external tool/version used to compute E*_warp."
cross_refs: ["fastervqa-no-compute-script"]
check_script: _audit_code/check_metric_scripts.py
paper_ref: "Table 2 E*_warp row; Figure 1 (E*_warp panel)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eval-gt-path-mismatch
category: bug
topic: "evaluation / reproduction script"
title: "inference.sh passes --gt datasets/test/UDM10/GT for 5 non-UDM10 eval blocks"
severity: high
confidence: high
status: finding
file: code/zhengchen1999__DOVE/inference.sh
line_start: 15
line_end: 25
quote: |
  # SPMCS
  python inference_script.py \
      --input_dir datasets/test/SPMCS/LQ-Video \
      --model_path pretrained_models/DOVE \
      --output_path results/DOVE/SPMCS \
      --is_vae_st \

  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/SPMCS \
      --metrics psnr,ssim,lpips,dists,clipiqa
claim: "In inference.sh, the eval_metrics.py call for SPMCS (and identically for YouHQ40, RealVSR, MVSR4x, VideoLQ) sets --gt to datasets/test/UDM10/GT instead of the matching dataset's GT; only the UDM10 block uses the correct GT. _audit_code/check_inference_sh_gt.py confirms 5/6 blocks mismatch."
concern: "eval_metrics.py matches GT↔pred by filename stem and skips names with no GT match (eval_metrics.py:174-176); pointing every dataset at UDM10/GT will silently skip all/most samples or score against unrelated frames, so the canonical reproduction script does not reproduce Table 2 for those 5 datasets."
resolution: "Fix each --gt line to point at the corresponding datasets/test/<DATASET>/GT (the README's single manual example at README lines 259-262 uses the correct GT)."
cross_refs: []
check_script: _audit_code/check_inference_sh_gt.py
paper_ref: "Table 2 (SPMCS, YouHQ40, RealVSR, MVSR4x rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-eval-off-by-one
category: bug
topic: "metrics / indexing"
title: "eval_dover.py off-by-one assigns DOVER scores to the wrong filenames"
severity: low
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
      for i, name in enumerate(pred_names):
          results[name] = dover_results[i-1]
claim: "Per-sample DOVER scores are indexed with `dover_results[i-1]` instead of `dover_results[i]`, so pred_names[0] gets the last score (index -1) and every name is shifted by one relative to evaluate_set's output order."
concern: "The per-sample DOVER values written to metrics_dover.json are mis-attributed to filenames; the dataset-level average (np.mean over all values) is unaffected because i-1 over 0..N-1 is a permutation, so the headline DOVER number is not changed, but any per-clip analysis is wrong."
resolution: "Change to `results[name] = dover_results[i]` (after confirming evaluate_set returns results in pred_names order)."
cross_refs: ["dover-external-dependency-missing"]
paper_ref: "Table 2 DOVER row (per-sample)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: div2k-image-count
category: difference
topic: "training data description"
title: "Paper says DIV2K '900 images'; README/official link is the 800-image train split"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  The image dataset is DIV2K [3], with 900 images, which follows the RealESRGAN [38] degradation process.
claim: "§4.1 states the image training set is DIV2K with 900 images, but the released README lists DIV2K-HR as 800 images and links the official DIV2K_train_HR.zip, which is the standard 800-image training split. The code uses a DIV2K_train_HR.txt filelist (built by prepare_dataset.py) that is not shipped, so the exact count cannot be re-derived from the repo."
concern: "The reported image-dataset size in the paper does not match the released dataset description; minor, but a reproduction using the linked data would use 800 images, not 900."
resolution: "Authors: confirm whether 800 or 900 images were used (e.g. DIV2K train 800 + 100 from val/other), and reconcile §4.1 with the README."
cross_refs: []
paper_ref: "Section 4.1 Datasets; README 'Train Datasets' table"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. This is a generative video-super-resolution paper, not a
classification/regression task with learned decision thresholds: train and test
datasets are entirely disjoint standard benchmarks, there is no train/test split to
leak across, and no statistical tests are reported. The in-training validation on
UDM10 (`train_ddp_one_s*.sh` validation block) only *logs* metrics to wandb/console;
`trainer.py` never uses a validation metric to select a "best" checkpoint (no
best-checkpoint or early-stopping logic — see trainer.py:817-827, which only logs),
so monitoring UDM10 during training is not test-set tuning leakage (cf. the prompt's
"if the loss had been a hidden NaN, would the reported metric change?" diagnostic —
no). N/A: pretraining-contamination (CogVideoX is a generative prior, and the metric
is reconstruction fidelity/quality against held-out GT, not memorisable labels);
temporal-integrity splitting (no time-ordered split task).

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | FasterVQA/DOVER/E*_warp metric scripts + timing harness absent; deps incomplete |
| bug         | 3          | high         | E*_warp import broken; inference.sh wrong GT for 5/6 datasets; DOVER off-by-one |
| difference  | 1          | low          | DIV2K 900 (paper) vs 800 (released) image count |
| methodology | 0          | -            | No leakage/stat issues; SR task with disjoint benchmarks, validation only logged |

## 5. Closing lists

**Top take-aways** (≤6, severity × confidence):
1. `fastervqa-no-compute-script` (missing, high) — FasterVQA (Tab. 2, Fig. 1) has no computation script.
2. `dover-external-dependency-missing` (missing, high) — DOVER metric needs an unbundled `DOVER` package.
3. `ewarp-broken-import` (bug, high) — E*_warp script imports a nonexistent module + bad RAFT path; cannot run.
4. `eval-gt-path-mismatch` (bug, high) — inference.sh evaluates 5/6 datasets against the wrong (UDM10) GT.
5. `running-time-no-harness` (missing, medium) — no timing script behind the headline "28× faster" claim.
6. `pyiqa-diffusers-not-in-requirements` (missing, low) — environment not rebuildable from requirements.txt alone.

**Items that genuinely look fine**:
- Training hyper-parameters in the shell scripts / `schemas/args.py` match §4.1 exactly (lr, seed, resolution, steps, AdamW betas, φ=0.8, λ1=λ2=1, sr_noise_step=399).
- The Stage-2 loss in `lora_one_s2_trainer.py` implements Eq. (3)–(7) (MSE + DISTS + frame-difference L1, image/video mixing by image_ratio).
- The empty-prompt embedding file is genuinely sha256("") = e3b0c442…, matching the "empty prompt bypasses text encoder" design (inference_script.py:423-428, 581-585).
- PSNR/SSIM/LPIPS/DISTS/CLIP-IQA are properly computed via pyiqa in `eval_metrics.py`.
- Pretrained weights & HQ-VSR via external Google Drive are a legitimate reason the repo is not fully self-contained (5B model, 4×A100 training).

**Open questions for the authors**:
- Exactly which tool/version computed the Table 2 FasterVQA, DOVER, and E*_warp columns and the Table 3 timings? (high severity, medium confidence that the released code can reproduce them — currently it cannot.)
- DIV2K image count: 800 (released link) or 900 (paper §4.1)?
