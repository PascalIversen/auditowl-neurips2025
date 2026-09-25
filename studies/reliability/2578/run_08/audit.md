# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (paper 2578)

## 1. Summary

The repository `code/zhengchen1999__DOVE/` is the authors' official code for DOVE
(NeurIPS 2025), an efficient one-step diffusion VSR model built by fine-tuning
CogVideoX1.5-5B. Its README, citation, and assets match this paper. The repo
contains: an inference entry point (`inference_script.py`) and runner
(`inference.sh`); a full-reference metric script (`eval_metrics.py`); a `finetune/`
training stack (Stage-1 latent-MSE trainer and Stage-2 pixel MSE+DISTS+frame-diff
trainer, accelerate launchers, RealBasicVSR/RealESRGAN degradation configs, a
RAFT copy, and `eval_dover.py` / `eval_ewarp.py` / `eval_vbench.py` for the VQA /
temporal metrics); demo videos; an empty-prompt embedding; and READMEs.

What I did. I read the paper (PDF + text extraction) and mapped every numbered
table/figure to code. I read the inference path, the eval scripts, both trainers,
the degradation configs, and the training launchers, and confirmed the latent-MSE
(Stage-1, Eq. 3) and pixel MSE+DISTS+frame-diff (Stage-2, Eqs. 4/6/7) losses and
the φ=0.8 image ratio, lr/iteration, and `sr_noise_step=399` settings match the
paper. I then ran one deterministic check
(`_audit_code/check_eval_artifacts.py`, output `_audit_code/out/eval_artifacts.csv`)
that confirms: (a) the `ewarp`, `DOVER`, and `VBench` modules the eval scripts
import are absent from the repo; (b) `eval_ewarp.py`'s chdir target and default
RAFT-weights path are wrong; (c) `eval_dover.py` maps DOVER scores to clips with an
off-by-one index; (d) all six `--gt` arguments in `inference.sh` point to
`datasets/test/UDM10/GT` even for the other five datasets; and (e) no code for the
Section-3.3 video-processing pipeline (aesthetic / scene / motion-mask cropping)
exists. The base CogVideoX weights, DOVE checkpoints, and datasets are external
Google-Drive / HuggingFace downloads (acceptable, but not in-repo).

I did not run the model (requires GPUs + multi-GB external weights, out of scope for
this sandbox); semantic claims about training fidelity are marked medium confidence.

## 2. Result-traceability table

| Paper artefact (Table/Fig) | Repo location | Computes value? | Status |
|---|---|---|---|
| Tab. 2 PSNR/SSIM/LPIPS/DISTS/CLIP-IQA (all datasets) | `eval_metrics.py` (+pyiqa) | yes | Present / runs |
| Tab. 2 + Fig. 1 DOVER column | `finetune/scripts/eval_dover.py:150` | no (imports absent `DOVER` pkg) + off-by-one (`:159`) | MISSING dep + BUG |
| Tab. 2 + Fig. 1 FasterVQA column | `finetune/scripts/eval_vbench.py:19-20,146` | no (imports absent `VBench`) | MISSING dep |
| Tab. 2 + Fig. 1 E*_warp column | `finetune/scripts/eval_ewarp.py:147-149,190` | no (`from ewarp import` absent; wrong RAFT/model paths) | MISSING dep + BUG |
| Tab. 1a training-strategy ablation (S1 / S2-I / S2-I/V) | `finetune/models/dove/*_trainer.py` + launchers | partial (loss code present; no driver/log producing the 4-column table) | Present (config) |
| Tab. 1b image-ratio (φ) ablation | `--image_ratio` arg + `lora_one_s2_trainer.py:125` | partial (config-driven; no sweep harness/logs) | Present (config) |
| Tab. 1c training-dataset ablation (YouHQ / OpenVid-1M / HQ-VSR) | (none) | no dataset-selection/prep code for the 0.4M-OpenVid / 38k-YouHQ sets | MISSING |
| Tab. 1d pipeline ablation (+Filter / +Motion) | (none) | no Sec-3.3 pipeline code at all | MISSING |
| Sec. 3.3 / Eq. 8 HQ-VSR construction pipeline (metadata→scene→quality→motion crop) | (none; README TODO unchecked) | no | MISSING |
| Tab. 3 / Fig. 1 running time (279.32 / 425.23 / 14.90 s, 28×) | (none) | no benchmark/timing script | MISSING |
| DOVE one-step inference (the SR outputs) | `inference_script.py` | yes (needs external weights) | Present / runs |

## 3. Findings

## missing

```yaml finding
id: video-pipeline-code-absent
category: missing
topic: "result traceability / dataset construction"
title: "Section-3.3 HQ-VSR video processing pipeline (and its Tab. 1d ablation) has no code"
severity: high
confidence: high
status: finding
file: README.md
line_start: 79
line_end: 81
quote: |
  - [x] Release training code.
  - [ ] Release the video processing pipeline.
  - [x] Release HQ-VSR dataset.
claim: "The paper's core contribution and Tab. 1d ablation rely on a four-step pipeline (metadata filtering, scene detection, CLIP-IQA/FasterVQA/DOVER quality filtering, and optical-flow motion-mask bounding-box cropping per Eq. 8) that builds HQ-VSR; the repo contains none of this code, and the README leaves 'Release the video processing pipeline' unchecked."
concern: "Tab. 1c (training-dataset) and Tab. 1d (+Filter/+Motion) ablations and the HQ-VSR dataset claim cannot be reproduced because no script implements the filtering or motion-area cropping (a `grep` for aesthetic/scenedetect/motion-mask/bounding-box returns nothing — see _audit_code/out/eval_artifacts.csv)."
resolution: "Authors: release the metadata/scene/quality-filtering and motion-area cropping scripts (Sec 3.3, Eq. 8), or confirm Tab. 1c/1d are not reproducible from the released code."
cross_refs: ["§3.3", "Table 1c", "Table 1d"]
check_script: _audit_code/check_eval_artifacts.py
paper_ref: "Section 3.3 and Table 1c/1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: vqa-warp-metric-deps-missing
category: missing
topic: "result traceability / evaluation metrics"
title: "DOVER, FasterVQA (VBench), and E*_warp metric scripts import packages absent from the repo"
severity: high
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 150
line_end: 150
quote: |
    from DOVER.evaluate_a_set_of_videos import evaluate_set as dover
claim: "eval_dover.py imports a `DOVER` package, eval_vbench.py appends a `VBench/` dir to sys.path (lines 19-20/146), and eval_ewarp.py imports `from ewarp import Ewarp` (line 149); none of `DOVER/`, `VBench/`, or `ewarp.py` exists anywhere in the repo (verified in _audit_code/out/eval_artifacts.csv), and they are not listed in requirements.txt."
concern: "Three of Table 2's eight metric columns (DOVER, FasterVQA, E*_warp) — including the headline DOVER values in Fig. 1 and Table 3 — cannot be reproduced because the scripts that compute them depend on third-party code that is neither vendored nor declared."
resolution: "Authors: vendor or pin the DOVER, VBench/FasterVQA, and Ewarp dependencies (with versions/weights) so the Table 2 VQA and temporal columns are reproducible; the README already lists this as a TODO (README.md:269)."
cross_refs: ["eval-ewarp-paths-broken", "dover-offbyone-index", "Table 2", "Table 3", "Figure 1"]
check_script: _audit_code/check_eval_artifacts.py
paper_ref: "Table 2 (DOVER/FasterVQA/E*_warp columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: running-time-no-script
category: missing
topic: "result traceability / efficiency claim"
title: "No script computes the running-time / 28x-speedup headline numbers (Tab. 3, Fig. 1)"
severity: medium
confidence: medium
status: finding
file: paper.pdf
quote: |
  Our method is approximately 28× faster than MGLD-VSR [50]. Even
  compared with the fastest compared method, VEnhancer [9], the DOVE is 8 times faster.
claim: "The abstract/Table 3/Fig. 1 report per-method wall-clock times (DOVE 14.90 s vs MGLD-VSR 425.23 s, '28x faster') on a 33-frame 720x1280 video, but no timing/benchmark harness exists in the repo (grep for time.time/perf_counter/latency finds only randomness use in degradation.py — see _audit_code/out/eval_artifacts.csv)."
concern: "The central efficiency claim ('28x speed-up') has no released measurement code, so the timing numbers cannot be independently reproduced from the repo."
resolution: "Authors: release the timing/benchmark script (with the exact frame count, resolution, dtype, and GPU) used to produce Table 3 and Fig. 1."
cross_refs: ["Table 3", "Figure 1"]
check_script: _audit_code/check_eval_artifacts.py
paper_ref: "Table 3 and Figure 1 (Time column)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: inference-sh-gt-hardcoded-udm10
category: bug
topic: "evaluation / reproduce script"
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
claim: "The 'Reproduce paper results' runner reuses `--gt datasets/test/UDM10/GT` in all six eval blocks (lines 11,23,35,48,61,73); five of them (SPMCS, YouHQ40, RealVSR, MVSR4x, VideoLQ) pair non-UDM10 predictions with UDM10 ground truth (verified in _audit_code/out/eval_artifacts.csv)."
concern: "Anyone following inference.sh computes the full-reference Table-2 metrics (PSNR/SSIM/LPIPS/DISTS) for five of six datasets against the wrong GT; because eval_metrics.py matches by basename, mismatched names are silently skipped ('No valid samples') or, if names collide, scored against the wrong reference — so the script as shipped does not reproduce Table 2."
resolution: "Authors: change each `--gt` to the matching `datasets/test/<DatasetName>/GT`; confirm Table 2 was produced with per-dataset GT, not UDM10's."
cross_refs: ["Table 2"]
check_script: _audit_code/check_eval_artifacts.py
paper_ref: "Table 2 (PSNR/SSIM/LPIPS/DISTS columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-offbyone-index
category: bug
topic: "evaluation metrics / DOVER"
title: "eval_dover.py assigns DOVER scores to clips with an off-by-one (i-1) index"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
    for i, name in enumerate(pred_names):
        results[name] = dover_results[i-1]
claim: "Iterating with `enumerate(pred_names)` starts at i=0, so the first clip is assigned `dover_results[-1]` (the last clip's score) and every other clip is shifted by one, mis-aligning per-clip DOVER scores (verified in _audit_code/out/eval_artifacts.csv)."
concern: "Per-sample DOVER values are mis-assigned by one position; the dataset mean is unchanged but per-clip results (and any inspection of them) are wrong, and combined with the missing DOVER package this script cannot produce correct per-clip output."
resolution: "Authors: index with `dover_results[i]` (not `i-1`), and confirm whether the reported DOVER values used this script."
cross_refs: ["vqa-warp-metric-deps-missing", "Table 2"]
check_script: _audit_code/check_eval_artifacts.py
paper_ref: "Table 2 (DOVER column)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eval-ewarp-paths-broken
category: bug
topic: "evaluation metrics / temporal consistency"
title: "eval_ewarp.py chdirs to a non-existent RAFT dir and defaults to a non-existent weights path"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_ewarp.py
line_start: 147
line_end: 149
quote: |
    raft_dir = os.path.join(original_dir, "RAFT")
    os.chdir(raft_dir)
    from ewarp import Ewarp as Ewarp
claim: "After the module-level `os.chdir(script_directory)` (line 7), `original_dir` is `finetune/scripts/`, so the script chdirs into `finetune/scripts/RAFT` which does not exist (the RAFT package is at `finetune/utils/RAFT`), then imports a module `ewarp` that exists nowhere in the repo; the default `--model` path `finetune/scripts/models/raft-things.pth` (line 190) is also absent (verified in _audit_code/out/eval_artifacts.csv)."
concern: "The E*_warp temporal-consistency column of Table 2 / Fig. 1 cannot be computed: the script raises FileNotFoundError on chdir or ImportError on `from ewarp import`, and the default RAFT checkpoint path points to a directory that does not exist."
resolution: "Authors: ship `ewarp.py`, fix the chdir target to `finetune/utils/RAFT`, and set the default `--model` to the bundled `finetune/utils/RAFT/raft-things.pth`."
cross_refs: ["vqa-warp-metric-deps-missing", "Table 2"]
check_script: _audit_code/check_eval_artifacts.py
paper_ref: "Table 2 (E*_warp column)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: stage1-batchsize-image-count
category: difference
topic: "experimental settings"
title: "Released DIV2K image count (900) and per-GPU batch differ slightly from in-repo defaults"
severity: low
confidence: medium
status: finding
file: finetune/train_ddp_one_s1.sh
line_start: 37
line_end: 38
quote: |
    --batch_size 2
    --gradient_accumulation_steps 1
claim: "The launchers use `--batch_size 2` per process with `gradient_accumulation_steps 1`; the paper states 'total batch size 8' on '4 NVIDIA A800-80G GPUs', consistent only under 4 data-parallel processes (4x2=8), which the script does not itself enforce. Separately the paper says DIV2K has '900 images' (Sec 4.1) while the README datasets table lists DIV2K-HR as 800 images."
concern: "The 8-vs-2 batch size and 900-vs-800 image-count statements require the reader to assume the 4-GPU launch and a specific DIV2K variant; neither is pinned in the script, so an exact reproduction depends on undocumented launch topology."
resolution: "Authors: document the number of data-parallel processes used (to reach total batch 8) and clarify whether DIV2K-HR (800) or DIV2K train+val (900) was used."
cross_refs: ["§4.1"]
paper_ref: "Section 4.1 (Implementation Details / Datasets)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A — No methodology defect found. This is a generative VSR paper with no
train/test split over shared units, no leakage surface, and no statistical
significance test to mis-apply: training uses HQ-VSR + DIV2K with synthetic
degradation; evaluation is on separate standard benchmarks (UDM10, SPMCS,
YouHQ40, RealVSR, MVSR4x, VideoLQ). The Stage-1 latent-MSE loss (Eq. 3) and
Stage-2 MSE+DISTS+frame-diff loss (Eqs. 4/6/7) implemented in the trainers match
the paper's described procedure, and the degradation configs implement the cited
RealBasicVSR/RealESRGAN pipeline. The defects above are missing-artefact and
wiring bugs, not invalid procedures.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                  |
|-------------|------------|--------------|------------------------------------------------------------------|
| missing     | 3          | high         | HQ-VSR pipeline absent; DOVER/FasterVQA/E*_warp deps absent; no timing script |
| bug         | 3          | high         | inference.sh GT hardcoded to UDM10; DOVER off-by-one; eval_ewarp paths/import broken |
| difference  | 1          | low          | batch-size and DIV2K image-count statements need launch context  |
| methodology | 0          | -            | No invalid procedure found; losses/degradation match paper       |

## 5. Closing lists

### Top take-aways (ranked by severity x confidence)
1. (missing, high) Section-3.3 HQ-VSR construction pipeline — and therefore the
   Table 1c/1d ablations — has no code in the repo (`video-pipeline-code-absent`).
2. (bug, high) `inference.sh` evaluates SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ
   against UDM10's ground truth, so the runner does not reproduce Table 2
   full-reference metrics (`inference-sh-gt-hardcoded-udm10`).
3. (missing, high) DOVER, FasterVQA (VBench), and E*_warp eval scripts import
   packages/modules absent from the repo, so three Table-2 columns + Fig. 1 DOVER
   are not reproducible (`vqa-warp-metric-deps-missing`).
4. (bug, medium) `eval_dover.py` mis-assigns per-clip DOVER scores with an
   off-by-one `i-1` index (`dover-offbyone-index`).
5. (bug, medium) `eval_ewarp.py` chdirs into a non-existent RAFT directory and
   imports a non-existent `ewarp` module (`eval-ewarp-paths-broken`).
6. (missing, medium) No script computes the running-time / "28x faster" headline
   numbers (`running-time-no-script`).

### Items that genuinely look fine
- Stage-1 latent MSE loss (`lora_one_s1_trainer.py:207`) matches Eq. 3.
- Stage-2 MSE + DISTS + frame-difference L1 loss (`lora_one_s2_trainer.py:238-290`)
  matches Eqs. 4/6/7, with the frame-diff term only on multi-frame (video) batches.
- φ=0.8 image ratio, lr 2e-5 / 5e-6, 10k / 500 iterations, `sr_noise_step=399`,
  seed 42 in the launchers match Section 4.1.
- Degradation configs implement the cited RealBasicVSR/RealESRGAN two-stage
  degradation; the RealVSR/MVSR4x `--upscale 1` (paired real data already at GT
  resolution) is consistent, not a defect.
- `eval_metrics.py` (PSNR/SSIM/LPIPS/DISTS/CLIP-IQA via pyiqa) is present and runs.

### Open questions for the authors
- Were the Table-2 DOVER/FasterVQA/E*_warp values produced by the in-repo scripts
  (currently non-importable), or by an external evaluation harness not released?
- Was Table 2 produced with per-dataset GT despite `inference.sh` hardcoding UDM10's
  GT, i.e. is the runner a copy-paste error rather than the procedure actually used?
- Which DIV2K variant (800 vs 900 images) and how many DP processes (for total
  batch 8) were used for the reported runs?
