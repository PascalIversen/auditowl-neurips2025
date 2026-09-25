# Code-repository audit — DOVE (NeurIPS 2025, paper 2578)

## 1. Summary

The repo `zhengchen1999/DOVE` is the authors' code for an efficient one-step
diffusion VSR model built by fine-tuning CogVideoX1.5-5B. It contains: a
dependency spec (`requirements.txt`), two-stage training code (`finetune/`,
launched via `train_ddp_one_s1.sh` / `train_ddp_one_s2.sh`), an inference
entrypoint (`inference_script.py` + `inference.sh`), a metric script
(`eval_metrics.py`), additional metric scripts under `finetune/scripts/`
(`eval_dover.py`, `eval_vbench.py`, `eval_ewarp.py`, `eval_metrics.py`), the
RealBasicVSR-style degradation pipeline (`finetune/datasets/`,
`configs/degradation*.yaml`), a README with reproduction commands, and
Google-Drive links to the HQ-VSR dataset, test sets, and pretrained weights.

I read the paper (PDF + text extraction) and the code, focused on the headline
quantitative claims (Table 2 SOTA comparison, Table 1 ablations, Table 3 / Fig 1
timing and DOVER). I cross-checked the training hyper-parameters and loss
implementation against Sec 3.2 / Sec 4.1, traced every reported metric to the
script that computes it, and ran one deterministic check script
(`_audit_code/check_eval_artefacts.py`, output `_audit_code/out/eval_artefacts.csv`)
covering: external-dependency presence for the three video-metric scripts, the
`--gt` argument in `inference.sh`, the off-by-one index in `eval_dover.py`, and a
repo-wide search for the Sec 3.3 data-construction pipeline. All checks are
read-only on `code/`.

Overall the core method is faithfully implemented: the training config
(10k stage-1 iters @ lr 2e-5, 500 stage-2 iters @ lr 5e-6, image ratio φ=0.8,
batch 2×4 GPUs = 8, `sr_noise_step=399`, DISTS weight = frame-diff weight = 1)
matches Sec 4.1, and the Stage-2 loss (MSE + DISTS + L1 frame-difference) matches
Eqs. (4)-(7). The findings concern (a) the absence of the Sec 3.3 video-processing
pipeline code, (b) the three video-quality metric scripts that import external
packages not bundled in the repo, and (c) two scripting bugs in the evaluation
helpers plus a wrong ground-truth path in `inference.sh`.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Tab. 2 PSNR/SSIM/LPIPS/DISTS/CLIP-IQA (all 6 datasets) | `eval_metrics.py` (pyiqa psnr,ssim,lpips,dists,clipiqa) | not re-run (no GPU/weights) | — | Code present; computation traceable |
| Tab. 2 / Tab. 3 / Fig. 1 DOVER | `finetune/scripts/eval_dover.py` | — | — | PARTIAL: imports absent `DOVER` pkg; off-by-one index (see `eval-dover-off-by-one`, `video-metric-scripts-missing-deps`) |
| Tab. 2 FasterVQA | (none; pyiqa supports `fastervqa` but no script calls it) | — | — | MISSING script (see `video-metric-scripts-missing-deps`) |
| Tab. 2 E*warp (flow warping error) | `finetune/scripts/eval_ewarp.py` | — | — | PARTIAL: imports absent `ewarp` module; wrong default RAFT path (see `eval-ewarp-broken`, `video-metric-scripts-missing-deps`) |
| Tab. 1a/b/c/d ablations (PSNR/LPIPS/CLIP-IQA/DOVER on UDM10) | training code + `eval_metrics.py`/`eval_dover.py` | — | — | PSNR/LPIPS/CLIP-IQA traceable; DOVER column depends on broken script |
| Tab. 1d "+Filter / +Motion" pipeline ablation; HQ-VSR (2,055 videos) construction | (none — Sec 3.3 pipeline not in repo) | — | — | MISSING (see `pipeline-code-missing`) |
| Tab. 3 / Fig. 1 running time (s) | (none — no timing harness in repo) | — | — | MISSING timing script (not separately filed; uninstrumented but non-conclusion-critical) |
| Sec 3.1/3.2 one-step v-prediction inference (Eq. 2), training losses (Eqs. 3-7) | `inference_script.py:394-503`, `finetune/models/dove/lora_one_s2_trainer.py:124-297` | matches | ✓ | Verified by reading |

Notes: PSNR/SSIM/LPIPS/DISTS/CLIP-IQA are fully covered. The video-quality
metrics (DOVER, FasterVQA, E*warp) — which appear in Tab. 2 for every dataset
and in the headline Fig. 1 / Tab. 3 — are not reproducible from the repo as
shipped (missing external code and two bugs). I did not re-execute any metric
(no GPU / multi-hundred-GB weights), so all "Matches paper" cells for the
computable metrics are left "—" rather than asserted.

## 3. Findings

## missing

```yaml finding
id: video-metric-scripts-missing-deps
category: missing
topic: "result traceability / evaluation code"
title: "DOVER, FasterVQA, E*warp metric scripts depend on external code not in repo"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 150
line_end: 150
quote: |
      from DOVER.evaluate_a_set_of_videos import evaluate_set as dover
claim: "eval_dover.py imports a DOVER package, eval_vbench.py imports VBench/evaluate, and eval_ewarp.py imports an ewarp module; none of DOVER/, VBench/, or ewarp.py exist anywhere in the repo (verified repo-wide in _audit_code/out/eval_artefacts.csv), and FasterVQA has no calling script at all."
concern: "DOVER, FasterVQA, and E*warp are reported in Table 2 for every dataset and headline the abstract/Fig. 1/Tab. 3, yet none of these numbers can be reproduced from the shipped code without third-party packages the repo neither vendors nor pins; the README's own TODO 'Add metric computation scripts for FasterVQA, DOVER, and E*warp' confirms they are unfinished."
resolution: "Authors: bundle or pin (with exact versions/commits) the DOVER, VBench, and RAFT/ewarp dependencies, and add a FasterVQA computation script, so Table 2's VQA and temporal-consistency columns are reproducible."
cross_refs: ["eval-dover-off-by-one", "eval-ewarp-broken"]
check_script: _audit_code/check_eval_artefacts.py
paper_ref: "Table 2 (FasterVQA, DOVER, E*warp rows); Table 3; Fig. 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pipeline-code-missing
category: missing
topic: "dataset construction"
title: "Sec 3.3 video-processing pipeline (HQ-VSR construction) not in repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  Step 1: Metadata Filtering. ... Step 2: Scene Filtering. ... Step 3: Quality
  Filtering. ... Step 4: Motion Processing. ... we introduce the motion area
  detection algorithm for localized processing
claim: "The paper describes a four-step pipeline (metadata/scene/quality/motion filtering with aesthetic, CLIP-IQA, FasterVQA, DOVER scoring and optical-flow motion-area cropping per Eq. 8) used to build HQ-VSR and ablated in Table 1d; a repo-wide search finds no code implementing scene detection, quality-score filtering, or motion-area cropping (only CLIP-IQA used as an eval metric)."
concern: "Table 1d's '+Filter/+Motion' ablation and the HQ-VSR dataset construction cannot be reproduced or audited; the README TODO 'Release the video processing pipeline' is explicitly unchecked, confirming the omission."
resolution: "Authors: release the data-construction pipeline code (the four filtering steps and motion-area cropping) so HQ-VSR and Table 1d can be reproduced."
cross_refs: ["§3.3", "Table 1d"]
check_script: _audit_code/check_eval_artefacts.py
paper_ref: "Section 3.3; Table 1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: eval-dover-off-by-one
category: bug
topic: "evaluation code"
title: "eval_dover.py assigns DOVER scores with an off-by-one index"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
      for i, name in enumerate(pred_names):
          results[name] = dover_results[i-1]
claim: "Inside `for i, name in enumerate(pred_names)` the code indexes `dover_results[i-1]`, so the first clip (i=0) is assigned the LAST clip's score (index -1) and every clip is shifted by one relative to its name."
concern: "Per-clip DOVER scores are mismatched to clip names; the reported per-dataset DOVER average (Table 2/Tab. 3/Fig. 1) is computed over a misaligned mapping (the mean is unchanged, but per-sample values and any name-based aggregation are wrong, and it signals the script was not validated)."
resolution: "Change `dover_results[i-1]` to `dover_results[i]`; confirm the order returned by `evaluate_set` matches `sorted(pred_names)`."
cross_refs: ["video-metric-scripts-missing-deps"]
check_script: _audit_code/check_eval_artefacts.py
paper_ref: "Table 2, DOVER rows"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eval-ewarp-broken
category: bug
topic: "evaluation code"
title: "eval_ewarp.py chdirs to script dir and uses a non-existent default RAFT path"
severity: low
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
claim: "The script chdirs into `<script_dir>/RAFT` and imports `ewarp`, but no `RAFT/` subdirectory or `ewarp.py` exists under finetune/scripts (RAFT lives at finetune/utils/RAFT and has no ewarp.py); its default `--model finetune/scripts/models/raft-things.pth` also points to a path that does not exist (the weight is at finetune/utils/RAFT/raft-things.pth)."
concern: "eval_ewarp.py cannot run as shipped, so the E*warp temporal-consistency column of Table 2 (a headline 'excellent temporal consistency' claim) is not reproducible from the repo."
resolution: "Authors: vendor the `ewarp` implementation, fix the chdir target to the real RAFT location, and correct the default `--model` path."
cross_refs: ["video-metric-scripts-missing-deps"]
check_script: _audit_code/check_eval_artefacts.py
paper_ref: "Table 2, E*warp rows; Fig. 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: inference-sh-wrong-gt
category: bug
topic: "evaluation driver"
title: "inference.sh passes UDM10 ground truth for every dataset's eval"
severity: medium
confidence: high
status: finding
file: inference.sh
line_start: 34
line_end: 37
quote: |
  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/YouHQ40 \
      --metrics psnr,ssim,lpips,dists,clipiqa
claim: "All six eval_metrics.py invocations in inference.sh hardcode `--gt datasets/test/UDM10/GT`, even for SPMCS, YouHQ40, RealVSR, MVSR4x, and VideoLQ (verified: 6/6 --gt lines point to UDM10/GT in _audit_code/out/eval_artefacts.csv)."
concern: "Run verbatim, the reference-based metrics (PSNR/SSIM/LPIPS/DISTS) for the five non-UDM10 datasets are computed against the wrong ground truth; eval_metrics.py silently skips clips whose names lack a matching GT file (eval_metrics.py:163-179), so most clips would be dropped, producing meaningless Table 2 fidelity numbers unless the user edits the script."
resolution: "Authors: set `--gt` to each dataset's own GT directory in inference.sh."
cross_refs: []
check_script: _audit_code/check_eval_artefacts.py
paper_ref: "Table 2 (per-dataset fidelity metrics)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

No findings. The implemented inference (one-step v-prediction, `sr_noise_step=399`,
empty-prompt embedding) and the Stage-1/Stage-2 losses match the paper's
descriptions (Eqs. 2-7) and the reported hyper-parameters (Sec 4.1). The
`--upscale 1` used for RealVSR/MVSR4x in `inference.sh` is consistent with those
being real captured LQ-HQ pairs at equal resolution, not a contradiction of the
"×4" claim (which concerns the synthetic-degradation datasets).

## methodology

No findings. This is a fine-tuning / generation task with fixed public benchmark
test sets (UDM10, SPMCS, YouHQ40, RealVSR, MVSR4x, VideoLQ) disjoint from the
training data (HQ-VSR from OpenVid-1M + DIV2K), so train/test leakage,
split-construction, and pair-split shortcut concerns are structurally
inapplicable. Baselines are established VSR/ISR methods compared on identical
benchmarks. No statistical-significance testing is claimed, so statistical-integrity
checks are N/A. The pretraining-contamination topic (CogVideoX priors) is a
generic generative prior, not memorisation of these test sets, and is out of
scope for this audit's evidence.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                  |
|-------------|------------|--------------|------------------------------------------------------------------|
| missing     | 2          | medium       | Sec 3.3 pipeline code absent; DOVER/FasterVQA/E*warp eval deps absent |
| bug         | 3          | medium       | eval_dover off-by-one; eval_ewarp broken; inference.sh wrong --gt |
| difference  | 0          | -            | Core method faithfully implemented                               |
| methodology | 0          | -            | Standard benchmark eval; no leakage / stat-integrity issues found |

### Top take-aways (≤6, ranked by severity × confidence)
1. (`missing`) DOVER / FasterVQA / E*warp metric scripts import external packages (`DOVER`, `VBench`, `ewarp`) absent from the repo and have no FasterVQA script — these Table 2 / Fig. 1 / Tab. 3 columns are not reproducible as shipped. [medium severity, high confidence]
2. (`bug`) `inference.sh` passes `datasets/test/UDM10/GT` as the ground truth for all six datasets; the documented reproduction commands compute fidelity metrics against the wrong references. [medium, high]
3. (`missing`) The Sec 3.3 four-step video-processing pipeline (HQ-VSR construction; Table 1d ablation) is not in the repo; README TODO confirms it is unreleased. [medium, high]
4. (`bug`) `eval_dover.py:159` uses `dover_results[i-1]` inside an `enumerate` loop, misaligning per-clip DOVER scores to clip names. [medium, high]
5. (`bug`) `eval_ewarp.py` chdirs to a non-existent `RAFT/` dir, imports an absent `ewarp` module, and defaults to a wrong RAFT weight path. [low, high]

### Items that genuinely look fine
- Training hyper-parameters in `train_ddp_one_s1.sh` / `train_ddp_one_s2.sh` match Sec 4.1 (10k/500 iters, lr 2e-5/5e-6, φ=0.8, batch 2×4=8, `sr_noise_step=399`, λ1=λ2=1).
- Stage-2 loss in `lora_one_s2_trainer.py:236-290` implements MSE + DISTS + L1 frame-difference exactly as Eqs. (4)-(7); the latent/pixel one-step v-prediction in `inference_script.py:394-503` matches Eq. (2).
- `eval_metrics.py` correctly computes PSNR/SSIM/LPIPS/DISTS/CLIP-IQA via pyiqa with proper FR/NR handling and resolution matching; dependency spec, training, inference, README, and pretrained-weight links are all present.

### Open questions for the authors
- Were the Table 2 DOVER / FasterVQA / E*warp numbers produced by an off-repo (private) version of these scripts? If so, please release the exact code used.
- Were the Table 2 fidelity metrics in the paper computed with the correct per-dataset GT (i.e., the `inference.sh` `--gt` bug is only in the shipped driver, not in the reported run)?
