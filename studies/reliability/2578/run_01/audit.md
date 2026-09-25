# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (#2578)

## 1. Summary

The repo `code/zhengchen1999__DOVE/` is the authors' code for DOVE, a one-step
video super-resolution (VSR) model built by fine-tuning CogVideoX1.5-5B. I
confirmed it is the right repo (README title, abstract, and citation all match
paper #2578). The repo contains: an inference script (`inference_script.py`),
the two-stage training code (`finetune/`, with stage-1 latent-MSE and stage-2
pixel MSE+DISTS+frame-diff losses faithfully implementing Eqs. 2–7), the
RealBasicVSR/RealESRGAN online degradation pipeline (`finetune/datasets/`,
`finetune/configs/degradation*.yaml`), evaluation scripts
(`eval_metrics.py`, `finetune/scripts/eval_*.py`), the empty-prompt embedding,
and demo videos. Trained DOVE weights and the HQ-VSR dataset are *not* in the
repo but are linked from the README (Google Drive). I read the paper PDF
(verifying Tables 1–3 and §3–4) and traced every Table-2 metric column and every
ablation row to code.

What I ran: read all training/inference/eval/degradation source; wrote
`_audit_code/check_missing_eval_modules.py` (read-only) to deterministically
verify which Table-2 metrics have computation code in the repo, whether the
HQ-VSR curation pipeline exists, and whether `pyiqa` is in `requirements.txt`.
Output: `_audit_code/out/missing_eval_modules.csv`.

The training/inference math is sound and matches the paper. The two reproducibility
gaps are: (a) the headline "video processing pipeline" that constructs the
HQ-VSR dataset (a stated contribution, used by the Table 1c/1d ablations) has no
code in the repo; (b) two of the eight Table-2 metric columns (E*warp, DOVER)
have no working in-repo computation — `eval_ewarp.py` imports a module
(`ewarp`) that does not exist anywhere in the repo, and the README's testing
section itself flags these scripts as TODO. Plus two helper-script bugs in the
evaluation drivers.

## 2. Result-traceability table

Paper artefacts checked against repo computation. "pyiqa" = computed by
`eval_metrics.py` / `inference_script.py` via the `pyiqa` library (PSNR, SSIM,
LPIPS, DISTS, CLIP-IQA, and FasterVQA are all available pyiqa metrics).

| Paper artefact | Repo location | Computed? | Status |
|---|---|---|---|
| Table 2 — PSNR, SSIM (fidelity) | `eval_metrics.py` (pyiqa) | yes | Present (cannot run numerically here; no weights/data/GPU) |
| Table 2 — LPIPS, DISTS, CLIP-IQA (perceptual) | `eval_metrics.py` (pyiqa) | yes | Present |
| Table 2 — FasterVQA | `eval_metrics.py` (pyiqa `fastervqa`) | yes | Present |
| Table 2 — DOVER | `finetune/scripts/eval_dover.py:150` → external `DOVER` pkg | partial | External dep, not in repo/requirements; per-sample off-by-one (see `dover-per-sample-offset`) |
| Table 2 — E*warp (temporal) | `finetune/scripts/eval_ewarp.py:149` → `from ewarp import Ewarp` | NO | MISSING module (see `ewarp-module-missing`) |
| Table 1a/1b/1c/1d ablations (PSNR/LPIPS/CLIP-IQA/DOVER on UDM10) | training shells + `eval_metrics.py`/`eval_dover.py` | partial | Training/eval code present; DOVER column shares the E*warp/DOVER gaps |
| Tab. 1c/1d — HQ-VSR construction (filter + motion ablations) | (none) | NO | MISSING curation pipeline (see `hq-vsr-pipeline-missing`) |
| Table 3 / Fig. 1 — running time (28× speed-up), step count | (none — wall-clock timing harness) | NO | Not scripted; secondary (timing is environment-dependent, weights absent) |
| DOVE trained weights (all numbers) | README Google-Drive links (not in repo) | — | External download (documented) |
| HQ-VSR dataset (2,055 videos) | README Google-Drive link (not in repo) | — | External download (documented) |

## 3. Findings

## missing

```yaml finding
id: hq-vsr-pipeline-missing
category: missing
topic: "data construction / video processing pipeline"
title: "Video processing pipeline that builds HQ-VSR (Sec 3.3, Fig 3) absent from repo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 80
line_end: 80
quote: |
  - [ ] Release the video processing pipeline.
claim: "The paper's four-step video processing pipeline (metadata/scene/quality filtering + optical-flow motion-mask bounding-box cropping, Sec 3.3 & Fig 3) — a stated contribution used to build the HQ-VSR dataset and ablated in Tables 1c and 1d — has no implementation in the repo; the README TODO marks it unreleased, and a repo-wide grep for scene detection, aesthetic/CLIP-IQA/FasterVQA/DOVER quality filtering, or motion-mask/bounding-box code finds nothing under finetune/datasets or finetune/scripts."
concern: "A core contribution and the +Filter/+Motion ablation rows (Table 1d) cannot be reproduced or audited; the released HQ-VSR.txt/dataset is the only artefact, with no code showing how 2,055 videos were curated from OpenVid-1M."
resolution: "Authors: release the curation pipeline code (scene detection, quality scoring with the four metrics, and the optical-flow motion-area cropping), or confirm the Table 1c/1d ablations were produced by code that is not in this repo."
cross_refs: ["§3.3", "Table 1c", "Table 1d", "Fig 3"]
check_script: _audit_code/check_missing_eval_modules.py
paper_ref: "Section 3.3 'Video Processing Pipeline'; Table 1c/1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ewarp-module-missing
category: missing
topic: "evaluation / temporal-consistency metric"
title: "eval_ewarp.py imports nonexistent `ewarp` module; E*warp (Table 2) not computable"
severity: medium
confidence: high
status: finding
file: finetune/scripts/eval_ewarp.py
line_start: 149
line_end: 149
quote: |
      from ewarp import Ewarp as Ewarp
claim: "eval_ewarp.py — the only script for the flow-warping error E*warp reported in every row of Table 2 — does `os.chdir(RAFT)` then `from ewarp import Ewarp`, but there is no `ewarp.py` anywhere in the repo (verified by find); the script therefore raises ImportError and cannot produce any E*warp value."
concern: "One of the eight Table-2 metric columns (temporal consistency, a headline claim for a VSR method) has no working computation in the repo, so that column is not reproducible from the released code."
resolution: "Authors: add the missing `ewarp.py` (the Ewarp implementation), as already flagged in the README testing TODO ('Add metric computation scripts for FasterVQA, DOVER, and E*warp')."
cross_refs: ["readme-eval-todo", "Table 2 E*warp"]
check_script: _audit_code/check_missing_eval_modules.py
paper_ref: "Table 2, E*warp column; Section 4.1 Evaluation Metrics"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pyiqa-deps-unlisted
category: missing
topic: "dependencies / environment"
title: "pyiqa (and DOVER/VBench) required for metrics but absent from requirements.txt"
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
claim: "eval_metrics.py and inference_script.py both `import pyiqa` (the library that computes PSNR/SSIM/LPIPS/DISTS/CLIP-IQA/FasterVQA for Table 2), and eval_dover.py / eval_vbench.py import external `DOVER` / `VBench` packages, but none of pyiqa, DOVER, or VBench appears in requirements.txt; `pip install -r requirements.txt` yields an environment that cannot run the evaluation."
concern: "The reproduction environment is under-specified: a fresh install per requirements.txt fails on the metric scripts; the README mentions `pip install pyiqa` only in prose, and DOVER/VBench have no install instructions or pinned versions."
resolution: "Authors: add pyiqa (pinned) to requirements.txt and document the exact DOVER / VBench versions and install steps used for Table 2."
cross_refs: ["ewarp-module-missing"]
check_script: _audit_code/check_missing_eval_modules.py
paper_ref: "Section 4.1 Evaluation Metrics; Table 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: inference-sh-wrong-gt-path
category: bug
topic: "evaluation driver script"
title: "inference.sh evaluates SPMCS/YouHQ40/RealVSR/MVSR4x against UDM10 ground truth"
severity: medium
confidence: high
status: finding
file: inference.sh
line_start: 21
line_end: 25
quote: |
  python eval_metrics.py \
      --gt datasets/test/UDM10/GT \
      --pred results/DOVE/SPMCS \
      --metrics psnr,ssim,lpips,dists,clipiqa
claim: "The provided reproduction script hardcodes `--gt datasets/test/UDM10/GT` for the SPMCS block (line 23) and identically for YouHQ40 (line 35), RealVSR (line 48), and MVSR4x (line 61), instead of each dataset's own GT folder; eval_metrics.py matches predictions to GT by filename stem and skips any pred with no matching GT name, so this either evaluates against the wrong reference or processes zero samples."
concern: "Running inference.sh as shipped does not reproduce the Table-2 full-reference metrics (PSNR/SSIM/LPIPS/DISTS) for four of the five FR datasets — the eval points at UDM10's GT for all of them."
resolution: "Authors: fix each eval block's `--gt` to point at the corresponding dataset's GT directory (e.g. `datasets/test/SPMCS/GT`)."
cross_refs: ["Table 2"]
paper_ref: "Table 2 (SPMCS, YouHQ40, RealVSR, MVSR4x rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-per-sample-offset
category: bug
topic: "evaluation / DOVER aggregation"
title: "eval_dover.py off-by-one maps each clip name to the previous clip's DOVER score"
severity: low
confidence: high
status: finding
file: finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
      for i, name in enumerate(pred_names):
          results[name] = dover_results[i-1]
claim: "When assembling per-sample DOVER results, the loop indexes `dover_results[i-1]`, so for i=0 the first clip name receives `dover_results[-1]` (the last score) and every other name receives the preceding clip's score — a cyclic shift of the name→score mapping."
concern: "Per-sample DOVER values are mislabeled; the overall average over `results` (np.mean) is unaffected because it sums the same set of values, so the reported Table-2 DOVER averages are not changed, but any per-clip DOVER use would be wrong."
resolution: "Authors: change `dover_results[i-1]` to `dover_results[i]`."
cross_refs: ["Table 2 DOVER column"]
paper_ref: "Table 2, DOVER column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

No methodologically-sound-but-paper-mismatched findings. The stage-1 latent MSE
(Eq. 3), stage-2 MSE+DISTS+frame-difference losses (Eqs. 4,6,7), φ=0.8 image
ratio, t=399 timestep, lr/iteration counts, and batch size all match the
training shells and the paper's Section 4.1. (Note: the stage-2 trainer also
contains unused edge-aware loss branches `ea_dists`/`ea_lpips`, but the shipped
`train_ddp_one_s2.sh` enables only `--dists_weight 1.0` and
`--frame_diff_weight 1.0`, matching the paper, so the extra branches are dead
code, not a discrepancy.)

## methodology

No methodology findings. This is a generative VSR task: training data (HQ-VSR +
DIV2K) and the six test sets are disjoint datasets, LQ is synthesized online
from HQ via the standard RealBasicVSR/RealESRGAN degradation, and evaluation is
per-clip reference/no-reference IQA/VQA — there is no classifier train/test
split, no label, and no leakage vector of the kind the checklists target.
Pretraining-contamination is N/A in the supervised-overlap sense (CogVideoX is a
generic T2V prior, and the test sets are standard public VSR benchmarks).

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 3          | high         | HQ-VSR curation pipeline absent; E*warp module absent; metric deps unlisted |
| bug         | 2          | medium       | inference.sh wrong GT path; eval_dover per-sample off-by-one |
| difference  | 0          | -            | Training/inference math matches the paper                  |
| methodology | 0          | -            | Generative VSR; no split/leakage/baseline issue found      |

## 5. Closing lists

**Top take-aways** (ranked by severity × confidence):
1. [missing] `hq-vsr-pipeline-missing` — the video processing pipeline that
   builds HQ-VSR (a headline contribution, used in Table 1c/1d) has no code;
   README TODO confirms it is unreleased. (high / high)
2. [missing] `ewarp-module-missing` — `eval_ewarp.py` imports a nonexistent
   `ewarp` module, so the E*warp temporal-consistency column of Table 2 is not
   computable from the repo. (medium / high)
3. [bug] `inference-sh-wrong-gt-path` — the shipped reproduction script
   evaluates four of five FR datasets against UDM10's GT, so Table-2 FR metrics
   for SPMCS/YouHQ40/RealVSR/MVSR4x are not reproduced as-shipped. (medium / high)
4. [missing] `pyiqa-deps-unlisted` — pyiqa/DOVER/VBench needed for metrics but
   not in requirements.txt; the env from requirements alone cannot run eval.
   (low / high)
5. [bug] `dover-per-sample-offset` — per-sample DOVER mislabeled by a cyclic
   off-by-one; reported averages unaffected. (low / high)

**Items that genuinely look fine**:
- Stage-1 / stage-2 trainers implement Eqs. 2–7 faithfully (velocity
  prediction, latent MSE, pixel MSE+DISTS+frame-diff, φ image-ratio mixing).
- Default `sr_noise_step=399` matches the paper's empirically-set t=399.
- The RealBasicVSR/RealESRGAN two-order online degradation (configs +
  `degradation.py`) matches the paper's stated LQ synthesis.
- PSNR/SSIM/LPIPS/DISTS/CLIP-IQA/FasterVQA are all computable via pyiqa in
  `eval_metrics.py`; the Y-channel conversion uses standard BT.601 coefficients.
- Trained DOVE weights and the HQ-VSR dataset, though not in the repo, are
  linked and documented in the README (legitimate external downloads).
- No hardcoded absolute paths in the Python sources (the `/data2/...` strings
  are example CLI args in the README, not in code).

**Open questions for the authors**:
- Was DOVER (and E*warp) for Table 2 computed with the in-repo scripts or an
  external harness? If external, which DOVER/RAFT/ewarp versions?
- Will the HQ-VSR curation pipeline be released so Table 1c/1d are reproducible?
- Is there a script for the Table-3 / Fig-1 running-time (28×) measurement, or
  was timing measured manually?
