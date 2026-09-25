# Audit — HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance (NeurIPS 2025, #4376)

## 1. Summary

The repo (`code/Bujiazi__HiFlow/`) is the official HiFlow code. It is a **single
inference entrypoint**: `run_hiflow.py` builds a customized Flux.1.0-dev pipeline
(`flux_pipeline_hiflow.py`, `transformer_flux.py`, `utils.py`) and generates **one
image per run** through the cascade 1024 → 2048 → 4096. There are exactly 4 Python
files, a `requirements.txt`, an `inference.sh`, and a README. The flow-aligned
guidance algorithm (initialization / direction / acceleration alignment) described
in the paper is faithfully implemented in `flux_pipeline_hiflow.py`
(`flowmatch_step`, lines 1007–1108) and matches the appendix pseudocode (Algorithm 1).

HiFlow is **training-free**, so there is correctly no training code (N/A). However,
the paper's *entire quantitative substance* — Table 1 (FID/FIDpatch/IS/ISpatch/CLIP
for HiFlow + 4 baselines at 2K and 4K), Table 2 (latency), and Table 3 (ablation) —
depends on machinery that is **not in the repo**: the 1K caption set, the 10K
LAION-High-Resolution reference images, the FID/IS/CLIP/patch metric computation,
the four baseline pipelines, the latency-timing harness, and the ablation driver.
I verified the absence programmatically.

Scripts I ran (read-only, under `_audit_code/`):
- `check_eval_artifacts.py` → `out/eval_artifacts.json`: greps all repo `.py` for
  FID/IS/CLIP/patch-metric/prompt-set/LAION/baseline/timing tokens. All ABSENT
  (the lone "Inception Score" hit is the bare substring `IS` in unrelated code).
- `check_run_config.py` → `out/run_config.txt`: extracts the default `run_hiflow.py`
  config and compares the cascade / LoRA settings against the paper.

## 2. Traceability table

Repo paths are relative to `code/Bujiazi__HiFlow/`. "Computes value" means code that
*produces* the reported number, not code that could in principle be run with
artefacts that are themselves absent.

| Paper artefact | Repo location | Computes value | Matches paper | Status |
|---|---|---|---|---|
| Tab.1 FID/FIDpatch/IS/ISpatch/CLIP, HiFlow @2K & @4K | (none) | — | — | MISSING (no metric code, no prompts, no LAION ref set) |
| Tab.1 same metrics, DemoFusion / DiffuseHigh / I-Max / BSRGAN @2K & @4K | (none) | — | — | MISSING (no baseline implementations) |
| Tab.2 Latency (sec.) HiFlow + 3 baselines @2K & @4K | (none) | — | — | MISSING (no timing harness, no baselines) |
| Tab.3 Ablation FID/IS/CLIP for −Ai/−Ad/−Aa | (none) | — | — | MISSING (no ablation driver; flags exist but no metric eval) |
| Fig.3a "FID score" curve / Fig.3 component analyses | (none) | — | — | MISSING |
| Fig.5 / Fig.6 / Fig.7 / Fig.8 qualitative panels | `run_hiflow.py` + pipeline | image only | n/a (visual) | PARTIAL — pipeline generates images; exact prompts/seeds for paper figures not all in repo |
| Eval prompts ("1K high-quality captions", App. Tab.H.1–H.3) | (none) | — | — | MISSING (prompt set not shipped) |
| Flow-aligned guidance algorithm (init/dir/accel alignment) | `flux_pipeline_hiflow.py:1007-1108` | algorithm faithfully implemented | ✓ (matches Alg.1) | Verified |
| Cascade schedule τ=[0.6,0.3,0.3] for 1K→2K→3K→4K | `run_hiflow.py:38-46` (1K→2K→4K) | runs, valid | ✗ default skips 3K stage | MISMATCH (low) |
| Table-1 config = base Flux (no LoRA) | `run_hiflow.py:18` loads LoRA by default | runs, valid | ✗ demo default loads a LoRA | MISMATCH (low) |

Routing: every MISSING row is a paper-described step that nothing in the repo
implements → category `missing` (consolidated into one owner finding per the
Single-Owner Rule, with sub-aspects cross-referenced). The two MISMATCH rows are
config differences where the code's behaviour is itself valid → `difference`.

## 3. Findings

## missing

```yaml finding
id: no-evaluation-pipeline
category: missing
topic: "result traceability / evaluation code"
title: "No code, prompts, or reference data to compute any reported metric (Tab.1/2/3)"
severity: high
confidence: high
status: finding
file: run_hiflow.py
line_start: 1
line_end: 52
quote: |
  import torch
  from flux_pipeline_hiflow import FluxPipeline
  from transformer_flux import FluxTransformer2DModel
  from utils import set_seeds
  import pdb
claim: "The repo's only entrypoint, run_hiflow.py, generates a single image from one hard-coded prompt; there is no FID/IS/CLIP/patch-metric computation, no 1K caption set, no 10K LAION-High-Resolution reference images, and no latency-timing or ablation driver anywhere in the 4 Python files."
concern: "Every quantitative claim in the paper (Tab.1 FID/FIDpatch/IS/ISpatch/CLIP at 2K and 4K, Tab.2 latency, Tab.3 ablation) is unreproducible from the repo: none of the values can be recomputed and none can be checked against the paper."
resolution: "Authors: please release the evaluation scripts (FID/IS/CLIP and the patch variants), the 1K caption list, the resolvable LAION-High-Resolution reference subset (or its image IDs), and the ablation/latency drivers."
cross_refs: ["no-baseline-implementations", "eval-prompt-set-missing", "§4.1", "§4.2", "§4.3"]
check_script: _audit_code/check_eval_artifacts.py
paper_ref: "Tables 1, 2, 3; §4.1 Evaluation"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-baseline-implementations
category: missing
topic: "baselines"
title: "Baseline methods (DemoFusion, DiffuseHigh, I-Max, BSRGAN) absent from repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  Baselines. The compared methods encompass DemoFusion [12], DiffuseHigh [26], I-Max [13], and
  an image super-resolution method, BSRGAN [58]. For a fair comparison, the training-free methods
  are tested on the same Flux model according to their official implementations.
claim: "Table 1 and Table 2 compare HiFlow against four baselines run on Flux, but the repo contains no baseline code, configs, or pointers to the exact adaptations used."
concern: "The 'fair comparison' and 'same Flux model' claims cannot be verified, and the comparative numbers cannot be reproduced, without the adapted baseline pipelines."
resolution: "Authors: release (or link to the exact forks/commits of) the Flux-adapted DemoFusion, DiffuseHigh, I-Max, and BSRGAN pipelines used to produce Tables 1–2."
cross_refs: ["no-evaluation-pipeline"]
paper_ref: "§4.1 Baselines; Tables 1, 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eval-prompt-set-missing
category: missing
topic: "data availability"
title: "1K evaluation caption set and 10K LAION reference subset not shipped"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  We collect 1K high-quality captions across various scenarios for diverse image generation. The CLIP [43] score is used to assess the prompt-following capability, Frechet Inception
  Distance [18] (FID) and Inception Score [47] (IS) are reported to measure image quality, in which FID
  is calculated between generated images and 10K real high-quality images (with at least 1024 × 1024
  resolution) sourced from LAION-High-Resolution [48].
claim: "The evaluation uses 1K self-collected captions and a 10K-image LAION-High-Resolution reference set; neither the caption file nor the image IDs/accessions for the reference subset are in the repo."
concern: "FID/IS depend on the exact reference image distribution and the exact prompt set; without them the reported metrics cannot be reproduced or audited for selection effects."
resolution: "Authors: include the caption list (paper says App. Tab.H.1–H.3) as a machine-readable file and provide the LAION image IDs or a fetch script for the 10K reference subset."
cross_refs: ["no-evaluation-pipeline"]
paper_ref: "§4.1 Evaluation; Appendix §C"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No findings. The pipeline is internally consistent; `import pdb` is unused but
harmless, and the flow-aligned-guidance reshaping (swin-patchify pad/crop in
`flowmatch_step`) is self-consistent. N/A for crashes — code was not executed
(requires an A100-class GPU and FLUX.1-dev download), but no static defect that
contradicts the code's own intent was found.

## difference

```yaml finding
id: default-loads-lora-not-table1-config
category: difference
topic: "evaluation consistency (config)"
title: "Default run loads an aesthetic LoRA, unlike the base-Flux Table-1 setup"
severity: low
confidence: high
status: finding
file: run_hiflow.py
line_start: 17
line_end: 22
quote: |
  # LoRA can be downloaded from https://civitai.com/models/832683/flux-pro-11-style-lora-extreme-detailer-for-flux-illustrious
  pipe.load_lora_weights("./lora_models/aidmaFLUXPro1.1-FLUX-v0.3.safetensors") # optional

  set_seeds(seed)

  prompt = "A robot standing in the rain reading newspaper, rusty and worn down, in a dystopian cyberpunk street, photo-realistic, urbanpunk. aidmaFLUXPro1.1"
claim: "The shipped default entrypoint loads the aidmaFLUXPro1.1 detail-enhancing LoRA (and inference.sh downloads it), whereas the Table-1 quantitative comparison uses base Flux.1.0-dev with no LoRA."
concern: "A reader running the repo out-of-the-box reproduces a LoRA-enhanced demo, not the base-model configuration behind the reported metrics; the divergence is cosmetic since the call is labelled optional and the metric setup is base Flux."
resolution: "Authors: comment out or gate the LoRA call by default, or add a base-Flux config matching the Table-1 evaluation."
cross_refs: ["no-evaluation-pipeline"]
paper_ref: "§4.1; Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: default-cascade-skips-3k-stage
category: difference
topic: "evaluation consistency (config)"
title: "Default cascade is 1K→2K→4K; paper specifies a 1K→2K→3K→4K schedule"
severity: low
confidence: medium
status: finding
file: run_hiflow.py
line_start: 38
line_end: 39
quote: |
      target_heights = [2048, 4096], 
      target_widths = [2048, 4096], 
claim: "The default script upscales 1024→2048→4096 (two upscale steps), but the paper's implementation details state the noise-adding ratio tau = [0.6, 0.3, 0.3] for a 1K→2K→3K→4K cascade (three upscale steps, including a 3072 stage)."
concern: "The shipped default cascade does not match the cascade schedule the paper describes for 4K generation; results may differ from the paper's 4K numbers, though both schedules are valid and the parameters are user-configurable."
resolution: "Authors: confirm which cascade (2-stage vs 3-stage to 4K) produced the Table-1 4K numbers and ship that as the default, or document that the intermediate 3K stage is optional."
cross_refs: []
check_script: _audit_code/check_run_config.py
paper_ref: "§4.1 Implementation details ('tau ... [0.6, 0.3, 0.3] for 1K → 2K → 3K → 4K')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No findings. HiFlow is a training-free inference-time method with no train/test
split, no learned parameters, and no statistical-test claims, so the leakage /
sample-independence / tuning-on-test / statistical-integrity checklists are
**N/A** (structurally inapplicable). The implemented guidance algorithm matches the
paper's equations and pseudocode; I found no invalid procedure in the code itself.
Note: the *inability to verify* baseline fairness and metric computation is owned by
the `missing` findings above, not re-filed here.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 3          | high         | No metric/eval code, no baselines, no prompts/reference data — all of Tab.1/2/3 unreproducible |
| bug         | 0          | -            | Pipeline internally consistent; no intent-contradicting defect found |
| difference  | 2          | low          | Default run uses a LoRA and a 2-stage cascade, unlike the paper's base-Flux / 3-stage-to-4K eval config |
| methodology | 0          | -            | Training-free inference method; split/leakage/tuning/stats checklists N/A; algorithm matches paper |

### Top take-aways (≤6, ranked)
1. **[missing] No evaluation pipeline (`no-evaluation-pipeline`, high/high).** Repo
   has 4 Python files and one demo entrypoint; no FID/IS/CLIP/patch-metric code,
   no prompts, no reference data. The entire quantitative case of the paper
   (Tab.1, Tab.2, Tab.3) is unreproducible from the artefact.
2. **[missing] Baselines absent (`no-baseline-implementations`, medium/high).** The
   four comparison methods are not in the repo, so the "fair comparison on the same
   Flux model" claim cannot be checked.
3. **[missing] Prompt set + LAION reference subset missing (`eval-prompt-set-missing`,
   medium/high).** FID/IS reference distribution and the 1K captions are not shipped.
4. **[difference] Default loads a LoRA, not base Flux (`default-loads-lora-not-table1-config`,
   low/high).** Out-of-the-box run does not reproduce the Table-1 base-model setup.
5. **[difference] Default cascade skips the 3K stage (`default-cascade-skips-3k-stage`,
   low/medium).** Paper describes 1K→2K→3K→4K; default script does 1K→2K→4K.

### Items that genuinely look fine
- The flow-aligned guidance core (`flowmatch_step`, `flux_pipeline_hiflow.py:1007-1108`)
  faithfully implements initialization, direction (FFT/DWT low-frequency fusion),
  and acceleration alignment, and matches appendix Algorithm 1 and Eqs. 9–13.
- Dependencies are pinned in `requirements.txt` (diffusers/transformers/torch versions
  fixed); the environment is rebuildable.
- Seeding (`utils.set_seeds`) sets python/numpy/torch/cuda seeds plus a generator,
  consistent with the paper's multi-seed (0/1/2) qualitative claim.
- Code is self-contained for its stated purpose (auto-downloads FLUX.1-dev); the GPU
  requirement (A100) is a legitimate hardware dependency, not an unexplained failure.

### Open questions for the authors
- Which cascade schedule (2-stage to 4K vs the paper's 3-stage 1K→2K→3K→4K) produced
  the Table-1 4K numbers? (`default-cascade-skips-3k-stage`)
- Were Table-1 metrics computed with base Flux only, or with any LoRA? (Confirms the
  default LoRA load is purely a demo, `default-loads-lora-not-table1-config`.)
