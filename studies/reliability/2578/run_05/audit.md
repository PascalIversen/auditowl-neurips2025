# Audit — DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution (paper 2578)

## 1. Summary

The repository `code/zhengchen1999__DOVE/` is the authors' official code for the
NeurIPS 2025 paper "DOVE". Its README, citation block, and asset links all match
this paper, so it is the author code. DOVE fine-tunes CogVideoX1.5-5B into a
one-step video super-resolution (VSR) model via a two-stage "latent–pixel"
strategy. The repo contains: inference (`inference_script.py`, `inference.sh`),
full-reference / IQA metric evaluation (`eval_metrics.py`,
`finetune/scripts/eval_metrics.py`), VQA / temporal-consistency metric stubs
(`finetune/scripts/eval_dover.py`, `eval_vbench.py`, `eval_ewarp.py`), two-stage
training (`finetune/train.py`, `train_ddp_one_s{1,2}.sh`, trainers under
`finetune/models/dove/`), degradation configs, and a RAFT optical-flow vendored
copy. Pretrained weights and the HQ-VSR dataset are hosted on Google Drive
(not in-repo).

What I did: read the paper (PDF + text extraction) and mapped every Table/Figure
number to a producing script; read the inference, evaluation, optimizer, and both
training trainers; and ran three read-only checks under `_audit_code/`:
`check_inference_gt_paths.py` (parses the `--gt`/`--pred` pairs in
`inference.sh`), `check_dover_offbyone.py` (reproduces the per-sample index map in
`eval_dover.py`), and `check_eval_imports.py` (filesystem existence of the
external modules the VQA/Ewarp scripts import). Outputs are in `_audit_code/out/`.

I did NOT run the model (requires CogVideoX weights, the HQ-VSR/test datasets, and
A100-class GPUs, none of which are in the sandbox), so dynamic findings are noted
as such.

## 2. Traceability table

The training/inference code matches the paper's described procedure (Eqs. 1-7),
so the method side is sound. The gaps are in the *evaluation* harness that
produces the reported numbers.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Tab. 2 PSNR/SSIM/LPIPS/DISTS/CLIP-IQA, UDM10 | `eval_metrics.py` via `inference.sh` (UDM10 block) | not run | n/a | Verified-present (GT path correct only for UDM10) |
| Tab. 2 PSNR/SSIM/LPIPS/DISTS, SPMCS/YouHQ40/RealVSR/MVSR4x | `inference.sh:22-25,34-37,47-50,60-63` use `--gt datasets/test/UDM10/GT` | wrong/empty | ✗ | BUG (eval-wrong-gt-dataset) |
| Tab. 2 DOVER (all datasets) | `finetune/scripts/eval_dover.py:150` imports `DOVER` package | not in repo | — | MISSING dep (dover-vbench-external-deps); per-sample off-by-one (dover-per-sample-offbyone) |
| Tab. 2 FasterVQA (all datasets) | `finetune/scripts/eval_vbench.py:146-150` imports `VBench` checkout | not in repo | — | MISSING dep (dover-vbench-external-deps) |
| Tab. 2 / Fig. 1 E*warp | `finetune/scripts/eval_ewarp.py:149` `from ewarp import Ewarp` | module absent | — | BUG/MISSING (ewarp-module-missing) |
| Tab. 1a/1b training-strategy & image-ratio ablations | `train_ddp_one_s{1,2}.sh` + trainers | not run | n/a | Verified-present (configs exist) |
| Tab. 1c training-dataset ablation (YouHQ/OpenVid/HQ-VSR) | (none; needs OpenVid-1M ≥1080p ~0.4M, YouHQ) | — | — | Not reproducible (external datasets) |
| Tab. 1d processing-pipeline ablation (+Filter,+Motion) | (none) | — | — | MISSING (pipeline-code-absent) |
| HQ-VSR construction (Sec. 3.3, Fig. 3: 4-step pipeline) | (none; `prepare_dataset.py` only lists files) | — | — | MISSING (pipeline-code-absent) |
| Tab. 3 / Fig. 1 running time, step, DOVER | (no timing script; DOVER via broken eval) | — | — | Not reproducible from code |
| Method: t=399 one-step v-prediction | `inference_script.py:535`, `lora_one_s1_trainer.py:167-209` | matches | ✓ | Verified |
| Eq. 3 latent MSE (Stage-1) | `lora_one_s1_trainer.py:207` | matches | ✓ | Verified |
| Eq. 4/7 MSE+DISTS+frame loss (Stage-2) | `lora_one_s2_trainer.py:238,253,285,290` | matches | ✓ | Verified |
| φ=0.8 image ratio | `train_ddp_one_s2.sh:33`, `lora_one_s2_trainer.py:125` | matches | ✓ | Verified |

## 3. Findings

## missing

```yaml finding
id: pipeline-code-absent
category: missing
topic: "result traceability / dataset construction"
title: "Four-step video processing pipeline (HQ-VSR / Table 1d) absent from repo"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/README.md
line_start: 80
line_end: 81
quote: |
  - [ ] Release the video processing pipeline.
  - [x] Release HQ-VSR dataset.
claim: "The paper's Sec. 3.3 / Fig. 3 video processing pipeline (metadata, scene, quality filtering with CLIP-IQA/FasterVQA/DOVER thresholds, and optical-flow motion-area cropping) that constructs HQ-VSR and produces the Table 1d ablation (+Filter / +Motion) is not implemented anywhere in the repo; `finetune/scripts/prepare_dataset.py` only enumerates file paths, and the README explicitly lists the pipeline as an unreleased TODO."
concern: "Table 1d (ablation on processing pipeline) and the HQ-VSR construction cannot be reproduced from the code; the dataset is shared but the curation procedure that the paper credits for the performance gains is not."
resolution: "Authors: release the metadata/scene/quality-filtering and motion-area-cropping scripts (Eq. 8) so the +Filter and +Motion rows of Table 1d are reproducible."
cross_refs: ["§3.3", "Table 1d"]
check_script: _audit_code/check_eval_imports.py
paper_ref: "Section 3.3 'Video Processing Pipeline'; Table 1d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-vbench-external-deps
category: missing
topic: "dependencies / evaluation metrics"
title: "DOVER and FasterVQA eval scripts import packages absent from repo and requirements"
severity: medium
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 150
line_end: 150
quote: |
    from DOVER.evaluate_a_set_of_videos import evaluate_set as dover
claim: "`eval_dover.py` imports a `DOVER` package and `eval_vbench.py:149` imports `from evaluate import calculate_final` after `chdir` into a `VBench/` directory; neither `DOVER` nor `VBench` exists in the repo (confirmed by `_audit_code/check_eval_imports.py`), and neither is listed in `requirements.txt`."
concern: "DOVER and FasterVQA columns appear for every dataset in Table 2 and in Fig. 1 / Table 3, but the scripts that compute them cannot run without external, unpinned third-party checkouts the README itself flags as TODO."
resolution: "Authors: vendor or pin the exact DOVER and VBench/FasterVQA versions (commit hashes) and add the import/setup instructions, or provide a self-contained metric script."
cross_refs: ["ewarp-module-missing", "Table 2", "Table 3"]
check_script: _audit_code/check_eval_imports.py
paper_ref: "Table 2 (DOVER, FasterVQA columns); README TODO line"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: deps-unpinned-pyiqa-diffusers
category: missing
topic: "dependencies / environment"
title: "Core eval/inference deps (pyiqa, diffusers, safetensors) not in requirements.txt"
severity: low
confidence: high
status: finding
file: code/zhengchen1999__DOVE/requirements.txt
line_start: 1
line_end: 19
quote: |
  accelerate>=1.1.1
  transformers>=4.46.2
  numpy==1.26.0
  torch>=2.5.0
  torchvision>=0.20.0
claim: "`requirements.txt` omits `pyiqa` (used by `eval_metrics.py` and `inference_script.py` for every IQA metric), `diffusers` (the CogVideoX pipeline backbone), and `safetensors`; they are only mentioned as loose, unpinned `pip install` lines in the README (`pip install diffusers[\"torch\"] ...`, `pip install pyiqa`)."
concern: "The environment cannot be rebuilt from `requirements.txt` alone, and the diffusers version is unpinned even though the pipeline depends on specific CogVideoX scheduler/transformer APIs (e.g. `scheduler.get_velocity`)."
resolution: "Authors: add and pin `pyiqa`, `diffusers`, and `safetensors` (with versions) in `requirements.txt`."
cross_refs: []
check_script: _audit_code/check_eval_imports.py
paper_ref: "README 'Dependencies' section"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: eval-wrong-gt-dataset
category: bug
topic: "evaluation harness / Table 2"
title: "inference.sh scores SPMCS/YouHQ40/RealVSR/MVSR4x against UDM10 ground truth"
severity: high
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
claim: "Every `eval_metrics.py` invocation in `inference.sh` hardcodes `--gt datasets/test/UDM10/GT`, including the SPMCS, YouHQ40, RealVSR, and MVSR4x blocks (lines 22-25, 34-37, 47-50, 60-63); only the UDM10 block uses its own GT. `_audit_code/check_inference_gt_paths.py` confirms 5 of 6 calls are dataset-mismatched."
concern: "Full-reference metrics (PSNR/SSIM/LPIPS/DISTS) in Table 2 for these four datasets are computed against the wrong dataset's ground truth; `eval_metrics.py` matches files by name stem (`process` line 174), so non-matching clips are silently skipped (yielding count=0 / no output) and any accidental name collisions produce meaningless scores — either way the provided script does not reproduce Table 2's FR columns for these datasets."
resolution: "Change each block's `--gt` to the matching `datasets/test/<DatasetName>/GT`. The script's own intent (per-dataset evaluation) is contradicted by the copy-paste UDM10 path."
cross_refs: ["Table 2"]
check_script: _audit_code/check_inference_gt_paths.py
paper_ref: "Table 2 (PSNR/SSIM/LPIPS/DISTS rows for SPMCS, YouHQ40, RealVSR, MVSR4x)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ewarp-module-missing
category: bug
topic: "evaluation metrics / temporal consistency"
title: "eval_ewarp.py imports a non-existent `ewarp` module and a wrong RAFT/model path"
severity: medium
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
claim: "`eval_ewarp.py` (the only script for the E*warp temporal-consistency metric) chdirs into `<script_dir>/RAFT` and does `from ewarp import Ewarp`, but there is no `ewarp.py` anywhere in the repo and no `finetune/scripts/RAFT/` directory (RAFT lives at `finetune/utils/RAFT/`); the default `--model finetune/scripts/models/raft-things.pth` (line 190) also does not exist (weights are at `finetune/utils/RAFT/raft-things.pth`). All three confirmed by `_audit_code/check_eval_imports.py`."
concern: "The E*warp column in Table 2 and Fig. 1 cannot be computed: the script raises FileNotFoundError on the chdir / ImportError on `from ewarp import Ewarp` before any metric is produced; this matches the README TODO that E*warp computation is unreleased."
resolution: "Authors: add the missing `ewarp.py` (the RAFT-based warping-error implementation) and fix the RAFT directory and default `--model` paths to point at `finetune/utils/RAFT/`."
cross_refs: ["dover-vbench-external-deps", "Table 2", "Fig. 1"]
check_script: _audit_code/check_eval_imports.py
paper_ref: "Table 2 (E*warp rows); README TODO line"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dover-per-sample-offbyone
category: bug
topic: "evaluation metrics / indexing"
title: "eval_dover.py maps per-sample DOVER scores with an off-by-one (i-1) shift"
severity: low
confidence: high
status: finding
file: code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py
line_start: 158
line_end: 159
quote: |
    for i, name in enumerate(pred_names):
        results[name] = dover_results[i-1]
claim: "When assembling per-sample DOVER scores, the first sorted clip (`i=0`) is assigned `dover_results[-1]` (the last clip's score) and every other clip is shifted by one. `_audit_code/check_dover_offbyone.py` reproduces this: all N per-sample entries are misaligned, while the reported average is unchanged because it sums the same multiset."
concern: "Per-sample DOVER values written to `metrics_dover.json` are wrong (clip ↔ score mismatch); the paper reports only dataset averages, so headline Table 2 DOVER numbers are not affected, but any per-clip analysis or future use of this output would be incorrect."
resolution: "Replace `dover_results[i-1]` with `dover_results[i]` (the index should match `enumerate`)."
cross_refs: ["dover-vbench-external-deps"]
check_script: _audit_code/check_dover_offbyone.py
paper_ref: "Table 2 (DOVER column)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: div2k-image-count
category: difference
topic: "training data / paper-code"
title: "Paper says DIV2K '900 images'; code/README use standard DIV2K_train_HR (800)"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  The image dataset is DIV2K [3], with 900 images
claim: "Sec. 4.1 states the image training set is DIV2K with 900 images, but `train_ddp_one_s2.sh:30` loads `DIV2K_train_HR.txt` and the README datasets table (line 120) lists DIV2K-HR as 800 images — the standard DIV2K training-HR split size."
concern: "The stated image count (900) does not match the standard DIV2K_train_HR split (800) referenced by the code and README; minor, but the exact training-image set affects the Stage-2 pixel-space ablations."
resolution: "Authors: confirm whether 900 vs 800 reflects extra images (e.g. validation HR) or is a typo; specify the exact image list used."
cross_refs: ["§4.1"]
paper_ref: "Section 4.1 'Datasets'; README datasets table"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The implemented training procedure (latent MSE in
Stage-1; MSE + DISTS + frame-difference loss in Stage-2, with image/video mixing
at φ=0.8) and the one-step v-prediction inference (t=399) faithfully match the
paper's Eqs. 1-7. The VSR evaluation protocol (separate synthetic/real test sets,
standard IQA/VQA metrics) is appropriate for the task. The evaluation defects
found are wiring/harness bugs and missing artefacts, not invalid procedures.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | medium       | HQ-VSR pipeline + Table 1d absent; DOVER/VBench/pyiqa/diffusers deps unprovided/unpinned |
| bug         | 3          | high         | inference.sh evaluates 4 datasets against UDM10 GT; E*warp module missing; DOVER per-sample off-by-one |
| difference  | 1          | low          | DIV2K count 900 (paper) vs 800 (code/README) |
| methodology | 0          | -            | Training/inference faithful to Eqs. 1-7; metrics appropriate |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **[bug] eval-wrong-gt-dataset** — `inference.sh` scores SPMCS/YouHQ40/RealVSR/MVSR4x against `UDM10/GT`; the provided commands do not reproduce Table 2's full-reference columns for these 4 datasets. (high/high)
2. **[bug] ewarp-module-missing** — the E*warp script imports a `ewarp` module that does not exist; the temporal-consistency metric (Table 2, Fig. 1) cannot be computed. (medium/high)
3. **[missing] dover-vbench-external-deps** — DOVER and FasterVQA scripts import third-party packages absent from the repo and requirements (README TODO). (medium/high)
4. **[missing] pipeline-code-absent** — the 4-step HQ-VSR construction pipeline (Sec. 3.3) and Table 1d ablation are not in the repo. (medium/high)
5. **[bug] dover-per-sample-offbyone** — `eval_dover.py` shifts per-sample DOVER scores by one (averages unaffected). (low/high)
6. **[missing] deps-unpinned-pyiqa-diffusers** — `pyiqa`/`diffusers`/`safetensors` missing/unpinned in `requirements.txt`. (low/high)

### Items that genuinely look fine
- One-step v-prediction inference with t=399 (`inference_script.py:535`,
  `process_video` lines 459-493) matches Eq. 2 and Sec. 3.1.
- Stage-1 latent MSE loss (`lora_one_s1_trainer.py:207`) matches Eq. 3.
- Stage-2 combined loss = MSE + DISTS + frame-difference L1
  (`lora_one_s2_trainer.py:238,253,285,290`) matches Eqs. 4, 6, 7; the
  frame-difference loss correctly uses adjacent-frame deltas.
- Image ratio φ=0.8 (`train_ddp_one_s2.sh:33`; sampling at
  `lora_one_s2_trainer.py:125`) matches the paper.
- Optimizer betas: paper lists β1/β2/β3 for "AdamW"; `optimizer_utils.py:92-96`
  passes only `(beta1, beta2)` to `torch.optim.AdamW` (β3 is silently ignored,
  used only by Prodigy/CAME), so this is harmless, not a defect.
- `eval_metrics.py` correctly distinguishes full-reference vs no-reference metrics
  and the VideoLQ block uses only `clipiqa` (NR), so its `--gt UDM10` is unused
  there — the GT mismatch genuinely affects only the 4 FR datasets above.
- Tiling/chunking write-coverage assertions (`inference_script.py:724-729`) guard
  against unwritten/double-written regions.

### Open questions for the authors
- Were Table 2's reported FR metrics for SPMCS/YouHQ40/RealVSR/MVSR4x produced by
  a corrected (per-dataset GT) version of `eval_metrics.py` not committed here?
- Which exact DOVER and VBench/FasterVQA versions (commits) produced the Table 2
  VQA columns, and what is the released `ewarp.py` implementation for E*warp?
- Does "DIV2K 900 images" reflect a non-standard image set, or is it 800?
