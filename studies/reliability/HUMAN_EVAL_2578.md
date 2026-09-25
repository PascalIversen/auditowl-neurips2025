# Human-eval worksheet — #2578 · 2578

**16 distinct defects** (the 10 PDF+text audit runs' findings, merged by defect). Detection = how many of the 10 runs surfaced the defect (high = robust; 1 = one run only). Severity & confidence are the auditor's own labels (spread shown where runs disagreed); the wording/quote is taken from the highest-confidence run that cited code.

Tick **one** box per defect (put an `x`):

- **correct & relevant** — true *and* a substantive reproducibility issue worth raising
- **correct but wrong severity** — true and worth raising, but the severity label is miscalibrated (e.g. an out-of-the-box crash with a trivial fix tagged high that's really low/medium)
- **correct but not relevant** — technically true but trivial / nitpick / already acknowledged
- **unsure** — can't decide without resources beyond the frozen repo + paper
- **false** — the claim misreads the code/paper and does not hold

Frozen code: `2578/code_frozen/`  ·  paper: `audits/2578/paper.pdf`

---

### F01 · inference.sh (the canonical 'reproduce paper results' runner) hardcodes `--gt datasets/test/UDM10/GT` for ALL six eval blocks, so SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ (5/6) are scored against UDM10's ground truth; eval_metrics.py matches by filename stem and silently skips mismatches, so the shipped script does not reproduce the Table-2 full-reference (PSNR/SSIM/LPIPS/DISTS) numbers

_category: Technical bug · topic: evaluation driver_

**severity: high  (varied: high, medium)  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** For SPMCS, YouHQ40, RealVSR, MVSR4x and VideoLQ the eval step is hardcoded to `--gt datasets/test/UDM10/GT` instead of the matching dataset's GT (5/6 blocks; only the first UDM10 block is correct). Verified by parsing inference.sh: see _audit_code/out/inference_sh_gt.json (n_mismatch=5).
- **Concern:** As shipped, the full-reference Tab.2 numbers (PSNR/SSIM/LPIPS/DISTS) for five datasets would be computed against UDM10 ground truth; since eval_metrics.py matches predictions to GT by filename (`eval_metrics.py:174`), non-matching names are silently skipped and the reproduction script cannot regenerate Tab.2 for those datasets.
- **Ask:** Authors: fix the `--gt` path in each block to the corresponding `datasets/test/<NAME>/GT`, and confirm the published Tab.2 numbers were computed against the correct per-dataset GT.
- **Evidence:** `inference.sh:22-50` · paper: Table 2 (SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ)
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r06#5)
- **Quoted at `inference.sh:22-50`:**
```
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
... (+11 more lines)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[x]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
severity is low. Does not affect the published findings.
---

### F02 · eval_ewarp.py (the only E*warp temporal-consistency script) imports a nonexistent `ewarp` module and chdir's into a missing `finetune/scripts/RAFT` dir (RAFT actually lives at finetune/utils/RAFT) with a wrong default checkpoint path — the E*warp column (every Tab 2 row, Fig 1) crashes on chdir/import; README §TODO confirms it unreleased

_category: Technical bug · topic: evaluation metrics / E*_warp_

**severity: high  (varied: high, medium)  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** The script `os.chdir`'s to its own dir (`finetune/scripts`) at line 7, then builds `raft_dir = <finetune/scripts>/RAFT` and chdir's into it before `from ewarp import Ewarp`; but RAFT actually lives at `finetune/utils/RAFT/` and contains no `ewarp.py` (verified `_audit_code/check_missing_metric_deps.py`). The default `--model finetune/scripts/models/raft-things.pth` (line 190) also does not exist; the checkpoint is at `finetune/utils/RAFT/raft-things.pth`.
- **Concern:** Even if the missing `ewarp.py` were supplied, the script would crash on the chdir to a nonexistent `finetune/scripts/RAFT` and on the wrong default RAFT checkpoint path, so the E*_warp temporal-consistency numbers (Tab. 2) are not reproducible as shipped.
- **Ask:** Authors: provide `ewarp.py`, point `raft_dir` and the default `--model` at `finetune/utils/RAFT/`, and verify the script runs end-to-end.
- **Evidence:** `finetune/scripts/eval_ewarp.py:146-152` · paper: Table 2 (E*_warp rows)
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r07#6)
- **Quoted at `finetune/scripts/eval_ewarp.py:146-152`:**
```
original_dir = os.getcwd()
raft_dir = os.path.join(original_dir, "RAFT")
os.chdir(raft_dir)
from ewarp import Ewarp as Ewarp
results, avg_score = Ewarp(args)
os.chdir(original_dir)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F03 · eval_dover.py per-sample off-by-one: scores assigned with `dover_results[i-1]`, so each clip name receives the PREVIOUS clip's DOVER score (i=0 gets the last). Dataset-mean is a cyclic permutation so headline Tab 2/3 averages are unaffected, but per-clip DOVER output written to metrics_dover.json is mislabeled

_category: Technical bug · topic: evaluation metrics_

**severity: high  (varied: medium, low)  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** Per-video DOVER scores are assigned with `dover_results[i-1]`, so the first sorted name gets the LAST video's score and every name is mapped to the previous video's score. The reported average is unaffected because indices {-1,0,...,count-2} still cover all videos (verified in _audit_code/out/dover_offbyone.json: mean_unchanged=true, per_sample_matches_correct=false).
- **Concern:** Any per-clip DOVER analysis or per-sample JSON written by this script is mislabeled; only the headline mean DOVER is unaffected, so this does not by itself invalidate Tab.2/Tab.3 DOVER averages.
- **Ask:** Change `dover_results[i-1]` to `dover_results[i]` so per-sample scores align with names.
- **Evidence:** `finetune/scripts/eval_dover.py:156-159` · paper: Table 2 DOVER row
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r06#6)
- **Quoted at `finetune/scripts/eval_dover.py:156-159`:**
```
results = {}

for i, name in enumerate(pred_names):
    results[name] = dover_results[i-1]
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
correct but invalidates no number in the paper as far as I can see it
---

### F04 · eval_dover.py imports an external `DOVER` package that is neither vendored in the repo nor listed in requirements.txt — DOVER (Tab 1, all of Tab 2, the Tab 3 'Performance' column, Fig 1) cannot be computed as shipped; README §TODO line 269 lists it unreleased

_category: Missing code / data · topic: result traceability / metrics_

**severity: medium  (varied: high, medium)  ·  confidence: high  ·  detection: 8/10 runs**

- **Claim:** eval_dover.py relies on `from DOVER.evaluate_a_set_of_videos import evaluate_set`, but no `DOVER` package/dir exists anywhere in the repo and `dover` is not listed in requirements.txt; there is no fetch/setup instruction for it.
- **Concern:** DOVER is a headline metric (Table 2, Table 3 'Performance' column, Fig. 1); without the bundled or pinned DOVER package the script cannot run, so the reported DOVER numbers are not reproducible from the release.
- **Ask:** Authors: vendor the DOVER code (or pin its version + weights and document the install), and provide the exact command used to produce the Table 2/3 DOVER values.
- **Evidence:** `code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py:150-154` · paper: Table 2 DOVER row; Table 3 Performance column
- **Found in runs:** r02, r03, r04, r05, r06, r07, r08, r09  (representative: r03#1)
- **Quoted at `code/zhengchen1999__DOVE/finetune/scripts/eval_dover.py:150-154`:**
```
from DOVER.evaluate_a_set_of_videos import evaluate_set as dover

input_path = os.path.abspath(input_path)

dover_results = dover(input_path, device=device)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
The authors wrapper of DOVER is missing.
---

### F05 · HQ-VSR video-processing pipeline (Sec 3.3 / Fig 3 / Eq 8: metadata + scene + quality filtering with CLIP-IQA/FasterVQA/DOVER thresholds + optical-flow motion-area bounding-box cropping) absent from the repo — the dataset-curation contribution and the Tab 1c/1d (+Filter/+Motion) ablations are not reproducible; README §TODO confirms it is unreleased (only the final dataset blob is linked)

_category: Missing code / data · topic: result traceability / dataset construction_

**severity: high  (varied: high, medium)  ·  confidence: high  ·  detection: 7/10 runs**

- **Claim:** The paper's core contribution and Tab. 1d ablation rely on a four-step pipeline (metadata filtering, scene detection, CLIP-IQA/FasterVQA/DOVER quality filtering, and optical-flow motion-mask bounding-box cropping per Eq. 8) that builds HQ-VSR; the repo contains none of this code, and the README leaves 'Release the video processing pipeline' unchecked.
- **Concern:** Tab. 1c (training-dataset) and Tab. 1d (+Filter/+Motion) ablations and the HQ-VSR dataset claim cannot be reproduced because no script implements the filtering or motion-area cropping (a `grep` for aesthetic/scenedetect/motion-mask/bounding-box returns nothing — see _audit_code/out/eval_artifacts.csv).
- **Ask:** Authors: release the metadata/scene/quality-filtering and motion-area cropping scripts (Sec 3.3, Eq. 8), or confirm Tab. 1c/1d are not reproducible from the released code.
- **Evidence:** `README.md:79-81` · paper: Section 3.3 and Table 1c/1d
- **Found in runs:** r01, r02, r05, r06, r07, r08, r09  (representative: r08#0)
- **Quoted at `README.md:79-81`:**
```
- [x] Release training code.
- [ ] Release the video processing pipeline.
- [x] Release HQ-VSR dataset.
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
still unreleased many month later.
---

### F06 · requirements.txt omits core dependencies (pyiqa, diffusers, safetensors) imported by inference_script.py / eval_metrics.py / trainer.py for every IQA metric and the CogVideoX backbone; they appear only as ad-hoc unpinned `pip install` lines in the README — the env is not rebuildable from requirements alone and the version-sensitive diffusers API (scheduler.get_velocity / get_3d_rotary_pos_embed) risks silent drift

_category: Missing code / data · topic: dependencies / environment_

**severity: low  (varied: medium, low)  ·  confidence: high  ·  detection: 7/10 runs**

- **Claim:** eval_metrics.py and inference_script.py both `import pyiqa` (the library that computes PSNR/SSIM/LPIPS/DISTS/CLIP-IQA/FasterVQA for Table 2), and eval_dover.py / eval_vbench.py import external `DOVER` / `VBench` packages, but none of pyiqa, DOVER, or VBench appears in requirements.txt; `pip install -r requirements.txt` yields an environment that cannot run the evaluation.
- **Concern:** The reproduction environment is under-specified: a fresh install per requirements.txt fails on the metric scripts; the README mentions `pip install pyiqa` only in prose, and DOVER/VBench have no install instructions or pinned versions.
- **Ask:** Authors: add pyiqa (pinned) to requirements.txt and document the exact DOVER / VBench versions and install steps used for Table 2.
- **Evidence:** `requirements.txt:1-20` · paper: Section 4.1 Evaluation Metrics; Table 2
- **Found in runs:** r01, r03, r05, r06, r07, r09, r10  (representative: r01#2)
- **Quoted at `requirements.txt:1-20`:**
```
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
... (+2 more lines)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
 Correct: pyiqa/diffusers imported but not in requirements.txt; DOVER/VBench external + undocumented. This is flagged as TODO but the repo is a year old.
---

### F07 · FasterVQA metric (reported for every Table 2 row and as a Fig 1 axis) has NO computation script anywhere in the repo — a tree-wide search finds 'fastervqa' only in the README §TODO line 269, which confirms the script is unreleased

_category: Missing code / data · topic: result traceability / metrics_

**severity: medium  (varied: high, medium)  ·  confidence: high  ·  detection: 6/10 runs**

- **Claim:** FasterVQA is reported in Tab. 2 and Fig. 1, but no script computes it: a repo-wide grep for 'fastervqa'/'faster_vqa' over all .py files returns no hits (see _audit_code/out/missing_eval_deps.json: fastervqa_script_present=false).
- **Concern:** A reported metric column has no traceable computation in the repo, so those FasterVQA numbers cannot be reproduced from the released code; the authors acknowledge this in the README TODO.
- **Ask:** Add the FasterVQA evaluation script (or document the exact external repo/commit and command used).
- **Evidence:** `code/zhengchen1999__DOVE/README.md:269` · paper: Table 2 (FasterVQA rows); Figure 1
- **Found in runs:** r02, r03, r04, r06, r07, r09  (representative: r02#1)
- **Quoted at `code/zhengchen1999__DOVE/README.md:269`:**
```
> **TODO:** Add metric computation scripts for FasterVQA, DOVER, and $E^*_{warp}$.
```

**Verdict:**   correct & relevant `[]`   correct but wrong severity `[x]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
severity should be medium. It is not a headline metric.
---

### F08 · No timing/benchmark harness for the headline '~28x faster' efficiency claim / Table 3 running times (e.g. DOVE 14.90s vs MGLD-VSR 425.23s on a 33-frame 720x1280 clip) — inference_script.py never measures or logs latency, so the central efficiency contribution cannot be reproduced

_category: Missing code / data · topic: result traceability / efficiency claim_

**severity: low  (varied: medium, low)  ·  confidence: medium  (varied: high, medium)  ·  detection: 4/10 runs**

- **Claim:** Table 3 and Figure 1 report per-method wall-clock running times (e.g. DOVE 14.90 s vs MGLD-VSR 425.23 s) and a '~28× faster' headline, but the repo contains no benchmarking/timing script that measures inference time on a 33-frame 720×1280 video; inference_script.py only runs SR and never times or logs latency.
- **Concern:** The central efficiency claim (28× speed-up) — the paper's main selling point — has no accompanying measurement code, so the timing comparison cannot be independently reproduced or audited for fairness.
- **Ask:** Authors: release the timing script (warm-up, repetitions, what is included/excluded, e.g. VAE tiling, model load) used for Table 3, or document the exact measurement protocol.
- **Evidence:** `paper.pdf` · paper: Table 3; Figure 1 (Time panel); Abstract '28×'
- **Found in runs:** r03, r06, r08, r10  (representative: r03#2)
- **Quoted at `paper.pdf`:**
```
For fairness, all methods are measured running time on the same A100 GPU, generating a
33-frame 720×1280 video. Our method is approximately 28× faster than MGLD-VSR [50].
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
confirmed, but the values do make sense (single forward and no sampling loop should lead to speed up). Magnitude is not reproducible though
---

### F09 · Paper §4.1 states the image training set is DIV2K with 900 images, but the README/official link and the DIV2K_train_HR filelist used by the code are the standard 800-image training split (900 = 800 train + 100 val) — minor description mismatch in the Stage-2 image data

_category: Paper–code mismatch · topic: training data description_

**severity: low  ·  confidence: medium  (varied: high, medium)  ·  detection: 4/10 runs**

- **Claim:** Sec 4.1 states the image training set is DIV2K with 900 images, but the repo README (line 119-120) lists DIV2K-HR as 800 images and links the official `DIV2K_train_HR.zip`, which contains exactly 800 HR training images.
- **Concern:** The image-data count in the paper does not match the dataset the code actually uses (the standard DIV2K train split is 800), a minor description inconsistency in a small part of the training data.
- **Ask:** Clarify whether 900 images were used (and from which DIV2K split), or correct the paper to 800.
- **Evidence:** `paper.pdf` · paper: Sec 4.1 Datasets; README Train Datasets table
- **Found in runs:** r02, r03, r05, r09  (representative: r09#8)
- **Quoted at `paper.pdf`:**
```
"The image dataset is DIV2K [3], with 900 images, which follows the RealESRGAN [38] degradation process."
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
borderline relevancy. Does not change any result. Rather a nitpick.
---

### F10 · Shipped training config (accelerate_config.yaml = 2 GPUs; both train shells set --batch_size 2 / gradient_accumulation_steps 1 -> global batch 4) does not match the paper's '4 NVIDIA A800-80G GPUs, total batch size 8'; the 4-GPU launch topology is not enforced by the scripts, so the default files reproduce batch 4, not 8

_category: Paper–code mismatch · topic: training configuration_

**severity: low  ·  confidence: medium  (varied: high, medium)  ·  detection: 2/10 runs**

- **Claim:** The released accelerate config launches 2 processes (2 GPUs) and both train shell scripts set `--batch_size 2 --gradient_accumulation_steps 1`, giving a global batch size of 4. The paper (§4.1) states training used '4 NVIDIA A800-80G GPUs with the total batch size 8'.
- **Concern:** The shipped configuration does not match the paper's reported training setup (batch size 4 vs 8 / 2 GPUs vs 4); reproducing with the default files would not match the paper's effective batch size, which can affect the trained weights.
- **Ask:** Authors: confirm the GPU count / batch configuration used for the released checkpoints, or update accelerate_config.yaml / the shell scripts to the paper's 4-GPU, batch-8 setting.
- **Evidence:** `finetune/accelerate_config.yaml:3-4` · paper: §4.1 Implementation Details ('4 NVIDIA A800-80G GPUs with the total batch size 8')
- **Found in runs:** r04, r08  (representative: r04#5)
- **Quoted at `finetune/accelerate_config.yaml:3-4`:**
```
gpu_ids: "0, 1"
num_processes: 2  # should be the same as the number of GPUs
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
This is a nitpick and will not change anything. accelerate_config.yaml is a machine-specific launch file which users can adapt.
---

### F11 · UDM10 is used as BOTH the every-500-step training-time validation monitor AND the Table-1 ablation / Table-2 benchmark set; checkpoints are saved at fixed step intervals (not selected by val metric) and there is no held-out split distinct from the benchmark, so manual design/iteration could be informed by UDM10 (soft leakage/design concern)

_category: Methodology / validity · topic: evaluation / model selection_

**severity: low  ·  confidence: low  ·  detection: 2/10 runs**

- **Claim:** Both training shells set the validation set to `../datasets/test/UDM10`, the same UDM10 used as a reported benchmark in Tab. 2 and as the sole evaluation set for all ablations in Tab. 1; validation metrics are logged during training. Checkpoints are saved at fixed `checkpointing_steps` (trainer.py:1002-1014), not chosen by best validation metric, so there is no automated test-loss leakage.
- **Concern:** Because UDM10 is monitored during training and is also the ablation/benchmark set, manual design or checkpoint choices could be informed by UDM10 performance; with only fixed-interval checkpointing this is a soft concern rather than a demonstrated leak.
- **Ask:** Authors: confirm UDM10 was not used to pick checkpoints/hyperparameters, or report the held-out validation set used for selection.
- **Evidence:** `finetune/train_ddp_one_s1.sh:62-73` · paper: Table 1 (ablations on UDM10), Table 2 (UDM10 benchmark)
- **Found in runs:** r06, r07  (representative: r07#8)
- **Quoted at `finetune/train_ddp_one_s1.sh:62-73`:**
```
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
    --eval_metric_list "psnr,ssim,lpips,dists,clipiqa"  # ["psnr", "ssim", "lpips", "dists", "clipiqa", "musiq", "maniqa", 'niqe']
)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[]`   unsure `[ ]`   false `[ ]`

**Notes:**
A bit of an anti pattern. very borderline non relevant. but the agent correctly concerns this as "soft concern" so I would say borderlien relevant.
---

### F12 · inference_script.py removes the SR-output padding as `pad*args.upscale`-should-be but hardcodes `pad*4` regardless of --upscale, so for the --upscale 1 datasets (RealVSR/MVSR4x) with H/W not divisible by 16 the output is over-cropped on the right/bottom edge, shifting the evaluated region and those FR metrics

_category: Technical bug · topic: inference / spatial padding_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** pad_h/pad_w are LR-pixel pads making H/W multiples of 16 (lines 228-229); the LR is then upscaled by args.upscale (line 672), so the SR padding region is pad*args.upscale, but removal hardcodes pad*4. inference.sh runs RealVSR and MVSR4x with --upscale 1 (lines 45, 58), where for non-multiple-of-16 input dims the code over-crops by 3*pad pixels (see _audit_code/out/pad_upscale_mismatch.json).
- **Concern:** For RealVSR/MVSR4x (upscale=1) with input width/height not divisible by 16, the SR output is over-cropped on the right/bottom edge, slightly shifting the evaluated region and the Tab. 2 fidelity metrics for those two real-world datasets.
- **Ask:** Use pad_h*args.upscale and pad_w*args.upscale in remove_padding_and_extra_frames; confirm whether RealVSR/MVSR4x inputs are multiples of 16 (if always, impact is nil).
- **Evidence:** `code/zhengchen1999__DOVE/inference_script.py:731` · paper: Table 2 (RealVSR, MVSR4x rows)
- **Found in runs:** r02  (representative: r02#6)
- **Quoted at `code/zhengchen1999__DOVE/inference_script.py:731`:**
```
video_generate = remove_padding_and_extra_frames(video_generate, pad_f, pad_h*4, pad_w*4)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F13 · inference.sh runs RealVSR/MVSR4x with `--upscale 1`, whereas §4.1 states 'all experiments use a scaling factor x4'; likely the correct operational choice for already-target-resolution real-world inputs, but the code and the blanket x4 statement differ

_category: Paper–code mismatch · topic: evaluation settings_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** For RealVSR and MVSR4x the inference script is invoked with --upscale 1 (no bilinear pre-upscale), whereas Sec 4.1 says 'All experiments are conducted with a scaling factor ×4.'
- **Concern:** If the LQ inputs for these phone-captured real-world datasets are already at target resolution, upscale=1 is the correct operational choice and the ×4 statement is a coarse description; flagged because the code and the blanket ×4 statement differ.
- **Ask:** Confirm that RealVSR/MVSR4x LQ-HQ pairs are stored at the same spatial resolution (so the effective SR factor is realized in the data, not the bilinear pre-upscale), and reconcile with the ×4 statement.
- **Evidence:** `code/zhengchen1999__DOVE/inference.sh:40-46` · paper: Sec 4.1 ('scaling factor ×4')
- **Found in runs:** r09  (representative: r09#9)
- **Quoted at `code/zhengchen1999__DOVE/inference.sh:40-46`:**
```
python inference_script.py \
    --input_dir datasets/test/RealVSR/LQ-Video \
    --model_path pretrained_models/DOVE \
    --output_path results/DOVE/RealVSR \
    --is_vae_st \
    --upscale 1 \
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:** Not a real mismatch. `--upscale` is just a bilinear pre-upscale to match target resolution. RealVSR LQ are already at GT resolution so `upscale=1` is required. §4.1's "×4" is the SR task factor, not this flag. SR is still ×4. The claim holds, it's just trivial

---

### F14 · Stage-2 trainer (lora_one_s2_trainer.py / args.py) exposes loss variants the paper never mentions — edge-aware DISTS (Sobel EdgeDetectionModel), LPIPS / edge-aware LPIPS, GAN losses (gen_cls_loss_weight, diffusion_gan_max_timestep), optical-flow flags; unused by the released defaults but undocumented

_category: Paper–code mismatch · topic: training loss / paper omission_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** The Stage-2 trainer exposes loss variants the paper never mentions: an edge-aware DISTS (`ea_dists_weight`, using a Sobel `EdgeDetectionModel`), LPIPS and edge-aware LPIPS, and GAN-related args (`gen_cls_loss_weight`, `diffusion_gan_max_timestep`) plus optical-flow flags in args.py. The shipped `train_ddp_one_s2.sh` sets only `--dists_weight 1.0` and `--frame_diff_weight 1.0`, which matches the paper's Eq. (4)/(7).
- **Concern:** These are unused-by-default alternative loss paths; they do not change the released training recipe, but their presence is undocumented in the paper and could matter if reviewers assume the released defaults are the only configuration used.
- **Ask:** Authors: confirm the paper results use only plain DISTS + frame-diff (the shell defaults) and that edge-aware/LPIPS/GAN paths were exploratory.
- **Evidence:** `finetune/models/dove/lora_one_s2_trainer.py:245-264` · paper: Section 3.2, Eq. (4), Eq. (7)
- **Found in runs:** r07  (representative: r07#7)
- **Quoted at `finetune/models/dove/lora_one_s2_trainer.py:245-264`:**
```
if self.args.ea_dists_weight > 0:
    dists_loss = self.dists_loss(pred_frame, gt_frame)
    edge_loss = self.dists_loss(
        self.edge_detection_model(pred_frame),
        self.edge_detection_model(gt_frame)
    )
    perceptual_loss = perceptual_loss + dists_loss + edge_loss
elif self.args.dists_weight > 0:
    dists_loss = self.dists_loss(pred_frame, gt_frame)
    perceptual_loss = perceptual_loss + dists_loss
elif self.args.ea_lpips_weight > 0:
    lpips_loss = self.lpips_loss(pred_frame, gt_frame)
    edge_loss = self.lpips_loss(
        self.edge_detection_model(pred_frame),
        self.edge_detection_model(gt_frame)
    )
    perceptual_loss = perceptual_loss + lpips_loss + edge_loss
elif self.args.lpips_weight > 0:
... (+2 more lines)
```

**Verdict:**   correct & relevant `[]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:** correct but the "GAN losses" overstate it: there is no discriminator/adversarial code at all. Seems to be some old research code.

---

### F15 · eval_metrics.py computes PSNR/SSIM on full RGB with no border crop by default; neither inference.sh nor the README example passes --test_y_channel/--crop, whereas many VSR papers report Y-channel PSNR/SSIM — the channel/crop convention is unstated, so reproduced fidelity numbers may differ

_category: Paper–code mismatch · topic: metric definition_

**severity: low  ·  confidence: low  ·  detection: 1/10 runs**

- **Claim:** PSNR/SSIM are computed on full RGB unless `--test_y_channel` and `--crop` are passed; neither inference.sh nor the README example passes them, so by default PSNR/SSIM use RGB with no border crop. Many VSR papers report Y-channel PSNR/SSIM.
- **Concern:** If the paper's Table 2 PSNR/SSIM were computed on Y channel (or with a border crop) but the shipped command uses RGB/no-crop, the reproduced fidelity numbers would differ; the paper does not state the channel/crop convention, so this is ambiguous.
- **Ask:** Authors: state whether Table 2 PSNR/SSIM are RGB or Y-channel and whether any border crop was applied, and ship the matching eval command.
- **Evidence:** `eval_metrics.py:245-246` · paper: §4.1 Evaluation Metrics (PSNR, SSIM)
- **Found in runs:** r10  (representative: r10#5)
- **Quoted at `eval_metrics.py:245-246`:**
```
parser.add_argument('--crop', type=int, default=0, help='Crop border size for PSNR/SSIM')
parser.add_argument('--test_y_channel', action='store_true', help='Use Y channel for PSNR/SSIM')
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
This is used for Table 2. Its applied uniformly. So correct, but it is rather irrelevant.
---

### F16 · eval_vbench.py chdir's into a nonexistent `finetune/scripts/VBench` dir and imports `from evaluate import calculate_final`; the VBench package is neither vendored nor in requirements (VBench is not a Table-2 metric, so headline impact is low)

_category: Technical bug · topic: metrics_

**severity: low  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** eval_vbench.py changes into 'VBench' under finetune/scripts/ (absent in repo, see check) and imports `from evaluate import calculate_final`; the VBench package is neither vendored nor in requirements.txt.
- **Concern:** The script crashes immediately; however VBench is not a metric reported in the main paper's Table 2, so impact on the headline numbers is low.
- **Ask:** Remove this helper or vendor/pin VBench and fix the path if any reported number depends on it.
- **Evidence:** `code/zhengchen1999__DOVE/finetune/scripts/eval_vbench.py:145-150` · paper: N/A (VBench not in main Table 2)
- **Found in runs:** r09  (representative: r09#5)
- **Quoted at `code/zhengchen1999__DOVE/finetune/scripts/eval_vbench.py:145-150`:**
```
original_dir = os.getcwd()
vbench_dir = os.path.join(original_dir, "VBench")
os.chdir(vbench_dir)
print(f"Changed directory to: {vbench_dir}")
from evaluate import calculate_final as Vbench
results, avg_score, dim_results, dim_avg = Vbench(input_path)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
real bug in irrelevant part of the code.  VBench is reported nowhere in the paper.
---

