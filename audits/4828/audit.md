# Code audit — WarpGAN (NeurIPS 2025, paper 4828)

## 1. Summary

WarpGAN is a 3D GAN inversion method for single-image novel-view face synthesis.
It trains (a) a Swin-Transformer encoder that inverts an image into the `W+`
latent space of a pretrained EG3D generator (`scripts/train_vanilla.py`,
`training/coach_vanilla.py`), and (b) a style-based novel-view inpainting
network "SVINet" built on a LaMa/FFC generator that fills occluded regions of a
depth-warped image (`scripts/train_inpainting.py`,
`training/coach_inpainting_static.py`, `models/saicinpainting/...`). Inference is
`scripts/infer.py`; editing via PTI is `scripts/run_pti.py` /
`scripts/editing_ptiG.py`. The paper's empirical claims are entirely in **Table 1**
(FID / ID-similarity on CelebA-HQ; LPIPS / FID / ID on MEAD at ±30°/±60°, plus
inference Time) and **Table 2** (a 7-row ablation of FID / ID on CelebA-HQ).
There are no statistical tests; the authors explicitly report no error bars
(checklist Q7, "image generation rather than prediction").

What I did: read the paper text and PDF; mapped every reported number to repo
code; greased the repo for any FID / ID-similarity / LPIPS-as-metric computation
(`_audit_code/check_metric_code.py`, scanning 205 `.py` files outside the
vendored `editings/` tree and the unrelated arcface IJB-C eval helper); inspected
`scripts/infer.py` to confirm it only writes PNGs and computes no metric; checked
dependency specification (`scripts/install_deps.sh`), checkpoint/data
availability, and the ablation toggles in `configs/train_inpainting.yaml`.

Headline result: the repository contains training and inference code and a full
dependency spec, but **no code that computes any of the quantitative metrics
reported in the paper** (FID, ID similarity, LPIPS). Every number in Table 1 and
Table 2 is therefore untraceable to a producing script. The evaluation
datasets used to compute those numbers (preprocessed CelebA-HQ and MEAD test
splits) are also absent. These are `missing` findings; nothing in the runnable
code is methodologically invalid for what it does, and no leakage/independence
concern applies to a generative novel-view task evaluated against external GT.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — CelebA-HQ FID (e.g. Ours 19.12) | (none — no FID code) | — | — | MISSING |
| Table 1 — CelebA-HQ ID (e.g. Ours 0.7882) | (none — no ID-sim metric code) | — | — | MISSING |
| Table 1 — MEAD LPIPS ±30°/±60° (Ours 0.2490 / 0.3008) | (none — LPIPS used only as training loss, no eval) | — | — | MISSING |
| Table 1 — MEAD FID ±30°/±60° (Ours 38.15 / 64.01) | (none) | — | — | MISSING |
| Table 1 — MEAD ID ±30°/±60° (Ours 0.8315 / 0.7741) | (none) | — | — | MISSING |
| Table 1 — Time (s) column | (none — `infer.py` does not time/report) | — | — | MISSING |
| Table 2 — 7-row ablation FID / ID (rows A–G) | configs toggle variants exist (`configs/train_inpainting.yaml:84-85`, `15-16`, `losses.latent.weight:65-66`) but no FID/ID code to score them | — | — | MISSING (metric code); ablation variants ARE configurable |
| Fig. 3 / Fig. 4 qualitative panels | `scripts/infer.py` synthesizes & saves images | images only | n/a (qualitative) | Partial (needs external ckpts/data) |
| Fig. 5 ablation/editing visuals | `scripts/infer.py`, `scripts/editing_ptiG.py` | images only | n/a (qualitative) | Partial |

Every quantitative cell routes to a MISSING row → the dominant finding below.

## 3. Findings

## missing

```yaml finding
id: no-metric-evaluation-code
category: missing
topic: "result traceability / evaluation"
title: "No code computes FID, ID-similarity, or LPIPS — all of Table 1 & 2 untraceable"
severity: high
confidence: high
status: finding
file: scripts/infer.py
line_start: 210
line_end: 213
quote: |
                        save_inp_dir = os.path.join(self.opts.exp_dir)
                        os.makedirs(save_inp_dir, exist_ok=True)
                        save_inp_path = os.path.join(save_inp_dir, f'{frame[i]}.png')
                        cv2.imwrite(save_inp_path, cv2.cvtColor(self.tensor2im(res[i]), cv2.COLOR_RGB2BGR))
claim: "The inference entrypoint only synthesizes and writes PNG images; it computes no metric. A repo-wide scan (_audit_code/check_metric_code.py over 205 .py files, excluding vendored editings/ and the arcface IJB-C helper) finds zero FID/Frechet/InceptionV3, zero ID-similarity, and zero LPIPS-as-metric code. LPIPS appears only as a training loss (criteria/lpips/, training/coach_*.py)."
concern: "Every reported number in Table 1 (CelebA-HQ FID/ID; MEAD LPIPS/FID/ID at ±30°/±60°; inference Time) and Table 2 (7-row FID/ID ablation) lacks any producing script, so none of the paper's quantitative claims can be reproduced or verified from the released code."
resolution: "Authors: please add the evaluation harness that loads synthesized vs ground-truth/reference images and computes FID, ArcFace ID-cosine similarity, and LPIPS for Tables 1 and 2 (and the timing measurement), specifying the exact FID/ID/LPIPS implementations and reference-set construction used."
cross_refs: ["eval-datasets-absent"]
check_script: _audit_code/check_metric_code.py
paper_ref: "Table 1; Table 2; §4.1 Evaluation metrics"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eval-datasets-absent
category: missing
topic: "data availability / evaluation"
title: "Preprocessed CelebA-HQ and MEAD test sets used for Table 1/2 are not in the repo"
severity: medium
confidence: high
status: finding
file: configs/paths_config.py
line_start: 1
line_end: 9
quote: |
  dataset_paths = {
  	'test': './data/celeba-hq_1000_align',
  	'train': './data/FFHQ-LPFF-EG3D_all',
  }
  
  dataset_static_paths = {
  	'test': './data/celeba-hq_1000_static_rebalanced',
  	'train': './data/FFHQ-EG3D_all_static_rebalanced',
  	'synth': './data/SynthData100000_rebalanced',
  }
claim: "Config points the test/eval pipeline at local directories ./data/celeba-hq_1000_align and ./data/celeba-hq_1000_static_rebalanced (and the paper also evaluates on MEAD), but the repo ships only 8 demo PNGs under data/test_img/ and these CelebA-HQ/MEAD directories are absent (data/ is gitignored)."
concern: "The exact 1000-image CelebA-HQ subset and the MEAD multi-view test split (with their preprocessing/pose extraction) that produce the Table 1/2 numbers are not provided, so even if metric code were added the precise reported values could not be reproduced."
resolution: "Authors: please release (or give an exact reconstruction recipe for) the preprocessed CelebA-HQ 1000-image subset and the MEAD test split, including the alignment/pose-extraction outputs and the list of identities/views used."
cross_refs: ["no-metric-evaluation-code"]
paper_ref: "§4.1 Datasets / Evaluation metrics"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pretrained-weights-external-only
category: missing
topic: "pretrained models / reproducibility"
title: "All pretrained weights live on an external Google Drive folder, not in the repo"
severity: low
confidence: high
status: question
file: README.md
line_start: 16
line_end: 18
quote: |
  ## Checkpoints
  We have uploaded the pre-trained models required for image preprocessing, training, inference, and editing to [Google Drive](https://drive.google.com/drive/folders/1G9PeyrCS1gTyF3C957l6Key__t44aiIE?usp=sharing). After downloading, please place them in the corresponding directories (`./pose_estimation`, `./pretrained_models`, `./editings`).
claim: "Inference and training require ./pretrained_models/inversion/gan_encoder.pt and ./pretrained_models/inpaintor/inpaintor.pt (configs/infer.yaml, configs/train_inpainting.yaml:30,60); the pretrained_models/ directory does not exist in the clone and all weights are gated behind a single Google Drive link."
concern: "Reproduction depends on an external, mutable Drive folder I did not retrieve; if it is incomplete or removed, neither inference nor the LaMa-perceptual-loss training path is runnable. (Routing as a question: external hosting is an acceptable mechanism, only the resolvability is unverified here.)"
resolution: "Authors: confirm the Drive folder contains every checkpoint referenced by the configs (gan_encoder.pt, inpaintor.pt, EG3D generator, LaMa_perceptual_loss_models, pose-estimation/Deep3DFaceRecon weights) and consider a checksummed download script."
cross_refs: []
paper_ref: "README Checkpoints section"
tags: [reforms:2, heil:bronze]
```

## bug

No findings. The scripts I read (`infer.py`, `train_inpainting.py`,
`train_vanilla.py`, configs) are internally consistent; seeds are set in
`infer.py:35-42`. I did not run the pipeline (requires GPU + external weights),
so runtime crashes cannot be fully excluded, but no static defect was found.

## difference

```yaml finding
id: reconstruction-perceptual-loss-impl
category: difference
topic: "loss function (paper vs code)"
title: "Paper's reconstruction perceptual loss L_P is implemented as LaMa ResNet-PL, not LPIPS"
severity: low
confidence: medium
status: question
file: configs/train_inpainting.yaml
line_start: 48
line_end: 62
quote: |
  perceptual:
    weight: 0
  adversarial:
    kind: r1
    weight: 10
    gp_coef: 0.001
    mask_as_fake_target: ${losses.with_mask}
    allow_scale_mask: True
  feature_matching:
    weight: 100
  resnet_pl:
    weight: 30
    weights_path: ./pretrained_models/LaMa_perceptual_loss_models
  lpips:
    weight: 0
claim: "The released SVINet config drives the reconstruction 'perceptual' term via LaMa's ResNet-PL (weight 30) and adds a feature-matching loss (weight 100) and r1 adversarial loss; the generic 'perceptual' and 'lpips' losses are zeroed."
concern: "The paper's Eq. (6) names a perceptual loss L_P with weight λ_P=30 but does not disclose it is LaMa ResNet-PL, nor mention the weight-100 feature-matching term, so the realized objective is broader than the paper states (paper omission)."
resolution: "Authors: confirm whether L_P in Eq. (6) is the LaMa ResNet perceptual loss and whether the feature-matching loss (weight 100) was part of the reported runs; if so, state it in the loss description."
cross_refs: []
paper_ref: "§3.3.3 Eq. (6), Eq. (10); §4.1 (λ_P=30)"
tags: [forensics:post-hoc-selection]
```

## methodology

No findings. This is a generative novel-view-synthesis task; the standard
leakage / sample-independence / train-test-split concerns do not structurally
apply:
- **Data splitting / leakage**: N/A — training is on FFHQ + EG3D-sampled
  synthetic pairs; evaluation is on disjoint external datasets (CelebA-HQ, MEAD)
  against their own ground-truth views. No classifier split to leak across.
- **Baselines**: the paper compares against 8 prior 3D-inversion methods
  (optimization- and encoder-based) under the stated EG3D setup; this is the
  appropriate baseline set for the task (a "naive predictor" baseline is not
  meaningful for image FID/ID). Not flaggable from the code.
- **Hyperparameter tuning on test set**: no evaluation/selection code is present
  at all, so no test-set-touching selection could be observed (the absence is
  the `no-metric-evaluation-code` finding, not a methodology defect).
- **Statistical integrity**: N/A — no statistical tests or error bars are
  reported, and the authors disclose this (checklist Q7).
- **Pretraining contamination**: EG3D is pretrained on FFHQ and the inversion
  encoder also trains on FFHQ; test sets (CelebA-HQ, MEAD) are separate face
  corpora. Standard for this subfield; no concrete overlap evidence in-repo to
  flag.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 3          | high         | No FID/ID/LPIPS metric code; eval datasets absent; weights external |
| bug         | 0          | -            | No static defect found in scripts read (not executed)       |
| difference  | 1          | low          | Reconstruction perceptual loss realized as LaMa ResNet-PL + FM, undisclosed |
| methodology | 0          | -            | Generative task; leakage/split/stat checks structurally N/A |

## 5. Closing lists

**Top take-aways** (ranked):
1. [missing, high] `no-metric-evaluation-code` — the repo contains no code that
   computes FID, ID-similarity, or LPIPS; every number in Table 1 and Table 2
   (and the inference-Time column) is untraceable to a producing script.
2. [missing, medium] `eval-datasets-absent` — the preprocessed CelebA-HQ 1000-image
   subset and MEAD multi-view test split that the metrics are computed on are not
   shipped (only 8 demo images), so the exact numbers cannot be reproduced.
3. [difference, low] `reconstruction-perceptual-loss-impl` — the paper's Eq. (6)
   perceptual term is realized as LaMa ResNet-PL plus an undisclosed weight-100
   feature-matching loss.
4. [missing, low/question] `pretrained-weights-external-only` — all checkpoints
   are gated behind a single Google Drive folder (resolvability unverified here).

**Items that genuinely look fine**:
- Dependency specification is complete and pinned (`scripts/install_deps.sh`,
  exact versions incl. CUDA wheels).
- Training and inference entrypoints for both the inversion encoder and SVINet
  are present and wired to configs.
- Ablation variants in Table 2 are reproducible through config toggles
  (`input_mirror`, `use_style`, `losses.latent.weight` for L_c, `synth.able`),
  i.e. the ablation *experiments* are not missing — only the metric scoring is.
- Inference seeds are set across numpy / random / torch / cuda
  (`scripts/infer.py:35-42`).

**Open questions for the authors**:
- Where is the evaluation script that produces Table 1/2 (FID/ID/LPIPS) and the
  timing in Table 1, and which FID/ID/LPIPS implementations were used?
- Will the preprocessed CelebA-HQ and MEAD test splits be released, or an exact
  reconstruction recipe provided?
- Is the Eq. (6) L_P the LaMa ResNet perceptual loss, and was the weight-100
  feature-matching loss part of the reported runs?
