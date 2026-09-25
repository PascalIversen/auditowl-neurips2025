# Code-repository audit — "Transformer brain encoders explain human high-level visual responses" (NeurIPS 2025, #1717)

## 1. Summary

The repository (`Hosseinadeli/transformer_brain_encoder`, audited at commit `d4de870`) implements
the Transformer Brain Encoder (TBEn) and two comparison encoders — a spatial-feature-factorized
encoder and an L2-penalised "linear" encoder — together with a training driver (`main.py`,
`engine.py`), the NSD/Algonauts data pipeline (`datasets/nsd.py`, `datasets/nsd_utils.py`),
three feature backbones (DINOv2, ResNet50, CLIP in `models/`), and a results notebook
(`visualize_results.ipynb`) that computes the headline encoding-accuracy metric (squared Pearson
correlation between predicted and ground-truth test fMRI, divided by the noise ceiling). The core
architecture and the metric computation are present and look methodologically reasonable; in
particular, model selection (`val_perf`, the epoch at which test predictions are dumped) is done
on a held-out validation split, not on the test set, so there is no obvious early-stopping leak.

The repo is, however, **missing the code for a large fraction of the paper's reported comparisons
and analyses**: there is no BERT backbone (Table 6 text-modality results), no DeepGaze saliency
baseline (Tables 1/3/5), no YOLO category-selectivity analysis (Tables 7–10), no BrainDiVE image
generation (Figs S9/S10), and no PCA-feature extraction for the PCA+regression baseline. It also
ships **no dependency specification, no trained weights, and no copy of the test-set fMRI / noise
ceiling / per-run predictions** that the reported numbers are computed against, and all data paths
are hardcoded to an internal `/engram/...` cluster. Finally, the paper's "10-fold cross
validation" is implemented as 10 *unseeded random 90/10 resamples* whose predictions are
ensembled — a different (and milder) procedure than k-fold CV.

I read `paper.pdf`/`paper_text.txt`, `README.md`, `main.py`, `engine.py`, `datasets/nsd.py`,
`datasets/nsd_utils.py`, `models/brain_encoder.py`, `models/backbone.py`, and the three notebooks
(`visualize_results.ipynb`, `run_model.ipynb`, `test_wrapper.ipynb`). I ran two read-only checks
under `_audit_code/`: `check_artifact_presence.py` (keyword/regex scan over all `.py` and decoded
`.ipynb` source; outputs `out/artifact_presence.txt`) and `check_cv_and_seed.py` (inspects the
split/seed mechanism; outputs `out/cv_seed.txt`).

## 2. Result-traceability table

"Repo location" lists the code that *computes* the value (not merely plots it). Even where code is
present, the reported numbers additionally require a local `test_split/test_fmri/*.npy`,
`noise_ceiling/*.npy`, and saved per-run predictions, none of which ship with the repo
(see `missing-test-fmri-and-weights`), so no value could be independently re-run.

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Table 1 Transformer (rois) DINOv2 (0.60/0.56/0.56/0.42) | `models/brain_encoder.py` + `visualize_results.ipynb` (`evaluate_model`) | code present, not runnable (no data/weights) | not re-run | Present-but-unverifiable |
| Table 1 Spatial-feature factorized | `models/brain_encoder.py:204-237` + notebook | code present | not re-run | Present-but-unverifiable |
| Table 1 Ridge regression (~900M) | `main.py:208-210` (fixed 0.02 L2) / notebook `learn_reg` (sklearn `LinearRegression`, no reg) | partial / mismatched | paper says grid-searched ridge penalty | MISMATCH (see `ridge-baseline-no-gridsearch`) |
| Table 1 PCA + regression | notebook loads `pca_run*/{train,val,test}.npy`; no extraction code | NO (load only) | — | MISSING (`missing-pca-feature-extraction`) |
| Table 1 CLS + regression | backbone supports `*_cls`; no table-producing driver | partial | — | weakly traceable |
| Tables 1/3/5 Saliency based integration (DeepGaze) | (none) | NO | — | MISSING (`missing-saliency-baseline`) |
| Table 2 training-set-size sweep (550…8800) | no size-sweep driver found | NO | — | MISSING (`missing-trainsize-sweep`) |
| Table 3 ResNet50 backbone results | `models/resnet.py` + `models/backbone.py` + notebook | code present | not re-run | Present-but-unverifiable |
| Table 4 vertices / layer-ensemble | `brain_encoder.py` voxels branch + notebook `evaluate_combined_model` | code present | not re-run | Present-but-unverifiable |
| Table 5 CLIP backbone results | `models/clip.py` + notebook | code present | not re-run | Present-but-unverifiable |
| Table 6 BERT text-modality results | (no BERT backbone in `models/backbone.py`) | NO | — | MISSING (`missing-bert-text-modality`) |
| Tables 7–10 category selectivity (YOLOv5/YOLOv8-face) | (none) | NO | — | MISSING (`missing-yolo-selectivity`) |
| Figs 2/3 surface & per-ROI accuracy | notebook (pycortex) | plotting of above values | not re-run | Present-but-unverifiable |
| Fig 4 layer-ensemble surface | notebook `evaluate_combined_model` (softmax on **val** corr) | code present | not re-run | Present-but-unverifiable |
| Fig 5 attention maps | `brain_encoder_wrapper.py` `attention()` + `test_wrapper.ipynb` | code present | not re-run | Present-but-unverifiable |
| Figs S5–S8 ROI-query cosine similarity | (none found) | NO | — | MISSING (`missing-supplementary-analyses`) |
| Figs S9–S10 BrainDiVE generated images | (none) | NO | — | MISSING (`missing-braindive`) |

## 3. Findings

### missing

```yaml finding
id: missing-bert-text-modality
category: missing
topic: "result traceability / text modality"
title: "No BERT backbone in repo; Table 6 text-modality results untraceable"
severity: high
confidence: high
status: finding
file: models/backbone.py
line_start: 58
line_end: 68
quote: |
    if 'resnet' in args.backbone_arch: 
        backbone = resnet_model(args.backbone_arch, train_backbone, return_interm_layers, args.dilation)
        num_channels = backbone.num_channels
    elif 'dinov2_q' in args.backbone_arch:
        backbone = dino_model_with_hooks(-1*args.enc_output_layer, return_interm_layers, return_cls)
        num_channels = backbone.num_channels
    elif 'dinov2' in args.backbone_arch:
        backbone = dino_model(-1*args.enc_output_layer, return_interm_layers, return_cls)
        num_channels = backbone.num_channels
    elif 'clip' in args.backbone_arch:
        backbone = clip_model(-1*args.enc_output_layer, return_interm_layers, return_cls)
claim: "The backbone factory branches only on resnet / dinov2_q / dinov2 / clip vision backbones; no BERT (or any text) backbone is implemented anywhere in the repo."
concern: "Table 6 reports a full BERT-backbone text-modality experiment (Transformer 0.27/0.27/0.33/0.27 vs Ridge), but no code in the repo can produce those numbers, so a headline cross-modality claim is unreproducible."
resolution: "Authors: please add the BERT-backbone branch and the caption/BERT feature extraction used for Table 6, or point to where it lives."
cross_refs: []
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Section 4.4 / Table 6"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-saliency-baseline
category: missing
topic: "baselines"
title: "DeepGaze saliency-integration baseline absent from repo (Tables 1/3/5)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  method uses saliency map of the image, instead of a learned spatial map, to integrate the tokens
claim: "The paper describes a DeepGaze-based 'Saliency based integration' baseline reported in Tables 1, 3 and 5, but no DeepGaze/saliency code exists anywhere in the repo (verified by regex scan over all .py and notebook source)."
concern: "A reported comparison baseline cannot be reproduced or checked, weakening the claim that TBEn beats generic feature reweighting."
resolution: "Authors: please add the DeepGaze saliency-map generation and integration code, or the saved saliency maps/weights used."
cross_refs: []
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Section 4 (Saliency based integration); Tables 1, 3, 5"
tags: [reforms:5, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-yolo-selectivity
category: missing
topic: "result traceability / supplementary analysis"
title: "YOLO-based category-selectivity analysis absent (Tables 7–10)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  To quantify the category selectivity of attention maps, we classified each pixel of the test set images using
claim: "Supplementary Tables 7–10 report quantitative category-selectivity of attention maps computed via YOLOv5 + YOLOv8-face, but no YOLO pixel-classification or selectivity-tabulation code is present in the repo."
concern: "The quantitative interpretability claim (attention selectivity matches ROI labels) cannot be reproduced from the code provided."
resolution: "Authors: please add the YOLO pixel-classification and the top-2k-pixel category-tabulation script that produced Tables 7–10."
cross_refs: []
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Supplementary A.2; Tables 7–10"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-braindive
category: missing
topic: "result traceability / supplementary analysis"
title: "BrainDiVE image-generation pipeline absent (Figs S9/S10)"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  Using our encoding model within the BrainDiVE framework, we generated 200 images predicted to
claim: "Supplementary Figs S9/S10 present diffusion-generated maximally-activating images per ROI cluster via BrainDiVE, but no diffusion / BrainDiVE-guidance code is present in the repo."
concern: "A qualitative supplementary analysis is not reproducible from the provided code."
resolution: "Authors: please add (or link) the BrainDiVE guidance code used with the encoding model."
cross_refs: []
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Supplementary A.4; Figs S9, S10"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-pca-feature-extraction
category: missing
topic: "baselines"
title: "PCA+regression baseline loads precomputed features; no PCA-fit code in repo"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  model is to first reduce the dimensionality of the features using Principle Component Analyses (PCA)
claim: "The PCA+regression (learn_reg) path in visualize_results.ipynb only loads precomputed feature .npy files from pca_run* directories; the PCA dimensionality-reduction fit that produced them is not in the repo (PCA is imported but never .fit on training features, per check_artifact_presence.py)."
concern: "The PCA+regression baseline in Table 1 cannot be reproduced because the feature-reduction step is not shipped; whether PCA was fit on train-only vs all data is also unverifiable."
resolution: "Authors: please add the PCA-fitting feature-extraction script, and confirm PCA components were fit on training images only."
cross_refs: []
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Section 4 (PCA + regression); Table 1"
tags: [reforms:5, leakage:L1.1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ridge-baseline-no-gridsearch
category: missing
topic: "baselines / hyperparameter tuning"
title: "Paper's grid-searched ridge penalty not implemented in repo"
severity: medium
confidence: high
status: finding
file: main.py
line_start: 207
line_end: 210
quote: |
        # add a ridge penalty to the linear model
        if 'cls' not in self.backbone_arch:
            if self.encoder_arch == 'linear':
                loss = loss + 0.02* outputs['l2_reg']
claim: "The paper states 'We used a grid search to select the best ridge penalty to maximize performance on the validation data' for the Ridge baseline, but the repo's linear encoder uses a single hardcoded 0.02 L2 weight, and the notebook's alternative learn_reg path uses unregularised sklearn LinearRegression — no grid search over the penalty exists anywhere (verified by check_artifact_presence.py)."
concern: "The strongest baseline (Ridge ~900M params) may be under-tuned relative to its paper description, so the reported gap between TBEn and Ridge could partly reflect an untuned baseline rather than the proposed method's advantage."
resolution: "Authors: please add the ridge-penalty grid-search code, or report the chosen penalty per subject/backbone and confirm it matches the reported Ridge numbers."
cross_refs: []
check_script: _audit_code/check_artifact_presence.py
paper_ref: "Section 4 (Ridge regression); Tables 1, 3, 5"
tags: [reforms:5, whalen:pitfall-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-dependencies
category: missing
topic: "expected code completeness"
title: "No dependency specification (requirements.txt / environment.yml) in repo"
severity: medium
confidence: high
status: finding
file: main.py
line_start: 1
line_end: 14
quote: |
  import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

  import numpy as np
  from tqdm import tqdm

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.utils.model_zoo
  import torchvision
  import torch.multiprocessing as mp
  from torch.utils.data.distributed import DistributedSampler
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.distributed import init_process_group, destroy_process_group
claim: "The repo ships no requirements.txt, environment.yml, setup.py, or pyproject.toml (none present in the file tree); the code imports torch, torchvision, scipy, sklearn, nilearn, pycortex (cortex), wandb, transformers, and a DINOv2/CLIP/timm stack with no pinned versions."
concern: "The environment cannot be deterministically rebuilt, and several imports (e.g. cortex, transformers, nilearn) are non-trivial to provision, so reproduction is impeded."
resolution: "Authors: please add a pinned requirements.txt / environment.yml covering all imported packages."
cross_refs: []
paper_ref: "NeurIPS checklist Q5 (open access to code)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-test-fmri-and-weights
category: missing
topic: "data / weights availability"
title: "Test fMRI, noise ceiling, saved predictions and weights absent; data_dir hardcoded"
severity: high
confidence: high
status: finding
file: main.py
line_start: 52
line_end: 56
quote: |
    parser.add_argument('--data_dir', default='../../../algonauts/algonauts_2023_challenge_data/', type=str)
    parser.add_argument('--parent_submission_dir', default='./algonauts_2023_challenge_submission/', type=str)
    
    parser.add_argument('--saved_feats', default=None, type=str) #'dinov2q'
    parser.add_argument('--saved_feats_dir', default='../../algonauts_image_features/', type=str) 
claim: "The headline encoding accuracies are computed in visualize_results.ipynb by loading ground-truth test fMRI (test_split/test_fmri/*.npy) and noise_ceiling/*.npy plus per-run saved predictions; none of these artefacts ship with the repo, there is no fetch script, and data_dir defaults to a relative '../../../algonauts/...' path (the notebooks hardcode '/engram/nklab/...')."
concern: "The Algonauts-2023 test split's ground-truth fMRI and noise ceiling are not publicly distributed and are not provided here, and no trained weights are included, so none of the reported numbers can be independently recomputed."
resolution: "Authors: please provide (or give a resolvable accession/fetch script for) the test fMRI + noise-ceiling arrays and trained checkpoints, and document the expected data_dir layout."
cross_refs: ["missing-dependencies", "hardcoded-absolute-paths"]
check_script: _audit_code/check_cv_and_seed.py
paper_ref: "Section 4 (noise-ceiling-normalised encoding accuracy)"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-trainsize-sweep
category: missing
topic: "result traceability"
title: "No training-set-size sweep driver (Table 2)"
severity: low
confidence: medium
status: finding
file: datasets/nsd.py
line_start: 229
line_end: 234
quote: |
        num_train = int(np.round(len(train_img_list) / 100 * 90))
        # Shuffle all training stimulus images
        idxs = np.arange(len(train_img_list))

        if args.run < 20:
            np.random.shuffle(idxs)
claim: "Table 2 reports TBEn accuracy at training-set sizes 550/1100/2200/4400/8800, but the data loader always fixes num_train at 90% of all training images with no size argument, and no subsampling driver was found."
concern: "The data-efficiency result (a stated selling point for small-scale experiments) is not reproducible from the provided code."
resolution: "Authors: please add the training-set-size subsampling code used for Table 2."
cross_refs: []
paper_ref: "Table 2"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-supplementary-analyses
category: missing
topic: "result traceability / supplementary analysis"
title: "ROI-query cosine-similarity analysis (Figs S5–S8) not in repo"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  We analyzed the representational similarity of learned ROI queries, and report the average cosine similarity between each pair of ROIs across 20 models
claim: "Supplementary Figs S5–S8 report pairwise cosine similarities of learned ROI queries averaged over 20 models, but no script computing or plotting query cosine-similarity matrices was found in the repo."
concern: "A supplementary interpretability analysis is not reproducible from the provided code."
resolution: "Authors: please add the ROI-query cosine-similarity computation/plotting code."
cross_refs: []
paper_ref: "Supplementary A.3; Figs S5–S8"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: hardcoded-absolute-paths
category: bug
topic: "reproducibility / hardcoded paths"
title: "Notebooks hardcode internal /engram cluster paths and chdir"
severity: low
confidence: high
status: finding
file: datasets/nsd_utils.py
line_start: 153
line_end: 154
quote: |
    #challenge_data_dir = '../algonauts_2023_challenge_data'
    challenge_data_dir = '/engram/nklab/algonauts/algonauts_2023_challenge_data/'
claim: "Plotting/eval code hardcodes challenge_data_dir = '/engram/nklab/algonauts/algonauts_2023_challenge_data/' (same string appears in visualize_results.ipynb, which also os.chdir's to '/engram/nklab/hossein/recurrent_models/transformer_brain_encoder/'); these paths exist only on the authors' cluster."
concern: "The evaluation/plotting code will crash out-of-the-box on any other machine, blocking reproduction even if the data were available."
resolution: "Authors: parameterise the cluster paths (CLI arg / config) and remove the hardcoded os.chdir."
cross_refs: ["missing-test-fmri-and-weights"]
paper_ref: "n/a"
tags: [heil:bronze, lones:stage-5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: cv-random-resample-not-kfold
category: difference
topic: "data splitting / cross-validation"
title: "Paper's '10-fold cross validation' is 10 unseeded random 90/10 resamples, ensembled"
severity: medium
confidence: high
status: finding
file: datasets/nsd.py
line_start: 229
line_end: 241
quote: |
        num_train = int(np.round(len(train_img_list) / 100 * 90))
        # Shuffle all training stimulus images
        idxs = np.arange(len(train_img_list))

        if args.run < 20:
            np.random.shuffle(idxs)

        if args.output_path:
            np.save(args.save_dir+ '/idxs.npy', idxs)
        
        # Assign 90% of the shuffled stimulus images to the training partition,
        # and 10% to the test partition
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
claim: "Each 'run' draws an independent random 90/10 train/val split via an unseeded np.random.shuffle (the seeds at main.py:30-31 are commented out); the eval notebook sets runs = arange(1,11) and averages the 10 runs' test predictions. This is a 10x random-resample ensemble, not k-fold cross-validation — no KFold/fold-partition code exists, so the held-out 10% folds overlap arbitrarily and do not partition the data."
concern: "The paper repeatedly calls this '10-fold cross validation'; the implemented procedure (overlapping random resamples, ensembled predictions) is itself valid but does not match the described k-fold protocol, and because runs are unseeded the exact splits are not reproducible."
resolution: "Authors: clarify whether results are from 10-fold CV or a 10x random-resample ensemble, and set/record seeds so the splits are reproducible."
cross_refs: ["missing-test-fmri-and-weights"]
check_script: _audit_code/check_cv_and_seed.py
paper_ref: "Section 4: 'we did 10-fold cross validation using the training set'"
tags: [reforms:4, lones:stage-4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

No `methodology` finding. Topics actively checked and found sound (a finding-free pass, not a skip):

- **Model selection / early stopping does not touch the test set.** In `main.py:432-474`, test
  predictions are dumped only when `val_perf` (validation correlation) improves; the test fMRI is
  never read during training. No test-loss-based checkpointing → no early-stopping leakage.
- **The held-out test set is genuinely held out by construction** (Algonauts last-3-sessions
  split); train/val splits operate only on the training images (`datasets/nsd.py:216-261`); the
  test loader (`:263-285`) is a separate directory.
- **Noise-ceiling normalisation and r²-capping** in `visualize_results.ipynb` follow the standard
  Algonauts procedure (explained variance capped at 1).
- **Pretraining contamination**: N/A in the strong sense — backbones are frozen and the encoder
  is trained from scratch on NSD; no claim depends on backbone train/test overlap beyond what the
  authors acknowledge.
- **Temporal integrity**: N/A — the "future sessions" split is the intended design, not a
  shuffled time series within features.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 10         | high         | BERT/saliency/YOLO/BrainDiVE/PCA code, deps, test fMRI/weights, ridge grid-search, size-sweep all absent |
| bug         | 1          | low          | Hardcoded /engram cluster paths break out-of-box runs      |
| difference  | 1          | medium       | "10-fold CV" is really a 10x unseeded random-resample ensemble |
| methodology | 0          | -            | Model selection on val (not test); no leakage found        |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)

1. **[missing] `missing-test-fmri-and-weights`** (high/high) — the held-out test fMRI, noise
   ceiling, per-run predictions, and trained weights are all absent with no fetch script and
   hardcoded internal paths, so *no* reported number can be independently recomputed.
2. **[missing] `missing-bert-text-modality`** (high/high) — no BERT backbone exists; the entire
   Table 6 cross-modality result is untraceable.
3. **[difference] `cv-random-resample-not-kfold`** (medium/high) — the paper's "10-fold cross
   validation" is implemented as 10 unseeded random 90/10 resamples ensembled together.
4. **[missing] `ridge-baseline-no-gridsearch`** (medium/high) — the paper's grid-searched ridge
   penalty is not implemented; the strongest baseline may be under-tuned vs its description.
5. **[missing] `missing-saliency-baseline`** (medium/high) — the DeepGaze saliency baseline
   (Tables 1/3/5) has no code.
6. **[missing] `missing-dependencies`** (medium/high) — no requirements/environment file ships
   with the repo.

### Items that genuinely look fine

- Test set is held out by construction; train/val resampling never touches the test images.
- Model/epoch selection uses validation performance, not test loss — no early-stopping leakage.
- The TBEn architecture, masked ROI readout, and the encoding-accuracy metric (noise-ceiling-
  normalised squared Pearson correlation, capped at 1) are implemented as described.
- DINOv2 / ResNet50 / CLIP backbone branches are all present and wired into `build_backbone`.

### Open questions for the authors

- Are the headline numbers from true 10-fold CV or a 10x random-resample ensemble, and what seeds
  were used? (`cv-random-resample-not-kfold`)
- What ridge penalty was selected per subject/backbone, and where is the grid-search code?
  (`ridge-baseline-no-gridsearch`)
- Can you release the test fMRI / noise ceiling (or an accession) and trained checkpoints so the
  reported accuracies can be verified? (`missing-test-fmri-and-weights`)
