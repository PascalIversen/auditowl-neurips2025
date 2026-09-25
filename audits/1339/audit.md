# Code-repository audit — E³AD: Embodied Cognition Augmented End2End Autonomous Driving (paper 1339)

## 1. Summary

The paper proposes **E³AD**: a two-stage pipeline that (1) trains a "Driving-Thinking Model"
(a Video Swin Transformer) by CLIP-style contrastive learning against a frozen EEG large model
(LaBraM) on a self-collected paired video–EEG dataset, and (2) freezes that model and injects its
features into mainstream end-to-end driving planners (VAD, UniAD, GenAD, LAW). It reports
open-loop nuScenes results (Table 1), Bench2Drive/CARLA closed-loop results (Table 2, A.6),
ablations (Table 3), a three-framework comparison (Table 4), hyper-parameter sweeps (A.3–A.5),
an L2/collision trade-off study (A.2), and a Singapore↔Boston cross-dataset study (A.7).

**Repo audited.** `code_links.txt` PRIMARY points at `github.com/AIR-DISCOVER/E-cubed-AD`; the
default `main` branch clone (`code/AIR-DISCOVER__E-cubed-AD/`) contains only a README that says
*"code coming soon"*. The real code is on the **non-default `E-VAD` branch**
(`code/AIR-DISCOVER__E-cubed-AD__E-VAD/`, HEAD `ad78ad0`, "Update E-VAD code", committed
2026-03-11 — after NeurIPS 2025). I audited that branch. It is a fork of VAD/BEVFormer
(`mmdet3d`) plus a `projects/eeg_vedio/` Stage-1 contrastive module and one Stage-2 integration
(`projects/mmdet3d_plugin/EAD/`, `type='EAD'`, a VAD-Base variant).

**What I ran** (read-only over `code/`; scripts under `_audit_code/`):
- `_audit_code/check_repo_completeness.py` → `out/repo_completeness.{txt,csv}`: file-existence /
  grep census of the dataset loader, weights, per-model integrations, closed-loop code,
  cross-dataset split, config count, and Stage-1 split logic.
- `_audit_code/check_video_branch_freeze.py` → `out/video_branch_freeze.txt`: reproduces
  `create_mlp()` naming to prove the hardcoded "trainable" video keys match no real parameter.
- Manual reads of the Stage-1 module (`main.py`, `run_model/train.py`, `contrastive_model.py`,
  `video_encoder.py`, configs), the Stage-2 detector/head (`EAD.py`, `EAD_head.py`), and the
  single config `projects/configs/EAD/EAD_based_pretrain.py`.

**Headline.** The repository reproduces at most **one** of the paper's many reported experiments
(open-loop E³AD(VAD-Base) on nuScenes) — and even that only in principle, because every trained /
pretrained weight and the entire self-collected dataset are absent. The Stage-1 training code
cannot import (missing dataset module), and even if it could, a key-name bug freezes the whole
video branch it is meant to train. Closed-loop, all non-VAD integrations, both alternative
frameworks, the ablations, and the cross-dataset study have no code at all. The reported gains are
single-run point estimates (fractions of a percentage point on a noisy metric) with no error bars
or significance test.

## 2. Result-traceability table

Legend — **Verified**: code computes the value in-repo; **PARTIAL**: computing code is present but
the reported number cannot be produced from the repo because required weights/data/config are
absent; **MISSING**: no code computes the value.

| Paper artefact | Repo location | Computed in repo? | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — E³AD(VAD-Base) L2/Collision (nuScenes open-loop) | `EAD.py:676-724`, `planner/metric_stp3.py` (via config `EAD_based_pretrain.py`) | metric code present; needs `Driving_thinking_model.safetensors` (absent) + nuScenes | unrunnable | PARTIAL (see `missing-pretrained-and-trained-weights`) |
| Table 1 — E³AD(VAD-Tiny) | (none) | — | — | MISSING (`missing-nonvad-integrations`) |
| Table 1 — E³AD(UniAD) | (none) | — | — | MISSING (`missing-nonvad-integrations`) |
| Table 1 — E³AD(GenAD) | (none) | — | — | MISSING (`missing-nonvad-integrations`) |
| Table 1 — E³AD(LAW) | (none) | — | — | MISSING (`missing-nonvad-integrations`) |
| Abstract/§4.3 — "collision … decreased by 0.08% (UniAD) / 0.04% (VAD-Base)" | VAD-Base metric code only | partial | unrunnable | PARTIAL / MISSING (UniAD) |
| Table 2 — Bench2Drive open+closed-loop (Avg.L2, DS, SR) | (none) | — | — | MISSING (`missing-closedloop-code`) |
| §4.3 — closed-loop completion / driving-score deltas | (none) | — | — | MISSING (`missing-closedloop-code`) |
| Table 3 — ablation (contrastive on/off, freeze, expert/novice EEG) | (none; needs Stage-1 retrains) | — | — | MISSING (`missing-frameworks-and-ablation-code`) |
| Table 4 — Framework 1 "Attach to Spatio-temporal" | (none; no AttnGate/TokenLearner) | — | — | MISSING (`missing-frameworks-and-ablation-code`) |
| Table 4 — Framework 2 "Interact with Ego Query" | (none) | — | — | MISSING (`missing-frameworks-and-ablation-code`) |
| Table 4 — Framework 3 "Interact with Planning Features" | `EAD_head.py:779-821` (`eeg_decoder`) | code present | unrunnable | PARTIAL |
| Table A.2 — L2-loss doubling / Ego-MLP | (none) | — | — | MISSING (`missing-frameworks-and-ablation-code`) |
| Tables A.3 / A.5 — Framework 2 / Framework 1 sweeps | (none) | — | — | MISSING (`missing-frameworks-and-ablation-code`) |
| Table A.4 — Framework 3 sweep | one point only (`EAD_based_pretrain.py` 4-layer `eeg_decoder`) | partial | unrunnable | PARTIAL |
| Table A.6 — closed-loop infraction breakdown | (none) | — | — | MISSING (`missing-closedloop-code`) |
| Table A.7 — Singapore↔Boston cross-dataset | (none; geo split not implemented) | — | — | MISSING (`missing-frameworks-and-ablation-code`) |
| §3.1 — Stage-1 dataset, 80:10:10 split, 1894/236/237 clips | (none; `RealCarDataset` absent) | — | — | MISSING (`missing-stage1-dataset-and-loader`) |
| §3.2/§4.1 — Stage-1 contrastive training of Driving-Thinking model | `contrastive_model.py`, `run_model/train.py` | present but cannot import; video branch frozen by bug | unrunnable | PARTIAL (`stage1-broken-entrypoint-imports`, `stage1-video-branch-frozen-by-key-mismatch`) |

No numbered figure/table value in the paper can be *reproduced* from the repository as shipped.

## 3. Findings

### missing

```yaml finding
id: missing-pretrained-and-trained-weights
category: missing
topic: "expected code completeness / weights"
title: "No LaBraM, video-backbone, or Driving-Thinking weights; Stage-2 asserts a missing checkpoint"
severity: high
confidence: high
status: finding
file: projects/mmdet3d_plugin/EAD/EAD.py
line_start: 105
line_end: 107
quote: |
        config["ckpt_load_path"] = "ckpts/Driving_thinking_model.safetensors"
        self.cognitive_video_encoder = VideoEncoder(config, project_dir)
        self.cognitive_video_encoder.load_ckpts()
claim: "The Stage-2 EAD detector loads the frozen Driving-Thinking model from ckpts/Driving_thinking_model.safetensors, and VideoEncoder.load_ckpts asserts the file exists (video_encoder.py:41); no such file, nor the LaBraM checkpoint (ckpts/eeg_encoder_base.pth) or the video base (ckpts/video_encoder_base.safetensors), is committed anywhere and there is no download script (check found 0 weight files of any type)."
concern: "Without the trained Driving-Thinking checkpoint (and the pretrained LaBraM/Swin weights) neither stage can run and none of the paper's numbers can be reproduced; retraining is also blocked because the dataset is absent."
resolution: "Release the trained Driving-Thinking checkpoint plus the LaBraM and video-backbone weights (or resolvable download URLs/accessions), and state which checkpoint produced each table."
cross_refs: ["missing-stage1-dataset-and-loader", "stage2-proj-mlp-shape-mismatch"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 1; §4.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-stage1-dataset-and-loader
category: missing
topic: "data / result traceability"
title: "Self-collected EEG–video dataset, its loader, and the 80:10:10 split are all absent"
severity: high
confidence: high
status: finding
file: projects/eeg_vedio/src/run_model/train.py
line_start: 18
line_end: 18
quote: |
  from data.real_car_dataset import RealCarDataset
claim: "Both Stage-1 training scripts (train.py:18, train_ddp.py:23) import RealCarDataset from a data.real_car_dataset module that does not exist (there is no src/data directory); the dataset itself is referenced only via hardcoded private paths in cfgs/train_config.yaml (/home/tsinghuaair/...); and no code implements the paper's shuffled 80:10:10 clip split (check: 0 hits for train_test_split/random_split/80:10:10)."
concern: "The paired EEG–video dataset that is the paper's central new asset, the code that reads/splits it, and thus the entire Stage-1 contribution cannot be inspected or reproduced."
resolution: "Release the self-collected dataset (or the data-collection/preprocessing outputs) and the RealCarDataset loader and split code; the data-availability statement promises this ('we commit to publicly releasing the dataset ... soon')."
cross_refs: ["stage1-broken-entrypoint-imports", "stage1-clip-split-leakage"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "§3.1; Fig. 1 caption (1894/236/237 clips)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-nonvad-integrations
category: missing
topic: "result traceability / baselines"
title: "Only a VAD-Base integration exists; E³AD(VAD-Tiny/UniAD/GenAD/LAW) have no code"
severity: high
confidence: high
status: finding
file: out/repo_completeness.csv
csv_row: 4
quote: |
  uniad_integration_hits,0,E3AD(UniAD) rows in Table 1 & Table 2
claim: "The repo contains exactly one integration config (projects/configs/EAD/EAD_based_pretrain.py, a VAD-Base variant) and zero references to UniAD, GenAD, LAW, or a VAD-Tiny integration (grep counts, vendored torchvision excluded); Table 1 nonetheless reports E³AD applied to VAD-Tiny, UniAD, GenAD, and LAW."
concern: "Four of the six E³AD table-1 rows — and the headline UniAD collision-improvement claim in the abstract/§4.3 — cannot be reproduced or checked because the integration code is not present."
resolution: "Provide the UniAD/GenAD/LAW/VAD-Tiny integration code and configs, or restrict the claims to the VAD-Base model that is actually shipped."
cross_refs: ["missing-closedloop-code"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 1; Abstract; §4.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-closedloop-code
category: missing
topic: "result traceability / closed-loop"
title: "No Bench2Drive/CARLA closed-loop code; Tables 2 and A.6 are unreproducible"
severity: high
confidence: high
status: finding
file: out/repo_completeness.csv
csv_row: 7
quote: |
  closedloop_carla_bench2drive_code_hits,0,"Table 2 / A.6 closed-loop DS,SR; carla only pinned in requirements.txt"
claim: "There is no Bench2Drive or CARLA driving/evaluation code in the repo (0 grep hits outside the vendored torchvision copy); 'carla=0.9.12' appears only as one pinned line in requirements.txt, and the driving-score / success-rate metrics (DS, SR) are computed nowhere."
concern: "The entire closed-loop evaluation — Table 2 (DS, SR), §4.3 completion/driving-score deltas, and the Table A.6 infraction breakdown, which the paper presents as its strongest evidence — has no computing code and cannot be verified."
resolution: "Release the Bench2Drive/CARLA harness (routes, agent wrapper, scoring) used to produce Table 2 and A.6."
cross_refs: ["missing-nonvad-integrations"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 2; Table A.6; §4.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-frameworks-and-ablation-code
category: missing
topic: "ablations / framework comparison"
title: "Frameworks 1 & 2, all ablations, hparam sweeps, and the cross-dataset study have no code"
severity: medium
confidence: high
status: finding
file: out/repo_completeness.csv
csv_row: 10
quote: |
  framework1_attngate_tokenlearner_present,False,Framework 1 'Attach to Spatio-temporal' (Eq.5-6); head only wires eeg_decoder (Framework 3)
claim: "Only Framework 3 ('Interact with Planning Features', the eeg_decoder path in EAD_head.py:785) is implemented; Framework 1 (AttnGate/TokenLearner, Eq. 5-6) and Framework 2 (ego-query interaction, Eq. 8) are absent, as are the Table 3 ablations (contrastive on/off, unfreeze, expert/novice/mixed EEG), the A.2 L2-loss/Ego-MLP study, the A.3/A.5 sweeps, and the A.7 Singapore↔Boston geographic split (singapore/boston appear only as nuScenes map metadata, not a train-on-one/test-on-other split)."
concern: "Table 3 (which is the paper's core evidence that gains come from EEG contrastive learning rather than extra capacity), Table 4, and appendix Tables A.2/A.3/A.5/A.7 cannot be reproduced or audited."
resolution: "Provide the Framework 1/2 code and the ablation/sweep/cross-dataset configs and drivers used for Tables 3, 4, A.2, A.3, A.5, and A.7."
cross_refs: ["missing-nonvad-integrations"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Tables 3, 4, A.2, A.3, A.5, A.7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-readme-repro-instructions
category: missing
topic: "expected code completeness / documentation"
title: "No README results table or reproduction commands; released code is an untagged post-hoc branch"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  ## E-cubed-AD
claim: "The E-VAD-branch README is 14 bytes with no results table, no run commands, and no data-preparation instructions; the default main branch README says only 'code coming soon'; the audited code lives on a non-default branch whose single commit is dated 2026-03-11 (after NeurIPS 2025) rather than a submission-tagged commit."
concern: "A reviewer has no documented path from repo to any paper number, and it is unclear which code state produced the reported results (moving branch, no tag)."
resolution: "Add a README with the exact commands and environment for each table, and tag the commit that generated the paper's numbers."
cross_refs: ["missing-pretrained-and-trained-weights"]
paper_ref: "NeurIPS checklist Q5 (open access to code)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: stage1-video-branch-frozen-by-key-mismatch
category: bug
topic: "training correctness"
title: "Hardcoded 'trainable' keys (proj.0/proj.2) match no parameter → whole video branch frozen"
severity: medium
confidence: high
status: finding
file: projects/eeg_vedio/src/models/contrastive_model.py
line_start: 30
line_end: 30
quote: |
        self.video_encoder_missing_keys = ["proj.0.weight", "proj.0.bias", "proj.2.weight", "proj.2.bias"]
claim: "The freeze loop (lines 40-44) keeps a video parameter trainable only if its name is in video_encoder_missing_keys; but the real projection-head parameters are named proj.1.* / proj.3.* (create_mlp prepends a ReLU at index 0), so the hardcoded proj.0/proj.2 names match nothing and, under the configured fine_tuning_policy='missing_trainable' (train_config.yaml:16), every video parameter gets requires_grad=False — the entire video branch (Swin + projection) is frozen."
concern: "This contradicts the paper's Eq. (4)/§3.2 ('only the video-branch parameters θv = {gv, hv} are updated'): as written, Stage-1 contrastive training cannot update the video branch at all, so the released training code does not learn the described video→cognition alignment."
resolution: "Fix the trainable-key names to proj.1.*/proj.3.* (and proj_norm.*), or set requires_grad on the intended modules directly; confirm which parameters were actually trained for the released checkpoint."
cross_refs: ["stage1-broken-entrypoint-imports"]
check_script: _audit_code/check_video_branch_freeze.py
paper_ref: "§3.2 Eq. (4); §4.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage1-broken-entrypoint-imports
category: bug
topic: "runnability / dead imports"
title: "Stage-1 entrypoint imports a non-existent module and wrong function signature"
severity: medium
confidence: high
status: finding
file: projects/eeg_vedio/src/main.py
line_start: 4
line_end: 4
quote: |
  from training.train import train_model
claim: "main.py imports train_model from a training package that does not exist (the training code is under run_model/, not training/); it also calls train_model(args) whereas run_model/train.py defines train_model(epochs, batch_size, learning_rate) — a different signature. Independently, run_model/train.py and train_ddp.py import the missing data.real_car_dataset module and read hardcoded absolute paths, so every Stage-1 entrypoint raises ImportError before any training begins."
concern: "The Stage-1 training code cannot be executed as shipped, so the paper's central Driving-Thinking-model training is not runnable from the repository."
resolution: "Point main.py at the actual training module and matching signature, remove hardcoded paths, and ship the dataset loader it depends on."
cross_refs: ["missing-stage1-dataset-and-loader", "stage1-video-branch-frozen-by-key-mismatch"]
paper_ref: "§3.2; §4.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: stage1-hparams-and-temperature-vs-paper
category: difference
topic: "hyperparameters / training config"
title: "Config uses 30 epochs, lr 1e-4, fixed temperature 0.1, 0.2 s clips vs paper 120 epochs, 2e-5, learnable τ, 2 s"
severity: low
confidence: medium
status: finding
file: projects/eeg_vedio/cfgs/train_config.yaml
line_start: 11
line_end: 16
quote: |
  model:
    epochs: 30
    batch_size: 16
    learning_rate: 1e-4
    temperature: 0.1
    fine_tuning_policy: "missing_trainable" # "all_trainable" or "missing_trainable"
claim: "The runnable Stage-1 config sets 30 epochs, lr 1e-4, and a fixed temperature 0.1 (contrastive_model.py:78 divides logits by the constant self.temperature), and data_duration 0.2 s (train_config.yaml:6); the paper §4.1 states 120 epochs, lr 2e-5, and Eq. (2) a learnable log-temperature β, on 2 s clips at 2 fps."
concern: "The shipped training recipe does not match the paper's stated recipe; the temperature is a constant, not the learnable β of Eq. (2)."
resolution: "Confirm which epochs/lr/temperature/clip-duration were used for the reported models and align the config (or the paper) accordingly."
cross_refs: []
paper_ref: "§4.1; §3.2 Eq. (2)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

```yaml finding
id: headline-gains-single-run-no-significance
category: methodology
topic: "statistical integrity / evaluation"
title: "Sub-0.1% collision gains reported as single point estimates with no variance or significance test"
severity: high
confidence: medium
status: finding
file: paper.pdf
quote: |
  significantly enhances the end-to-end planning performance of baseline models
claim: "Tables 1-4 and A.2-A.7 report single numbers with no error bars, seeds, or significance test (NeurIPS checklist Q7 answers 'Yes' to error bars, but none appear); the main open-loop improvements are ~0.04-0.08 percentage points of collision rate on nuScenes open-loop planning — a metric known to be dominated by a handful of samples and by ego-status — yet the abstract describes the effect as 'significant'."
concern: "Effects of a fraction of a percentage point from single runs are within the documented noise of nuScenes open-loop planning; without multiple seeds and a significance test the headline 'significant improvement' claim is not supported."
resolution: "Report mean ± std over multiple seeds and a paired significance test (e.g. per-sample McNemar on collisions), for both the proposed models and the baselines under identical evaluation."
cross_refs: []
paper_ref: "Abstract; Table 1; §4.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### questions (evidence incomplete — for the authors)

```yaml finding
id: stage1-clip-split-leakage
category: methodology
topic: "sample independence / data splitting"
title: "Shuffled clip-level 80:10:10 split over 20 drivers risks subject/temporal leakage (unverifiable)"
severity: medium
confidence: low
status: question
file: paper.pdf
quote: |
  The structured data was shuffled and divided into two parts with a ratio of 80:10:10.
claim: "The paper splits 2 s clips cut from the continuous sessions of only 20 drivers (10 expert, 10 novice) across 14 condition segments by shuffling at the clip level; temporally adjacent (near-duplicate) clips and other clips from the same driver/condition can then land in both train and test."
concern: "If any model-selection or reported Stage-1 metric depends on this split, subject/temporal near-duplicate leakage could inflate it; the split code is not in the repo (RealCarDataset absent) so this cannot be verified either way."
resolution: "Provide the split implementation and confirm whether the split is subject-disjoint; report a driver-held-out contrastive result if selection touched the val/test clips."
cross_refs: ["missing-stage1-dataset-and-loader"]
paper_ref: "§3.1; Fig. 1 caption"
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

```yaml finding
id: stage2-proj-mlp-shape-mismatch
category: bug
topic: "weight loading / architecture"
title: "Stage-2 video projection is [200,200] but Stage-1 config is [512,200] — strict=False may drop it"
severity: medium
confidence: low
status: question
file: projects/eeg_vedio/cfgs/video_encoder_inference.yaml
line_start: 3
line_end: 7
quote: |
  video_encoder:
    ckpt_load_path: "ckpts/first_try/video_encoder_ft_1.safetensors"
    # ckpt_load_path: None
    # ckpt_save_path: "ckpts/video_encoder_ft_1.safetensors"
    mlp_layers: [200, 200]
claim: "EAD.py (Stage 2) builds the cognitive VideoEncoder from this inference config with mlp_layers=[200,200] (proj = 1024->200->200), whereas the Stage-1 train_config.yaml uses mlp_layers=[512,200] (proj = 1024->512->200); VideoEncoder.load_ckpts calls load_state_dict(strict=False), which silently skips shape-mismatched proj.1/proj.3 weights."
concern: "If the released Driving-Thinking checkpoint was trained with [512,200], loading it into the [200,200] Stage-2 model would leave the projection randomly initialised, so the 'cognitive' features injected into planning would not be the contrastively-aligned ones."
resolution: "Confirm the exact proj architecture of the released checkpoint matches the Stage-2 model, and log load_state_dict missing/unexpected keys at inference to prove the projection actually loads."
cross_refs: ["missing-pretrained-and-trained-weights"]
paper_ref: "§4.1 (two-layer MLP adapter, 200-d space)"
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 6          | high         | Weights, dataset, all non-VAD integrations, all closed-loop, ablations, and docs absent |
| bug         | 2 (+1 question) | medium  | Video branch frozen by key-name mismatch; Stage-1 entrypoints don't import |
| difference  | 1          | low          | Stage-1 epochs/lr/temperature/clip-duration differ from paper |
| methodology | 1 (+1 question) | high    | Sub-0.1% single-run gains, no error bars/significance |

## 5. Closing lists

**Top take-aways** (ranked by severity × confidence):
1. `missing-pretrained-and-trained-weights` *(missing, high/high)* — no trained/pretrained weights and no fetch; Stage-2 asserts a missing checkpoint, so nothing runs.
2. `missing-stage1-dataset-and-loader` *(missing, high/high)* — the self-collected dataset, its loader, and the 80:10:10 split are entirely absent.
3. `missing-closedloop-code` *(missing, high/high)* — no CARLA/Bench2Drive code; the closed-loop results (Table 2, A.6) cannot be reproduced.
4. `missing-nonvad-integrations` *(missing, high/high)* — only VAD-Base exists; UniAD/GenAD/LAW/VAD-Tiny rows have no code.
5. `headline-gains-single-run-no-significance` *(methodology, high/medium)* — fraction-of-a-percent gains from single runs with no significance test undercut the "significant improvement" claim.
6. `stage1-video-branch-frozen-by-key-mismatch` *(bug, medium/high)* — a key-name bug freezes the video branch Stage-1 is supposed to train.

**Items that genuinely look fine** (actively checked):
- Stage-2 open-loop metrics are *computed* in-repo (`EAD.py:676-724` + `planner/metric_stp3.py`), not merely plotted — so the VAD-Base open-loop pipeline is real, just unrunnable without weights/data.
- Stage-2 inference uses only the nuScenes front-camera queue, no EEG (`EAD.py:410-414`), and the cognitive encoder is frozen (`EAD.py:111-112`) — consistent with the paper's "no EEG at stage 2, model frozen" fairness claim.
- Ego status is not fed to the planner (`ego_his_encoder=None`, `ego_lcf_feat_idx=None` in `EAD_based_pretrain.py:81`), matching the Table 1 caption "ego status was not used".
- Dependencies are specified and pinned (`requirements.txt`, 204-line conda export).
- Evaluating on the nuScenes `val` split as the test set is standard for this benchmark (hidden test labels) — not a finding.

**Open questions for the authors**:
- `stage1-clip-split-leakage` — is the Stage-1 80:10:10 split subject-disjoint, or can clips from one driver appear in both train and test?
- `stage2-proj-mlp-shape-mismatch` — does the released Driving-Thinking checkpoint's projection head actually load into the [200,200] Stage-2 model (no silently dropped keys)?
- Which exact commit / branch and which checkpoints produced each table, given the audited code is an untagged branch dated after publication?
