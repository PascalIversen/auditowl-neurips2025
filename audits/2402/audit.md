# Code-Repository Audit — SignViP (NeurIPS 2025, paper 2402)

## 1. Summary

The repository `umnooob/signvip` (commit `54a28b3`) is the author code for *"Advanced
Sign Language Video Generation with Compressed and Quantized Multi-Condition
Tokenization"* (SignViP). The paper is a three-stage generative pipeline (Sign Video
Diffusion Model → FSQ Autoencoder → Multi-Condition Token Translator). The repo contains
the three training entry points (`train_stage1_multi_cond.py`, `train_stage2_multi_cond.py`,
`train_compress_vq_multicond.py`, `train_multihead_t2vqpgpt.py`), the model definitions
(`models/`), dataset preprocessing scripts (`scripts/`, `signdatasets/`), inference/video
generation scripts (`combined_t2s_eval.py`, `eval_compress_vq_video.py`,
`eval_compress_video_from_origin.py`, `eval_multihead_t2vqpgpt.py`), a `requirements.txt`,
and three metric calculators (`metrics/calculate_fid.py`, `calculate_fvd.py`,
`calculate_ssim.py`).

What I did: read the paper (PDF + plain-text), enumerated every reported table/figure
metric, and grepped the entire repo for the code that computes each. I wrote one read-only
check, `_audit_code/check_metric_coverage.py`, which (a) searches for computation code for
each paper metric, (b) tests existence of README-referenced files, and (c) counts files
with hardcoded absolute paths. Its outputs are in `_audit_code/out/`. I did **not** run the
pipeline (it requires multi-GPU, unreleased checkpoints, and the RWTH-2014T/How2Sign video
corpora).

Headline result of the audit: the **evaluation metrics that constitute every comparison
table in the paper are not computed by any code in the repo.** The four eval scripts only
*generate videos / pose tokens*; no script ingests generated and ground-truth videos and
produces FID, CLIP-FID, FVD, IDS, PSNR, SSIM, LPIPS, or Hand-SSIM for the comparison
tables, and there is no back-translation (SLT) model or BLEU/ROUGE/COMET computation at all.
The three metric calculators that do exist are wired only into training-time validation
(checkpoint selection), not into a table-producing evaluation.

## 2. Traceability table

Result of `_audit_code/check_metric_coverage.py` (see `out/metric_coverage.csv`). "Computes"
means code that takes generated + reference data and produces the reported number; video/
token *generation* scripts are not sufficient (Rule G).

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — video back-translation BLEU/ROUGE/COMET (both datasets, all methods) | (none) | — | — | MISSING (no SLT model, no BLEU/ROUGE/COMET code) |
| Table 2 — pose back-translation BLEU/ROUGE/COMET | (none) | — | — | MISSING |
| Table 3 — FID | `metrics/calculate_fid.py` exists but is never called (only a commented import in `train_stage1_multi_cond.py:27`) | No (not invoked) | — | MISSING (no driver computes the Table 3 FID) |
| Table 3 — CLIP-FID | (none) | — | — | MISSING |
| Table 3 — FVD | `metrics/calculate_fvd.py` (called only in `train_stage2_multi_cond.py` validation) | No standalone table driver | — | MISSING (computed only for checkpoint selection, not the comparison table) |
| Table 3 — IDS (YOLO5Face + Arc2Face) | (none) | — | — | MISSING |
| Table 4 — PSNR | (none) | — | — | MISSING |
| Table 4 — SSIM | `metrics/calculate_ssim.py` (called only in `train_stage1_multi_cond.py` validation) | No standalone table driver | — | MISSING |
| Table 4 — LPIPS | (none) | — | — | MISSING |
| Table 4 — Hand SSIM | (none) | — | — | MISSING |
| Table 5 — ablation (w/o 3D Hands / w/o poses) FVD, Hand SSIM, BLEU-4, ROUGE | (none — depends on missing FVD/SSIM/BLEU) | — | — | MISSING |
| Table 6 — human evaluation vote% | (none; human study) | — | — | N/A (human study, not code) |
| Table 7 — direct-condition-prediction back-translation | (none) | — | — | MISSING |
| Table 8 — #params, inference time | `combined_t2s_eval.py:291-369` prints param counts | params: yes; time: not logged to a table | unverified | PARTIAL |
| Table 9 — pretrained-init ablation FID/FVD | (none — depends on missing FID; FVD only in training val) | — | — | MISSING |
| Tables 10,11 — Norm. DTW token-translation | `train_multihead_t2vqpgpt.py:208-214` (`fastdtw`, in training validation) | DTW: yes (training val); BLEU-4/ROUGE columns: no | unverified | PARTIAL |
| Fig. 4(d) — FSQ vs VQ codebook usage | (none) | — | — | MISSING |
| Fig. 4(e) — FSQ vs VQ reconstruction loss | training loss exists in `train_compress_vq_multicond.py`; no FSQ-vs-VQ sweep driver | No comparison driver | — | MISSING |
| Token-level accuracy | `eval_multihead_t2vqpgpt.py:108-135` | yes | not a paper table number | (computed but unreported) |

Every comparison-table metric routes to MISSING. The single largest defect (no metric code
for any comparison table) is filed once below and cross-referenced.

## 3. Findings

## missing

```yaml finding
id: comparison-table-metrics-not-computed
category: missing
topic: "result traceability"
title: "No code computes the metrics in any comparison table (Tables 1-9)"
severity: high
confidence: high
status: finding
file: combined_t2s_eval.py
line_start: 451
line_end: 593
quote: |
    # Dataset evaluation mode
    else:
        # Load datasets for batch processing
        dataloaders = create_dataset_dataloader(t2s_cfg, t2s_tokenizer)
        t2s_model, *dataloaders = accelerator.prepare(t2s_model, *dataloaders)
claim: "The four eval scripts (combined_t2s_eval.py, eval_compress_vq_video.py, eval_compress_video_from_origin.py, eval_multihead_t2vqpgpt.py) only generate videos / pose tokens and save them to disk; none ingests generated + ground-truth videos and computes FID, CLIP-FID, FVD, IDS, PSNR, SSIM, LPIPS, Hand-SSIM, BLEU, ROUGE, or COMET for the paper's comparison tables. The repo-wide grep in _audit_code/out/metric_coverage.csv finds zero files computing CLIP-FID, IDS, PSNR, LPIPS, Hand-SSIM, BLEU, ROUGE, or COMET."
concern: "Every headline quantitative claim in Tables 1-9 (the entire empirical case for state-of-the-art performance) is untraceable to code, so the reported numbers cannot be reproduced or verified."
resolution: "Authors: please provide the evaluation driver(s) that load generated and reference videos/poses and emit each table's metrics (FID/CLIP-FID/FVD/IDS, PSNR/SSIM/LPIPS/Hand-SSIM, and the back-translation BLEU/ROUGE/COMET)."
cross_refs: ["backtranslation-slt-model-missing", "fid-calculator-never-invoked", "fvd-ssim-only-in-training-val"]
check_script: _audit_code/check_metric_coverage.py
paper_ref: "Tables 1-9"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: backtranslation-slt-model-missing
category: missing
topic: "evaluation / back-translation"
title: "Back-translation SLT model and BLEU/ROUGE/COMET pipeline absent"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  "To evaluate the semantic consistency of the generated sign language videos, we follow
  ProTran [58] to train two SLP models [6] to translate sign language videos (i.e., the
  video back-translation model) and poses (i.e., the pose back-translation model) into
  texts, respectively."
claim: "The paper's Appendix C describes training a video and a pose back-translation (SLT) model used to produce all BLEU/ROUGE/COMET numbers in Tables 1, 2, and 7, but the repo contains no SLT model definition, no training script for it, no inference/back-translation script, and no BLEU/ROUGE/COMET computation (confirmed by repo-wide grep, _audit_code/out/metric_coverage.csv)."
concern: "The semantic-fidelity claims (a core contribution) rest entirely on back-translation metrics that no released code can produce, so the semantic comparison is unreproducible."
resolution: "Authors: please release the back-translation SLT models (or checkpoints) and the script that computes BLEU/ROUGE/COMET from generated videos/poses."
cross_refs: ["comparison-table-metrics-not-computed"]
check_script: _audit_code/check_metric_coverage.py
paper_ref: "Appendix C; Tables 1, 2, 7"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fid-calculator-never-invoked
category: missing
topic: "result traceability"
title: "FID calculator present but never called by any live code path"
severity: medium
confidence: high
status: finding
file: train_stage1_multi_cond.py
line_start: 27
line_end: 28
quote: |
  # from metrics.calculate_fid import calculate_fid as calc_fid  # 添加 FID 计算导入
  from metrics.calculate_ssim import calculate_ssim as calc_ssim  # 添加 SSIM 计算导入
claim: "metrics/calculate_fid.py defines calculate_fid, but its only import in the repo (train_stage1_multi_cond.py:27) is commented out; no other file calls calculate_fid. Thus the FID/CLIP-FID/Table 9 FID numbers have no producing code."
concern: "The FID column in Table 3 and Table 9 cannot be reproduced because the FID function is dead code (never invoked)."
resolution: "Authors: point to the script that actually invokes FID computation over generated vs. ground-truth frames, or confirm FID was computed off-repo."
cross_refs: ["comparison-table-metrics-not-computed"]
check_script: _audit_code/check_metric_coverage.py
paper_ref: "Table 3 (FID), Table 9"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fvd-ssim-only-in-training-val
category: missing
topic: "result traceability"
title: "FVD/SSIM computed only inside training validation, no table driver"
severity: medium
confidence: high
status: finding
file: train_stage2_multi_cond.py
line_start: 386
line_end: 394
quote: |
        fvd_result = calc_fvd(
            fvd_all_real,  # 真实视频 [N, T, C, H, W]
            fvd_all_generated,  # 生成视频 [N, T, C, H, W]
claim: "calculate_fvd is invoked only inside the validation/checkpoint-selection loop of train_stage2_multi_cond.py (and calculate_ssim only inside train_stage1_multi_cond.py validation). There is no standalone evaluation script that computes the FVD/SSIM values reported in the cross-method comparison Tables 3 and 4."
concern: "The FVD and SSIM table numbers (and the FVD-based ablations in Table 5/Fig 4) are produced for checkpoint selection on validation data, not by a reproducible test-set evaluation matching the reported protocol, so the table values are not directly traceable."
resolution: "Authors: provide the test-set evaluation script that computes the FVD/SSIM/Hand-SSIM table values for SignViP and each baseline under the same protocol."
cross_refs: ["comparison-table-metrics-not-computed"]
check_script: _audit_code/check_metric_coverage.py
paper_ref: "Tables 3, 4, 5; Figure 4(a)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baseline-code-absent
category: missing
topic: "baselines"
title: "No code for any baseline (SignGAN, SignGen, MoMP, ControlNet, AnimateAnyone)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "in video back-translation comparison, SignViP outperforms all competing methods,
  including SignGAN [61], its enhanced version using AnimateAnyone [29], and SignGen [47]."
claim: "Tables 1-4 compare against SignGAN, SignGAN+AnimateAnyone, SignGen, MoMP+ControlNet, MoMP+AnimateAnyone, ProTran, Adversarial, MDN, ControlNet, and AnimateAnyone, but the repo contains no implementation, config, or invocation for any of these baselines (no files matching these method names outside the README/citation)."
concern: "Without baseline code under the same split/metric/preprocessing it cannot be verified that the comparisons are fair (e.g. equal tuning budget), and the SOTA claim cannot be reproduced."
resolution: "Authors: release the baseline configs/checkpoints or the harness used to evaluate the competing methods under the identical protocol."
cross_refs: ["comparison-table-metrics-not-computed"]
paper_ref: "Tables 1, 2, 3, 4"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pretrained-checkpoints-not-released
category: missing
topic: "expected code completeness"
title: "Trained checkpoints required for evaluation not released (How2Sign 'coming soon')"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 77
line_end: 81
quote: |
  ## 📦 Model Checkpoint

  * RWTH-T Models: [[huggingface]](https://huggingface.co/umnooob/signvip/tree/main/RWTH)
  * How2Sign Models: *comming soon*
claim: "The eval scripts require trained checkpoints (e.g. cfg.modules.ckpt, condition_encoder, unet, vq_model). The README provides only RWTH-T weights and marks How2Sign weights as 'comming soon', so the How2Sign half of every results table cannot be reproduced from released artefacts."
concern: "All How2Sign results (a full column of every comparison table) depend on weights that are not released, blocking reproduction of half the empirical claims."
resolution: "Authors: release the How2Sign checkpoints (and confirm the RWTH-T HuggingFace link contains all stages: diffusion model, FSQ autoencoder, token translator, and back-translation models)."
cross_refs: ["comparison-table-metrics-not-computed"]
paper_ref: "Tables 1-9 (How2Sign columns)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-references-nonexistent-files
category: missing
topic: "documentation / reproducibility commands"
title: "README training commands reference files that do not exist in the repo"
severity: low
confidence: high
status: finding
file: README.md
line_start: 88
line_end: 99
quote: |
  # Single frame training
  accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 2 --gpu_ids "0,1" \
    train_stage_1.py --config "configs/stage1/stage_1_multicond_RWTH.yaml"

  # Temporal-Attention Layer training
  accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 2 --gpu_ids "0,1" \
    train_stage_2.py --config "configs/stage2/stage_2_RWTH.yaml"
claim: "The README's Stage I commands call train_stage_1.py and train_stage_2.py, and the structure section lists train.sh, but none of these files exists in the repo (the actual files are train_stage1_multi_cond.py and train_stage2_multi_cond.py). README also lists scripts/RWTH-T/3_process_annotation.py whereas the repo has 3_process_annotion.py. Verified by _audit_code/out/readme_files.csv."
concern: "The exact reproduce-commands in the README fail as written, so a reader cannot run the documented pipeline without guessing the correct filenames."
resolution: "Authors: update the README to the actual script names, or add the referenced wrapper files."
cross_refs: []
check_script: _audit_code/check_metric_coverage.py
paper_ref: "README Training section"
tags: [lones:stage-1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-absolute-paths
category: bug
topic: "configuration / portability"
title: "Eval/training scripts and configs default to hardcoded /deepo_data absolute paths"
severity: medium
confidence: high
status: finding
file: combined_t2s_eval.py
line_start: 52
line_end: 59
quote: |
    parser.add_argument(
        "--t2s_config",
        type=str,
        help="Config file for text-to-pose model (GPT)",
        default="/deepo_data/signvip_v2/configs/gpt/eval_multihead_t2vqpgpt_RWTH.yaml",
    )
    parser.add_argument(
        "--vq_config",
claim: "19 of the repo's .py files hardcode absolute paths under /deepo_data/ as defaults (confirmed in _audit_code/out/hardcoded_paths.csv), and the eval configs (e.g. configs/gpt/eval_multihead_t2vqpgpt_RWTH.yaml) point ckpt to /t2pgpt/model.bin and datasets to /deepo_data/... and /RWTH/...; none of these paths exist outside the authors' machine."
concern: "The scripts will not run out-of-the-box: every default config/checkpoint/dataset path must be hand-edited, and the configs reference paths that are absent from the release, impeding reproduction."
resolution: "Authors: replace absolute defaults with repo-relative paths or required CLI args, and document where each checkpoint/dataset must be placed."
cross_refs: ["pretrained-checkpoints-not-released"]
check_script: _audit_code/check_metric_coverage.py
paper_ref: "N/A (engineering)"
tags: [lones:stage-1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: token-accuracy-computed-not-reported
category: difference
topic: "evaluation consistency"
title: "eval_multihead_t2vqpgpt.py computes token accuracy, a metric not in the paper"
severity: low
confidence: medium
status: finding
file: eval_multihead_t2vqpgpt.py
line_start: 108
line_end: 135
quote: |
        # Calculate accuracy
        for i in range(pose_latents.shape[0]):
            pred = pred_idx[i, : pose_len[i]]
            gt = pose_latents[i, : pose_len[i]]
            # print(f"pred: {pred.shape}, gt: {gt.shape}")
            valid_acc.append((pred == gt).sum().item() / pose_len[i].item())
claim: "The token-translator evaluation script computes exact token-match accuracy against ground-truth tokens, but the paper explicitly avoids token accuracy (Appendix G: 'Directly computing the accuracy of token translation ... presents significant challenges') and reports normalized DTW instead. The released eval script computes a metric the paper says is unreliable, not the DTW metric the paper reports."
concern: "The released token-translator evaluation does not implement the paper's stated DTW-based protocol (Tables 10/11); the implemented accuracy metric is internally valid but does not match what the paper reports."
resolution: "Authors: confirm whether the DTW numbers in Tables 10/11 were produced by the training-validation DTW path (train_multihead_t2vqpgpt.py:208-214) rather than this eval script, and release the DTW evaluation entry point."
cross_refs: ["comparison-table-metrics-not-computed"]
paper_ref: "Appendix G; Tables 10, 11"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A — the procedure that the present code actually implements (diffusion training,
FSQ autoencoding, autoregressive token translation, condition augmentation, scheduled
sampling) is methodologically standard and I found no leakage, no test-set tuning, and no
statistical-integrity defect within the *implemented* code. The audit's concerns are about
absence of the evaluation/metric/baseline code (filed under `missing`), not about an invalid
implemented procedure. The paper reports no formal significance tests, so statistical-integrity
checks are not applicable. Sample-independence/temporal-integrity checks are N/A: this is a
conditional video-generation task with fixed dataset-provided train/val/test splits, not a
predictive task with a leakage-prone split.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 7          | high         | No code computes any comparison-table metric; back-translation SLT model, baselines, How2Sign weights all absent |
| bug         | 1          | medium       | 19 files hardcode /deepo_data absolute paths; configs point to nonexistent ckpt/data paths |
| difference  | 1          | low          | Token-translator eval computes token accuracy, not the paper's DTW metric |
| methodology | 0          | -            | Implemented procedure is sound; concerns are absence (→ missing), not invalidity |

## 5. Closing lists

**Top take-aways** (ranked by severity × confidence):
1. (`missing`) No code computes the metrics in **any** comparison table (FID/CLIP-FID/FVD/IDS, PSNR/SSIM/LPIPS/Hand-SSIM) — Tables 1-9 are untraceable. [high]
2. (`missing`) The back-translation SLT model and BLEU/ROUGE/COMET pipeline (the basis of all semantic-fidelity claims) are entirely absent. [high]
3. (`missing`) No baseline implementation/config (SignGAN, SignGen, MoMP, ControlNet, AnimateAnyone), so SOTA comparisons cannot be reproduced or checked for fairness. [medium]
4. (`missing`) How2Sign checkpoints are not released ("coming soon"), blocking half of every results table. [medium]
5. (`missing`) FID calculator exists but is dead code (only a commented import); FVD/SSIM run only inside training validation, not a table-producing test evaluation. [medium]
6. (`bug`) 19 scripts and the eval configs hardcode absolute `/deepo_data` paths and reference checkpoint/data paths absent from the release. [medium]

**Items that genuinely look fine** (actively checked):
- Condition augmentation rate `pose_aug_rate: 0.001` in `configs/stage2/stage_2_RWTH.yaml:4` matches the paper's stated p = 10⁻³.
- Scheduled-sampling 40% token replacement: `train_multihead_t2vqpgpt.py:464-470` with `p_keep: 0.6` matches "40% of input tokens randomly replaced."
- FSQ config (4 latent channels, level 5 → vocab 625) matches `configs/gpt/multihead_t2vqpgpt_RWTH.yaml` `codebook_size: 625`.
- Dependencies are fully pinned (`requirements.txt`, 31 packages with exact versions).
- The metric calculators present (`calculate_fid.py`, `calculate_fvd.py`, `calculate_ssim.py`) implement standard, correct FID/FVD/SSIM formulas.

**Open questions for the authors:**
- Were the comparison-table metrics computed by an unreleased evaluation harness? If so, please release it.
- Does the RWTH-T HuggingFace checkpoint bundle include all three pipeline stages plus the two back-translation models?
- Were Tables 10/11 DTW values produced by the training-validation DTW path rather than the released `eval_multihead_t2vqpgpt.py` (which computes token accuracy)?
