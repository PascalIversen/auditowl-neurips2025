# Audit — #2170 Seg4Diff: Unveiling Open-Vocabulary Semantic Segmentation in Text-to-Image Diffusion Transformers

## Summary

The repo (`cvlab-kaist/Seg4Diff`, commit `bb500ca`) is the authors' official
Detectron2-based implementation. It vendors a full copy of HuggingFace
`diffusers` under `code/.../diffusers/` (treated as a dependency, **not** the
contribution, except where the authors patched the SD3/Flux joint-attention
processor to extract I2T attention — that patch *is* the paper's core mechanism
and was audited). The paper's own contribution lives in `seg4diff/` (the three
meta-architectures `Seg4DiffOVSS` / `Seg4DiffUnsup` / `Seg4DiffTrainer`, the
SD3/SD3.5/Flux backbones, the criterion/matcher), `configs/`, `train_net.py`,
and the `eval_*.sh` driver scripts.

I mapped the headline artefacts — Table 1 (open-vocab seg), Table 2
(unsupervised seg), Tables 3–4 (generation), Fig. 8 (per-layer seg), the
layer-9 "semantic grounding expert" analysis, and MAGNET LoRA fine-tuning — to
the released code. The zero-shot extraction (mean over heads of the I2T block of
the attention matrix, Eq. 10–11) is faithfully implemented in the patched
attention processor (`attention_processor.py:1095-1115`). I wrote four
deterministic checks under `_audit_code/` (all run read-only, outputs in
`_audit_code/out/`):

- `check_gt_only_prompt.py` — the open-vocab eval prompt is built from the GT
  classes present in each image, not the full dataset vocabulary.
- `check_hparams.py` — paper-stated LR / effective batch size vs released YAMLs.
- `check_attn_mlp.py` — MAGNET training uses an undisclosed learned conv head.
- `check_readme_lora_download.py` — README's SA-1B weight-download snippet
  points to the COCO repo.

Headline reproducibility verdict: the analysis pipeline and zero-shot extraction
are present and faithful, and pretrained LoRA weights are released. However, the
released **open-vocabulary** evaluation (Table 1) hard-codes an *oracle* prompt
containing only the classes actually present in each image and restricts the
argmax to those classes — an easier setting than the full-vocabulary protocol
used by the baselines it is tabulated against. This is partially acknowledged in
prose but is the dominant interpretability caveat on the headline table. Several
faithfulness gaps (training LR, SA-1B effective batch size, an undisclosed
trainable conv head in MAGNET) and a README download bug round out the findings.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 zero-shot seg (Seg4Diff SD3/SD3.5/Flux mIoU) | `seg4diff/seg4diff_model_ovss.py` + `attention_processor.py:1095-1115` + `eval_ovss.sh` | not re-run (needs GPU + SD3 weights) | code present; **GT-only prompt** caveat | Verified-present / METHODOLOGY |
| Table 1 +MAGNET rows (SA-1B/COCO) | `eval_ovss.sh MODEL.WEIGHTS LORA_PATH_*` + released LoRA on HF | n/a | code present, weights released | Verified-present |
| Table 2 unsupervised seg | `seg4diff/seg4diff_model_unsup.py` (`mask_merge` KL clustering + Hungarian match) | not re-run | code present (DiffSeg/DiffCut mask-proposal protocol) | Verified-present |
| Fig. 8 per-layer seg (peaks @ layer 9) | `configs/eval_ovss.yaml ATTENTION_LAYERS:[9]` + backbone layer cache | not re-run | code present (sweep by passing other layers) | Verified-present |
| Tables 3–4 generation (CLIPScore / T2I-CompBench++) | (none in repo) | — | — | MISSING (no CLIPScore / CompBench eval script) |
| §4.1 "lr = 1e-5" | `configs/train_coco.yaml:62`, `train_sa1b.yaml:62` BASE_LR=1e-4 (×0.1 backbone mult) | BASE_LR=1e-4 | ✗ (paper number ≠ config number) | DIFFERENCE |
| §4.1 "effective batch size 16" | `train_coco.yaml` 8×2=16 ✓; `train_sa1b.yaml` 4×2=8 ✗ | 16 / 8 | ✗ for SA-1B | DIFFERENCE |
| §3.5 MAGNET = "LoRA adapter" + mask loss | `train_*.yaml USE_ATTN_MLP: True` + trainable `AttentionScoreLayer` | — | ✗ (extra trained head undisclosed) | DIFFERENCE |
| §4.1 "10k images from … SA-1B" | `datasets/captions/train_caption_sa1b.json` has 15169 entries | 15169 | ✗ minor (subset selected at train time) | noted, low severity |
| README pretrained-weights download | `README.md:79-85` SA-1B block uses COCO repo_id | bug=1 | ✗ | BUG |

## missing

```yaml finding
id: generation-metric-scripts-missing
category: missing
topic: "result traceability"
title: "No code to compute CLIPScore / T2I-CompBench++ (Tables 3–4)"
severity: medium
confidence: high
status: finding
file: seg4diff/seg4diff_model_train.py
line_start: 677
line_end: 709
quote: |
      @torch.no_grad()
      def generate_image(self, cond_prompts):

          print("Generating unconditional image...")
          for i in range(2):
              image = self.backbone.pipe(
                  prompt="",
                  guidance_scale=7.5,
                  generator=torch.manual_seed(42+i),
                  num_inference_steps=28,
              )
claim: "The only image-generation code in the repo is generate_image(), which renders a handful of qualitative validation images during training; there is no script that generates the benchmark image sets (500/5000/1000 images per §A.5) or computes the CLIPScore and T2I-CompBench++ numbers reported in Tables 3 and 4."
concern: "The generation-quality headline numbers (Tables 3–4, abstract claim that MAGNET 'also enhances generative fidelity') cannot be reproduced from the released code, which contains no CLIPScore or T2I-CompBench++ evaluation."
resolution: "Authors: please release the generation + CLIPScore + T2I-CompBench++ evaluation scripts (and the prompt sets / seeds) used to produce Tables 3 and 4."
cross_refs: []
check_script: null
paper_ref: "Tables 3 and 4; §4.3; §A.5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: readme-sa1b-download-points-to-coco
category: bug
topic: "released artefacts"
title: "README SA-1B LoRA download snippet uses the COCO repo_id"
severity: low
confidence: high
status: finding
file: README.md
line_start: 79
line_end: 85
quote: |
  # Download SA1B-trained lora weights
  LORA_PATH_SA1B = hf_hub_download(
      repo_id="chyun/seg4diff-coco-lora",
      filename="lora_weights.pth",
      cache_dir="/path/to/save/coco",
  )
  print("Downloaded to: ", LORA_PATH_SA1B)
claim: "The 'Download SA1B-trained lora weights' block downloads from repo_id 'chyun/seg4diff-coco-lora' (the COCO repo), not the SA-1B repo; cache_dir is also '/path/to/save/coco'."
concern: "A user following the README to fetch the SA-1B-trained weights silently gets the COCO weights instead, so the Table 1/2 'SA-1B' MAGNET rows cannot be reproduced via the documented path."
resolution: "Fix the SA-1B block to point at the SA-1B HuggingFace repo (e.g. chyun/seg4diff-sa1b-lora, which the README's prose already links)."
cross_refs: []
check_script: _audit_code/check_readme_lora_download.py
paper_ref: "Tables 1 & 2, '+ MAGNET / SA-1B' rows"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: magnet-undisclosed-attn-mlp-head
category: difference
topic: "method faithfulness (paper omission)"
title: "MAGNET training adds a trainable conv head the paper never describes"
severity: medium
confidence: high
status: finding
file: configs/train_coco.yaml
line_start: 10
line_end: 10
quote: |
  USE_ATTN_MLP: True
claim: "Both released training configs set USE_ATTN_MLP: True, which instantiates a trainable 3-layer Conv2d 'AttentionScoreLayer' (seg4diff_model_train.py:113, :856-865) whose output is added as a residual to the I2T attention maps before the mask loss (seg4diff_model_train.py:336-345); its parameters require_grad=True and are optimised alongside the LoRA adapter."
concern: "The paper describes MAGNET as a LoRA adapter trained with a flow-matching + mask loss (§3.5, Fig.10); a separately-trained convolutional refinement head on the attention maps is a result-affecting component that is not disclosed, so the reported MAGNET segmentation gains may be partly attributable to an undescribed learned module rather than to 'strengthening the I2T attention' alone."
resolution: "Authors: describe the AttentionScoreLayer (architecture, that it is trained, how its weights are released) or provide an ablation showing the MAGNET gains hold with USE_ATTN_MLP=False."
cross_refs: []
check_script: _audit_code/check_attn_mlp.py
paper_ref: "§3.5 (MAGNET); Fig. 10"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: train-lr-mismatch
category: difference
topic: "hyperparameters"
title: "Released training LR (1e-4 base) differs from paper's stated lr = 1e-5"
severity: low
confidence: high
status: finding
file: configs/train_coco.yaml
line_start: 58
line_end: 58
quote: |
      BASE_LR: 0.0001
claim: "Both training configs set BASE_LR: 0.0001 (1e-4); with BACKBONE_MULTIPLIER 0.1 the LoRA params see an effective 1e-5 but the base/attn_mlp LR is 1e-4. The paper states a single 'lr = 1e-5'."
concern: "The released base learning rate is 10x the paper's stated value; only after the 0.1 backbone multiplier does the transformer-LoRA LR reach 1e-5, and the un-multiplied modules (e.g. the attn_mlp head) train at 1e-4, so the paper's flat 'lr = 1e-5' does not describe the released optimisation."
resolution: "Authors: clarify which learning rate the paper's '1e-5' refers to and reconcile with BASE_LR=1e-4 × BACKBONE_MULTIPLIER=0.1 in the configs."
cross_refs: ["train-batchsize-sa1b-mismatch"]
check_script: _audit_code/check_hparams.py
paper_ref: "§4.1 Implementation Details ('lr = 1 × 10^-5')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: train-batchsize-sa1b-mismatch
category: difference
topic: "hyperparameters"
title: "SA-1B training config yields effective batch size 8, paper states 16"
severity: low
confidence: high
status: finding
file: configs/train_sa1b.yaml
line_start: 54
line_end: 55
quote: |
    IMS_PER_BATCH: 4
    GRADIENT_ACCUMULATION_STEPS: 2
claim: "train_sa1b.yaml has IMS_PER_BATCH 4 and GRADIENT_ACCUMULATION_STEPS 2 => effective batch size 8; train_coco.yaml has 8×2 = 16. The paper states a single effective batch size of 16 for training."
concern: "The SA-1B MAGNET runs (a Table 1/2 row) use half the effective batch size the paper reports, a faithfulness gap in the disclosed training setup."
resolution: "Authors: confirm the SA-1B effective batch size (8 vs 16) and update the paper or the config to match."
cross_refs: ["train-lr-mismatch"]
check_script: _audit_code/check_hparams.py
paper_ref: "§4.1 ('per-device batch size 4 … effective batch size of 16')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ovss-gt-only-prompt-oracle
category: methodology
topic: "evaluation validity / open-vocabulary protocol"
title: "Open-vocab seg eval feeds only the image's GT classes as the prompt (oracle)"
severity: high
confidence: high
status: finding
file: seg4diff/seg4diff_model_ovss.py
line_start: 149
line_end: 162
quote: |
          idxs = self.get_gt_indices(batched_inputs)
          
          prompt = None
          num_classes = len(self.test_class_texts)        

          assert not self.training

          selected_idxs = idxs
          # if "background" in self.test_class_texts:
          #     selected_idxs = [i for i in selected_idxs if self.test_class_texts[i] != "background"]
          
          # Ensure indices are Python ints for list indexing
          classnames= [self.test_class_texts[int(i)].split("-")[0] for i in selected_idxs]
          prompt = " ".join(classnames)
claim: "For each image the eval reads the ground-truth semantic map (get_gt_indices -> bincount of present class IDs), sets selected_idxs to exactly those present classes, builds the text prompt from only those classnames, and writes predictions into `outputs[:, selected_idxs]` (line 229-230) so the per-pixel argmax can only choose among classes truly present in the image."
concern: "This is an oracle class set: the standard open-vocabulary protocol scores every pixel against the full dataset vocabulary (the baselines in Table 1 — ProxyCLIP, CorrCLIP, DiffSegmenter, iSeg — do so), whereas Seg4Diff is told which classes are in each image and never has to reject absent classes, making the Table 1 mIoU comparison not apples-to-apples and likely inflating the proposed method relative to the tabulated baselines."
resolution: "Authors: report Seg4Diff open-vocab mIoU with the full per-dataset vocabulary in the prompt and argmax (the GT_ONLY_PROMPT=False path), or re-tabulate baselines under the same GT-restricted setting; clarify which setting Table 1's baseline numbers were obtained under."
cross_refs: []
check_script: _audit_code/check_gt_only_prompt.py
paper_ref: "Table 1; §4.2 ('CLIP-based methods process entire classnames … not identical to our evaluation setting')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 1 | medium | No CLIPScore / T2I-CompBench++ code for Tables 3–4. |
| bug | 1 | low | README SA-1B weight download points to the COCO repo. |
| difference | 3 | medium | Undisclosed trainable conv head in MAGNET; LR and SA-1B batch-size config↔paper gaps. |
| methodology | 1 | high | Open-vocab eval uses an oracle (GT-only) class set, unlike the tabulated baselines. |

## Top take-aways

1. **[methodology, high/high]** Table 1's "open-vocabulary" seg uses an oracle:
   the prompt and argmax are restricted to the GT classes present in each image
   (`ovss-gt-only-prompt-oracle`), an easier setting than the full-vocabulary
   protocol of the baselines it is compared against. Partially acknowledged in
   §4.2 prose but it materially affects the headline comparison.
2. **[difference, medium/high]** MAGNET training silently includes a separately
   trained 3-layer conv head (`AttentionScoreLayer`, `USE_ATTN_MLP: True`) the
   paper never describes (`magnet-undisclosed-attn-mlp-head`).
3. **[missing, medium/high]** No script computes the CLIPScore / T2I-CompBench++
   generation numbers in Tables 3–4 (`generation-metric-scripts-missing`).
4. **[difference, low/high]** Training LR in the configs (BASE_LR 1e-4) does not
   match the paper's stated `lr = 1e-5` (`train-lr-mismatch`).
5. **[difference, low/high]** SA-1B training config gives effective batch size 8,
   not the paper's stated 16 (`train-batchsize-sa1b-mismatch`).
6. **[bug, low/high]** README's SA-1B LoRA-download snippet downloads the COCO
   weights (`readme-sa1b-download-points-to-coco`).

## Items that genuinely look fine

- The core mechanism (mean-over-heads of the I2T sub-block of the attention
  matrix, Eq. 10–11) is faithfully implemented in the patched joint-attention
  processor (`attention_processor.py:1095-1115`, slice `[:, :, :4096, 4096:4096+77]`).
- The headline zero-shot rows (Seg4Diff SD3 / SD3.5 / Flux, Table 1) run with
  `NORM_BEFORE_MERGE: False` / `NORM_AFTER_MERGE: False`, consistent with the
  paper's "without any refinement or postprocessing" claim (`eval_ovss.yaml:19-20`).
- Pretrained SA-1B and COCO LoRA weights are released on HuggingFace and wired
  into `eval_ovss.sh` via `MODEL.WEIGHTS`.
- The 10k COCO caption subset is shipped in-repo (`train_caption_coco.json`,
  10001 entries); SD3/SD3.5/Flux backbones download automatically.
- The unsupervised eval (Table 2) uses null/`<pad>` prompts + KL-based
  mask-proposal merging + Hungarian matching to GT — the standard
  DiffSeg/DiffCut mask-proposal protocol (matching at eval is part of the
  protocol, not leakage); disclosed in §A.4.
- The timestep `t = 8 of 28` and rectified-flow noise interpolation are wired
  through `scale_noise` consistently in inference (`sd3_backbone.py:319-332`).

## Open questions for the authors

- Under which prompt setting (full vocabulary vs GT-restricted) were the Table 1
  *baseline* numbers obtained? If the baselines use the full vocabulary and
  Seg4Diff uses the GT-restricted set, the comparison is not controlled.
- Are the trained `AttentionScoreLayer` weights included in the released
  `lora_weights.pth`, and are they loaded at MAGNET evaluation time (the OVSS
  eval config defaults `USE_ATTN_MLP` to False)?
- The SA-1B caption file holds 15169 entries while §4.1 says "10k images from
  SA-1B"; how is the 10k subset selected at train time (low severity)?
