# Audit — #2021 "RESAnything: Attribute Prompting for Arbitrary Referring Segmentation"

## Summary

RESAnything is a zero-shot, training-free referring-expression-segmentation (RES)
method built by chaining an MLLM (Chain-of-Thought "attribute prompting") with
SAM mask proposals and CLIP/MLLM-based mask selection. The paper's headline
empirical claims are quantitative benchmark numbers (Tables 1–5: RefCOCO/+/g,
ReasonSeg, COCO-Tasks, the authors' new ABO-Image-ARES dataset) and a new ~3K-
instance benchmark dataset (ABO-Image-ARES) presented as a core contribution.

The author code was recovered manually from the project page
(`github.com/suikei-wang/RESAnything`) and cloned under
`code/suikei-wang__RESAnything/`. I read every source file
(`demo.py`, `demo_batch.py`, `generation.py`, `similarity.py`, `sam.py`,
`sam_utils.py`, `prompts.py`, `config.py`, `config.yaml`, `prompts.yaml`,
`README.md`) read-only, mapped the paper's pipeline (SAM proposals → reference/
candidate attribute text generation → text-to-text + text-to-image MLLM/CLIP
selection, Algorithm 1) onto the code, and built two deterministic checks under
`_audit_code/`:
`check_repo_inventory.py` (lists all 8 `.py` files; greps for any eval/metric/
benchmark/IoU/dataset-loader code; extracts the README disclaimer verbatim) and
`check_backbone_and_prompts.py` (resolves the MLLM backbone the code actually
loads vs the paper's default, and searches for any bounding-box visual-prompt
generation).

Headline reproducibility verdict: **source-present but bottom-tier**. The
released repo is, by its own README, a *re-implementation* whose original code is
withheld under a "protected license" and whose prompts "may not be the original
version" — so the released artifact is not the artifact that produced the paper's
numbers, and (this being an LLM-prompting method) the prompts *are* the method.
On top of that provenance gap, the repo contains **no evaluation/benchmark
harness at all** (no IoU computation, no dataset loaders, no script that
reproduces any number in any table), the **ABO-Image-ARES dataset** (a stated
core contribution) is **not released** ("ASAP"), the **default MLLM backbone in
the code (Qwen2.5-VL-7B) is neither the paper's default (Pixtral-12B) nor the
listed ablation (Qwen2-VL)**, and the **bounding-box visual prompt** that the
paper says is part of its chosen two-prompt configuration is **not implemented**.
The NeurIPS checklist Q5 (open code/data) is answered `[NA]`, whose official
meaning is "no experiments requiring code" — false for this paper.

Every quantitative artefact in the paper therefore traces to **(none)** in the
repo. The repo is a runnable single-image/batch *demo* of the method, not a
reproduction package.

## Traceability table (Rule G)

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 RefCOCO/+/g cIoU/gIoU (e.g. RESAnything 68.5 RefCOCO val) | (none) | — | — | MISSING (no eval harness) |
| Table 2 ReasonSeg 74.6 gIoU / 72.5 cIoU | (none) | — | — | MISSING |
| Table 2 ABO-Image-ARES 78.2 gIoU / 72.4 cIoU | (none) — and dataset itself absent | — | — | MISSING |
| Table 3 COCO-Tasks mIoU@0.5 (14-task avg) | (none) | — | — | MISSING |
| Table 4 prompting ablation (gIoU/cIoU, ReasonSeg) | (none) | — | — | MISSING |
| Table 5 visual-prompt ablation (mask+bbox = 72.2 gIoU, RefCOCO testA) | (none); bbox prompt not implemented | — | — | MISSING |
| Table 5 MLLM-backbone ablation (Pixtral 74.6 / Claude 76.2 / Qwen2-VL 74.2) | (none); code loads Qwen2.5-VL-7B | — | — | MISSING |
| Supp. mask-selector / CLIP-as-RNN / RefCOCOm / g-RefCOCO tables | (none) | — | — | MISSING |
| Abstract: "~3K curated RES instances" benchmark (ABO-Image-ARES) | (none) — "release ASAP" | — | — | MISSING |
| Default MLLM = Pixtral 12B (paper §4) | config.yaml:2 loads Qwen2.5-VL-7B | n/a | ✗ | MISMATCH |
| Default config uses bbox + mask-cropped visual prompts (paper §3.1) | generation.py:53 / similarity.py:65 use cropped only | n/a | ✗ | MISMATCH |
| CLIP combined-score threshold "set to 1 for all experiments" (paper §3.2) | similarity.py:190-222 selection has no such threshold | n/a | ✗ | MISMATCH (low) |

Routing note: every MISSING row above is owned by a single `missing` finding
(`no-eval-or-benchmark-harness`) for the eval code, plus a dedicated
`abo-ares-dataset-missing` for the dataset; the two MISMATCH backbone/bbox rows
are `difference` findings. They are not re-filed per row (Single-Owner Rule).

## missing

```yaml finding
id: reimplementation-original-withheld
category: missing
topic: "repository provenance / result traceability"
title: "Released repo is a re-implementation; original result-producing code withheld under protected license"
severity: high
confidence: high
status: finding
file: code/suikei-wang__RESAnything/README.md
line_start: 13
line_end: 13
quote: |
    <i>This repository contains a re-implementation of the codebase. The initial version is subject to a protected license that restricts redistribution. The prompts provided in this repository may not be the original version, and optimal performance may require further iterative refinement and tuning.</i>
claim: "The README explicitly states the public repo is a re-implementation, that the original codebase is withheld under a protected license, and that the released prompts may differ from the originals and may need further tuning to reach the reported performance."
concern: "The released artifact is not the artifact that generated the paper's reported numbers; for an LLM-prompting method the prompts ARE the method, so withholding/altering them breaks the traceability from reported results to runnable code and the README itself disclaims that the reported performance is reproducible."
resolution: "Authors: release the exact prompts and code (or a versioned snapshot) used to produce Tables 1–5, or state explicitly which reported numbers are reproducible from this repo and which are not."
cross_refs: ["no-eval-or-benchmark-harness", "checklist-q5-misanswered"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "README.md (released repo), and paper §4 Experiment"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-eval-or-benchmark-harness
category: missing
topic: "result traceability / expected code completeness"
title: "No evaluation/benchmark harness: no code computes any reported IoU number"
severity: high
confidence: high
status: finding
file: code/suikei-wang__RESAnything/README.md
line_start: 118
line_end: 131
quote: |
    RESAnything/
    ├── config.py              # Configuration loader
    ├── config.yaml           # Main configuration file
    ├── demo.py               # Single image processing
    ├── demo_batch.py         # Batch processing
    ├── generation.py         # Text generation utilities
    ├── prompts.py           # Prompt management
    ├── prompts.yaml         # Prompt templates
    ├── sam_utils.py         # SAM processing utilities
    ├── similarity.py        # Similarity computation
    ├── requirements.txt     # Python dependencies
    ├── sam_vit_h_4b8939.pth # SAM model checkpoint
    └── Qwen2.5-VL-7B-Instruct/ # Qwen model directory
claim: "The repo's own File Structure listing enumerates only configuration, single-image/batch demo, generation, similarity, and SAM-utility files — no evaluation/benchmark module, no dataset loader (RefCOCO/ReasonSeg/COCO-Tasks/ABO-ARES), no ground-truth handling, and no gIoU/cIoU/mIoU computation. The deterministic grep (check_repo_inventory.py) confirms the only 'iou' hits across all 8 .py files are SAM's pred_iou_thresh/stability hyperparameters; nothing reproduces any number in Tables 1–5."
concern: "Every quantitative claim in the paper (all benchmark tables and headline numbers) is untraceable to runnable code, so none of the reported results can be reproduced from this repo."
resolution: "Authors: release the evaluation scripts (dataset loaders, gIoU/cIoU/mIoU computation, per-benchmark drivers) that produced Tables 1–5, with exact commands."
cross_refs: ["reimplementation-original-withheld", "abo-ares-dataset-missing"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Tables 1–5; §4 Experiment"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: abo-ares-dataset-missing
category: missing
topic: "dataset availability"
title: "ABO-Image-ARES benchmark (core contribution) is not released"
severity: high
confidence: high
status: finding
file: code/suikei-wang__RESAnything/README.md
line_start: 133
line_end: 134
quote: |
    ## 📊 ABO-ARES Dataset
    We will try to release ABO-ARES dataset ASAP.
claim: "The README states the ABO-Image-ARES dataset is not yet released ('try to release ... ASAP'); no dataset files, build script, or accession for it exist in the repo."
concern: "ABO-Image-ARES (~3K curated RES instances) is listed as a core contribution and underpins Table 2 (right) and several supplementary tables; without it those results cannot be reproduced and the dataset contribution cannot be inspected."
resolution: "Authors: release ABO-Image-ARES (the 2,482 images / 2,989 expression-segment pairs) with annotations and the extraction/annotation pipeline, or provide a resolvable accession."
cross_refs: ["no-eval-or-benchmark-harness"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "§4 ABO-Image-ARES benchmark; contributions list (intro); Table 2 right"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No technical bugs were found that block the demo's intent. The single-image and
batch inference paths import consistently, the SAM/MLLM/CLIP wiring is internally
coherent, and `selection()` implements the Algorithm-1 cascade. (N/A is not
claimed — this is a genuine finding-free pass for this category. Note: a hardware
dependency on CUDA/`flash_attention_2` is a legitimate environment constraint,
not a bug.)

## difference

```yaml finding
id: backbone-mismatch-qwen-vs-pixtral
category: difference
topic: "evaluation consistency (model backbone)"
title: "Code default MLLM is Qwen2.5-VL-7B; paper's default is Pixtral-12B"
severity: medium
confidence: high
status: finding
file: code/suikei-wang__RESAnything/config.yaml
line_start: 1
line_end: 4
quote: |
    model:
      name: "Qwen/Qwen2.5-VL-7B-Instruct"
      dtype: "bfloat16"
      attn_implementation: "flash_attention_2"
claim: "The released pipeline hardcodes Qwen2.5-VL-7B-Instruct as the MLLM backbone (config.yaml; demo.py:52 / demo_batch.py:129 instantiate Qwen2_5_VLForConditionalGeneration), whereas the paper states 'We use Pixtral 12B as the MLLM' as the default and reports Pixtral-12B in Table 5; the listed Qwen ablation is 'Qwen2-VL' (not Qwen2.5-VL)."
concern: "The released code runs a different (and different-generation) model from the one that produced the headline numbers, so even with prompts fixed the repo cannot reproduce the reported Pixtral-12B results; the substituted model is not even the Qwen2-VL ablation row."
resolution: "Authors: confirm whether the reported numbers used Pixtral-12B (as stated) and provide a config/loader for it, and clarify the Qwen2-VL vs Qwen2.5-VL discrepancy."
cross_refs: ["reimplementation-original-withheld"]
check_script: _audit_code/check_backbone_and_prompts.py
paper_ref: "paper §4 (line 621) 'We use Pixtral 12B'; Table 5 (right)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: bbox-visual-prompt-not-implemented
category: difference
topic: "evaluation consistency (visual prompts)"
title: "Paper's chosen bbox+mask-cropped prompt config: bbox prompt absent from code"
severity: medium
confidence: high
status: finding
file: code/suikei-wang__RESAnything/generation.py
line_start: 53
line_end: 55
quote: |
    mask_dir = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + '-SAM', 'cropped')
    mask_images = [f for f in os.listdir(mask_dir) if f.endswith('.jpg') or f.endswith('.png')]
    mask_images_path = [os.path.join(mask_dir, mask_image) for mask_image in mask_images]
claim: "Candidate-text generation and the text-to-image comparison (similarity.py:65) consume only the 'cropped' mask images; sam_utils.py saves only cropped/overlay/binary-txt outputs, and no source file generates a bounding-box visual prompt. The only 'bounding box' string in the repo is a prohibition inside the candidate prompt (prompts.yaml:28)."
concern: "The paper states its chosen/default configuration uses two visual prompts — bounding box (V^b) and mask cropped (V^m) — and Table 5 reports that mask+bbox is the best config (72.2 gIoU); the released method implements only the mask-cropped half, so it does not match the configuration that produced the reported best numbers."
resolution: "Authors: release the bounding-box visual-prompt generation and the multi-prompt candidate/comparison logic, or clarify that the reported numbers used mask-cropped only."
cross_refs: ["reimplementation-original-withheld"]
check_script: _audit_code/check_backbone_and_prompts.py
paper_ref: "paper §3.1 (lines 366-367); Table 5 (left), line 855"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: clip-threshold-not-in-selection
category: difference
topic: "evaluation consistency (selection logic)"
title: "Paper's CLIP combined-score threshold (=1) absent from selection fallback"
severity: low
confidence: medium
status: finding
file: code/suikei-wang__RESAnything/similarity.py
line_start: 217
line_end: 222
quote: |
    max_clip_score = max(info.get('avg_clip', -float('inf'))
                         for info in json_data.values())
    clip_best = [mask_id for mask_id, info in json_data.items()
                 if info.get('avg_clip', -float('inf')) == max_clip_score]
    if clip_best:
        return clip_best
    return []
claim: "When no candidate gets a positive MLLM yes/no, selection() returns the mask with the highest avg_clip unconditionally, with no threshold check; the paper (lines 556-557) says it checks whether the combined CLIP score exceeds a threshold 'set to 1 for all experiments' and otherwise returns the reference-text explanation (i.e. no mask)."
concern: "The released selection always emits a mask, whereas the paper's procedure can abstain (return T_ref) when no candidate clears the CLIP threshold, which would change the 'target invisible/irrelevant' cases that the paper specifically handles."
resolution: "Authors: confirm the combined-CLIP threshold and abstention behaviour used for the reported numbers, and reconcile with the released selection()."
cross_refs: ["reimplementation-original-withheld"]
paper_ref: "paper §3.2 (lines 554-559); Algorithm 1 lines 17-23"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: checklist-q5-misanswered
category: methodology
topic: "reporting / reproducibility checklist"
title: "NeurIPS checklist Q5 (open code/data) answered [NA] though paper runs experiments"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
    5. Open access to data and code
    Answer: [NA]
    Justification: Code and data are not included in the submission.
    Guidelines:
    • The answer NA means that paper does not include experiments requiring code.
claim: "The reproducibility checklist answers Q5 (open access to data and code) as [NA], whose official definition printed directly beneath it is 'the answer NA means that paper does not include experiments requiring code' — but RESAnything is an empirical paper whose claims rest entirely on code-driven experiments (Tables 1–5)."
concern: "The [NA] answer mischaracterises the submission's reproducibility status: the correct answer is 'No' (code/data withheld), and [NA] obscures from reviewers that no code or data was provided for an experiment-heavy paper."
resolution: "Authors: correct the checklist to 'No' (or release the code/data), since the paper plainly includes experiments requiring code."
cross_refs: ["reimplementation-original-withheld", "no-eval-or-benchmark-harness"]
paper_ref: "NeurIPS Paper Checklist, item 5 (Open access to data and code)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | Re-implementation w/ withheld original; no eval harness; ABO-ARES dataset unreleased |
| bug         | 0          | -            | Demo paths run coherently; no intent-contradicting defect found |
| difference  | 3          | medium       | Backbone Qwen2.5-VL vs paper's Pixtral-12B; bbox visual prompt absent; CLIP threshold absent |
| methodology | 1          | medium       | Checklist Q5 answered [NA] for an experiment-driven paper |

## Top take-aways (≤6, ranked by severity × confidence)

1. **[missing] Released repo is a re-implementation; the original result-producing
   code and the original prompts are withheld** (`reimplementation-original-withheld`,
   high/high). For an LLM-prompting method the prompts are the method, and the
   README itself disclaims that reported performance is reproducible.
2. **[missing] No evaluation/benchmark harness exists** — no IoU computation, no
   dataset loaders, no driver reproduces any number in Tables 1–5
   (`no-eval-or-benchmark-harness`, high/high).
3. **[missing] ABO-Image-ARES dataset (a core contribution) is not released**
   ("ASAP") (`abo-ares-dataset-missing`, high/high).
4. **[methodology] Checklist Q5 mis-answered `[NA]`** for an experiment-driven
   paper; correct answer is "No" (`checklist-q5-misanswered`, medium/high).
5. **[difference] Code default MLLM is Qwen2.5-VL-7B, not the paper's default
   Pixtral-12B** (nor the Qwen2-VL ablation) (`backbone-mismatch-qwen-vs-pixtral`,
   medium/high).
6. **[difference] The bounding-box visual prompt — part of the paper's chosen
   best config — is not implemented** (`bbox-visual-prompt-not-implemented`,
   medium/high).

## Items that genuinely look fine

- The single-image (`demo.py`) and batch (`demo_batch.py`) inference pipelines
  import consistently and wire SAM → MLLM (Qwen2.5-VL) → CLIP → selection in the
  order the paper's Fig. 2 describes.
- `selection()` (similarity.py:190-222) faithfully implements the Algorithm-1
  cascade priority: both-positive → text-to-text → text-to-image → CLIP
  fallback (modulo the missing abstention threshold, filed as
  `clip-threshold-not-in-selection`).
- SAM hyperparameters in `config.yaml` (points_per_side, pred_iou_thresh=0.92,
  stability=0.92, crop layers) are consistent between `config.yaml`, `sam.py`,
  and `sam_utils.py`; CUDA / flash-attention dependence is a legitimate hardware
  constraint, not a defect.
- `requirements.txt` is a fully pinned conda export (dependencies are
  reproducible at the environment level).

## Open questions for the authors

- Which reported numbers (if any) in Tables 1–5 are reproducible from THIS
  re-implementation with THESE prompts, given the README's disclaimer that
  "optimal performance may require further iterative refinement and tuning"?
- Did the reported numbers use Pixtral-12B (as stated in §4) — and if so, will the
  Pixtral configuration and the bounding-box visual-prompt code be released?
- What is the combined-CLIP abstention threshold actually used for the reported
  results, and does the abstention ("return T_ref") path affect any table?
