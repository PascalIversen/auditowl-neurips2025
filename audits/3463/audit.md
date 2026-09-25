# Code audit — Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations (NeurIPS 2025, #3463)

## 1. Summary

The paper ("DiTF") proposes a *training-free* recipe for extracting semantic-correspondence
features from Diffusion Transformers (Flux, SD3, SD3-5, Pixart-α). Two components are
claimed as contributions: (a) **channel-wise modulation with AdaLN** — extract the
pre-AdaLN feature `z_k` and re-apply `ẑ_k = (1+γ)·LayerNorm(z_k) + β` (Eq. 8–9), and
(b) a **channel-discard** strategy that zeroes the channels still holding weak massive
activations. Headline results: +9.4% over DIFT on SPair-71k, SOTA on AP-10K, plus
PF-Pascal, ADE20K segmentation, HPatches geometric and a temporal task. The strongest
("†") numbers integrate DINOv2 features with PCA fusion (Eq. 14).

The submission ships **no GitHub repo**; the authors' code is `code/openreview_supplementary.zip`.
I extracted it (read-only) to `_audit_code/extracted/` (a nested `code.zip` inside a
`Supplementary Material/` folder). The released code base is a single SPair-71k evaluation
entry point: `eval_spair.py` + `test_spair.sh`, plus a fork of the public Flux repo under
`src/flux/` (`model.py`, `modules/layers.py`, `feat_flux.py`, `util.py`, …). I read every
`.py`, traced feature/AdaLN shapes, and ran three deterministic checks under `_audit_code/`
(`check_chunk_crash.py`, `check_missing_pieces.py`; outputs in `_audit_code/out/`). Model
weights load from HuggingFace (`black-forest-labs/FLUX.1-dev`) and resolve.

Path note: code paths below are relative to the extracted code root
`_audit_code/extracted/Supplementary Material/code_extracted/code/`.

Bottom line: only the Flux semantic-correspondence path on SPair-71k is wired up, and even
that **crashes under the exact command the authors ship** (`--ensemble_size 1` makes a
`chunk(2)` over a size-1 batch fail). The **channel-discard** contribution is **absent**
from the code, the **DINOv2 + PCA fusion** that produces every "†" SOTA row is **absent**
(the only PCA helper is dead code), and the ablation toggles (Original / +AdaLN / +discard;
condition t vs c) have no implementation. Evaluation scripts for PF-Pascal, AP-10K,
ADE20K, HPatches and the temporal task are not present.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 / Table 3, `DiTFflux` SPair-71k PCK@0.10 = 67.1 (per-point) | `eval_spair.py` (per-point print L279/L287) | not run (crashes, see `chunk2-cfg-crash`); also img_size 768 in shell vs 960 in paper App. Table 7 | — | BLOCKED / config mismatch |
| Table 2/3 `DiTFsd3-5`, `DiTFSD3`, `DiTFpixart-α` rows | (none — `eval_spair.py` only accepts `--dift_model flux`, L100-103) | — | — | MISSING |
| Table 2/3 **all "†" rows** (DiTF + DINOv2, the SOTA numbers, e.g. 72.2 SPair, 69.4 AP-10K-C.S.) | (none — no DINOv2 extraction; PCA helper `pca_feature_pair` defined but never called) | — | — | MISSING |
| Table 3 / Fig. 6 AP-10K (I.S./C.S./C.F.) results | (none — no AP-10K eval; `feat_utils.get_dataset_info` mentions `ap10k` but is unused) | — | — | MISSING |
| Table 3 PF-Pascal results | (none) | — | — | MISSING |
| Table 4 ablation: Original / +AdaLN / +Channel discard | (none — AdaLN applied unconditionally L187-207; no discard code; no toggle flags) | — | — | MISSING |
| Table 5 condition ablation (original / c / t / t&c) | (none — condition is fixed to `t & c` via `prepare_txt`; no flag) | — | — | MISSING |
| Table 6 ADE20K segmentation (DiTFflux 54.8 mIoU) | (none — no segmentation head / training code) | — | — | MISSING |
| Table 8 HPatches geometric (cv2.findHomography) | (none) | — | — | MISSING |
| Temporal correspondence task (App. mentions DAVIS) | (none) | — | — | MISSING |
| Fig. 2–5 massive-activation / AdaLN visualisations | (none — no activation-magnitude plotting script) | — | — | MISSING |

Every quantitative artefact is either MISSING or BLOCKED; none could be recomputed from the
released code.

## 3. Findings

## missing

```yaml finding
id: channel-discard-missing
category: missing
topic: "core method / channel discard"
title: "Channel-discard strategy (a headline contribution) is absent from the code"
severity: high
confidence: high
status: finding
file: eval_spair.py
line_start: 180
line_end: 190
quote: |
                src_ft = rearrange(src_ft_raw, "b c h w -> b (h w) c")
                src_ft = pre_norm(src_ft)
                src_ft = rearrange(src_ft, "b (h w) c -> b c h w", h=H, w=W)
                
                
                # src_shift_raw, src_scale_raw = src_ada[1][0].unsqueeze(0).unsqueeze(2).unsqueeze(3), src_ada[1][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                
                src_shift = src_ada[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                src_scale = src_ada[0][1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                
                src_ft = (1 + src_scale) * src_ft + src_shift
claim: "The eval applies AdaLN modulation but performs no channel-discard (no dimension is ever zeroed); grep over the whole repo finds 0 occurrences of 'discard' and no channel-zeroing in the feature path."
concern: "Channel discard is listed as one of the paper's contributions and is the Table-4 ablation row that adds +1.8 PCK@0.10 on SPair (65.3 to 67.1), so the released code cannot reproduce the reported DiTFflux 67.1 number nor the ablation."
resolution: "Provide the channel-discard implementation: which dimensions are zeroed, how they are selected (fixed indices vs per-image), and at which point in eval_spair.py it runs."
cross_refs: ["ablation-toggles-missing"]
check_script: _audit_code/check_missing_pieces.py
paper_ref: "Sec 4.3 'Channel discard'; Table 4 row '+ Channel discard'"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dinov2-pca-fusion-missing
category: missing
topic: "result traceability / SOTA rows"
title: "DINOv2 integration + PCA fusion absent; every '†' SOTA row not reproducible"
severity: high
confidence: high
status: finding
file: eval_spair.py
line_start: 32
line_end: 61
quote: |
  def pca_feature_pair(feat1, feat2, q=1024):
      
      
      B, C, H, W = feat1.shape
      # data = data.float().permute(0,2,3,1).squeeze(0).reshape((-1, C))
      feat1 = rearrange(feat1, "b c h w -> b (h w) c")
      feat1 = feat1.float().squeeze(0)
      
      feat2 = rearrange(feat2, "b c h w -> b (h w) c")
      feat2 = feat2.float().squeeze(0)
      
      cat_desc_dino = torch.cat((feat1, feat2), dim=0) # (1, 1, num_patches**2, dim)
      mean = torch.mean(cat_desc_dino, dim=0, keepdim=True)
      centered_features = cat_desc_dino - mean
      U, S, V = torch.pca_lowrank(centered_features, q=q)
      reduced_features = torch.matmul(centered_features, V[:, :q]) # (t_x+t_y)x(d)
      processed_co_features = reduced_features
      feat1 = processed_co_features[:H*W, :]
      feat2 = processed_co_features[H*W:, :]
      
      
      feat1 = rearrange(feat1, "(h w) c -> c h w", h=H, w=W)
      
      feat1 = feat1.unsqueeze(0)
      
      feat2 = rearrange(feat2, "(h w) c -> c h w", h=H, w=W)
      
      feat2 = feat2.unsqueeze(0)
      
      return feat1, feat2
claim: "There is no DINOv2 feature extraction or concatenation anywhere in the released code (0 DINOv2 identifiers), and the only PCA helper, pca_feature_pair, is defined with q=1024 but never called (0 call sites)."
concern: "Every '†' (DiTF+DINOv2) row in Tables 2-3 — which are the paper's state-of-the-art numbers (e.g. SPair 72.2, AP-10K-C.S. 69.4) — depends on DINOv2 concatenation and the Eq.14 PCA fusion (paper says output dim 1280, code's dead helper uses q=1024); none of this is implemented."
resolution: "Release the DINOv2 extraction (11th-layer token facet), the concatenation, and a PCA-fusion call site at output dim 1280; clarify the 1024-vs-1280 discrepancy."
cross_refs: ["channel-discard-missing"]
check_script: _audit_code/check_missing_pieces.py
paper_ref: "Tables 2-3 '†' rows; Appendix Eq. 14 (PCA, dim 1280)"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: nonflux-and-otherdataset-evals-missing
category: missing
topic: "experimental protocol coverage"
title: "Only Flux-on-SPair eval shipped; SD3/SD3-5/Pixart and all other benchmarks absent"
severity: high
confidence: high
status: finding
file: eval_spair.py
line_start: 100
line_end: 103
quote: |
    if args.dift_model == 'flux':
        dift = Featurizer4Eval(cat_list=all_cats, ensemble_size=args.ensemble_size)
    else:
        raise Exception("model must be in [flux] ")
claim: "eval_spair.py only supports --dift_model flux and only the SPair-71k benchmark; there is no script for SD3/SD3-5/Pixart-alpha features nor for PF-Pascal, AP-10K, ADE20K segmentation, HPatches geometric, or the temporal task."
concern: "The DiTFsd3-5 / DiTFSD3 / DiTFpixart rows (Tables 2-3, Fig. 6), all AP-10K / PF-Pascal / ADE20K / HPatches / temporal results, and the per-model ablations have no code that produces them, so most reported tables are unreproducible from this submission."
resolution: "Provide the feature extractors for SD3, SD3-5, Pixart-alpha and the evaluation harnesses for each non-SPair benchmark."
cross_refs: []
check_script: _audit_code/check_missing_pieces.py
paper_ref: "Tables 2-6, 8; Figs 6, 12"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-toggles-missing
category: missing
topic: "ablations"
title: "No toggle for Original / +AdaLN / +discard or condition (t,c) ablations"
severity: medium
confidence: high
status: finding
file: eval_spair.py
line_start: 310
line_end: 322
quote: |
    parser.add_argument('--dataset_path', type=str, default='./dataset/SPair-71k', help='path to spair dataset')
    parser.add_argument('--dataset', type=str, default='SPair', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='./spair_ft/', help='path to save features')
    parser.add_argument('--dift_model', choices=['flux'], default='flux', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[512, 512],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', default=261, type=int, help='t for diffusion')
    parser.add_argument('--ft_index', nargs='+', type=int, default=[12, 14], help='which upsampling block to extract the ft map') ###调参[0,57]
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    parser.add_argument('--guidance_scale', default=1, type=float, help='ensemble size for getting an image ft map')
claim: "The CLI exposes no flag to disable AdaLN, disable channel discard, or vary the AdaLN condition (t only / c only / t&c); AdaLN is applied unconditionally and the text prompt 'a photo of a {cat}' is always built, so the Table-4 (Original/+AdaLN/+discard) and Table-5 (condition) ablations cannot be produced without code changes the authors did not ship."
concern: "Ablation rows central to the paper's central claim (AdaLN alone gives +20% absolute) have no reproducible path in the code."
resolution: "Add flags to run the Original (no AdaLN) and condition variants, or release the scripts used for Tables 4-5."
cross_refs: ["channel-discard-missing"]
paper_ref: "Tables 4, 5"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-readme-no-deps
category: missing
topic: "expected code completeness / environment"
title: "No README, no dependency/environment specification for the eval scripts"
severity: medium
confidence: high
status: finding
file: eval_spair.py
line_start: 13
line_end: 29
quote: |
  import cv2
  import matplotlib.pyplot as plt
  import gc
  import copy
  from sklearn.decomposition import PCA
  from torchvision import transforms as T
  from math import sqrt
  import seaborn as sns
  import base64
  from io import BytesIO
  
  import warnings
  
  warnings.filterwarnings('ignore')
  
  import numpy as np
  from scipy.spatial.distance import cosine
claim: "The repo has no README and no environment/requirements file for the audited scripts; eval_spair.py imports cv2, sklearn, scipy, seaborn, matplotlib, diffusers, einops, etc., none of which are pinned. The only requires.txt present (src/flux.egg-info/requires.txt) is the upstream Flux package list and omits these."
concern: "Without a dependency spec or run instructions beyond a single hard-wired shell line, the environment that produced the numbers cannot be rebuilt; the NeurIPS checklist claims 'exact command and environment' are provided."
resolution: "Add a README with reproduce commands and a pinned environment (requirements.txt/conda) covering cv2, scikit-learn, scipy, seaborn, diffusers versions."
cross_refs: []
paper_ref: "NeurIPS checklist Q5 (open access to code, 'exact command and environment')"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: chunk2-cfg-crash
category: bug
topic: "evaluation entry point"
title: "Shipped command crashes: chunk(2) over size-1 batch when --ensemble_size 1"
severity: high
confidence: high
status: finding
file: eval_spair.py
line_start: 170
line_end: 177
quote: |
                src_ft_raw = output_dict[data['src_imname']].cuda()
                B,C,H,W = src_ft_raw.shape
                src_ada = ada_dict[data['src_imname']].cuda()
                
                
                feat_pred_uncond, feat_pred_text = src_ft_raw.chunk(2)
                # src_ft_raw = feat_pred_uncond + guidance_scale * (feat_pred_text - feat_pred_uncond)
                src_ft_raw = feat_pred_text
claim: "The saved feature tensor has batch dim equal to ensemble_size (Featurizer4Eval.forward, feat_flux.py:149, no mean over the ensemble). test_spair.sh sets --ensemble_size 1, so src_ft_raw is [1, C, H, W]; src_ft_raw.chunk(2) over dim 0 returns ONE chunk, and 'feat_pred_uncond, feat_pred_text = src_ft_raw.chunk(2)' raises ValueError (verified in _audit_code/check_chunk_crash.py)."
concern: "Running the exact provided command (test_spair.sh, --ensemble_size 1) crashes before any PCK is computed, so the released code does not reproduce its own headline number as shipped."
resolution: "Either remove the chunk(2)/CFG split (it is vestigial — the guidance line is commented out) or document that ensemble_size must be an even number >= 2; clarify what the two halves are meant to represent given the forward does no uncond/cond batch construction."
cross_refs: ["ensemble-not-averaged"]
check_script: _audit_code/check_chunk_crash.py
paper_ref: "test_spair.sh (--ensemble_size 1)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ensemble-not-averaged
category: bug
topic: "feature ensembling"
title: "Ensemble dimension never averaged; treated as CFG halves then only index 0 used"
severity: medium
confidence: high
status: finding
file: src/flux/feat_flux.py
line_start: 195
line_end: 199
quote: |
        unet_ft = torch.cat(model_output_copy, dim=1)
        # print(unet_ft.shape)
        # unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
        
        return unet_ft.cpu(), torch.cat((mod.shift, mod.scale), dim=1).cpu()
claim: "The mean over the ensemble dimension (unet_ft.mean(0)) is commented out, so the saved feature keeps batch=ensemble_size. In eval, src_ft_raw.chunk(2) splits this batch into two halves, keeps the second half (feat_pred_text), then F.normalize/indexing uses only src_ft[0,...] (eval_spair.py:241), so the remaining ensemble copies are discarded rather than averaged."
concern: "The 'ensemble_size' described as averaging repeated noisy passes does not average anything in the released code; the effective feature is from a single (noise-specific) forward, which both changes the method and makes results sensitive to the unseeded torch.randn noise (feat_flux.py:160)."
resolution: "Confirm whether ensemble averaging was used for the reported numbers; if so, restore unet_ft.mean(0) and remove the chunk(2) split. Also seed the per-image noise for determinism."
cross_refs: ["chunk2-cfg-crash"]
paper_ref: "Sec 5.1 / test_spair.sh '--ensemble_size'"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: imgsize-768-vs-960
category: difference
topic: "evaluation config"
title: "Released command uses 768x768 input; paper App. Table 7 specifies 960x960 for DiTs"
severity: low
confidence: high
status: finding
file: test_spair.sh
line_start: 1
line_end: 1
quote: |
  CUDA_VISIBLE_DEVICES=1 python eval_spair.py --dataset spair --dataset_path spair-71k_path --save_path dfm_feats/spair_flux_img768 --dift_model flux --img_size 768 768 --t 260 --ft_index 28 --ensemble_size 1
claim: "The provided run command sets --img_size 768 768 (and save dir 'spair_flux_img768'), whereas the paper Appendix Table 7 / text states the DiT input image size is 960x960 (t=260, block k=28 match)."
concern: "Input resolution materially affects PCK; the shipped command does not match the paper's stated 960x960 setting, so a reviewer following the script would evaluate a different configuration than the one reported."
resolution: "Confirm the input size used for the reported DiTFflux SPair numbers (768 vs 960) and align the script/paper."
cross_refs: []
paper_ref: "Appendix Table 7; text 'input image size as 960x960 for DiTs'"
tags: [reforms:6]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding is raised. The implemented procedure (extract pre-AdaLN feature,
re-apply AdaLN, cosine-match keypoints, PCK@0.1 with per-point and per-image variants on
the SPair test split) is a standard, sound zero-shot semantic-correspondence evaluation
and does not, as written, introduce leakage or an inappropriate metric. The blocking issues
are absence (`missing`) and breakage (`bug`), not an invalid procedure. N/A for temporal /
pretraining-contamination split checks at the code level: DiTF is training-free and the
test set is the standard SPair-71k test split, so there is no train/test split to leak
across in the released eval.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | Channel discard, DINOv2+PCA SOTA rows, non-Flux & non-SPair evals, ablation toggles, README/deps all absent |
| bug         | 2          | high         | Shipped command crashes (chunk(2) on size-1 batch); ensemble never averaged |
| difference  | 1          | low          | Run command uses 768x768 vs paper's 960x960 |
| methodology | 0          | -            | Implemented SPair eval is sound; blockers are absence/breakage, not invalid method |

## 5. Closing lists

**Top take-aways** (ranked by severity × confidence):
1. (`bug`, high/high) `chunk2-cfg-crash` — the exact provided command (`--ensemble_size 1`) crashes on `chunk(2)` of a size-1 batch; the code does not run as shipped.
2. (`missing`, high/high) `dinov2-pca-fusion-missing` — every "†" state-of-the-art row (DiTF+DINOv2, e.g. SPair 72.2, AP-10K-C.S. 69.4) has no implementation; the PCA helper is dead code.
3. (`missing`, high/high) `channel-discard-missing` — the channel-discard contribution and its Table-4 ablation row (+1.8 PCK) are not in the code at all.
4. (`missing`, high/high) `nonflux-and-otherdataset-evals-missing` — only Flux-on-SPair is wired; SD3/SD3-5/Pixart and PF-Pascal/AP-10K/ADE20K/HPatches/temporal evals are absent.
5. (`bug`, medium/high) `ensemble-not-averaged` — ensemble averaging is commented out; the feature is a single unseeded-noise pass, not an average.
6. (`missing`, medium/high) `ablation-toggles-missing` — no flags to disable AdaLN/discard or vary the AdaLN condition, so Tables 4–5 cannot be reproduced.

**Items that genuinely look fine**:
- AdaLN re-application in eval (`(1+scale)*LayerNorm(x)+shift`, eval_spair.py:181-207) matches Eq. 8–9, with the saved `mod` (shift/scale) coming from the same single block whose pre-AdaLN feature is extracted (model.py:168-174).
- Model/VAE weights load from a resolvable HuggingFace repo (`black-forest-labs/FLUX.1-dev`, util.py:43-61); no dead local-only checkpoint path forces failure.
- PCK@0.1 computation and the per-point vs per-image accounting (eval_spair.py:231-296) is standard and consistent with the "per point for (U) methods" convention stated in the paper.
- Block-index bookkeeping (double blocks 0–18, single blocks 19+i, `ft_index 28` → single block 9) and the `H=W=48` derivation from saved feature shape are internally consistent for 768px input.

**Open questions for the authors**:
- What were the actual `ensemble_size`, input resolution (768 vs 960), and noise seeding used for the reported DiTFflux SPair number, given the shipped command crashes and does not average the ensemble?
- Will the channel-discard, DINOv2 integration, PCA fusion, and the SD3/SD3-5/Pixart and non-SPair evaluation scripts be released? Several were shipped only as `.pyc` (`feat_dinov2`, `feat_flux_v1`, `feat_flux_cross_image`) with no source.
