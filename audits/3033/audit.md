# Audit — One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models (NeurIPS 2025, #3033)

## Summary

The cloned repository `code/surkovv__sdxl-unbox/` is the author code for this paper.
Provenance is confirmed: the first author (Viacheslav Surkov) and co-authors match;
the paper's footnote 2 names the project page `https://sdxl-unbox.epfl.ch` as the hub
that "links to all mentioned repositories", and this repo is `surkovv/sdxl-unbox`. The
repository's `README.MD` is for the earlier arXiv version ("Unpacking SDXL Turbo",
arXiv 2410.22366), which the NeurIPS paper extends (adding the "one-step is enough"
generalization framing, RIEBench, and FLUX); the code contents (SDLens hooked SDXL
pipeline, TopK SAE training, 4 block checkpoints) are exactly the SDXL-Turbo machinery
the paper describes.

What the repo provides: (1) `SDLens/` — the hooked SDXL pipeline used to cache/intervene
on transformer-block residual updates; (2) `SAE/` + `scripts/train_sae.py` — the TopK SAE
implementation and training loop (matches Appendix K hyperparameters: k, nf=5120, auxk=256,
bs=4096, lr=1e-4, Adam betas, geometric-median pre-bias init, decoder unit-norm); (3)
`scripts/collect_latents_dataset.py` — the 1-step SDXL Turbo latent-collection script; (4)
four pretrained SAE checkpoints (k=10, nf=5120) for down.2.1, mid.0, up.0.0, up.0.1; (5)
`utils/hooks.py` intervention primitives and a gradio demo (`app.py`/`app.ipynb`) plus a
qualitative `example.ipynb`.

What I did: I read the SAE model/training/dataloader code, the hooked pipeline, the hook
primitives, both notebooks, and `requirements.txt`; I read the paper's main text and
Appendices C, D, F, K, L. I wrote two read-only check scripts under `_audit_code/`:
`check_imports_and_savepath.py` (AST-confirms an ImportError in `example.ipynb` and a
free-variable NameError in `SAETrainingConfig.save_path`) and `check_traceability.py`
(greps every `.py` file and notebook code cell for the computation primitive each
quantitative artefact requires). Outputs are in `_audit_code/out/`.

Headline conclusion of the audit: the repo faithfully reproduces the SAE **training** and
the **qualitative** interventions, but the **quantitative evaluation harness for every
numbered figure and table is absent** (Fig. 2, Fig. 3, Fig. 5, Fig. 6, Table 1, Table 5),
and the entire **FLUX** track — a headline contribution — has no code at all. Two
small but real runtime bugs are present in the documented entry points. No methodological
leakage/validity defect was found in the code that *is* present (the SAE objective, the
CFG add-to-cond/subtract-from-uncond intervention, and the EV metric are all correct).

## Result-traceability coverage table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Training EV metric (used for Fig. 2 & Fig. 3 EV) | `scripts/train_sae.py:227-238` (`explained_variance`) | metric fn present, logged to wandb during training | n/a (logged, not the plotted sweep/multistep numbers) | PARTIAL |
| Fig. 2 EV vs k and vs expansion factor (line-search sweep) | (none) | — | — | MISSING (no sweep driver computes plotted EV) |
| Fig. 3 (left) EV across denoising steps, 4-step/20-step SDXL | (none) | — | — | MISSING (no multi-step EV evaluation) |
| Fig. 3 (right) SAE feature overlap = cosine sim of adjacent-timestep coeffs | (none) | — | — | MISSING (no cosine-similarity/overlap code) |
| Fig. 5 RIEBench: LPIPS vs CLIP-sim, SAE/neuron/steering, 9 categories | (none) | — | — | MISSING (no LPIPS, no CLIP-score, no SAM2, no RIEBench harness) |
| Fig. 5 feature-selection score eq.(8) (SAE) / eq.(9) (neurons) | (none) | — | — | MISSING |
| Fig. 6 per-block feature-selection counts by edit category | (none) | — | — | MISSING (depends on absent RIEBench selection) |
| Table 1 per-block ablation LPIPS, mean over 20 prompts | `utils/hooks.py:118-120` (`ablate_block` primitive only) | hook defined, never called by any driver; no LPIPS | — | MISSING (no driver computes the 28 values) |
| Table 5 reconstruction Mean/Median Manhattan, LPIPS, EV (%) | `utils/hooks.py:110-115` (`reconstruct_sae_hook` primitive only) | hook defined, never called; no Manhattan/LPIPS/EV-in-image driver | — | MISSING (no driver computes the table) |
| FLUX SAE training (layer 18, k=20, nf=12288) — App. C | (none) | — | — | MISSING (no FLUX code anywhere) |
| FLUX interventions Figs 1,8,9,35–43; FLUX RIEBench Fig.14 | (none) | — | — | MISSING (no FLUX pipeline/hook code) |
| SAE training hyperparameters (k, nf=5120, auxk=256, λ=1/32, bs=4096, lr=1e-4) | `SAE/sae_utils.py:5-25`, `scripts/train_sae.py:241-304` | matches App. K | ✓ | Verified |
| Pretrained checkpoints (4 blocks, k=10, nf=5120) | `checkpoints/*/final/config.json` | k=10, n_dirs=5120, d_model=1280, auxk=256 | ✓ | Verified |
| Latent dataset = 1.5M LAION-COCO 1-step generations | `scripts/collect_latents_dataset.py:19,28` | uses `guangyil/laion-coco-aesthetic` (aesthetic-filtered re-host), default 30000×50=1.5M | partial | PARTIAL (dataset variant differs from cited [54]) |
| CFG intervention: add to cond, subtract from uncond (App. B) | `utils/hooks.py:44-57` (`add_feature_on_area_base_both`) | subtract from `output[0][0]` (uncond), add to `output[0][1]` (cond) | ✓ | Verified |

## missing

```yaml finding
id: flux-track-no-code
category: missing
topic: "result traceability / FLUX"
title: "Entire FLUX contribution has no code (training, interventions, RIEBench)"
severity: high
confidence: high
status: finding
file: scripts/collect_latents_dataset.py
line_start: 29
line_end: 29
quote: |
    pipe = HookedStableDiffusionXLPipeline.from_pretrained('stabilityai/sdxl-turbo')
claim: "Every pipeline, hook, and script in the repo targets SDXL Turbo only; there is no FLUX pipeline class, no FLUX SAE training, and no FLUX intervention code (grep for 'flux'/'schnell'/'FluxPipeline'/'layer 18' over all .py files and notebook code cells returns zero matches; only SDXL Turbo is loaded)."
concern: "FLUX generalization is a headline contribution ('we consider this a crucial result', Sec. 1.1; Fig. 1 row 4; Figs 8, 9, 35-43; App. C; Fig. 14), but none of the FLUX results can be reproduced or inspected from this repository."
resolution: "Authors: please release the FLUX SAE-training and intervention code (App. C says k=20, nf=12288, layer-18 activations of Flux-schnell) or point to the exact repository/commit that produces Figs 1(row4), 8, 9, 14, 35-43."
cross_refs: ["riebench-eval-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "Sec. 1.1 'Additionally, we train SAEs ... FLUX Schnell'; Appendix C"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: riebench-eval-missing
category: missing
topic: "result traceability / RIEBench"
title: "No code computes Fig. 5 / Fig. 6 RIEBench metrics (LPIPS, CLIP-sim, SAM2, selection)"
severity: high
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 13
quote: |
  diffusers==0.29.2
  gradio==4.44.1
  torch>=2.4.0
  numpy
  matplotlib
  pillow
  wandb
  einops
  transformers
  accelerate
  huggingface_hub
  git+https://github.com/wendlerc/clip-retrieval.git
claim: "No file in the repo computes the LPIPS distance, the CLIP-similarity increase, the grounded-SAM2 segmentation masks, the eq.(8)/(9) feature-selection scores, or the PIEBench-derived edit-category harness that produce Fig. 5 and Fig. 6; correspondingly lpips/sam2/groundingdino are not even declared as dependencies."
concern: "Fig. 5 and Fig. 6 carry the paper's central quantitative claim (SAE features match neuron baselines 'while requiring several orders of magnitude fewer features' and reveal block specialization); none of these numbers can be reproduced from this repository."
resolution: "Authors: please release the RIEBench evaluation harness (the companion repo wendlerc/RIEBench appears to be it) and pin it from this repo, including LPIPS/CLIP-score computation and the SAM2 mask generation, so Fig. 5/6 are reproducible."
cross_refs: ["flux-track-no-code", "table5-recon-no-driver", "table1-ablation-no-driver"]
check_script: _audit_code/check_traceability.py
paper_ref: "Sec. 4.1 (Fig. 5), Sec. 4.2 (Fig. 6)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fig2-fig3-ev-sweep-missing
category: missing
topic: "result traceability / explained variance"
title: "No driver produces Fig. 2 (EV vs k / expansion) or Fig. 3 (EV across steps, feature overlap)"
severity: high
confidence: high
status: finding
file: scripts/train_sae.py
line_start: 227
line_end: 238
quote: |
  def explained_variance(recons, x):
      # Compute the variance of the difference
      diff = x - recons
      diff_var = torch.var(diff, dim=0, unbiased=False)

      # Compute the variance of the original tensor
      x_var = torch.var(x, dim=0, unbiased=False)

      # Avoid division by zero
      explained_var = 1 - diff_var / (x_var + 1e-8)

      return explained_var.mean()
claim: "The repo defines a per-batch explained_variance() that is logged to wandb during training, but there is no script that (a) sweeps k and expansion factor and aggregates EV for Fig. 2, or (b) runs the SAEs over 4-step/20-step SDXL generations and computes EV per denoising step plus the adjacent-timestep cosine-similarity 'feature overlap' for Fig. 3 (no cosine_similarity / overlap code exists)."
concern: "Fig. 2 and Fig. 3 (including the headline 'one-step is enough' generalization-across-steps result) cannot be regenerated; only a training-time scalar EV exists, not the plotted sweep or multi-step curves."
resolution: "Authors: please release the evaluation notebook/script that computes the Fig. 2 sweep and the Fig. 3 multi-step EV and feature-overlap curves."
cross_refs: ["riebench-eval-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "Fig. 2, Fig. 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: table5-recon-no-driver
category: missing
topic: "result traceability / Table 5"
title: "Table 5 reconstruction numbers have a hook primitive but no driver to compute them"
severity: medium
confidence: high
status: finding
file: utils/hooks.py
line_start: 110
line_end: 115
quote: |
  @torch.no_grad()
  def reconstruct_sae_hook(sae, module, input, output):
      diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
      activated = sae.encode(diff)
      reconstructed = sae.decoder(activated) + sae.pre_bias
      return (input[0] + reconstructed.permute(0, 3, 1, 2).to(output[0].device),)
claim: "reconstruct_sae_hook is defined but never called by any script or notebook; nothing computes the pixel Manhattan mean/median distance, the LPIPS column, or the per-block EV (%) over 100 prompts that populate Table 5 (24 rows × 4 columns)."
concern: "The Table 5 reconstruction-quality numbers (and Fig. 18) cannot be reproduced from the repo; only the intervention primitive exists."
resolution: "Authors: please release the script that drives reconstruct_sae_hook over 100 LAION-COCO prompts and computes the Manhattan/LPIPS/EV columns of Table 5."
cross_refs: ["riebench-eval-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "Appendix L, Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: table1-ablation-no-driver
category: missing
topic: "result traceability / Table 1"
title: "Table 1 per-block ablation LPIPS has a hook primitive but no driver/LPIPS computation"
severity: medium
confidence: high
status: finding
file: utils/hooks.py
line_start: 118
line_end: 120
quote: |
  @torch.no_grad()
  def ablate_block(module, input, output):
      return input
claim: "ablate_block (returns the block input, i.e. zeroes the residual update) is defined but never called by any script or notebook; nothing iterates over the 28 blocks, ablates each, generates 20 random prompts, and computes the mean LPIPS distance reported in Table 1."
concern: "Table 1 (the rebuttal-added quantitative causal-impact ranking of resnet/attention blocks) cannot be reproduced from the repo."
resolution: "Authors: please release the script that drives ablate_block per block over 20 prompts and computes the mean LPIPS in Table 1."
cross_refs: ["riebench-eval-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "Appendix D, Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: example-nb-import-error
category: bug
topic: "documented entry point"
title: "example.ipynb imports add_feature_on_area / replace_with_feature, which do not exist"
severity: medium
confidence: high
status: finding
file: utils/hooks.py
line_start: 40
line_end: 44
quote: |
  @torch.no_grad()
  def add_feature_on_area_base(sae, feature_idx, activation_map, module, input, output):
      return add_feature_on_area_base_both(sae, feature_idx, activation_map, module, input, output)

  @torch.no_grad()
  def add_feature_on_area_base_both(sae, feature_idx, activation_map, module, input, output):
claim: "The first code cell of example.ipynb runs `from utils import add_feature_on_area, replace_with_feature`, but utils/hooks.py (re-exported via `from .hooks import *`) defines only add_feature_on_area_base/_base_both/_base_cond/_turbo and replace_with_feature_base/_turbo — there is no symbol named add_feature_on_area or replace_with_feature."
concern: "The README directs users to example.ipynb for analysis examples, yet its very first cell raises ImportError, so the documented usage example does not run as shipped."
resolution: "Rename the imports in example.ipynb to the *_turbo (or *_base) variants, or add aliases add_feature_on_area / replace_with_feature in utils/hooks.py."
cross_refs: []
check_script: _audit_code/check_imports_and_savepath.py
paper_ref: "README.MD 'See example.ipynb for analysis examples'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: savepath-free-variable
category: bug
topic: "training entry point"
title: "SAETrainingConfig.save_path references free name save_path_base, not self.save_path_base"
severity: medium
confidence: high
status: finding
file: SAE/sae_utils.py
line_start: 23
line_end: 25
quote: |
      @property
      def save_path(self):
          return os.path.join(save_path_base, f'{self.block_name}_k{self.k}_hidden{self.n_dirs}_auxk{self.auxk}_bs{self.bs}_lr{self.lr}')
claim: "save_path_base is stored as an instance field of SAETrainingConfig (line 12) yet the save_path property references the bare name save_path_base instead of self.save_path_base; there is no module-level or builtin save_path_base, so evaluating cfg.save_path raises NameError."
concern: "train_sae.py calls cfg.save_path at every save_interval (line 187) and at the end of training (line 204), so a training run crashes with NameError the first time it tries to checkpoint, preventing the SAEs from being saved."
resolution: "Change `save_path_base` to `self.save_path_base` in the save_path property of SAETrainingConfig."
cross_refs: []
check_script: _audit_code/check_imports_and_savepath.py
paper_ref: "README.MD step 2.2 `python scripts/train_sae.py`"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: laion-coco-aesthetic-variant
category: difference
topic: "training data"
title: "Code trains on guangyil/laion-coco-aesthetic, paper cites LAION-COCO [54]"
severity: low
confidence: high
status: finding
file: scripts/collect_latents_dataset.py
line_start: 28
line_end: 28
quote: |
      dataset = load_dataset("guangyil/laion-coco-aesthetic", split="train", columns=["caption"], streaming=True).shuffle(seed=42)
claim: "The latent-collection script streams prompts from the HuggingFace dataset `guangyil/laion-coco-aesthetic` (an aesthetic-filtered third-party re-host), whereas the paper states the SAEs were trained on '1.5M prompts from the LAION-COCO [54]', citing the original LAION blog dataset (laion.ai/blog/laion-coco)."
concern: "The actual training prompts come from an aesthetic-filtered subset, not the original LAION-COCO captioning set the paper cites; both are valid prompt sources, but the description and the artefact differ."
resolution: "Authors: clarify whether the reported SAEs were trained on guangyil/laion-coco-aesthetic and update the citation, or provide the script variant that uses the original LAION-COCO."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "Sec. 3 'Training' and ref [54]"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: config-default-k20
category: difference
topic: "hyperparameter config"
title: "Shipped SAE/config.json defaults to k=20, paper's main SDXL SAEs use k=10"
severity: low
confidence: medium
status: finding
file: SAE/config.json
line_start: 2
line_end: 12
quote: |
      "sae_configs": [
          {
              "d_model": 1280,
              "n_dirs": 5120,
              "k": 20
          },
          {
              "d_model": 1280,
              "n_dirs": 640,
              "k": 20
          }
      ],
claim: "The training config shipped in the repo sets k=20 for both SAEs, while the paper's main SDXL Turbo SAEs (and the provided checkpoints) use k=10 (Appendix K: 'The value of k is set to 10'); the checkpoint folders are k10_hidden5120."
concern: "A user running the documented `python scripts/train_sae.py` reproduces a k=20 SAE, not the paper's main k=10 SAE; this is a template/default mismatch rather than a result error since the released checkpoints are correctly k=10."
resolution: "Authors: set the example config k to 10 (and document the k/expansion sweep used for Fig. 2), or note that config.json is only a template."
cross_refs: []
check_script: _audit_code/check_imports_and_savepath.py
paper_ref: "Appendix K 'The value of k is set to 10'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding. The procedures implemented by the code that is present are
sound: the TopK SAE objective with auxiliary dead-feature loss, geometric-median pre-bias
initialization, per-step decoder unit-norm and gradient projection match Gao et al. [19]
and the paper's Appendix K; the explained-variance metric (`train_sae.py:227-238`) is a
standard `1 - Var(x-x̂)/Var(x)`; and the classifier-free-guidance intervention
(`utils/hooks.py:44-57`) correctly adds the feature to the conditional pass and subtracts
it from the unconditional pass, exactly as Appendix B describes. The SAEs are an
unsupervised reconstruction model with no train/test label split, so leakage / sample-
independence / temporal-integrity checks are N/A. The quantitative-evaluation defects are
all *absence* of code (routed to `missing`), not invalid implemented procedures.

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                             |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 5          | high         | No eval code for Fig 2/3/5/6, Table 1/5, or the FLUX track. |
| bug         | 2          | medium       | example.ipynb ImportError; train save_path NameError.       |
| difference  | 2          | low          | aesthetic LAION-COCO variant; config defaults to k=20.      |
| methodology | 0          | -            | Implemented procedures (SAE, EV, CFG intervention) sound.   |

## Top take-aways (ranked)

1. **[missing] FLUX has no code at all** (`flux-track-no-code`, high/high) — a headline
   contribution (Fig. 1 row 4, Figs 8/9/35–43, App. C, Fig. 14) is unreproducible from this repo.
2. **[missing] RIEBench evaluation harness absent** (`riebench-eval-missing`, high/high) —
   Fig. 5/6 (the central quantitative claim: SAEs match neurons with orders-of-magnitude
   fewer features) has no LPIPS/CLIP-score/SAM2/selection code; the companion repo
   `wendlerc/RIEBench` is referenced but not part of the audited artefact.
3. **[missing] Fig. 2 & Fig. 3 generalization curves unreproducible** (`fig2-fig3-ev-sweep-missing`,
   high/high) — only a training-time scalar EV exists; no k/expansion sweep, no multi-step EV,
   no feature-overlap cosine code.
4. **[missing] Table 1 & Table 5 numbers have only hook primitives** (`table1-ablation-no-driver`,
   `table5-recon-no-driver`, medium/high) — `ablate_block`/`reconstruct_sae_hook` are defined
   but never driven, and no LPIPS/Manhattan computation exists.
5. **[bug] Training entry point crashes on first checkpoint** (`savepath-free-variable`,
   medium/high) — `cfg.save_path` raises `NameError` (free `save_path_base`), so
   `python scripts/train_sae.py` cannot save SAEs as shipped.
6. **[bug] Documented example notebook fails on cell 0** (`example-nb-import-error`,
   medium/high) — `from utils import add_feature_on_area, replace_with_feature` references
   names that do not exist in `utils/hooks.py`.

## Items that genuinely look fine

- SAE model (`SAE/sae.py`): TopK encoder/decoder, tied init, unit-norm decoder, auxiliary
  dead-feature reconstruction, geometric-median pre-bias — matches Gao et al. [19] and App. K.
- Training hyperparameters (`SAE/sae_utils.py`, `scripts/train_sae.py`): k, nf=5120, auxk=256,
  λ=1/32, bs=4096, lr=1e-4, Adam betas, dead-token threshold 10M/bs — all match App. K.
- Pretrained checkpoints: 4 blocks, k=10, nf=5120, d_model=1280, auxk=256 — match the paper's
  "best" SDXL configuration.
- `explained_variance` (`train_sae.py:227-238`): correct standard EV definition.
- CFG intervention (`utils/hooks.py:44-57`): correctly adds to cond / subtracts from uncond,
  matching Appendix B; the uncond/cond chunk order (`output[0][0]` uncond, `output[0][1]` cond)
  is correct for diffusers SDXL.
- `app.py` uses the correctly-named `*_base`/`*_turbo` hooks (the ImportError is confined to
  example.ipynb).

## Open questions for the authors

- Is the released RIEBench evaluation code (LPIPS/CLIP/SAM2/feature-selection) the companion
  repo `wendlerc/RIEBench`, and at which commit does it reproduce Fig. 5/6? It is referenced
  by the project page but is not part of this code release.
- Were the reported SAEs trained on `guangyil/laion-coco-aesthetic` rather than the cited
  original LAION-COCO? (`laion-coco-aesthetic-variant`.)
- Will the FLUX SAE-training and intervention code (App. C; k=20, nf=12288, layer-18) be
  released, given it underpins a stated "crucial result"?
