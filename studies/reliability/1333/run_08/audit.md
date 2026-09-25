# Audit — Latent Harmony (NeurIPS 2025, paper #1333)

## 1. Summary

The paper proposes **Latent Harmony**, a two-stage VAE framework for UHD all-in-one
image restoration: Stage 1 trains an "LH-VAE" with progressive degradation perturbation
(PDPS), a DINOv2-based degradation-invariance loss `LInv`, and a latent equivariance loss
`LEqv`; Stage 2 first trains a latent-space restoration network `Rθ` with a restoration
loss `LRes`, then applies high-frequency-guided LoRA fine-tuning (FHF-LoRA on the encoder,
PHF-LoRA on the decoder) with an inference-time blend parameter `α`. Tables 1–5 report
PSNR/SSIM/LPIPS/FID/NIQE/user-study numbers on UHD-LL, UHD-blur, UHD-haze, UHDN, UHD-rain,
UHD-snow, and a Gendeg standard-resolution benchmark.

The cloned repo (`code/lyd-2022__Latent-Harmony/`) is a **BasicSR-derived training skeleton**.
It contains the VAE architecture (`RAVAE_EQ`), a LoRA wrapper (`LoRAConv2d`/`RAVAEHFLora`),
a Stage-1 trainer (`EQVAEModel`), a Stage-2 trainer (`VAEadapter`), losses, a paired-image
dataset, a single training entrypoint (`basicsr/train.py`), and two **example** config YAMLs.
The README is one line (`# Latent-Harmony`). The NeurIPS checklist (item 5) states the code
would be released "upon acceptance," so the authors did not claim reproducibility from this
artefact at submission; nevertheless the repo is the public author code and I audit what it
implements vs. what the paper describes.

What I did:
- Read the paper (PDF + text extraction) and both config YAMLs; mapped every reported table
  to repo code.
- Statically (AST + yaml, importing nothing from the repo to avoid the hardcoded
  `sys.path` hacks) verified that every `type:` name in the configs resolves to a registered
  class — `_audit_code/check_registry_vs_configs.py` →
  `_audit_code/out/registry_vs_configs.csv`.
- File-existence checks for dependency spec, datasets, weights, download/eval scripts, and
  the Stage-2 restoration network — `_audit_code/check_missing_artifacts.py` →
  `_audit_code/out/missing_artifacts.csv`.
- Read every model/arch/loss file end-to-end to compare implemented procedure with the
  paper's Stage-1/Stage-2 equations.

Headline conclusions:
1. **None of the quantitative results in Tables 1–5 are reproducible from this repo**: no
   datasets, no trained weights, no evaluation/inference script, no dependency spec, no
   README commands.
2. The **central Stage-2 latent restoration network `Rθ` (paper Eq. 7) is absent** — the
   Stage-2 trainer only LoRA-fine-tunes the VAE and never builds or trains an `Rθ`.
3. The Stage-2 example config asks for an architecture (`RAVAE`) that is **not registered**,
   so `RAVAEHFLora` construction raises `KeyError` — Stage 2 cannot start as shipped.

## 2. Traceability table

Every quantitative artefact in the paper, mapped to repo code that *computes* it. "compute
location" means a script/function producing the number — not a plot/formatter.

| Paper artefact | Repo location that computes it | Status |
|---|---|---|
| Table 1 (4-deg) PSNR/SSIM/LPIPS, all rows incl. "Ours 27.32/.926…" | (none — no eval script, no data, no weights) | MISSING |
| Table 2 (6-deg) PSNR/SSIM/LPIPS, all rows incl. "Ours 27.14/.925…" | (none) | MISSING |
| Table 3 standard-res LPIPS/FID (PromptIR/Diff-Plugin/CosAE ± Ours) | (none) | MISSING |
| Table 4 generalization (unseen + composite) PSNR/SSIM/LPIPS | (none) | MISSING |
| Table 5(a) component ablation (w/o LInv, LEqv, PDPS, FHF/PHF-LoRA …) | (none — no ablation harness) | MISSING |
| Table 5(b) inference-time comparison (LH 0.43 s) | (none) | MISSING |
| Table 5(c) restoration-net ablation (Restormer/NAFNet/SFHformer ±Ours) | (none — Rθ not in repo) | MISSING |
| Table 5(d) α-sweep PSNR/SSIM/LPIPS/User (α=0.2…0.8) | `set_alpha` exists (LHVAE_hflora_arch.py:83), but no eval/α-sweep script, no weights | MISSING |
| Fig. 2 motivation (t-SNE, CDCS, DCT, loss curves, HF-LoRA) | (none) | MISSING |
| Fig. 4 visual results | (none — no inference script) | MISSING |
| FLOPs/Params (e.g. "Ours 3.6G / 1.2M") | (none — no FLOPs/params script) | MISSING |

Single underlying cause for the whole table: the repo ships training code only, with no data,
no weights, no eval entrypoint. Owned by `results-not-traceable` (and, for Table 5(c) /
Stage-2 numbers specifically, by `stage2-restoration-net-missing`).

## 3. Findings

## missing

```yaml finding
id: results-not-traceable
category: missing
topic: "result traceability"
title: "No eval script, datasets, or weights — Tables 1-5 not reproducible"
severity: high
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The repo contains a single training entrypoint (basicsr/train.py) and two example config YAMLs, but no evaluation/inference/test script, no datasets (configs point to non-existent ./datasets/...), no trained weights (./weights/stage1_eqvae.pth, ./weights/dinov2_vits14.pth absent), no data download/prepare scripts, and a one-line README with no reproduction commands. _audit_code/out/missing_artifacts.csv: every checked artefact is MISSING."
concern: "None of the PSNR/SSIM/LPIPS/FID/NIQE/runtime/FLOPs numbers in Tables 1-5 or any figure can be regenerated or verified from the repository; there is no code path that computes a reported value end-to-end."
resolution: "Authors: release the evaluation harness that computes the table metrics, the dataset accessions/download scripts (UHD-LL, UHD-blur, UHD-haze, UHDN, UHD-rain, UHD-snow, Gendeg), the trained checkpoints, and exact reproduction commands."
cross_refs: ["stage2-restoration-net-missing", "missing-dependency-spec", "§5", "Tables 1-5"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Tables 1-5; Figs 2,4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage2-restoration-net-missing
category: missing
topic: "method completeness / Stage 2"
title: "Stage-2 latent restoration network Rθ and LRes training are absent"
severity: high
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 132
line_end: 139
quote: |2
      def _forward_g(self):
          out = self.net_g(self.lq, alpha=self.alpha)
          if isinstance(out, (list, tuple)):
              if len(out) >= 3:
                  return out[0], out[1], out[2]
              if len(out) == 2:
                  return out[0], out[1], None
          return out, None, None
claim: "The Stage-2 trainer VAEadapter builds only net_g (the LoRA-wrapped VAE, RAVAEHFLora) and an optional HF discriminator; its forward passes the degraded input self.lq directly through the VAE. There is no latent restoration network Rθ and no LRes (||Dψ*(Rθ(Eφ*(Ideg))) - Iclean||1) training step anywhere in the model or repo (grep for net_r/restoration-net/LRes finds nothing; see _audit_code/out/missing_artifacts.csv)."
concern: "Paper Section 4.2 / Eq. 7 makes Rθ the core of Stage 2 ('a latent space restoration network Rθ … predict a restored latent zres = Rθ(zdeg)') and Table 5(c) ablates Rθ across Restormer/NAFNet/SFHformer; without Rθ the implemented Stage 2 is only HF-LoRA on raw VAE reconstruction, so the method that produces the restoration numbers is not in the repo."
resolution: "Authors: provide the Rθ architecture (paper states SFHformer is the default), the LRes pre-training stage, and the wiring that feeds zres = Rθ(zdeg) into the decoder before HF-LoRA fine-tuning."
cross_refs: ["results-not-traceable", "§4.2", "Table 5(c)", "Eq.7"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 4.2, Eq. 7; Table 5(c)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-dependency-spec
category: missing
topic: "environment / dependencies"
title: "No dependency specification (no requirements.txt / setup.py / env.yml)"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 8
line_end: 8
quote: |
  import pyiqa
claim: "The repo imports third-party packages (pyiqa, torch, torchvision, einops, tqdm, cv2, yaml, and DINOv2 via torch.hub.load('facebookresearch/dinov2', ...)) but ships no requirements.txt, setup.py, setup.cfg, pyproject.toml, environment.yml, or Pipfile (see _audit_code/out/missing_artifacts.csv: dependency_spec=MISSING)."
concern: "The runtime environment cannot be rebuilt deterministically; unpinned/unlisted dependencies (e.g. a pyiqa version compatible with the 'lpips-vgg' metric, a torch version matching the DINOv2 hub model) block reproduction."
resolution: "Authors: add a pinned requirements file or environment.yml listing all imports and versions, including the DINOv2 source and pyiqa version."
cross_refs: ["results-not-traceable"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "NeurIPS checklist Q5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-readme-and-commands
category: missing
topic: "documentation / reproducibility"
title: "README is a one-line stub with no results table or run commands"
severity: low
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The entire README.md is the single line '# Latent-Harmony' (16 bytes; see _audit_code/out/missing_artifacts.csv readme_bytes=16). There are no dataset-preparation instructions, no Stage-1/Stage-2 run commands, no results table, and no checkpoint links."
concern: "A user cannot determine how to prepare data, which config to run, or in what order the two stages execute, so even the training code is not actionable as shipped."
resolution: "Authors: add a README documenting data layout, the two-stage run commands, expected outputs, and a results table tying configs to paper numbers."
cross_refs: ["results-not-traceable"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Repository README"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-vae-type-unregistered
category: bug
topic: "config / arch registry"
title: "stage2 config requests vae type 'RAVAE' which is not registered → KeyError"
severity: high
confidence: high
status: finding
file: basicsr/archs/LHVAE_hflora_arch.py
line_start: 36
line_end: 38
quote: |2
          cfg = deepcopy(vae_config)
          arch_type = cfg.pop('type')
          self.vae = ARCH_REGISTRY.get(arch_type)(**cfg)
claim: "RAVAEHFLora reads vae_config['type'] and calls ARCH_REGISTRY.get(arch_type). configs/stage2_hflora.yml sets network_g.vae_config.type: RAVAE (line 42), but no class named 'RAVAE' is registered — only 'RAVAE_EQ' and 'RAVAEHFLora' (verified by AST scan in _audit_code/out/registry_vs_configs.csv: stage2_hflora.yml,RAVAE,UNREGISTERED). ARCH_REGISTRY.get raises KeyError when the name is absent (basicsr/utils/registry.py:62-66)."
concern: "Building the Stage-2 model from the shipped example config crashes immediately with KeyError(\"No object named 'RAVAE' found in 'arch' registry!\"), so Stage 2 cannot be trained as released without editing the config to 'RAVAE_EQ'."
resolution: "Authors: change stage2 vae_config.type to 'RAVAE_EQ' (or register a 'RAVAE' alias) and confirm the intended backbone."
cross_refs: ["§4.2"]
check_script: _audit_code/check_registry_vs_configs.py
paper_ref: "configs/stage2_hflora.yml line 42"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-experiments-root
category: bug
topic: "hardcoded absolute paths"
title: "Training output dir hardcoded to author machine path /fs-computility/..."
severity: medium
confidence: high
status: finding
file: basicsr/utils/options.py
line_start: 158
line_end: 164
quote: |2
      if is_train:
          experiments_root = osp.join('/fs-computility/ai4sData/liuyidi/model', 'experiments', opt['name'])
          opt['path']['experiments_root'] = experiments_root
          opt['path']['models'] = osp.join(experiments_root, 'models')
          opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
          opt['path']['log'] = experiments_root
          opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
claim: "For every training run (is_train=True), the experiments root (where checkpoints, logs, training states, and visualizations are written) is hardcoded to the absolute author-cluster path /fs-computility/ai4sData/liuyidi/model/experiments/<name>, instead of the stock BasicSR osp.join(root_path, 'experiments', opt['name'])."
concern: "On any machine without that exact directory tree, training fails to write checkpoints/logs (FileNotFoundError / PermissionError) or silently writes to an unexpected absolute location, blocking out-of-the-box training reproduction."
resolution: "Authors: derive experiments_root from root_path (or a config field) rather than a hardcoded absolute path."
cross_refs: ["dead-syspath-appends"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "basicsr/utils/options.py:159"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dead-syspath-appends
category: bug
topic: "hardcoded absolute paths"
title: "Multiple hardcoded sys.path.append to absent author-machine dirs"
severity: low
confidence: high
status: finding
file: basicsr/train.py
line_start: 8
line_end: 8
quote: |
  sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
claim: "Several modules unconditionally append absolute author-machine paths to sys.path: basicsr/train.py:8 ('/fs-computility/ai4sData/liuyidi/code/LatentGen'), basicsr/archs/LHVAE_arch.py:7 (same), basicsr/utils/modules/attention.py:8 ('/code/UHDFour_code-main'), basicsr/archs/encoder_3.py:8 ('/code/UHDformer-main'). The imports that actually run resolve to in-repo basicsr.* modules, so these appends are dead leftovers, but they reveal the code was never path-portability-checked."
concern: "Although harmless at import time (no active import depends on these dirs), the hardcoded absolute paths are a portability smell and, combined with the live experiments_root hardcode, indicate the released code was not validated outside the author's filesystem."
resolution: "Authors: remove the dead sys.path.append lines."
cross_refs: ["hardcoded-experiments-root"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "basicsr/train.py:8; LHVAE_arch.py:7; attention.py:8; encoder_3.py:8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: eqv-loss-on-perturbed-latent
category: difference
topic: "Stage-1 equivariance loss"
title: "LEqv applied to perturbed-image latent, paper defines it on the clean latent"
severity: low
confidence: medium
status: finding
file: basicsr/models/EQvae_model.py
line_start: 283
line_end: 293
quote: |2
      def _compute_eqv_loss(self, z):
          if self.lambda_eqv <= 0:
              return z.new_zeros(())
          if self.eqv_scale_factor <= 0 or self.eqv_scale_factor >= 1:
              raise ValueError(f'eqv.scale_factor must be in (0, 1), got {self.eqv_scale_factor}')

          z_down = F.interpolate(z, scale_factor=self.eqv_scale_factor, mode='bilinear', align_corners=False)
          net_g = self.get_bare_model(self.net_g)
          pred_down = net_g.decode(z_down)
          gt_down = F.interpolate(self.gt, size=pred_down.shape[2:], mode='bilinear', align_corners=False)
          return F.l1_loss(pred_down, gt_down)
claim: "The equivariance loss downsamples z, the latent of the *perturbed* input pdps_input (z comes from net_g(self.pdps_input) at optimize_parameters line 299), decodes it, and compares to a downsampled clean GT. Paper Eq. 5 defines LEqv = ||Dψ(zdown) - Idown||1 with zdown = Downs(zclean), i.e. the latent of the *clean* image Iclean, not the perturbed one."
concern: "The implemented equivariance constraint operates on a different (degradation-perturbed) latent than the paper specifies; the procedure is still a valid latent-equivariance regularizer, so this is a paper↔code difference rather than an invalid method, but it changes what the constraint actually enforces."
resolution: "Authors: clarify whether LEqv should use the clean-image latent (as in Eq. 5) or the perturbed-input latent (as coded); align code or paper accordingly."
cross_refs: ["§4.1", "Eq.5"]
paper_ref: "Section 4.1, Eq. 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding is filed. The data-splitting / leakage / baseline / tuning topics are
**N/A at the code level**: the repo provides no split-generation, no evaluation harness, no
baselines, and no datasets, so there is no implemented procedure whose statistical validity
can be assessed — those concerns are subsumed by the `missing` findings above (you cannot
audit a leak in code that does not exist). The paired-image dataset
(`basicsr/data/paired_image_dataset.py`) only reads pre-split gt/lq folders supplied by the
user; train/val are separate directories, so no in-repo split logic leaks.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | No data/weights/eval/deps; Stage-2 Rθ absent → Tables 1-5 untraceable |
| bug         | 3          | high         | stage2 config requests unregistered 'RAVAE' (KeyError); hardcoded experiments_root; dead sys.path hacks |
| difference  | 1          | low          | LEqv computed on perturbed-input latent vs. clean latent (Eq. 5) |
| methodology | 0          | -            | N/A — no implemented eval/split/baseline procedure to assess |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] results-not-traceable** — no eval script, datasets, weights, or commands;
   none of Tables 1–5 / figures are reproducible from the repo. (high/high)
2. **[bug] stage2-vae-type-unregistered** — `vae_config.type: RAVAE` is not a registered
   class; `RAVAEHFLora` build raises `KeyError`, so Stage 2 cannot start as shipped. (high/high)
3. **[missing] stage2-restoration-net-missing** — the core Stage-2 latent restoration network
   `Rθ` and its `LRes` pre-training (Eq. 7, Table 5(c)) are absent. (high/high)
4. **[missing] missing-dependency-spec** — no requirements/env file; environment cannot be
   rebuilt. (medium/high)
5. **[bug] hardcoded-experiments-root** — training output dir hardcoded to
   `/fs-computility/ai4sData/liuyidi/model/...`, breaking training on other machines. (medium/high)
6. **[difference] eqv-loss-on-perturbed-latent** — `LEqv` uses the perturbed-input latent,
   not the clean-image latent of Eq. 5. (low/medium)

### Items that genuinely look fine
- Stage-1 PDPS implementation (`EQvae_model.py:206-224`) matches Eq. 3: three branches
  (clean / synthetic-degraded / clean↔degraded interpolation) selected by probabilities
  p0/p1/p2, with monotonically increasing severity `t` and interpolation `beta = t`.
- `LInv` (`_compute_inv_loss`, `EQvae_model.py:270-281`) aligns the perturbed-input latent to
  DINOv2 patch tokens of the clean GT, matching Eq. 4's intent (`z'_deg → fVFM`).
- Inference `α` blending (`RAVAEHFLora.set_alpha`, `LHVAE_hflora_arch.py:83-91`) implements
  `enc_scale = α`, `dec_scale = 1-α`, matching Eq. "ϕ = ϕ* + αΔϕ, ψ = ψ* + (1-α)Δψ".
- LoRA delta is zero-initialized on `lora_up` (`lora_arch.py:50`), so an untrained adapter is a
  no-op — correct LoRA behavior.
- Stage-1 config types (`EQVAEModel`, `RAVAE_EQ`, `KlLoss`, `L1Loss`, `PairedImageDataset`)
  all resolve to registered classes (`_audit_code/out/registry_vs_configs.csv`).

### Open questions for the authors
- Was the released `basicsr/` skeleton the exact code that produced Tables 1–5, or a
  re-implementation? (The absent `Rθ`, missing eval harness, and `RAVAE` config typo suggest
  the public repo is a partial/cleaned subset.)
- For `LEqv`: clean-image latent (Eq. 5) or perturbed-input latent (code)?
- Which dataset versions/accessions and DINOv2 weights are required, and what are the exact
  two-stage run + evaluation commands?
