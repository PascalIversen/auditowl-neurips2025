# Audit — Latent Harmony: Synergistic Unified UHD Image Restoration (paper 1333)

## Summary

The cloned repo `code/lyd-2022__Latent-Harmony/` is a stripped-down BasicSR-style
training framework containing two model trainers (`EQVAEModel` = Stage 1 LH-VAE,
`VAEadapter` = Stage 2 HF-LoRA), the VAE architecture (`RAVAE_EQ`), a LoRA wrapper,
losses, metrics, a data loader, and two **example** YAML configs
(`configs/stage1_eqvae.yml`, `configs/stage2_hflora.yml`). The README is a single line
(`# Latent-Harmony`). I read the paper (PDF + text extraction), enumerated the reported
tables/figures, and statically analyzed the repo: I walked the file tree, parsed the
arch registry with AST, traced the import graph of `basicsr/__init__.py`, traced the
Stage-2 forward/optimization path, and checked for reproducibility artefacts (datasets,
weights, requirements, test/inference scripts). Deterministic checks live in
`_audit_code/check_import_and_registry.py` (output
`_audit_code/out/import_and_registry.csv`). I did **not** execute the repo's own code
(it requires torch/cv2/pyiqa/DINOv2 weights and 4K UHD datasets that are not present);
the run-blocking findings below are confirmed by static parsing, not by speculation.

Two issues make the code unrunnable as shipped even before any data is supplied:
(1) `basicsr/__init__.py` imports a `.test` submodule that does not exist in the repo, so
`import basicsr` (and therefore `python basicsr/train.py`) raises `ModuleNotFoundError`;
(2) the Stage-2 config instantiates a VAE of `type: RAVAE`, but no class named `RAVAE`
is registered (only `RAVAE_EQ`), so `RAVAEHFLora.__init__` raises `KeyError` from the
registry. On top of these, the experiment root path is hardcoded to the authors' cluster
filesystem, and the entire evaluation pipeline behind every paper number (datasets,
trained weights, inference/metric script, FLOPs/params computation) is absent — consistent
with the paper's own checklist answer "We will release the source code upon acceptance".

## Traceability table (Rule G)

Every quantitative claim in the paper is produced by an off-repo pipeline. No dataset,
no trained weight, and no inference/metric-reporting script is present; the two configs
are `*_example` training configs with `metrics: ~` (validation metrics disabled) and dummy
`./datasets/...` paths. The PSNR/SSIM/LPIPS/FID/NIQE/FLOPs/Params values therefore have no
in-repo computation path.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (4-deg) PSNR/SSIM/LPIPS, FLOPs, Params for "Ours" and all baselines | (none — no test/inference script, no weights, no datasets) | — | — | MISSING |
| Table 2 (6-deg) PSNR/SSIM/LPIPS, FLOPs, Params | (none) | — | — | MISSING |
| Table 3 standard-resolution LPIPS/FID (PromptIR/Diff-Plugin/CosAE ±Ours) | (none) | — | — | MISSING |
| Table 4 generalization PSNR/SSIM/LPIPS (unseen/composite) | (none) | — | — | MISSING |
| Table 5(a)-(d) ablations (Latent Harmony, latent restoration net, runtime, α-sweep) | (none — no ablation driver) | — | — | MISSING |
| Fig. 2 (t-SNE, CDCS, DCT, fine-tune loss, HF-LoRA) motivation plots | (none) | — | — | MISSING |
| Fig. 4 visual results (4 degradations) | (none — no inference script) | — | — | MISSING |
| FLOPs (256×256) / Params columns ("3.6G / 1.2M" for Ours) | (none — no flop-counting script; `thop` used only in dead `__main__` of LHVAE_arch.py with an absolute-path config) | — | — | MISSING |
| Inference-time α fidelity/perception trade-off (§4.3) | `LHVAE_hflora_arch.py:83-91` `set_alpha` exists, but no inference script exercises it | — | — | MISSING (no driver) |

All rows route to the `missing` finding `repro-pipeline-and-data-absent`.

## missing

```yaml finding
id: repro-pipeline-and-data-absent
category: missing
topic: "result traceability / completeness"
title: "No datasets, weights, inference/metric script, or FLOPs code — no paper number is reproducible"
severity: high
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The repo ships only Stage-1/Stage-2 training code plus two *_example* configs (metrics disabled, dummy ./datasets paths). There is no dataset, no pretrained/trained checkpoint, no test/inference/eval script, no FLOPs/params script, and no ablation driver; the README is a one-line title with no install/data/run instructions or results table."
concern: "Every value in Tables 1-5 and Figs 2/4 (PSNR/SSIM/LPIPS/FID/NIQE, FLOPs, params, runtime, α-sweep, ablations) is produced by an off-repo pipeline; none can be recomputed from this repository, so the headline state-of-the-art and efficiency claims are unverifiable."
resolution: "Release the UHD-LL/UHD-Blur/UHD-Haze/UHD-Snow/UHD-Rain (+GenDeg) data access, the trained checkpoints, an inference+metric script that prints the table numbers, the FLOPs/params script, and the ablation drivers, with exact commands in the README."
cross_refs: ["import-missing-test-module", "stage2-ravae-arch-unregistered", "stage2-restoration-network-rtheta-absent"]
check_script: _audit_code/check_import_and_registry.py
paper_ref: "Tables 1-5, Figures 2 and 4; Checklist Q5 'We will release the source code upon acceptance'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage2-restoration-network-rtheta-absent
category: missing
topic: "evaluation consistency (paper vs code)"
title: "Stage-2 latent restoration network Rθ and its LRes training (Eq. 7/9) not implemented"
severity: high
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 132
line_end: 139
quote: |
  def _forward_g(self):
        out = self.net_g(self.lq, alpha=self.alpha)
        if isinstance(out, (list, tuple)):
            if len(out) >= 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                return out[0], out[1], None
        return out, None, None
claim: "Stage-2 (`VAEadapter`) feeds the degraded input straight through the wrapped VAE (`RAVAEHFLora` -> RAVAE.forward), reconstructing lq and aligning its high-frequency band to gt's; there is no latent restoration network Rθ in the forward or optimization path. `restormer.py` defines a TransformerBlock but no model or config instantiates a restoration network, and nothing trains the paper's LRes (Eq. 7)."
concern: "The paper's Stage 2 (§4.2, Eqs. 7-9) centers on a latent restoration network Rθ that predicts a restored latent zres=Rθ(zdeg), trained with LRes and embedded inside the HF-LoRA GAN loss (Eq. 9, Dψ*+ΔψLoRA(Rθ(Eφ*(Ideg)))); the implemented Stage 2 omits Rθ entirely, so the released code does not realize the described restoration method that produced the paper's numbers."
resolution: "Provide the Rθ definition, the script that trains it with LRes (Eq. 7), and the HF-LoRA optimization wired through Rθ as in Eq. 9; confirm whether the paper numbers were produced with Rθ in the loop."
cross_refs: ["repro-pipeline-and-data-absent"]
paper_ref: "§4.2 Stage Two, Eq. (7) and Eq. (9)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dependencies-unspecified
category: missing
topic: "dependencies / environment"
title: "No requirements/environment file; heavy unpinned deps (torch, cv2, pyiqa, einops, DINOv2 hub)"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "There is no requirements.txt, environment.yml, setup.py, or pyproject.toml. The code imports torch, torchvision, cv2, pyiqa, einops, scipy, tqdm, yaml and dynamically pulls DINOv2 via torch.hub.load('facebookresearch/dinov2', ...) with a local weight path; none of these versions are pinned or listed."
concern: "The environment cannot be rebuilt deterministically; metric/back-end version drift (e.g. pyiqa LPIPS/FID, OpenCV, DINOv2) directly changes reported numbers."
resolution: "Add a pinned requirements.txt / environment.yml listing torch, torchvision, opencv-python, pyiqa, einops, scipy, and the exact DINOv2 weights/version, plus CUDA/cuDNN versions."
cross_refs: ["repro-pipeline-and-data-absent"]
check_script: _audit_code/check_import_and_registry.py
paper_ref: "Checklist Q5 (code release)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dino-and-vae-weights-missing
category: missing
topic: "pretrained weights / external artefacts"
title: "Stage-1 needs DINOv2 weights, Stage-2 needs a Stage-1 VAE checkpoint; neither is present or fetched"
severity: medium
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 35
line_end: 37
quote: |
  network_g:
    type: RAVAEHFLora
    pretrain_vae_path: ./weights/stage1_eqvae.pth
claim: "Stage-2 config requires `./weights/stage1_eqvae.pth` and Stage-1 requires `./weights/dinov2_vits14.pth` (stage1_eqvae.yml dino.local_weight_path), with `_init_dino_backbone` raising ValueError if the DINO path is empty. No `weights/` directory or download script exists in the repo."
concern: "Both stages depend on external checkpoints that are neither bundled nor fetchable from the repo; Stage 1 hard-fails without the DINOv2 weight and Stage 2 silently `load_state_dict(..., strict=False)` against a missing file path, so neither stage can be reproduced as shipped."
resolution: "Provide the trained Stage-1 VAE checkpoint and a script/URL to obtain the exact DINOv2-ViT-S/14 weights, or document the precise download steps."
cross_refs: ["repro-pipeline-and-data-absent"]
paper_ref: "Stage 1 LInv (DINOv2 semantic features); Stage 2 pretrain_vae_path"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: import-missing-test-module
category: bug
topic: "import / packaging"
title: "basicsr/__init__.py imports `.test` but basicsr/test.py does not exist — import crashes"
severity: high
confidence: high
status: finding
file: basicsr/__init__.py
line_start: 7
line_end: 7
quote: |
  from .test import *
claim: "`basicsr/__init__.py` executes `from .test import *`, but there is no `basicsr/test.py` in the repo (only `__init__.py` and `train.py` at the package root). Any `import basicsr` — including `basicsr/train.py`'s `from basicsr.data import ...`, which triggers the package `__init__` — raises ModuleNotFoundError: No module named 'basicsr.test'."
concern: "The training entrypoint cannot be imported/run at all; the framework is broken on first import before any data or config is involved."
resolution: "Either add the missing `basicsr/test.py` (the standard BasicSR inference entrypoint, which would also supply the absent evaluation pipeline) or remove the `from .test import *` line."
cross_refs: ["repro-pipeline-and-data-absent"]
check_script: _audit_code/check_import_and_registry.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage2-ravae-arch-unregistered
category: bug
topic: "registry / configuration"
title: "Stage-2 requests VAE arch `RAVAE` (unregistered); RAVAEHFLora raises KeyError"
severity: high
confidence: high
status: finding
file: basicsr/archs/LHVAE_hflora_arch.py
line_start: 36
line_end: 38
quote: |
  cfg = deepcopy(vae_config)
        arch_type = cfg.pop('type')
        self.vae = ARCH_REGISTRY.get(arch_type)(**cfg)
claim: "`stage2_hflora.yml` sets `network_g.vae_config.type: RAVAE`, and `RAVAEHFLora.__init__` calls `ARCH_REGISTRY.get('RAVAE')`. AST scan of basicsr/archs shows the registered arch classes are {LoRAConv2d, RAVAEHFLora, RAVAE_EQ, UNetDiscriminatorSN} — no class named `RAVAE`. `Registry.get` raises `KeyError(\"No object named 'RAVAE' found in 'arch' registry!\")` when the name is absent."
concern: "Stage-2 model construction crashes before training begins; the only Stage-1 VAE that exists is `RAVAE_EQ`, so the config type is simply wrong (or the `RAVAE` class was never committed), making the entire HF-LoRA stage unrunnable as configured."
resolution: "Register/define the `RAVAE` class, or change the config's `vae_config.type` to the actually-registered `RAVAE_EQ` (and confirm its keyword args match)."
cross_refs: ["repro-pipeline-and-data-absent", "import-missing-test-module"]
check_script: _audit_code/check_import_and_registry.py
paper_ref: "n/a (configs/stage2_hflora.yml)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-experiments-root-path
category: bug
topic: "hardcoded absolute paths"
title: "experiments_root hardcoded to authors' cluster path; breaks training for anyone else"
severity: medium
confidence: high
status: finding
file: basicsr/utils/options.py
line_start: 158
line_end: 164
quote: |
  if is_train:
        experiments_root = osp.join('/fs-computility/ai4sData/liuyidi/model', 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
claim: "On the training path, the experiment output root is unconditionally set to the absolute path `/fs-computility/ai4sData/liuyidi/model/experiments/<name>`, ignoring `root_path`. Two more absolute paths exist: `sys.path.append('/fs-computility/ai4sData/liuyidi/code/LatentGen')` in basicsr/train.py:8 and basicsr/archs/LHVAE_arch.py:7."
concern: "Any user without write access to `/fs-computility/...` cannot run training (checkpoint/log/visualization writes fail), and the hardcoded sys.path entries point to a private machine; this confirms the repo was not prepared for external reproduction."
resolution: "Derive experiments_root from `root_path`/`opt['name']` (as upstream BasicSR does) and remove the absolute `sys.path.append` lines."
cross_refs: []
check_script: _audit_code/check_import_and_registry.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: stage2-inference-alpha-blends-lora-output-not-weights
category: difference
topic: "inference-time control (α)"
title: "α blends LoRA output scale at runtime; paper describes blending LoRA weight deltas"
severity: low
confidence: medium
status: finding
file: basicsr/archs/LHVAE_hflora_arch.py
line_start: 83
line_end: 91
quote: |
  def set_alpha(self, alpha):
        alpha = float(max(0.0, min(1.0, alpha)))
        self.alpha = alpha
        enc_scale = alpha
        dec_scale = 1.0 - alpha
        for layer in self.enc_lora_layers:
            layer.set_runtime_scale(enc_scale)
        for layer in self.dec_lora_layers:
            layer.set_runtime_scale(dec_scale)
claim: "At inference, α scales the encoder-LoRA additive output by α and the decoder-LoRA additive output by (1-α) per forward (`out = base + lora_out * scale*runtime_scale`, lora_arch.py:81-86)."
concern: "The paper's Eq./§4.3 writes the control as a blend of weight deltas ϕ=ϕ*+α·Δϕ_LoRA and ψ=ψ*+(1-α)·Δψ_LoRA; scaling the LoRA *output* per-layer is mathematically equivalent to scaling Δ for a single linear LoRA layer, so this is a benign reformulation, but the two are described differently and only the code version is exercised."
resolution: "Confirm the inference α-control used for the paper's α-sweep (Table 5d) matches this runtime-scaling implementation; both are valid for linear LoRA, so this is a low-severity faithfulness note."
cross_refs: ["stage2-restoration-network-rtheta-absent"]
paper_ref: "§4.3 Inference-Time Control"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No standalone methodology finding could be grounded in the *runnable* code: the Stage-2
procedure the paper relies on (latent restoration network Rθ + HF-LoRA over Rθ) is not
present (filed as `missing`), and the Stage-1/Stage-2 trainers that *are* present do not,
on inspection, implement an invalid procedure (no train/test leakage is observable because
no evaluation/split code exists in the repo at all). Per Rule B / Rule L, I do not invent a
methodology finding; the dominant issues are absence (missing) and brokenness (bug).

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                   |
|-------------|------------|--------------|-------------------------------------------------------------------|
| missing     | 4          | high         | No data/weights/inference/metric/ablation code; Rθ + LRes absent. |
| bug         | 3          | high         | `import basicsr` crashes (.test); Stage-2 `RAVAE` unregistered; hardcoded cluster path. |
| difference  | 1          | low          | α-control scales LoRA output vs paper's weight-delta blend (equivalent). |
| methodology | 0          | -            | No invalid procedure groundable in the runnable code (see note).  |

## Top take-aways (≤6, ranked by severity × confidence)

1. **[bug] `import basicsr` crashes** — `basicsr/__init__.py:7` imports a non-existent `.test`
   module, so `basicsr/train.py` cannot even be imported (`import-missing-test-module`).
2. **[bug] Stage-2 unrunnable** — config asks for VAE `type: RAVAE`, but only `RAVAE_EQ` is
   registered; `RAVAEHFLora` raises `KeyError` (`stage2-ravae-arch-unregistered`).
3. **[missing] No reproducible pipeline** — no datasets, no checkpoints, no inference/metric
   script, no FLOPs/ablation code; not one number in Tables 1-5 / Figs 2,4 traces to repo
   code (`repro-pipeline-and-data-absent`).
4. **[missing] Restoration network Rθ absent** — the paper's Stage-2 core (Rθ + LRes, Eqs. 7/9)
   is not implemented; Stage-2 trains only VAE LoRA on lq→recon HF alignment
   (`stage2-restoration-network-rtheta-absent`).
5. **[bug] Hardcoded cluster path** — `options.py:158-164` writes all training outputs to
   `/fs-computility/ai4sData/liuyidi/...` unconditionally (`hardcoded-experiments-root-path`).
6. **[missing] Unspecified environment & missing weights** — no requirements file; Stage-1
   DINOv2 weight and Stage-2 VAE checkpoint are required but absent
   (`dependencies-unspecified`, `dino-and-vae-weights-missing`).

## Items that genuinely look fine

- The registry/auto-import machinery (`basicsr/utils/registry.py`, `archs/__init__.py`) is
  correct; `RAVAE_EQ`, `RAVAEHFLora`, `UNetDiscriminatorSN`, `LoRAConv2d`, the three models,
  and the losses register as expected (verified by AST scan).
- `LoRAConv2d` (lora_arch.py) is internally consistent: zero-init `lora_up` (LoRA starts as a
  no-op), correct rank/alpha scaling, and a valid fused-`_delta_weight` einsum.
- The Stage-1 PDPS / KL / DINO-LInv / EQv loss assembly in `EQVAEModel.optimize_parameters`
  is internally coherent (progressive degradation severity ramps with t; KL fallback present;
  eqv scale-factor guarded to (0,1)).

## Open questions for the authors

- Were the paper's reported numbers produced with the restoration network Rθ in the loop
  (Eq. 7/9)? If so, which script trains/evaluates it, and how does it connect to the released
  `VAEadapter` Stage-2 trainer? (`stage2-restoration-network-rtheta-absent`)
- Is `configs/stage2_hflora.yml`'s `vae_config.type: RAVAE` a typo for `RAVAE_EQ`, or was a
  separate `RAVAE` class omitted from the release? (`stage2-ravae-arch-unregistered`)
- Does the α-sweep in Table 5(d) use the runtime-output-scaling α implemented here, or the
  weight-delta blend written in §4.3? (`stage2-inference-alpha-blends-lora-output-not-weights`)
