# Audit — Latent Harmony (paper 1333)

## 1. Summary

The paper proposes **Latent Harmony (LH)**, a two-stage VAE framework for UHD all-in-one
image restoration: Stage 1 trains an **LH-VAE** (`RAVAE_EQ`) with KL + L1 reconstruction +
a DINOv2-based degradation-invariance loss `LInv` (Eq. 4), a latent-equivariance loss
`LEqv` (Eq. 5), and a progressive degradation perturbation strategy (PDPS, Eq. 3); Stage 2
trains a **latent restoration network `R_θ`** (Eq. 7) and then **HF-LoRA** adapters
(FHF-LoRA on the encoder, PHF-LoRA on the decoder via a GAN loss, Eqs. 8–9) with an
inference parameter α. The headline results are PSNR/SSIM/LPIPS state-of-the-art numbers on
UHD benchmarks (Tables 1–2), standard-resolution generalization (Tables 3–4), and ablations
(Table 5).

The cloned repo `code/lyd-2022__Latent-Harmony/` is a BasicSR-derived tree containing the
VAE arch (`LHVAE_arch.py`), a LoRA-wrapped VAE (`LHVAE_hflora_arch.py`), a discriminator,
two trainer models (`EQvae_model.py` for Stage 1, `VAEadapter_model.py` for Stage 2), losses,
data utilities, and two example YAML configs. The README is a single line; the NeurIPS
checklist (Q5) states the code/models will be released only "upon acceptance."

What I did: read the paper text + the full `code/` tree; mapped configs to registered
arch/model classes; traced the Stage-1 and Stage-2 training/forward paths; and ran one
read-only static check under `_audit_code/` (`check_registry_and_rtheta.py`) confirming
(a) the Stage-2 config references an arch `RAVAE` that is not registered, and (b) no
standalone restoration network `R_θ` exists anywhere and `RAVAEHFLora.forward` only calls
the VAE. I did **not** run training/inference (datasets, DINOv2 weights, and the Stage-1
checkpoint are all absent, and no eval entrypoint exists).

The repo is best characterised as a **partial training-code drop**: the restoration network
`R_θ` — the component that actually produces the paper's restoration numbers — is absent, no
evaluation/inference code exists, no checkpoints or datasets are shipped, dependencies are
unpinned, the Stage-2 config crashes on an unregistered arch name, and `train.py` hardcodes
an author-machine absolute path. None of the quantitative artefacts in the paper can be
reproduced from this repo.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (4-deg) PSNR/SSIM/LPIPS, "Ours" 29.70/.877/.2502 | (none — no eval script; configs set `metrics: ~`) | — | — | MISSING |
| Table 2 (6-deg) "Ours" 29.24/.920/.1822 | (none) | — | — | MISSING |
| Table 3 standard-res LPIPS/FID (PromptIR/Diff-Plugin/CosAE /w Ours) | (none — no integration code for these backbones) | — | — | MISSING |
| Table 4 generalization PSNR/SSIM/LPIPS (unseen/composite) | (none) | — | — | MISSING |
| Table 5(a) ablations (w/o LInv, LEqv, PDPS, FHF/PHF-LoRA, …) | (no ablation harness; some toggles exist via λ=0) | — | — | MISSING |
| Table 5(b) inference time (LH 0.43s) | (none) | — | — | MISSING |
| Table 5(c) Restormer/NAFNet/SFHformer Base vs +Ours | (none — those restoration nets are not in repo) | — | — | MISSING |
| Table 5(d) α-sweep PSNR/SSIM/LPIPS/User | (no α-sweep eval; α used only as LoRA scale at train) | — | — | MISSING |
| Fig. 2 motivation (t-SNE/CDCS/DCT/loss curves) | (none) | — | — | MISSING |
| Eq. 7 latent restoration network `R_θ` (`z_res = R_θ(z_deg)`) | (none) | — | — | MISSING (see missing-restoration-network-rtheta) |
| Eq. 2 `LVAE` (L1 recon + KL) | `EQvae_model.py:318-328` | implemented | ✓ | Verified (logic present) |
| Eq. 4 `LInv` (DINOv2 align of perturbed latent to clean) | `EQvae_model.py:270-281` | implemented | ✓ (with extra learned 1×1 projector) | Verified |
| Eq. 5 `LEqv` (`D(z_down)` vs `I_down`, z from clean) | `EQvae_model.py:283-293` | uses latent of perturbed input, not clean | ✗ | MISMATCH (see eqv-latent-from-perturbed) |
| Eq. 3 PDPS perturbation | `EQvae_model.py:186-224` | implemented | ✓ | Verified (logic present) |
| Eqs. 8–9 HF-LoRA / HF-GAN over `R_θ` output | `VAEadapter_model.py:148-210` | operates on VAE recon of `lq`, no `R_θ` | ✗ | MISMATCH/MISSING |

## 3. Findings

## missing

```yaml finding
id: missing-restoration-network-rtheta
category: missing
topic: "result traceability / core method"
title: "Latent restoration network R_theta (Eq. 7) is absent from the repo"
severity: high
confidence: high
status: finding
file: basicsr/archs/LHVAE_hflora_arch.py
line_start: 122
line_end: 133
quote: |
  def forward(self, x, alpha=None):
        if alpha is not None:
            self.set_alpha(alpha)

        out = self.vae(x)
        if isinstance(out, (list, tuple)):
            if len(out) >= 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                z, posterior = self.vae.encode(x, self.vae.sample_posterior)
                return out[0], out[1], z
        raise RuntimeError('Unexpected output format from wrapped VAE.')
claim: "Stage 2 (VAEadapter) wraps only the VAE+LoRA (RAVAEHFLora); its forward decodes the VAE reconstruction of the degraded input and never invokes the latent restoration network R_theta that the paper defines as z_res = R_theta(z_deg) (Eq. 7) and trains with L_Res. No SFHformer/NAFNet/Restormer-as-restorer or any 'restoration net' class exists in the repo (verified by AST/text scan in _audit_code/out/registry_and_rtheta.txt)."
concern: "All of the paper's restoration numbers (Tables 1-5) are produced by the Decoder(R_theta(Encoder(I_deg))) pipeline; with R_theta absent, the pipeline that yields every headline metric cannot be reproduced, and Stage 2 as shipped only reconstructs the degraded image."
resolution: "Authors: please release the R_theta architecture and the Stage-2 restoration-network training code (Eq. 7), and wire R_theta into the Stage-2 forward path."
cross_refs: ["§4.2", "stage2-no-eval-numbers"]
check_script: _audit_code/check_registry_and_rtheta.py
paper_ref: "Section 4.2, Eq. 7; Tables 1-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-eval-inference-entrypoint
category: missing
topic: "result traceability / evaluation code"
title: "No evaluation/inference script; configs disable metrics, so no Table number is computed"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 120
line_end: 125
quote: |
  val:
    val_freq: 1000
    save_img: false
    pbar: true
    suffix: ~
    metrics: ~
check_script: _audit_code/check_registry_and_rtheta.py
claim: "Only basicsr/train.py is an executable entrypoint; there is no test/inference/eval script. Both example configs set val.metrics: ~, so VAEadapter.nondist_validation / EQVAEModel.nondist_validation compute no PSNR/SSIM/LPIPS (with_metrics is False). No script computes the values reported in Tables 1-5 or the FID/LPIPS in Tables 3-4."
concern: "Every quantitative artefact in the paper is untraceable to code: nothing in the repo produces the reported PSNR/SSIM/LPIPS/FID/runtime numbers."
resolution: "Authors: provide the evaluation entrypoint, the metric configuration used (PSNR/SSIM/LPIPS via pyiqa), and the exact commands and dataset paths to reproduce each table."
cross_refs: ["missing-restoration-network-rtheta"]
paper_ref: "Tables 1-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-and-backbone-experiments-absent
category: missing
topic: "ablations / cross-architecture experiments"
title: "Table 5(c) backbones (Restormer/NAFNet/SFHformer +Ours) and Table 3 backbone integrations absent"
severity: high
confidence: high
status: finding
file: basicsr/archs/LHVAE_arch.py
line_start: 25
line_end: 25
quote: |
  from basicsr.archs.restormer import TransformerBlock
claim: "Restormer/NAFNet/SFHformer appear in the repo only as internal attention building blocks (e.g. restormer.TransformerBlock used inside the VAE encoder/decoder), not as the swappable latent restoration network compared in Table 5(c); and PromptIR/Diff-Plugin/CosAE 'w/ Ours' integrations from Table 3 are not present. A repo-wide AST/text scan finds no SFHformer/NAFNet/Restormer restoration-net class (see _audit_code/out/registry_and_rtheta.txt)."
concern: "The robustness-across-architectures claim (Table 5c) and the standard-resolution integration claim (Table 3) are not backed by any code, so these experiments are unverifiable."
resolution: "Authors: provide the latent-restoration-network ablation harness (Restormer/NAFNet/SFHformer variants) and the PromptIR/Diff-Plugin/CosAE integration scripts used for Tables 3 and 5(c)."
cross_refs: ["missing-restoration-network-rtheta"]
check_script: _audit_code/check_registry_and_rtheta.py
paper_ref: "Table 3; Table 5(c)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-weights-and-data
category: missing
topic: "expected completeness / weights & data"
title: "No trained weights, no datasets, and required pretrained files are not shipped"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 33
line_end: 35
quote: |
  network_g:
    type: RAVAEHFLora
    pretrain_vae_path: ./weights/stage1_eqvae.pth
claim: "Stage 2 requires ./weights/stage1_eqvae.pth (Stage-1 VAE checkpoint) and Stage 1 requires ./weights/dinov2_vits14.pth (DINOv2; EQvae_model.py:111-116 raises if dino.local_weight_path is empty). Neither file, nor any model checkpoint, nor the UHD datasets (dataroot ./datasets/train/{gt,lq}) are present in the repo (only basicsr/metrics/niqe_pris_params.npz exists)."
concern: "Neither stage can be run end-to-end as shipped: Stage 1 needs external DINOv2 weights and the UHD paired datasets, Stage 2 additionally needs the Stage-1 checkpoint; none are provided or fetchable, blocking reproduction."
resolution: "Authors: release the Stage-1/Stage-2 checkpoints and a data-preparation script (or accessions/URLs) for the UHD-LL/UHD-blur/UHD-haze/UHD-rain/UHD-snow/UHDN datasets, and document DINOv2 weight acquisition."
cross_refs: []
paper_ref: "Section 4; NeurIPS checklist Q5/Q13"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: deps-unpinned-no-requirements
category: missing
topic: "dependencies / environment"
title: "No requirements/setup/environment file; dependencies unlisted and unpinned"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The repo has no requirements.txt/setup.py/environment.yml/pyproject.toml. Third-party imports include torch, torchvision, cv2, einops, numpy, scipy, PIL, pyiqa, tqdm, yaml (none version-pinned), plus a torch.hub fetch of DINOv2. The README is a single line with no install or run instructions."
concern: "The environment cannot be rebuilt deterministically, undermining reproducibility of any result even if the missing code/weights were supplied."
resolution: "Authors: add a pinned requirements file and a README with install commands, dataset layout, and the exact train/eval commands per table."
cross_refs: []
paper_ref: "NeurIPS checklist Q5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: ravae-type-not-registered
category: bug
topic: "config / arch registry"
title: "Stage-2 config references arch type 'RAVAE' that is not registered (only 'RAVAE_EQ')"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 44
quote: |
    vae_config:
      type: RAVAE
      embed_dim: 4
      sample_posterior: true
claim: "RAVAEHFLora.__init__ does ARCH_REGISTRY.get(cfg.pop('type'))(**cfg) (LHVAE_hflora_arch.py:37-38). The stage2 config sets vae_config.type: RAVAE, but no class named 'RAVAE' is registered in ARCH_REGISTRY (only 'RAVAE_EQ', 'RAVAEHFLora', 'LoRAConv2d', 'UNetDiscriminatorSN'; verified in _audit_code/out/registry_and_rtheta.txt). Registry.get raises KeyError on a missing name (registry.py:62-66)."
concern: "Running Stage 2 with the shipped config crashes at network construction with KeyError(\"No object named 'RAVAE' found in 'arch' registry!\"), so Stage-2 training cannot start as provided."
resolution: "Authors: register a 'RAVAE' arch (or change the config type to 'RAVAE_EQ' and confirm the embed_dim=4 base VAE matches the Stage-1 checkpoint)."
cross_refs: []
check_script: _audit_code/check_registry_and_rtheta.py
paper_ref: "Section 4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-author-syspath
category: bug
topic: "portability / hardcoded path"
title: "train.py prepends a hardcoded author-machine absolute path to sys.path"
severity: medium
confidence: high
status: finding
file: basicsr/train.py
line_start: 7
line_end: 8
quote: |
  import sys
  sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
claim: "The training entrypoint unconditionally appends an absolute cluster path (/fs-computility/ai4sData/liuyidi/code/LatentGen) to sys.path."
concern: "On any other machine this path does not exist; while sys.path.append of a missing dir does not itself crash, it signals the code depends on modules outside the released tree (LatentGen) and was run from an environment not reproducible from the repo."
resolution: "Authors: remove the hardcoded path and confirm all imported modules are inside the released repository (or list the external LatentGen package as a dependency)."
cross_refs: ["deps-unpinned-no-requirements"]
paper_ref: "N/A"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: stage2-no-eval-numbers
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Stage-2 trains FHF/PHF-LoRA on VAE reconstruction of lq, not on R_theta restoration as in Eqs. 8-9"
severity: high
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 132
line_end: 142
quote: |
    def _forward_g(self):
        out = self.net_g(self.lq, alpha=self.alpha)
        if isinstance(out, (list, tuple)):
            if len(out) >= 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                return out[0], out[1], None
        return out, None, None

    def _hf(self, x):
        return extract_hf_fft(x, cutoff_ratio=self.hf_cutoff_ratio)
claim: "Stage-2 recon = VAE+LoRA reconstruction of the degraded input lq (no restoration network). The HF fidelity loss compares HF(recon) to HF(gt) and the HF-GAN discriminates HF(recon) vs HF(gt). The paper's Eq. 8 uses D(E(I_deg)) but Eq. 9 explicitly puts R_theta in the loop: HF(D(R_theta(E(I_deg)))). The code's HF-GAN therefore omits R_theta present in the paper's equation."
concern: "What the code optimizes (HF alignment of a plain VAE autoencoding) differs from the paper's described Stage-2 objective that operates on the restored latent z_res = R_theta(z_deg); the implemented procedure cannot produce the paper's restoration outputs."
resolution: "Authors: confirm whether the released Stage-2 forward should route through R_theta (Eq. 7/9); if so, the restoration network and its wiring are missing (see missing-restoration-network-rtheta)."
cross_refs: ["missing-restoration-network-rtheta", "no-eval-inference-entrypoint"]
paper_ref: "Section 4.2, Eqs. 8-9"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eqv-latent-from-perturbed
category: difference
topic: "loss definition (paper vs code)"
title: "LEqv decodes the latent of the PERTURBED input, not the clean-image latent in Eq. 5"
severity: low
confidence: high
status: finding
file: basicsr/models/EQvae_model.py
line_start: 283
line_end: 293
quote: |
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
claim: "Paper Eq. 5 defines LEqv = ||D(Downs(z_clean)) - Downs(I_clean)||_1 with z_clean from the clean image. In optimize_parameters (lines 298-330) z is the latent of the PDPS-perturbed input (self.net_g(self.pdps_input)), and that perturbed-image latent is passed to _compute_eqv_loss, while the target gt_down is the downsampled clean image."
concern: "The equivariance constraint is applied to the perturbed-image latent rather than the clean-image latent the equation specifies; the implemented variant is a defensible (degradation-robust) alternative but does not match Eq. 5."
resolution: "Authors: clarify whether LEqv should use the clean-image latent (Eq. 5) or the perturbed-input latent as implemented; align code or paper accordingly."
cross_refs: []
paper_ref: "Section 4.1, Eq. 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

The paper's LInv (Eq. 4) is `d(z'_deg, f_VFM)` with no learned projection; the code
interpolates `z` to the DINO token grid and, when channel counts differ (latent is 4-ch,
DINOv2-small is 384-ch), inserts a trainable 1×1 conv that is jointly optimized. Because I
cannot confirm from the repo alone whether this projector was part of the reported runs, this
is filed as a `question`.

```yaml finding
id: linv-extra-learned-projector
category: difference
topic: "loss definition (paper vs code)"
title: "LInv adds an undocumented trainable 1x1 conv projector before DINOv2 alignment"
severity: low
confidence: medium
status: question
file: basicsr/models/EQvae_model.py
line_start: 144
line_end: 153
quote: |
    def _ensure_inv_projector(self, in_ch, out_ch):
        if in_ch == out_ch:
            return None
        if self.inv_projector is not None:
            if self.inv_projector.in_channels == in_ch and self.inv_projector.out_channels == out_ch:
                return self.inv_projector
        self.inv_projector = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0).to(self.device)
        if hasattr(self, 'optimizer_g'):
            self.optimizer_g.add_param_group({'params': self.inv_projector.parameters()})
        return self.inv_projector
claim: "LInv aligns the encoder latent to DINOv2 features through a learned 1x1 conv projector added on the fly and optimized with optimizer_g (EQvae_model.py:270-281, 144-153); the paper states a direct feature-space distance d(z'_deg, f_VFM) with no projector."
concern: "A learned projector can absorb the distribution gap, weakening the intended degradation-invariance constraint and adding an undescribed result-affecting component (paper omission)."
resolution: "Authors: confirm whether a learned projection layer was part of LInv in the reported runs; if so, describe it; if not, remove it."
cross_refs: []
paper_ref: "Section 4.1, Eq. 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

N/A for a standalone methodology finding: the procedures that would be the subject of
methodology scrutiny (the restoration pipeline `R_θ` and the evaluation harness) are not
present in the repo, so the dominant defects are `missing`/`bug`/`difference`. No additional
independent methodology defect was verifiable in the code that *is* present (seeding is
comprehensive: `random`/`numpy`/`torch`/`cuda` are all seeded in `utils/misc.py:12-18`).

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | R_theta restoration net, all eval/ablation code, weights/data, deps all absent |
| bug         | 2          | high         | Stage-2 config uses unregistered arch 'RAVAE' (KeyError); hardcoded author sys.path |
| difference  | 3          | high         | Stage-2 optimizes VAE recon w/o R_theta; LEqv uses perturbed latent; LInv adds learned projector |
| methodology | 0          | -            | No independently verifiable methodology defect in the present code |

## Top take-aways (≤6, ranked by severity × confidence)

1. **[missing] `missing-restoration-network-rtheta`** — The latent restoration network `R_θ`
   (Eq. 7), the core of every reported restoration number, is absent; Stage 2 only
   autoencodes the degraded input. (high/high)
2. **[missing] `no-eval-inference-entrypoint`** — No evaluation/inference script and
   `metrics: ~` in both configs ⇒ no code computes any Table 1–5 / FID / runtime number.
   (high/high)
3. **[bug] `ravae-type-not-registered`** — Stage-2 config references arch `RAVAE`, which is
   not registered (only `RAVAE_EQ`); Stage-2 build raises `KeyError`. (high/high)
4. **[missing] `ablation-and-backbone-experiments-absent`** — Table 3 backbone integrations
   and Table 5(c) Restormer/NAFNet/SFHformer comparisons have no code. (high/high)
5. **[difference] `stage2-no-eval-numbers`** — The implemented Stage-2 HF-LoRA/HF-GAN runs on
   a plain VAE reconstruction, diverging from Eqs. 8–9 which route through `R_θ`. (high/high)
6. **[missing] `missing-weights-and-data`** — No checkpoints, no UHD datasets, and required
   DINOv2 / Stage-1 weights are not shipped or fetchable. (high/high)

## Items that genuinely look fine

- **Random seeding** is comprehensive: `utils/misc.py:12-18` seeds `random`, `numpy`,
  `torch`, and CUDA; `options.py:109-114` offsets by rank.
- **Registry mechanism** and auto-import of `*_arch.py` / model / loss modules are correct;
  `RAVAE_EQ`, `RAVAEHFLora`, `UNetDiscriminatorSN`, `KlLoss`, `GANLoss`, `L1Loss` are all
  present and registered.
- **PDPS (Eq. 3)** is faithfully implemented with the three branches and a monotonic
  schedule (`EQvae_model.py:186-224`).
- **HF extraction** (`utils/hf_ops.py`) is a sound FFT high-pass; the alternating
  FHF/PHF-LoRA schedule and selective `requires_grad` freezing match the paper's
  alternating-optimization description.

## Open questions for the authors

- Was the released `VAEadapter` Stage-2 ever intended to include `R_θ`, or is the LoRA-only
  autoencoding the actual Stage-2 path? (Determines whether `missing-restoration-network-rtheta`
  is a code-release gap or a deeper mismatch.)
- Which arch should `vae_config.type` be in `stage2_hflora.yml` (`RAVAE` vs `RAVAE_EQ`), and
  does its `embed_dim`/`ddconfig` match the Stage-1 checkpoint state dict?
- Was a learned projector part of `LInv` in the reported runs (`linv-extra-learned-projector`)?
