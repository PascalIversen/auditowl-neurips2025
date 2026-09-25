# Audit — Latent Harmony: Synergistic Unified UHD Image Restoration (paper 1333)

## Summary

The paper proposes "Latent Harmony", a two-stage VAE framework for UHD all-in-one image
restoration: Stage 1 trains an LH-VAE with progressive degradation perturbation (PDPS), a
DINOv2 semantic-invariance loss (L_Inv) and a latent equivariance loss (L_Eqv); Stage 2
trains a latent restoration network R_theta with a frozen VAE (L_Res) and then applies
high-frequency-guided LoRA (FHF-LoRA on the encoder for fidelity, PHF-LoRA on the decoder
for perception), with an inference parameter α blending the two. Tables 1–5 report PSNR/
SSIM/LPIPS/FID, FLOPs, params, runtime and an α-sweep against ~11 baselines.

The cloned repo (`code/lyd-2022__Latent-Harmony/`) is a BasicSR-derived **training-only
skeleton**. It contains a Stage-1 trainer (`EQvae_model.py`) and a Stage-2 trainer
(`VAEadapter_model.py`), VAE/LoRA architectures, losses, and two example configs. It
contains **no** evaluation/inference entrypoint, **no** dependency specification, an empty
(`# Latent-Harmony`) README, **no** pretrained weights, and **no** data. The NeurIPS
checklist (Q5) states "We will release the source code upon acceptance" — consistent with
this being a partial/placeholder release rather than the code that produced the paper's
numbers. Several architecture modules are explicit "placeholder/substitute/fallback"
re-implementations.

What I ran (read-only, under `_audit_code/`, outputs in `_audit_code/out/`):
- `check_ravae_registration.py` — AST scan of auto-imported `*_arch.py` files: arch type
  `RAVAE` requested by `configs/stage2_hflora.yml` is **not** registered (only `RAVAE_EQ`,
  `RAVAEHFLora`, `UNetDiscriminatorSN`, `LoRAConv2d`).
- `check_stage2_no_restoration_net.py` — confirms Stage-2 has no R_theta / L_Res; `net_g`
  is the VAE itself, optimized only by HF fidelity + GAN losses.
- `check_completeness.py` — confirms no eval entrypoint / no `test.py`, no requirements,
  empty README, no weights, no data files.

## Result-traceability table

Every quantitative artefact in the paper depends on (a) trained weights, (b) the
benchmark data, and (c) an evaluation script that computes the metrics. None of the three
exists in the repo, and the default configs set `val.metrics: ~` (no metric computation).

| Paper artefact (value)                                   | Repo location for the computed value | Computed? | Matches | Status |
|----------------------------------------------------------|--------------------------------------|-----------|---------|--------|
| Table 1 (4-deg): Ours PSNR avg 29.70 / SSIM .877 / LPIPS .2502 | (none — no eval script, no weights, no data) | — | — | MISSING |
| Table 2 (6-deg): Ours PSNR avg 29.24 / SSIM .920 / LPIPS .1822 | (none) | — | — | MISSING |
| Table 3 (std-res LPIPS/FID, +Ours rows)                  | (none) | — | — | MISSING |
| Table 4 (generalization PSNR/SSIM/LPIPS)                 | (none) | — | — | MISSING |
| Table 5(a) ablation (L_Inv/L_Eqv/PDPS/LoRA rows)         | (none — no ablation harness)         | — | — | MISSING |
| Table 5(b) inference-time (LH 0.43 s)                    | (none)                               | — | — | MISSING |
| Table 5(c) backbone swap (Restormer/NAFNet/SFHformer)    | (none; arch is a "lightweight" stub) | — | — | MISSING |
| Table 5(d) α-sweep (PSNR/SSIM/LPIPS/User)                | (none)                               | — | — | MISSING |
| FLOPs/Params (3.6G / 1.2M)                                | only `thop` profile in `LHVAE_arch.py:__main__` (debug, hardcoded path) | — | — | MISSING |
| Fig. 2 motivation (CDCS/t-SNE/DCT/loss curves)           | (none)                               | — | — | MISSING |

All rows are MISSING → routed to `missing` finding `no-eval-or-results-pipeline`.

## missing

```yaml finding
id: no-eval-or-results-pipeline
category: missing
topic: "result traceability"
title: "No evaluation/inference pipeline, weights, or data — no paper number is reproducible"
severity: high
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The repo has only a Stage-1 and Stage-2 training entrypoint (basicsr/train.py). There is no basicsr/test.py or any inference/metric-computing script, no requirements file, no pretrained weights (*.pth/*.ckpt), and no benchmark data; the README is a single title line. Default configs set val.metrics: ~ so even validation computes no metrics."
concern: "Every PSNR/SSIM/LPIPS/FID/FLOPs/runtime value in Tables 1–5 and every Fig. 2 motivation curve is untraceable: nothing in the repo computes the reported numbers, and the trained models and benchmark datasets needed to do so are absent."
resolution: "Authors: provide the evaluation/inference scripts that produce each table, the trained checkpoints, the dataset access (or fetch scripts), and a requirements/environment file with the exact commands per table."
cross_refs: ["no-dependency-spec", "repo-is-placeholder-reimplementation"]
check_script: _audit_code/check_completeness.py
paper_ref: "Tables 1–5; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage2-missing-restoration-network
category: missing
topic: "stage 2 method"
title: "Paper's latent restoration network R_theta and L_Res step are absent from Stage 2"
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
claim: "The Stage-2 network RAVAEHFLora wraps only the VAE (self.vae) plus LoRA adapters; forward() just runs the VAE on the degraded input. There is no latent restoration network R_theta, and the Stage-2 trainer (VAEadapter_model.py) never trains an R_theta nor uses a main restoration loss L_Res — its only losses are the HF fidelity L1 and the HF-GAN loss."
concern: "The paper's entire Stage 2 (Eq. 7: z_res = R_theta(z_deg), L_Res = ||D_psi*(z_res) - I_clean||_1, trained first with a frozen VAE, then HF-LoRA refinement) is the core of the method and the source of the restoration results; the code implements only a VAE that maps the degraded image to itself through LoRA-adapted encoder/decoder, so the released code cannot reproduce the restoration described."
resolution: "Authors: provide the R_theta restoration-network definition, the L_Res pre-training step, and confirm how z_res enters the decoder; clarify whether the released Stage-2 code is complete."
cross_refs: ["no-eval-or-results-pipeline", "stage2-ravae-arch-missing"]
check_script: _audit_code/check_stage2_no_restoration_net.py
paper_ref: "Section 4.2, Eq. (7)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec
category: missing
topic: "environment / dependencies"
title: "No requirements/environment file though many third-party deps are imported"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 8
line_end: 8
quote: |
  import pyiqa
claim: "The repo imports torch, torchvision, pyiqa, einops, cv2, numpy, scipy, PIL, tqdm and yaml at module load, but contains no requirements.txt, setup.py, environment.yml or any other dependency specification (confirmed by check_completeness.py)."
concern: "The environment cannot be rebuilt deterministically; in particular unpinned pyiqa/torch/torchvision versions affect the LPIPS/metric values the paper reports."
resolution: "Authors: add a pinned requirements.txt or environment.yml listing all dependencies and versions."
cross_refs: ["no-eval-or-results-pipeline"]
check_script: _audit_code/check_completeness.py
paper_ref: "NeurIPS checklist Q5 (code release)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: repo-is-placeholder-reimplementation
category: missing
topic: "repository provenance"
title: "Key architecture modules are placeholder/substitute stubs, not the paper's components"
severity: high
confidence: medium
status: finding
file: basicsr/archs/vgg_arch.py
line_start: 4
line_end: 8
quote: |
  class VGGFeatureExtractor(nn.Module):
      """Fallback VGG feature extractor placeholder.

      Returns a dict with requested layer names, each mapped to the input itself.
      """
claim: "Several components are explicitly self-described as stand-ins: VGGFeatureExtractor 'returns the input itself' (vgg_arch.py:4-8); restormer.TransformerBlock is a 'Lightweight spatial transformer-style block' (restormer.py:5); fourmer.ProcessBlock is a 'Lightweight substitute for Fourmer process block' (fourmer.py:5). The paper states the latent restoration network is SFHformer, which does not appear here."
concern: "If the perceptual/feature and restoration components are placeholders rather than the real modules, the repo is not the artefact that produced the paper's numbers, and the reported results (which depend on these exact modules) cannot be reproduced from it."
resolution: "Authors: confirm whether these are placeholders, and release the actual SFHformer restoration network and the real VGG/perceptual modules used for the reported metrics."
cross_refs: ["no-eval-or-results-pipeline", "stage2-missing-restoration-network"]
paper_ref: "Section 5.3 (SFHformer); Table 1/2 (LPIPS)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-ravae-arch-missing
category: bug
topic: "stage 2 config / arch registry"
title: "Stage-2 config requests unregistered arch 'RAVAE' — network build raises KeyError"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 42
quote: |
  vae_config:
    type: RAVAE
claim: "configs/stage2_hflora.yml sets vae_config.type: RAVAE, and RAVAEHFLora.__init__ does ARCH_REGISTRY.get(arch_type)(**cfg) (LHVAE_hflora_arch.py:38). The only arch names registered by the auto-imported *_arch.py modules are RAVAE_EQ, RAVAEHFLora, UNetDiscriminatorSN and LoRAConv2d (verified by AST scan); 'RAVAE' is never registered, so ARCH_REGISTRY.get('RAVAE') raises KeyError."
concern: "The Stage-2 training config cannot instantiate its network as written; Stage-2 training crashes at startup, so the provided config does not run."
resolution: "Authors: change vae_config.type to RAVAE_EQ (the registered class) or register a class named RAVAE, and confirm which VAE the Stage-2 results used."
cross_refs: ["stage2-missing-restoration-network"]
check_script: _audit_code/check_ravae_registration.py
paper_ref: "Section 4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-absolute-paths
category: bug
topic: "portability / hardcoded paths"
title: "Hardcoded author-machine absolute paths break runs on any other system"
severity: medium
confidence: high
status: finding
file: basicsr/utils/options.py
line_start: 158
line_end: 160
quote: |
  if is_train:
        experiments_root = osp.join('/fs-computility/ai4sData/liuyidi/model', 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
claim: "options.py:159 hardcodes the experiments root to '/fs-computility/ai4sData/liuyidi/model/...'. Additional hardcoded sys.path.append paths exist: train.py:8 and LHVAE_arch.py:7 ('/fs-computility/ai4sData/liuyidi/code/LatentGen'), attention.py:8 ('/code/UHDFour_code-main'), encoder_3.py:8 ('/code/UHDformer-main')."
concern: "Training writes checkpoints/logs to an absolute path that does not exist on a reviewer's machine, and the sys.path.append lines point at directories not in the repo; the code is not portable as released."
resolution: "Authors: make experiments_root relative to root_path (as the test branch already does) and remove the absolute sys.path.append calls."
cross_refs: []
check_script: _audit_code/check_completeness.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: phf-lora-uses-fidelity-loss
category: difference
topic: "stage 2 losses"
title: "Decoder/PHF-LoRA step also applies HF fidelity L1, not the perception-only loss of Eq. 9"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 190
line_end: 202
quote: |
            recon, posterior, z = self._forward_g()
            hf_pred = self._hf(recon)

            l_total = recon.new_tensor(0.0)
            l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
            l_total = l_total + l_hf_fid
            loss_dict['l_hf_fid'] = l_hf_fid

            if self.net_d_hf is not None and self.cri_gan is not None and self.lambda_gan > 0:
                pred_fake_g = self.net_d_hf(hf_pred)
                l_hf_gan = self.cri_gan(pred_fake_g, True, is_disc=False) * self.lambda_gan
                l_total = l_total + l_hf_gan
claim: "In the PHF-LoRA (decoder) optimization step the loss is l_hf_fid (HF fidelity L1, the same loss used for the encoder FHF-LoRA step) plus l_hf_gan; the decoder LoRA is updated by both."
concern: "The paper states PHF-LoRA is guided by a perception-oriented high-frequency loss implemented as a GAN loss (Eq. 9) only, decoupled from the fidelity loss; including the fidelity L1 in the decoder step contradicts the described 'differentiated mechanisms' and the perception-vs-fidelity decoupling that motivates the α control."
resolution: "Authors: clarify whether the decoder LoRA was trained with the fidelity L1 in addition to the GAN loss, and reconcile with Eq. 9 / Section 3.3."
cross_refs: ["stage2-missing-restoration-network"]
paper_ref: "Section 4.2, Eq. (9); Section 3.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage1-recon-input-is-degraded
category: difference
topic: "stage 1 VAE loss"
title: "Stage-1 reconstruction encodes the perturbed image, not I_clean as in Eq. 2"
severity: low
confidence: high
status: finding
file: basicsr/models/EQvae_model.py
line_start: 298
line_end: 318
quote: |
        self.pdps_input, t_value, pdps_branch = self._build_pdps_input(current_iter)
        raw_out = self.net_g(self.pdps_input)
claim: "Stage-1 feeds the PDPS-perturbed/degraded image (pdps_input) through the VAE and computes the reconstruction loss l_recon = cri_pix(recon, gt) against the clean image (line 318)."
concern: "Eq. 2 writes L_VAE = ||D_psi(E_phi(I_clean)) - I_clean||_1 (encode the CLEAN image); the code reconstructs the clean target from the degraded input. This is a valid restoration-style objective but differs from the equation as written, so the paper's formula does not match the implemented loss."
resolution: "Authors: update Eq. 2 to reflect that the encoder input is the PDPS-perturbed image, or confirm which input the reported Stage-1 used."
cross_refs: []
paper_ref: "Section 4.1, Eq. (2)–(3)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding is supportable: because there is no runnable end-to-end pipeline,
no weights, and no data, I cannot execute the actual procedure (splits, leakage,
baselines, statistics) to judge its validity. The data uses standard BasicSR paired
folder loading with separate train/val directories (no in-code split to inspect), and the
α-blend and L_Eqv implementations match the paper. Methodological soundness of the
reported experiments cannot be assessed from this artefact.

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 4          | high         | No eval pipeline/weights/data; Stage-2 R_theta absent; placeholder archs |
| bug         | 2          | high         | Stage-2 config requests unregistered arch RAVAE (crash); hardcoded paths |
| difference  | 2          | medium       | PHF-LoRA includes fidelity L1 (Eq.9); Stage-1 recon input differs (Eq.2) |
| methodology | 0          | -            | Not assessable — no runnable end-to-end procedure to evaluate           |

## Top take-aways

1. (`missing`, high) No evaluation/inference pipeline, no trained weights, no data, empty
   README — **no number in Tables 1–5 or Fig. 2 is traceable** (`no-eval-or-results-pipeline`).
2. (`missing`, high) Stage 2's restoration network R_theta and L_Res pre-training (Eq. 7),
   the core of the method, are **entirely absent** (`stage2-missing-restoration-network`).
3. (`bug`, high) Stage-2 config requests unregistered arch `RAVAE` → KeyError at startup;
   the provided Stage-2 config does not run (`stage2-ravae-arch-missing`).
4. (`missing`, high/med) Multiple architecture modules are self-labelled placeholder/
   substitute stubs (VGG "returns the input itself"); the repo is likely not the artefact
   that produced the paper's numbers (`repo-is-placeholder-reimplementation`).
5. (`difference`, medium) Decoder/PHF-LoRA is trained with the HF fidelity L1 in addition
   to the GAN loss, contradicting the perception-only Eq. 9 (`phf-lora-uses-fidelity-loss`).
6. (`bug`/`missing`, medium) Hardcoded author-machine absolute paths and no dependency
   spec block portable execution (`hardcoded-absolute-paths`, `no-dependency-spec`).

## Items that genuinely look fine

- Stage-1 latent equivariance loss L_Eqv (EQvae_model.py:283-293) matches Eq. 5:
  decode(downsample(z)) vs downsample(I_clean), L1.
- Inference α-blend (LHVAE_hflora_arch.py:83-91): enc_scale = α, dec_scale = 1-α, matching
  ϕ = ϕ* + α∆ϕ, ψ = ψ* + (1-α)∆ψ in Section 4.3.
- LoRA implementation (lora_arch.py) is a correct rank-r down/up conv adapter with zero-init
  up-projection and a merge/unmerge path.
- PDPS three-branch mixture (EQvae_model.py:206-224) implements Eq. 3 (no-perturb / synth /
  interpolate) with a monotonically increasing schedule t = iter/total_iter.

## Open questions for the authors

- Is the released code the exact artefact used for the paper, or a partial/placeholder
  re-implementation? (Several modules are labelled "placeholder"/"substitute"/"fallback".)
- Where is the SFHformer latent restoration network (R_theta) and the Stage-2 L_Res step?
- Will the evaluation scripts, checkpoints, datasets, and a pinned environment be released?
