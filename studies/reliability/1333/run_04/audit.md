# Audit — Latent Harmony (paper 1333)

## 1. Summary

The paper proposes **Latent Harmony**, a two-stage VAE framework for UHD all-in-one
image restoration: Stage 1 trains an LH-VAE with progressive degradation perturbation
(PDPS), a DINOv2 semantic-invariance loss (`LInv`), and a latent equivariance loss
(`LEqv`); Stage 2 first trains a *latent-space restoration network* `Rθ` with a
restoration loss `LRes` (Eq. 7), then fine-tunes encoder/decoder LoRA adapters
(FHF-LoRA / PHF-LoRA, Eq. 8–9) with an inference-time control parameter α.

The cloned repo `code/lyd-2022__Latent-Harmony/` is a BasicSR-style training
codebase. It contains: the LH-VAE arch (`RAVAE_EQ`), the Stage-1 trainer
(`EQVAEModel`), a LoRA-wrapped VAE (`RAVAEHFLora`), the Stage-2 trainer
(`VAEadapter`), an HF-FFT helper, an HF discriminator, standard BasicSR losses /
datasets / metrics, and two **`*_example`** config files. There is no README content
(`README.md` is a single line), no dependency specification, no dataset or weight
download script, and no evaluation/inference entrypoint — only `basicsr/train.py`.

What I did (read-only):
- Read the paper (`paper.pdf` via `paper_text.txt`), both configs, both Stage-1/2
  models, the LoRA arch, the main VAE arch, the HF op, the losses, the dataset, and
  `train.py`.
- Wrote `_audit_code/check_ravae_registry.py`, a static AST/regex scan that
  enumerates classes registered in `ARCH_REGISTRY` (only `*_arch.py` files are
  auto-imported) and compares them to the `type:` values requested by each config.
  Output: `_audit_code/out/ravae_registry.txt`.

The single most consequential observation: **the latent-space restoration network
`Rθ` — the component that actually performs restoration in the paper's Stage 2 — is
absent from the repo.** Stage-2 training instead pushes the degraded image directly
through the VAE (encode→decode) with LoRA, so the headline restoration tables
(Tables 1–2) cannot be reproduced by this code. Independently, the Stage-2 config is
mis-wired (`vae_config.type: RAVAE`, which is not a registered class), so Stage-2
training crashes on construction.

The NeurIPS checklist (Q5) answers **[No]** to open code/data ("We will release the
source code upon acceptance"); this repo is a post-submission release whose contents
do not let a reader reproduce any reported number end-to-end.

## 2. Result-traceability table

Every quantitative artefact in the paper, and the repo location that *computes* the
underlying value (a plotting/formatting script does not satisfy this). "Repo
location" = a script/function that runs an evaluation producing the number.

| Paper artefact (value) | Repo location that computes it | Status |
|---|---|---|
| Table 1 — 4-degradation PSNR/SSIM/LPIPS, "Ours" (e.g. Avg PSNR 29.70) | (none — no Rθ, no eval driver, no metrics wired) | MISSING |
| Table 2 — 6-degradation PSNR/SSIM/LPIPS, "Ours" (e.g. Avg PSNR 29.24) | (none) | MISSING |
| Table 3 — standard-res LPIPS/FID (CosAE/PromptIR/Diff-Plugin /w Ours) | (none; FID not implemented anywhere) | MISSING |
| Table 4 — generalization PSNR/SSIM/LPIPS (unseen + composite) | (none) | MISSING |
| Table 5(a) — component ablation (PSNR/SSIM/LPIPS) | (no ablation driver; losses toggle by config but no eval) | MISSING |
| Table 5(b) — inference-time comparison (s) | (none) | MISSING |
| Table 5(c) — Restormer/NAFNet/SFHformer base vs +Ours | (none; none of these restoration nets exist in repo) | MISSING |
| Table 5(d) — metrics vs α | `RAVAEHFLora.set_alpha` exists, but no eval driver produces the α-swept numbers | MISSING |
| Fig. 2(a–e) — t-SNE / CDCS / DCT / loss curves / HF-LoRA bars | (none) | MISSING |
| Table 1/2 — FLOPs & Params ("Ours" 3.6G / 1.2M) | `__main__` of `LHVAE_arch.py` profiles `RAVAE_EQ` via thop, but path is a hardcoded author dir; not for the full pipeline | MISSING |

Routing: every row is MISSING. The dominant cause is the absent `Rθ` restoration
network plus the absence of any evaluation/inference driver and metric wiring; these
are filed once each below and cross-referenced rather than re-filed per row.

## 3. Findings

## missing

```yaml finding
id: restoration-network-rtheta-absent
category: missing
topic: "result traceability / core method"
title: "Latent restoration network Rθ (Eq. 7/9) is absent; Stage-2 restores via VAE only"
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
claim: "Stage-2 training forwards the degraded input self.lq only through net_g (a RAVAEHFLora, i.e. the LoRA-wrapped VAE encode→decode); no latent restoration network Rθ is instantiated or invoked anywhere in the repo."
concern: "The paper's restoration result depends on Rθ predicting a restored latent zres=Rθ(zdeg) (Eq. 7, and Eq. 9 runs HF(Dψ(Rθ(Eϕ(Ideg))))); with Rθ missing, the code merely reconstructs the degraded image with HF-LoRA and cannot produce the headline restoration numbers in Tables 1–5."
resolution: "Authors: please add the latent restoration network (paper says SFHformer; Table 5c also lists Restormer/NAFNet) and the LRes pre-training stage (Eq. 7), and show where zres = Rθ(zdeg) enters the Stage-2 pipeline."
cross_refs: ["stage2-ravae-not-registered", "no-eval-entrypoint", "§4.2", "Eq. 7", "Eq. 9", "Table 5(c)"]
check_script: _audit_code/check_ravae_registry.py
paper_ref: "Section 4.2 (Eq. 7, 9); Table 5(c)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: lres-pretraining-stage-absent
category: missing
topic: "training protocol"
title: "Stage-2 LRes pre-training of Rθ with frozen VAE (Eq. 7) is not implemented"
severity: high
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 148
line_end: 167
quote: |
      def optimize_parameters(self, current_iter):
          loss_dict = OrderedDict()
          hf_gt = self._hf(self.gt)

          if self._is_fhf_step(current_iter):
              # FHF-LoRA: update encoder LoRA only.
              self._set_training_stage(train_enc=True, train_dec=False, train_disc=False)
              self.optimizer_enc.zero_grad()

              recon, posterior, z = self._forward_g()
              hf_pred = self._hf(recon)
              l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
              l_hf_fid.backward()
              self.optimizer_enc.step()

              self.output = recon
              self.posterior = posterior
              self.latent = z
              loss_dict['l_hf_fid'] = l_hf_fid
              loss_dict['stage'] = recon.new_tensor(0.0)
claim: "The VAEadapter trainer only alternates FHF-LoRA (encoder) and PHF-LoRA (decoder) HF-alignment steps; there is no phase that trains a restoration network with the standard restoration loss LRes = ||Dψ*(zres) - Iclean||1 (Eq. 7) before LoRA fine-tuning."
concern: "Eq. 7 (training Rθ with frozen VAE) is described as the first sub-step of Stage 2 and is the source of the restored latent; its absence means the described training protocol is not reproducible from this repo."
resolution: "Authors: provide the LRes pre-training script/config (which model, which loss, frozen-VAE setting) and how its checkpoint feeds the LoRA stage."
cross_refs: ["restoration-network-rtheta-absent", "Eq. 7"]
paper_ref: "Section 4.2, Eq. 7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-eval-entrypoint
category: missing
topic: "result traceability / evaluation"
title: "No evaluation/inference entrypoint and no metric computation for Tables 1–5"
severity: high
confidence: high
status: finding
file: basicsr/metrics/__init__.py
line_start: 1
line_end: 8
quote: |
  from copy import deepcopy

  from basicsr.utils.registry import METRIC_REGISTRY
  from .niqe import calculate_niqe
  from .psnr_ssim import calculate_psnr, calculate_ssim

  __all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']
claim: "The repo ships only basicsr/train.py; there is no test.py / inference / benchmark script, the two provided configs set `metrics: ~` (no metrics), no config evaluates a trained model on UHD-LL/UHD-blur/etc., and no FID metric exists at all (paper Table 3 reports FID)."
concern: "Every PSNR/SSIM/LPIPS/FID/NIQE/user-study/FLOPs/runtime number in Tables 1–5 and Figure 2 is therefore untraceable to code that computes it, so none of the reported results can be reproduced or verified."
resolution: "Authors: add the evaluation entrypoint(s) and metric configuration used to produce each table (including FID), with the exact commands and the trained checkpoints."
cross_refs: ["restoration-network-rtheta-absent", "missing-deps-and-data", "Table 1", "Table 2", "Table 3", "Table 4", "Table 5"]
paper_ref: "Tables 1–5; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-deps-and-data
category: missing
topic: "expected code completeness"
title: "No dependency spec, no dataset/weights download, empty README, missing checkpoints"
severity: high
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "README.md is a single header line; there is no requirements.txt/setup.py/environment.yml, no dataset-prep or download script (configs hardcode ./datasets/train/{gt,lq}), and the referenced ./weights/dinov2_vits14.pth and ./weights/stage1_eqvae.pth are absent with no fetch instructions."
concern: "The environment cannot be rebuilt, the UHD benchmark data is not obtainable from the repo, the DINOv2 backbone and the Stage-1 VAE checkpoint are not provided, and there are no commands/results table — so the submission is not self-contained or runnable end-to-end."
resolution: "Authors: add a pinned dependency file, dataset acquisition instructions/scripts, the DINOv2 and Stage-1 checkpoints (or download links), and a README with the exact reproduce commands and a results table."
cross_refs: ["no-eval-entrypoint", "hardcoded-author-paths"]
paper_ref: "NeurIPS checklist Q5 (code release: No)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-ravae-not-registered
category: bug
topic: "configuration / arch registry"
title: "Stage-2 config requests vae_config.type=RAVAE, which is not a registered arch"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 42
quote: |
    vae_config:
      type: RAVAE
claim: "The Stage-2 config sets the wrapped VAE type to 'RAVAE', but the only VAE class registered in ARCH_REGISTRY is 'RAVAE_EQ' (the Stage-1 config correctly uses RAVAE_EQ). RAVAEHFLora.__init__ calls ARCH_REGISTRY.get(arch_type), and Registry.get raises KeyError when the name is absent."
concern: "Stage-2 training crashes at network construction (KeyError: No object named 'RAVAE' found in 'arch' registry), so the provided Stage-2 config cannot run as shipped."
resolution: "Authors: change the type to RAVAE_EQ (or register a RAVAE class) and confirm the Stage-2 config runs."
cross_refs: ["restoration-network-rtheta-absent"]
check_script: _audit_code/check_ravae_registry.py
paper_ref: "Section 4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-author-paths
category: bug
topic: "portability / dead paths"
title: "Hardcoded author-machine absolute paths injected into sys.path and __main__"
severity: medium
confidence: high
status: finding
file: basicsr/train.py
line_start: 7
line_end: 8
quote: |
  import sys
  sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
claim: "train.py (and LHVAE_arch.py line 7) append the author-only absolute path /fs-computility/ai4sData/liuyidi/code/LatentGen to sys.path; LHVAE_arch.py's __main__ further opens /fs-computility/.../options/debug.yml."
concern: "These author-specific paths do not exist on any other machine; while sys.path.append of a missing dir is silently ignored, it signals the code was not cleaned for release and the __main__ profiling path is dead, undermining the claimed FLOPs/Params profiling route."
resolution: "Authors: remove hardcoded absolute paths and make the profiling/debug entrypoints use repo-relative paths."
cross_refs: ["missing-deps-and-data"]
paper_ref: "Tables 1–2 (FLOPs/Params)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: eqv-loss-uses-perturbed-latent
category: difference
topic: "equivariance loss"
title: "LEqv downsamples the perturbed latent, not zclean=Eϕ(Iclean) as in Eq. 5"
severity: low
confidence: medium
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
claim: "The equivariance loss downsamples z, where z is the latent of the PDPS-perturbed input (optimize_parameters forwards self.pdps_input and reuses that z), whereas paper Eq. 5 defines zdown=Downs(zclean) with zclean=Eϕ(Iclean) from the clean image."
concern: "Using the perturbed latent instead of the clean latent is a valid-but-different equivariance constraint; it mildly couples the equivariance objective with the degradation perturbation rather than enforcing clean-image scale equivariance as written."
resolution: "Authors: clarify whether LEqv is intended on the clean latent (Eq. 5) or the perturbed latent; align code and paper."
cross_refs: ["§4.1", "Eq. 5"]
paper_ref: "Section 4.1, Eq. 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: configs-are-examples-not-paper-settings
category: difference
topic: "reproducibility / configs"
title: "Shipped configs are *_example with toy batch sizes and no paper hyperparameters"
severity: medium
confidence: medium
status: question
file: configs/stage1_eqvae.yml
line_start: 1
line_end: 4
quote: |
  name: stage1_eqvae_example
  model_type: EQVAEModel
  num_gpu: 1
  manual_seed: 123
claim: "Both configs are named *_example with batch_size_per_gpu: 2 and num_gpu: 1, and the paper's appendix is said to hold the actual training/eval setup (data splits, hyperparameters, compute) that is not present in the repo."
concern: "Without the real training configs (LoRA rank, loss weights, schedules, batch/GPU counts, dataset roots) the reported numbers cannot be reproduced; the example configs do not document the settings behind Tables 1–5."
resolution: "Authors: ship the exact configs used for each reported experiment, or point to where they live."
cross_refs: ["missing-deps-and-data", "no-eval-entrypoint"]
paper_ref: "Section 5 / appendix (implementation details)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

No methodology finding is filed. The portions of the method that *are* implemented
(Stage-1 PDPS, DINOv2 `LInv`, KL, the HF-FFT alignment, the LoRA injection and α
mixing) are individually sound as written. The central problems are absence of the
restoration network / evaluation harness (`missing`) and a construction-time
mis-wiring (`bug`), not an invalid-but-runnable procedure. (Reproducibility of the
reported numbers is unverifiable because there is no runnable end-to-end pipeline
and no released data/weights — see the `missing` findings, not a methodology defect.)

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 4          | high         | Core Rθ restoration net, LRes stage, eval harness, deps/data all absent |
| bug         | 2          | high         | Stage-2 config requests unregistered `RAVAE`; hardcoded author paths |
| difference  | 2          | medium       | LEqv on perturbed (not clean) latent; only toy *_example configs shipped |
| methodology | 0          | -            | Implemented sub-modules are sound; gaps are absence/mis-wiring, not invalid logic |

## 5. Closing lists

### Top take-aways (≤6, severity × confidence)
- **[missing] `restoration-network-rtheta-absent`** — the latent restoration network
  `Rθ` (Eq. 7/9), which performs the actual restoration, is not in the repo;
  Stage-2 restores via VAE+LoRA only. Headline Tables 1–5 not reproducible. (high/high)
- **[bug] `stage2-ravae-not-registered`** — Stage-2 config requests
  `vae_config.type: RAVAE`, but only `RAVAE_EQ` is registered → `KeyError` at
  construction; Stage-2 cannot run as shipped. (high/high)
- **[missing] `no-eval-entrypoint`** — no inference/benchmark script, no metric
  wiring, no FID; every reported number is untraceable to code that computes it. (high/high)
- **[missing] `missing-deps-and-data`** — no requirements file, no dataset/weights
  download, empty README, missing DINOv2 + Stage-1 checkpoints. (high/high)
- **[missing] `lres-pretraining-stage-absent`** — the Eq. 7 `LRes` pre-training of
  `Rθ` (first sub-step of Stage 2) is not implemented. (high/high)
- **[bug] `hardcoded-author-paths`** — author-only absolute paths in `train.py` /
  `LHVAE_arch.py`; release was not cleaned. (medium/high)

### Items that genuinely look fine
- Stage-1 PDPS schedule (`_build_pdps_input` / `_apply_synth_deg`) implements the
  three-branch perturbation (clean / synth / interpolation) with monotone `t`,
  matching Eq. 3's structure (clean=gt, real-degraded=lq).
- `LInv` aligns the perturbed-input latent to DINOv2 features of the clean image
  (`_compute_inv_loss`), consistent with Eq. 4.
- The LoRA injection (`LHVAE_hflora_arch._inject_lora`) wraps encoder/decoder
  convs and `set_alpha` mixes encoder (α) and decoder (1−α) LoRA, matching the
  inference-time control of Eq. in §4.3.
- The HF-FFT high-pass extractor (`extract_hf_fft`) is a standard, correct
  high-pass mask in the Fourier domain.

### Open questions for the authors
- Where is `Rθ` (SFHformer / Restormer / NAFNet) and the code wiring `zres =
  Rθ(zdeg)` into Stage 2? (drives `restoration-network-rtheta-absent`)
- Are the shipped `*_example` configs the actual paper settings, or are the real
  configs/checkpoints/eval scripts withheld pending release?
  (`configs-are-examples-not-paper-settings`)
- Is `LEqv` intended on the clean or the perturbed latent?
  (`eqv-loss-uses-perturbed-latent`)
