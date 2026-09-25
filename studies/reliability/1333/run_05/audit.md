# Audit — Latent Harmony (NeurIPS 2025, paper #1333)

## 1. Summary

The paper proposes "Latent Harmony" (LH), a two-stage VAE framework for UHD all-in-one
image restoration: Stage 1 trains a regularized VAE (LH-VAE) with progressive degradation
perturbation (PDPS), a DINOv2 semantic-invariance loss (L_Inv), and a latent equivariance
loss (L_Eqv); Stage 2 first trains a latent restoration network R_theta with a restoration
loss L_Res (VAE frozen), then fine-tunes encoder/decoder HF-LoRA adapters with a
high-frequency fidelity loss (FHF-LoRA) and a GAN-based high-frequency perceptual loss
(PHF-LoRA), with an inference-time control parameter alpha.

The cloned repo (`code/lyd-2022__Latent-Harmony/`) is a BasicSR fork containing 49 Python
files: the VAE architecture (`RAVAE_EQ`), a LoRA wrapper (`LoRAConv2d`, `RAVAEHFLora`), a
UNet HF discriminator, two trainer models (`EQVAEModel` Stage 1, `VAEadapter` Stage 2),
losses, datasets, metrics, and the generic BasicSR `train.py`. There are two example
configs (`stage1_eqvae.yml`, `stage2_hflora.yml`). The README is a single line (`# Latent-
Harmony`). The NeurIPS checklist (item 5) states "We will release the source code upon
acceptance of the paper."

What I did. I read the paper (PDF + text extraction), enumerated the repo, read the two
trainer models, the VAE and LoRA architectures, the two configs, the loss registry, and the
options/training entry point. I wrote two read-only checks under `_audit_code/`:
`check_registry_vs_configs.py` (statically maps every `type:` in the configs to the
`@*_REGISTRY.register()` class names found in the source) and `check_completeness.py`
(inventories dependency specs / eval scripts / weights / data / README). Outputs are in
`_audit_code/out/`. I did not run training (no data, no GPU, and `pyiqa` is not installed).

Headline result: the repository is **training-only and incomplete**. It contains no
evaluation/inference code, no metric computation wired into validation, no dependency
specification, no datasets, and no pretrained weights — so none of the paper's quantitative
results (Tables 1–5, Fig. 2/3 panels) can be reproduced or even traced to a computing
script. Two concrete code defects also block running as configured: the Stage-2 config
references an unregistered arch `RAVAE` (instantiation raises `KeyError`), and the
experiments root is hardcoded to an author-specific absolute path. Finally, the Stage-2
trainer as written does **not** include the latent restoration network R_theta that the
paper centers Stage 2 on (Eq. 7, Table 5c): it only fine-tunes the VAE to reconstruct the
degraded input's high frequencies, so it cannot perform the restoration the tables report.

## 2. Result-traceability table

Every quantitative artefact in the paper is checked for a script that *computes* the value.
"(none)" means no script in the repo produces the number; plotting/formatting alone does
not count (Rule G).

| Paper artefact | Repo location (computes value?) | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1: 4-degradation PSNR/SSIM/LPIPS for Ours + 11 baselines | (none) — no eval script; val `metrics: ~` | — | — | MISSING |
| Table 2: 6-degradation PSNR/SSIM/LPIPS, Ours + baselines | (none) | — | — | MISSING |
| Table 1/2: FLOPs (3.6G), Params (1.2M) of "Ours" | (none) — no FLOPs/param-count script | — | — | MISSING |
| Table 3: standard-res LPIPS/FID for PromptIR/Diff-Plugin/CosAE +/- Ours | (none) — methods not in repo | — | — | MISSING |
| Table 4: generalization PSNR/SSIM/LPIPS (unseen + composite) | (none) | — | — | MISSING |
| Table 5(a): ablation (w/o L_Inv, L_Eqv, PDPS, FHF/PHF-LoRA, ...) | (none) — no ablation harness/configs | — | — | MISSING |
| Table 5(b): inference time (LH 0.43 s vs others) | (none) — no timing script | — | — | MISSING |
| Table 5(c): R_theta swap (Restormer/NAFNet/SFHformer, Base vs +Ours) | (none) — no R_theta arch present at all | — | — | MISSING (see no-restoration-network-rtheta) |
| Table 5(d): alpha-sweep PSNR/SSIM/LPIPS/User (alpha 0.2–0.8) | `set_alpha` exists, but no sweep/eval driver | — | — | MISSING |
| Fig. 2(a) t-SNE, (b) CDCS, (c) DCT, (d/e) loss curves | (none) — no analysis scripts | — | — | MISSING |
| User-study scores (Fig.2e, Table 5d) | (none) — no study data/code | — | — | MISSING |

All rows are MISSING: the repo ships the two training models but nothing that evaluates a
trained model or computes any reported number. (Per Rule G this routes to a single
representative `missing` finding for evaluation/metric code, cross-referenced by the others,
rather than one finding per table row.)

## 3. Findings

## missing

```yaml finding
id: no-eval-or-metric-code
category: missing
topic: "result traceability"
title: "No evaluation/inference code; validation computes no metrics — no reported number traceable"
severity: high
confidence: high
status: finding
file: configs/stage1_eqvae.yml
line_start: 116
line_end: 121
quote: |
  val:
    val_freq: 1000
    save_img: false
    pbar: true
    suffix: ~
    metrics: ~
claim: "The repo contains only the two BasicSR training models and the generic train.py; there is no test/inference/evaluation script, and both configs set val.metrics to null (~), so even validation computes no PSNR/SSIM/LPIPS/FID/NIQE. _audit_code/check_completeness.py finds zero eval/test/infer scripts and zero shell scripts."
concern: "None of the paper's quantitative results (Tables 1-5, Fig. 2/3 panels) can be reproduced or traced to a script that computes them; the entire results section is unsupported by runnable code."
resolution: "Provide the evaluation scripts (with the exact commands, datasets, and pretrained checkpoints) that compute every PSNR/SSIM/LPIPS/FID/NIQE/FLOPs/runtime value in Tables 1-5 and the figures."
cross_refs: ["no-deps-spec", "missing-weights-and-data", "no-restoration-network-rtheta", "readme-empty-no-repro-commands"]
check_script: _audit_code/check_completeness.py
paper_ref: "Tables 1-5; Fig. 2-3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-restoration-network-rtheta
category: missing
topic: "evaluation consistency / Stage-2 architecture"
title: "Latent restoration network R_theta (Eq.7, Table 5c) is absent; Stage-2 only auto-encodes the degraded input"
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
claim: "Stage-2 network RAVAEHFLora.forward passes the degraded input x straight through the VAE (encode->decode) with no latent restoration network. The whole repo has no R_theta arch: grep for SFHformer/NAFNet/Restormer-as-restoration returns only an attention-block 'restormer' option inside the VAE, and Stage-2 model VAEadapter._forward_g (basicsr/models/VAEadapter_model.py:132-139) calls net_g(self.lq) directly. The paper defines z_res = R_theta(z_deg) trained with L_Res (Eq.7) and ablates R_theta in Table 5(c)."
concern: "As coded, Stage-2 only fine-tunes the VAE to reconstruct the degraded input's high frequencies (z_res = z_deg), so it cannot map degraded latents to clean ones; the restoration step the paper's results depend on is not implemented."
resolution: "Provide the R_theta arch, the Stage-2a training that optimizes only theta with L_Res (Eq.7) under a frozen VAE, and the wiring that feeds z_res (not z_deg) into the decoder during HF-LoRA fine-tuning and inference."
cross_refs: ["no-eval-or-metric-code", "stage2-ravae-arch-missing"]
paper_ref: "Section 4.2, Eq. 7-9; Table 5(c)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-weights-and-data
category: missing
topic: "expected code completeness"
title: "No datasets, no pretrained VAE/DINOv2 weights, no fetch scripts; configs point to non-existent paths"
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
claim: "Stage-2 requires a Stage-1 checkpoint at ./weights/stage1_eqvae.pth and Stage-1 requires DINOv2 weights at ./weights/dinov2_vits14.pth (configs/stage1_eqvae.yml:95); both the ./weights and ./datasets directories are absent (_audit_code/check_completeness.py: weights_dir_exists=false, datasets_dir_exists=false, zero weight files besides niqe params). No download/data-prep script exists and the paper checklist says code is unreleased ('upon acceptance')."
concern: "Neither stage can be trained or evaluated out-of-box: the required pretrained weights and the UHD-LL/UHD-blur/UHD-haze/UHD-rain/UHD-snow/Gendeg datasets are neither shipped nor fetchable, so results cannot be reproduced."
resolution: "Release the Stage-1 LH-VAE checkpoint, the DINOv2 weights pointer, and dataset download/preparation instructions (accessions or URLs) for every benchmark used in Tables 1-5."
cross_refs: ["no-eval-or-metric-code"]
check_script: _audit_code/check_completeness.py
paper_ref: "Section 5; checklist item 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-deps-spec
category: missing
topic: "expected code completeness / dependencies"
title: "No dependency specification (no requirements.txt / environment.yml / setup.py)"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 9
line_end: 9
quote: |
  import pyiqa
claim: "_audit_code/check_completeness.py finds zero dependency specs in the repo. Yet the code imports non-trivial third-party packages such as pyiqa (here), torch, torchvision, and (via torch.hub) the facebookresearch/dinov2 hub repo (basicsr/models/EQvae_model.py:115), none of which are pinned or listed."
concern: "The environment cannot be reconstructed; unpinned/unlisted dependencies (pyiqa, dinov2 hub, torch versions) make reproduction non-deterministic and likely to break."
resolution: "Add a requirements.txt or environment.yml pinning torch/torchvision/pyiqa and document the DINOv2 source/version."
cross_refs: ["no-eval-or-metric-code"]
check_script: _audit_code/check_completeness.py
paper_ref: "checklist item 6/8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-empty-no-repro-commands
category: missing
topic: "documentation / reproducibility"
title: "README is a single title line — no results table, no commands to reproduce"
severity: low
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The entire README is the single line shown (_audit_code/check_completeness.py: readme_nonblank_lines == ['# Latent-Harmony']). There are no setup, training, evaluation, or data-preparation instructions and no results table."
concern: "A reviewer has no exact commands or environment to reproduce any result; combined with the missing eval code, the submission is not self-documenting."
resolution: "Add a README with environment setup, dataset preparation, exact train/eval commands per table, and a results table tying commands to reported numbers."
cross_refs: ["no-eval-or-metric-code", "no-deps-spec"]
check_script: _audit_code/check_completeness.py
paper_ref: "checklist item 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-ravae-arch-missing
category: bug
topic: "configuration / registry"
title: "Stage-2 config sets vae_config.type: RAVAE, but no 'RAVAE' arch is registered -> KeyError at build"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 42
quote: |
  vae_config:
    type: RAVAE
claim: "RAVAEHFLora.__init__ does ARCH_REGISTRY.get(cfg.pop('type'))(**cfg) (basicsr/archs/LHVAE_hflora_arch.py:37-38). The only registered VAE arch is 'RAVAE_EQ' (LHVAE_arch.py:130); there is no class named 'RAVAE'. _audit_code/check_registry_vs_configs.py reports type 'RAVAE' in stage2_hflora.yml as UNRESOLVED. Registry.get raises KeyError for an unknown name (basicsr/utils/registry.py:62-66)."
concern: "Building the Stage-2 network from the shipped config crashes immediately with KeyError: No object named 'RAVAE', so Stage-2 cannot run as configured."
resolution: "Change vae_config.type to RAVAE_EQ (or register a RAVAE alias) and confirm the Stage-1 checkpoint key layout matches the wrapped arch."
cross_refs: ["no-restoration-network-rtheta"]
check_script: _audit_code/check_registry_vs_configs.py
paper_ref: "Section 4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-experiments-root
category: bug
topic: "hardcoded absolute path"
title: "Training output root hardcoded to author machine path /fs-computility/ai4sData/liuyidi/model"
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
claim: "For every training run the experiments root is unconditionally set to the absolute author-specific path '/fs-computility/ai4sData/liuyidi/model/experiments/<name>', ignoring root_path. make_exp_dirs(opt) (train.py:105) then tries to create directories there."
concern: "On any other machine this path is not writable / does not exist, so training fails when creating the experiment directory; checkpoints/logs cannot be written."
resolution: "Derive experiments_root from root_path (e.g. osp.join(root_path, 'experiments', opt['name'])) as upstream BasicSR does, or make it configurable."
cross_refs: ["stale-sys-path-appends"]
check_script: _audit_code/check_registry_vs_configs.py
paper_ref: "N/A (engineering)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stale-sys-path-appends
category: bug
topic: "hardcoded absolute path / dead imports"
title: "Multiple hardcoded sys.path.append to non-existent author/devbox directories"
severity: low
confidence: high
status: finding
file: basicsr/train.py
line_start: 7
line_end: 8
quote: |
  import sys
  sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
claim: "train.py prepends an author-specific absolute path to sys.path; the same pattern recurs in basicsr/archs/LHVAE_arch.py:7 (/fs-computility/.../LatentGen), basicsr/utils/modules/attention.py:8 (/code/UHDFour_code-main), and basicsr/archs/encoder_3.py:8 (/code/UHDformer-main). LHVAE_arch.py:467 also opens a hardcoded /fs-computility/.../debug.yml in its __main__."
concern: "These point to directories that do not exist outside the authors' machines; while sys.path.append of a missing dir is silently tolerated, it signals the released tree was not cleaned and risks importing the wrong module if such a path happens to exist."
resolution: "Remove the hardcoded sys.path.append/open statements; rely on package-relative imports."
cross_refs: ["hardcoded-experiments-root"]
paper_ref: "N/A (engineering)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: phf-decoder-uses-fidelity-l1-not-only-gan
category: difference
topic: "evaluation consistency (paper vs code)"
title: "PHF-LoRA (decoder) step also minimizes the HF fidelity L1, not only the GAN perceptual loss"
severity: low
confidence: medium
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 193
line_end: 201
quote: |
            l_total = recon.new_tensor(0.0)
            l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
            l_total = l_total + l_hf_fid
            loss_dict['l_hf_fid'] = l_hf_fid

            if self.net_d_hf is not None and self.cri_gan is not None and self.lambda_gan > 0:
                pred_fake_g = self.net_d_hf(hf_pred)
                l_hf_gan = self.cri_gan(pred_fake_g, True, is_disc=False) * self.lambda_gan
                l_total = l_total + l_hf_gan
claim: "In the PHF-LoRA (decoder) step the decoder LoRA is updated with l_hf_fid (HF L1 fidelity, always added) plus the GAN loss, whereas the paper's PHF-LoRA is guided by a perception-oriented loss only (Eq. 9, L_HFGAN), with the HF fidelity loss reserved for the encoder FHF-LoRA (Eq. 8)."
concern: "Adding the fidelity L1 to the decoder objective contradicts the paper's clean perception/fidelity separation and could blunt the perceptual specialization the alpha trade-off relies on."
resolution: "Clarify whether the decoder LoRA is intended to also see the HF L1 term; if not, remove l_hf_fid from the decoder step to match Eq. 9."
cross_refs: []
paper_ref: "Section 4.2, Eq. 8-9"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A — the procedure that *is* implemented (Stage-1 VAE with PDPS/L_Inv/L_Eqv; LoRA fine-
tuning) is, where present, methodologically standard (train/val are separate folders; LoRA
is a normal low-rank adapter; the inference alpha-mixing matches the paper). The decisive
problems are absence (no eval code, no R_theta, no data/weights/deps) and breakage (config
arch / hardcoded path), not an invalid-but-runnable procedure. No leakage, metric-fit, or
statistical-integrity defect is verifiable from the artefact because no evaluation runs.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | No eval/metric code, no R_theta, no data/weights, no deps spec, empty README. |
| bug         | 3          | high         | Stage-2 config arch unregistered (crash); hardcoded experiments root; stale sys.path. |
| difference  | 1          | low          | Decoder PHF-LoRA step also uses HF fidelity L1, not only GAN (Eq.9). |
| methodology | 0          | -            | No runnable evaluation -> nothing methodological is verifiable. |

## 5. Closing lists

### Top take-aways (ranked by severity x confidence)
1. **[missing] no-eval-or-metric-code** — no evaluation/inference script and `val.metrics: ~`, so no number in Tables 1-5 or Fig. 2/3 traces to a computing script (high/high).
2. **[missing] no-restoration-network-rtheta** — the latent restoration network R_theta central to Stage 2 (Eq.7, Table 5c) is absent; Stage-2 only auto-encodes the degraded input (high/high).
3. **[bug] stage2-ravae-arch-missing** — Stage-2 config `vae_config.type: RAVAE` is unregistered; `ARCH_REGISTRY.get('RAVAE')` raises KeyError, so Stage-2 cannot build (high/high).
4. **[missing] missing-weights-and-data** — no datasets, no Stage-1/DINOv2 weights, no fetch scripts; configs point at absent `./weights` and `./datasets` (high/high).
5. **[bug] hardcoded-experiments-root** — output root hardcoded to `/fs-computility/.../model`, so training fails to create dirs on any other machine (medium/high).
6. **[missing] no-deps-spec** — no requirements.txt/environment.yml; pyiqa, torch, dinov2-hub unpinned (medium/high).

### Items that genuinely look fine
- LoRA wrapper (`LoRAConv2d`) is a standard low-rank adapter; zero-init of `lora_up` makes the adapter a no-op at start.
- Inference alpha-mixing in `RAVAEHFLora.set_alpha` (enc_scale=alpha, dec_scale=1-alpha) matches the paper's phi=phi*+alpha*dphi, psi=psi*+(1-alpha)*dpsi.
- Stage-1 `L_Inv` aligns the perturbed-image latent with DINOv2 features of the clean image, matching Eq. 4 (`fVFM = VFM(I_clean)`).
- Train/val use separate `dataroot` folders (no obvious train/val overlap in the config wiring).
- All loss classes named in the configs (`L1Loss`, `KlLoss`, `GANLoss`) are registered and present.

### Open questions for the authors
- Where is the Stage-2a training of R_theta with L_Res (Eq.7), and where is R_theta swapped in for Table 5(c)? Was the released tree intended to include it?
- Is `vae_config.type: RAVAE` a typo for `RAVAE_EQ`, or is a separate `RAVAE` class expected to be released?
- Were the Tables 1-5 numbers produced by code not in this repo (e.g., the `/fs-computility/.../LatentGen` tree referenced by the stale sys.path appends)? If so, that code is the artefact that needs releasing.
