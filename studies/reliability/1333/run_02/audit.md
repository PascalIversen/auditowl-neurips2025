# Audit — Latent Harmony (NeurIPS 2025, paper #1333)

## 1. Summary

The paper proposes "Latent Harmony" (LH), a two-stage VAE-based framework for
UHD all-in-one image restoration: Stage 1 trains a regularised LH-VAE (PDPS
perturbation + DINOv2 semantic-invariance loss `LInv` + latent equivariance loss
`LEqv`); Stage 2 first trains a **latent-space restoration network `Rθ`** (Eq. 7)
with a frozen VAE, then fine-tunes the VAE with high-frequency-guided LoRA
adapters (FHF-LoRA on encoder, PHF-LoRA on decoder), with an inference-time
`α` blend (Eq. 8, 9, §4.3). Headline results are SOTA PSNR/SSIM/LPIPS on
4- and 6-degradation UHD benchmarks (Tables 1–2), standard-resolution
adaptability (Table 3), generalisation (Table 4), and ablations + runtime +
`α`-sweep (Table 5).

The cloned repo `code/lyd-2022__Latent-Harmony/` is a stripped BasicSR fork. It
contains: the Stage-1 trainer (`EQVAEModel`), the Stage-2 trainer (`VAEadapter`),
the VAE arch (`RAVAE_EQ`), the LoRA-wrapped VAE (`RAVAEHFLora`), a LoRA conv
(`LoRAConv2d`), losses, a HF-FFT op, a paired-image dataset, and two example
configs. It is **README-empty** (16 bytes: just the title), with **no test/
inference/eval entrypoint**, **no dependency spec**, **no pretrained weights**,
and **no data**. The NeurIPS checklist (Q5) states code/data are *not* released
("released upon acceptance").

What I did: read the paper (PDF + extraction), read every `.py` and both
configs, and ran two static checks under `_audit_code/` (no torch import, repo
read-only): `check_registry_and_artifacts.py` enumerates registered arch classes
vs. config-referenced types and tests existence of promised artefacts. Output in
`_audit_code/out/registry_and_artifacts.json`.

The decisive issue: **the latent restoration network `Rθ` — the core of Stage 2
and the producer of every restoration number in Tables 1–5 — is absent from the
repo.** The Stage-2 trainer simply auto-encodes the degraded input through the
VAE; no module maps `zdeg → zres`. Combined with the absent baselines, weights,
data, and metric harness, **no quantitative result in the paper is reproducible
or even traceable to code that computes it.** Separately, the Stage-2 example
config references an unregistered arch type (`RAVAE`) and so crashes on
construction.

## 2. Result-traceability table

Every value below was searched for a script/function that *computes* it (not
merely plots/formats). "Repo location = (none)" means no such computation exists.

| Paper artefact | Repo location (computes the value?) | Computed? | Matches? | Status |
|---|---|---|---|---|
| Table 1 — PSNR/SSIM/LPIPS, 4 degradations, "Ours" + 11 baselines | (none) — no eval/inference script, `metrics: ~` in both configs, no baselines in repo | — | — | MISSING |
| Table 2 — PSNR/SSIM/LPIPS, 6 degradations, "Ours" + 11 baselines | (none) | — | — | MISSING |
| Table 3 — standard-res adaptability (PromptIR/Diff-Plugin/CosAE ± Ours) | (none) — none of these three host methods exist in repo | — | — | MISSING |
| Table 4 — generalisation (unseen + composite), Ours vs HAIR/UHD-processer | (none) | — | — | MISSING |
| Table 5(a) — ablation (w/o LInv / LEqv / PDPS / FHF / PHF / LoRA / FT) | (none) — no ablation harness; no eval | — | — | MISSING |
| Table 5(b) — inference time (DreamUIR/Histformer/UHDprocesser/LH) | (none) | — | — | MISSING |
| Table 5(c) — restoration-net ablation (Restormer/NAFNet/SFHformer Base vs +Ours) | (none) — no restoration network of any kind in repo | — | — | MISSING |
| Table 5(d) — α sweep (PSNR/SSIM/LPIPS/User @ α=0.2..0.8) | (none) — α blend implemented (`RAVAEHFLora.set_alpha`) but no eval driver produces these numbers | — | — | MISSING |
| Fig. 2(a–e) — t-SNE / CDCS / DCT / loss / HF-LoRA motivation plots | (none) | — | — | MISSING |
| FLOPs 3.6G / Params 1.2M ("Ours", Tables 1–2) | (none) — `__main__` in `LHVAE_arch.py` profiles `RAVAE_EQ` only via an absolute-path debug.yml; not a wired script | — | — | MISSING |
| Eq. 7 `LRes = ‖Dψ*(Rθ(Eϕ*(Ideg))) − Iclean‖₁` (Stage-2 base restoration) | (none) — no `Rθ` module; Stage-2 trainer auto-encodes Ideg | — | — | MISSING |

Every quantitative row is MISSING. There are no MISMATCH rows because no value
is computed by any code path in the repo.

## 3. Findings

## missing

```yaml finding
id: missing-restoration-network
category: missing
topic: "core method / result traceability"
title: "Latent restoration network Rθ (Stage-2 core) is absent from the repo"
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
claim: "The Stage-2 network RAVAEHFLora.forward simply runs the degraded input through the VAE (encode->decode); its only learnable parts are encoder/decoder LoRA adapters. There is no module that maps the degraded latent z_deg to a restored latent z_res."
concern: "The paper's Stage 2 (Eq. 7) is built around a latent restoration network Rθ (z_res = Rθ(z_deg)) that produces every restoration result in Tables 1-5; with no Rθ in the repo the pipeline can only reconstruct the degraded image, so no headline restoration number is traceable to code that computes it."
resolution: "Provide the Rθ implementation (paper says SFHformer; Table 5c also tests Restormer/NAFNet), its training script (the LRes step of Eq. 7), and how its latent feeds the LoRA-VAE decode."
cross_refs: ["stage2-config-no-restoration-net", "missing-eval-and-baselines", "stage2-ravae-type-unregistered"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "Eq. 7 (§4.2); Table 5(c) 'latent space restoration network ... adopts SFHformer'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-eval-and-baselines
category: missing
topic: "result traceability / evaluation"
title: "No evaluation/inference entrypoint and no metric computation; every table is unreproducible"
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
claim: "The repo has only basicsr/train.py as an entrypoint; there is no test.py/inference.py/eval.py, and both configs set val.metrics to null (~), so no PSNR/SSIM/LPIPS/FID/NIQE is ever computed. None of the 11+ comparison baselines (AIRNet, PromptIR, Histoformer, HAIR, UHD-processer, ...) are present either."
concern: "Tables 1-5 (all PSNR/SSIM/LPIPS/FID/runtime/ablation numbers and the entire baseline comparison) cannot be produced or checked from the repo, because nothing computes a metric and no baseline code exists."
resolution: "Provide the evaluation script that loads checkpoints, runs full-size 4K inference, and computes PSNR/SSIM/LPIPS/FID/NIQE, plus the baseline configs/weights used for Tables 1-4."
cross_refs: ["missing-restoration-network", "missing-weights-and-data"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "Tables 1-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-weights-and-data
category: missing
topic: "expected code completeness"
title: "Pretrained weights and datasets are referenced by config but absent; no fetch script"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 33
line_end: 36
quote: |
  network_g:
    type: RAVAEHFLora
    pretrain_vae_path: ./weights/stage1_eqvae.pth
    pretrain_param_key: params
claim: "Stage-2 config loads ./weights/stage1_eqvae.pth and Stage-1 config loads ./weights/dinov2_vits14.pth and reads data from ./datasets/train/{gt,lq}; none of these files/dirs exist in the repo and there is no download/fetch script or accession."
concern: "Stage 2 cannot start without the Stage-1 VAE checkpoint, Stage 1 cannot start without the DINOv2 weights (LInv hard-errors if the path is empty), and neither stage has training data, so the pipeline cannot be run as shipped."
resolution: "Release the LH-VAE checkpoint and DINOv2 weights (or a documented download), and provide data-preparation scripts / accessions for the UHD benchmarks (UHD-LL, UHD-blur, UHD-haze, UHD-rain, UHD-snow, UHD denoising)."
cross_refs: ["missing-eval-and-baselines"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "§5 Experiments; NeurIPS checklist Q5 ('released upon acceptance')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage2-config-no-restoration-net
category: missing
topic: "core method"
title: "Stage-2 config wires no restoration network into the pipeline"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 33
line_end: 41
quote: |
  network_g:
    type: RAVAEHFLora
    pretrain_vae_path: ./weights/stage1_eqvae.pth
    pretrain_param_key: params
    rank: 8
    alpha: 1.0
    dropout: 0.0
    target_min_kernel: 3
    vae_config:
claim: "The only network in the Stage-2 config is RAVAEHFLora (frozen VAE + LoRA) plus a HF discriminator; there is no network_r / restoration sub-config and the VAEadapter trainer never instantiates or calls a restoration network."
concern: "This is the config-side manifestation of the absent Rθ: as configured, Stage 2 fine-tunes a VAE that auto-encodes the degraded image, which cannot reproduce the paper's restoration results."
resolution: "Add the restoration-network entry to the Stage-2 config and the trainer code that uses it; clarify whether Rθ weights are also released."
cross_refs: ["missing-restoration-network"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "§4.2 Stage Two"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-deps-and-readme
category: missing
topic: "expected code completeness / reproducibility"
title: "No dependency specification and an empty README (no commands, no results table)"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The repo has no requirements.txt / setup.py / environment.yml (confirmed by _audit_code), yet code imports third-party packages such as pyiqa (basicsr/models/sr_model.py:6, VAEadapter_model.py:8) and torch/torchvision; the README is a 16-byte title with no install/run instructions and no results table."
concern: "The environment cannot be reconstructed (unpinned/unlisted deps) and there are no exact commands to reproduce any reported number, failing the minimum completeness expected of a submission."
resolution: "Add a pinned dependency file (torch/torchvision/pyiqa/etc. versions) and a README with a results table and the exact train/test commands."
cross_refs: ["missing-eval-and-baselines"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "NeurIPS checklist Q5/Q6"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-ravae-type-unregistered
category: bug
topic: "configuration / runtime"
title: "Stage-2 config requests arch type 'RAVAE' which is not registered (KeyError on build)"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 42
quote: |
  vae_config:
      type: RAVAE
claim: "RAVAEHFLora.__init__ does ARCH_REGISTRY.get(cfg.pop('type'))(**cfg) on vae_config; vae_config.type is 'RAVAE', but only RAVAE_EQ, RAVAEHFLora, LoRAConv2d and UNetDiscriminatorSN are registered (confirmed by _audit_code: RAVAE_registered=false), and Registry.get raises KeyError for an unknown name (basicsr/utils/registry.py:62-65)."
concern: "Building the Stage-2 network with the provided config crashes immediately with KeyError: \"No object named 'RAVAE' found in 'arch' registry!\", so the shipped Stage-2 example cannot run."
resolution: "Change vae_config.type to 'RAVAE_EQ' (or register a 'RAVAE' arch) so the Stage-2 config constructs."
cross_refs: ["stage2-config-no-restoration-net"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "n/a (configuration defect)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-experiments-root
category: bug
topic: "hardcoded path"
title: "experiments_root hardcoded to an author-specific absolute path"
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
claim: "On any training run the experiment root (logs, checkpoints, visualizations) is forced to the absolute author path /fs-computility/ai4sData/liuyidi/model/experiments/<name> instead of being derived from root_path (the upstream BasicSR uses osp.join(root_path, 'experiments', opt['name']))."
concern: "On a reviewer machine this path does not exist and is typically not writable, so training fails to save checkpoints / states (or scatters them to an unexpected absolute location), impeding reproduction."
resolution: "Derive experiments_root from root_path (or make it configurable) rather than hardcoding an author-specific absolute path."
cross_refs: ["hardcoded-syspath"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-syspath
category: bug
topic: "hardcoded path"
title: "sys.path.append to a non-existent author-specific directory in train.py and arch"
severity: low
confidence: high
status: finding
file: basicsr/train.py
line_start: 7
line_end: 8
quote: |
  import sys
  sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
claim: "train.py and LHVAE_arch.py prepend the author's absolute path /fs-computility/ai4sData/liuyidi/code/LatentGen to sys.path; LHVAE_arch.py's __main__ also opens .../LatentGen/options/debug.yml."
concern: "These dead absolute paths are remnants of the author environment; they do not break import on their own (append silently ignores missing dirs) but signal the repo was not cleaned for release and the __main__ profiling block (the only place FLOPs/params could be computed) cannot run as shipped."
resolution: "Remove the author-specific sys.path.append lines and the hardcoded debug.yml path; ship a runnable FLOPs/params script."
cross_refs: ["hardcoded-experiments-root"]
check_script: _audit_code/check_registry_and_artifacts.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: eqv-loss-uses-perturbed-latent
category: difference
topic: "stage-1 equivariance loss"
title: "LEqv downsamples the perturbed-input latent, not the clean latent z_clean as in Eq. 5"
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
claim: "The equivariance loss downsamples z (the latent of the PDPS-perturbed input passed in from optimize_parameters) and compares decode(z_down) to a downsampled clean image; paper Eq. 5 defines z_down = Downs(z_clean) on the CLEAN-image latent."
concern: "Code uses the perturbed-input latent rather than the clean latent for the equivariance constraint; the constraint is still a valid scale-equivariance regulariser, so this is a faithfulness mismatch rather than an invalid procedure (low severity, and ambiguous because z could be argued to stand in for the encoded clean signal)."
resolution: "Confirm whether LEqv should operate on the clean-image latent (Eq. 5) or the perturbed-input latent, and align code/paper accordingly."
cross_refs: []
paper_ref: "Eq. 5 (§4.1)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A for an independent methodology defect: the procedure that would need to be
judged for validity (Stage-2 latent restoration + evaluation) is not present in
the repo, so its soundness cannot be assessed from code. The dominant problems
are absence (missing) and a broken config (bug), which take priority under the
routing rules. No leakage/metric/baseline-validity finding can be grounded
without the eval and restoration code; the relevant concerns are filed as
`missing` (no baselines run under a shared harness; no held-out metric
computation) rather than fabricated `methodology` verdicts.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | Core Rθ, eval harness, baselines, weights, data, deps, README all absent. |
| bug         | 3          | high         | Stage-2 config references unregistered 'RAVAE' (KeyError); hardcoded author paths. |
| difference  | 1          | low          | LEqv uses perturbed-input latent vs clean latent in Eq. 5. |
| methodology | 0          | -            | Restoration/eval code absent -> cannot ground a methodology verdict (routed to missing). |

## 5. Closing lists

### Top take-aways (ranked)
1. [missing] `missing-restoration-network` — the Stage-2 latent restoration
   network `Rθ`, which produces every restoration number in Tables 1–5, does
   not exist in the repo; Stage 2 only auto-encodes the degraded image.
2. [missing] `missing-eval-and-baselines` — no inference/eval entrypoint, no
   metric computation (`metrics: ~`), no baselines; Tables 1–5 are
   unreproducible and untraceable.
3. [missing] `missing-weights-and-data` — Stage-1 VAE checkpoint, DINOv2
   weights, and all datasets are referenced but absent, with no fetch script.
4. [bug] `stage2-ravae-type-unregistered` — the shipped Stage-2 config requests
   arch `RAVAE`, which is unregistered, so network build raises `KeyError`.
5. [missing] `stage2-config-no-restoration-net` — the Stage-2 config wires no
   restoration network at all (config-side view of take-away 1).
6. [bug] `hardcoded-experiments-root` — all training outputs are forced to an
   author-specific absolute path that will not exist on a reviewer machine.

### Items that genuinely look fine
- LoRA blending matches the paper's inference control: `enc_scale = α`,
  `dec_scale = 1 − α` via `set_alpha` (`LHVAE_hflora_arch.py:83-91`,
  `lora_arch.py:81-86`); zero-init of `lora_up` makes adapters identity at start.
- HF extraction (`hf_ops.py`) is a standard FFT high-pass and is used
  consistently for `LHFFid` (encoder step) and the GAN/decoder step in
  `VAEadapter_model.py`, matching the FHF/PHF split described in §4.2.
- Stage-1 alternating PDPS branches (no-perturb / synth / interpolate) with
  probabilities `p0,p1,p2` summing to 1 are implemented and renormalised
  (`EQvae_model.py:206-224`), consistent with Eq. 3.
- Stage-2 alternating optimization (`_is_fhf_step`) alternates encoder-LoRA and
  decoder-LoRA(+discriminator) steps as the paper describes.

### Open questions for the authors
- Where is the restoration network `Rθ` (SFHformer / Restormer / NAFNet
  variants in Table 5c) and its `LRes` training step (Eq. 7)? Will its weights
  be released?
- Were Tables 1–5 produced by this repository, or by a fuller internal codebase
  (e.g. the absolute `.../LatentGen` tree referenced in `sys.path.append`)?
- For `LEqv`, should the equivariance constraint use the clean-image latent
  (Eq. 5) or the perturbed-input latent (as coded)?
