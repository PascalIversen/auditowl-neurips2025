# Code-repository audit — Paper 1333 "Latent Harmony"

## 1. Summary

The paper proposes **Latent Harmony**, a two-stage UHD image-restoration framework:
Stage 1 trains an **LH-VAE** (`RAVAE_EQ`) with progressive degradation perturbation
(PDPS), a DINOv2 semantic-invariance loss `LInv`, and a latent equivariance loss `LEqv`;
Stage 2 introduces a **latent restoration network `R_theta`** (paper says SFHformer,
Table 5c) plus **HF-LoRA** fine-tuning of the VAE (encoder FHF-LoRA for fidelity,
decoder PHF-LoRA + GAN for perception), with an inference control `alpha`.

The cloned repo `code/lyd-2022__Latent-Harmony/` is a partial **BasicSR fork** containing:
the VAE arch (`basicsr/archs/LHVAE_arch.py`), a LoRA wrapper
(`basicsr/archs/lora_arch.py`, `LHVAE_hflora_arch.py`), a Stage-1 trainer
(`basicsr/models/EQvae_model.py`), a Stage-2 trainer
(`basicsr/models/VAEadapter_model.py`), two example configs, losses, metrics, and the
generic BasicSR `train.py`. The README is a single line (`# Latent-Harmony`).

What I did: read the paper PDF/text and every code file; mapped each paper artefact to
code; wrote two read-only static checks under `_audit_code/`:
`check_registry_vs_configs.py` (every arch/model `type:` in the configs vs the registry)
and `check_repo_completeness.py` (presence of `R_theta`, test/eval scripts, dependency
spec, datasets, weights). Outputs are in `_audit_code/out/`.

Headline observations: (a) the repo contains **no restoration network `R_theta`** at all,
so the Stage-2 pipeline that produces every reported restoration number is not
implemented; (b) **no test/evaluation/inference harness** exists, so none of the metric
values in Tables 1–5 are traceable to code; (c) **no dataset, no pretrained weights, no
dependency specification**; (d) the Stage-2 config references an **unregistered arch
`RAVAE`**, so `stage2_hflora.yml` crashes at model build; (e) several **hardcoded
absolute paths** (`/fs-computility/...`, `/code/...`) bind the code to the authors'
machine. The NeurIPS checklist (item 5) answers **[No]** to open code/data access
("We will release the source code upon acceptance"), consistent with this being a partial
snapshot rather than the result-producing repository.

## 2. Result-traceability table

Every quantitative artefact in the paper, mapped to the code that computes it. "Repo
location" is where the *value-producing* computation would live (not a plotter).

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (4-deg) PSNR/SSIM/LPIPS for "Ours" and 11 baselines | (none — no eval harness; no `R_theta`) | — | — | MISSING |
| Table 2 (6-deg) PSNR/SSIM/LPIPS for "Ours" and baselines | (none) | — | — | MISSING |
| Table 3 standard-resolution LPIPS/FID (PromptIR/Diff-Plugin/CosAE +Ours) | (none — no integration code for these 3 methods) | — | — | MISSING |
| Table 4 generalization PSNR/SSIM/LPIPS (unseen + composite) | (none) | — | — | MISSING |
| Table 5(a) ablation (w/o LInv, LEqv, PDPS, FHF/PHF-LoRA, fine-tuning) | (none — no ablation driver; PDPS/LInv/LEqv code exists only inside Stage-1 trainer) | — | — | MISSING |
| Table 5(b) inference-time comparison (12.3/8.4/1.2/0.43 s) | (none) | — | — | MISSING |
| Table 5(c) latent-restoration-net ablation (Restormer/NAFNet/SFHformer ±Ours) | (none — `R_theta` not in repo) | — | — | MISSING |
| Table 5(d) metrics vs alpha (0.2/0.4/0.6/0.8) | (none — `alpha` blend implemented in `RAVAEHFLora.set_alpha`, but no eval that sweeps it) | — | — | MISSING |
| Fig. 2 motivation panels (t-SNE, CDCS, DCT, loss curves, HF-LoRA bars) | (none — no analysis scripts) | — | — | MISSING |
| FLOPs 3.6G / Params 1.2M ("Ours", Tables 1–2) | `basicsr/archs/LHVAE_arch.py:474-478` (thop profile in `__main__`, hardcoded debug.yml path) | not runnable as-is | — | MISSING/uncheckable |
| Stage-1 training (PDPS + LInv + LEqv) | `basicsr/models/EQvae_model.py:295-336` | trains LH-VAE | n/a (no metric) | Present (code only) |
| Stage-2 HF-LoRA training | `basicsr/models/VAEadapter_model.py:148-210` | fine-tunes VAE only (no `R_theta`) | n/a | Present but incomplete |

Net: **0 of the paper's reported numbers are reproducible from this repo** — there is no
evaluation entrypoint and the core restoration network is absent.

## 3. Findings

## missing

```yaml finding
id: no-restoration-network-rtheta
category: missing
topic: "result traceability / Stage-2 pipeline"
title: "Latent restoration network R_theta absent from entire codebase"
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
claim: "Stage-2 forward passes the degraded input directly through the VAE (net_g = RAVAEHFLora wrapping the VAE); no latent restoration network R_theta is instantiated or applied anywhere in the repo (grep for SFHformer/NAFNet/Restormer-as-restoration/net_r returns nothing in basicsr/models)."
concern: "The paper's Stage 2 (§4.2, Eq. 7-9) and Table 5(c) center on a latent restoration network R_theta (default SFHformer) that maps z_deg to z_res; without it the code cannot perform restoration, so none of the reported PSNR/SSIM/LPIPS numbers can be produced by this repo."
resolution: "Authors: provide the R_theta restoration-network code (SFHformer/NAFNet/Restormer variants from Table 5c) and the script that trains LRes (Eq. 7) and runs it inside the Stage-2 loss path (Eq. 9)."
cross_refs: ["no-test-eval-script", "stage2-ravae-type-unregistered"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "§4.2 Eq. 7-9; Table 5(c)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-test-eval-script
category: missing
topic: "evaluation harness / reproducibility"
title: "No test/evaluation/inference entrypoint; Tables 1-5 and Fig. 2 untraceable"
severity: high
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The repo ships only the generic BasicSR train.py and two example training configs; there is no test.py / inference / evaluation / ablation / analysis script (check_repo_completeness.py: test_eval_infer_scripts = []), and the one-line README gives no commands."
concern: "Every quantitative result in the paper (Tables 1-5, Fig. 2 motivation panels) lacks a value-producing script, so no reported number can be reproduced or checked."
resolution: "Authors: provide the evaluation harness that loads the UHD benchmark test splits and computes the PSNR/SSIM/LPIPS/FID/NIQE/runtime values in Tables 1-5, plus the analysis code for Fig. 2."
cross_refs: ["no-restoration-network-rtheta", "no-data-or-weights"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Tables 1-5; Fig. 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-data-or-weights
category: missing
topic: "datasets and pretrained weights"
title: "No datasets, no pretrained VAE/DINOv2 weights, no fetch script"
severity: high
confidence: high
status: finding
file: configs/stage1_eqvae.yml
line_start: 8
line_end: 11
quote: |
    dataroot_gt: ./datasets/train/gt
    dataroot_lq: ./datasets/train/lq
    io_backend:
      type: disk
claim: "Configs point at ./datasets/... and ./weights/dinov2_vits14.pth / ./weights/stage1_eqvae.pth, but the repo contains no datasets/ or weights/ directory and no *.pth/image files (check_repo_completeness.py: data_or_weight_dirs=[], weight_files=[], image_files=[]); there is no download/fetch script."
concern: "The UHD benchmarks (UHD-LL, UHD-blur, UHD-haze, UHDN, UHD-rain, UHD-snow) and the required DINOv2 weights are neither bundled nor fetchable, so neither stage can be trained or run as shipped."
resolution: "Authors: provide working dataset download/preparation scripts (or accessions) and the DINOv2 / Stage-1 VAE checkpoints the configs reference."
cross_refs: ["no-test-eval-script"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "§5 (UHD benchmarks); configs stage1/stage2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec
category: missing
topic: "environment / dependencies"
title: "No dependency specification (requirements/setup/env); deps unpinned"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The repo has no requirements.txt / setup.py / pyproject.toml / environment.yml / Dockerfile (check_repo_completeness.py: dependency_spec_files=[]); it imports torch, cv2, numpy, einops, torchvision, tqdm, and pyiqa, plus torch.hub DINOv2, none pinned."
concern: "Without a pinned environment the runtime (and especially pyiqa/torch versions that affect metric values) cannot be reconstructed, blocking reproduction."
resolution: "Authors: add a pinned requirements.txt / environment.yml listing torch, pyiqa, einops, opencv, torchvision and their versions."
cross_refs: []
check_script: _audit_code/check_repo_completeness.py
paper_ref: "NeurIPS checklist item 6/8 (env)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-ravae-type-unregistered
category: bug
topic: "config / registry"
title: "Stage-2 config requests unregistered arch 'RAVAE'; model build crashes"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 42
quote: |
    vae_config:
      type: RAVAE
claim: "RAVAEHFLora.__init__ does self.vae = ARCH_REGISTRY.get(arch_type)(**cfg) with arch_type='RAVAE' (LHVAE_hflora_arch.py:37-38); the only registered VAE arch is 'RAVAE_EQ' (registry scan: registered_archs = [LoRAConv2d, RAVAEHFLora, RAVAE_EQ, UNetDiscriminatorSN]). Registry.get raises KeyError for an unknown name (registry.py:62-66)."
concern: "Running `train.py -opt configs/stage2_hflora.yml` raises KeyError(\"No object named 'RAVAE' found in 'arch' registry!\") before any training step, so Stage 2 cannot run as shipped."
resolution: "Authors: set vae_config.type to the actual registered arch (RAVAE_EQ) or register a 'RAVAE' class; confirm which VAE arch produced the paper results."
cross_refs: ["no-restoration-network-rtheta"]
check_script: _audit_code/check_registry_vs_configs.py
paper_ref: "§4.2 (Stage-2 LH-VAE backbone)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-experiments-root
category: bug
topic: "hardcoded absolute paths"
title: "experiments_root hardcoded to authors' machine path"
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
claim: "For any training run, all output dirs (models, logs, checkpoints, visualizations) are forced under the absolute path /fs-computility/ai4sData/liuyidi/model/experiments/<name>, ignoring the run's root_path."
concern: "On any machine other than the authors' cluster, make_exp_dirs/copy_opt_file/save will write to (or fail to create) an inaccessible absolute path, so checkpoints and logs are misplaced or training aborts."
resolution: "Authors: derive experiments_root from root_path (as upstream BasicSR does) instead of a hardcoded cluster path."
cross_refs: ["hardcoded-syspath-appends"]
paper_ref: "n/a (engineering)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-syspath-appends
category: bug
topic: "hardcoded absolute paths"
title: "Multiple sys.path.append to non-existent absolute author paths"
severity: low
confidence: high
status: finding
file: basicsr/archs/encoder_3.py
line_start: 8
line_end: 8
quote: |
  sys.path.append('/code/UHDformer-main')
claim: "Several modules append absolute author-machine paths to sys.path: encoder_3.py:8 ('/code/UHDformer-main'), modules/attention.py:8 ('/code/UHDFour_code-main'), train.py:8 and LHVAE_arch.py:7 ('/fs-computility/ai4sData/liuyidi/code/LatentGen')."
concern: "These reveal the code was carved out of a larger private tree; the appends are no-ops only because nothing under them is imported here, but they signal that the released snapshot is incomplete relative to the authors' working tree."
resolution: "Authors: remove the absolute sys.path.append lines and confirm no hidden dependency lived under those paths."
cross_refs: ["hardcoded-experiments-root"]
paper_ref: "n/a (engineering)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: stage2-decoder-lora-active-during-fhf
category: difference
topic: "Stage-2 alternating optimization"
title: "FHF step keeps decoder LoRA active (alpha=0.5) instead of frozen base decoder"
severity: low
confidence: medium
status: question
file: basicsr/models/VAEadapter_model.py
line_start: 152
line_end: 161
quote: |
          if self._is_fhf_step(current_iter):
              # FHF-LoRA: update encoder LoRA only.
              self._set_training_stage(train_enc=True, train_dec=False, train_disc=False)
              self.optimizer_enc.zero_grad()

              recon, posterior, z = self._forward_g()
              hf_pred = self._hf(recon)
              l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
              l_hf_fid.backward()
              self.optimizer_enc.step()
claim: "During the FHF (encoder-fidelity) step the decoder LoRA is frozen (train_dec=False) but still contributes to the forward pass, because _forward_g calls net_g(lq, alpha=0.5) and set_alpha sets dec_scale=1-alpha=0.5 (LHVAE_hflora_arch.py:83-91), so the decoder is psi*+0.5*delta_psi, not the frozen base psi* of Eq. 8."
concern: "Paper Eq. 8 evaluates the fidelity loss with the decoder at its frozen base parameters D_{psi*}; the code instead applies a half-strength decoder LoRA, a mismatch between the described and implemented FHF objective (the implemented version is still valid, hence difference)."
resolution: "Authors: confirm whether FHF-LoRA training should set alpha=1 (decoder LoRA off) during encoder steps to match Eq. 8, or clarify the intended decoder state."
cross_refs: ["no-restoration-network-rtheta"]
paper_ref: "§4.2 Eq. 8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

N/A — the procedure that produces the paper's results (Stage-2 restoration via `R_theta`
+ HF-LoRA, plus the benchmark evaluation) is not present in the repo, so there is no
implemented end-to-end procedure to assess for methodological validity. The Stage-1 and
partial Stage-2 code that *is* present (PDPS, LInv, LEqv, HF-LoRA losses) is internally
consistent with the paper's equations as far as it goes; issues are absence (`missing`) and
breakage (`bug`), not an invalid-but-running procedure. No data leakage or invalid-split
concern is assessable because no split-generation or evaluation code exists.

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 4 | high | No R_theta, no eval harness, no data/weights, no dependency spec |
| bug | 3 | high | Stage-2 config uses unregistered arch RAVAE; hardcoded absolute paths |
| difference | 1 | low | FHF step keeps decoder LoRA half-active vs Eq. 8's frozen decoder (question) |
| methodology | 0 | - | Not assessable: result-producing pipeline absent |

## 5. Closing lists

### Top take-aways (ranked by severity x confidence)
1. **[missing] no-restoration-network-rtheta** — the latent restoration network `R_theta`
   (the core of Stage 2 and Table 5c) is not in the repo; the code cannot restore images.
2. **[missing] no-test-eval-script** — no evaluation/inference/ablation harness exists, so
   none of Tables 1-5 or Fig. 2 are traceable to code.
3. **[bug] stage2-ravae-type-unregistered** — `stage2_hflora.yml` requests arch `RAVAE`,
   which is unregistered; `train.py` on this config crashes with `KeyError` at model build.
4. **[missing] no-data-or-weights** — no datasets, no DINOv2/VAE checkpoints, no fetch
   scripts; neither stage runs as shipped.
5. **[bug] hardcoded-experiments-root** — all training outputs are forced under a hardcoded
   `/fs-computility/...` cluster path, breaking runs on any other machine.
6. **[missing] no-dependency-spec** — no pinned environment; metric-affecting deps
   (pyiqa/torch) cannot be reconstructed.

### Items that genuinely look fine
- Stage-1 trainer (`EQvae_model.py`) implements PDPS Eq. 3 with `gt`=clean / `lq`=real
  degraded mapping, `LInv` aligning the perturbed-input latent to DINOv2(clean) features
  (Eq. 4), and `LEqv` decoding a downsampled latent against the downsampled clean image
  (Eq. 5) — all consistent with the paper.
- The loss classes the configs reference (`L1Loss`, `KlLoss`, `GANLoss`) and the archs
  `RAVAE_EQ`, `RAVAEHFLora`, `UNetDiscriminatorSN`, `LoRAConv2d` are all present and
  correctly registered (registry scan).
- `LoRAConv2d` correctly freezes the base conv and exposes only low-rank params; the
  encoder/decoder `alpha` blend (`set_alpha`) matches the inference-time control of §4.3.
- The HF extraction (`hf_ops.extract_hf_fft`) is a standard FFT high-pass and matches the
  HF-alignment intent of Eq. 8.

### Open questions for the authors
- Is the cloned `lyd-2022/Latent-Harmony` the repository that produced the paper's numbers,
  or an early partial snapshot? (NeurIPS checklist item 5 = [No], code "released upon
  acceptance"; abstract promises code at the same URL.)
- For Stage 2, should `alpha` be set to 1 during FHF steps so the decoder is at its frozen
  base (Eq. 8)? (finding `stage2-decoder-lora-active-during-fhf`)
- Which VAE arch (`RAVAE_EQ` vs an unreleased `RAVAE`) and which `R_theta` configuration
  produced the headline 3.6G / 1.2M efficiency numbers in Tables 1-2?
