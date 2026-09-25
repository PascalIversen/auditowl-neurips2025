# Audit — Latent Harmony (paper 1333)

## 1. Summary

The repo `code/lyd-2022__Latent-Harmony/` is a BasicSR-style image-restoration
training framework. It contains exactly two training entry configurations
(`configs/stage1_eqvae.yml`, `configs/stage2_hflora.yml`), one generic training
loop (`basicsr/train.py`), two custom model trainers
(`basicsr/models/EQvae_model.py` = Stage-1 LH-VAE, `basicsr/models/VAEadapter_model.py`
= Stage-2 HF-LoRA), and supporting VAE / LoRA / discriminator architectures.
The README is empty, there is no dependency specification, and there is no
evaluation/inference/metric-computation script of any kind.

What I did: I read the paper (PDF + text extraction) and mapped each headline
artefact (Tables 1–5) to repo code; I read every model/arch/loss/util file that
is exercised by the two configs; and I ran two read-only static checks under
`_audit_code/` — `check_arch_registry.py` (which arch/model names are actually
registered vs referenced by the configs) and `check_completeness.py` (presence of
dependency specs, eval scripts, and the latent restoration network Rθ). I did
not execute the repo (no GPU, no data, no weights, and Stage-2 cannot even
construct its network — see `stage2-ravae-arch-unregistered`).

The dominant problems are completeness/reproducibility: the central Stage-2
component (latent restoration network Rθ, paper Eq. 7 and Eq. 9) is absent, no
script computes any reported number, no dependencies are listed, and the Stage-2
config crashes at network construction because it names an unregistered arch
(`RAVAE`).

## 2. Result-traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (4-deg) PSNR/SSIM/LPIPS, "Ours" | (none — no eval/metric script) | — | — | MISSING |
| Table 2 (6-deg) PSNR/SSIM/LPIPS, "Ours" | (none) | — | — | MISSING |
| Table 3 standard-res (PromptIR/Diff-Plugin/CosAE "/w Ours") | (none — no integration code) | — | — | MISSING |
| Table 4 generalization (unseen / composite) | (none) | — | — | MISSING |
| Table 5(a) ablation (w/o LInv, LEqv, PDPS, FHF/PHF-LoRA, …) | partial: losses exist in `EQvae_model.py`/`VAEadapter_model.py`; no ablation driver or eval | — | — | MISSING (no ablation/eval harness) |
| Table 5(b) inference time (LH 0.43 s) | (none — no timing/inference script) | — | — | MISSING |
| Table 5(c) backbone ablation (Restormer/NAFNet/SFHformer + Ours) | (none — Rθ network absent) | — | — | MISSING |
| Table 5(d) α sweep PSNR/SSIM/LPIPS/User | α plumbing exists (`RAVAEHFLora.set_alpha`); no eval producing the numbers | — | — | MISSING (no eval) |
| Eq. 2 LVAE (L1 + KL) | `EQvae_model.py:318-328` | code present | n/a (not a number) | Verified (code) |
| Eq. 3 PDPS perturbation | `EQvae_model.py:206-224`, `186-204` | code present | n/a | Verified (code) |
| Eq. 4 LInv (DINOv2 alignment) | `EQvae_model.py:270-281` | code present | n/a | Verified (code) |
| Eq. 5 LEqv (equivariance) | `EQvae_model.py:283-293` | code present | n/a | Verified (code) |
| Eq. 7 LRes (Rθ restoration net) | (none — Rθ not implemented) | — | — | MISSING |
| Eq. 8 LHFFid (encoder FHF-LoRA) | `VAEadapter_model.py:152-167` | code present (but no Rθ in path) | partial | see findings |
| Eq. 9 LHFGAN (decoder PHF-LoRA) | `VAEadapter_model.py:168-210` | code present; decoder also gets L1 HF-fid loss | ✗ (paper: GAN only) | MISMATCH → difference |

Every quantitative artefact in the paper (all of Tables 1–5) is MISSING from the
code: there is no script that computes PSNR/SSIM/LPIPS/FID/NIQE/runtime, and no
inference entry point.

## 3. Findings

## missing

```yaml finding
id: missing-eval-inference-code
category: missing
topic: "result traceability"
title: "No evaluation/inference/metric script: none of Tables 1-5 can be reproduced"
severity: high
confidence: high
status: finding
file: basicsr/train.py
line_start: 215
line_end: 217
quote: |
  if __name__ == '__main__':
      root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
      train_pipeline(root_path)
claim: "The only executable entry point in the repo is train.py (a training loop). check_completeness.py finds zero test/inference/eval scripts; the only __main__ files are train.py, diffjpeg.py, LHVAE_arch.py, Res_four.py."
concern: "Every reported number (PSNR/SSIM/LPIPS in Tables 1-2, FID/LPIPS in Table 3, generalization in Table 4, ablations and runtime in Table 5) has no producing script, so none of the paper's quantitative claims can be reproduced or checked."
resolution: "Provide the evaluation/inference scripts that compute the metrics in Tables 1-5 (with the exact commands and the datasets/weights they consume)."
cross_refs: ["missing-rtheta-restoration-net", "missing-dependency-spec"]
check_script: _audit_code/check_completeness.py
paper_ref: "Tables 1-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-rtheta-restoration-net
category: missing
topic: "method completeness"
title: "Latent restoration network Rθ (Stage-2 core, Eq. 7/9) is absent"
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
claim: "The Stage-2 network RAVAEHFLora is just a frozen VAE plus encoder/decoder LoRA; its forward only does VAE encode->decode of the degraded input. There is no latent restoration network Rθ anywhere (check_completeness.py: rtheta_restoration_net_refs => NONE), so zres = Rθ(zdeg) (Eq. 7) and Rθ(Eϕ*(Ideg)) (Eq. 9) are not implemented."
concern: "Stage Two's headline mechanism — a latent restoration network Rθ that denoises the degraded latent before decoding — is missing, so the released code cannot perform the restoration the paper evaluates; the Table 5(c) Restormer/NAFNet/SFHformer-as-Rθ ablation is also unimplementable."
resolution: "Release the Rθ latent restoration network, its Stage-2-pre training (Eq. 7), and how it is wired into the HF-LoRA forward pass."
cross_refs: ["missing-eval-inference-code"]
check_script: _audit_code/check_completeness.py
paper_ref: "Section 4.2, Eq. 7 and Eq. 9; Table 5(c)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-dependency-spec
category: missing
topic: "expected code completeness"
title: "No dependency specification (requirements/setup/env) for a multi-dependency repo"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 9
line_end: 13
quote: |
  import pyiqa

  from basicsr.archs import build_network
  from basicsr.losses import build_loss
  from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
claim: "The repo imports third-party packages (pyiqa, torch, torchvision, tqdm) and downloads DINOv2 via torch.hub.load('facebookresearch/dinov2', ...) (EQvae_model.py:115), yet check_completeness.py finds dependency_spec_files => NONE (no requirements.txt, setup.py, environment.yml, pyproject.toml)."
concern: "Without pinned dependencies the environment cannot be reliably rebuilt; unpinned pyiqa/torch versions can change metric values and break the (frozen) APIs used here."
resolution: "Add a requirements.txt / environment.yml pinning torch, torchvision, pyiqa, tqdm, numpy, pyyaml and the DINOv2 hub revision."
cross_refs: []
check_script: _audit_code/check_completeness.py
paper_ref: "NeurIPS checklist Q5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-weights-and-data-and-readme
category: missing
topic: "data / weights / availability"
title: "Empty README, no pretrained weights or datasets, code-availability deferred"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  Answer:[No]
  Justification:We will release the source code upon acceptance of the paper.
claim: "The NeurIPS checklist (Q5) states code will be released only upon acceptance; the README is empty (0 lines beyond the title), and the configs point to absent artefacts: DINOv2 weights ./weights/dinov2_vits14.pth (stage1) and the Stage-1 VAE checkpoint ./weights/stage1_eqvae.pth (stage2), plus datasets ./datasets/train|val that are not provided and have no fetch script."
concern: "No reproduction instructions, no trained weights, and no resolvable dataset path means a reader cannot run either stage end-to-end as released."
resolution: "Provide a README with exact commands, the DINOv2 and Stage-1 VAE checkpoints (or a download script), and dataset preparation instructions/links."
cross_refs: ["missing-dependency-spec"]
paper_ref: "NeurIPS checklist Q5; configs/stage1_eqvae.yml:94-95, configs/stage2_hflora.yml:35"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-standardres-and-generalization-experiments
category: missing
topic: "result traceability"
title: "Standard-resolution integration (Table 3) and generalization (Table 4) experiments have no code"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 18
line_end: 20
quote: |
  @MODEL_REGISTRY.register()
  class VAEadapter(BaseModel):
      """Stage-2 trainer for HF-LoRA alternating optimization."""
claim: "The repo contains only the two stage trainers (EQVAEModel, VAEadapter). There is no code integrating LH-VAE into PromptIR / Diff-Plugin / CosAE (Table 3) nor any unseen/composite-degradation generalization harness (Table 4)."
concern: "Two full experimental sections of the paper (standard-resolution versatility and generalization) cannot be reproduced because their drivers are absent."
resolution: "Release the PromptIR/Diff-Plugin/CosAE integration code and the unseen/composite-degradation evaluation scripts."
cross_refs: ["missing-eval-inference-code"]
paper_ref: "Tables 3 and 4 (Sections 5.2)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-ravae-arch-unregistered
category: bug
topic: "configuration / registry"
title: "Stage-2 config names unregistered arch 'RAVAE' -> KeyError at network build"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 42
quote: |
  vae_config:
    type: RAVAE
claim: "stage2_hflora.yml sets vae_config.type: RAVAE, and RAVAEHFLora.__init__ resolves it via ARCH_REGISTRY.get(arch_type)(**cfg) (LHVAE_hflora_arch.py:37-38). check_arch_registry.py shows the only registered archs are LoRAConv2d, RAVAEHFLora, RAVAE_EQ, UNetDiscriminatorSN — 'RAVAE' is NOT registered, and Registry.get raises KeyError for missing names (registry.py:62-66)."
concern: "Stage-2 training crashes during network construction (KeyError: No object named 'RAVAE'), so the released Stage-2 pipeline cannot run with its own example config."
resolution: "Register a 'RAVAE' arch or change the config to 'RAVAE_EQ' (the only compatible VAE); confirm the intended base VAE."
cross_refs: []
check_script: _audit_code/check_arch_registry.py
paper_ref: "configs/stage2_hflora.yml"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-experiments-root
category: bug
topic: "hardcoded paths"
title: "Training output root hardcoded to an author-specific absolute path"
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
claim: "For every training run the experiments root is forced to the absolute path /fs-computility/ai4sData/liuyidi/model/experiments/<name>, overriding the root_path-relative default that standard BasicSR uses."
concern: "On any machine other than the authors' cluster this path does not exist / is not writable, so checkpoints, logs, and visualizations cannot be saved (or training errors when creating dirs), blocking reproduction."
resolution: "Derive experiments_root from root_path (e.g. osp.join(root_path, 'experiments', opt['name'])) or make it configurable."
cross_refs: ["hardcoded-syspath-author-dirs"]
paper_ref: null
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-syspath-author-dirs
category: bug
topic: "hardcoded paths"
title: "sys.path.append / open() to author-only absolute directories in core modules"
severity: low
confidence: high
status: finding
file: basicsr/archs/LHVAE_arch.py
line_start: 6
line_end: 7
quote: |
  import sys
  sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
claim: "Several imported modules append non-existent author-machine paths at import time: LHVAE_arch.py:7 and train.py:8 ('/fs-computility/ai4sData/liuyidi/code/LatentGen'), attention.py:8 ('/code/UHDFour_code-main'), encoder_3.py:8 ('/code/UHDformer-main'); LHVAE_arch.py:467 also opens '/fs-computility/.../debug.yml' in its __main__."
concern: "These appends are dead on other machines (silently no-ops at import but signal the code was never decoupled from the authors' filesystem; the __main__ open() crashes), indicating the released code was not made portable."
resolution: "Remove the hardcoded sys.path.append/open lines or replace with repo-relative imports."
cross_refs: ["hardcoded-experiments-root"]
paper_ref: null
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: decoder-lora-gets-fidelity-loss
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Decoder PHF-LoRA is trained with the HF fidelity L1 loss in addition to the GAN loss"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 188
line_end: 202
quote: |
            # 2) Decoder LoRA (generator) step
            self.optimizer_dec.zero_grad()
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
claim: "In the PHF-LoRA (decoder) step the code optimizes l_hf_fid (high-frequency L1 fidelity loss) plus the adversarial GAN loss; the fidelity term dominates (lambda_hf_fid=1.0 vs lambda_gan=0.1)."
concern: "Paper Eq. 9 specifies the decoder LoRA is driven solely by the perception-oriented GAN loss LHFGAN; adding the fidelity L1 term contradicts the paper's clean perception/fidelity decoupling and changes what PHF-LoRA optimizes."
resolution: "Confirm whether the decoder LoRA should be optimized by the GAN loss alone (Eq. 9); if the L1 HF-fidelity term is intended, document it in the method."
cross_refs: []
paper_ref: "Section 4.2, Eq. 9"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: training-alpha-not-freezing-opposite-lora
category: difference
topic: "training procedure (paper vs code)"
title: "During each LoRA step the opposite LoRA is scaled by alpha=0.5, not frozen at base"
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
claim: "Stage-2 always forwards with alpha = train.alpha = 0.5 (VAEadapter_model.py:54, :133), so during the encoder (FHF) step the decoder LoRA still contributes at scale 0.5 and during the decoder (PHF) step the encoder LoRA contributes at scale 0.5."
concern: "Paper Eq. 8 states that when optimizing FHF-LoRA the decoder uses its frozen base parameters ψ* (and symmetrically for PHF-LoRA); applying the opposite LoRA at scale 0.5 during training deviates from that frozen-base description (the gradient path differs from the equations)."
resolution: "Clarify whether the opposite LoRA should be disabled (scale 0) during each alternating step, matching Eq. 8's ϕ*/ψ* frozen-base formulation."
cross_refs: []
paper_ref: "Section 4.2, Eq. 8; Section 4.3 (α)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No standalone methodology finding. Standard leakage / data-splitting / sample-
independence checks are largely N/A: this is paired image restoration with
separate train/val dataset roots and per-image L1/HF losses (no cross-sample
splitting in-repo). The most consequential methodological concern — that no
held-out test evaluation or naive baseline is computed in-repo — is owned by
`missing-eval-inference-code` (the metrics simply have no producing code), so it
is cross-referenced rather than re-filed here.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | No eval code, no Rθ network, no deps/weights/data; whole Tables 1-5 untraceable |
| bug         | 3          | high         | Stage-2 crashes on unregistered 'RAVAE'; hardcoded author-only output/import paths |
| difference  | 2          | medium       | Decoder LoRA also gets HF-fidelity L1 (Eq.9 says GAN only); alpha not freezing opposite LoRA |
| methodology | 0          | -            | Splitting/leakage checks N/A; eval-absence owned by missing |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. [missing] `missing-eval-inference-code` — no script computes any number in Tables 1-5 (high/high).
2. [missing] `missing-rtheta-restoration-net` — the Stage-2 latent restoration network Rθ (Eq. 7/9) is entirely absent (high/high).
3. [bug] `stage2-ravae-arch-unregistered` — Stage-2 config names unregistered arch `RAVAE`, crashing at build (high/high).
4. [missing] `missing-dependency-spec` — no requirements/setup/env; environment not rebuildable (medium/high).
5. [bug] `hardcoded-experiments-root` — training output path forced to an author-only absolute directory (medium/high).
6. [difference] `decoder-lora-gets-fidelity-loss` — PHF-LoRA decoder also optimizes HF-fidelity L1, not GAN-only (medium/high).

### Items that genuinely look fine
- Stage-1 losses faithfully implement the paper: PDPS (Eq. 3) in `EQvae_model.py:186-224`, LInv vs DINOv2(Iclean) (Eq. 4) in `:270-281`, LEqv (Eq. 5) in `:283-293`, LVAE = L1 + λKL·KL (Eq. 2) in `:318-328`.
- `extract_hf_fft` (`hf_ops.py`) is a standard FFT high-pass; LoRAConv2d (`lora_arch.py`) is a correct, zero-initialized LoRA wrapper.
- DiffJPEG.forward accepts a scalar `quality` (diffjpeg.py:467), so the PDPS JPEG call `_apply_synth_deg` (`EQvae_model.py:200-202`) is wired correctly.
- α inference control (ϕ=ϕ*+α∆ϕ, ψ=ψ*+(1-α)∆ψ) matches the paper's Section 4.3 description (`LHVAE_hflora_arch.py:83-91`).

### Open questions for the authors
- Where is the latent restoration network Rθ and its Stage-2-pre training (Eq. 7), and how is it inserted into the HF-LoRA forward pass? (drives `missing-rtheta-restoration-net`)
- Which base VAE arch is `RAVAE` meant to be (is `RAVAE_EQ` the intended Stage-2 backbone)? (drives `stage2-ravae-arch-unregistered`)
- Is the decoder PHF-LoRA intended to optimize the GAN loss alone (Eq. 9) or the GAN + HF-fidelity combination implemented in code? (drives `decoder-lora-gets-fidelity-loss`)
