# Code-repository audit — Latent Harmony (NeurIPS 2025, #1333)

## 1. Summary

The paper proposes **Latent Harmony (LH)**, a two-stage UHD all-in-one image-restoration
framework: Stage 1 trains a regularized VAE (LH-VAE) with progressive degradation
perturbation (PDPS), a DINOv2 semantic-invariance loss (`LInv`) and a latent-equivariance
loss (`LEqv`); Stage 2 trains a latent restoration network `Rθ` (paper says SFHformer) with a
frozen VAE and then applies high-frequency LoRA (FHF-LoRA on the encoder, PHF-LoRA on the
decoder) with an inference-time fidelity/perception knob `α`. The paper reports SOTA on
4-/6-degradation UHD benchmarks (Tables 1–2), standard-resolution generalization (Tables 3–4),
ablations (Table 5a/c/d), efficiency (Table 5b), and a user study.

The cloned repo (`lyd-2022/Latent-Harmony`, single commit `da910f4` "first", dated 2026-02-18,
untagged `main`, 0.4 MB) is a **BasicSR fork** containing only: model archs
(`RAVAE_EQ`, `RAVAEHFLora`, `LoRAConv2d`, `UNetDiscriminatorSN`), Stage-1/Stage-2 trainers
(`EQVAEModel`, `VAEadapter`), losses, a paired-image dataset, `basicsr/train.py`, and two
**example** YAML configs with placeholder dataset paths. The paper's own NeurIPS checklist
(Q5) states "[No] … We will release the source code upon acceptance," so this is a
preliminary/partial upload.

What I did: read the paper (PDF + text extraction) and every Python file under `basicsr/` and
both configs. I ran two deterministic checks under `_audit_code/`:
`check_registry_vs_configs.py` (registered class names vs `type:` references in configs) and
`check_completeness.py` (presence of dependency spec / license / eval entrypoint / restoration
arch / weights / datasets, plus hardcoded-absolute-path scan). Outputs are in
`_audit_code/out/`. Findings below.

## 2. Traceability table

Every quantitative artefact in the paper is traced to the code that would *compute* it. There
is **no evaluation/inference entrypoint, no metric configuration (`metrics: ~` in both
configs), no pretrained weights, no datasets, no baseline code, and no FLOPs/params/runtime
script** (the only `thop.profile` call lives in a dead `__main__` block of `LHVAE_arch.py:465-481`
that opens a hardcoded path). Consequently every numeric result is MISSING from the repo.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (4-deg) PSNR/SSIM/LPIPS, all rows incl. "Ours" 29.70/.877/.2502 | (none) | — | — | MISSING (no eval driver, no metrics, no weights) |
| Table 2 (6-deg) PSNR/SSIM/LPIPS, "Ours" 29.24/.920/.1822 | (none) | — | — | MISSING |
| Table 1/2 FLOPs (3.6G) & Params (1.2M) for "Ours" | (none; thop call only in dead `__main__`) | — | — | MISSING |
| Table 3 standard-res LPIPS/FID (PromptIR/Diff-Plugin/CosAE +Ours) | (none) | — | — | MISSING (no integration code, no baselines) |
| Table 4 generalization PSNR/SSIM/LPIPS (unseen + composite) | (none) | — | — | MISSING |
| Table 5a ablation (w/o LInv, LEqv, PDPS, FHF/PHF-LoRA, …) | components exist in `EQVAEModel`/`VAEadapter`; no ablation harness/driver | — | — | MISSING (no ablation runner, no metrics) |
| Table 5b inference time (LH 0.43 s) | (none) | — | — | MISSING |
| Table 5c restoration-net ablation (Restormer/NAFNet/SFHformer Base vs +Ours) | (none); SFHformer/NAFNet arch not in repo, `Rθ` not wired | — | — | MISSING (see `missing-restoration-network-rtheta`) |
| Table 5d α-sweep PSNR/SSIM/LPIPS/User | `set_alpha` in `LHVAE_hflora_arch.py:83-91` implements α blend; no sweep/eval driver | — | — | MISSING (mechanism present, no driver/metrics) |
| Fig. 2 motivation (t-SNE, CDCS, DCT, loss curves, HF-LoRA radar) | (none) | — | — | MISSING |
| Fig. 4 visual results | (none, no weights/data) | — | — | MISSING |
| User study scores (Fig.2e, Table 5d) | (none) | — | — | MISSING |

Because not a single reported number is reproducible from the repo as shipped, the dominant
issue is `missing`. The remaining findings (`bug`, `difference`) concern correctness/fidelity
of the code that *is* present.

## 3. Findings

## missing

```yaml finding
id: no-eval-or-reproduction-pipeline
category: missing
topic: "result traceability"
title: "No evaluation/inference code, metrics, weights, datasets, or baselines for any table/figure"
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
claim: "The repo ships only two BasicSR training configs (with metrics disabled) and basicsr/train.py; there is no test/eval/inference entrypoint, no pretrained weights, no datasets, no FLOPs/params/runtime script, and no baseline implementations."
concern: "None of the paper's headline numbers (Tables 1-5, Figs 2/4, user study) can be reproduced or even recomputed from the repo, so every quantitative claim is untraceable to code."
resolution: "Release the evaluation pipeline (metric configs, dataset preparation, pretrained checkpoints, FLOPs/runtime scripts) and the baseline code/configs used for the comparison tables."
cross_refs: ["missing-restoration-network-rtheta", "missing-deps-license"]
check_script: _audit_code/check_completeness.py
paper_ref: "Tables 1-5; Figures 2,4; NeurIPS checklist Q5 ([No], release upon acceptance)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-restoration-network-rtheta
category: missing
topic: "Stage-2 restoration network"
title: "Stage-2 latent restoration network Rθ (and its LRes training) absent from the code"
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
claim: "The Stage-2 trainer forwards the degraded input straight through the LoRA-wrapped VAE (net_g is RAVAEHFLora, whose forward is just self.vae(x)); there is no latent restoration network Rθ, no SFHformer/NAFNet/Restormer restorer, and no LRes pre-training step (paper Eq.7)."
concern: "The paper's Stage-2 method (Eqs.7-9) hinges on Rθ predicting a restored latent z_res=Rθ(z_deg); without Rθ the released stage-2 code optimizes VAE reconstruction of the degraded image, not the restoration pipeline the paper evaluates, so the core method and Table 5c (Restormer/NAFNet/SFHformer +Ours) cannot be reproduced."
resolution: "Provide the Rθ architecture(s), the LRes pre-training script (Eq.7), and the code wiring Rθ into the HF-LoRA loop (Eq.9)."
cross_refs: ["no-eval-or-reproduction-pipeline", "stage2-hf-loss-omits-rtheta"]
check_script: _audit_code/check_completeness.py
paper_ref: "Section 4.2, Eqs. 7-9; Table 5c"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-deps-license
category: missing
topic: "expected code completeness"
title: "No dependency specification, README instructions, or license"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "The README is a single title line; there is no requirements.txt/environment.yml/setup.py/pyproject.toml, no run instructions, and no LICENSE, despite non-trivial deps (torch, pyiqa, DINOv2 via torch.hub, thop)."
concern: "The environment cannot be rebuilt deterministically and there is no documented command to run anything, blocking reproduction."
resolution: "Add a pinned dependency file, a README with exact reproduction commands and a results table, and a license."
cross_refs: ["no-eval-or-reproduction-pipeline"]
check_script: _audit_code/check_completeness.py
paper_ref: "NeurIPS checklist Q4/Q5"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-dinov2-weights-no-fetch
category: missing
topic: "external assets / DINOv2"
title: "DINOv2 semantic-loss weights required by Stage-1 but not provided or fetchable"
severity: medium
confidence: high
status: finding
file: basicsr/models/EQvae_model.py
line_start: 110
line_end: 116
quote: |
  def _init_dino_backbone(self):
      if not self.dino_local_weight_path:
          raise ValueError('Stage-1 LInv enabled but train.dino.local_weight_path is empty.')

      logger = get_root_logger()
      dino_model = torch.hub.load('facebookresearch/dinov2', self.dino_model_name, pretrained=False)
      state = torch.load(self.dino_local_weight_path, map_location='cpu')
claim: "Stage-1 LInv (paper Eq.4) loads DINOv2 from a local weight path (config default ./weights/dinov2_vits14.pth); the repo ships no weights and no download script, and raises if the path is empty/missing."
concern: "Stage-1 training cannot run as configured without an externally obtained DINOv2 checkpoint that is neither bundled nor fetched, so the LInv regularizer (an ablated component, Table 5a) is not reproducible out of the box."
resolution: "Bundle a fetch script / documented URL+hash for the DINOv2 checkpoint, or wire pretrained=True so torch.hub downloads it."
cross_refs: ["no-eval-or-reproduction-pipeline"]
paper_ref: "Section 4.1, Eq. 4 (LInv via DINOv2)"
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-experiments-root
category: bug
topic: "hardcoded absolute paths"
title: "Output directory hardcoded to the authors' cluster path; training cannot start elsewhere"
severity: high
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
claim: "parse_options() (is_train branch) overrides the experiments root to the absolute path /fs-computility/ai4sData/liuyidi/model/experiments/<name> instead of the repo-relative path used by upstream BasicSR."
concern: "On any other machine this path does not exist; make_exp_dirs() and copy_opt_file() write logs/checkpoints there, so train.py fails at startup for any external user."
resolution: "Restore the BasicSR default osp.join(root_path, 'experiments', opt['name']) or make the root configurable."
cross_refs: ["hardcoded-syspath-debug-yml"]
check_script: _audit_code/check_completeness.py
paper_ref: "N/A (engineering)"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage2-vae-type-unregistered
category: bug
topic: "config/registry mismatch"
title: "Stage-2 config requests VAE type 'RAVAE', which is not a registered arch"
severity: high
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 41
line_end: 43
quote: |
  vae_config:
    type: RAVAE
    embed_dim: 4
claim: "stage2_hflora.yml asks RAVAEHFLora to build an inner VAE of type 'RAVAE', but the only registered arch classes are RAVAE_EQ, RAVAEHFLora, LoRAConv2d, UNetDiscriminatorSN (see _audit_code/out/registry_vs_configs.json)."
concern: "ARCH_REGISTRY.get('RAVAE') raises KeyError, so Stage-2 cannot be instantiated with the shipped config; the example config does not match the registered code."
resolution: "Change vae_config.type to RAVAE_EQ (or register a RAVAE class) so the Stage-2 example config runs."
cross_refs: []
check_script: _audit_code/check_registry_vs_configs.py
paper_ref: "Section 4.2"
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-syspath-debug-yml
category: bug
topic: "hardcoded absolute paths"
title: "sys.path.append and debug.yml hardcoded to the authors' cluster filesystem"
severity: low
confidence: high
status: finding
file: basicsr/archs/LHVAE_arch.py
line_start: 6
line_end: 7
quote: |
  import sys
  sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
claim: "LHVAE_arch.py and train.py both append /fs-computility/ai4sData/liuyidi/code/LatentGen to sys.path, and the LHVAE_arch.py __main__ block opens /fs-computility/.../options/debug.yml (line 467)."
concern: "These absolute paths only exist on the authors' cluster; the sys.path hack and the dead __main__ smoke test are non-portable artefacts (lower severity since the appended path is not strictly required for import of the audited modules)."
resolution: "Remove the hardcoded sys.path.append lines and the dead __main__ block referencing debug.yml."
cross_refs: ["hardcoded-experiments-root"]
check_script: _audit_code/check_completeness.py
paper_ref: "N/A (engineering)"
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: stage2-hf-loss-omits-rtheta
category: difference
topic: "Stage-2 HF-LoRA loss vs paper Eqs. 8-9"
title: "FHF/PHF-LoRA losses computed on VAE self-reconstruction, not through Rθ as in Eqs. 8-9"
severity: medium
confidence: high
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 157
line_end: 160
quote: |
          recon, posterior, z = self._forward_g()
          hf_pred = self._hf(recon)
          l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
          l_hf_fid.backward()
claim: "The Stage-2 HF fidelity loss is L1 between HF(vae(lq)) and HF(gt); the GAN branch likewise discriminates HF(vae(lq)). Paper Eq.8 uses Dψ*(Eϕ*+Δϕ(Ideg)) (no Rθ in the fidelity term) but Eq.9 explicitly routes through Rθ: HF(Dψ*+Δψ(Rθ(Eϕ*(Ideg)))), which the code cannot do because Rθ is absent."
concern: "What the code implements (VAE reconstruction of the degraded image) differs from the paper's described Stage-2 objective, which restores via Rθ; the implemented version is internally consistent but is not the evaluated method."
resolution: "Confirm whether the released stage-2 trainer is the one used for the paper, and add the Rθ-in-the-loop variant matching Eq.9."
cross_refs: ["missing-restoration-network-rtheta"]
paper_ref: "Section 4.2, Eqs. 8-9"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fhf-step-decoder-lora-active
category: difference
topic: "alternating optimization / frozen decoder"
title: "During FHF (encoder) step the decoder LoRA is still applied (α=0.5), unlike Eq.8's frozen-base decoder Dψ*"
severity: low
confidence: medium
status: finding
file: basicsr/models/VAEadapter_model.py
line_start: 152
line_end: 159
quote: |
        if self._is_fhf_step(current_iter):
            # FHF-LoRA: update encoder LoRA only.
            self._set_training_stage(train_enc=True, train_dec=False, train_disc=False)
            self.optimizer_enc.zero_grad()

            recon, posterior, z = self._forward_g()
            hf_pred = self._hf(recon)
            l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
claim: "_forward_g uses the model's self.alpha (config train.alpha=0.5), so set_alpha(0.5) scales decoder LoRA by 1-α=0.5 during the encoder-only step; the forward pass therefore goes through a LoRA-modified decoder, not the frozen base decoder Dψ* of Eq.8 (decoder LoRA is non-trainable here, but still contributes to the forward output)."
concern: "Eq.8 specifies the decoder uses frozen base parameters ψ* when optimizing the encoder LoRA; the code instead keeps the decoder LoRA active at scale 0.5, a benign-but-different procedure (low severity, marked medium confidence due to ambiguity over intended α scheduling during training vs inference)."
resolution: "Clarify the intended training-time α: should the FHF step use α=1 (decoder LoRA off) to match Eq.8's Dψ*? Pass a step-specific alpha if so."
cross_refs: ["stage2-hf-loss-omits-rtheta"]
paper_ref: "Section 4.2, Eq. 8; Section 4.3 (α)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A — the procedure the code *actually implements* (a regularized VAE trainer plus an HF-LoRA
fine-tuner) is methodologically reasonable for what it does; the substantive problems are that
the core restoration network is missing (routed to `missing`) and that what is present diverges
from the paper (routed to `difference`). With no eval pipeline, dataset, or held-out protocol
in the repo, there is nothing implemented whose *validity* could be assessed (no leakage path,
no metric computation, no split logic to inspect). No methodology finding is asserted rather
than invented.

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 4 | high | No eval pipeline/weights/data; Stage-2 restoration network Rθ entirely absent. |
| bug | 3 | high | Hardcoded cluster output path blocks training; Stage-2 config requests unregistered `RAVAE`. |
| difference | 2 | medium | Stage-2 HF losses run on VAE self-recon (no Rθ), unlike paper Eqs. 8-9. |
| methodology | 0 | - | Nothing evaluable is implemented; no invented findings. |

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `no-eval-or-reproduction-pipeline`** — no evaluation/inference code, metrics,
   weights, datasets, baselines, or FLOPs/runtime scripts; every number in Tables 1–5 and
   Figs 2/4 is untraceable. (high/high)
2. **[missing] `missing-restoration-network-rtheta`** — the Stage-2 latent restoration network
   `Rθ` (paper's SFHformer/NAFNet/Restormer) and its `LRes` pre-training are absent; the
   shipped Stage-2 code optimizes VAE reconstruction, not restoration. (high/high)
3. **[bug] `hardcoded-experiments-root`** — `options.py` forces the output root to
   `/fs-computility/ai4sData/liuyidi/model/...`, so `train.py` cannot start on any other
   machine. (high/high)
4. **[bug] `stage2-vae-type-unregistered`** — the Stage-2 example config requests VAE
   `type: RAVAE`, which is not a registered arch; instantiation raises `KeyError`. (high/high)
5. **[difference] `stage2-hf-loss-omits-rtheta`** — FHF/PHF-LoRA losses are computed on
   `HF(vae(lq))`, not through `Rθ` as Eqs. 8–9 specify. (medium/high)
6. **[missing] `missing-deps-license`** — no dependency spec, run instructions, or license.
   (medium/high)

### Items that genuinely look fine
- The LoRA wrapper (`lora_arch.py`) is correct: zero-init `lora_up`, frozen base conv, additive
  `scale*runtime_scale` path, fuse/unfuse via the right einsum.
- The α inference blend (`set_alpha`: enc←α, dec←1−α) matches the paper's
  `ϕ=ϕ*+αΔϕ, ψ=ψ*+(1−α)Δψ` (Section 4.3).
- PDPS in `EQvae_model.py` implements the three-branch perturbation of Eq.3 with a
  monotone severity schedule `t=iter/total_iter`, and the probabilities are renormalized.
- `extract_hf_fft` is a sound FFT high-pass (fftshift, radial mask, ifft real part).
- Stage-1 `LEqv` (`_compute_eqv_loss`) matches Eq.5 (decode a downsampled latent, compare to a
  downsampled GT).

### Open questions for the authors
- Is the uploaded repo the exact code that produced the paper's tables, or a cleaned partial
  release? (Single commit "first", dated after NeurIPS 2025, untagged `main`.)
- Where is `Rθ` and the `LRes` training stage (Eq. 7)? Were the Table 1–5 numbers produced
  with an Rθ-in-the-loop pipeline not present here?
- During the FHF step, is the decoder intended to use the frozen base (α=1) per Eq. 8, or the
  α=0.5 blend the code uses? (`fhf-step-decoder-lora-active`.)
