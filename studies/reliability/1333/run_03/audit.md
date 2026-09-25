# Code-repository audit — Paper 1333, "Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement"

## 1. Summary

The cloned repo `code/lyd-2022__Latent-Harmony/` is a BasicSR-style training scaffold
for the paper's two-stage VAE framework. It contains: a Stage-1 LH-VAE trainer
(`basicsr/models/EQvae_model.py` + arch `RAVAE_EQ` in `basicsr/archs/LHVAE_arch.py`)
implementing PDPS / L_Inv (DINOv2) / L_Eqv; a Stage-2 HF-LoRA trainer
(`basicsr/models/VAEadapter_model.py` + arch `RAVAEHFLora` in
`basicsr/archs/LHVAE_hflora_arch.py`) with an FFT high-pass loss
(`basicsr/utils/hf_ops.py`) and an HF GAN; two *example* YAML configs under `configs/`;
and the generic BasicSR data/loss/metric utilities.

What I did: read both configs, both Stage models, the two LH-VAE archs, the LoRA
module, the HF-extraction op, the loss bank, the paired-image dataset, the training
entrypoint (`basicsr/train.py`) and option parser (`basicsr/utils/options.py`). I
greped the whole tree for a restoration network (SFHformer/R_theta), for evaluation/
inference/test entrypoints, for pretrained weights, for a dependency spec, and for
hardcoded absolute paths. I wrote one read-only static check,
`_audit_code/check_arch_registry.py`, that confirms the arch type `RAVAE` requested by
`configs/stage2_hflora.yml` is not registered in any `*_arch.py`
(`_audit_code/out/arch_registry.csv`).

Headline context: the paper's NeurIPS checklist Q5 answers **"[No] We will release the
source code upon acceptance"** (paper.pdf, checklist item 5), and the README is empty
(0 lines of content). The repo is a partial scaffold, not the artefact that produced
Tables 1–5. None of the paper's quantitative results can be reproduced from it.

## 2. Result-traceability table

Every numbered table in the paper reports PSNR/SSIM/LPIPS/FID for the proposed method and
baselines. Reproducing any of them requires: (a) the latent restoration network R_theta
(paper says SFHformer, Table 5c), (b) trained VAE + LoRA weights, (c) the UHD benchmark
datasets, and (d) an evaluation/inference script. None of (a)–(d) is present.

| Paper artefact                                   | Repo location                | Computed value | Matches | Status                                   |
|--------------------------------------------------|------------------------------|----------------|---------|------------------------------------------|
| Table 1 "Ours" PSNR/SSIM/LPIPS (4 degradations)  | (none — no eval script, no R_theta, no weights) | — | — | MISSING (see no-eval-script, no-restoration-network) |
| Table 2 "Ours" PSNR/SSIM/LPIPS (6 degradations)  | (none)                       | —              | —       | MISSING                                  |
| Table 3 standard-res LPIPS/FID (PromptIR/Diff-Plugin/CosAE /w Ours) | (none — no integration code) | — | — | MISSING                          |
| Table 4 Generalization (unseen + composite)      | (none)                       | —              | —       | MISSING                                  |
| Table 5a ablation (w/o L_Inv, L_Eqv, PDPS, FHF/PHF-LoRA, …) | (none — no ablation harness) | —      | —       | MISSING (no ablation driver)             |
| Table 5b inference time (DreamUIR/Hist/UHDproc/LH) | (none)                     | —              | —       | MISSING                                  |
| Table 5c latent-net ablation (Restormer/NAFNet/SFHformer +Ours) | (none — networks not wired into Stage 2) | — | — | MISSING |
| Table 5d α sweep (PSNR/SSIM/LPIPS/User vs α)     | `set_alpha`/`forward(alpha=)` exists, but no eval script | — | — | MISSING (control implemented, not evaluated) |
| Fig. 2 motivation (CDCS/t-SNE/DCT/loss curves)   | (none)                       | —              | —       | MISSING (no analysis code)               |
| FLOPs 3.6G / Params 1.2M ("Ours", Tables 1–2)    | `__main__` profiler in LHVAE_arch.py (needs cluster path) | — | — | MISSING (depends on hardcoded path; VAE-only, no R_theta) |

Every row is MISSING; the dominant causes are the three `missing` findings below. The
Stage-2 config additionally cannot instantiate at all (a `bug`).

## 3. Findings

## missing

```yaml finding
id: no-restoration-network-stage2
category: missing
topic: "result traceability / Stage-2 pipeline"
title: "Stage-2 latent restoration network R_theta (SFHformer) is absent; Stage-2 only reconstructs through the VAE"
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
claim: "Stage-2 forward calls net_g (a RAVAEHFLora that wraps only the VAE encoder/decoder, see LHVAE_hflora_arch.py:122-133 where forward does out = self.vae(x)); the recon it returns is an autoencoder reconstruction of the degraded input lq, never the output of a latent restoration network."
concern: "The paper's Stage 2 (Eq. 7-9, Sec 4.2) restores via R_theta on the latent (z_res = R_theta(z_deg); Table 5c says R_theta is SFHformer), but no restoration network exists anywhere in the repo (grep for SFHformer/R_theta/net_r returns nothing), so the released code cannot produce the restored images behind any reported metric."
resolution: "Authors: please add the latent restoration network R_theta (SFHformer/Restormer/NAFNet) and wire it into the Stage-2 forward path (Eq. 9 uses D(R_theta(E(I_deg)))); confirm whether the published numbers used R_theta and not bare VAE reconstruction."
cross_refs: ["no-eval-inference-script", "§4.2", "Table 5c"]
paper_ref: "Section 4.2, Eq. (7)-(9); Table 5(c)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-eval-inference-script
category: missing
topic: "result traceability / evaluation"
title: "No evaluation, inference, or test entrypoint exists; no table/figure number is reproducible"
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
claim: "The only executable entrypoint in the repo is basicsr/train.py (training). A repo-wide search for files matching *test*/*eval*/*infer*/*demo* returns nothing, and no script computes PSNR/SSIM/LPIPS/FID over a benchmark, runs the α sweep, or produces the motivation figures."
concern: "Per the result-traceability rule, every value in Tables 1-5 and Fig. 2 must trace to code that computes it; here no script computes any reported number, so all reported results are untraceable from the artefact."
resolution: "Authors: provide the evaluation/inference scripts (with the exact commands and the benchmark loaders) used to produce Tables 1-5, including the α-sweep and ablation drivers."
cross_refs: ["no-restoration-network-stage2", "no-deps-weights-readme"]
paper_ref: "Tables 1-5; Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-deps-weights-readme
category: missing
topic: "code completeness"
title: "Empty README, no dependency spec, no pretrained weights, no datasets — repo is not runnable end-to-end"
severity: high
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Latent-Harmony
claim: "README.md contains only the title line (no install/run/results instructions). There is no requirements.txt / setup.py / environment.yml; no pretrained VAE or LoRA weights (only basicsr/metrics/niqe_pris_params.npz, a metric helper, ships); no datasets; and configs point pretrain_vae_path/local_weight_path/dataroot_* at non-existent placeholder paths (./weights/stage1_eqvae.pth, ./weights/dinov2_vits14.pth, ./datasets/train/gt)."
concern: "A minimally complete submission needs a dependency spec, training + evaluation code, weights (or a fetch), and a README with reproduce commands; none is present, and the NeurIPS checklist Q5 states code is unreleased ('We will release the source code upon acceptance'), so the environment cannot be rebuilt and results cannot be reproduced."
resolution: "Authors: add a pinned dependency list, a README with exact reproduce commands and a results table, and either ship the trained Stage-1/Stage-2 weights and DINOv2 checkpoint or provide resolvable download links plus dataset access instructions."
cross_refs: ["no-eval-inference-script", "configs-are-examples"]
paper_ref: "NeurIPS checklist Q5 (Open access to data and code) = [No]"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: stage2-ravae-type-not-registered
category: bug
topic: "Stage-2 instantiation"
title: "Stage-2 config requests arch type 'RAVAE', which is never registered → KeyError at startup"
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
claim: "RAVAEHFLora.__init__ reads vae_config['type'] and calls ARCH_REGISTRY.get(arch_type). configs/stage2_hflora.yml:42 sets vae_config.type: RAVAE, but the only arches registered via @ARCH_REGISTRY.register() in *_arch.py are RAVAE_EQ, RAVAEHFLora, LoRAConv2d, UNetDiscriminatorSN (see _audit_code/out/arch_registry.csv). No class named 'RAVAE' is registered."
concern: "Registry.get('RAVAE') raises KeyError (basicsr/utils/registry.py:62-66 raises when name is missing), so building the Stage-2 network with the shipped config crashes before any training/eval can run."
resolution: "Authors: either register the intended backbone under the name 'RAVAE' or change configs/stage2_hflora.yml vae_config.type to 'RAVAE_EQ' (the actually-registered arch), and confirm which VAE class produced the paper's Stage-2 results."
cross_refs: []
check_script: _audit_code/check_arch_registry.py
paper_ref: "configs/stage2_hflora.yml line 42"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-experiments-root
category: bug
topic: "reproducibility / hardcoded paths"
title: "experiments_root hardcoded to the authors' cluster path /fs-computility/... breaks any off-cluster run"
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
claim: "On every training run, all output directories (models, logs, training_states, visualization) are placed under the absolute path /fs-computility/ai4sData/liuyidi/model/experiments/<name>, ignoring root_path / cwd; train.py then calls make_exp_dirs(opt) and copy_opt_file to that root."
concern: "This absolute path exists only on the authors' machine; on any other system make_exp_dirs / copyfile will fail (no such directory / permission denied), so the provided training entrypoint cannot run as shipped without editing the source."
resolution: "Authors: derive experiments_root from root_path (e.g. osp.join(root_path, 'experiments', opt['name'])) or expose it as a config/CLI option, as upstream BasicSR does."
cross_refs: []
paper_ref: "basicsr/utils/options.py"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: configs-are-examples
category: difference
topic: "experimental setup / configs"
title: "Shipped configs are labelled 'example' with placeholder data paths and generic hyper-parameters, not the configs that produced the paper numbers"
severity: low
confidence: high
status: finding
file: configs/stage2_hflora.yml
line_start: 1
line_end: 5
quote: |
  name: stage2_hflora_example
  model_type: VAEadapter
  num_gpu: 1
  manual_seed: 123

claim: "Both configs are named *_example (stage1_eqvae_example, stage2_hflora_example), use generic single-GPU settings, batch_size_per_gpu: 2, total_iter: 100000, and placeholder data roots ./datasets/train/gt etc.; they do not encode the six-degradation / four-degradation training recipes, per-degradation datasets, or the α and LoRA-rank settings that back Tables 1-5."
concern: "The paper reports specific UHD all-in-one results (Tables 1-2) and an α sweep (Table 5d), but the released configs are illustrative templates, so the exact experimental settings used for the paper cannot be recovered from the repo."
resolution: "Authors: release the actual training/eval configs (data manifests, degradation mix, iters, LoRA rank/alpha, λ_Inv/λ_Eqv/λ_hf_fid/λ_gan, α values) used for each reported table."
cross_refs: ["no-deps-weights-readme"]
paper_ref: "Tables 1-5; Sec. 5 Experiments"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No standalone methodology finding. The implemented procedures I could inspect
(Stage-1 PDPS/L_Inv/L_Eqv in EQvae_model.py; the LoRA alternating optimisation and
α-blend in VAEadapter_model.py / LHVAE_hflora_arch.py) are internally consistent with
the paper's described logic, and the data pipeline uses pre-separated train/val folders
(no in-code split, no obvious leakage). The blocking problems are absence and a crash
(routed to `missing`/`bug` above), not an invalid-but-runnable procedure. The most
consequential methodology-adjacent concern — that Stage-2 as shipped optimises a bare
VAE reconstruction with no restoration network — is filed as `missing`
(no-restoration-network-stage2) because the component is simply not there.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 3          | high         | No R_theta restoration net, no eval/inference scripts, no deps/weights/README/data |
| bug         | 2          | high         | Stage-2 'RAVAE' arch type unregistered (KeyError); hardcoded /fs-computility experiments_root |
| difference  | 1          | low          | Shipped configs are 'example' templates, not the paper's recipes       |
| methodology | 0          | -            | Inspected procedures internally consistent; no standalone finding      |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing] no-restoration-network-stage2** — the latent restoration network R_theta (SFHformer) central to Stage 2 and every reported number is absent; Stage-2 code only autoencodes the input. (high/high)
2. **[missing] no-eval-inference-script** — no script computes any value in Tables 1-5 or Fig. 2; results are untraceable. (high/high)
3. **[bug] stage2-ravae-type-not-registered** — Stage-2 config's `vae_config.type: RAVAE` is unregistered → KeyError; Stage 2 cannot instantiate. (high/high)
4. **[missing] no-deps-weights-readme** — empty README, no dependency spec, no weights/datasets; checklist Q5 says code unreleased; not runnable end-to-end. (high/high)
5. **[bug] hardcoded-experiments-root** — training writes to `/fs-computility/ai4sData/liuyidi/model/...`; breaks off the authors' cluster. (medium/high)
6. **[difference] configs-are-examples** — shipped configs are labelled templates, not the per-table recipes. (low/high)

### Items that genuinely look fine
- Stage-1 losses (PDPS Eq. 3, L_Inv via DINOv2 Eq. 4, L_Eqv Eq. 5) match the paper's described forms in `EQvae_model.py` (`_build_pdps_input`, `_compute_inv_loss`, `_compute_eqv_loss`).
- The inference α-blend matches the paper (`set_alpha`: enc_scale=α, dec_scale=1−α; LHVAE_hflora_arch.py:83-91 vs paper §4.3 ϕ=ϕ*+αΔϕ, ψ=ψ*+(1−α)Δψ).
- LoRA is correctly zero-initialised on the up projection (`nn.init.zeros_(self.lora_up.weight)`, lora_arch.py:50) so adapters start as identity.
- The paired-image dataset uses separate train/val folders; no in-code split or train/test leakage.
- All loss/arch types other than `RAVAE` referenced by the configs are registered and present.

### Open questions for the authors
- Did the published Tables 1-5 use a latent restoration network R_theta, and if so which checkpoint/config? (The repo contains none.)
- Which VAE class is the intended `RAVAE` backbone for Stage 2 — is it `RAVAE_EQ`, or a separate unreleased class?
- Will the evaluation/inference harness, trained weights, and per-table configs be released, given checklist Q5 = [No]?
