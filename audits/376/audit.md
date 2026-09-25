# Code-repository audit — SparseDiff (NeurIPS 2025, paper #376)

## 1. Summary

The repository `tsinghua-fib-lab/SparseDiff` is the official implementation of
*"Sparse Diffusion Autoencoder for Test-time Adapting Prediction of Complex
Systems."* SparseDiff has three trainable modules described in the paper
(Appendix B): a codebook-based **sparse encoder** (a VQ-VAE), a **probe-graph
diffusive predictor** (a GRAND-style graph neural ODE), and an **unconditional
diffusion decoder** (a UNet DDPM). The paper reports long-term prediction
results on five systems (Lambda-Omega, Navier-Stokes, Swift-Hohenberg,
Cylinder-Flow, and a real-world SEVIR climate dataset) in Table 1, a
graph-construction comparison (Table 2), ablations (Fig. 4), robustness
(Fig. 5), generalization (Fig. 6), an efficiency trade-off (Fig. 7), and an
appendix sparse/irregular study (Tables 4–5).

The repo ships code for **one system (SH)** only: `train_sh.py` (which trains
**only** the diffusion UNet), `datasets.py`, `utils.py`, a single
`config/sh.yaml`, the three model definitions, and one inference notebook
`sample_sh.ipynb`. Data and weights are not in the repo and are offered via a
Proton Drive link. The notebook is the only evaluation entrypoint; it runs a
single SH trajectory for 80 steps and prints NMSE/SSIM/RMSE.

What I did: read the paper (PDF + text extraction), all `*.py` files, the
config, and the full `sample_sh.ipynb` cell; then ran deterministic checks under
`_audit_code/` (`check_artifacts.py` → `_audit_code/out/artifacts.json`)
enumerating which training scripts, datasets, checkpoints, systems, and
adaptation logic exist versus what the paper/README promise. I stayed read-only
on `code/`.

## 2. Result-traceability table

Every quantitative artefact in the paper is evaluated by a script/notebook only
if a computation that *produces* the number exists in the repo. The repo
contains a single hard-coded SH inference notebook and no driver that sweeps
systems, seeds, baselines, or ablations.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1: RMSE/SSIM for 5 systems × 5 models, "10 runs" mean±std | `sample_sh.ipynb` computes RMSE/SSIM/NMSE for SH only, `test_tra=1`, single run | — | — | MISSING (no multi-system/multi-run/baseline driver) |
| Abstract / Intro: "49.99% average error reduction over baselines" | (none) | — | — | MISSING (depends on Table 1, no script aggregates it) |
| Table 2: graph-construction comparison (Ours/GKN/GRAND+kNN/GRAND+spectral, LO) | (none) | — | — | MISSING (no GKN/kNN/spectral variants in repo) |
| Fig. 4 ablations: uniform vs random vs codebook probes; edge-weight on/off | (none) | — | — | MISSING (no ablation driver) |
| Fig. 5 robustness: codebook size sweep; noise sweep | (none; `config/sh.yaml` has `noise: 0.0`, `num_embeddings: 30`) | — | — | MISSING |
| Fig. 6 generalization: SSIM vs Reynolds (CY OOD) | (none) | — | — | MISSING (no CY data/driver) |
| Fig. 7 accuracy/efficiency trade-off vs update interval; vs G-LED | (none) | — | — | MISSING (no G-LED baseline, no timing harness) |
| Table 3: trainable params (Encoder 0.023M, Predictor 1.32M, Diffusion 26.3M; baselines) | partial — `train_sh.py:46` prints diffusion params, but active `config/sh.yaml` is the "55M" net, not the 26M one | ~55M net active | ✗ (config mismatch) | MISMATCH / see `diffusion-config-55m-not-26m` |
| Tables 4–5: sparse-sampling / irregular wave-equation results | (none) | — | — | MISSING |
| §3.3 test-time adaptation via latent-consistency score χt < τ | not implemented; notebook re-encodes on a fixed `change_step` interval | — | — | MISSING (see `dynamic-adaptation-not-implemented`) |
| Baselines FNO/ConvLSTM/UNet/G-LED (Table 1, Figs. 3/7) | (none) | — | — | MISSING (no baseline code) |

The single number the notebook *can* compute (SH RMSE/SSIM/NMSE for one
trajectory) cannot be matched against Table 1 without the weights and data
(offered only via an external Proton Drive link, not retrieved here) and is a
single trajectory, not the "10 runs / different initial conditions" the table
reports.

## 3. Findings

## missing

```yaml finding
id: vqvae-grand-training-missing
category: missing
topic: "training code / completeness"
title: "No training code for sparse encoder (VQ-VAE) or probe-graph predictor"
severity: high
confidence: high
status: finding
file: train_sh.py
line_start: 12
line_end: 14
quote: |
  from datasets import get_dataset
  from model import DDPM, UNet_new
  from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP, print0
claim: "The only training script, train_sh.py, imports and trains exclusively the diffusion UNet (DDPM(nn_model=UNet_new(...)), backward/optim at lines 38-39, 130-133); no script anywhere in the repo constructs an optimizer or calls .backward() on the VQVAE or the GraphModel predictor (verified in _audit_code/out/artifacts.json: vqvae_backward_or_optim=false, predictor_backward_or_optim=false, backward only in train_sh.py)."
concern: "Two of the paper's three core contributions — the codebook-based sparse encoder and the GRAND probe-graph predictor — cannot be reproduced because the code that fits the codebook (Eq. 4) and trains the predictor is absent."
resolution: "Authors: please add the training scripts for the VQ-VAE sparse encoder and the GRAND predictor, including the codebook objective (Eq. 4) and the predictor loss."
cross_refs: ["only-sh-system-shipped", "headline-results-no-driver"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Appendix B; Eq. 4; §3.1.1, §3.2"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: headline-results-no-driver
category: missing
topic: "result traceability"
title: "No driver computes Table 1, the 49.99% headline, Table 2, or any ablation/robustness/generalization figure"
severity: high
confidence: high
status: finding
file: sample_sh.ipynb
line_start: 201
line_end: 205
quote: |
  test_tra = 1
  start_step = 2*T   # >= T
  t_input = torch.tensor([1.0])
  change_step = opt.sample["change_step"]
claim: "The sole evaluation entrypoint is sample_sh.ipynb, which evaluates the SH system on a single trajectory (test_tra = 1) and prints RMSE/SSIM/NMSE; there is no code that loops over the five systems, the four baselines (FNO/ConvLSTM/UNet/G-LED), the '10 runs', the graph-construction variants (Table 2), or any ablation/robustness/generalization experiment (verified in _audit_code/out/artifacts.json)."
concern: "The paper's headline claim (average 49.99% error reduction over baselines) and every numbered table/figure cannot be reproduced from the repo because no script computes those numbers."
resolution: "Authors: please provide the evaluation driver(s) that aggregate the 5-system / multi-run Table 1 numbers, the baseline runs, and the ablation/robustness/generalization figures."
cross_refs: ["vqvae-grand-training-missing", "only-sh-system-shipped"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Abstract; Table 1; Table 2; Figs. 4-7"
tags: [reforms:2, reforms:4, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: only-sh-system-shipped
category: missing
topic: "datasets / systems coverage"
title: "Only the SH system has config/data hooks; 4 of 5 reported systems and all baselines absent"
severity: medium
confidence: high
status: finding
file: config/sh.yaml
line_start: 1
line_end: 3
quote: |

  dataset: 'sh'
  noise: 0.0
claim: "The repo ships exactly one config (config/sh.yaml) and one system-specific notebook (sample_sh.ipynb); datasets.py references absolute paths for ns/cy/lo/sevir but no configs, notebooks, or data exist for them, and no baseline (FNO/ConvLSTM/UNet/G-LED) code is present (verified: config_files=['config/sh.yaml'], _audit_code/out/artifacts.json)."
concern: "The repo demonstrates only one of the five systems reported in Table 1 and contains none of the baseline implementations, so cross-system and comparative claims cannot be reproduced."
resolution: "Authors: please add configs, data-loading, and run instructions for the LO/NS/CY/SEVIR systems and the four baselines, or scope the claims to what is released."
cross_refs: ["headline-results-no-driver", "hardcoded-absolute-data-paths"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Table 1; Appendix A, C"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dynamic-adaptation-not-implemented
category: missing
topic: "test-time adaptation (core method)"
title: "Latent-consistency χt < τ dynamic re-encoding (§3.3) not implemented; notebook uses fixed interval"
severity: high
confidence: medium
status: finding
file: sample_sh.ipynb
line_start: 213
line_end: 213
quote: |
          for code_step in range(start_step, start_step + pred_steps, change_step):  
claim: "The notebook re-encodes the probe topology on a fixed stride (change_step, from config) inside a plain range() loop; nowhere does the code compute the latent-consistency score χt = (1/k) Σ cos(E(v_{t-T:t}), c_i) or compare it to a threshold τ to trigger re-encoding (verified: mentions_threshold_tau_adaptation=false; the only 'cosine' token in the repo is a VQ config flag, _audit_code/out/artifacts.json)."
concern: "The paper's headline contribution is an *adaptive* test-time re-encoding driven by a latent-consistency threshold; the released code instead re-encodes at a fixed cadence, so the dynamic-update mechanism that the paper credits for accuracy/efficiency (Fig. 7) is absent from the code."
resolution: "Authors: please point to or add the χt/τ dynamic-update implementation described in §3.3, or clarify that the released results used a fixed re-encoding interval."
cross_refs: ["headline-results-no-driver"]
check_script: _audit_code/check_artifacts.py
paper_ref: "§3.3 (latent consistency score χt, threshold τ); Fig. 7"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-absolute-data-paths
category: bug
topic: "reproducibility / paths"
title: "datasets.py hardcodes absolute /data5/chengjingwen/... paths"
severity: medium
confidence: high
status: finding
file: datasets.py
line_start: 199
line_end: 206
quote: |
      elif name == 'sh':
          path = '/data5/chengjingwen/sh/uv.npy'
      elif name == 'ns':
          path = '/data5/chengjingwen/ns/uv.npy'
      elif name == 'cy':
          path = '/data5/chengjingwen/cy/uv.npy'
      elif name == 'sevir_tem':
          path = '/data5/chengjingwen/sevir_tem/uv.npy'
claim: "get_dataset() resolves every system to an absolute path under /data5/chengjingwen/, an author-machine-specific location that does not exist on any other machine (verified: hardcoded_abs_paths_datasets lists 9 such paths, _audit_code/out/artifacts.json)."
concern: "train_sh.py calls get_dataset(name=opt.dataset) and will fail with FileNotFoundError on any other machine; the README instructs downloading data to ./data/sh, which the training path ignores (the notebook uses ./data/{system}, but train_sh.py does not)."
resolution: "Authors: replace the hardcoded /data5/... paths with the documented ./data/{system}/ relative paths used by the README and notebook."
cross_refs: ["only-sh-system-shipped"]
check_script: _audit_code/check_artifacts.py
paper_ref: "README 'Download Dataset'"
tags: [heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: train-multi-gpu-default-true
category: bug
topic: "CLI / entrypoint"
title: "train_sh.py defaults --multi_gpu and --use_amp to True, so single-GPU command in README crashes"
severity: low
confidence: medium
status: finding
file: train_sh.py
line_start: 200
line_end: 201
quote: |
      parser.add_argument("--use_amp", action='store_true', default=True, help="Enable automatic mixed precision.")
      parser.add_argument("--multi_gpu", action="store_true", default=True, help="Enable multi-GPU training.")
claim: "Both --use_amp and --multi_gpu use action='store_true' with default=True, so they are always True regardless of CLI flags; the documented single-GPU invocation `python train_sh.py` therefore enters the multi_gpu branch and reads os.environ['LOCAL_RANK'] (line 209), which is unset outside torchrun."
concern: "The README's single-GPU command `python train_sh.py` raises KeyError: 'LOCAL_RANK' because multi_gpu cannot be disabled from the CLI; CUDA_VISIBLE_DEVICES is also hardcoded to '3,6' (line 206)."
resolution: "Authors: set default=False for --multi_gpu/--use_amp (or use BooleanOptionalAction) so single-GPU training works as documented."
cross_refs: []
paper_ref: "README 'Train the model'"
tags: [heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: pde-guidance-disabled-in-notebook
category: difference
topic: "evaluation procedure (decoder guidance)"
title: "Released SH eval uses no-guidance sampler; the PDE/observation-guided sampler (Eq. 3) is commented out"
severity: medium
confidence: medium
status: finding
file: sample_sh.ipynb
line_start: 283
line_end: 284
quote: |
                      # ddpm_samples = ema_model.ddim_guided_sample_full_sh(n_sample = batch_size*sample_step, size = size, steps = opt.sample["ddim_step"], eta = opt.sample["eta"], zeta_obs = opt.sample["zeta_obs"], zeta_pde = opt.sample["zeta_pde"], ratio = opt.sample["ratio"], reconstructed = reconstructed, data_opt = opt.data, notqdm=False)  # shape: (B*sample_step, C, H, W)
                      ddpm_samples = ema_model.ddim_sample_from_reconstructed_sh(n_sample = batch_size*sample_step, size = size, steps = 50, eta = opt.sample["eta"], zeta_obs = opt.sample["zeta_obs"], zeta_pde = opt.sample["zeta_pde"], ratio = opt.sample["ratio"], reconstructed = reconstructed, data_opt = opt.data, notqdm=False)  # shape: (B*sample_step, C, H, W)
claim: "The active call is ddim_sample_from_reconstructed_sh (a method that, despite its zeta_obs/zeta_pde signature, denoises the filled reconstruction over a short n_T/5 schedule), while the guided sampler ddim_guided_sample_full_sh that applies the Bayes/observation+PDE gradient guidance of Eq. 3 is commented out (verified: notebook_pde_guided_call_active=false, notebook_active_sampler shows the from_reconstructed variant active, _audit_code/out/artifacts.json)."
concern: "The paper's decoder is described as a guided diffusion reconstruction (Eq. 3, §3.1.2); the released notebook uses an unguided short-schedule denoiser instead, so the evaluated procedure differs from the described one (both are individually plausible diffusion samplers, hence a difference rather than a bug)."
resolution: "Authors: confirm which sampler produced the reported numbers and, if it is the guided one, restore that call in the released notebook."
cross_refs: []
check_script: _audit_code/check_artifacts.py
paper_ref: "Eq. 3; §3.1.2 Guided Diffusion Decoder"
tags: [forensics:post-hoc-selection]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: diffusion-config-55m-not-26m
category: difference
topic: "model architecture / hyperparameters"
title: "Active diffusion config is the 55M UNet, not the 26M net Table 3 reports"
severity: low
confidence: medium
status: finding
file: config/sh.yaml
line_start: 20
line_end: 36
quote: |
  network:   # 55M - for test
    image_shape: [1, 128, 128]
    n_channels: 128
    ch_mults: [1, 2, 2, 2] # [1, 2, 4]
    is_attn: [False, True, True, True]
    dropout: 0.1
    n_blocks: 3
    use_res_for_updown: True

  # network:   # 26M - final use
  #   image_shape: [1, 128, 128]
  #   n_channels: 128
  #   ch_mults: [1, 2, 2] # [1, 2, 4]
  #   is_attn: [False, False, True]
  #   dropout: 0.1
  #   n_blocks: 2
  #   use_res_for_updown: False
claim: "The uncommented network block is labelled '55M - for test' (ch_mults [1,2,2,2], n_blocks 3, is_attn [F,T,T,T]); the block labelled '26M - final use' is commented out. Appendix B / Table 3 report the diffusion module at 26.3M with ch_mults [1,2,2], is_attn [False,False,True], n_blocks 2 — matching the *commented* block, not the active one."
concern: "The diffusion architecture loaded by default does not match the parameter count and hyperparameters the paper reports in Table 3/Appendix B; results reproduced with the shipped config would use a larger model than the paper describes."
resolution: "Authors: confirm which network config produced the reported numbers and ship that one uncommented (or update Table 3)."
cross_refs: []
paper_ref: "Appendix B; Table 3 (Diffusion 26.3M)"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: single-trajectory-eval-no-std
category: methodology
topic: "evaluation / statistical support"
title: "Released evaluation is a single trajectory, but Table 1 reports mean±std over 10 runs"
severity: medium
confidence: medium
status: finding
file: sample_sh.ipynb
line_start: 201
line_end: 202
quote: |
  test_tra = 1
  pred_steps = 80
claim: "The released SH evaluation runs exactly one test trajectory (test_tra = 1) for 80 prediction steps and reports point RMSE/SSIM/NMSE; Table 1 reports averages 'from 10 runs' with standard deviations, and the SH RMSE is reported over a longer horizon (paper states >100-step long-term prediction)."
concern: "A single-trajectory, 80-step run cannot produce the mean±std over 10 runs / >100 steps reported in Table 1, so the released entrypoint cannot substantiate the reported variability or long-horizon numbers; the basis of the reported standard deviations is not reproducible from the code."
resolution: "Authors: provide the multi-trajectory / multi-run evaluation loop and the prediction horizon used for Table 1, including how the 10 runs and standard deviations were computed."
cross_refs: ["headline-results-no-driver"]
paper_ref: "Table 1 caption ('standard deviation from 10 runs'); §4.1 ('more than 100 steps')"
tags: [reforms:7, stats:auc-ci]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 4 | high | No VQ-VAE/predictor training; no driver for Table 1/2, baselines, ablations; χt/τ adaptation absent |
| bug | 2 | medium | Hardcoded absolute data paths; multi-GPU flag cannot be disabled |
| difference | 2 | medium | Unguided sampler active (Eq. 3 guidance commented); active 55M config ≠ 26M Table 3 |
| methodology | 1 | medium | Released eval is one trajectory; Table 1 claims 10-run mean±std |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `vqvae-grand-training-missing`** — Two of three core modules (sparse encoder, probe-graph predictor) have no training code; only the diffusion UNet trains. (high/high)
2. **[missing] `headline-results-no-driver`** — No script computes Table 1, the 49.99% headline, Table 2, or any ablation/robustness/generalization figure; the one notebook does single-trajectory SH only. (high/high)
3. **[missing] `dynamic-adaptation-not-implemented`** — The latent-consistency χt < τ dynamic re-encoding that is the paper's headline test-time-adaptation mechanism is not in the code (fixed-interval re-encoding instead). (high/medium)
4. **[missing] `only-sh-system-shipped`** — Only 1 of 5 reported systems has config/notebook hooks; no baselines shipped. (medium/high)
5. **[difference] `pde-guidance-disabled-in-notebook`** — The released eval uses an unguided short-schedule sampler; the Eq. 3 guided sampler is commented out. (medium/medium)
6. **[bug] `hardcoded-absolute-data-paths`** — `datasets.py` resolves all data to author-only `/data5/chengjingwen/...` paths, breaking `train_sh.py` elsewhere. (medium/high)

### Items that genuinely look fine
- Min-max normalization statistics in the notebook are computed from the **training** array (`x_train`) and applied to test data (`sample_sh.ipynb`, the `xmin/xmax` block), so the test set does not leak into the normalization scale. Good practice.
- The VQ-VAE, GRAND predictor, DDPM, and UNet model definitions are present and internally coherent (`model/vq_vae.py`, `model/grand_predictor.py`, `model/DDPM.py`, `model/unet_new.py`); the DDPM training loss (Eq. 2) and DDIM samplers match standard formulations.
- Seeding is comprehensive for the diffusion trainer (`utils.py:init_seeds` sets random/numpy/torch/cuda and cuDNN deterministic).
- Dependencies are listed in the README (though unpinned).

### Open questions for the authors
- `dynamic-adaptation-not-implemented` (high severity, medium confidence): is the χt/τ adaptive re-encoding implemented anywhere, or were the reported numbers produced with fixed-interval re-encoding?
- `pde-guidance-disabled-in-notebook` (medium/medium): which decoder sampler (guided vs. from-reconstructed) produced the reported SH numbers?
- `diffusion-config-55m-not-26m` (low/medium): which network config matches Table 3, and which produced the results?
