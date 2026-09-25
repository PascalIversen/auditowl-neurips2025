# Code-repository audit — Unlocking Dataset Distillation with Diffusion Models (LD3M), NeurIPS 2025 (paper 2194)

## 1. Summary

The audited repo `Brian-Moser/prune_and_distill` (commit `d6912268`, moving `main`) is the
abstract-footnote URL cited by the paper. Its `readme.md` and citation block identify it as the
code for a **different** publication — *"Distill the Best, Ignore the Rest: A Study in Latent
Dataset Distillation on Core-Sets"* (IJCNN 2025), a loss-value-based **pruning** study — that
"is based on LD3M and GLaD". So the repo contains the LD3M implementation, but embedded inside a
pruning project: all README run-commands and the dataset builder are wired for loss-pruned
core-sets (`--percent`, `--order`, per-class JSON files), a procedure the NeurIPS LD3M paper never
mentions.

I read the paper (`paper.pdf`) and supplementary, traced every headline number/table/figure to
code, and read the core method files: the modified DDIM reverse step that implements Eq. 7
(`src/latentdiff/ldm/models/diffusion/ddim.py`), the LD3M wiring in `src/glad_utils.py`
(`ldm_latent_to_im_encode`, `ldm_backward_encode`, `prepare_LDM_latents_encode`, `build_dataset`,
`eval_loop_ldm_encode`, `load_ldm`), the DC/DM/MTT LD3M entrypoints (`src/distill_*_LD3M.py`),
`src/utils.py::evaluate_synset/get_dataset`, `requirements.txt`, and `install.sh`. Deterministic
checks live in `_audit_code/check_traceability.py` (output `_audit_code/out/traceability.json`):
they confirm no code computes the gradient-norm table, the SNR figure, the LPIPS-diversity number,
or peak-memory/timing table values, that the LD3M data builder hard-depends on per-class pruning
JSON files, and that zero such JSON files ship with the repo.

The implementation read READ-ONLY; no repo file was modified.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Eq. 7 modified reverse step (headline contribution) | `src/latentdiff/ldm/models/diffusion/ddim.py:200` | n/a (impl. present) | partial (applied to `pred_x0`, uses VQ-quantized `x_T`; paper writes weighting on mean μθ with `z_T`) | Present, faithfulness nuance (see `eq7-residual-on-predx0`) |
| Table 3 ImageNet 1IPC 128² (DC/DM/MTT, GLaD vs LD3M) | `src/distill_{dc,dm,mtt}_LD3M.py` + `src/distill_{dc,dm,mtt}.py`, eval `glad_utils.py:499` | n/a (needs ImageNet-1k + LDM ckpt + W&B) | not runnable as documented w/o pruning JSONs | see `pruning-json-missing`, `selective-best-acc` |
| Table 4 ImageNet 10IPC 128² | same as above | — | — | same |
| Table 5 256² DC IPC=1 (ImageNet/FFHQ/Random) | `distill_dc_LD3M.py --res=256`, `load_ldm` (ffhq/cin256) | — | — | Present (entrypoint), unverifiable |
| Table 2a CIFAR-10 IPC=1 (LD3M) | (none — `load_ldm` loads only ImageNet/FFHQ LDM; README has no CIFAR LD3M command) | — | — | MISSING path (`cifar-ld3m-missing`) |
| Table 2b CIFAR-10 IPC=50 (LD3M 73.2) | (none) | — | — | MISSING |
| Table 6 ablation w/ vs w/o diffusion | (no flag toggling the residual / diffusion-off variant found) | — | — | MISSING (`ablation-diffusion-toggle-missing`) |
| Table 7 ablation (c only / +Z / +Eq.7) | (no config switch to disable Eq. 7 or freeze Z found) | — | — | MISSING (cross-ref `ablation-diffusion-toggle-missing`) |
| Table 8 ablation init (Gauss vs encoded real) | `prepare_LDM_latents_encode` encodes real images; Gaussian-init branch not exposed | — | — | partial |
| Table 1 gradient norms ‖∂L/∂Z‖ vs T=10..90 | (none) | — | — | MISSING (`trace-table1-gradnorm-missing`) |
| Fig. 4 accuracy/time vs T (574 vs 693 min) | `glad_utils.py::ldm_time_measurement` measures one run; no table/figure producer; no GLaD-vs-LD3M sweep | — | — | MISSING producer (`trace-timing-missing`) |
| Fig. 4 / §5.2 peak GPU mem 29.4 vs 31.2 GB | (none — no `max_memory_allocated`) | — | — | MISSING |
| Fig. 6 SNR of gradient norms (w/ vs w/o mod) | (none) | — | — | MISSING (cross-ref `trace-table1-gradnorm-missing`) |
| Supp. LPIPS diversity 0.386 vs 0.420 | (none — no `lpips`) | — | — | MISSING |

Deterministic backing: `_audit_code/out/traceability.json`.

## 3. Findings

## missing

```yaml finding
id: pruning-json-missing
category: missing
topic: "result traceability / data preparation"
title: "ImageNet LD3M data builder hard-depends on per-class pruning JSONs that are absent"
severity: high
confidence: high
status: finding
file: src/glad_utils.py
line_start: 33
line_end: 39
quote: |
        json_file = os.path.join(json_path, f'class_{c}_top_{percent}_{order}.json')
        try:
            with open(json_file, 'r') as f:
                json_files[c] = set(json.load(f))  # Load the image paths
        except FileNotFoundError:
            print(f"Warning: JSON file for class {c} not found.")
            json_files[c] = set()  # Empty set if file not found
claim: "Every ImageNet LD3M entrypoint calls build_dataset(...) which, unless --percent=100, keeps only images whose path appears in `class_{c}_top_{percent}_{order}.json` loaded from json_path (the filter at line 51 is `if percent == 100 or img_path in json_files[class_label]`); on FileNotFoundError the per-class set is empty, so the real training set becomes empty for that class."
concern: "All README commands pass --percent=60/20 and json_path defaults to a private cluster path `/netscratch/bmoser/...`; no such JSON ships in the repo (0 found), so the documented runs cannot reproduce the paper's full-dataset LD3M numbers and silently train on an empty/degenerate real set."
resolution: "Authors: ship the pruning JSON files or document running with --percent=100 (the full-dataset setting the LD3M paper actually uses); confirm which percent produced each paper table."
cross_refs: ["repo-is-pruning-paper", "selective-best-acc"]
check_script: _audit_code/check_traceability.py
paper_ref: "§5.1 Datasets & Evaluation; Tables 3-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: trace-table1-gradnorm-missing
category: missing
topic: "result traceability"
title: "No code computes Table 1 gradient norms or Fig. 6 SNR (the vanishing-gradient evidence)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "Our analysis confirms that in standard diffusion models, gradient norms for Z decrease nearly tenfold as T increases from 10 to 90 (see Table 1)."
claim: "Table 1 reports ‖∂L/∂Z‖×10^4 for T=10..90 and Fig. 6 reports SNR of gradient norms with vs without the modification; a repo grep finds no gradient-norm or SNR computation (no `.norm()` of dL/dZ, no `SNR`)."
concern: "The empirical motivation for the whole method (vanishing gradients, and that the residual fixes them) has no producing script, so these numbers cannot be reproduced or checked."
resolution: "Authors: provide the script that computes the per-T gradient norms (Table 1) and the SNR analysis (Fig. 6)."
cross_refs: ["trace-timing-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "Table 1; Fig. 6 (Supp.)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: trace-timing-missing
category: missing
topic: "result traceability"
title: "No producer for Fig. 4 time/accuracy sweep or the peak-memory/runtime numbers"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "Using T = 20, LD3M requires slightly less peak GPU memory on an A100-40GB compared to GLaD (29.4GB vs 31.2GB) and completes the distillation process faster (574 min vs 693 min)."
claim: "ldm_time_measurement (glad_utils.py:1072) times a single forward pass, but no script produces the Fig. 4 accuracy-vs-T sweep, the 574/693-minute distillation times, or the 29.4/31.2 GB peak-memory figures (no max_memory_allocated anywhere in src)."
concern: "The efficiency claims (memory, runtime, optimal T trade-off) cannot be reproduced from the repo."
resolution: "Authors: provide the benchmarking script that measured peak memory and end-to-end distillation time for LD3M vs GLaD across T."
cross_refs: ["trace-table1-gradnorm-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "§5.2; Fig. 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: cifar-ld3m-missing
category: missing
topic: "result traceability"
title: "No LD3M entrypoint/LDM for the CIFAR-10 results in Table 2"
severity: medium
confidence: medium
status: finding
file: src/glad_utils.py
line_start: 775
line_end: 784
quote: |
  def load_ldm(res, args=None):
      if args.rand_g:
          config = OmegaConf.load("./src/latentdiff/configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
          G = load_model_from_config(config, None)
      elif args.ffhq:
          config = OmegaConf.load("./src/latentdiff/configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
          G = load_model_from_config(config, "./src/latentdiff/models/ldm/ffhq256/model.ckpt")
      else:
          config = OmegaConf.load("./src/latentdiff/configs/latent-diffusion/cin256-v2.yaml")
          G = load_model_from_config(config, "./src/latentdiff/models/ldm/cin256-v2/model.ckpt")
claim: "load_ldm only loads the ImageNet (cin256) or FFHQ LDM; there is no CIFAR-10 LDM load path, and the README gives no CIFAR LD3M command, yet Table 2 reports LD3M CIFAR-10 results (IPC=1 and IPC=50=73.2)."
concern: "The CIFAR-10 LD3M numbers in Table 2 have no clearly runnable producing path in the repo."
resolution: "Authors: provide the exact command and LDM checkpoint used to distill CIFAR-10 with LD3M (Table 2a/2b)."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "Table 2 (a) and (b)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-diffusion-toggle-missing
category: missing
topic: "ablations"
title: "No flag/config to disable the Eq. 7 residual or the diffusion process for Tables 6-7"
severity: medium
confidence: medium
status: finding
file: src/latentdiff/ldm/models/diffusion/ddim.py
line_start: 196
line_end: 200
quote: |
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        pred_x0 = (1. - index/T) * pred_x0 + (index/T) * x_T
claim: "The residual mix at line 200 is always applied; there is no CLI flag or config that turns it off ('w/o diffusion' / 'learnable c only' / '+Z' variants in Tables 6-7), nor a separate AE-only entrypoint, so the ablation configurations cannot be reproduced from the released code."
concern: "Tables 6 and 7 (the ablations that establish *why* LD3M works — the +1.2 pp diffusion gain and the +Eq.7 jump from 22.3% to 28.1%) have no toggle in the code to reproduce the ablated conditions."
resolution: "Authors: expose the flags used to produce the w/o-diffusion, c-only, and +Z (no Eq.7) ablation rows, or point to the branch/script that does."
cross_refs: ["eq7-residual-on-predx0"]
check_script: _audit_code/check_traceability.py
paper_ref: "Tables 6 and 7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: deps-unpinned
category: missing
topic: "reproducibility / dependencies"
title: "Dependencies unpinned; key LDM stack versions only in install.sh, paper says torch 1.10.1"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 13
quote: |
  torch
  torchvision
  tqdm
  wandb
  numpy
  kornia
  ema-pytorch
  einops
  dill
  scipy
  timm < 0.9
  ftfy
  regex
claim: "requirements.txt lists no versions except `timm < 0.9`; install.sh pins only pytorch-lightning==1.6.5 and omegaconf>=2.0.0, and the paper (Supp. F) states torch 1.10.1 / torchvision 0.11.2, which are not pinned anywhere."
concern: "The environment cannot be deterministically rebuilt; the LDM code (taming-transformers, ldm) is version-sensitive, so unpinned torch/lightning versions can break or silently alter behaviour."
resolution: "Authors: pin torch/torchvision and the LDM dependency versions actually used."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "Supplementary §F Hardware and Software"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone runtime bug was confirmed as scientifically load-bearing beyond the empty-dataset
consequence already filed under `pruning-json-missing`. The hardcoded W&B-key placeholder
(`wandb.login(key="FILL IN YOUR W&B KEY")`, e.g. `src/distill_dc_LD3M.py:27`) and the private
absolute paths (`/netscratch/...`, `/ds/imagenet`) are engineering friction, not method defects;
noted, not filed.

## difference

```yaml finding
id: repo-is-pruning-paper
category: difference
topic: "repository provenance"
title: "Released repo and README are for a different (pruning) paper; LD3M is treated as a dependency"
severity: medium
confidence: high
status: finding
file: readme.md
line_start: 1
line_end: 1
quote: |
  # Distill the Best, Ignore the Rest: Improving Dataset Distillation with Loss-Value-Based Pruning
claim: "The footnote repo's README, title, and BibTeX describe the IJCNN-2025 core-set *pruning* study and state the work 'is based on LD3M and GLaD'; the README's reproduction commands all carry pruning flags (--percent, --order) and a get_losses_imagenet.py loss-ranking script the LD3M paper never describes."
concern: "The NeurIPS LD3M paper points readers to a repo whose documented protocol (loss-pruned core-sets) differs from the paper's protocol (standard full-dataset GLaD setup); there is no LD3M-specific README, results table, or exact reproduce commands, so a reader cannot tell which settings reproduce the LD3M tables."
resolution: "Authors: provide an LD3M-specific README (or branch) with the exact full-dataset commands and a results table mapping commands to Tables 2-8."
cross_refs: ["pruning-json-missing"]
paper_ref: "Abstract footnote (code URL); §5.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eq7-residual-on-predx0
category: difference
topic: "core method implementation"
title: "Eq. 7 residual is applied to predicted x0 with a VQ-quantized z_T, not to the mean μθ as written"
severity: low
confidence: medium
status: finding
file: src/latentdiff/ldm/models/diffusion/ddim.py
line_start: 145
line_end: 200
quote: |
        outs = self.p_sample_ddim(img, self.model.first_stage_model.quantize(x_T)[0], total_steps-1, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
claim: "Eq. 7 in the paper weights the predicted *mean* μθ(c,z_t,γt) against z_T; the code instead linearly mixes the predicted clean latent `pred_x0` with `x_T` (ddim.py:200) and the residual `x_T` is the VQ-*quantized* encoded latent (`first_stage_model.quantize(x_T)[0]`, ddim.py:148), not the noised z_T the equation/Algorithm 1 names. The decay weight index/T runs 1->0 as the paper's t/T does, so the decay direction matches."
concern: "The implemented residual target (pred_x0, quantized) differs from the equation's target (mean μθ, z_T); both are plausible variants, so this is a faithfulness gap rather than an invalid procedure, but it means the released code does not literally implement Eq. 7."
resolution: "Authors: confirm whether the paper equation or the pred_x0/quantized form was used for the reported numbers, and reconcile Eq. 7 / Algorithm 1 with the code."
cross_refs: ["ablation-diffusion-toggle-missing"]
paper_ref: "Eq. 7; Algorithm 1; §4.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: selective-best-acc
category: methodology
topic: "hyperparameter tuning / model selection"
title: "Reported accuracy is the max test accuracy over distillation checkpoints (selection on test set)"
severity: medium
confidence: medium
status: finding
file: src/glad_utils.py
line_start: 549
line_end: 556
quote: |
        acc_test_mean = np.mean(np.max(accs_test, axis=1))
        acc_test_std = np.std(np.max(accs_test, axis=1))
        best_dict_str = "{}".format(model_eval)
        if acc_test_mean > best_acc[best_dict_str]:
            best_acc[best_dict_str] = acc_test_mean
            best_std[best_dict_str] = acc_test_std
            save_this_it = True
claim: "eval_loop_ldm_encode runs at multiple distillation iterations and keeps `best_acc` = the maximum mean test accuracy seen across those evaluation checkpoints; the headline/best metric is therefore the best distillation iteration chosen by the test-set accuracy itself (no held-out validation split exists in get_dataset)."
concern: "Selecting the reported number as the best-over-iterations on the test set inflates accuracy and is test-set model selection; it is applied symmetrically to GLaD here, but absolute numbers and the 'up to 4.8 pp' gains are optimistic without a validation-selected checkpoint."
resolution: "Authors: report accuracy at a fixed iteration (or a validation-selected checkpoint) rather than the max-over-iterations test accuracy, or confirm which `accs`/`best_acc` field populated each table."
cross_refs: ["pruning-json-missing"]
paper_ref: "§5.1 (mean ± std over 5 runs); Tables 3-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

Note (not a finding): I initially suspected per-epoch test-set selection inside
`evaluate_synset`, but it evaluates on the test set only once after the full training loop
(`src/utils.py:514-516`), so `np.max(accs_test, axis=1)` is a max over a length-1 list — no
within-eval leakage. The remaining selection-on-test concern is the across-distillation-iteration
`best_acc`, filed above.

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 6 | high | Pruning JSONs absent (LD3M data builder broken as documented); no producers for Table 1, Fig. 4/6, CIFAR & ablation paths; deps unpinned |
| bug | 0 | - | Hardcoded W&B key / private paths noted as engineering friction, not method bugs |
| difference | 2 | medium | Repo/README is the IJCNN pruning paper; Eq. 7 implemented on pred_x0 with quantized z_T |
| methodology | 1 | medium | Reported accuracy is max-over-distillation-iterations test accuracy (selection on test) |

## 5. Closing lists

### Top take-aways (ranked)
1. **`pruning-json-missing`** (missing, high/high): every documented ImageNet LD3M command depends on per-class loss-pruning JSON files that do not ship and live at a private cluster path; with `--percent<100` the builder silently drops to an empty real set, so the headline tables are not reproducible as documented.
2. **`repo-is-pruning-paper`** (difference, medium/high): the linked repo is the code for a *different* (IJCNN pruning) paper that "is based on LD3M"; there is no LD3M-specific README, results table, or reproduce commands.
3. **`trace-table1-gradnorm-missing`** (missing, medium/high): the vanishing-gradient evidence (Table 1 gradient norms, Fig. 6 SNR) that motivates the whole method has no producing code.
4. **`trace-timing-missing`** (missing, medium/high): the efficiency claims (574 vs 693 min, 29.4 vs 31.2 GB, Fig. 4 sweep) have no producing script.
5. **`selective-best-acc`** (methodology, medium/medium): reported accuracy is the best test accuracy across distillation checkpoints (test-set model selection; no validation split).
6. **`cifar-ld3m-missing`** + **`ablation-diffusion-toggle-missing`** (missing, medium): no CIFAR-10 LD3M path and no flags to reproduce the Table 6/7 ablation conditions.

### Items that genuinely look fine
- The Eq. 7 residual decay direction is correct: `index/T` runs 1→0 over the reverse process (ddim.py:200), matching the paper's linearly-decaying skip from z_T.
- Both the latent Z (`f_latents`) and conditioning c (`latents`) are made leaf tensors with `requires_grad_(True)` and both receive gradients (`prepare_LDM_latents_encode`, `ldm_backward_encode`), matching the paper's "optimize Z and c end-to-end".
- Latent initialization encodes randomly selected real images via the LDM encoder (`prepare_LDM_latents_encode:99-105`), matching §4.2.
- Gradient checkpointing is used in the reverse step (`ddim.py:173,179`), matching §4.3.
- LDM unconditional guidance scale defaults to 3.0 (`distill_*_LD3M.py`), matching Supp. §C.
- `evaluate_synset` does NOT select the best epoch on the test set within a run (single end-of-training test eval, `utils.py:514-516`).
- The ImageNet main path loads the public cin256-v2 ImageNet LDM, matching §5.1.

### Open questions for the authors
- Which `--percent` produced each paper table — is the answer `--percent=100` (full dataset), making the pruning machinery a no-op for the LD3M paper? (`pruning-json-missing`)
- Was the paper's Eq. 7 (mean μθ, z_T) or the released code's form (pred_x0, VQ-quantized x_T) used for the reported numbers? (`eq7-residual-on-predx0`)
- Which field (`acc_test_mean` at fixed iter vs `best_acc` max-over-iters) populated Tables 3-5? (`selective-best-acc`)
- Where are the scripts for CIFAR-10 LD3M, the gradient-norm/SNR analyses, the timing/memory benchmarks, and the Table 6/7 ablation toggles?
