# Code audit — RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers (paper 646)

## 1. Summary

RoPECraft is a training-free video motion-transfer method built on the Wan2.1-1.3B
text-to-video diffusion transformer. It (a) extracts optical flow from a reference
video and warps the rotary positional embeddings (RoPE) of the DiT, then (b)
optimizes a per-timestep additive RoPE delta during the first few denoising steps
with a flow-matching + Fourier-phase objective. The paper also proposes a new
metric, Fréchet Trajectory Distance (FTD).

The repository (`berkegokmen1__RoPECraft`, single commit `da5788b`) contains exactly
two Python files plus a README and `requirements.txt`:
- `generate.py` — runs the full method on **one** input video / prompt and writes one
  output mp4.
- `ftd.py` — computes FTD for **one** (reference, generated) video pair using
  CoTracker3.

There is **no** code for the other reported metrics (CD-FVD, CLIP similarity, Motion
Fidelity), **no** implementation of any of the five baselines (GWTF, SMM, MOFT,
DitFlow, ConMo), **no** DAVIS prompt dataset or evaluation/aggregation harness, and
**no** ablation or user-study code. The entire quantitative content of the paper
(Tables 1–5, runtime numbers, all headline numbers in §5.4) therefore has no
producing code in the repo.

What I did: read both Python files and the README; read the paper (Tables 1–5, §4
method/algorithms, §5 metrics/baselines, Eq. (2)/(3)); ran static checks under
`_audit_code/` (`check_static_issues.py`) for a use-before-definition bug, the
broken `--frames 81` path, and dependency listing. I could not execute the pipeline
(needs an H200-class GPU, ~40GB VRAM, and downloads Wan2.1 + CoTracker3), so dynamic
findings are limited to static analysis.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — Ours MF/CD-FVD/CLIP/FTD(FG)/FTD(FG+BG) = 0.5816 / 1284.58 / 0.2350 / 0.2644 / 0.2584 (mean±std over DAVIS) | (none — no eval harness, no CD-FVD/CLIP/MF code, no baselines) | — | — | MISSING |
| Table 1 — all baseline rows (GWTF, SMM, MOFT, DitFlow×2, ConMo) | (none — no baseline implementations) | — | — | MISSING |
| Table 1 — runtime 109.231±3.112 s etc. | (none — no timing harness) | — | — | MISSING |
| Table 2 — ablation Alg.1+opt / Alg.2+opt / +phase (MF, CLIP, FTD) | (none — no ablation/aggregation driver; flags exist in `generate.py` but no script sweeps them or computes MF/CLIP) | — | — | MISSING |
| Table 3 — (t,s) hyperparameter ablation (MF, CD-FVD, CLIP, FTD) | (none) | — | — | MISSING |
| Tables 4–5 — user study percentages | (none — N/A as code, human study) | — | — | MISSING (human study, expected absent) |
| FTD metric definition (Eq. 3, §5.2) | `ftd.py:128-181` | RMS of squared discrete Fréchet distances over kept tracks | ✓ (formula matches) | Verified (formula only; no dataset values) |
| Fig. 1/4/5/6/7/10 qualitative generations | `generate.py` (single-video inference) | one video per run | n/a (qualitative) | Partial (pipeline present, not runnable here) |

## 3. Findings

## missing

```yaml finding
id: no-eval-harness-baselines-metrics
category: missing
topic: "result traceability"
title: "No code for Table 1/2/3 metrics, baselines, or aggregation"
severity: high
confidence: high
status: finding
file: README.md
line_start: 65
line_end: 75
quote: |
  ## Evaluation

  The paper details a new motion transfer metric called `Fréchet Trajectory Distance` (FTD). You can compute a sample FTD metric from a generated video provided in `assets/`.

  ```bash
  python ftd.py \
      --reference_video assets/blackswan.mp4 \
      --target_video assets/output/A_child_in_a_duck_co.mp4 \
      --mask_path assets/mask/blackswan \
      --num_points 100
  ```
claim: "The only evaluation script shipped is ftd.py, which computes FTD for a single (reference, generated) video pair; there is no code for CD-FVD, CLIP similarity, or Motion Fidelity, no implementation of any of the five baselines (GWTF, SMM, MOFT, DitFlow, ConMo), no DAVIS prompt set, and no harness that aggregates over a dataset to produce the mean±std numbers in Table 1."
concern: "Every quantitative claim in the paper (Tables 1-3, runtime, all §5.4 headline numbers) is untraceable to repo code, so the central claim that RoPECraft 'outperforms all recently published methods' quantitatively cannot be reproduced or checked from this repository."
resolution: "Authors: please release the evaluation harness (CD-FVD, CLIP, MF computation), the DAVIS prompt list, the adapted baseline implementations, and the script that aggregates per-video metrics into the Table 1/2/3 numbers."
cross_refs: ["frames81-broken"]
check_script: _audit_code/check_static_issues.py
paper_ref: "Tables 1-3, §5.4"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: cotracker-dep-unlisted
category: missing
topic: "dependencies"
title: "CoTracker3 (required for FTD/MF) is not a listed dependency and defaults to a hardcoded path"
severity: medium
confidence: high
status: finding
file: ftd.py
line_start: 17
line_end: 17
quote: |
  DEFAULT_COTRACKER_ROOT = Path("/home/ubuntu/co-tracker")
claim: "ftd.py obtains trajectories from CoTracker3 via torch.hub, defaulting to a hardcoded local clone path /home/ubuntu/co-tracker, falling back to downloading facebookresearch/co-tracker at runtime; CoTracker is not mentioned in requirements.txt and frechetdist is unpinned."
concern: "FTD (and the paper's MF metric, which §5.1 says also uses Co-Tracker3) cannot be computed without an unlisted external repo fetched at runtime, and frechetdist has no version pin, so the metric environment is not reproducibly specified."
resolution: "Authors: add CoTracker3 install instructions / pin, pin frechetdist, and remove or document the hardcoded /home/ubuntu/co-tracker default."
cross_refs: ["no-eval-harness-baselines-metrics"]
check_script: _audit_code/check_static_issues.py
paper_ref: "§5.1 'For MF and FTD, the trajectories are obtained using Co-Tracker3'"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: frames81-broken
category: bug
topic: "frame-count handling"
title: "--frames 81 crashes: RoPE num_frames stays 13 and global latents are hardcoded to 13"
severity: medium
confidence: high
status: finding
file: generate.py
line_start: 855
line_end: 873
quote: |
        indices = torch.linspace(0, gt_flow_feats.size(0) - 1, 13 if num_frames == 49 else 21).long()
        gt_flow_feats = gt_flow_feats[indices]

        custom_rope = CustomWanRotaryPosEmbed(
            attention_head_dim=transformer.config.attention_head_dim,
            patch_size=transformer.config.patch_size,
            max_seq_len=transformer.config.rope_max_seq_len,
            u=gt_flow_feats[:, 0, :, :].to("cpu"),
            v=gt_flow_feats[:, 1, :, :].to("cpu"),
            divisor=divisor,
            enable_smoothing=enable_smoothing,
            theta=theta,
            mu=mu,
            std=std,
            mu_time=mu_time,
            std_time=std_time,
            enable_lookup_norm=enable_lookup_norm,
            lookup_norm_thr=lookup_norm_thr,
            # num_frames=13 if num_frames == 49 else 21,
        )
claim: "For --frames 81 the code selects 21 flow frames (line 855) but the num_frames argument to CustomWanRotaryPosEmbed is commented out (line 873), so it keeps its default num_frames=13 (def at line 148) and reshapes the 21*W rotary rows into [13, W, H, -1] (line 324), which fails; additionally the global latents tensor is hardcoded to 13 latent frames (`torch.randn(1, 16, 13, 60, 104)`, line 1028)."
concern: "The advertised --frames 81 mode (an explicit argparse choice) cannot run, so any 81-frame results in the paper are not reproducible from this code without edits."
resolution: "Authors: confirm whether 81-frame results were produced with this code; if so, restore the num_frames argument and parameterize the latents tensor by frame count."
cross_refs: ["no-eval-harness-baselines-metrics"]
check_script: _audit_code/check_static_issues.py
paper_ref: "generate.py --frames choices=[49,81]"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: delta-p-undefined-non-train
category: bug
topic: "control flow"
title: "delta_p referenced but never defined when train_mode=False"
severity: low
confidence: high
status: finding
file: generate.py
line_start: 944
line_end: 953
quote: |
            # train mode or not, perform forward passes to get x_t-1
            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=custom_image_rotary_emb + delta_p,
                )[0]
claim: "delta_p is assigned only inside the `if train_mode:` optimization loop (line 915); the always-executed no_grad forward at line 952 references delta_p, so with train_mode=False (the inference-only branch that pre-allocates zero tunable_image_rotary_emb at line 899) delta_p is undefined and raises NameError."
concern: "The non-training (zero-delta) inference path is dead code that would crash; only train_mode=True (hardcoded in generate(), line 1067) works, so the paper's 'no optimization' ablation rows cannot be produced by toggling this flag."
resolution: "Authors: confirm train_mode=False was never used, or define delta_p in the else branch (e.g. delta_p = sum(tunable_image_rotary_emb[:i+1]))."
cross_refs: ["no-eval-harness-baselines-metrics"]
check_script: _audit_code/check_static_issues.py
paper_ref: "Table 2 ablation rows (Alg.1+opt etc.)"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: generate-hardcodes-divisor-smoothing
category: difference
topic: "hyperparameters"
title: "generate() hardcodes divisor=8 and smoothing params overriding documented defaults"
severity: low
confidence: medium
status: question
file: generate.py
line_start: 1069
line_end: 1079
quote: |
        train_mode=True,
        tuned_rope_dtype=torch.complex64,
        divisor=8,  # 16
        enable_smoothing=True,  # True
        mu=21,
        std=11,  ## 21,11
        mu_time=5,
        std_time=3,  # 11,5
        enable_lookup_norm=True,
        lookup_norm_thr=1.0,
        n_replace_gt_mod=range(args.n_replace_gt_mod),
        num_optim_steps=args.num_optim_steps,
        tune_wo_warped_uv=(1 - args.start_with_uv_warped),
claim: "The driver generate() overrides the documented tune_p defaults with divisor=8 (inline comment '# 16'), enable_lookup_norm=True (tune_p default False), and temporal-smoothing mu_time=5/std_time=3, while the inline comments record different prior values; these flow-warping hyperparameters are not stated in the main paper (deferred to an unavailable Supplementary)."
concern: "The actual hyperparameters used to produce results differ from the in-code defaults and carry contradictory comments, and the main paper gives no values, so the exact configuration behind the reported numbers is not independently verifiable."
resolution: "Authors: state in the paper (or repo) the divisor, smoothing kernel sizes, lookup-norm setting, and theta actually used for the Table 1 results."
cross_refs: []
paper_ref: "§5.1 'The hyper parameters are detailed in Supplementary.'"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ftd-point-count-half
category: difference
topic: "metric definition"
title: "Default FTD samples n total points (n/2 fg + n/2 bg), not n foreground AND n background"
severity: low
confidence: medium
status: question
file: ftd.py
line_start: 146
line_end: 151
quote: |
    if use_fg_mask_only:
        queries = sample_points_inside_mask(mask, n_points, device)
    else:
        fg_points = max(1, n_points // 2)
        bg_points = max(0, n_points - fg_points)
        queries = sample_points_from_mask(mask, fg_points, bg_points, device)
claim: "Fig. 8 / §5.2 describe sampling 'n foreground (red) and n background (green)' seeds (2n total), but the released ftd.py splits the single --num_points argument into n/2 foreground and n/2 background (n total) in the default (FG+BG) mode."
concern: "The released metric samples half as many points per region as the figure describes; with the default --num_points 100 this is 50 fg + 50 bg rather than 100+100, a definitional difference that could shift FTD values."
resolution: "Authors: clarify the intended point count per region and which setting produced the Table 1 FTD numbers."
cross_refs: ["no-eval-harness-baselines-metrics"]
paper_ref: "Fig. 8 caption / §5.2"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ablation-tables-inconsistent-with-main
category: methodology
topic: "internal consistency / selective reporting"
title: "Tables 2-3 absolute MF/CLIP/FTD for the full method differ sharply from Table 1"
severity: medium
confidence: low
status: question
file: paper.pdf
quote: |
  Alg.2 + opt. + phase 0.7210 0.1656 0.2060
claim: "Table 2's full method (Alg.2+opt+phase) reports MF=0.7210, CLIP=0.1656, FTD=0.2060, and Table 3's best (10,5) row reports MF=0.5675, CLIP=0.1664, FTD=0.2633, whereas Table 1 'Ours' reports MF=0.5816, CLIP=0.2350, FTD(FG)=0.2644 for ostensibly the same final method; the CLIP values in particular differ by ~0.07 and MF/FTD differ substantially across tables."
concern: "The same final configuration yields materially different metric values across tables with no stated explanation (different subset, point count, or seed?), which undermines the reliability of the ablation comparisons and the main-table numbers; with no producing code (see no-eval-harness-baselines-metrics) the discrepancy cannot be resolved from the repo."
resolution: "Authors: explain why Tables 1, 2, and 3 report different absolute metrics for the full method (e.g., different evaluation subsets or point sampling), and report all tables on the same protocol."
cross_refs: ["no-eval-harness-baselines-metrics"]
paper_ref: "Tables 1, 2, 3"
tags: [stats:statcheck, forensics:post-hoc-selection]
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 2          | high         | No eval harness / baselines / CD-FVD/CLIP/MF; CoTracker unlisted |
| bug         | 2          | medium       | --frames 81 crashes; delta_p undefined in non-train branch  |
| difference  | 2          | low          | Hardcoded undocumented hyperparams; FTD point count halved  |
| methodology | 1          | medium       | Tables 1/2/3 metrics inconsistent for the same method (question) |

## 5. Closing lists

### Top take-aways (≤6, ranked)
1. **[missing, high]** `no-eval-harness-baselines-metrics` — the repo ships only single-video inference and single-pair FTD; none of Tables 1–3, the baselines, the CD-FVD/CLIP/MF metrics, the DAVIS prompts, or any aggregation are present, so the entire quantitative comparison is untraceable.
2. **[missing, medium]** `cotracker-dep-unlisted` — FTD/MF depend on CoTracker3, which is unlisted, hardcoded to `/home/ubuntu/co-tracker`, and `frechetdist` is unpinned.
3. **[bug, medium]** `frames81-broken` — the advertised `--frames 81` mode cannot run (RoPE num_frames stuck at 13; latents hardcoded to 13 frames).
4. **[methodology, medium / low-conf]** `ablation-tables-inconsistent-with-main` — the same final method reports sharply different MF/CLIP/FTD across Tables 1, 2, and 3 with no explanation.
5. **[bug, low]** `delta-p-undefined-non-train` — the `train_mode=False` inference path references an undefined `delta_p` and would crash.
6. **[difference, low]** `generate-hardcodes-divisor-smoothing` — driver hardcodes `divisor=8` and smoothing params that contradict in-code comments and are absent from the main paper.

### Items that genuinely look fine
- The FTD computation in `ftd.py` faithfully implements Eq. (3): RMS over squared discrete Fréchet distances (`fd**2` then `sqrt(mean)`), with resolution-invariant W/H coordinate normalization and occlusion-aware fill/drop matching §5.2.
- The core method in `generate.py` is internally consistent with §4: motion-augmented RoPE (Algorithm 2) via cumulative flow displacement, the additive per-timestep RoPE delta optimized with `phase_constraint_loss` (unit-circle cos/sin L1, matching Eq. 2) plus MSE flow-matching loss, CFG, and the Wan2.1-1.3B backbone (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`).
- "Training-free" is accurate: only the RoPE delta is optimized at inference; model weights are frozen (`transformer.requires_grad_(False)`).
- Seeding is comprehensive (`seed_everything` covers torch/cuda/numpy/random) and a fixed latent is used for deterministic runs.
- Most pip dependencies are version-pinned in `requirements.txt`.

### Open questions for the authors
- Where is the evaluation/aggregation harness, the DAVIS prompt set, and the adapted baseline code that produced Tables 1–3? (`no-eval-harness-baselines-metrics`)
- Why do Tables 1, 2, and 3 report different absolute metric values for the same final method? (`ablation-tables-inconsistent-with-main`)
- Which exact flow-warping hyperparameters (divisor, smoothing, theta, lookup-norm) produced the reported numbers, and was `--frames 81` ever used? (`generate-hardcodes-divisor-smoothing`, `frames81-broken`)
