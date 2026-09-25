# Audit — HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis (NeurIPS 2025, #2911)

## Summary

The repository `code/aigc3d__HyPlaneHead/` is confirmed author code: its git remote is
`github.com/aigc3d/HyPlaneHead`, the README title/abstract/author list match the paper
verbatim, and it implements the paper's hy-plane representation (the `TriPlane_SingSph`
render modes and `TriPlaneSingSphGenerator_flatten`). It is a fork of NVlabs/eg3d +
SizheAn/PanoHead (acknowledged in the README).

The release is **inference-only**: it ships `gen_samples.py` (image/shape generation
from a pretrained `.pkl`), the EG3D rendering/network code, the stock EG3D metric library
(`metrics/`), and a download link for one pretrained 3+1 checkpoint (the OSS URL resolves,
HTTP 200, ~563 MB). It does **not** ship any training code (no `train.py`, no
`training_loop`), no dataset/preprocessing code, no config files for the 16 Table-1
variants, and no driver that actually computes FID/FID-random. Consequently none of the
32 numbers in Table 1 (the paper's only quantitative results) can be reproduced from this
repo; Figures 1, 5, 6 are qualitative and have no number to trace.

I read the paper (`paper.pdf`, §3–4 and Table 1) and audited `training/volumetric_rendering/renderer.py`,
`training/triplane.py`, `training/volumetric_rendering/near_equal_area_warping.py`,
`gen_samples.py`, `metrics/metric_main.py`, `metrics/metric_utils.py`,
`metrics/frechet_inception_distance.py`, `environment.yml`, `README.md`, and
`torch_utils/persistence.py`. I wrote three deterministic checks under `_audit_code/`
(run read-only, outputs in `_audit_code/out/`):
- `check_run_model_undefined_out.py` — AST proof that `ImportanceRenderer.run_model`
  uses/returns `out` without ever assigning it and never calls the `decoder`.
- `check_repo_completeness.py` — inventory of training/dataset/metric-driver/config/
  per-variant generator artefacts.
- `check_warping_snippet_bugs.py` — AST proof that the standalone
  `near_equal_area_warping.py` snippet references undefined names (`coordinates_sph`, `N`).

I also documented a persistence caveat (`_audit_code/out/persistence_caveat.txt`): EG3D's
`@persistence.persistent_class` pickles class source at train time, so the *released
checkpoint* may still run `gen_samples.py` via its own pickled source even though the
*repo's* `renderer.py` is broken.

## Result-traceability coverage table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|----------------|---------------|----------------|---------------|--------|
| Table 1, FID/FID-random for all 16 representation configs (32 numbers, e.g. HyPlaneHead 3+1 area-bias FID=8.14 / FID-random=9.88) | metric library exists (`metrics/frechet_inception_distance.py`), but NO driver computes/sweeps them; no training code; no dataset | — | — | MISSING (no metric driver, no training code, no configs) |
| Table 1 rows 14 & 16: area-bias split 384×384/384×128/384×128/128×128 (incl. headline 3+1 result) | `training/volumetric_rendering/renderer.py:435,454` only do even `H//2` 2×2 split | — | — | MISSING (no area-bias split impl) |
| Table 1 rows 15 & 16: Hy-plane (2+2) variant | only `TriGridGenerator` and `TriPlaneSingSphGenerator_flatten` (3+1) exist (`training/triplane.py:53,197`) | — | — | MISSING (no 2+2 generator) |
| FID-random metric (decouple conditioning pose from rendering pose) | `metrics/metric_utils.py:282-288` (`mode=='back'`) implements it, but no driver sets `mode` / `dataset_kwargs` | — | — | PARTIAL (logic present, never invoked; needs dataset) |
| FID metric (standard 50k) | `metrics/frechet_inception_distance.py:21-40` (stock EG3D) | — | — | PARTIAL (computes FID, but no driver/dataset to produce Table-1 values) |
| Pretrained HyPlaneHead (3+1) checkpoint | OSS URL in `README.md:37`, HTTP 200, 563 MB | — | — | Verified (download resolves) |
| Documented inference (`gen_samples.py`) producing images/shapes | `gen_samples.py:146` → `renderer.run_model` raises `NameError` (`renderer.py:469-471`) | — | — | MISMATCH (crashes on repo source; see run-model-undefined-out) |
| Fig. 5 qualitative SOTA comparison | `gen_samples.py` (image gen only) | — | — | N/A (qualitative, no numeric claim) |
| Fig. 6 single-view PTI inversion (PanoHead/SphereHead/HyPlaneHead) | (none) — no PTI/inversion script in repo | — | — | MISSING (no inversion code) |
| Fig. 1 unified-feature-map visualization | (none) — no feature-map dump/visualization script | — | — | MISSING (no visualization code) |
| Near-equal-area warping (Eq. 1–3) | `renderer.py:201-294` (integrated) and `near_equal_area_warping.py:31-106` (snippet, broken) | matches Eq. structurally | ~ | PARTIAL (integrated version present; standalone snippet broken) |

## missing

```yaml finding
id: no-training-code
category: missing
topic: "expected code completeness"
title: "No training code: Table 1 (the paper's only quantitative results) cannot be reproduced"
severity: high
confidence: high
status: finding
file: README.md
line_start: 35
line_end: 45
quote: |
  ## Getting Started

  1. Download the pre-trained checkpoint from [OSS_Link](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/checkpoints/nips2025/hyplanehead/hyplanehead-ckpt.pkl) and place it under the `model` directory.
  2. Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames.

  ## Generating Samples

  ```bash
  # Generate images and shapes (as .mrc files) using pre-trained model
  python gen_samples.py --trunc=0.7 --seeds=0-2 --network model/hyplanehead-ckpt.pkl --outdir=output
  ```
claim: "The repo is inference-only; it documents only gen_samples.py and a single pretrained checkpoint, and contains no train.py / training_loop, no dataset code, and no config files for the 16 Table-1 variants (verified by _audit_code/check_repo_completeness.py)."
concern: "Every quantitative result in the paper lives in Table 1 (16 representation configs × FID/FID-random); none of these numbers can be produced from the released code because the training pipeline, dataset, and per-variant configs are entirely absent."
resolution: "Authors: please release the training entrypoint, dataset/preprocessing code, and the config files for all 16 Table-1 rows so the FID/FID-random numbers can be reproduced."
cross_refs: ["no-metric-driver", "missing-area-bias-and-2plus2"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 1; §4.1 'All experiments are trained on eight NVIDIA V100 GPUs'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-metric-driver
category: missing
topic: "result traceability"
title: "No driver computes FID / FID-random; the metric library is never invoked"
severity: high
confidence: high
status: finding
file: metrics/metric_main.py
line_start: 44
line_end: 50
quote: |
  def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
      assert is_valid_metric(metric)
      opts = metric_utils.MetricOptions(**kwargs)

      # Calculate.
      start_time = time.time()
      results = _metric_dict[metric](opts)
claim: "The stock EG3D metric library (calc_metric / fid50k / FeatureStats) is present, but nothing in the repo ever calls calc_metric, instantiates MetricOptions, or sets opts.mode='back'/dataset_kwargs (verified: grep for callers of calc_metric returns only the definition)."
concern: "Without an entrypoint that configures the generator, the dataset's camera distribution, and the FID-random 'back' mode, the 32 FID/FID-random values in Table 1 have no computational source in the repo and cannot be checked."
resolution: "Authors: please provide the calc_metrics/evaluation driver script (the one that produced Table 1), including how FID-random's 'back' mode and the dataset camera labels are wired."
cross_refs: ["no-training-code", "fid-random-never-invoked"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 1; §4.2 FID and FID-random definitions"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-area-bias-and-2plus2
category: missing
topic: "ablations / model variants"
title: "Area-bias split and Hy-plane (2+2) variants (incl. headline result) have no implementation"
severity: high
confidence: high
status: finding
file: training/volumetric_rendering/renderer.py
line_start: 452
line_end: 463
quote: |
        elif render_mode == 'TriPlane_SingSph_flatten':
            N, _n_planes, C, H, W = planes.shape
            split_H, split_W = H // 2, W // 2
            planes = planes.unfold(3, split_H, split_W).unfold(4, split_H, split_W)  
            planes = planes.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
            planes = planes.view(N, 4, C, split_H, split_W)
            if 'sph2cir_flag' not in options:
                options['sph2cir_flag'] = False
            if 'cir2squ_flag' not in options:
                options['cir2squ_flag'] = False
            sampled_features = sample_from_triplane_singsph(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', \
                            box_warp=options['box_warp'], sph2cir_flag=options['sph2cir_flag'], cir2squ_flag=options['cir2squ_flag'])
claim: "The only unify-split implemented is the even 2x2 split (split_H = split_W = H//2). The paper's area-bias split (Table 1 rows 14,16: 384x384/384x128/384x128/128x128) and the Hy-plane (2+2) variant (rows 15,16) have no code: only TriGridGenerator and the 3+1 TriPlaneSingSphGenerator_flatten classes exist (verified _audit_code/check_repo_completeness.py)."
concern: "The paper's best-reported HyPlaneHead configuration is the area-bias 3+1 (FID=8.14 / FID-random=9.88, Table 1 row 14) and the generality argument rests on Hy-plane (2+2); neither variant is implemented, so the headline ablation conclusions cannot be reproduced."
resolution: "Authors: please release the area-bias splitting code (the 384/128 partition) and the Hy-plane (2+2) generator, or clarify which checkpoint corresponds to which Table-1 row."
cross_refs: ["no-training-code"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 1 rows 14,15,16; §4.2 'we split a 512x512 feature map into four parts via area-bias splitting'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-inversion-code
category: missing
topic: "result traceability"
title: "No PTI single-view inversion code for Figure 6"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  We compare our method with PanoHead and SphereHead on 3D full-head reconstruction from a
  single-view image using Pivotal Tuning Inversion (PTI) Roich et al. (2022). As shown in fig. 6,
claim: "§4.3 reports a single-view 3D-aware GAN inversion experiment (Fig. 6) using PTI, but the repo contains no inversion / PTI / encoder script (only gen_samples.py for forward generation)."
concern: "The Fig. 6 inversion comparison against PanoHead/SphereHead cannot be reproduced from the repo; this is qualitative so impact is low."
resolution: "Authors: please release the PTI inversion script used for Fig. 6, or note it was run with an external PTI implementation."
cross_refs: []
paper_ref: "§4.3 Single-view 3D-aware GAN Inversion; Fig. 6"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: run-model-undefined-out
category: bug
topic: "rendering pipeline"
title: "ImportanceRenderer.run_model returns undefined 'out'; decoder never called"
severity: medium
confidence: high
status: finding
file: training/volumetric_rendering/renderer.py
line_start: 465
line_end: 471
quote: |
        else:
            raise ValueError(f"no such render_mode: {render_mode}")

        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        out['xyz'] = sample_coordinates#.permute(0,2,1)[...,None]
        return out
claim: "Every branch of run_model assigns the queried features to `sampled_features`, but the function then references and returns `out` — which is never assigned in this function — and never calls the `decoder` argument (AST-verified by _audit_code/check_run_model_undefined_out.py). The caller `forward` expects out['rgb']/out['sigma'] (lines 358-359)."
concern: "run_model raises NameError/UnboundLocalError on `out['sigma']`/`out['xyz']`, so the repo's renderer cannot run as written; the decoder (OSGDecoder) is never invoked, so no rgb/sigma are produced — `gen_samples.py` (G.synthesis -> renderer.forward -> run_model) crashes from repo source."
resolution: "Insert the missing `out = decoder(sampled_features, sample_directions)` before line 468 (as in upstream EG3D). Caveat: EG3D's @persistence.persistent_class pickles class source at train time, so the released checkpoint may still run via its own pickled source (see _audit_code/out/persistence_caveat.txt) — please confirm the repo source matches the training-time source."
cross_refs: ["warping-snippet-undefined-names"]
check_script: _audit_code/check_run_model_undefined_out.py
paper_ref: "§3.3 'Features are queried from each plane and volumetrically rendered'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: warping-snippet-undefined-names
category: bug
topic: "near-equal-area warping"
title: "Standalone near_equal_area_warping.py snippet references undefined names"
severity: low
confidence: high
status: finding
file: training/volumetric_rendering/near_equal_area_warping.py
line_start: 85
line_end: 104
quote: |
      bs, C, H, W = sphere_features.shape
      coordinates = (2/box_warp) * coordinates 
      theta, phi, _radius = cartesian_to_spherical(coordinates_sph)

      if sph2cir_flag:
          print('convert coordinates on the spherical plane to its corresponding coordinates on the circle plane (Lambert azimuthal equal-area projection)')
          theta_denorm, phi_denorm = denormalize_theta_phi(theta, phi)
          r_circle, theta_circle = spherical_to_circle(theta_denorm, phi_denorm)
          u_cir, v_cir = circle_polar2cartesian(r_circle, theta_circle)
          if cir2squ_flag:
              print('convert coordinates on the circle plane to its corresponding UV coordinates on the square feature map (elliptical grid mapping)')
              u_squ, v_squ = cir2squ_mapping(u_cir, v_cir)
              projected_coordinates_sph = torch.stack([u_squ, v_squ], dim=-1).unsqueeze(1)
          else:
              projected_coordinates_sph = torch.stack([u_cir, v_cir], dim=-1).unsqueeze(1)
      else:
          print('original theta-phi version')
          projected_coordinates_sph = torch.stack([theta, phi], dim=-1).unsqueeze(1)
      output_sphere_features = torch.nn.functional.grid_sample(sphere_features, projected_coordinates_sph.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(bs, 1, N, C)
claim: "The standalone 'core implementation' file passed `coordinates` but calls cartesian_to_spherical(coordinates_sph) (undefined) at line 87 and reshapes to (bs,1,N,C) with N undefined at line 104 (AST-verified by _audit_code/check_warping_snippet_bugs.py)."
concern: "The file the repo presents as the 'Core implementation of near-equal-area warping' cannot run as shipped (NameError on coordinates_sph / N); the working version lives in renderer.py (sample_from_triplane_singsph), so impact is documentation-only."
resolution: "Authors: fix the snippet (use `coordinates` and `M`/`N` consistently) or mark it clearly as pseudo-code; the integrated renderer.py version is the one to use."
cross_refs: ["run-model-undefined-out"]
check_script: _audit_code/check_warping_snippet_bugs.py
paper_ref: "§3.1 Near-Equal-Area Warping; Eq. (1)-(3)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: fid-random-never-invoked
category: difference
topic: "evaluation consistency"
title: "FID-random ('back' mode) is implemented but never invoked; standard EG3D mode is the default"
severity: low
confidence: medium
status: finding
file: metrics/metric_utils.py
line_start: 273
line_end: 289
quote: |
      # set mode
      mode = opts.mode
      
      # Main loop.
      while not stats.is_full():
          images = []
          for _i in range(batch_size // batch_gen):
              z = torch.randn([batch_gen, G.z_dim], device=opts.device)
              
              if mode == 'back':
                  # back
                  w = G.mapping(z=z, c=c_cond.repeat(batch_gen, 1), truncation_psi=1)
                  img = G.synthesis(ws=w, c=next(c_iter), **opts.G_kwargs)['image']
              else:
                  # front and all
                  img = G(z=z, c=next(c_iter), **opts.G_kwargs)['image']
claim: "FID-random (decoupling conditioning camera c_cond from a different random render camera c_iter) is implemented as `mode=='back'`, with default mode='all' (metric_utils.py:37) computing conventional FID. The conditioning pose for 'back' is hardcoded to the front pose (LookAtPoseSampler.sample(3.14/2, 3.14/2, ...), lines 268-271)."
concern: "The paper's FID-random definition says c_con is *randomly sampled from the dataset camera distribution* (§4.2), whereas the code fixes c_con to the front pose; the two procedures differ, and no driver selects mode='back' to produce the FID-random column at all."
resolution: "Authors: confirm whether FID-random in Table 1 was computed with c_con fixed to the front pose (as in code) or randomly sampled per the paper text, and release the driver that sets mode='back'."
cross_refs: ["no-metric-driver"]
paper_ref: "§4.2 'we first randomly sample a camera parameter c_con from the dataset's camera distribution'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The repo ships no training/evaluation procedure to assess for
leakage, tuning-on-test, baseline fairness, or statistical errors — those concerns are
structurally inapplicable to an inference-only release and are instead captured as
`missing` (no-training-code, no-metric-driver). The integrated rendering math
(`sample_from_triplane_singsph`, the LAEA + elliptical-grid warping) is itself a valid
implementation of the described representation; no invalid procedure was found in the code
that *is* present.

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                |
|-------------|------------|--------------|----------------------------------------------------------------|
| missing     | 4          | high         | No training code, no metric driver, no area-bias/2+2 variants — Table 1 unreproducible. |
| bug         | 2          | medium       | run_model returns undefined `out` (repo source crashes); standalone warping snippet broken. |
| difference  | 1          | low          | FID-random fixes c_con to front pose vs paper's "randomly sampled". |
| methodology | 0          | -            | N/A — inference-only release; no procedure present to evaluate. |

## Top take-aways

1. **(missing, high)** `no-training-code` — the release is inference-only; no training
   code, dataset, or configs, so none of Table 1's 32 FID/FID-random numbers (the paper's
   only quantitative evidence) can be reproduced.
2. **(missing, high)** `no-metric-driver` — the EG3D metric library is present but nothing
   ever calls `calc_metric` / sets FID-random mode, so the reported metric values have no
   computational source in the repo.
3. **(missing, high)** `missing-area-bias-and-2plus2` — the headline HyPlaneHead config
   (area-bias 3+1, FID=8.14) and the Hy-plane (2+2) variant have no implementation; only
   the even-split 3+1 generator exists.
4. **(bug, medium)** `run-model-undefined-out` — `ImportanceRenderer.run_model` references
   and returns an undefined `out` and never calls the decoder; the repo's renderer crashes
   as written (the released checkpoint may still run via EG3D's pickled-source persistence).
5. **(difference, low)** `fid-random-never-invoked` — FID-random's conditioning pose is
   hardcoded to the front pose, whereas the paper says c_con is randomly sampled from the
   dataset camera distribution.
6. **(bug, low)** `warping-snippet-undefined-names` — the standalone
   `near_equal_area_warping.py` "core implementation" references undefined `coordinates_sph`
   and `N`; the working code lives in `renderer.py`.

## Items that genuinely look fine

- **Provenance**: confirmed author code (remote `aigc3d/HyPlaneHead`, README matches paper,
  hy-plane render modes implemented).
- **Pretrained checkpoint**: the OSS download URL in the README resolves (HTTP 200, 563 MB).
- **Dependencies**: `environment.yml` pins python/pytorch/cudatoolkit and lists `mrcfile`,
  `pyspng`, etc.; the `import mrcfile` in `gen_samples.py` is covered.
- **Integrated warping math**: `renderer.py:201-294` implements LAEA (Eq. 1) +
  elliptical-grid (Eq. 2-3) consistently with the paper; the dual-sphere weighting
  (`spherical_weights_w_bias`, `sample_from_sphtriplane`) matches Eq. (4)'s intent.
- **FID metric core**: `frechet_inception_distance.compute_fid` is the standard EG3D/Heusel
  implementation.

## Open questions for the authors

- Does the released `gen_samples.py` actually run end-to-end from the checkpoint despite the
  on-disk `run_model` bug (i.e., does the pickled training-time source differ from the repo
  source)? (relates to `run-model-undefined-out`)
- Was FID-random in Table 1 computed with the conditioning pose fixed to front (as the code
  does) or randomly sampled per the paper text? (relates to `fid-random-never-invoked`)
- Which Table-1 row corresponds to the released checkpoint, and will the area-bias / 2+2 /
  training / dataset / config artefacts be released to reproduce the full table?
