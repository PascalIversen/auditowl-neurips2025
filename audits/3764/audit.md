# Code-Repository Audit — DNAEdit (paper 3764)

## 1. Summary

DNAEdit is a **training-free** rectified-flow image-editing method (DNA + MVG) built
on pretrained SD3.5-medium and FLUX.1-dev, evaluated on PIE-Bench and a new
long-prompt benchmark DNA-Bench. The paper's headline evidence is entirely
quantitative tables: reconstruction metrics (Table 1: MSE/LPIPS/SSIM/NFE),
PIE-Bench editing (Table 2), DNA-Bench editing (Table 3), and a module ablation
(Table 4), all over six metrics (structure distance, PSNR, LPIPS, MSE, SSIM, CLIP
whole/edited).

The cloned repo (`code/xiechenxi99__DNAEdit_code/`, single commit `436e5b4`)
contains the **editing/inference pipeline only**:
`scripts/run_script_dnaedit.py` (driver) and `scripts/DNAEdit_utils.py`
(`DNAEdit_SD3`, `DNAEdit_FLUX` implementing Algorithms 1+2), a Gradio demo
`app.py`, a Wan-2.1 video-editing port under `wan-DNAEdit/`, two YAML configs, and
`DNA-Bench/long_mapping_file.json` (700 prompt records). It produces edited images
saved to disk; it does **not** compute any of the reported metrics, does not ship
the PIE-Bench / DNA-Bench images, has no dependency specification, and has no
ablation harness for Table 4. I read every Python file, the two configs, the
DNA-Bench JSON, and the README; I verified the DNA-Bench JSON record count and
average prompt length, and I traced the loop-variable shadowing and list-indexing
in `DNAEdit_utils.py` with a CPU-only script under `_audit_code/`.

I stayed strictly read-only on everything under `code/`.

### Scripts I ran
- `_audit_code/trace_indexing.py` — control-flow trace of the inversion/editing
  loops (checks the inner `for i in range(k)` shadowing and `dx_lst[i-jmp]`
  bounds). Output: `_audit_code/out/trace_indexing.txt`.
- `python3 -c` one-liners: DNA-Bench JSON has **700** records, average source-prompt
  length **33.1** words (paper reports 33.17, Sec. 4.1); grep sweeps for any
  metric code (`lpips|ssim|psnr|clip|structure|skimage|torchmetrics`) — **none**
  outside the Wan port; search for any `requirements*/environment*` file — **none**.

## 2. Traceability table

Every reported number is an evaluation **metric**. The repo contains no code that
computes any metric, so all rows that report a metric value are MISSING. The
editing pipeline that *produces the images the metrics would be computed on* is
present (cross-referenced where relevant).

| Paper artefact | Repo location (computation) | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 reconstruction MSE/LPIPS/SSIM (DNA 0.010/0.110/0.830, baselines) | (none) | — | — | MISSING (no metric code) |
| Table 1 NFE column (DNA=56, etc.) | (none) | — | — | MISSING |
| Fig. 1(c) reconstruction-MSE-vs-timestep curve | (none) | — | — | MISSING |
| Table 2 PIE-Bench: structure dist / PSNR / LPIPS / MSE / SSIM / CLIP whole+edited (DNAEdit + 11 baselines) | (none) | — | — | MISSING (no metric code; no baseline code) |
| Table 2 "Rank Avg." column | (none) | — | — | MISSING |
| Table 3 DNA-Bench: same 6 metrics (DNAEdit + 5 baselines) | (none) | — | — | MISSING |
| Table 4 ablation Exp 1–6 (Interpolation / +DNA / +ResOffset / +MVG variants) | (none) — no on/off toggles in `DNAEdit_utils.py` | — | — | MISSING (no ablation harness) |
| DNAEdit editing images (input to all metrics) — SD3 | `scripts/DNAEdit_utils.py:111-282` (`DNAEdit_SD3`) | n/a (image) | n/a (no GPU run) | Present (not executed) |
| DNAEdit editing images — FLUX | `scripts/DNAEdit_utils.py:286-482` (`DNAEdit_FLUX`) | n/a (image) | n/a | Present (not executed) |
| DNA-Bench: 700 prompt records, avg 33.17 words | `DNA-Bench/long_mapping_file.json` | 700 records, 33.1 words | ✓ | Verified (JSON only; images absent) |

## 3. Findings

## missing

```yaml finding
id: no-metric-computation-code
category: missing
topic: "result traceability / evaluation code"
title: "No code computes any reported metric (Tables 1-4, Fig 1c)"
severity: high
confidence: high
status: finding
file: scripts/run_script_dnaedit.py
line_start: 134
line_end: 139
quote: |
            # make sure to create the directories before saving
            save_dir = f"outputs/{args.save}/"
            save_dir = os.path.join(save_dir,"/".join(data_dict["image_path"].split('/')[:-1]))
            os.makedirs(save_dir, exist_ok=True)
            image_tar = image_tar[0].resize((original_width,original_height),Image.BILINEAR)
            image_tar.save(f"{save_dir}/{data_dict['image_path'].split('/')[-1]}")
claim: "The only driver script runs the editing pipeline and saves edited images to disk; the repo contains no code that computes MSE, PSNR, LPIPS, SSIM, structure distance, CLIP similarity, NFE, or the rank average reported in Tables 1-4 and Fig. 1(c). A grep across scripts/ and app.py for lpips|ssim|psnr|clip|structure|skimage|torchmetrics returns nothing."
concern: "Every quantitative claim in the paper (all reconstruction and editing metrics, the ablation, the headline 'best performance') is untraceable to code, so none of the numbers can be reproduced from this repository."
resolution: "Authors: please provide the evaluation scripts that compute structure distance, PSNR, LPIPS, MSE, SSIM, CLIP whole/edited, NFE, and the rank average from saved edited images and PIE-Bench/DNA-Bench masks."
cross_refs: ["pie-bench-data-and-path-missing", "no-ablation-harness"]
check_script: _audit_code/out/trace_indexing.txt
paper_ref: "Tables 1, 2, 3, 4; Fig. 1(c)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-specification
category: missing
topic: "expected code completeness / dependencies"
title: "No requirements/environment file; only two package versions named in README"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 49
line_end: 51
quote: |
    ### 1️⃣ Install

    1. Environment: torch=2.3.1 diffusers==0.30.1
claim: "The repo has no requirements.txt, environment.yml, setup.py, or pyproject.toml (verified by find). The README names only torch=2.3.1 and diffusers==0.30.1, while the code additionally imports numpy, PIL, yaml, gradio, ipdb, opencv (wan), and a specific diffusers internal (retrieve_timesteps from diffusers.pipelines.stable_diffusion)."
concern: "The Python environment cannot be reconstructed deterministically; in particular `diffusers==0.30.1` API compatibility for the FLUX/SD3 pipeline internals used here is unpinned for all other dependencies, blocking reproduction."
resolution: "Authors: please add a pinned requirements.txt / environment.yml covering torch, diffusers, transformers, numpy, pillow, pyyaml, and any CUDA constraints."
cross_refs: []
paper_ref: "Checklist Q5 (open access to code)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pie-bench-data-and-path-missing
category: missing
topic: "data availability"
title: "PIE-Bench images/JSON not shipped; driver reads a hardcoded path absent from repo"
severity: medium
confidence: high
status: finding
file: scripts/run_script_dnaedit.py
line_start: 62
line_end: 73
quote: |
        with open('../PIE-bench/long_mapping_file.json','r',encoding='utf-8') as f:
            dataset_dict= json.load(f)

        for data_dict in dataset_dict.values():
            
            src_prompt = data_dict["original_prompt"]
            tar_prompt = data_dict["editing_prompt"]
            print(src_prompt)
            print(tar_prompt)
            negative_prompt =  "" # optionally add support for negative prompts (SD3)
            image_src_path = data_dict["image_path"]
            image_src_path = os.path.join("../PIE-bench/annotation_images",image_src_path)
claim: "The driver opens a hardcoded relative path `../PIE-bench/long_mapping_file.json` and reads images from `../PIE-bench/annotation_images`, neither of which exists in the repo; PIE-Bench is a third-party benchmark that is not bundled. DNA-Bench ships only `DNA-Bench/long_mapping_file.json` (700 prompt records) — the corresponding images are not shipped either (find for .jpg/.png under DNA-Bench returns 0)."
concern: "Out of the box the only runnable experiment crashes on a missing file; the user must externally obtain PIE-Bench and edit the path, and the new DNA-Bench images are absent, so the long-prompt results cannot be reproduced from the repo alone."
resolution: "Authors: please ship or give a resolvable download for the PIE-Bench and DNA-Bench images, and replace the hardcoded `../PIE-bench/...` path with a config/CLI argument (README step 3 already implies it should point at DNA-Bench/long_mapping_file.json)."
cross_refs: ["no-metric-computation-code"]
paper_ref: "Sec. 4.1 (PIE-Bench, DNA-Bench)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-ablation-harness
category: missing
topic: "ablations"
title: "Table 4 ablation variants (Exp 1-5) cannot be run — no on/off toggles in code"
severity: medium
confidence: high
status: finding
file: scripts/DNAEdit_utils.py
line_start: 272
line_end: 282
quote: |
        x=mvg
        x_ref = x_ref.to(torch.float32)
        x_ref=x_ref+(t_prev-t_curr)*(noise_pred_tgt-v_lst[i-jmp])
        x_ref = x_ref.to(noise_pred_tgt.dtype)
        v = noise_pred_tgt*x+(random_noise-x_ref)/t_curr*(1-x)
        # v = noise_pred_tgt
        random_noise=random_noise.to(torch.float32)
        # random_noise = scheduler.step(noise_pred_tgt, timestep, random_noise, return_dict=False)[0]
        random_noise+=v*(t_prev-t_curr)
        random_noise=random_noise.to(v.dtype)
claim: "The editing functions always apply the full DNA + ResOffset + MVG configuration (Table 4 Exp 6). There are no flags/branches to disable DNA, disable ResOffset (dx_lst), or disable MVG to reproduce Table 4 Exps 1-5 (Interpolation / +DNA / +ResOffset / +DNA+ResOffset / +DNA+MVG); a grep for resoffset|ablation|use_mvg|use_dna|interpolation across the scripts and configs returns nothing."
concern: "The module-ablation table — the paper's evidence that DNA, ResOffset, and MVG each contribute — is not reproducible because the variant configurations are not implemented or exposed."
resolution: "Authors: please provide the ablation toggles (or separate scripts) that reproduce Exp 1-5 of Table 4 under the same split and metrics as Exp 6."
cross_refs: ["no-metric-computation-code"]
paper_ref: "Table 4 (ablation Exp 1-6)"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: app-undefined-name-and-missing-import
category: bug
topic: "demo app"
title: "app.py uses StableDiffusion3Pipeline without import and undefined model_choice"
severity: low
confidence: high
status: finding
file: app.py
line_start: 98
line_end: 108
quote: |
        elif model_type == 'SD3':
            pipe = StableDiffusion3Pipeline.from_pretrained("/home/notebook/data/personal/S9055029/hf/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
        pipe.to("cuda")
        loaded_pipe=pipe
        loaded_scheduler = pipe.scheduler

        

        
        # ========================================================================
        
        return f"✅ 模型 '{model_choice}' 加载成功！"
claim: "load_model() references StableDiffusion3Pipeline (never imported — only FluxPipeline is imported at line 6) on the SD3 branch, and the success return on line 108 references model_choice, a name defined only as a Gradio component in module scope, not in this function's parameters (the parameter is model_type). Model paths are also hardcoded to an author-specific absolute path /home/notebook/data/personal/S9055029/hf/..."
concern: "The Gradio demo raises NameError on the SD3 model branch (undefined StableDiffusion3Pipeline) and would also NameError on the success message; the hardcoded absolute checkpoint paths fail on any other machine."
resolution: "Authors: import StableDiffusion3Pipeline, replace model_choice with model_type in the f-string, and make checkpoint paths configurable. (Affects the demo only, not the reported metrics.)"
cross_refs: []
paper_ref: "README 'Inference on Your Image'"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: readme-mvg-recommendation-vs-paper
category: difference
topic: "hyperparameters"
title: "README recommends MVG coefficient ~0.85 while paper fixes eta=0.8 (configs use 0.8)"
severity: low
confidence: high
status: finding
file: README.md
line_start: 119
line_end: 123
quote: |
    *MVG Coefficient*
    - **Recommended**: ~0.85 (0.80 - 0.90)
    - **Effect**: Controls editing strength
    - Higher values (0.9-1.0): Stronger editing effects, more dramatic changes
    - Lower values (0.7-0.8): More faithful to original image, subtle modifications
claim: "The paper states 'the MVG coefficient eta is fixed at 0.8' (Sec. 4.1) and both benchmark configs (configs/DNAEdit_SD3_exp.yaml:10, configs/DNAEdit_FLUX_exp.yaml:11) set mvg: 0.8, matching the paper. The README's interactive-demo guidance instead recommends ~0.85. The README also notes the demo's higher-mvg = stronger-editing mapping, which is the same direction the code implements (v = noise_pred_tgt*mvg + (random_noise-x_ref)/t_curr*(1-mvg))."
concern: "Only a documentation/guidance mismatch for the interactive demo; the result-producing configs match the paper's eta=0.8, so reported numbers are unaffected."
resolution: "Authors: clarify that 0.8 was used for all reported benchmarks and that 0.85 is only a demo suggestion."
cross_refs: []
paper_ref: "Sec. 4.1 ('eta is fixed at 0.8')"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. This is a training-free, inference-only editing method
evaluated on a fixed third-party benchmark (PIE-Bench) and an extension of it
(DNA-Bench); there is no model training, no train/test split, no tuning loop, and
no statistical test in the repo, so the usual leakage/splitting/tuning failure
modes are structurally inapplicable (N/A). The editing algorithm as implemented
(`DNAEdit_utils.py`) faithfully follows Algorithms 1-2 of the paper as far as can
be verified by inspection; I could not execute it (no GPU / no checkpoints / no
benchmark images), so any numerical-faithfulness question is a reproducibility
(missing-evaluation) issue, not a methodology one.

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 4 | high | No metric code for any table/figure; no deps; benchmark data + ablation harness absent. |
| bug | 1 | low | Gradio demo (app.py) NameErrors on SD3 branch; demo only. |
| difference | 1 | low | README demo guidance (mvg~0.85) vs paper/config eta=0.8; results unaffected. |
| methodology | 0 | - | N/A — training-free inference-only method; no split/tuning/stats in scope. |

## 5. Closing lists

### Top take-aways (≤6, ranked)
1. **[missing]** No code computes any reported metric — every number in Tables 1-4
   and Fig. 1(c) is untraceable to the repo (`no-metric-computation-code`, high/high).
2. **[missing]** Table 4 ablation variants (Exp 1-5) have no toggles in the code;
   only the full config (Exp 6) is implemented (`no-ablation-harness`, medium/high).
3. **[missing]** PIE-Bench/DNA-Bench images not shipped and the driver opens a
   hardcoded `../PIE-bench/...` path absent from the repo
   (`pie-bench-data-and-path-missing`, medium/high).
4. **[missing]** No requirements/environment file; only two package versions named
   (`no-dependency-specification`, medium/high).
5. **[bug]** `app.py` references an unimported `StableDiffusion3Pipeline` and an
   undefined `model_choice`; demo only (`app-undefined-name-and-missing-import`, low/high).
6. **[difference]** README recommends MVG ~0.85 vs paper/config 0.8; results
   unaffected (`readme-mvg-recommendation-vs-paper`, low/high).

### Items that genuinely look fine
- The DNA + MVG editing algorithm (`DNAEdit_SD3`, `DNAEdit_FLUX`) implements
  Algorithms 1-2 by inspection, including the residual-offset reuse (`dx_lst`) and
  the source-velocity reuse (`v_lst`) the paper describes for NFE savings.
- Seeds are set comprehensively (random, numpy, torch, cuda) in the driver
  (`run_script_dnaedit.py:57-60`).
- The inner `for i in range(k)` loop-variable shadowing is benign: with k=1 the
  break-check at the top of the outer loop uses the outer index before the inner
  loop runs, and `dx_lst`/`v_lst` indexing in the editing loop stays in bounds
  (verified by `_audit_code/trace_indexing.py`: 24 list entries, editing uses
  indices 0-23).
- DNA-Bench JSON matches the paper: 700 records, average source-prompt length 33.1
  words (paper: 33.17, Sec. 4.1).
- The MVG coefficient in both result-producing configs (0.8) matches the paper.

### Open questions for the authors
- Were any of the baseline metrics in Tables 2-3 recomputed by the authors under a
  shared environment (paper says "official implementations and default settings"),
  or copied from the original papers? No baseline or metric code is present to
  verify either way. (Tracks `no-metric-computation-code`.)
- Will the released code include the evaluation harness and the DNA-Bench images,
  given the checklist (Q5) states data are public and code/data will be released on
  acceptance?
