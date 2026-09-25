#!/usr/bin/env python3
"""
make_worksheets.py — human-eval worksheet per paper for the reliability runs.

Uses the project's MERGED-BY-DEFECT (soft) matcher, not the strict anchor one:
the 10 runs' raw findings are grouped into distinct defects (the same adversarial
re-clustering behind studies/reliability/data/merged_clusters.json and the D-numbered
census worksheet of the original human-eval apparatus, since removed). Because the new PDF+text runs produced their
own findings, the clusters below were re-derived for THESE runs; fids are
`r{run}#{idx}` into studies/reliability/<paper>/run_NN/findings.json.

Unlike the blinded census worksheet, this one SHOWS severity + confidence +
detection rate (the reviewer asked for them) and uses a 4-way verdict box.

Run:  python studies/reliability/make_worksheets.py
Writes: studies/reliability/HUMAN_EVAL_<pid>.md   (one per paper)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

RB = Path(__file__).resolve().parent
FID_RE = re.compile(r"r(\d+)#(\d+)")
CONF_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}
SEV_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}
CATLAB = {"missing": "Missing code / data", "difference": "Paper–code mismatch",
          "bug": "Technical bug", "methodology": "Methodology / validity"}

# Paper folder names
PAPERS = {
    "1829": "1829",
    "1333": "1333",
    "2657": "2657",
    "2578": "2578",
    "2371": "2371",
}

# --- MERGED-BY-DEFECT clusters, re-derived for the reliability runs --------
# Each entry: (label, [fids]).  Every raw finding is assigned to exactly one
# defect (validated at the end against the per-run totals).
CLUSTERS = {
"1829": [
 ("Baseline & OrthoTrans/NormLin plug-in model code absent — Tables 1/5/6 not reproducible (no iTransformer/PatchTST/RLinear/Timer etc. in repo)",
  ["r01#0","r02#0","r03#2","r07#0","r07#2","r08#0","r10#0"]),
 ("Headline benchmark datasets (ECL/ETT/Traffic/Solar/PEMS/Exchange/METR) and their precomputed Q-matrices are not shipped",
  ["r02#1","r03#3","r04#1","r05#2","r08#1","r10#3"]),
 ("OLinear-C per-dataset channel-correlation matrices (COV_channel.npy) missing for every dataset except weather",
  ["r03#1","r07#1"]),
 ("No code computes the reported std dev / 99% CIs / Student's t-test (Tables 13/14/15/26)",
  ["r02#2","r05#0","r06#0","r10#1"]),
 ("'Seven random seeds' claimed but code uses a single hardcoded seed / robustness runs never explicitly seeded",
  ["r05#1","r10#5"]),
 ("requirements.txt fully unpinned (no versions) and lists 'pywt' — wrong PyPI name (should be PyWavelets); install fails",
  ["r02#3","r03#0","r04#0","r05#3","r06#1","r06#2","r07#3","r08#2","r10#2"]),
 ("patoolib and tqdm imported at startup but absent from requirements.txt (undeclared dependencies)",
  ["r09#0"]),
 ("Basis-ablation scripts request an unregistered model key 'OLinear_wavelet_concat' → KeyError",
  ["r01#1","r07#4","r10#4"]),
 ("OLinear-C missing-channel-file guard uses `assert ValueError(...)` (always truthy; never raises)",
  ["r03#4","r07#5"]),
 ("OrthoTrans adds undocumented learnable additive terms delta1/delta2 absent from the paper's transform equations",
  ["r04#2","r09#1"]),
 ("Reported metric is selected on the test set (best-of-N runs / min over test-batch-sizes & decoder stages by test MSE+MAE)",
  ["r01#2","r01#3","r04#4","r05#4","r06#4","r06#5"]),
 ("Q-matrix generation notebook is single-file with hardcoded Windows paths, fixed lags, and is not wired into the pipeline",
  ["r02#4","r03#5"]),
 ("Early-stopping patience / epoch count in shipped scripts (patience 5–8, 30 epochs) disagree with the paper (patience 10, 50 epochs)",
  ["r02#5","r04#3","r06#3"]),
 ("Reported 'MASE' scales by the test-target 1-step naive error, not the in-sample seasonal naive denominator",
  ["r02#6"]),
 ("Per-dataset hyperparameters are tuned over ranges but the selection criterion is left unstated",
  ["r02#7"]),
 ("run.py references Exp_Short_Term_Forecast / Exp_Long_Term_Forecast_Partial / Exp_Imputation classes that are never imported",
  ["r08#3"]),
],
"1333": [
 ("Latent restoration network R_theta (Stage-2 core, Eq.7/9) is absent — Stage-2 only auto-encodes via the VAE; L_Res training not implemented",
  ["r01#0","r02#0","r02#3","r03#0","r04#0","r04#1","r05#1","r06#1","r07#0","r07#7","r08#1","r09#1","r10#1"]),
 ("No evaluation / inference / metric entrypoint — none of Tables 1–5 or Fig. 2 can be computed from the repo",
  ["r01#1","r02#1","r03#1","r04#2","r05#0","r06#0","r07#1","r08#0","r09#0","r10#0"]),
 ("Datasets and pretrained VAE/DINOv2 weights are referenced by config but not shipped; no fetch script",
  ["r01#2","r02#2","r05#2","r07#3","r09#3"]),
 ("No dependency specification (no requirements.txt / environment.yml / setup.py) for a heavy multi-dependency repo",
  ["r01#3","r02#4","r03#2","r04#3","r05#3","r06#2","r07#4","r08#2","r09#2","r10#2"]),
 ("README is an empty one-line stub (no commands, no results table); code-availability deferred",
  ["r05#4","r08#3","r10#3"]),
 ("Stage-2 config requests VAE arch type 'RAVAE' which is not registered → KeyError at network build",
  ["r01#4","r02#5","r03#3","r04#4","r05#5","r06#4","r07#5","r08#4","r09#5","r10#5"]),
 ("experiments_root / training output dir hardcoded to the authors' cluster path — breaks any other machine",
  ["r01#5","r02#6","r03#4","r05#6","r06#5","r08#5","r09#6","r10#6"]),
 ("Hardcoded author-machine absolute paths injected via sys.path.append in core modules (non-existent elsewhere)",
  ["r01#6","r02#7","r04#5","r05#7","r07#6","r08#6","r10#7"]),
 ("LoRA alpha blending mismatch — opposite LoRA scaled at runtime by alpha=0.5 instead of frozen at base / weight-delta blending",
  ["r01#7","r09#7","r10#9"]),
 ("Decoder PHF-LoRA step also minimizes the HF-fidelity L1 loss, not the perception-only (GAN) loss of Eq. 9",
  ["r05#8","r06#6","r10#8"]),
 ("LEqv is computed on the latent of the PERTURBED input, not the clean-image latent z_clean as in Eq. 5",
  ["r02#8","r04#6","r07#8","r08#7"]),
 ("Stage-1 reconstruction encodes the perturbed image, not I_clean as defined in Eq. 2",
  ["r06#7"]),
 ("Shipped configs are '*_example' placeholders with toy batch sizes and generic (non-paper) hyperparameters",
  ["r03#5","r04#7"]),
 ("Key architecture modules are placeholder/substitute stubs (e.g. vgg_arch), not the paper's components",
  ["r06#3"]),
 ("Backbone-integration experiments absent — Table 5(c) backbones (Restormer/NAFNet/SFHformer) and Table 3/4 integration/generalization have no code",
  ["r07#2","r10#4"]),
 ("basicsr/__init__.py imports `.test` but basicsr/test.py does not exist → import crashes on load",
  ["r09#4"]),
 ("LInv adds an undocumented trainable 1x1 conv projector before the DINOv2 alignment",
  ["r07#9"]),
],
"2657": [
 ("TCCM's training-epoch count (the paper's only data-dependent hyperparameter) is a hardcoded 47-branch if/elif table (values 1..5000); no code implements the unsupervised CSM / Improved Contrast Score Margin selection the paper claims (some values commented 'Chosen first from \"100 or 1\"')",
  ["r01#0","r02#1","r03#0","r03#4","r04#0","r05#2","r06#0","r07#0","r08#0","r09#1","r10#1"]),
 ("Explainability — a titular contribution (RQ3) — has no code: no per-feature residual attribution, no MNIST 1-vs-7 figure (Fig 4, AUROC 0.76), no ExactMatch/Jaccard metrics (Table 3); the model returns only a scalar L2 score",
  ["r01#2","r02#0","r03#1","r04#2","r05#0","r06#2","r07#1","r09#0","r10#0"]),
 ("No code for the Friedman + Nemenyi post-hoc significance tests / critical-difference diagrams (Figs 21-22, App D.5); aggregation stops at mean per-detector ranks",
  ["r01#1","r03#2","r04#1","r05#1","r06#1","r07#2","r09#2"]),
 ("requirements.txt pins torch==1.13.1 while the paper (App B.3) and README state PyTorch 2.0 / Python 3.9.21",
  ["r01#5","r02#4","r03#5","r04#7","r05#5","r07#4","r10#6"]),
 ("run_knn.sh invokes `python Full_experiments.py` but the driver is `FullExperiments.py` (camelCase) — file-not-found on every call; model index 50 also maps to INNE_semisup, not KNN",
  ["r01#4","r02#3","r04#5","r05#3","r09#4","r10#2"]),
 ("run_semisupervise.sh loops `for j in {45..50}` (6 indices) but comments '7 models'; the 7th force-inductive model at index 51 is never launched",
  ["r05#4","r06#4","r07#3","r08#1","r09#3","r10#3"]),
 ("Only TCCM gets per-dataset tuned epochs (1..5000) via determine_FMAD_hyperparameters; every baseline runs its source-default config — fair-comparison concern (paper does state baselines use defaults)",
  ["r01#6","r02#5","r05#6","r06#6","r08#2"]),
 ("The CSM epoch criterion compares top-k 'predicted anomalies' vs presumed inliers, but the training split is normal-only — the criterion needs anomalies, implying it is computed on the test set (leakage)",
  ["r04#8","r05#7","r09#7"]),
 ("README says `AggregateResults.py --semi_only` produces Rank_ROC.pdf / Rank_PR.pdf (Fig 2 box plots), but the script imports no plotting library and writes only CSVs",
  ["r03#3","r09#6","r10#5"]),
 ("AblationStudies.py implements only Figs 12-15; no code for Fig 16 (feature normalization), Fig 17 (time-interpolated inputs z_t=tz), or Fig 18 (TCCM vs AE+TimeEmbedding)",
  ["r02#2","r04#3","r06#3"]),
 ("model_worker enqueues a 2-tuple on out-of-try exceptions (OOM / limit_memory failure), but run_model_with_timeout always unpacks 4 elements → the real error is masked by a ValueError",
  ["r01#3","r10#4"]),
 ("determine_FMAD_hyperparameters is an if/elif chain with no final else; an unmatched dataset name would raise UnboundLocalError (all 47 shipped datasets do resolve)",
  ["r04#6"]),
 ("FullExperiments.py overrides GOAD's n_epoch to 25 (GOAD's own default is 1), contradicting the paper's 'baselines use default configurations' statement",
  ["r09#5"]),
 ("ContaminationStudies re-stacks already-StandardScaler'd train+test arrays and applies a fresh 50/50 split — double-scaling / split inconsistency",
  ["r03#6"]),
 ("Figure 1 (vector-field visualization), Figure 6 (anomaly-mismatch boxplots validating Props 3-5), and the §C.3 representation-collapse tracking have no shipping code",
  ["r04#4"]),
 ("AggregateResults defaults to semi_only=True, dropping the 7 transductive models — the README's main-result command may not match the paper's figure population",
  ["r06#5"]),
],
"2578": [
 ("HQ-VSR video-processing pipeline (Sec 3.3 / Fig 3 / Eq 8: metadata + scene + quality filtering with CLIP-IQA/FasterVQA/DOVER thresholds + optical-flow motion-area bounding-box cropping) absent from the repo — the dataset-curation contribution and the Tab 1c/1d (+Filter/+Motion) ablations are not reproducible; README §TODO confirms it is unreleased (only the final dataset blob is linked)",
  ["r01#0","r02#0","r05#0","r06#3","r07#2","r08#0","r09#0"]),
 ("FasterVQA metric (reported for every Table 2 row and as a Fig 1 axis) has NO computation script anywhere in the repo — a tree-wide search finds 'fastervqa' only in the README §TODO line 269, which confirms the script is unreleased",
  ["r02#1","r03#0","r04#0","r06#0","r07#0","r09#1"]),
 ("eval_dover.py imports an external `DOVER` package that is neither vendored in the repo nor listed in requirements.txt — DOVER (Tab 1, all of Tab 2, the Tab 3 'Performance' column, Fig 1) cannot be computed as shipped; README §TODO line 269 lists it unreleased",
  ["r02#2","r03#1","r04#2","r05#1","r06#1","r07#1","r08#1","r09#2"]),
 ("eval_ewarp.py (the only E*warp temporal-consistency script) imports a nonexistent `ewarp` module and chdir's into a missing `finetune/scripts/RAFT` dir (RAFT actually lives at finetune/utils/RAFT) with a wrong default checkpoint path — the E*warp column (every Tab 2 row, Fig 1) crashes on chdir/import; README §TODO confirms it unreleased",
  ["r01#1","r02#3","r03#4","r04#1","r05#4","r06#2","r07#6","r08#5","r09#4","r10#0"]),
 ("inference.sh (the canonical 'reproduce paper results' runner) hardcodes `--gt datasets/test/UDM10/GT` for ALL six eval blocks, so SPMCS/YouHQ40/RealVSR/MVSR4x/VideoLQ (5/6) are scored against UDM10's ground truth; eval_metrics.py matches by filename stem and silently skips mismatches, so the shipped script does not reproduce the Table-2 full-reference (PSNR/SSIM/LPIPS/DISTS) numbers",
  ["r01#3","r02#4","r03#5","r04#3","r05#3","r06#5","r07#4","r08#3","r09#7","r10#3"]),
 ("eval_dover.py per-sample off-by-one: scores assigned with `dover_results[i-1]`, so each clip name receives the PREVIOUS clip's DOVER score (i=0 gets the last). Dataset-mean is a cyclic permutation so headline Tab 2/3 averages are unaffected, but per-clip DOVER output written to metrics_dover.json is mislabeled",
  ["r01#4","r02#5","r03#6","r04#4","r05#5","r06#6","r07#5","r08#4","r09#6","r10#4"]),
 ("No timing/benchmark harness for the headline '~28x faster' efficiency claim / Table 3 running times (e.g. DOVE 14.90s vs MGLD-VSR 425.23s on a 33-frame 720x1280 clip) — inference_script.py never measures or logs latency, so the central efficiency contribution cannot be reproduced",
  ["r03#2","r06#4","r08#2","r10#1"]),
 ("requirements.txt omits core dependencies (pyiqa, diffusers, safetensors) imported by inference_script.py / eval_metrics.py / trainer.py for every IQA metric and the CogVideoX backbone; they appear only as ad-hoc unpinned `pip install` lines in the README — the env is not rebuildable from requirements alone and the version-sensitive diffusers API (scheduler.get_velocity / get_3d_rotary_pos_embed) risks silent drift",
  ["r01#2","r03#3","r05#2","r06#7","r07#3","r09#3","r10#2"]),
 ("Paper §4.1 states the image training set is DIV2K with 900 images, but the README/official link and the DIV2K_train_HR filelist used by the code are the standard 800-image training split (900 = 800 train + 100 val) — minor description mismatch in the Stage-2 image data",
  ["r02#7","r03#7","r05#6","r09#8"]),
 ("Shipped training config (accelerate_config.yaml = 2 GPUs; both train shells set --batch_size 2 / gradient_accumulation_steps 1 -> global batch 4) does not match the paper's '4 NVIDIA A800-80G GPUs, total batch size 8'; the 4-GPU launch topology is not enforced by the scripts, so the default files reproduce batch 4, not 8",
  ["r04#5","r08#6"]),
 ("UDM10 is used as BOTH the every-500-step training-time validation monitor AND the Table-1 ablation / Table-2 benchmark set; checkpoints are saved at fixed step intervals (not selected by val metric) and there is no held-out split distinct from the benchmark, so manual design/iteration could be informed by UDM10 (soft leakage/design concern)",
  ["r06#8","r07#8"]),
 ("inference_script.py removes the SR-output padding as `pad*args.upscale`-should-be but hardcodes `pad*4` regardless of --upscale, so for the --upscale 1 datasets (RealVSR/MVSR4x) with H/W not divisible by 16 the output is over-cropped on the right/bottom edge, shifting the evaluated region and those FR metrics",
  ["r02#6"]),
 ("inference.sh runs RealVSR/MVSR4x with `--upscale 1`, whereas §4.1 states 'all experiments use a scaling factor x4'; likely the correct operational choice for already-target-resolution real-world inputs, but the code and the blanket x4 statement differ",
  ["r09#9"]),
 ("Stage-2 trainer (lora_one_s2_trainer.py / args.py) exposes loss variants the paper never mentions — edge-aware DISTS (Sobel EdgeDetectionModel), LPIPS / edge-aware LPIPS, GAN losses (gen_cls_loss_weight, diffusion_gan_max_timestep), optical-flow flags; unused by the released defaults but undocumented",
  ["r07#7"]),
 ("eval_metrics.py computes PSNR/SSIM on full RGB with no border crop by default; neither inference.sh nor the README example passes --test_y_channel/--crop, whereas many VSR papers report Y-channel PSNR/SSIM — the channel/crop convention is unstated, so reproduced fidelity numbers may differ",
  ["r10#5"]),
 ("eval_vbench.py chdir's into a nonexistent `finetune/scripts/VBench` dir and imports `from evaluate import calculate_final`; the VBench package is neither vendored nor in requirements (VBench is not a Table-2 metric, so headline impact is low)",
  ["r09#5"]),
],
"2371": [
 ("No code for the five generative baselines (TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4) — the Table 1 & 2 comparison rows (and the normalized-average / rank columns built on them) cannot be reproduced",
  ["r01#0","r02#0","r03#0","r04#0","r05#0","r06#0","r07#0","r08#0","r09#0","r10#0"]),
 ("Section 6 / Figure 2 text-conditioning kNN-classifier accuracy (0.81) has no classifier or evaluation code in the released package",
  ["r01#1","r02#1","r03#1","r04#1","r05#2","r06#2","r07#1","r08#1","r09#1","r10#2"]),
 ("ED and DTW pair the i-th real window with the i-th synthetic sample by raw array index (no distributional matching) and only compare min(n) samples, silently discarding surplus generated curves",
  ["r01#3","r02#2","r02#5","r03#6","r04#2","r05#3","r06#6","r07#4","r09#7"]),
 ("Table 2 'WQL' is computed from a single point forecast (~0.5*MAE / un-normalised mean pinball loss), not a true Weighted Quantile Loss over quantile forecasts",
  ["r01#4","r03#5","r05#4","r06#7","r09#5"]),
 ("No driver / aggregation script computes the Table 1 & 2 normalized-average and average-rank columns across the 12-dataset benchmark (and no per-dataset configs are shipped)",
  ["r05#1","r06#1","r08#2","r09#3","r10#1"]),
 ("SHAP-RE (SHR) reshape collapses the channel axis — valid only for univariate input; silently wrong / crashes for multivariate (C>1)",
  ["r02#3","r05#6","r06#3","r10#3"]),
 ("SHAP-RE (SHR) uses an unseeded RNG and is nondeterministic (varies ~13% run-to-run on identical inputs)",
  ["r03#7","r04#3","r07#3"]),
 ("MASE denominator uses the forecast window's own naive error, not the in-sample / training-set (seasonal) naive error of the paper's definition",
  ["r02#6","r06#5","r07#2","r09#6"]),
 ("ED is computed on StandardScaler (zero-mean/unit-variance) data, but Appendix B.2 defines it on [0,1]-scaled inputs",
  ["r01#2","r02#4","r06#4"]),
 ("Config `seed` is not propagated to the SDForger LLM (RNG hardcoded to 42), undermining the paper's '5 random seeds' averaging",
  ["r05#5","r08#5","r09#4"]),
 ("TTM utility evaluation depends on tsfm_public / granite-tsfm, which is absent from the env files and installed from an unpinned main / moving model revision",
  ["r03#2","r08#3","r09#2"]),
 ("TrainingArguments hardcodes bf16=True, contradicting the float32 config and breaking the documented MPS run path",
  ["r03#4","r08#4"]),
 ("TTM RMSE/MASE/WQL are averaged over all forecast channels, not just the stated target channel",
  ["r03#8","r10#5"]),
 ("Table D.5 generation-time comparison has no timing / benchmark script",
  ["r03#3"]),
 ("Shipped univariate config diverges from the paper's reported settings (k=5 and L0=2000 with 1 sample vs the paper's k=3)",
  ["r10#4"]),
],
}


def load_runs(paper):
    runs = {}
    for rd in sorted((RB / paper).glob("run_*")):
        fj = rd / "findings.json"
        if fj.exists():
            runs[int(rd.name.split("_")[1])] = json.loads(fj.read_text()).get("findings", [])
    return runs


def resolve(runs, fid):
    m = FID_RE.match(fid)
    r, i = int(m.group(1)), int(m.group(2))
    items = runs.get(r, [])
    return (r, items[i]) if 0 <= i < len(items) else (r, None)


def agree(vals):
    seen = [v for v in vals if v]
    if not seen:
        return "—"
    uniq = sorted(set(seen), key=lambda v: -SEV_RANK.get(v, 0))
    mode = max(set(seen), key=seen.count)
    return mode if len(uniq) == 1 else f"{mode}  (varied: {', '.join(uniq)})"


def representative(runs, fids):
    best = None
    for fid in fids:
        r, f = resolve(runs, fid)
        if f is None:
            continue
        score = (CONF_RANK.get((f.get("confidence") or "").lower(), 0),
                 1 if (f.get("quote") or "").strip() and (f.get("file") or "").strip() else 0,
                 len(f.get("quote") or ""))
        if best is None or score > best[0]:
            best = (score, r, f, fid)
    return (best[1], best[2], best[3]) if best else (None, None, None)


def loc(f):
    fl = f.get("file", "") or ""
    ls, le = f.get("line_start"), f.get("line_end")
    if ls:
        fl += f":{ls}" + (f"-{le}" if le and le != ls else "")
    return fl


def build(pid, paper):
    runs = load_runs(paper)
    total = len(runs)
    clusters = CLUSTERS[pid]
    cards = []
    for label, fids in clusters:
        det = len({FID_RE.match(x).group(1) for x in fids})
        rep_run, rep, rep_fid = representative(runs, fids)
        if rep is None:
            continue
        sev = agree([(resolve(runs, x)[1] or {}).get("severity", "").lower() for x in fids])
        conf = agree([(resolve(runs, x)[1] or {}).get("confidence", "").lower() for x in fids])
        cat = rep.get("category", "?")
        cards.append((det, SEV_RANK.get((rep.get("severity") or "").lower(), 0),
                      label, fids, det, sev, conf, cat, rep, rep_run, rep_fid))
    cards.sort(key=lambda c: (-c[0], -c[1]))

    out = []
    out.append(f"# Human-eval worksheet — #{pid} · {paper}\n")
    out.append(f"**{len(cards)} distinct defects** (the 10 PDF+text audit runs' findings, merged by defect). "
               f"Detection = how many of the {total} runs surfaced the defect (high = robust; 1 = one run only). "
               f"Severity & confidence are the auditor's own labels (spread shown where runs disagreed); "
               f"the wording/quote is taken from the highest-confidence run that cited code.\n")
    out.append("Tick **one** box per defect (put an `x`):\n")
    out.append("- **correct & relevant** — true *and* a substantive reproducibility issue worth raising")
    out.append("- **correct but wrong severity** — true and worth raising, but the severity label is miscalibrated (e.g. an out-of-the-box crash with a trivial fix tagged high that's really low/medium)")
    out.append("- **correct but not relevant** — technically true but trivial / nitpick / already acknowledged")
    out.append("- **unsure** — can't decide without resources beyond the frozen repo + paper")
    out.append("- **false** — the claim misreads the code/paper and does not hold\n")
    out.append(f"Frozen code: `{paper}/code_frozen/`  ·  paper: `audits/{paper}/paper.pdf`\n")
    out.append("---\n")

    for n, c in enumerate(cards, 1):
        _, _, label, fids, det, sev, conf, cat, rep, rep_run, rep_fid = c
        runs_hit = sorted({f"r{FID_RE.match(x).group(1)}" for x in fids})
        out.append(f"### F{n:02d} · {label}\n")
        out.append(f"_category: {CATLAB.get(cat, cat)} · topic: {rep.get('topic','?')}_\n")
        out.append(f"**severity: {sev}  ·  confidence: {conf}  ·  detection: {det}/{total} runs**\n")
        if rep.get("claim"):
            out.append(f"- **Claim:** {rep['claim'].strip()}")
        if rep.get("concern"):
            out.append(f"- **Concern:** {rep['concern'].strip()}")
        if rep.get("resolution"):
            out.append(f"- **Ask:** {rep['resolution'].strip()}")
        ev = [f"`{loc(rep)}`"] if loc(rep) else []
        if rep.get("paper_ref"):
            ev.append(f"paper: {rep['paper_ref']}")
        if ev:
            out.append("- **Evidence:** " + " · ".join(ev))
        out.append(f"- **Found in runs:** {', '.join(runs_hit)}  (representative: {rep_fid})")
        q = (rep.get("quote") or "").rstrip()
        if q:
            ql = q.split("\n")
            if len(ql) > 18:
                ql = ql[:18] + [f"... (+{len(q.splitlines())-18} more lines)"]
            out.append(f"- **Quoted at `{loc(rep) or rep.get('file','')}`:**")
            out.append("```\n" + "\n".join(ql) + "\n```")
        out.append("")
        out.append("**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   "
                   "correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`\n")
        out.append("**Notes:**\n")
        out.append("---\n")

    fn = RB / f"HUMAN_EVAL_{pid}.md"
    fn.write_text("\n".join(out) + "\n")
    return fn, len(cards), total


def validate():
    ok = True
    for pid, paper in PAPERS.items():
        runs = load_runs(paper)
        per_run = {r: len(v) for r, v in runs.items()}
        assigned = {}
        for label, fids in CLUSTERS[pid]:
            for x in fids:
                assigned[x] = assigned.get(x, 0) + 1
        # every run#idx assigned exactly once
        allfids = {f"r{r:02d}#{i}" for r, n in per_run.items() for i in range(n)}
        got = set(assigned)
        dup = {k: v for k, v in assigned.items() if v > 1}
        missing = allfids - got
        extra = got - allfids
        n_assigned = sum(per_run.values())
        print(f"#{pid}: {sum(per_run.values())} raw findings, {len(CLUSTERS[pid])} defects, "
              f"missing={len(missing)} dup={len(dup)} extra={len(extra)}")
        if missing:
            print("   MISSING:", sorted(missing))
        if dup:
            print("   DUP:", dup)
        if extra:
            print("   EXTRA (bad fid):", sorted(extra))
        ok &= not (missing or dup or extra)
    return ok


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None  # build a single paper, e.g. `make_worksheets.py 2371`
    if not validate():
        raise SystemExit("cluster coverage check FAILED — fix CLUSTERS before rendering")
    for pid, paper in PAPERS.items():
        if only and pid != only:
            continue
        fn, n, total = build(pid, paper)
        print(f"wrote {fn.name}: {n} defects from {total} runs")
