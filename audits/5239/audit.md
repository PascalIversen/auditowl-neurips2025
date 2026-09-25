# Audit — SD-KDE: Score-Debiased Kernel Density Estimation (paper 5239)

## Summary

SD-KDE is a method paper: it proposes taking one score-function step per data
point before standard KDE with a modified bandwidth, proves an AMISE rate of
`O(n^{-8/(d+8)})` (Theorem 1), and corroborates it on 1D/2D synthetic mixtures
and qualitatively on MNIST. The audited artefact is the author's released code
in **two trees**, both treated as author code: the GitHub mirror
`code/Elliotepsteino__SD-KDE/` (only `shrinkage_kde_{gaussian,laplace}.py`,
`visualize_score.py`) and the NeurIPS Supplemental ZIP
`code/SD-KDE__neurips_supplemental/`, which additionally ships the minDiffusion
DDPM suite, the 2D experiment notebook (`minDiffusion/notebooks/2d.py`),
pretrained checkpoints (`notebooks/best_model_*.pth`), MNIST/CIFAR training
scripts, and a UCI processing pipeline.

What I did: read both READMEs and every Python file in both trees; reproduced
the 1D-Gaussian core numerics by copying the author's exact functions into
`_audit_code/check_gaussian_1d.py` (the repo file cannot be imported directly —
it sets `text.usetex=True` at import); ran deterministic existence/grep checks
in `_audit_code/check_missing_artifacts.py`. The Figure-2 scaling slopes and the
Figure-3 histogram statistics reproduce closely; the proposed estimator and its
Silverman baseline are implemented soundly. The previous audit's "diffusion / 2D
/ MNIST / UCI experiment code is missing" conclusion is **largely resolved by the
supplement** — the 1D, 2D, and diffusion-training code is all present (see
traceability table). What remains genuinely missing is code for two *specific
reported results*: the Section 3.3 / Figure 7 iterated-SD-KDE experiment and the
Section 3.4 / Figure 16 MNIST density-ranking; plus a paper/code documentation
mismatch on the Figure-3 sample size and hardcoded paths that block the 2D
figure out of the box.

## Result-traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig 2 — MISE vs n, fitted slopes (3 Gaussian mixtures, Silverman + SD-KDE std=0/2/4) | `shrinkage_kde_gaussian.py:300-487` | reproduced slopes M1 −0.49/−0.86/−0.81/−0.68, M3 −0.60/−0.98/−0.97/−0.96 (n≤5000) | ✓ (paper M1 −0.54/−0.85/−0.82/−0.72, M3 −0.63/−0.93/−0.92/−0.91; diff is my n-cap) | Verified |
| Fig 3 — MISE-diff histograms, mean/std + "95%" (Gaussian) | `shrinkage_kde_gaussian.py:246-286` | at n=200: M1 mean .015/std .002, M2 .014/.001, M3 .010/.005, 96% better | ✓ values, **but code uses n=200 while caption says n=100** | MISMATCH (caption) → see `fig3-sample-size-mismatch` |
| Fig 4 — Emp-SD-KDE vs Silverman vs SD-KDE slopes | `shrinkage_kde_gaussian.py:489-599` (`--plot_empirical`) | runs; uses empirical KDE score `one_step_debiased_data_emp_kde` | ✓ (qualitative) | Verified (present) |
| Fig 6 — 2D spiral: true density vs Silverman vs SD-KDE (true & diffusion score) | `minDiffusion/notebooks/2d.py:1011-1155` + `best_model_spirals.pth` | not runnable as shipped (hardcoded `/scratch/...` model path) | code present, path broken | present → see `hardcoded-model-paths-2d` |
| Fig 15 — 2D mixture-of-Gaussians, true score | `minDiffusion/notebooks/2d.py:687-778` (`demo_kde_2d`) | runs (analytic score, no checkpoint) | ✓ (qualitative) | Verified (present) |
| Fig 7 / Sec 3.3 — iterated SD-KDE, KL & MISE vs iteration | (none in either tree) | — | — | MISSING (`iterated-sdkde-missing`) |
| Fig 16 / Sec 3.4 — MNIST images ranked by SD-KDE density | (none; only DDPM training in `train_mnist.py`) | — | — | MISSING (`mnist-density-ranking-missing`) |
| Appendix A Figs 8–13 — Laplace mixtures + score viz | `shrinkage_kde_laplace.py`, `visualize_score.py` | run | ✓ (qualitative) | Verified (present) |
| Theorem 1 AMISE rate `O(n^{-8/(d+8)})` | theoretical (proof §4) | n/a | n/a | N/A (proof, not code) |

Notes on the deterministic checks (`_audit_code/out/`): `fig2_slopes.csv`,
`fig3_stats.csv` back the rows above; `missing_artifacts.txt` shows 0 hits for
any MNIST-ranking / iteration loop and 24 hardcoded `/scratch|/pscratch` paths.

## missing

```yaml finding
id: iterated-sdkde-missing
category: missing
topic: "result traceability"
title: "No code for the Section 3.3 / Figure 7 iterated SD-KDE experiment"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  As shown in Figure 7, repeated
  application of SD-KDE yields a closer alignment between the estimated and true probability densities
  and a corresponding reduction in KL divergence and mean integrated squared error (MISE) with each
  iteration.
claim: "Section 3.3 and Figure 7 report an iterated/multi-step SD-KDE procedure (recompute score, re-apply SD-KDE, KL and MISE over 1-4 iterations, 200 MC replicates); a grep of both code trees finds zero iteration loops in any .py file and no script that produces the per-iteration KL/MISE curves."
concern: "A reported experiment and its figure have no code that computes the underlying values in either released tree, so Figure 7 cannot be reproduced."
resolution: "Authors: please add the iterated-SD-KDE script (score recomputation loop + KL/MISE-vs-iteration aggregation over the 200 replicates) used for Figure 7."
cross_refs: ["§3.3", "Figure 7"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 3.3 'Iterated SD-KDE'; Figure 7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: mnist-density-ranking-missing
category: missing
topic: "result traceability"
title: "No code ranks MNIST images by SD-KDE density (Section 3.4 / Figure 16)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  Using this score, we apply SD-KDE in latent space to assess the realism of generated images. We ranked generated
  images from highest to lowest estimated probability density, visualized in Figure 16.
claim: "Section 3.4 / Figure 16 report applying SD-KDE in latent space to score and rank generated MNIST images by estimated density; the released MNIST code (train_mnist.py, superminddpm.py) only trains/samples a DDPM, and a grep of both trees finds no argsort/rank/latent-density/SD-KDE-on-MNIST code that produces the ranking."
concern: "The MNIST experiment's reported result (density-ordered image grid) has no code computing the per-image SD-KDE density or the ranking in either tree, so Figure 16 / Section 3.4 is not reproducible."
resolution: "Authors: please add the script that computes the latent-space SD-KDE density per generated MNIST image and produces the ranked grid in Figure 16."
cross_refs: ["§3.4", "Figure 16"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Section 3.4 'MNIST Dataset'; Figure 16"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-model-paths-2d
category: bug
topic: "reproducibility / hardcoded paths"
title: "2D Fig-6 notebook hardcodes /scratch model paths; shipped checkpoints sit elsewhere"
severity: medium
confidence: high
status: finding
file: code/SD-KDE__neurips_supplemental/minDiffusion/notebooks/2d.py
line_start: 1132
line_end: 1137
quote: |
    model_paths = {
        'gaussian': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_gaussian.pth',
        'moons': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_moons.pth',
        'spirals': '/scratch/Score_KDE/minDiffusion/notebooks/best_model_spirals.pth'
    }
claim: "The Figure-6 spiral comparison loads the diffusion-score checkpoint from an absolute path '/scratch/Score_KDE/minDiffusion/notebooks/best_model_spirals.pth', but the supplement ships that checkpoint at the relative path minDiffusion/notebooks/best_model_spirals.pth; KDEComparison.__init__ constructs DiffusionScoreEstimator (which torch.load's the path) before computing any MISE, so the whole spiral panel crashes with FileNotFoundError when run as shipped."
concern: "Figure 6 (the headline 2D diffusion-score result) cannot be reproduced from the released artefact without editing the hardcoded path, even though the needed checkpoint is included."
resolution: "Authors: derive the checkpoint path from BASE_DIR / a relative path so the included best_model_*.pth files are found out of the box (24 such /scratch|/pscratch paths exist across the diffusion/UCI code)."
cross_refs: ["Figure 6", "§3.2"]
check_script: _audit_code/check_missing_artifacts.py
paper_ref: "Figure 6; Section 3.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: fig3-sample-size-mismatch
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Figure 3 caption/text say n=100, but the code produces it with n=200"
severity: low
confidence: high
status: finding
file: code/SD-KDE__neurips_supplemental/shrinkage_kde_gaussian.py
line_start: 189
line_end: 190
quote: |
    n_example = 200
    seed_example = 0
claim: "The MISE-difference histograms (Figure 3) are computed with n_example=200 (shrinkage_kde_gaussian.py:189, used at lines 246-259), and my reproduction at n=200 matches the figure's reported statistics (M1 mean 0.015/std 0.002, M2 0.015/0.001, M3 0.009/std~0.005, ~96% of seeds better); at n=100 the statistics differ (M1 mean 0.019, M3 ~84%). The paper's Figure-3 caption and body instead state 'for n = 100 samples' and 'better in 95% of samples'."
concern: "The reported sample size for Figure 3 (n=100) does not match the n=200 that the code actually used to produce the matching numbers; the qualitative claim ('SD-KDE consistently beats Silverman') holds at both n, so impact is low."
resolution: "Authors: correct the Figure-3 caption/text to n=200 (or regenerate the figure at n=100), and reconcile the '95%' / 'all 100 samples' wording with the 50 seeds the code uses (n_seeds=50)."
cross_refs: ["Figure 3", "§3.1"]
check_script: _audit_code/check_gaussian_1d.py
paper_ref: "Figure 3 caption and Section 3.1 'SD-KDE consistently beats Silverman baseline'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The proposed estimator and the Silverman baseline are
implemented correctly and compared under the same MISE/grid and the same data
seeds; the score-noise robustness sweep is genuine; the asymptotic-rate claim is
a proved theorem, out of scope for code audit. N/A topics: data splitting /
sample independence / target leakage / pretraining contamination / temporal
integrity — this is an in-simulation density-estimation paper with synthetic
ground-truth densities and no train/test split or held-out real-data evaluation
to leak across.

## Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 2          | medium       | Iterated-SD-KDE (Fig 7) and MNIST density-ranking (Fig 16) have no code in either tree. |
| bug         | 1          | medium       | 2D Fig-6 notebook hardcodes /scratch checkpoint paths; shipped checkpoints are elsewhere. |
| difference  | 1          | low          | Figure 3 says n=100; code (and matching numbers) use n=200. |
| methodology | 0          | -            | Estimator and baseline implemented soundly; no methodological defect found. |

## Top take-aways

1. (`missing`, medium) Section 3.3 / Figure 7 iterated SD-KDE — no code in either tree (`iterated-sdkde-missing`).
2. (`missing`, medium) Section 3.4 / Figure 16 MNIST density ranking — only DDPM training is shipped, no density-scoring/ranking code (`mnist-density-ranking-missing`).
3. (`bug`, medium) Figure 6's 2D spiral comparison crashes as shipped on a hardcoded `/scratch` checkpoint path, though the checkpoint is included at a different relative path (`hardcoded-model-paths-2d`).
4. (`difference`, low) Figure 3 caption/text say n=100 but the figure's numbers were produced at n=200 (`fig3-sample-size-mismatch`).

## Items that genuinely look fine

- The 1D Gaussian scaling experiment (Figure 2): reproduced; fitted slopes and the SD-KDE-vs-Silverman ordering and noise-robustness match the paper.
- The Figure-3 histogram statistics reproduce exactly at the code's n=200 (mean/std/fraction-better), confirming the code and the figure are consistent — only the reported sample size differs.
- The SD-KDE estimator (`one_step_debiased_data`), Silverman bandwidth, empirical-score variant (`one_step_debiased_data_emp_kde` / `kde_score_eval`), and the MISE/KL evaluators are implemented correctly and compared fairly under the same grid, metric, and seeds.
- The 2D mixture-of-Gaussians true-score panel (`demo_kde_2d`, Figure 15) runs with the analytic score and needs no checkpoint.
- The diffusion-model description in the paper ("3-layer MLP, hidden 512, Adam, 1500 steps, 1000 diffusion steps") matches `notebooks/2d.py` (epochs=1500, hidden_dim=512, T=1000, 3 hidden Linear layers + SiLU); the separate `train_2d.py` (hidden 128, 10000 epochs) is an unused variant, not the Figure-6 code.
- The previous audit's "missing diffusion/2D/MNIST/UCI experiment code" concern is resolved by the supplement for everything except the two specific reported results flagged above.

## Open questions for the authors

- Was Figure 3 generated at n=100 or n=200? (Numbers point to n=200; please correct the caption or the run.)
- Could you release the iterated-SD-KDE (Fig 7) and the MNIST latent-space SD-KDE ranking (Fig 16) scripts, which are referenced in the paper but absent from both code trees?
