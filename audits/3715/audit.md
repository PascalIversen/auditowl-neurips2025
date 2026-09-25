# Code audit — Diffusion Models and the Manifold Hypothesis: Log-Domain Smoothing is Geometry Adaptive (NeurIPS 2025, #3715)

## 1. Summary

The paper is largely theoretical (Propositions 3.1/3.7/3.8, Theorems 3.6/4.1) with a set of
illustrative empirical figures. The repository `samuel-howard/log_smoothing` contains a small
JAX/Flax codebase: four `src/` modules (`diffusion.py` — variance-exploding diffusion + analytic
empirical score via `logsumexp`; `smoothing.py` — Gaussian and manifold-translated score smoothing;
`vae.py` — convolutional VAE; `plotting.py` — helpers) and six Jupyter notebooks that drive the
figures. MNIST data ships as `.npy` files under `data/`. There is no top-level README beyond a
one-line citation, no dependency-pinned reproduction command list (an `environment.yml` exists), and
`.gitignore` excludes `figures/` and `params/`.

What I did:
- Mapped every numbered figure to the notebook/cell that produces it (traceability table below).
- Read all four `src/` modules and all six notebooks (cell dumps).
- Verified the empirical-score / VE-diffusion math is internally consistent with paper eq. (9)
  (`v(t)=t`, `g(t)=1`, `log p̂_t = LSE(-‖x−x_i‖²/2t)`).
- Ran `_audit_code/check_artifacts.py` to confirm which loaded/saved artefacts are absent
  (output: `_audit_code/out/artifacts.json`).
- Grepped all notebooks for the VAE-checkpoint dependency, the smoothing-sample-count (`n=`)
  settings, and for any code generating Figure 3.

Theory (proofs, assumptions, bounds) is out of scope for a code-repository audit and is not assessed
here except where a figure depends on a computation.

## 2. Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Fig. 1 (lima-bean, isotropic smoothing σ=0.02→0.12) | `low_dim_plots.ipynb` cells 3-5 | yes | qualitative (no numbers) | Verified |
| Fig. 2 (KDE vs score-smoothing, lima-bean) | `low_dim_plots.ipynb` cells 6-7 | yes | qualitative | Verified |
| **Fig. 3 (tailored/aligned kernels → different manifolds; "alter dimension and connectivity")** | (none) | — | — | **MISSING** |
| Fig. 4 left (circle samples) | `circle2d_log_likelihood.ipynb` cell 18 | yes | qualitative | Verified |
| Fig. 4 right (population NLL U-curve, 1000 pts) | `circle2d_log_likelihood.ipynb` cells 16-17 | yes | qualitative; uses T=5.0 not paper's T=9.0 | MISMATCH (param) |
| Fig. 5 (wavy circle, σ=0.1/0.25/1.0) | `low_dim_plots.ipynb` cells 8-10 | yes | qualitative | Verified |
| Fig. 6 (latent MNIST: score vs KDE reconstructions) | `mnist_KDE_vs_gaussian.ipynb` cells 33-38 | yes, but needs VAE checkpoint | qualitative | MISSING dependency |
| Fig. 7 (L2 to data vs L2 to M, latent MNIST) | `mnist_KDE_vs_gaussian.ipynb` cells 28-30 | yes, but needs VAE checkpoint; n=1000 vs paper-stated 50000 | qualitative | MISSING dependency |
| Fig. 8 left (bump: L2 to data vs M) | `bump_compare_smoothing.ipynb` cells 19-33 | yes | qualitative | Verified |
| Fig. 8 right (bump anisotropy) | `bump_compare_smoothing.ipynb` cells 10,35 | yes | qualitative | Verified |
| Fig. 9 left (pixel MNIST: L2 to data vs M) | `mnist_compare_smoothing.ipynb` cells 24-35 | yes, but needs VAE checkpoint | qualitative | MISSING dependency |
| Fig. 9 right (FID vs distance) | `mnist_fid_computations.ipynb` cells 5-9 (reads samples saved by `mnist_compare`) | yes, but needs VAE checkpoint + saved samples | qualitative | MISSING dependency |
| Fig. 10 (per-digit PCA explained variance) | `mnist_KDE_vs_gaussian.ipynb` cell 12 | yes, but needs VAE checkpoint | qualitative | MISSING dependency |
| Fig. 16 (FID for digits 2 and 7) | `mnist_compare_smoothing.ipynb` (digit switch) + `mnist_fid_computations.ipynb` | yes, but needs VAE checkpoint | qualitative | MISSING dependency |

Note: the paper reports no tables and no numeric headline statistics; all figures are qualitative
trend plots (distance/FID/anisotropy vs distance-to-data, sample montages). The traceability
question therefore reduces to "does runnable code produce each figure", which is the basis for the
findings below.

## 3. Findings

### missing

```yaml finding
id: vae-checkpoint-absent
category: missing
topic: "result traceability / pretrained model"
title: "VAE checkpoint required for all MNIST figures (6,7,9,10,16) is gitignored and absent"
severity: high
confidence: high
status: finding
file: mnist_KDE_vs_gaussian.ipynb
line_start: 152
line_end: 154
quote: |
      "# Load pretrained params\n",
      "with open('params/vae_params.pkl', 'rb') as f:\n",
      "    params = pkl.load(f)"
claim: "The MNIST notebooks load a pretrained VAE from params/vae_params.pkl (the training call directly above is commented out); the same load appears in mnist_compare_smoothing.ipynb. The file does not exist in the repo and .gitignore lists `params/`."
concern: "Every MNIST figure (6, 7, 9 latent+pixel, 10 PCA, 16) depends on this checkpoint; without it the notebooks raise FileNotFoundError at the load cell, and the manifold construction in pixel space (decoding a latent triangle) is checkpoint-specific, so the paper's MNIST results cannot be reproduced as shipped."
resolution: "Authors: please commit params/vae_params.pkl (or a download link), or uncomment/clarify the exact training command + seed so the checkpoint can be regenerated identically."
cross_refs: ["readme-no-repro-instructions", "smoothing-n-mismatch"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Appendix G.2 (VAE training) and Figures 6,7,9,10,16"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: figure3-no-code
category: missing
topic: "result traceability"
title: "Figure 3 (tailored kernels selecting different manifolds) has no generating code"
severity: medium
confidence: high
status: finding
file: low_dim_plots.ipynb
line_start: 47
line_end: 47
quote: |
      "## Figure 1"
claim: "low_dim_plots.ipynb contains sections for Figures 1, 2 and 5 only; no notebook contains the construction of the tangent-aligned / level-set-aligned kernels that Figure 3 uses to make the same training data generate samples on different manifolds (wavy-circle vs base circle, and the dimension/connectivity-altering kernels in Figure 3 right)."
concern: "Figure 3 is a headline illustration of the paper's geometric-bias claim ('by choosing a kernel that aligns with certain geometric structures ... the diffusion model is biased to interpolate along the corresponding manifold'), yet no code in the repo produces it, so it is not reproducible or verifiable."
resolution: "Authors: please add the notebook/cells that construct the aligned kernels and generate the Figure 3 panels, or point to where they live."
cross_refs: []
paper_ref: "Figure 3 and Section 2.3 / 4.1"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-no-repro-instructions
category: missing
topic: "expected completeness / reproduction"
title: "README has no setup, data, checkpoint, or run instructions and no results mapping"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  Research code for the paper 'Diffusion Models and the Manifold Hypothesis: Log-Domain Smoothing is Geometry Adaptive' (https://arxiv.org/abs/2510.02305), by Tyler Farghly, Peter Potaptchik, Samuel Howard, George Deligiannidis, and Jakiw Pidstrigach.
claim: "The entire README is a single citation line. There is no statement of which notebook produces which figure, no instruction to train/obtain the VAE checkpoint that figures depend on, and no command to reproduce any result; environment.yml exists but is never referenced."
concern: "A reviewer cannot map figures to code or rebuild the environment and artefacts without reverse-engineering the notebooks, and the missing-checkpoint failure (see vae-checkpoint-absent) is undocumented, undermining the paper's 'open access to code with sufficient instructions' checklist answer."
resolution: "Authors: add a README section listing dependencies (point to environment.yml), how to obtain/train the VAE checkpoint, and a notebook→figure map."
cross_refs: ["vae-checkpoint-absent"]
paper_ref: "NeurIPS checklist Q5 (open access to data and code)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

No technical bugs found. The diffusion sampler, analytic empirical score (`log_hat_pt`/
`empirical_eps_fn`), and the VAE decoder bottleneck arithmetic (`dim_mults=(1,2,2,2)`, 3 downsamples
→ 4×4 spatial, hardcoded `bottleneck_dim=4`) are internally consistent. The `Decoder` docstring
comments mention an "8-dimensional latent space" while the config uses `latent_dim=32`, but the code
reads `latent_dim` from config, so this is a stale comment, not a defect.

### difference

```yaml finding
id: smoothing-n-mismatch
category: difference
topic: "evaluation fidelity (paper vs code)"
title: "Latent-MNIST figures shipped with n=1000 smoothing samples; comment says paper used 50000"
severity: low
confidence: high
status: finding
file: mnist_KDE_vs_gaussian.ipynb
line_start: 671
line_end: 671
quote: |
      "smoothed_emp_eps_fn = get_smoothed_fn(data_empirical_eps_fn, smoothing_param, n=1000) #n=50000)     # increase n for better smoothing accuracy\n",
claim: "In mnist_KDE_vs_gaussian.ipynb the Gaussian-smoothed score is estimated with n=1000 Monte-Carlo kernel samples in every plotting cell, with an inline `#n=50000` and 'increase n for better smoothing accuracy' note; the paper (App. G.2) and the bump notebook use 50000. The committed code thus produces a coarser smoothing estimate than the figures it claims to reproduce."
concern: "As shipped, re-running the notebook does not reproduce the paper's Figure 6/7 smoothing quality; the score estimate is a 50x coarser Monte-Carlo approximation, which can shift the L2-to-manifold curve."
resolution: "Authors: set n to the value used for the published figures (or document that n=1000 was used) so the committed code reproduces the figures."
cross_refs: ["vae-checkpoint-absent"]
paper_ref: "Appendix G.2 ('we use 1000 smoothing samples at each generation step')"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: circle-nll-T-mismatch
category: difference
topic: "hyperparameters (paper vs code)"
title: "Figure 4 circle NLL uses T=5.0 in code vs T=9.0 stated in Appendix G.1"
severity: low
confidence: medium
status: finding
file: circle2d_log_likelihood.ipynb
line_start: 325
line_end: 326
quote: |
      "sigma_max = jnp.sqrt(5.0)\n",
      "ve = VE_diffuser(sigma_min=sigma_min, sigma_max=sigma_max)\n",
claim: "The VE diffuser active during the Figure 4 NLL plotting loop is the one defined with sigma_max=sqrt(5.0), giving T = sigma_max**2 = 5.0 (VE_diffuser.__init__ sets self.T = sigma_max**2). Appendix G.1 states the circle experiment uses T = 9. All other notebooks use sigma_max=3.0 → T=9, consistent with the paper."
concern: "The published Figure 4 endpoint time differs from the value reported in Appendix G.1, so the committed notebook does not reproduce that figure under the documented setting; the discrepancy is benign for the qualitative U-shape but affects exact values."
resolution: "Authors: confirm whether the Figure 4 NLL plot used T=5 or T=9, and align the notebook with Appendix G.1."
cross_refs: []
check_script: _audit_code/check_T_settings.py
paper_ref: "Appendix G.1 ('a variance exploding diffusion model with T = 9')"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

No methodology findings rise to the level of a defect. Notes that I checked and judged sound or
disclosed:

- **Manifold-adapted kernel uses ground-truth manifold.** The "Adapted" curves in Figures 8/9 smooth
  along the true manifold (`interpolants`) via `get_manifold_translated_smoothed_fn`. This is an
  oracle, but the paper explicitly discloses it ("the sampler 'knows' the manifold only via the
  smoothing mechanism; the empirical score itself uses only the training dataset", §5.2) and presents
  it as a theoretical/illustrative comparison, not a deployable method. Not a hidden flaw.
- **Small sample counts (N=100 generations; no error bars / single seed).** Figures 7/8/9 average
  over 100–500 generations at a single PRNG seed with no variance bands. Because the paper's claims
  are qualitative (curve ordering, trend direction) rather than a quantitative win at a threshold,
  this is a robustness caveat rather than a conclusion-changing defect; recorded here, not filed.
- **No held-out test set in the ML sense.** N/A — the paper studies smoothing of an empirical score
  on synthetic/controlled manifolds; there is no train/test generalisation claim of the predictive
  kind, so split/leakage checks are structurally inapplicable.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | VAE checkpoint for all MNIST figures absent; Figure 3 has no code; README has no repro instructions |
| bug         | 0          | -            | sampler / score / VAE arithmetic internally consistent |
| difference  | 2          | low          | latent-MNIST n=1000 vs paper 50000; Figure 4 NLL uses T=5 vs stated T=9 |
| methodology | 0          | -            | oracle-manifold kernel is disclosed; claims are qualitative |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] VAE checkpoint absent** (`vae-checkpoint-absent`, high/high): `params/vae_params.pkl`
   is gitignored and not present, yet every MNIST figure (6, 7, 9, 10, 16) loads it; the notebooks
   fail at the load cell as shipped.
2. **[missing] Figure 3 has no code** (`figure3-no-code`, medium/high): the headline geometric-bias
   figure (tailored kernels selecting different manifolds) is produced by no notebook.
3. **[missing] README has no reproduction instructions** (`readme-no-repro-instructions`,
   medium/high): one-line README, no environment/checkpoint/run guidance, no figure map.
4. **[difference] Latent-MNIST n=1000 vs paper 50000** (`smoothing-n-mismatch`, low/high): committed
   smoothing-sample count is 50x coarser than the figures it reproduces.
5. **[difference] Figure 4 NLL T=5 vs stated T=9** (`circle-nll-T-mismatch`, low/medium).

### Items that genuinely look fine
- The VE diffusion + analytic empirical score (`src/diffusion.py`) matches paper eq. (9):
  `log p̂_t = LSE(-‖x−x_i‖²/2t)` with `v(t)=t`, `g(t)=1`; the reverse sampler and last-step noise mask
  are coherent.
- `src/smoothing.py` Gaussian smoothing (`get_smoothed_fn`) correctly Monte-Carlo-averages the score
  over `x + σ·z` offsets, matching the convolution `k * ∇log p̂_t` in eq. (6).
- VAE decoder shape arithmetic is consistent (`dim_mults=(1,2,2,2)` → 4×4 bottleneck) despite a stale
  "8-dimensional" docstring comment.
- Low-dimensional Figures 1, 2, 5 are fully reproducible from `low_dim_plots.ipynb` with bundled code
  (no external checkpoint).
- MNIST data is bundled (`data/*.npy`, standard 60k/10k split) — no missing-data issue there.

### Open questions for the authors
- Was the Figure 4 NLL plot generated with T=5 (as the committed code does) or T=9 (Appendix G.1)?
- Which `n` (smoothing samples) produced the published Figures 6/7 — 1000 (committed) or 50000
  (comment)?
- Can the VAE checkpoint be released or its training command/seed pinned so the MNIST manifolds and
  figures are reproducible?
