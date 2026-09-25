# Code audit — Mitigating Instability in High Residual Adaptive Sampling for PINNs via Langevin Dynamics (NeurIPS 2025, #3680)

## 1. Summary

The repository (`neurips2025-las/LAS-implementation`, commit `db29b90`) is a collection
of ten self-contained Jupyter notebooks, one per PDE benchmark
(`LAS-mains/<PDE>/code/PINN_training.ipynb`): Burgers1D, Allen-Cahn1D, KdV1D,
Schrödinger1D, Convection1D, Burgers2D, Heat2D, and DF-Heat 4D/6D/8D. Each notebook
defines a `PINN` class plus five samplers (`RADSampler`, `R3Sampler`, `L_INFSampler`,
`LASSampler`, and inline `random-r`/`fixed`), and a `__main__` block that trains the
chosen sampler (`--method`) five times (`repeat = [0,1,2,3,4]`) and dumps per-run loss
and relative-L2 arrays to `../models/`. The proposed method (LAS) is implemented as a
Langevin update over a persistent collocation population (`LASSampler.update`).

What I did: extracted and read all ten notebooks; compared the implemented samplers and
hyperparameters against the paper's §5.1-5.5 and Table 1; checked for seeding,
dependency specs, aggregation/plotting/reporting code, and data-file presence. I wrote
three read-only checks under `_audit_code/` (seed grep across all notebooks; file/format
inventory was done inline via `find`). The four PDEs that need reference data
(`AC.mat`, `burgers_shock.mat`, `KDV.npz`, `NLS.mat`) ship their data; the rest use
closed-form analytic solutions defined in-code, so data availability is complete.

Headline picture: the code that *trains* the models and *computes* the relative-L2 error
is present and looks methodologically reasonable (no test-set leakage into model
selection — the saved checkpoint is chosen on training loss and is never reloaded for
evaluation; the reported metric is computed against the held-out analytic/reference
solution). The main gaps are reproducibility-side: (i) no RNG is ever seeded although the
paper repeatedly claims "five random seeds"; (ii) there is no code that aggregates the
per-run arrays into the Table 1 mean±std values, no figure-generating code, and no code
for the Appendix G/H/J baseline loss-balance search, architecture, and tuning
experiments; (iii) no dependency specification exists.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1: relative L2 (mean ± std) per PDE × method × {8,10} layers | per-run `rel_l2` arrays saved by each `PINN.Train` (e.g. Allen-Cahn notebook `Train`); **no script computes mean±std** | per-run array only | — | MISSING (aggregation absent) |
| Table 1: 8/10-layer sweep | `--layers` CLI arg (default 8); 10 must be passed manually; no driver loops layers | n/a | — | PARTIAL (runnable, no harness) |
| Fig. 3 (max Hessian eigenvalue / steepness; rel-L2 curve, 4 layers) | (none) — no Hessian-eigenvalue or plotting code in repo | — | — | MISSING |
| Fig. 4 (Allen-Cahn boxplots over layers / learning rates, 5 seeds) | (none) — no boxplot/aggregation code | — | — | MISSING |
| Fig. 5 / Appendix I (computation cost vs Npde) | (none) — `time` imported but no timing-collection or plotting code | — | — | MISSING |
| §5.4 / Appendix G: "extensive search for optimal loss balance terms and hyperparameter configurations of the baselines" | (none) — loss weights are hardcoded per-PDE; no search code | — | — | MISSING (no search code) |
| Appendix H: robustness across model architectures | (none) | — | — | MISSING |
| Appendix J: LAS tuning (lL, re-initialization) results | LAS `L_iter`/re-init commented out in `LASSampler.update`; no tuning sweep code | — | — | MISSING |
| §5.1 protocol: η=0.001, StepLR γ=0.9 every 5000, 200k iters, Npde=1000 | `Train`: `optim.Adam(lr=1e-3)`, `StepLR(step_size=5000, gamma=0.9)`; `--epochs` default 200000; `--Nf` default 1000 | matches | ✓ | Verified |
| §5.2 LAS defaults: τ=0.002 (1-2D) / 0.01 (4-8D), β=0.2, lL=1 | `self.las = LASSampler(..., L_iter=1, beta=0.2, tau=…)` across all 10 notebooks | matches | ✓ | Verified |
| "five random seeds" (§5.1, Fig. 4 caption, §5.3) | `repeat = [0,1,2,3,4]` loop, **no seed set anywhere** | unseeded repeats | ✗ (no seeds exist) | MISMATCH |

## 3. Findings

## missing

```yaml finding
id: no-result-aggregation-or-figures
category: missing
topic: "result traceability"
title: "No code aggregates per-run arrays into Table 1 / figures or runs Appendix G/H/J experiments"
severity: high
confidence: high
status: finding
file: LAS-mains/Allen-Cahn-equation1D/code/PINN_training.ipynb
line_start: 1
line_end: 1
quote: |
  import matplotlib.pyplot as plt
claim: "Each notebook's `__main__` only `torch.save`s per-run loss and relative-L2 arrays to ../models/ (e.g. `.rel_l2_<method>_<layers>_<Nf>_<i>`); the repo contains no script that reads these arrays and computes the Table 1 mean±std values, produces any figure (Fig. 3 Hessian-eigenvalue/steepness, Fig. 4 boxplots, Fig. 5 cost), or runs the Appendix G loss-balance search, Appendix H architecture, or Appendix J tuning experiments."
concern: "None of the paper's quantitative artefacts (Table 1 numbers, every figure, the baseline loss-balance search, the architecture and tuning studies) can be reproduced from the repo without re-implementing the entire aggregation/selection/plotting pipeline, and the rule used to reduce each per-run array to a single reported number (final epoch vs. minimum over training) is not in the code."
resolution: "Provide the aggregation/plotting scripts that turn the saved per-run arrays into Table 1 and Figures 3-5, and the driver code for the Appendix G/H/J experiments; state explicitly whether the reported relative L2 is the final-epoch value or the minimum over the recorded curve."
cross_refs: ["no-rng-seed-control"]
check_script: _audit_code/check_seeds.py
paper_ref: "Table 1; Figures 3-5; Appendix G, H, I, J"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-specification
category: missing
topic: "environment / dependencies"
title: "No requirements.txt / environment.yml / pinned versions"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 1
quote: |
  # Mitigating Instability in High Residual Adaptive Sampling for PINNs via Langevin Dynamics [Neurips 2025 Spotlight]
claim: "The repo has no dependency specification of any kind (no requirements.txt, environment.yml, setup.py, or pyproject.toml anywhere outside .git); the only stated versions are implicit imports of torch, numpy, scipy, matplotlib."
concern: "Without pinned dependency versions the environment cannot be reconstructed, and PINN training results are sensitive to PyTorch/CUDA numerical behaviour, which matters given the paper's central claims are about training stability."
resolution: "Add a requirements.txt or environment.yml pinning at least torch, numpy, scipy versions used for the reported runs."
cross_refs: []
paper_ref: "Reproducibility checklist (paper_text.txt:1157-1201)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: no-rng-seed-control
category: bug
topic: "reproducibility / seeding"
title: "No RNG seeded despite paper's repeated 'five random seeds' claim"
severity: medium
confidence: high
status: finding
file: LAS-mains/Allen-Cahn-equation1D/code/PINN_training.ipynb
line_start: 455
line_end: 456
quote: |
        repeat = [0, 1, 2, 3, 4]
        for i in repeat:
claim: "Every one of the ten notebooks runs five repetitions via `repeat = [0,1,2,3,4]` but never calls `torch.manual_seed`, `np.random.seed`, `random.seed`, or `torch.cuda.manual_seed` (verified across all 10 notebooks: 0 seed calls each, see _audit_code/out/seeds.txt); the loop index `i` is used only to name output files, not to seed anything."
concern: "The paper repeatedly states results are over 'five random seeds' (§5.1, Fig. 4 caption, §5.3), but no seeds exist: the five runs differ only by uncontrolled nondeterminism (weight init, collocation sampling, GPU nondeterminism), so the runs are not reproducible and 'seed' is a misnomer for what the code does."
resolution: "Either seed each of the five runs deterministically (e.g. `torch.manual_seed(i)`, `np.random.seed(i)`) and document it, or correct the paper to say 'five independent runs' rather than 'five random seeds'."
cross_refs: ["no-result-aggregation-or-figures"]
check_script: _audit_code/check_seeds.py
paper_ref: "paper_text.txt:570, 658, 677 ('five random seeds')"
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: las-k-exponent-unused
category: difference
topic: "method hyperparameters"
title: "Paper states LAS uses residual exponent k=2; LAS sampler normalizes the raw gradient and ignores k"
severity: low
confidence: medium
status: finding
file: LAS-mains/Allen-Cahn-equation1D/code/PINN_training.ipynb
line_start: 99
line_end: 104
quote: |
        for t in range(1, self.L_iter + 1):
            grad = phy_lf(model, samples, self.device)
            scaler = torch.sqrt(torch.sum((grad+1e-16)**2, axis = 1)).reshape(-1,1)
            grad = grad/scaler
            with torch.no_grad():
                samples = samples + self.tau * grad + self.beta*torch.sqrt(torch.tensor(2 * self.tau, device=self.device)) * torch.randn(samples.shape, device=self.device)
claim: "`LASSampler.update` drives the Langevin step with the gradient of the (squared) residual produced by `cal_domain_grad`, then L2-normalizes it (`grad/scaler`); it never applies the residual exponent `k`. The paper (§5.2) lists 'the residual exponent k = 2' as part of the LAS configuration."
concern: "The k value reported as part of LAS's configuration has no effect in the LAS code path (it is consumed only by RADSampler), so the paper's hyperparameter description does not correspond to what LAS actually computes; the implemented LAS update (gradient of squared residual, unit-normalized) is itself a valid sampling rule."
resolution: "Clarify in the paper whether k=2 refers to the residual definition used to form the LAS energy/gradient, or remove it from the LAS configuration; confirm the unit-normalized squared-residual gradient is the intended LAS update."
cross_refs: []
paper_ref: "paper_text.txt:560-562 ('the residual exponent k = 2 ... for the LAS configuration')"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding. The evaluation procedure the code implements is sound: the
relative-L2 metric is computed against the analytic/reference solution on a fixed
evaluation grid that is independent of the collocation points; the only model checkpoint
saved is selected on *training* loss and is never reloaded to produce the reported test
metric, so there is no test-set model selection. Loss weights are identical across all
sampling methods within each PDE (the method is a CLI flag, weights are a fixed
per-PDE constant), so the default-setting comparison in Table 1 is symmetric. (The
separate Appendix G "tuned-baseline" experiments are not in the repo — routed to the
`no-result-aggregation-or-figures` MISSING finding, not graded here.)

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                  |
|-------------|------------|--------------|------------------------------------------------------------------|
| missing     | 2          | high         | No aggregation/figure/Appendix-experiment code; no dependency spec |
| bug         | 1          | medium       | No RNG ever seeded despite "five random seeds" claim             |
| difference  | 1          | low          | Paper lists LAS k=2 but LAS code ignores k                       |
| methodology | 0          | -            | Eval procedure checked and looks sound (no test-set selection)  |

## 5. Closing lists

**Top take-aways**
- `[missing]` No code aggregates the saved per-run arrays into Table 1 mean±std, produces any figure, or runs the Appendix G/H/J baseline-tuning / architecture / LAS-tuning experiments — the headline quantitative artefacts are not reproducible from the repo (`no-result-aggregation-or-figures`, high/high).
- `[bug]` No RNG is seeded in any of the ten notebooks although "five random seeds" is claimed repeatedly; the five runs are uncontrolled nondeterminism (`no-rng-seed-control`, medium/high).
- `[missing]` No dependency specification (requirements/environment) exists (`no-dependency-specification`, medium/high).
- `[difference]` Paper lists residual exponent k=2 as part of the LAS configuration, but the LAS sampler normalizes the raw squared-residual gradient and never uses k (`las-k-exponent-unused`, low/medium).

**Items that genuinely look fine**
- LAS step-size τ (0.002 for 1-2D, 0.01 for 4-8D), β=0.2, lL=1 in the code match §5.2.
- Optimizer/scheduler/iteration protocol (Adam lr=1e-3, StepLR γ=0.9 every 5000, 200k iters, Npde=1000) matches §5.1.
- Reference data for the four data-driven PDEs (`AC.mat`, `burgers_shock.mat`, `KDV.npz`, `NLS.mat`) is present; the remaining PDEs use in-code analytic solutions.
- Relative-L2 is computed against the held-out reference solution; the saved checkpoint is chosen on training loss and never reloaded for evaluation, so there is no test-set model selection.
- Loss weights are identical across all sampling methods within each PDE notebook, so the Table 1 default-setting comparison is symmetric.

**Open questions for the authors**
- Is each reported Table 1 relative-L2 the final-epoch value or the minimum over the recorded learning curve? (Not determinable from the repo; selection rule lives only in absent aggregation code.)
- Where is the Appendix G/H/I/J code (baseline loss-balance search, architecture robustness, computation-cost timing, LAS tuning)? It is referenced in the paper but absent from the repo.
