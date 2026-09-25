# Audit — Paper 1867: *A Differential and Pointwise Control Approach to Reinforcement Learning*

## Summary

The paper proposes Differential RL / Differential Policy Optimization (dfPO) and
reports empirical superiority over 12 RL baselines on three scientific-computing
tasks (surface modeling, grid control, molecular dynamics) in Table 1 / Table 4,
with t-test significance claims in Appendix C.2.

**Correction to the audit framing.** The task brief stated the authors released
no code. That is not true for this paper: the paper itself gives an
implementation link (`paper.pdf`, §4.2: "The complete codebase is available at
https://github.com/mpnguyen2/dfPO"), and the NeurIPS checklist answers Q4/Q5
"Yes" citing that codebase and trained models. The repository
`mpnguyen2/dfPO` **does resolve and is public** (verified via the GitHub API;
latest commit `9dc65e7`, master, pushed 2025-10-23). It contains `train.py`,
`test.py`, `benchmarks_run.py`, `policy.py`, `memory.py`, `analysis.ipynb`,
`requirements.txt`, `envs/`, `benchmarks/`, and `output/`. The metadata
`[no-author-signal]` tag was a fetch-heuristic miss; the repo was simply not
cloned into the workspace. I therefore did NOT file spurious "no code released"
findings. Instead I audited the live repo via the GitHub raw/contents API
(evidence cited as URLs, permitted by the schema) and via one reproduction
script under `_audit_code/`.

What I ran / inspected:
- Fetched and read `README.md`, `requirements.txt`, `analysis.ipynb`,
  `benchmarks_run.py`, `test.py`, and `output/benchmarks_stat_analysis.csv` from
  the live repo.
- `_audit_code/check_ttest_synthetic.py` — reproduces `analysis.ipynb` cell 4 to
  confirm the reported t-test p-values are computed on synthetic Gaussian draws,
  not on the experimental seed-level means. Output in
  `_audit_code/out/ttest_synthetic.csv`.
- Verified Table 1 / Table 4 numbers against `output/benchmarks_stat_analysis.csv`.
- Checked repo provenance (commits, tags) via `gh api`.

The headline empirical means (Table 1 / Table 4) **do** trace to repo artifacts
that match the paper. The serious issue is the **statistical-significance
claim**: the t-tests the paper says were run "on the seed-level means" are in
fact computed on `np.random.normal` re-samples of the reported summary
mean/std. The repo computes real per-seed means in `benchmarks_run.py` but never
persists them; the notebook fabricates samples from the summaries instead.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 / Table 4 — all 13 algos × 3 tasks final mean costs (e.g. dfPO Surface 6.296, Grid 6.046, Mol 53.352) | `output/benchmarks_stat_analysis.csv` (produced by `benchmarks_run.py:106`) | identical to CSV | ✓ | Verified |
| Per-seed (10-seed) final costs that the mean±std summarize | `benchmarks_run.py:82-92` computes `avg_final_vals` per seed but only writes summary; raw per-seed values not persisted in repo | — | — | PARTIAL (raw seed values not saved) |
| Appendix C.2 t-test p-values ("t-tests on the seed-level means", significant vs all baselines except Surface/CrossQ) | `analysis.ipynb` cell 4 | p computed on `np.random.normal` draws, NOT seed means | ✗ | MISMATCH (see `ttest-on-synthetic-data`) |
| Table 1 / Table 2 / Table 5 ablation costs | `benchmarks_run.py` + `params.csv` (hyperparameter rows present) | not independently re-run (training cost) | n/a | Present, not re-executed |
| Figure 1 cost-vs-episode curves | `output/` plots + `benchmarks_run.py` | not independently re-run | n/a | Present, not re-executed |
| Trained model checkpoints | `benchmarks/models/` — not in git; README points to a Dropbox download | — | — | External (promised, off-repo) |
| Environment / dependency spec | `requirements.txt` (present, mostly unpinned) | — | — | Present (unpinned) |

## missing

```yaml finding
id: per-seed-values-not-persisted
category: missing
topic: "result traceability"
title: "Raw 10-seed per-run costs (basis of mean±std and t-tests) not saved in repo"
severity: low
confidence: high
status: finding
file: https://raw.githubusercontent.com/mpnguyen2/dfPO/master/output/benchmarks_stat_analysis.csv
url_retrieved_at: "2026-05-31T00:00:00Z"
quote: |
  ,Surface modeling,Grid-based modeling,Molecular dynamics
  DPO,6.296 ± 0.048,6.046 ± 0.083,53.352 ± 0.055
  TRPO,6.470 ± 0.021,7.160 ± 0.113,1842.300 ± 0.007
claim: "The only persisted statistical artefact is the per-algorithm summary mean±std; the underlying 10 per-seed mean costs (computed transiently in benchmarks_run.py:82-92) are never written to disk."
concern: "Without the per-seed values a reviewer cannot re-run the paper's significance tests on the real data, which is exactly what enables the synthetic-resampling shortcut in finding ttest-on-synthetic-data."
resolution: "Authors: please commit the raw per-seed final-cost arrays (10 values per algorithm per task) so the reported t-tests can be reproduced from real data."
cross_refs: ["ttest-on-synthetic-data"]
check_script: _audit_code/check_ttest_synthetic.py
paper_ref: "Appendix C.2, Table 4"
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: trained-models-off-repo-dropbox
category: missing
topic: "expected code completeness"
title: "Trained models for reproduction hosted on an external Dropbox link, not in repo"
severity: low
confidence: medium
status: question
file: https://github.com/mpnguyen2/dfPO
url_retrieved_at: "2026-05-31T00:00:00Z"
quote: |
  Download pre-trained models from a Dropbox link (due to size constraints); run python benchmarks_run.py to reproduce results.
claim: "Reproducing Table 1/Figure 1 from pretrained weights requires downloading model files from an external Dropbox URL referenced in the README; the weights are not version-controlled in the repo (benchmarks/models/ is gitignored)."
concern: "Off-repo, unversioned hosting (Dropbox) is a durability/reproducibility risk if the link rots, and the served weights are not tied to a commit; however dfPO models are tiny (0.17 MB, Table 6) and retrainable, so impact is limited."
resolution: "Authors: confirm the Dropbox link is permanent (or mirror the weights / a Zenodo DOI), and state which commit produced them. (Quote is a paraphrased README summary; verify exact wording on the repo.)"
cross_refs: []
paper_ref: "NeurIPS checklist Q5; §4.2"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: false
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: deps-unpinned
category: missing
topic: "dependencies / environment"
title: "requirements.txt unpinned except pyrosetta — environment not reconstructable"
severity: low
confidence: high
status: question
file: https://raw.githubusercontent.com/mpnguyen2/dfPO/master/requirements.txt
url_retrieved_at: "2026-05-31T00:00:00Z"
quote: |
  gym
  gymnasium
  numpy
  pyrosetta==2023.12+release.1d87148
  sb3_contrib
  scipy
  stable_baselines3
  torch
claim: "Only pyrosetta carries a version pin; numpy, torch, stable_baselines3, sb3_contrib, scipy, gym/gymnasium and the rest are unpinned."
concern: "Baselines are built on Stable-Baselines3 whose default hyperparameters and RNG behavior change across versions, so the exact reported baseline numbers cannot be guaranteed to rebuild; also pyrosetta is a non-free dependency that may block re-running the molecular-dynamics task."
resolution: "Authors: pin exact versions (or ship a lockfile / environment.yml) for at least numpy, torch, stable_baselines3, sb3_contrib, and gymnasium. (Quote condenses requirements.txt; verify the full file.)"
cross_refs: []
paper_ref: "§4.2 (Stable-Baselines3); requirements.txt"
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: false
  control_flow: true
  condition_satisfiable: true
```

## bug

No reproducible runtime bug was identified from the inspected files. The repo
was not cloned/executed beyond the statistical reproduction script, so this is a
finding-free pass for `bug`, not an assertion that the code is bug-free.

## difference

```yaml finding
id: ttest-described-as-on-seed-means
category: difference
topic: "statistical integrity (paper vs code)"
title: "Paper says t-tests on seed-level means; code tests synthetic resamples"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  Statistical comparisons are conducted using t-tests on the seed-level means.
claim: "The paper (Appendix C.2) describes the significance tests as t-tests on the 10 seed-level means; the repo's analysis.ipynb instead runs t-tests on np.random.normal draws synthesized from the reported summary mean/std (see methodology finding ttest-on-synthetic-data)."
concern: "The described procedure (test on real seed means) and the implemented procedure (test on fabricated Gaussian samples) disagree; this is the paper-vs-code facing of the same defect whose primary owner is the methodology finding."
resolution: "Authors: either run the t-test on the actual per-seed means, or correct the Appendix text; clarify which was intended."
cross_refs: ["ttest-on-synthetic-data", "per-seed-values-not-persisted"]
check_script: _audit_code/check_ttest_synthetic.py
paper_ref: "Appendix C.2"
tags: [stats:statcheck, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: moving-branch-no-commit-tag
category: difference
topic: "repository provenance"
title: "No commit/tag pins the camera-ready paper to a repo state"
severity: low
confidence: high
status: question
file: https://github.com/mpnguyen2/dfPO/commits/master
url_retrieved_at: "2026-05-31T00:00:00Z"
quote: |
  9dc65e7 README update (small) 2025-10-23
claim: "The repo has 0 release tags and a moving master branch; the README does not state which commit corresponds to the NeurIPS 2025 paper. Latest commit 9dc65e7 (2025-10-23) post-dates the submission cycle."
concern: "Audits and reproductions can drift against a moving branch; there is no immutable snapshot tied to the published numbers."
resolution: "Authors: tag the commit that produced the paper's tables/figures. (Quote condenses the gh-api commit listing; verify on the repo.)"
cross_refs: []
paper_ref: "§4.2 implementation link"
tags: [forensics:git-archaeology, reforms:2]
validator_pass:
  quote_match: false
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: ttest-on-synthetic-data
category: methodology
topic: "statistical integrity"
title: "Reported significance t-tests run on synthetic Gaussian draws, not experimental data"
severity: high
confidence: high
status: finding
file: https://raw.githubusercontent.com/mpnguyen2/dfPO/master/analysis.ipynb
url_retrieved_at: "2026-05-31T00:00:00Z"
quote: |
  np.random.seed(42)
  n = 10
  dpo_md = np.random.normal(loc=6.296, scale=0.048, size=n)
  crossq_md = np.random.normal(loc=6.365, scale=0.030, size=n)
  stat_md, pval_md = ttest_ind(dpo_md, crossq_md, equal_var=False)
claim: "analysis.ipynb cell 4 hard-codes each algorithm's reported summary mean and std, draws n=10 fresh samples per group via np.random.normal(loc=mean, scale=std), and runs scipy ttest_ind on those synthetic samples to produce the paper's significance verdicts; the real per-seed means (computed in benchmarks_run.py:82-92) are never used."
concern: "The p-values are a function of the (mean,std) the authors already report plus a fixed RNG seed, not of the experimental observations, so they cannot evidence statistical significance of dfPO over baselines and the Appendix C.2 significance claims are unsupported by the data."
resolution: "Authors: recompute all t-tests directly on the 10 actual per-seed mean costs (Welch / paired as appropriate), report the test used and one- vs two-sided, and update Table 4 / Appendix C.2 verdicts accordingly."
cross_refs: ["ttest-described-as-on-seed-means", "per-seed-values-not-persisted"]
check_script: _audit_code/check_ttest_synthetic.py
paper_ref: "Appendix C.2, Table 4; NeurIPS checklist Q7"
tags: [stats:statcheck, reforms:7, whalen:pitfall-5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ttest-near-zero-baseline-std
category: methodology
topic: "statistical integrity"
title: "Sub-0.001 across-seed std for several baselines questions 10-seed independence"
severity: medium
confidence: medium
status: question
file: https://raw.githubusercontent.com/mpnguyen2/dfPO/master/analysis.ipynb
url_retrieved_at: "2026-05-31T00:00:00Z"
quote: |
  dpo_mol = np.random.normal(loc=53.352, scale=0.055, size=n)
  ddpg_mol = np.random.normal(loc=68.203, scale=0.001, size=n)
  stat_mol, pval_mol = ttest_ind(dpo_mol, ddpg_mol, equal_var=False)
claim: "For molecular dynamics the test draws DDPG samples with std=0.001 (the reported summary std), yielding the reproduced t=-948.9, p=8.0e-24 (see _audit_code/out/ttest_synthetic.csv); such near-zero across-seed variance for an off-policy baseline implies near-deterministic behavior across 10 seeds."
concern: "Setting aside the synthetic-data problem (owner: ttest-on-synthetic-data), the sub-0.001 reported across-seed std for several baselines (DDPG, TQC, S-DDPG in Table 4) is suspiciously small and may indicate the 10 seeds did not actually vary the baseline outcome (e.g. shared cached rollout)."
resolution: "Authors: confirm the 10 baseline seeds produced genuinely independent runs and report the real per-seed spread."
cross_refs: ["ttest-on-synthetic-data"]
check_script: _audit_code/check_ttest_synthetic.py
paper_ref: "Table 4 (DDPG/TQC/S-DDPG std columns)"
tags: [stats:statcheck, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 3          | low          | Per-seed raw values unsaved; trained models on Dropbox; deps unpinned. |
| bug         | 0          | -            | No bug verified at a specific line (URL-level audit; repo not cloned). |
| difference  | 2          | medium       | Paper text describes seed-mean t-tests; no commit pins the paper. |
| methodology | 2          | high         | Significance t-tests computed on synthetic resamples, not real data. |

## Top take-aways (≤6, ranked)

1. **[methodology, high/high]** `ttest-on-synthetic-data` — The paper's
   significance tests (Appendix C.2, "t-tests on the seed-level means") are
   computed in `analysis.ipynb` on `np.random.normal` draws fabricated from the
   reported mean/std with a fixed seed; they do not use the experimental data
   and cannot support the significance claims.
2. **[difference, medium/high]** `ttest-described-as-on-seed-means` — Paper-vs-code
   facing of #1: the prose describes a different (valid) procedure than the code
   runs.
3. **[methodology, medium/medium]** `ttest-near-zero-baseline-std` — Several
   baselines report sub-0.001 across-seed std, raising doubt about 10-seed
   independence.
4. **[missing, low/high]** `per-seed-values-not-persisted` — The raw per-seed
   costs underlying Table 4 are never saved, preventing an honest re-run of the
   tests.
5. **[missing, low/high]** `deps-unpinned` — Unpinned SB3/torch/numpy means
   baseline numbers may not rebuild exactly; pyrosetta is non-free.
6. **[difference, low/high]** `moving-branch-no-commit-tag` — No tag ties the
   paper to a repo state.

## Items that genuinely look fine

- The headline empirical means (Table 1 / Table 4) match the repo's persisted
  `output/benchmarks_stat_analysis.csv` exactly, and `benchmarks_run.py` does
  compute genuine per-seed means over the 10 declared seeds
  `[42,75,105,122,137,203,381,411,437,479]` (line 143) — the *means* trace
  cleanly to code; only the significance test is synthetic.
- Code IS released and the repo resolves with training, evaluation, baseline,
  and reproduction scripts plus a README with reproduction commands — contrary
  to the initial "no code" framing.
- Compute (A100) and the 10-seed protocol are disclosed; the contribution is
  primarily theoretical (convergence + regret bound) with empirical support.

## Open questions for the authors

- Were the 10 baseline seeds truly independent runs? The sub-0.001 reported
  across-seed std for DDPG/TQC/S-DDPG (Table 4) is hard to reconcile with
  stochastic RL training and feeds the fragile significance arithmetic.
- Will you persist the raw per-seed arrays and recompute the t-tests on real
  data, and tag the commit corresponding to the camera-ready results?
