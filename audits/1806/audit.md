# Audit — BayeSQP: Bayesian Optimization through Sequential Quadratic Programming (paper 1806)

## 1. Summary

The repository `brunzema/bayesqp` is a **plug-and-play Python package** implementing
the BayeSQP optimizer (an `__init__`, the `BayeSQP` driver class, GP/Hessian models,
the SOCP subproblem solver, config dataclasses, BoTorch problem wrappers, and five
demo notebooks under `examples/`). It is published as the `bayesqp` PyPI package and
is intended for practitioners to run BayeSQP on their own BoTorch problems.

I read every source file under `src/bayesqp/` (`bayesqp.py`, `subproblems.py`,
`models.py`, `configs.py`, `utils.py`, `objective_wrappers.py`, `__init__.py`), the
`README.md`, the `examples/` notebooks, and `pyproject.toml`; and I read the paper's
empirical section (§5, Figures 4–6, Tables 1–2) and the NeurIPS reproducibility
checklist. I cross-checked the implemented algorithm (local Sobol-sphere subsampling,
the uncertainty-aware second-order cone subproblem B-SUB, the slack fallback, and the
constrained-posterior-sampling line search) against the paper's §4 description — it is
**faithful**. I ran two read-only checks under `_audit_code/`:
`check_reset_path_undefined_attrs.py` (AST scan confirming two instance attributes are
read but never assigned) and `check_experiment_artifacts.py` (repo-wide grep confirming
no baseline/benchmark/seed-loop/result code exists).

**Headline finding:** the repo is the *method library only*. It contains **none** of the
code that produces the paper's reported numbers — no baseline implementations (logEI,
TuRBO, SAASBO, MPD, C-logEI, SCBO), no benchmark functions (random-Fourier-feature
within-model objectives, Ackley, Hartmann variants used in Table 1, Speed Reducer,
Gramacy), no 32-seed driver, and no result/figure/table generation. The paper's
reproducibility statement claims "we provide instructions to reproduce all experiments,"
which the repository does not support. The core method code itself is sound and matches
the paper; the only code defects are two latent crashes in the (default-off) restart path.

## 2. Traceability table

Every quantitative artefact in the paper is traced to repo code that would *compute* it.
"(none)" means no script/function/notebook in the repo computes the value. Backed by
`_audit_code/out/experiment_artifacts.json` (repo-wide scan for baseline/benchmark/seed
terms: no hits beyond a kernel-config name and spurious notebook-JSON matches).

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 4 (a–c) unconstrained within-model: BayeSQP vs logEI/TuRBO/SAASBO/MPD, d∈{4..96}, 32 seeds | (none) | — | — | MISSING (no baselines, no within-model RFF objective, no seed loop) |
| Fig. 5 (a–c) constrained within-model: BayeSQP vs SCBO/C-logEI; runtime; δ-sensitivity at d=64 | (none) | — | — | MISSING |
| Fig. 6 Gramacy optimization paths | (none) | — | — | MISSING (no Gramacy benchmark) |
| Table 1 Ackley5D/Hartmann/Ackley20D (+constr.) medians & quantiles, feas. counts | (none) | — | — | MISSING (no Table-1 benchmark harness) |
| Table 2 Speed Reducer performance & avg. runtime (SCBO/C-logEI/BayeSQP) | (none) | — | — | MISSING (no Speed Reducer benchmark) |
| Abstract/§5 headline "BayeSQP outperforms SOTA from dimension 16 onward" | (none) | — | — | MISSING (depends on Fig. 4/5 above) |
| Algorithm B-SUB subproblem (Eqs. 10–18) | `src/bayesqp/subproblems.py:497-772` | n/a (solver) | ✓ (faithful) | Verified (method present, sound) |
| Local Sobol-sphere subsampling r=ε·u^(1/d) (§4.4) | `src/bayesqp/bayesqp.py:238-252` | n/a | ✓ | Verified |
| Line search via constrained posterior sampling, M=3, Eq. 20 selection (§4.3) | `src/bayesqp/bayesqp.py:443-555` | n/a | ✓ | Verified |
| Hessian of GP posterior mean (RBF) (§3) | `src/bayesqp/models.py:317-366` | n/a | ✓ | Verified |
| Default δf=δc=0.2, K=d+1 (§5) | `src/bayesqp/configs.py:38-42`, `bayesqp.py:70-73` | n/a | ✓ | Verified |

Routing: every MISSING row is owned by the single `missing` finding
`experiment-harness-missing` (Single-Owner Rule); they are not re-filed per row.

## 3. Findings

### missing

```yaml finding
id: experiment-harness-missing
category: missing
topic: "result traceability / experiment reproduction"
title: "Repo is the method library only; no code reproduces any paper experiment"
severity: high
confidence: high
status: finding
file: code/brunzema__bayesqp/README.md
line_start: 9
line_end: 9
quote: |
  The repository contains a _plug-and-play_ implementation of BayeSQP (NeurIPS 2025). BayeSQP as a framework aims to combine ideas from both Bayesian optimization (BO) and ideas from sequential quadratic optimization to effectively solve potentially constrained black-box optimization problems.
claim: "The repository ships only the BayeSQP optimizer package plus demo notebooks; it contains no baseline implementations (logEI, TuRBO, SAASBO, MPD, C-logEI, SCBO), no paper benchmark functions (within-model RFF objectives, Ackley variants of Table 1, Speed Reducer, Gramacy), no 32-seed driver, and no result/figure/table generation code."
concern: "None of the paper's quantitative claims (Figs. 4-6, Tables 1-2, the headline 'outperforms SOTA from dimension 16 onward') can be reproduced or verified from the repository; the comparative results that establish the paper's central empirical claim are entirely off-repo."
resolution: "Authors: please release the experiment harness (baseline configs, the within-model RFF benchmark generators, Speed Reducer and Gramacy definitions, the 32-seed driver, and the table/figure scripts), or point to where it lives."
cross_refs: ["repro-statement-overclaims-instructions"]
check_script: _audit_code/check_experiment_artifacts.py
paper_ref: "§5 Empirical evaluations; Figures 4-6; Tables 1-2"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: reset-path-undefined-attrs
category: bug
topic: "restart / reset logic"
title: "use_resets=True crashes: self.subsampling_option and self.tol never defined"
severity: medium
confidence: high
status: finding
file: code/brunzema__bayesqp/src/bayesqp/bayesqp.py
line_start: 583
line_end: 587
quote: |
                        new_x, new_y, next_constraints = self._generate_local_samples(
                            x_k=x_k,
                            n=self.n_subsamples,
                            option=self.subsampling_option,
                        )
claim: "Inside the `use_resets` reset branch, `self.subsampling_option` (line 586) and `self.tol` (used by `_check_reset_conditions`, line 662) are read but are never assigned anywhere in the package; the config field is `self.config.tol` / `self.config.local_sample_strategy`."
concern: "Enabling the documented restart feature (`use_resets=True`, default False) raises AttributeError mid-run, so the restart-on-stagnation behavior described in the paper's practical-considerations section cannot actually be exercised through the public config."
resolution: "Authors: replace `self.subsampling_option` with `self.config.local_sample_strategy` and `self.tol` with `self.config.tol`, and add a test that runs with `use_resets=True`."
cross_refs: []
check_script: _audit_code/check_reset_path_undefined_attrs.py
paper_ref: "§4.4 Practical considerations (restart behavior)"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unconstrained-subproblem-returns-none
category: bug
topic: "error handling"
title: "Unconstrained solver returns bare None on failure; caller unpacks two values"
severity: low
confidence: high
status: finding
file: code/brunzema__bayesqp/src/bayesqp/subproblems.py
line_start: 970
line_end: 971
quote: |
            )
            return None
claim: "`solve_bayesqp_unconstrained` returns a bare `None` on solver exception (lines 971 and 979), but the caller unpacks `res, info = solve_bayesqp_unconstrained(...)` at bayesqp.py:376."
concern: "If the unconstrained subproblem solver hits an exception, the optimizer crashes with a TypeError on the tuple unpack instead of degrading gracefully (the constrained path returns a valid failed-result tuple via create_failed_bayesqp_result, so the inconsistency is the code's own intent)."
resolution: "Authors: return `create_failed_bayesqp_result(n, 0), {...}` (a 2-tuple) on the unconstrained failure paths, matching the constrained solver's contract."
cross_refs: []
paper_ref: "§4.1 B-SUB (unconstrained case)"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: repro-statement-overclaims-instructions
category: difference
topic: "reproducibility statement vs repo contents"
title: "Paper claims instructions/scripts to reproduce all experiments; repo has neither"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  All benchmarks are publicly available and we provide instructions to reproduce
  all experiments. The implementation is provided under https://github.com/brunzema/
  bayesqp and easily accessible via PyPI.
claim: "The NeurIPS reproducibility checklist (open-access-to-code item) answers 'Yes' and states the authors provide instructions to reproduce all experiments; the linked repo provides only the method package and demo notebooks, with no experiment instructions, baseline configs, or result scripts (see finding experiment-harness-missing)."
concern: "Readers relying on the availability statement will not find the promised reproduction instructions/scripts in the repository, so the stated reproducibility avenue does not match the released artefact."
resolution: "Authors: either add the reproduction instructions and scripts to the repo, or amend the statement to scope it to the method implementation plus the paper's Appendix A hyperparameter tables."
cross_refs: ["experiment-harness-missing"]
paper_ref: "NeurIPS Checklist Q5 (Open access to data and code), 'Justification'; §4 Experimental Result Reproducibility (Appendix A)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

No methodology findings. The implemented procedure (local curvature estimation via GP
Hessian, uncertainty-aware SOCP search direction, posterior-sampling line search,
slack fallback) matches the paper's §3–§4 description and is internally valid as a
local black-box optimizer. There is no train/test split, dataset, or statistical test
in the released code to audit (the experiments are off-repo, owned by
`experiment-harness-missing`), so the leakage / split / metric / statistics checklists
are N/A to the repository contents.

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 1 | high | No code reproduces any paper experiment (Figs. 4-6, Tables 1-2). |
| bug | 2 | medium | Restart path reads two never-defined attrs; unconstrained solver returns bare None. |
| difference | 1 | medium | Repro statement promises reproduction instructions/scripts the repo lacks. |
| methodology | 0 | - | Released method code is faithful to paper; experiments are off-repo (N/A). |

## 5. Closing lists

**Top take-aways**
1. `experiment-harness-missing` (`missing`, high/high): the repo is the BayeSQP package only — none of the comparative results behind the paper's central claim are reproducible from it.
2. `repro-statement-overclaims-instructions` (`difference`, medium/high): the checklist promises reproduction instructions/scripts that are absent from the linked repo.
3. `reset-path-undefined-attrs` (`bug`, medium/high): `use_resets=True` crashes on `self.subsampling_option` / `self.tol`, both never assigned.
4. `unconstrained-subproblem-returns-none` (`bug`, low/high): unconstrained solver failure path returns a bare `None` while the caller unpacks a 2-tuple.

**Items that genuinely look fine**
- The B-SUB SOCP construction (objective quadratic + linearized uncertainty-aware constraints + box constraints + second-order cones from the GP joint covariance Cholesky) matches the paper's formulation (`subproblems.py`).
- Local sub-sampling uses the exact Sobol-sphere transform `r = ε·u^(1/d)` described in §4.4 (`bayesqp.py:238-252`).
- The line-search next-iterate selection (best feasible, else least total violation) matches Eq. 20 (`bayesqp.py:150-159`, `530-555`).
- Defaults δf=δc=0.2 and K=d+1 ("auto") match §5 (`configs.py:38-42`, `bayesqp.py:70-73`).
- Dependencies are declared in `pyproject.toml` (numpy, scipy, cvxopt, botorch, gpytorch) and a `uv.lock` is committed; `seed_everything` seeds random/numpy/torch/CUDA.

**Open questions for the authors**
- Where does the experiment harness (baselines, RFF within-model generators, Speed Reducer, Gramacy, 32-seed driver, table/figure scripts) live? Is it intended to be released?
- Is the audited single-commit `main` the state used for the paper, or should a tagged submission commit be referenced?
