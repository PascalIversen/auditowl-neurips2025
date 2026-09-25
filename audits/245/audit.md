# Audit — Adaptive Data-Borrowing for Improving Treatment Effect Estimation using External Controls (NeurIPS 2025, #245)

## 1. Summary

The repository is the OpenReview supplementary `code/openreview_supplementary.zip`
(`NIPS25_code_submit_final/`). It contains two parts: `example/` (the synthetic
illustration of Section 7 / Figure 1) and `application/` (the quantitative
experiments of Section 8 — Tables 3 and Figures 2–3 on Linear, Nonlinear ("exp")
and NSW/PSID data). The method has two selectors: an adaptive-LASSO bias selector
(`lasso_selector.py`, baseline of Gao et al. [17]) and the proposed
influence-function selector (`influence_selector.py`), an AIPW borrowing estimator
(`estimator.py:aipw_borrow`), data generators (`simulate_data.py`), and a driver
(`main.py`). Pre-computed result tables ship under `application/results/`.

What I did (scripts in `_audit_code/`, outputs in `_audit_code/out/`):
- Extracted the zip read-only into `_audit_code/extracted/` and read every `.py`.
- `check_select_samples.py` — verified the obfuscated `argsort(...)[::-1][-top_k:]`
  selection actually returns the K *smallest*-score (most comparable) samples (it does).
- `check_sim_reproduce.py` — regenerated the simulated Linear CSVs with the shipped
  generator (seed 42) and confirmed byte-for-byte (atol 1e-6) reproduction of the
  shipped `dataset/simulated_data/*.csv`.
- Cross-checked the shipped `results/markdowntable/...` against paper Table 3 for
  Linear (`linear_mu_0.1/control_n_100`), Nonlinear (`exp_mu_0.1/control_n_100`) and
  NSW (`NSW/control_n_80`) — the reported `std` values match exactly.
- AST/grep checks for an optimal-K (`S*`) selection step, dependency files, dead
  imports, replication loops, and data-file row counts.

Overall: the code that produces the per-Top-K numbers in Table 3 / Figure 2 is
present and the values reproduce from the shipped artefacts. The main gaps are
(a) the paper's headline "data-driven optimal-K selection" step has no code, (b) no
dependency specification, (c) a single-realization (no Monte-Carlo) evaluation
underlying the "std"/"bias" comparisons, and (d) a per-group MinMaxScaler in the
NSW pipeline that scales each group (and the outcome Y) independently.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 3 Linear, ˆτif std Top-K=10 (0.0963) / \|bias\| (0.0005) | `application/main.py` → `results/markdowntable/linear/linear_mu_0.1/control_n_100/valid_markdowntable_top_k_10.txt` | se=0.09633, bias=0.00059 | ✓ | Verified |
| Table 3 Linear, ˆτlasso Top-K=10 (0.0994/0.0082) | same file | se=0.09947, bias=0.00820 | ✓ | Verified |
| Table 3 Linear, ˆτfull (0.0912/0.0951) | same file | se=0.09118, bias=0.09507 | ✓ | Verified |
| Table 3 Linear, ˆτif Top-K=50 (0.0902/0.0075) | `.../control_n_100/...top_k_50.txt` | se=0.09030, bias=0.00751 | ✓ | Verified |
| Table 3 Nonlinear, ˆτif/ˆτlasso/ˆτfull std Top-K=10 (0.2090/0.2093/0.1862) | `results/markdowntable/exp/exp_mu_0.1/control_n_100/...top_k_10.txt` | 0.20904/0.20936/0.18622 | ✓ | Verified |
| Table 3 NSW, ˆτif/ˆτlasso/ˆτfull std Top-K=10 (0.0131/0.0142/0.0149) | `results/markdowntable/NSW/control_n_80/...top_k_10.txt` | 0.01311/0.01428/0.01499 | ✓ | Verified |
| Table 3 ˆτaipw std (Linear 0.1075, NSW 0.0134) | `estimator.py:aipw_borrow` (`se_rct`) | 0.1074–0.1080, 0.01349 | ✓ (rounding) | Verified |
| Figure 2 (MSE vs Top-K, all 3 datasets) | `main.py:159-199` (`bv` = bias²+var) + plotting `main.py:232-429` | per-K `bv` lists | ✓ | Verified (values present) |
| Figure 3 (sensitivity µ2∈{0.2,0.3,0.4}) | `main.py` linear loop over `mu_=[...,0.2,0.3,0.4,...]`; `results/.../linear_mu_0.{2,3,4}` | per-K tables | ✓ | Verified (tables present) |
| Appendix B.1 (control-size sensitivity Nsc∈{70,80,90} / {70,75,85}) | `main.py` `control_list`/`control_n_list`; `results/.../control_n_*` | per-K tables | ✓ | Verified (tables present) |
| Simulated datasets (NE=400, Nt=300, Nc=100, NO=800) | `simulate_data.py` → shipped CSVs | reproduces exactly (seed 42) | ✓ | Verified |
| Step 2: "Find optimal K that minimizes MSE" / S* (Sec. 6, eq. ~ line 425) | (none) | — | — | MISSING (no argmin-over-K code) |
| PSID NO = 123 samples (paper §8.1) | `dataset/NSW/PSID_control.txt` | 128 rows | ✗ | MISMATCH (count) |
| Figure 1 / Section 7 synthetic example | `example/main_example.py` + `example/*.py` | — | n/a (illustrative) | Present (import-path issues, see bug) |

Scope filters (Rule L): Temporal integrity — N/A (no time dimension; earnings years
are static covariates). Pretraining contamination — N/A (no pretrained
model/encoder/embeddings; models are fit from scratch per run).

## 3. Findings

## missing

```yaml finding
id: optimal-k-selection-absent
category: missing
topic: "result traceability / method step"
title: "Headline data-driven optimal-K (S*) selection step has no implementation"
severity: medium
confidence: high
status: finding
file: application/main.py
line_start: 159
line_end: 199
quote: |
    for i in range(len(top_k)):
        print("top_k: ", top_k[i])
        lasso_idx = selector_lasso.select_samples(top_k=top_k[i])
        data_ext_lasso["S"] = 0
        data_ext_lasso.loc[lasso_idx, "S"] = 1
claim: "main.py loops over a fixed grid of Top-K values and reports bias/std/MSE for each K, but no code selects the optimal K = argmin_k MSE(S_k)."
concern: "A core contribution and method step ('We develop a data-driven approach to select the optimal subset ... based on the MSE', Sec. 6 Step 2 / eq. near line 425) is never executed; only per-K curves are produced, so the claimed automatic selection of S* cannot be reproduced."
resolution: "Authors: provide the script that computes K* = argmin over the per-K MSE (bv) and reports the resulting single estimate, or clarify that K was chosen manually from the figures."
cross_refs: []
paper_ref: "Section 6 (Step 2, MSE-minimising selection); Contributions bullet line 71-72"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec
category: missing
topic: "environment / dependencies"
title: "No requirements.txt / environment file; unlisted deps incl. scikit-optimize"
severity: medium
confidence: high
status: finding
file: application/lasso_selector.py
line_start: 1
line_end: 6
quote: |
    from sklearn.linear_model import LassoCV, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from skopt import BayesSearchCV
    from skopt.space import Real
    from sklearn.base import BaseEstimator
    from sklearn.neural_network import MLPRegressor
claim: "The repo ships no requirements.txt / environment.yml / setup.py (confirmed by find), yet imports torch, sklearn, skopt (scikit-optimize), prettytable, scipy, tqdm, joblib, matplotlib, pandas, numpy at module top level."
concern: "The environment cannot be rebuilt from a pinned spec, and `import skopt` (a non-default package) is required at import time even though BayesSearchCV/Real are never used, so the modules will fail to import if scikit-optimize is absent."
resolution: "Authors: add a pinned requirements.txt (with versions of torch/sklearn/skopt etc.) and remove the unused skopt / MLPRegressor imports if not needed."
cross_refs: []
paper_ref: "Reproducibility / code availability"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: simulate-data-pandas-import-scope
category: bug
topic: "imports / module scope"
title: "simulate_data.py uses pd.DataFrame but imports pandas only inside __main__"
severity: low
confidence: high
status: finding
file: application/simulate_data.py
line_start: 23
line_end: 23
quote: |
        rct_data = pd.DataFrame({
claim: "generate_rct_data/generate_external_data reference `pd` (lines 23, 45, 100, 123) but `import pandas as pd` appears only inside `if __name__ == \"__main__\"` (line 139); there is no top-level pandas import (verified by AST)."
concern: "Running the file as a script works (the __main__ block imports pandas before calling the generators), but importing simulate_data as a module and calling the generators raises NameError: name 'pd' is not defined (reproduced in _audit_code/out/check_sim_reproduce.txt before a workaround)."
resolution: "Move `import pandas as pd` to module top level."
cross_refs: []
check_script: _audit_code/check_sim_reproduce.py
paper_ref: "Section 8.1 simulation data generation"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: example-import-mismatch
category: bug
topic: "Section 7 example entrypoint"
title: "example/main_example.py has an unresolvable import and a wrong-package class"
severity: low
confidence: medium
status: question
file: example/main_example.py
line_start: 1
line_end: 4
quote: |
    import matplotlib.pyplot as plt
    from example.generate_data import generate_demonstration_data
    from lasso_selector import SelectiveBorrowingLasso
    from influence_selector import Influenceselector
claim: "The Section-7 entrypoint imports `from example.generate_data import ...` (a package-qualified path that only resolves if run from the repo root with `example` as a package) and imports `SelectiveBorrowingLasso`/`Influenceselector` from bare module names, while the sibling files `example/lasso_selector.py` and `example/influence_selector.py` are the intended versions."
concern: "Depending on the working directory the script either fails to import `example.generate_data` or binds `SelectiveBorrowingLasso` to the application-folder class whose constructor requires a `dataset` argument and whose `select_samples` has a different signature than the call at line 25, so the illustrative example (Figure 1) is not runnable as shipped without adjusting the import paths."
resolution: "Authors: confirm the exact run command/working directory for the example, and make the imports consistent (use relative imports or run as a package)."
cross_refs: []
paper_ref: "Section 7 / Figure 1"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## difference

```yaml finding
id: psid-sample-count
category: difference
topic: "dataset description"
title: "Paper states PSID NO=123 but shipped PSID_control.txt has 128 rows"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  while the PSID dataset includes NO = 123 samples.
claim: "The paper (§8.1) states the external PSID control set has NO = 123 samples; the shipped dataset/NSW/PSID_control.txt contains 128 rows, and main.py's NSW top_k grid runs up to 128, consistent with 128 external samples."
concern: "The reported external-control sample size does not match the data actually used; minor, but the exact N used in the NSW experiment differs from the text."
resolution: "Authors: reconcile the PSID sample count (123 vs 128) in the text and code."
cross_refs: []
check_script: _audit_code/out/check_sim_reproduce.txt
paper_ref: "Section 8.1, line 504"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: single-realization-no-replication
category: methodology
topic: "statistical integrity / evaluation"
title: "Reported std and bias come from one dataset realization with no Monte-Carlo repetition"
severity: medium
confidence: medium
status: finding
file: application/estimator.py
line_start: 54
line_end: 60
quote: |
    bias = np.abs(tau_borrow - tau_exp)
    variance = np.var(psi) / X.shape[0]
    variance_1 = np.var(chi) / X.shape[0]
    se_1 = np.sqrt(variance_1)

    bv = bias ** 2 + variance
    se = np.sqrt(variance)
claim: "The 'std' column of Table 3 is the analytic plug-in standard error sqrt(var(psi)/n) of a single AIPW point estimate, and 'bias' is |tau_borrow - tau_aipw| on a single simulated dataset (data CSVs generated once with seed 42; main.py has no outer replication loop, confirmed by grep)."
concern: "Comparative claims that ˆτif 'has a smaller standard deviation' and 'smaller bias' than baselines rest on a single dataset draw with no Monte-Carlo averaging or repeated-seed variation, so the differences may not be stable and no sampling distribution of the estimators is estimated."
resolution: "Authors: report mean +/- std over many simulation replicates (multiple seeds) rather than the analytic SE on one realization, so the std/bias comparisons reflect genuine sampling variability."
cross_refs: []
paper_ref: "Section 8.2, Table 3 (std/|bias|)"
tags: [reforms:7, whalen:pitfall-4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: nsw-per-group-minmax-scaling
category: methodology
topic: "preprocessing / real-data pipeline"
title: "NSW pipeline MinMax-scales treated, control, external (and Y) groups independently"
severity: medium
confidence: medium
status: finding
file: application/main.py
line_start: 41
line_end: 51
quote: |
        scaler_rct_treat = MinMaxScaler()
        data_rct_treated[numeric_cols] = scaler_rct_treat.fit_transform(data_rct_treated[numeric_cols])
        scaler_rct_control = MinMaxScaler()
        data_rct_control[numeric_cols] = scaler_rct_control.fit_transform(data_rct_control[numeric_cols])
        scaler_ext = MinMaxScaler()
        data_ext[numeric_cols] = scaler_ext.fit_transform(data_ext[numeric_cols])

        X = ["age", "education", "Black", "Hispanic", "married", "nodegree", "RE74", "RE75"]

        X_ext = data_ext[X].values
        y_ext = data_ext["Y"].values
claim: "For NSW, three separate MinMaxScalers are fit independently on the treated, RCT-control, and external groups (numeric_cols includes the outcome Y), so each group's covariates and outcome are mapped to its own [0,1] range."
concern: "The comparability bias b = |mu_ext(X) - mu_rct(X)| and the borrowed-outcome estimates are then computed across groups whose covariate/outcome scales differ; a model trained on RCT-scaled X applied to external-scaled X (and per-group-rescaled Y) mixes incompatible scales, which can distort both the selector ranking and the AIPW estimate on the NSW experiment."
resolution: "Authors: fit a single scaler on the pooled covariates (or none) and avoid scaling the outcome per group; confirm whether the NSW results change under a shared scaling."
cross_refs: []
paper_ref: "Section 8.1 Real-Data Application; Table 3 NSW"
tags: [leakage:L1.1, reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 2          | medium       | Optimal-K (S*) selection step absent; no dependency spec.    |
| bug         | 2          | low          | pandas import-scope; Section-7 example import mismatch.       |
| difference  | 1          | low          | PSID N=123 (paper) vs 128 rows (data).                       |
| methodology | 2          | medium       | Single-realization std/bias; NSW per-group MinMax scaling.   |

### Top take-aways (≤6, ranked by severity × confidence)
1. [missing] `optimal-k-selection-absent` — the paper's data-driven optimal-K (S*)
   selection, a headline contribution, has no code; only per-K curves are produced. (high confidence)
2. [missing] `no-dependency-spec` — no requirements/env file; non-default `skopt` is
   a hard import dependency. (high confidence)
3. [methodology] `single-realization-no-replication` — Table 3 "std"/"bias" come from
   one simulated dataset draw (analytic SE), no Monte-Carlo replication. (medium confidence)
4. [methodology] `nsw-per-group-minmax-scaling` — NSW covariates and outcome scaled
   independently per group, mixing incompatible scales in the selector/estimator. (medium confidence)
5. [difference] `psid-sample-count` — PSID N=123 in text vs 128 rows in the data. (high confidence)
6. [bug] `simulate-data-pandas-import-scope` — pandas imported only in `__main__`;
   importing the module breaks the generators. (high confidence)

### Items that genuinely look fine
- The obfuscated selection `np.argsort(scores)[::-1][-top_k:]` correctly returns the
  K smallest-score (most comparable) samples (verified, `_audit_code/out/check_select_samples.txt`).
- The shipped simulated CSVs reproduce exactly from `simulate_data.py` with seed 42
  (`_audit_code/out/check_sim_reproduce.txt`).
- Table 3 std/bias values (Linear, Nonlinear, NSW) match the shipped result tables,
  which are computed by `main.py`/`aipw_borrow` (not merely re-plotted).
- The τ̂aipw benchmark (`tau_exp`) is effectively constant across Top-K as expected
  for an R-restricted estimator (verified across K=10..800).
- Simulated sample sizes (NE=400, Nt≈300, Nc≈100, NO=800) and NSW shapes (185/260/PSID)
  match the paper; treatment assignment is constant-propensity (valid "completely at random").

### Open questions for the authors
- Was K* chosen automatically (and if so by which script) or read off the figures?
  (relates to `optimal-k-selection-absent`)
- Are the Table 3 numbers a single realization or averaged over replicates? If single,
  how stable are the ˆτif < baseline comparisons? (relates to `single-realization-no-replication`)
- Is the per-group MinMax scaling of NSW (including Y) intentional, and do the NSW
  results survive a shared/no scaling? (relates to `nsw-per-group-minmax-scaling`)
</content>
</invoke>
