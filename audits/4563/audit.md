# Audit — Automated Model Discovery via Multi-modal & Multi-step Pipeline (NeurIPS 2025, #4563)

## Summary

The repository `code/kaist-ami__Automated-Model-Discovery/` is confirmed to be
THIS paper's official author code: the README title, the five-author roster, the
OpenReview id `qGFvTIMS3W`, the arXiv link `2509.25946`, and the project page
all match the paper and metadata. The code implements the proposed GP-kernel
discovery pipeline (`main_gp.py`) and the symbolic-regression extension
(`main_sr.py`), with VLM modules in `utils/analyzer_utils.py`,
`utils/vision_score_utils.py`, GP fitting in `gpy_fit_func.py`, kernel parsing in
`utils/cfg_utils.py`, and the gpss-research univariate time-series CSVs shipped in
`data/`. The pipeline calls the OpenAI API (closed-source GPT-4o-mini) for both
AnalyzerVLM and EvaluatorVLM, so it is not fully self-contained — a legitimate
exemption per the protocol, but the API non-determinism is itself a
reproducibility caveat.

I audited by reading every Python source file and the paper, then ran
deterministic checks under `_audit_code/` (`check_imports_and_paths.py`,
output `_audit_code/out/imports_and_paths.json`). The checks confirm: (1) three
modules are imported but absent from the repo (`parse_functools`,
`vision_score_gpt2_0105_func`, `cfgs.asmd_cfg`) so the two entrypoints cannot be
imported; (2) 63 lines hardcode the authors' private absolute paths
(`/home/mok/...`, `/node_data_2/...`), including the only data loader and the
EvaluatorVLM few-shot reference images; (3) the `rmse()`/`r2_score()` functions
that would produce the Tables 1/4 metrics are defined but never called — the
driver scripts compute only BIC, validation-MSE, LOO and emit plots, never the
reported train/test RMSE; (4) none of the five GP baselines or three SR baselines
are implemented in the repo; (5) `colorlog`, imported by both entrypoints, is in
neither `requirements.txt` nor `requirements.yaml`. The net effect: the released
code does not run as shipped and contains no script that computes the headline
numbers in any table or figure. This is overwhelmingly a `missing` /
reproducibility story; the few `difference`/`bug` items are secondary.

## Result-traceability coverage table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|-----------------------------|----------------------------|----------------|---------------|-------------------------------|
| Table 1 RMSE (Ours GPT-4o-mini, 7 datasets, train+test) | `main_gp.py` produces only BIC/val-MSE/LOO + plots; `rmse()` in `utils/dataload_utils.py:361` never called | — | — | MISSING (no RMSE computation) |
| Table 1 RMSE Ours (Qwen2.5-VL) row | Qwen path in `utils/main_utils.py:209-232` (model object commented out); no RMSE | — | — | MISSING |
| Table 1 RMSE Ours (GPT-4o) row | `main_gp.py --model gpt-4o`; no RMSE computation | — | — | MISSING |
| Table 1 baselines: GP(SE), ARIMA, Prophet, Automatic Statistician, BoxLM | (none — no baseline scripts in repo) | — | — | MISSING (no baseline code) |
| Table 2 ablation (Analyzer/Evaluator on/off, MSE, 4 datasets) | `USE_ANALYZER`/`USE_VISUAL_SCORE` flags hardcoded `True` in `main_gp.py:96,98`; no MSE-vs-test export | — | — | MISSING (flags fixed; no metric output) |
| Table 3 α grid (0/30/50/70/100, Airline+Radio test RMSE) | `--alpha` flag exists (`main_gp.py:78`); no RMSE export | — | — | PARTIAL (knob present, metric not computed) |
| Table 4 SR R²/RMSE (Ours + SGA/ICSR-V/LLM-SR) | `main_sr.py`; `r2_score`/`rmse` in `utils/dataload_utils.py:361-381` never called; no baseline code | — | — | MISSING |
| Fig. 2 qualitative predictions | `main_gp.py:489-529` plots final models | plots only | — | PARTIAL (plot, not a metric) |
| Fig. 3 MSE text-only vs multimodal (Δ −0.0952 …) | `USE_VISUAL_SCORE` flag in `main_gp.py:98`; no MSE export script | — | — | MISSING |
| Fig. 4 LLM-vs-VLM analysis-step counts | no step-count logging/aggregation script | — | — | MISSING |
| Fig. 5 single- vs multi-step MSE (Δ −0.0171 …) | no single-step-restriction code path or MSE export | — | — | MISSING |
| Fig. 7 BIC vs VIC selection | BIC (`utils/main_utils.py:37`) + VIC scoring present; no figure-producing script | partial | — | PARTIAL |
| Fig. 8 train/test MSE over rounds | per-round `val mse`/BIC logged; test-MSE-over-rounds not computed | — | — | MISSING |
| Fig. 9 Spearman ρ (VIC vs LL / gen-gap / human) ρ=0.7995/0.841/0.777/0.580 | no spearman/correlation/human-eval code in repo | — | — | MISSING |
| Fig. 10 init-from-AnalyzerVLM vs random | `initialize_mods_already_tuned` vs random init exist (`gpy_fit_func.py`); no figure script | partial | — | PARTIAL |
| Table A6 SR-on-real-data + Ours(SR+GP) | no SR+GP hybrid driver, no RMSE export | — | — | MISSING |
| App. A.2 VIC derivation (Eqs. 7–16) | theoretical; `VIC = α·Eval − BIC` implemented via `sort_genetic_pools` (`utils/main_utils.py:88-128`) | matches form | ✓ | Verified (formula form) |
| Data: gpss-research univariate CSVs | `data/*-train.csv`,`*-test.csv` present, but loader reads `/home/mok/...` (`utils/dataload_utils.py:69`) | files present, path wrong | ✗ | MISMATCH |

Every numeric artefact in every table and figure is either MISSING (no computing
script) or PARTIAL (a knob/plot exists but the reported number is not produced by
repo code). No table value could be recomputed.

## missing

```yaml finding
id: missing-rmse-table1-table4
category: missing
topic: "result traceability"
title: "No code computes the train/test RMSE (Table 1) or R²/RMSE (Table 4)"
severity: high
confidence: high
status: finding
file: utils/dataload_utils.py
line_start: 361
line_end: 362
quote: |
  def rmse(y_true, y_pred):
      return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
claim: "rmse() and r2_score() are defined in dataload_utils.py but have zero call sites anywhere in the repo (verified by AST/regex scan); main_gp.py and main_sr.py compute only BIC, validation MSE, and LOO, and use testX/testY solely for plotting (main_gp.py:497, main_sr.py:528). No script evaluates the final selected model's RMSE/R² on the test region."
concern: "The headline RMSE numbers in Table 1 and the R²/RMSE numbers in Table 4 (the paper's central quantitative claims) cannot be reproduced or traced to any computation in the released code."
resolution: "Authors: please add (or point to) the script that computes the test-region RMSE/R² for the final selected model and the baselines that populate Tables 1 and 4."
cross_refs: ["missing-baselines", "broken-entrypoint-imports"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Table 1; Table 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-baselines
category: missing
topic: "baselines"
title: "No implementation of any of the eight reported baselines"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  "we compare our pipeline against five competing methods ranging from traditional forecasting methods to the latest LLM-based model discovery approaches: Gaussian Process Regression with Squared Exponential kernel, ARIMA [52], Facebook Prophet [54], Automatic Statistician [14, 33], and BoxLM [30]"
claim: "A repo-wide keyword scan (ARIMA, prophet, BoxLM, statsmodels, pmdarima, auto_arima, SGA, ICSR, LLM-SR) finds zero baseline implementations outside of prompt strings; the appendix says these baselines were implemented (e.g. 'For BoxLM implementation, we have followed...', ARIMA p=2,d=1,q=2, Prophet changepoint_prior_scale=0.1)."
concern: "The comparative claim that 'our pipeline outperforms the others' (Table 1, Table 4) is not reproducible because no baseline code is shipped; the paper describes specific baseline settings (App. A.3) that nothing in the repo implements."
resolution: "Authors: release the ARIMA/Prophet/GP-SE/Automatic-Statistician/BoxLM and SGA/ICSR/LLM-SR baseline scripts (or links) used to produce the comparison tables."
cross_refs: ["missing-rmse-table1-table4"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Section 4.1; Appendix A.3; Table 1; Table 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-cfgs-module
category: missing
topic: "missing module"
title: "cfgs.asmd_cfg imported by analyzer/vision modules is absent from repo"
severity: high
confidence: high
status: finding
file: utils/analyzer_utils.py
line_start: 16
line_end: 16
quote: |
  from cfgs.asmd_cfg import parse_asmd_cfg as parse_kernel
claim: "Both utils/analyzer_utils.py:16 and utils/vision_score_utils.py:14 do `from cfgs.asmd_cfg import parse_asmd_cfg`; there is no `cfgs/` package or `parse_asmd_cfg` definition anywhere in the repo (the actual kernel parser lives in utils/cfg_utils.py as parse_kernel_name)."
concern: "Importing the AnalyzerVLM or EvaluatorVLM modules raises ModuleNotFoundError, so the core pipeline cannot run as shipped."
resolution: "Authors: add the missing `cfgs/asmd_cfg.py` (or fix the imports to `utils.cfg_utils`)."
cross_refs: ["broken-entrypoint-imports", "missing-vision-score-module"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Section 3.2; 3.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: broken-entrypoint-imports
category: missing
topic: "missing module"
title: "parse_functools imported by both entrypoints does not exist"
severity: high
confidence: high
status: finding
file: main_gp.py
line_start: 49
line_end: 49
quote: |
  from parse_functools import param_init, build_hierarchical_dict, get_param_dict, random_update_hierarchical_dict
claim: "main_gp.py:49 and main_sr.py:49 import `parse_functools`; no `parse_functools.py` (or installable package of that name) exists in the repo, confirmed by file scan."
concern: "Both top-level entrypoints fail at import time with ModuleNotFoundError, so neither `python3 main_gp.py ...` nor `python3 main_sr.py ...` (the README's run commands) can execute."
resolution: "Authors: include the `parse_functools.py` module that defines param_init / build_hierarchical_dict / get_param_dict / random_update_hierarchical_dict."
cross_refs: ["missing-cfgs-module", "missing-vision-score-module"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "README run command"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-vision-score-module
category: missing
topic: "missing module"
title: "vision_score_gpt2_0105_func imported by main_utils is absent"
severity: high
confidence: high
status: finding
file: utils/main_utils.py
line_start: 1
line_end: 1
quote: |
  from vision_score_gpt2_0105_func import get_structure_sim_response, get_mean_sim_response, get_confidence_sim_response
claim: "utils/main_utils.py:1 imports `vision_score_gpt2_0105_func`, which is not present in the repo; main_utils is imported by main_gp.py / main_sr.py / vision_score_utils.py, so the failure cascades to every entrypoint."
concern: "The module that defines the EvaluatorVLM structure/mean/confidence scoring helpers is missing, so importing the pipeline crashes."
resolution: "Authors: include `vision_score_gpt2_0105_func.py` (or remove the dead import if these helpers are superseded by utils/vision_score_utils.py)."
cross_refs: ["broken-entrypoint-imports", "missing-cfgs-module"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Section 3.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-fig9-correlation-human
category: missing
topic: "result traceability"
title: "No code computes Fig. 9 Spearman correlations or the human-evaluation study"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "Spearman's ρ : 0.7995 ... Spearman's ρ : 0.841 ... Spearman's ρ : 0.777 ... Spearman's ρ : 0.580"
claim: "A repo-wide scan for spearman/corrcoef/.corr(/human finds no correlation computation and no human-evaluation pipeline; Fig. 9 reports four Spearman ρ values (VIC vs likelihood, vs generalization gap, and VIC criteria vs human ratings) with no producing code."
concern: "The validity claim that VIC correlates with likelihood, the generalization gap, and human judgment cannot be reproduced from the repo."
resolution: "Authors: release the script that computes the Spearman correlations in Fig. 9 and the collected human-rating data."
cross_refs: []
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Figure 9; Appendix A.5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-ablation-figs-3-4-5-8
category: missing
topic: "ablations"
title: "Ablation/analysis figures (Figs 3,4,5,8) have no metric-export code"
severity: medium
confidence: high
status: finding
file: main_gp.py
line_start: 96
line_end: 98
quote: |
    USE_ANALYZER = True
    USE_INHERIT_AT_NO_ISLAND = True
    USE_VISUAL_SCORE = True
claim: "The multi-modal/single-modal (Fig.3) and Analyzer/Evaluator (Table 2) ablations are controlled only by these hardcoded booleans (no CLI flags); the LLM-vs-VLM step count (Fig.4), single-step restriction (Fig.5), and train/test-MSE-over-rounds (Fig.8) have no logging or aggregation script that emits the plotted quantities."
concern: "None of the ablation/analysis figures can be reproduced: the toggles must be edited by hand and, even then, no script computes the MSE deltas or step counts the figures report."
resolution: "Authors: expose the ablation toggles as CLI flags and release the scripts that aggregate the per-figure MSE deltas / step counts."
cross_refs: ["missing-rmse-table1-table4"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Figures 3, 4, 5, 8; Table 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-colorlog-dep
category: missing
topic: "dependencies"
title: "colorlog imported by both entrypoints but absent from requirements"
severity: medium
confidence: high
status: finding
file: main_gp.py
line_start: 38
line_end: 38
quote: |
  from colorlog import ColoredFormatter
claim: "main_gp.py:38 and main_sr.py:38 import colorlog, which appears in neither requirements.txt nor requirements.yaml (verified by grep)."
concern: "Following the README's `pip install -r requirements.txt` then running an entrypoint raises ModuleNotFoundError for colorlog, so the documented environment is incomplete."
resolution: "Authors: add colorlog (and any other unlisted imports) to requirements.txt / requirements.yaml, or drop the unused import."
cross_refs: ["broken-entrypoint-imports"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "README Environment Setup"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-sr-dataset-and-hybrid
category: missing
topic: "data / experiments"
title: "Symbolic-regression datasets and the Ours(SR+GP) hybrid (Table A6) are not in the repo"
severity: medium
confidence: high
status: finding
file: main_sr.py
line_start: 138
line_end: 138
quote: |
    datapath = '/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/data'
claim: "The SR datasets are not shipped (README instructs downloading them externally) and the code points to a private absolute path; there is also no driver implementing the Ours(SR+GP) hybrid that Table A6 reports (no code combining a PySR-style initial composition with the GP pipeline for 2 rounds)."
concern: "Table 4 / Table A6 SR results are not reproducible from the repo: the input data path is private and the hybrid SR+GP experiment is absent."
resolution: "Authors: ship or give a resolvable accession for the SR data folders, and release the SR+GP hybrid driver used for Table A6."
cross_refs: ["hardcoded-private-paths"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Table 4; Table A6; Appendix A.6"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-private-paths
category: bug
topic: "hardcoded paths"
title: "Data loader and EvaluatorVLM few-shot images use nonexistent private absolute paths"
severity: high
confidence: high
status: finding
file: utils/dataload_utils.py
line_start: 69
line_end: 71
quote: |
      path = f'/home/mok/module/icml25/gpss-research/data/tsdlr_9010_csv/mok'
      train_df = pd.read_csv(f'{path}/{data_name}-train.csv')
      test_df = pd.read_csv(f'{path}/{data_name}-test.csv')

claim: "load_gp_data reads from the hardcoded path /home/mok/module/icml25/gpss-research/data/tsdlr_9010_csv/mok (which does not exist), even though the same CSVs are shipped in the repo's ./data directory; 63 lines across the codebase hardcode /home/mok/... or /node_data_2/... paths, including the EvaluatorVLM few-shot reference images in vision_score_utils.py:82-86 and 203-206 that the scoring loop loads via encode_image()."
concern: "Even after the missing modules are supplied, the pipeline crashes with FileNotFoundError on data load and again on EvaluatorVLM few-shot image loading, so no result can be produced on a fresh checkout."
resolution: "Authors: replace the hardcoded /home/mok and /node_data_2 paths with repo-relative paths (e.g. ./data) and ship the few-shot reference images, or load them relative to the repo."
cross_refs: ["data-loader-path-vs-readme", "missing-sr-dataset-and-hybrid"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Appendix A.3 (dataset)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: init-code-hardcoded-path
category: bug
topic: "hardcoded paths"
title: "AnalyzerVLM's injected data-access code embeds the private path"
severity: medium
confidence: high
status: finding
file: utils/prompts.py
line_start: 8
line_end: 8
quote: |
  train_df = pd.read_csv('/home/mok/module/icml25/gpss-research/data/tsdlr_9010_csv/mok/%s-train.csv')
claim: "INIT_CODE (prompts.py:8), the data-access snippet injected into and executed by AnalyzerVLM during multi-step analysis (via insert_init_code in analyzer_utils.py), reads the same nonexistent /home/mok/... path."
concern: "Even if the entrypoint imports were fixed, AnalyzerVLM's executed analysis code would fail to load the data, breaking the multi-step analysis loop that is the paper's core mechanism."
resolution: "Authors: parameterize INIT_CODE with a repo-relative data directory."
cross_refs: ["hardcoded-private-paths"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Appendix A.8 (AnalyzerVLM prompts / INIT_CODE)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: data-loader-path-vs-readme
category: difference
topic: "evaluation consistency"
title: "main_sr.py uses argparse + hardcoded paths, contradicting the README run command"
severity: low
confidence: high
status: finding
file: main_sr.py
line_start: 64
line_end: 66
quote: |
  if __name__ == '__main__':    
      parser = argparse.ArgumentParser(description="gpr kernel selection")
      parser.add_argument('-d','--data', nargs='+', type=int, help='dataset', default=[1,2,3,4,5,6,7,8,9,10])
claim: "The README instructs `python3 main_sr.py experiment/function=nguyen/nguyen1` (Hydra/OmegaConf-style override syntax), but main_sr.py actually parses `--data <int>` integer indices into a hardcoded `datasets` list of private absolute paths (lines 107-136); no Hydra config is used."
concern: "A user following the README's documented command cannot select a dataset; the invocation interface disagrees with the running code."
resolution: "Authors: align the README run command with the actual argparse interface (and fix the data paths)."
cross_refs: ["hardcoded-private-paths", "missing-sr-dataset-and-hybrid"]
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "README 'Running the Code' (Symbolic Regression)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: noise-injection-not-described
category: difference
topic: "evaluation consistency / paper omission"
title: "Data loader and run script support label-noise injection not described for Table 1"
severity: low
confidence: medium
status: finding
file: main_gp.sh
line_start: 5
line_end: 11
quote: |
  # python3 main_gp.py --noise 0.03 --data 2 --set_se_const True --model gpt-4o-mini
  # # python3 main_gp.py --noise 0 --data 3 --set_se_const True --model gpt-4o-mini
  # python3 main_gp.py --noise 0.03 --data 4 --set_se_const True --model gpt-4o-mini
  python3 main_gp.py --noise 0 --data 1 --set_se_const True --model gpt-4o-mini
  # python3 main_gp.py --noise 0.03 --data 7 --set_se_const True --model gpt-4o-mini
  # python3 main_gp.py --noise 0.03 --data 8 --set_se_const True --model gpt-4o-mini
  # python3 main_gp.py --noise 0.03 --data 9 --set_se_const True --model gpt-4o-mini
claim: "load_gp_data (dataload_utils.py:78-79) adds Gaussian label noise scaled by `--noise`; the provided run script main_gp.sh contains commented invocations with --noise 0.03 / 0.05 for several datasets, while the paper's Table 1 description does not mention adding synthetic noise to the clean benchmark datasets."
concern: "It is unclear whether any Table 1 number used noised inputs; the active command uses --noise 0, but the commented variants suggest noise experiments were run and the figure/table provenance is ambiguous."
resolution: "Authors: confirm Table 1 used noise=0 for all datasets, and document the role of the --noise experiments."
cross_refs: []
check_script: _audit_code/check_imports_and_paths.py
paper_ref: "Table 1; Section 4.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No standalone methodology finding is filed. The proposed selection criterion
`VIC = α·EvaluatorVLM − BIC` is implemented in the score sorting
(`utils/main_utils.py:88-128`) consistent with the paper's Eq. 3 / A.16 form, and
the 9:1 train/validation split with a held-out forward test region
(`utils/dataload_utils.py:104-112`) is methodologically sound and temporally
forward (no shuffling). EvaluatorVLM uses GPT-4o-mini, whose validity cannot be
independently re-derived from the repo, but that is captured under the
reproducibility/missing findings rather than as a distinct leakage/validity
defect. Because the released code cannot run and computes none of the reported
metrics, I cannot establish a methodology defect in the *actual* implemented
procedure beyond what is already owned by the `missing`/`bug` findings — filing
one would be extrapolation (Rule B).

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                   |
|-------------|------------|--------------|---------------------------------------------------|
| missing     | 9          | high         | No RMSE/baseline/figure code; 3 modules absent    |
| bug         | 2          | high         | Data loader + few-shot images use dead private paths |
| difference  | 2          | low          | README run cmd ≠ argparse; undocumented noise knob |
| methodology | 0          | -            | Implemented criterion/split look sound; nothing to file |

## Top take-aways

1. **No script computes any reported table/figure number** — `rmse()`/`r2_score()`
   are defined but never called; the drivers emit only BIC/val-MSE/LOO + plots.
   Tables 1, 4, A6 and Figs 3/4/5/8/9 are untraceable. (`missing`, high/high)
2. **The two entrypoints cannot even be imported**: `parse_functools`,
   `cfgs.asmd_cfg`, and `vision_score_gpt2_0105_func` are imported but absent.
   (`missing`, high/high)
3. **None of the eight baselines (ARIMA, Prophet, GP-SE, Automatic Statistician,
   BoxLM, SGA, ICSR, LLM-SR) are implemented** in the repo, so the comparative
   "we outperform" claim is not reproducible. (`missing`, high/high)
4. **The only data loader and the EvaluatorVLM few-shot images read nonexistent
   private absolute paths** (`/home/mok/...`, `/node_data_2/...`, 63 lines); the
   shipped `./data` CSVs are never used. Fresh checkout crashes on data load.
   (`bug`, high/high)
5. **Fig. 9 correlations and the human-evaluation study have no producing code**;
   the VIC-validity claim cannot be reproduced. (`missing`, medium/high)
6. **`colorlog` is imported by both entrypoints but is in no requirements file**,
   so the documented environment setup is incomplete. (`missing`, medium/high)

## Items that genuinely look fine

- The gpss-research univariate CSVs (13 train/test pairs) are shipped in `data/`
  with the correct two-column (time, value) format matching the loader's
  expectations.
- The GP train/validation split is a forward 9:1 slice with a separate held-out
  forward test region (`utils/dataload_utils.py:104-112`) — no shuffling, no
  obvious temporal leakage.
- VIC is implemented in the score-sorting logic in the `α·score − BIC` form that
  matches Eq. 3 / A.16; the derivation in App. A.2 is internally consistent.
- `requirements.txt`/`requirements.yaml` pin the key libraries the paper names
  (GPy==1.13.2, GPy-ABCD==1.2.3, openai==1.38.0, lark==1.2.2), matching the README.

## Open questions for the authors

- Were the released files an incomplete snapshot? The three missing modules and
  63 private-path lines suggest the public repo was copied from a private working
  tree without the supporting files — can a runnable, self-contained version be
  released?
- Which exact script produced the Table 1 / Table 4 numbers, and can it be added?
- Did any Table 1 result use `--noise > 0`, given the commented noised invocations
  in `main_gp.sh`?
