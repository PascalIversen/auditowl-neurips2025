# Code-repository audit — EG-CFG: Execution-Guided Line-by-Line Code Generation (paper 4468)

## 1. Summary

EG-CFG is an **inference-time** decoding method (no training): for each programming
task it samples candidate line continuations with beam search, executes them against
the task's test cases to extract execution traces, injects those traces into the prompt,
and uses Classifier-Free Guidance (CFG, Eq. 13) to interpolate between the prior and the
trace-conditioned distribution at every token. It runs many agents in parallel (a grid of
decoding configs × CFG strengths γ) and accepts the first candidate that passes the test
cases. Headline claims are SOTA accuracy on MBPP (96.6%), MBPP-ET (73.0%), HumanEval
(99.4%), HumanEval-ET (89.02%), DS-1000 (69.9%), CodeContests (60.6%).

The repository (`code/boazlavon__eg_cfg/`) contains the full method: the EG-CFG inference
loop (`eg_cfg/`), the trace dumper (`traces_dumper/`), dataset loaders, evaluation code,
configs, `environment.yml`, a thorough README, and — critically for this re-audit — four
**now-populated git submodules** under `submodules/`: a modified HuggingFace `transformers`
fork, a modified `trepan-xpy` debugger fork, plus `trepan` and `xpython`.

What I did: read the paper (`paper.pdf`/`paper_text.txt`), README, and all core
`eg_cfg/` modules; inspected the four submodules' contents and git state
(`git submodule status`); traced the test-case data flow from dataset loading →
guidance-signal generation → final evaluation; and wrote
`_audit_code/check_submodule_and_tests.py` (output in `_audit_code/out/checks.txt`) which
deterministically confirms (a) the `transformers` fork is populated and carries the
`_eg_cfg_sample` method with EG-CFG modification markers and the `apply_guidance` call,
(b) all four submodules are non-empty (1754 / 206 / 51 / 43 files), (c) for MBPP the
guidance test cases and the evaluation test cases are the identical `problem["test_list"]`,
and (d) the `apply_guidance` formula is algebraically equivalent to paper Eq. 13.

**Re-audit headline:** the previously-empty submodules are now fully present. The required
modified `transformers/generation/utils.py` (containing `_eg_cfg_sample`, the CFG
interpolation, and the EG-CFG modification blocks) and the modified `trepan-xpy` trace
emitter are **present in the released artefact**. Any prior conclusion that the required
modified forks were absent is **reversed** by this clone.

## 2. Result-traceability table

This is an empirical-method paper; "computed value" means the script that runs the method
and produces per-task pass/fail, from which an accuracy is obtained. The repo ships the
method end-to-end but does **not** ship an aggregation script that divides passed tasks by
total to emit the headline percentages, nor the full per-task run outputs (`trials/` is
git-ignored), nor any RSR computation.

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Method itself (CFG Eq.13 decoding) | `submodules/transformers/generation/utils.py:3133-3325`, `eg_cfg/probs_utils.py:38-47` | yes (impl present) | implements Eq.13 | Verified present |
| Execution-trace signal (Eq.9-11) | `traces_dumper/`, `submodules/trepan-xpy/`, `eg_cfg/code_generation_adapter.py` | yes | — | Verified present |
| Per-task pass/fail → accuracy | `eg_cfg/eg_cfg_session_manager.py:587-620`, `eg_cfg/eval_et.py` | per-task only | N/A | Present (per-task) |
| Table 1 MBPP 96.6% / 83.2% | (no aggregation script) | — | — | MISSING aggregation (see `missing-accuracy-aggregation`) |
| Table 1/2/3/4 RSR columns | (none in repo) | — | — | MISSING (see `missing-rsr-script`) |
| Tables 1-4 full per-task run JSONs | `trials/**` is `.gitignore`d; only 9-task `analysis/mbpp_analysis/` subset shipped | — | — | Not shipped (expected for method paper; reproducible by re-run) |
| Table 5 runtime Mean±SD | (no script computes/aggregates these) | — | — | MISSING (see `missing-runtime-stats`) |
| Table 6 ablations (no beam / γ=1 / minimal trace) | configs toggle these (`minimal_trace`, beam off, γ), but no ablation driver/aggregation | partial (toggles exist) | — | Partial (see `missing-ablation-driver`) |
| README "9 of 17 MBPP tasks have invalid tests" | `analysis/mbpp_analysis/` (9 task folders + summary) | yes (manual analysis) | matches README | Verified |

## 3. Findings

## missing

```yaml finding
id: missing-accuracy-aggregation
category: missing
topic: "result traceability"
title: "No script aggregates per-task pass/fail into the headline accuracy numbers"
severity: medium
confidence: high
status: finding
file: code/boazlavon__eg_cfg/analysis/scripts/print_passed.py
line_start: 10
line_end: 23
quote: |
  def find_passed_task_ids(json_dir):
      json_dir = Path(json_dir)
      for path in json_dir.rglob("*.json"):
          match = TASK_ID_RE.search(path.name)
          if not match:
              continue  # skip files that don't match the pattern
          task_id = match.group(1)
          try:
              with open(path) as f:
                  data = json.load(f)
                  if isinstance(data, dict) and data.get("passed") is True:
                      print(task_id)
          except Exception as e:
              pass
claim: "The only result-reading utility prints the IDs of passed tasks; no committed script divides #passed by the benchmark total (500, 164, 1000, ...) to produce the reported accuracies (96.6%, 99.4%, etc.)."
concern: "The headline accuracy values in Tables 1-4 cannot be regenerated by running a committed script; a reviewer must count passed task IDs and divide by the benchmark size by hand."
resolution: "Authors: please add the script that converts the per-task `passed` flags into the reported accuracy (and over which trial directory it is run)."
cross_refs: ["missing-full-trials", "missing-rsr-script"]
check_script: _audit_code/check_submodule_and_tests.py
paper_ref: "Tables 1-4 (Acc. % columns)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-rsr-script
category: missing
topic: "result traceability"
title: "RSR (Relative Success Rate) column reported in every table but never computed in repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "RSR: Relative Success Rate = Accuracy gain over baseline normalized to full success."
claim: "Every results table reports an RSR (%) column, but a repo-wide search (`grep -rni 'rsr|relative'` over all .py, excluding submodules) finds no code that computes RSR."
concern: "The RSR values (e.g. 80.23, 94.04, 32.29) cannot be traced to any committed computation; they can only be re-derived if one already knows the formula and the baseline accuracies."
resolution: "Authors: provide the script computing RSR from EG-CFG accuracy and the baseline accuracy, including which baseline run each RSR uses."
cross_refs: ["missing-accuracy-aggregation"]
check_script: _audit_code/check_submodule_and_tests.py
paper_ref: "Tables 1-4 (RSR % columns); README line 102"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-runtime-stats
category: missing
topic: "result traceability"
title: "Table 5 per-task runtime Mean ± SD has no computing/aggregation script"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  "Table 5: Per-task runtime statistics (in seconds) for each model and method on MBPP."
claim: "Per-task JSONs record `stats.duration`, but no committed script reads those durations across tasks to compute the Mean ± SD reported in Table 5 (123.23 ± 344.91, 271.37 ± 271.45, etc.)."
concern: "The runtime comparison (a fairness argument central to the 'wall-clock time is the fair metric' claim) is not reproducible from a committed script."
resolution: "Authors: add the script that aggregates per-task `duration` into the Table 5 statistics, for both EG-CFG and the baselines."
cross_refs: []
paper_ref: "Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-ablation-driver
category: missing
topic: "ablations"
title: "Ablation toggles exist but no driver reproduces Table 6 rows"
severity: low
confidence: medium
status: question
file: code/boazlavon__eg_cfg/configs/session_config.local.json
line_start: 1
line_end: 9
quote: |
  {
    "gammas": [0.0, 0.5, 1.0, 3.0],
    "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deployment_type": "local",
    "dataset": "mbpp",
    "is_prod": true,
    "results_dir": "trials/local_results",
    "use_global_cache": true,
    "debug_mode": true
  }
claim: "The three Table 6 ablations map to togglable settings (γ=1 via `gammas`, `minimal_trace` in the README config table, and disabling beam search), but no committed config/driver fixes these to the exact ablation settings nor aggregates the four resulting accuracies."
concern: "Table 6 ablation numbers (58.2, 75.2, 76.4 ...) are not reproducible by running a committed, named configuration; the reader must reconstruct each ablation's config."
resolution: "Authors: please provide the exact configs (or a driver) used for the 'no beam search', 'γ=1', and 'minimal trace' ablation rows."
cross_refs: ["missing-accuracy-aggregation"]
paper_ref: "Table 6"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

```yaml finding
id: missing-full-trials
category: missing
topic: "repository completeness"
title: "Full per-task run outputs not shipped (only a 9-task analysis subset)"
severity: low
confidence: high
status: finding
file: code/boazlavon__eg_cfg/.gitignore
line_start: 1
line_end: 4
quote: |
  **.pyc
  output/**
  results*/**
  trials/**
claim: "`trials/`, `results*/`, and `output/` are git-ignored, so none of the per-task result JSONs that underlie Tables 1-5 are in the repo; only `analysis/mbpp_analysis/` (9 tasks) is committed."
concern: "Because the headline numbers are produced by selecting, across ~192 (config × γ) attempts per task, the first that passes the test cases, the absence of the run logs means the exact selection that yielded each reported accuracy cannot be independently inspected without a full (and expensive) re-run."
resolution: "Authors: consider releasing the per-task solved-task JSONs (or the `solved_tasks/` caches) for at least one benchmark so the reported accuracy can be recomputed directly."
cross_refs: ["methodology-best-of-n-selection", "missing-accuracy-aggregation"]
paper_ref: "Tables 1-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No findings. The wiring I traced (the CFG hook in the `transformers` fork, the
guidance-strength dispatch by γ, the local-vs-endpoint paths, the stopping criteria, and
the test-execution harness) is internally consistent. The `environment.yml` pins
`transformers==4.49.0`, `trepanxpy==1.1.1`, `x-python==1.5.1` from PyPI and then
`redirect_env_to_submodules.py` symlinks the local modified forks over them — a coherent
(if unusual) setup, not a bug.

## difference

No findings rising to a reportable level. The implemented procedure matches the paper's
description (multi-agent grid, CFG Eq.13, AST executable extraction, trace injection). The
one notable nuance — that reported accuracy is best-of-many-configs with selection by test
passing — is described in the paper (multi-agent controller, Appendix C Algorithm 1: "return
first valid solution and terminate other agents"), so it is owned by `methodology` below
rather than filed as an undisclosed difference.

## methodology

```yaml finding
id: methodology-best-of-n-selection
category: methodology
topic: "evaluation protocol / selection on test labels"
title: "Reported accuracy is best-of-~192 candidates selected by passing the evaluation test cases"
severity: high
confidence: high
status: finding
file: code/boazlavon__eg_cfg/eg_cfg/eg_cfg_session_manager.py
line_start: 588
line_end: 618
quote: |
            else:
                solution_results = run_tests(solution, test_cases_to_eval, io_flag)
                solution_entry = format_results(
                    solution, solution_results, general_error, tb
                )
claim: "For each task the framework sweeps a grid of decoding configs (paper: t∈{6 values}, d∈{4}, s=3, two prompt templates = 48 configs) × γ∈{0,0.5,1,3}, i.e. up to ~192 generations, and `solve_single_problem` accepts the first whose `passed` flag is True, where `passed` is computed by `run_tests(solution, test_cases_to_eval, ...)` against the *evaluation* test cases. The reported metric is therefore pass@(many candidates) with the selection oracle being the evaluation tests themselves, not pass@1."
concern: "Because the candidate that is reported is the first to pass the evaluation tests, the held-out test labels are used to choose which of ~192 generations to keep; this inflates accuracy relative to a single-shot or test-blind-selection protocol and is the mechanism behind the reported gains, so the headline SOTA numbers must be read as 'best of a large candidate budget chosen by the test cases', not single-attempt accuracy."
resolution: "Authors: report pass@1 (single fixed config, no test-based selection) alongside the best-of-grid number, and clarify in the tables that the reported accuracy uses test-case-passing as the cross-config selection criterion; state the exact candidate budget per benchmark."
cross_refs: ["methodology-mbpp-guidance-equals-eval", "missing-full-trials"]
check_script: _audit_code/check_submodule_and_tests.py
paper_ref: "Sec 3.4 / Appendix C Alg.1; Tables 1-4"
tags: [reforms:6, whalen:pitfall-2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: methodology-mbpp-guidance-equals-eval
category: methodology
topic: "evaluation protocol / MBPP test reuse"
title: "On MBPP the execution-guidance tests are byte-identical to the evaluation tests"
severity: medium
confidence: high
status: finding
file: code/boazlavon__eg_cfg/eg_cfg/eg_cfg_session_manager.py
line_start: 235
line_end: 235
quote: |
        test_cases_to_prompt = problem["test_list"]
claim: "For MBPP, the tests fed into the prompt and executed to build the execution-guidance signal are `problem[\"test_list\"]` (line 235), and the tests used for final scoring are `eval_problem[\"test_list\"]` (line 495) — the same MBPP 3-test list. So on MBPP the model is steered, token by token, by the exact unit tests it is then graded on."
concern: "The whole method optimizes generation toward passing the supplied tests; when those tests are also the evaluation tests (MBPP, and the public tests used as the within-run stop signal generally), the headline MBPP numbers reflect fitting-to-the-eval-tests rather than generalization — the paper itself flags MBPP/MBPP-ET tests as the non-hidden case, so the concern is the magnitude, not concealment."
resolution: "Authors: confirm that the MBPP accuracy is computed on the same 3 tests used for guidance, and (ideally) report MBPP performance on a held-out test set disjoint from the guidance tests, as already done for HumanEval/CodeContests via `eval_test_list`."
cross_refs: ["methodology-best-of-n-selection"]
check_script: _audit_code/check_submodule_and_tests.py
paper_ref: "Sec 4 'Evaluation Benchmark'; Table 1"
tags: [leakage:L1.1, reforms:6]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | medium       | No aggregation/RSR/runtime/ablation-driver scripts; full per-task run logs not shipped. Method code itself is complete. |
| bug         | 0          | -            | Wiring is internally consistent; modified forks present and coherent. |
| difference  | 0          | -            | Code matches the paper's described procedure. |
| methodology | 2          | high         | Reported accuracy = best-of-~192 candidates selected by passing eval tests; on MBPP guidance tests == eval tests. |

## 5. Closing lists

**Top take-aways (≤6, ranked):**
1. [methodology] Headline SOTA accuracies are best-of-~192 (config × γ) candidates, with the
   first candidate that *passes the evaluation tests* being the one reported — test-label
   selection, not pass@1 (`methodology-best-of-n-selection`, high/high).
2. [methodology] On MBPP, the execution-guidance tests are identical to the evaluation tests,
   so the MBPP numbers measure fitting to the graded tests (`methodology-mbpp-guidance-equals-eval`,
   medium/high).
3. [missing] No committed script turns per-task pass/fail into the reported accuracy
   percentages (`missing-accuracy-aggregation`, medium/high).
4. [missing] RSR columns appear in every table but no code computes RSR
   (`missing-rsr-script`, medium/high).
5. [missing] Full per-task run outputs are git-ignored; only a 9-task analysis subset is
   shipped, so the exact selected solutions behind the tables are not inspectable
   (`missing-full-trials`, low/high).
6. [missing] Table 5 runtime stats and Table 6 ablation rows lack committed
   aggregation/driver scripts (`missing-runtime-stats`, `missing-ablation-driver`, low).

**Items that genuinely look fine (actively checked):**
- All four git submodules are populated; the modified `transformers` fork
  (`submodules/transformers/generation/utils.py`) contains `_eg_cfg_sample` plus the
  EG-CFG modification blocks and the `apply_guidance` call — the required modified fork is
  present (reverses the prior empty-submodule conclusion).
- The modified `trepan-xpy` debugger fork is present and is what `traces_dumper/runner.py`
  invokes (`["trepan-xpy", script_path]`).
- `apply_guidance` (`eg_cfg/probs_utils.py:38-47`) is algebraically equivalent to paper
  Eq. 13 (verified numerically in `_audit_code/out/checks.txt`).
- Test-case separation is correct where the paper claims hidden tests: HumanEval and
  CodeContests guide on public tests (`test_list`) and evaluate on held-out tests
  (`eval_test_list`); CodeContests evaluates on `private_tests` + `generated_tests`.
- Dependencies are fully pinned (`environment.yml`, python 3.9.19, exact versions);
  README gives concrete reproduce commands; data deps (`data/humaneval/humaneval.json`,
  MBPP prompt JSONs) are present.
- README's "9 of 17 MBPP tasks have flawed benchmark tests" is backed by the committed
  `analysis/mbpp_analysis/` evidence.

**Open questions for the authors:**
- What is the per-benchmark candidate budget actually used for each reported number, and
  what is pass@1 under a single fixed config without test-based cross-config selection?
- Which trial directory and aggregation procedure produced each table's accuracy and RSR,
  and can the per-task solved-task caches be released?
- For MBPP, is there a held-out evaluation disjoint from the guidance tests?
