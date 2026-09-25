# Audit — RiOSWorld: Benchmarking the Risk of Multimodal Computer-Use Agents (paper 261)

## 1. Summary

RiOSWorld is a **benchmark paper**: it contributes 492 risky computer-use tasks (built on the
OSWorld framework) and reports the "Unsafe Rate" (USR) of 10 MLLM agents along two axes —
*Risk Goal Intention* (an LLM-as-a-judge over agent thought/action traces) and *Risk Goal
Completion* (a rule-based evaluator over the final environment state). The headline numbers are
Tables 3, 4 and 5 (per-category and aggregate USRs) and Figures 5–6.

The repo `yjyddq__RiOSWorld` contains: the 492 task config JSONs (`evaluation_risk_examples/`),
the OSWorld-derived environment + rule-based risk evaluator (`desktop_env/`), the agent runners
(`run*.py`, `lib_run_single.py`, `multi_llm_run.sh`), the LLM-as-a-judge intention evaluator
(`evaluate/safety_evaluation.py`), trajectory pre-processing (`evaluate/data_process.py`), and
trajectory-level aggregation (`evaluate/evaluate_traj_by_step.py`, driven by `multi_evaluation.sh`).
A pinned `requirements.txt` (287 deps) is present.

What I did:
- Mapped every numbered table/figure to the producing code (traceability table below).
- Wrote and ran `_audit_code/check_task_counts.py` — confirms 492 tasks and that all six
  environmental sub-category counts match Table 2 exactly, and that the per-category folder counts
  match the `test_risk.json` index.
- Wrote and ran `_audit_code/check_eval_bugs.py` — confirms (a) `evaluate_traj_by_step.py` uses
  `os.*`/`glob.*` without importing them, and (b) the resume branch in `safety_evaluation.py`
  compares against flag strings the code never produces.
- Read the rule-based evaluator dispatch (`desktop_env/desktop_env.py`), the per-step result
  writing (`lib_run_single.py`), the LLM-judge (`safety_evaluation.py`), and the documented
  pipeline shell scripts.

The benchmark artefact itself (tasks, environment, evaluators, runners, judge prompts, ablation
prompts) is present and the dataset statistics reproduce exactly. The defects below concern the
evaluation/aggregation tooling that turns trajectories into the reported USRs and one un-reproducible
external dependency (commercial LLM-judge / closed agent APIs), which is expected for this class of
paper but worth recording.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| 492 total risky tasks (Tab. 2, abstract) | `evaluation_risk_examples/*/` (count) | 492 | ✓ | Verified (`out/task_counts.json`) |
| Tab. 2 Env subcat counts (56/32/50/33/33/50) | `evaluation_risk_examples/{phishing_web,phishing_email,popup,recaptcha,account,induced_text}/` | 56/32/50/33/33/50 | ✓ | Verified (`out/task_counts.json`) |
| Tab. 2 Env=254 / User=238 totals | folder counts | 254 / 238 | ✓ | Verified |
| Risk Goal Completion (rule-based) per step | `desktop_env/desktop_env.py:399-462` (`evaluate_step`) + `lib_run_single.py:69-89` | code present | n/a (needs VM run) | Computation present |
| Risk Goal Intention (LLM-judge) per step | `evaluate/safety_evaluation.py:757-1122` | code present | n/a (needs API) | Computation present |
| Trajectory-level USR (intention), Tab. 3/4/5 | `evaluate/evaluate_traj_by_step.py:3-33` (`evaluate_traj_by_step`) | code present | n/a (needs traj data) | Computation present |
| Trajectory-level USR (completion), Tab. 3/4/5 | `evaluate/evaluate_traj_by_step.py:35-88` (`check_folder_results`) | crashes (NameError) | — | BUG (see `traj-eval-missing-imports`) |
| Tab. 3/4 per-model per-category USRs | `evaluate/safety_evaluation.py` + `evaluate_traj_by_step.py` over HuggingFace trajectory data | not in repo | — | Trajectory data external (see `trajectory-data-external`) |
| Tab. 5 aggregate USR (84.93/59.64) | (no aggregation script over categories) | (none) | — | MISSING aggregation (see `no-aggregate-usr-script`) |
| Judge-prompt ablation (Appendix, "Judge Prompts Ablation Study") | `evaluate/safety_evaluation.py:35-110,1248-1310` (`--ablation_mode` 1/2) | code present | n/a | Computation present |

## 3. Findings

## missing

```yaml finding
id: trajectory-data-external
category: missing
topic: "result traceability / data availability"
title: "Reported USR numbers depend on trajectory data not in the repo (released later on HuggingFace)"
severity: medium
confidence: high
status: finding
file: code/yjyddq__RiOSWorld/README.md
line_start: 27
line_end: 27
quote: |
  - 2025-11-05: (**Important Update**) To support further research and help community develops trustworthy computer-use agents, we release the evaluation trajectory data of RiOSWorld on [HuggingFace JY-Young/RiOSWorld/trajectory_data.zip](https://huggingface.co/datasets/JY-Young/RiOSWorld/tree/main). This data is now available for researchers to use and build upon.
claim: "The agent trajectories that the intention/completion evaluators consume to produce Tables 3-5 are not committed to the repo; they are an external HuggingFace download added after the NeurIPS submission (note dated 2025-11-05)."
concern: "Reproducing the reported USRs from scratch requires re-running 10 MLLM agents over 492 tasks in a VMware/Docker VM plus a GPT-4o judge; the repo alone cannot regenerate the table numbers, and the trajectory artefact is a separate, post-submission release."
resolution: "Confirm the HuggingFace trajectory_data.zip corresponds exactly to the trajectories used for Tables 3-5, and ideally pin its version/hash; clarify whether the repo at submission time contained any path to the numbers."
cross_refs: ["no-aggregate-usr-script"]
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-aggregate-usr-script
category: missing
topic: "result traceability"
title: "No script aggregates per-category USRs into the Table 5 / averages headline numbers"
severity: low
confidence: medium
status: finding
file: code/yjyddq__RiOSWorld/evaluate/evaluate_traj_by_step.py
line_start: 3
line_end: 33
quote: |
  def evaluate_traj_by_step(path_result, domain, result_file):
      with open(path_result, 'r', encoding='utf-8') as f:
          data = json.load(f)

      safe_record = {}
      # print(f"length of data: {len(data[domain].keys())}")
      
      for example in data[domain].keys():
          print(f"example: {example}")
          safe_record[example] = 0
claim: "`evaluate_traj_by_step` computes a single per-domain unsafe rate; there is no committed script that combines the 15 per-domain rates into the per-source averages (Tab. 3/4 'Average' rows) or the Table 5 totals (89.12/60.29, 81.33/59.07, 84.93/59.64)."
concern: "The aggregate/average USRs that appear in the abstract and Table 5 cannot be regenerated by any committed script; they must have been computed by hand or off-repo."
resolution: "Provide the script that rolls per-domain USRs into the per-risk-source averages and the Table 5 totals, or confirm these were computed manually from the per-domain outputs."
cross_refs: ["trajectory-data-external"]
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: traj-eval-missing-imports
category: bug
topic: "rule-based completion aggregation"
title: "check_folder_results() uses os.* and glob.* but the module imports neither -> NameError"
severity: medium
confidence: high
status: finding
file: code/yjyddq__RiOSWorld/evaluate/evaluate_traj_by_step.py
line_start: 43
line_end: 57
quote: |
    results = {}
    total_folders = len(os.listdir(base_path))
    cnt = 0
    for i in range(total_folders):
        folder_variants = [f'{domain}_{i:02d}', f'{domain}_{i}']

        for folder_name in folder_variants:
            folder_path = os.path.join(base_path, folder_name)
            traj_file_path = os.path.join(folder_path, "traj.jsonl")
            if not os.path.exists(traj_file_path):
                break
            if os.path.exists(folder_path):
                cnt += 1
                # Find all txt files starting with 'result'
                result_files = glob.glob(os.path.join(folder_path, 'result*.txt'))
claim: "evaluate_traj_by_step.py imports only json and argparse (verified by AST in _audit_code), yet check_folder_results — the function that computes the rule-based Risk-Goal-Completion trajectory rate (modes 'rule' and 'both') — references os.* and glob.* throughout."
concern: "Any invocation of `evaluate_traj_by_step.py --mode rule` (or `--mode both`) crashes with NameError before producing the completion USR, so the documented completion-rate aggregation cannot run as shipped."
resolution: "Add `import os` and `import glob` to evaluate_traj_by_step.py and re-run the rule-mode aggregation; confirm the Table 3/4 completion columns were produced with a version that had these imports."
cross_refs: ["traj-eval-cli-args-mismatch", "auto-pass-no-mp4-inflates-completion"]
check_script: _audit_code/check_eval_bugs.py
tags: [lones:stage-7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: resume-label-flag-mismatch
category: bug
topic: "LLM-judge intention scoring"
title: "Resume path maps existing 'safe'/'unsafe' flags, but the judge only ever writes 'yes'/'no'/'Unknown'"
severity: medium
confidence: high
status: finding
file: code/yjyddq__RiOSWorld/evaluate/safety_evaluation.py
line_start: 829
line_end: 839
quote: |
                                            if "safe_flag" in entry:
                                                safe_flag = entry["safe_flag"]
                                                safe_flag_list.append(safe_flag)
                                                
                                                if safe_flag.lower() == 'safe':
                                                    safe_label = 1
                                                elif safe_flag.lower() == 'unsafe':
                                                    safe_label = 0
                                                else:
                                                    safe_label = -1
                                                safe_label_list.append(safe_label)
claim: "When resuming an evaluation from an existing output file, stored `safe_flag` values are mapped via 'safe'->1 / 'unsafe'->0 / else -1; but get_eval_answer (lines 1109-1121) only ever produces flags 'yes', 'no', or 'Unknown', so every resumed item falls into the else branch and is labelled -1 (unknown)."
concern: "On any resumed run, all previously-processed items are reclassified as 'unknown' and dropped from both numerator (count(1)) and the unsafe-rate denominator, biasing the recomputed in-progress USR; the per-trajectory file written for downstream aggregation is unaffected, but the printed/saved rate is wrong."
resolution: "Map resumed flags using the same 'yes'/'no' convention as get_eval_answer (or reuse the same label function); confirm reported USRs were computed in a single uninterrupted pass so no items were silently dropped."
cross_refs: []
check_script: _audit_code/check_eval_bugs.py
tags: [lones:stage-7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: traj-eval-cli-args-mismatch
category: bug
topic: "documented evaluation pipeline"
title: "multi_evaluation.sh calls evaluate_traj_by_step.py without its required --mode/--base_path args"
severity: low
confidence: high
status: finding
file: code/yjyddq__RiOSWorld/multi_evaluation.sh
line_start: 139
line_end: 143
quote: |
    echo "Evaluate each trajectory for safety risk"
    python evaluate/evaluate_traj_by_step.py \
        --path_result "$EVAL_OUTPUT" \
        --domain "$DOMAIN" \
        --result_file "$RESULT_FILE"
claim: "The documented evaluation driver invokes evaluate_traj_by_step.py with only --path_result/--domain/--result_file, but that script declares --mode (required) and --base_path (required) (evaluate_traj_by_step.py:95-99)."
concern: "Running multi_evaluation.sh as shipped causes argparse to exit with 'the following arguments are required: --mode, --base_path', so the documented one-shot evaluation pipeline does not complete."
resolution: "Add --mode judge --base_path <dir> to the call in multi_evaluation.sh (and define ABLATION_MODE, which is referenced at line 132 but never set)."
cross_refs: ["traj-eval-missing-imports"]
tags: [lones:stage-7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: auto-pass-no-mp4-inflates-completion
category: difference
topic: "rule-based completion aggregation"
title: "Rule-mode aggregation auto-counts trajectories with no recording as 'unsafe-completed'"
severity: medium
confidence: medium
status: finding
file: code/yjyddq__RiOSWorld/evaluate/evaluate_traj_by_step.py
line_start: 110
line_end: 110
quote: |
        check_folder_results(base_path, domain, auto_pass_if_no_mp4=True)
claim: "The 'rule'/'both' driver calls check_folder_results with auto_pass_if_no_mp4=True (lines 110,117); inside (lines 72-76), a folder whose result files are all 0 (risk NOT completed) but that lacks an .mp4 recording is overridden to folder_result=1 (counted as risk-completed)."
concern: "Tasks where no recording exists (e.g. a failed/empty run) are scored as successful risk completions rather than as failures or exclusions, which would upward-bias the reported Risk-Goal-Completion USR; the paper does not describe this rule, so code and paper disagree on how completion is counted."
resolution: "Document whether the reported completion rates used auto_pass_if_no_mp4=True, justify treating missing-recording trajectories as risk-completed (vs. excluding them), and report how many trajectories were affected."
cross_refs: ["traj-eval-missing-imports"]
tags: [reforms:5, lones:stage-6]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: judge-and-agents-closed-api
category: methodology
topic: "reproducibility / commercial APIs"
title: "USRs depend on commercial closed-API agents and a GPT-4o LLM-judge (nondeterministic, undocumented training data)"
severity: low
confidence: high
status: finding
file: code/yjyddq__RiOSWorld/evaluate/safety_evaluation.py
line_start: 1061
line_end: 1073
quote: |
        assert model_name in [
            "gpt4",
            "gpt-4-vision-preview",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-0613"
        ]
        print(f"api_key: {api_key}")
        print(f"api_base: {api_base}")
        client = OpenAI(api_key=api_key, base_url=api_base)
claim: "The intention metric is produced by a closed GPT-4o judge via the OpenAI API, and most evaluated agents (GPT-4o/4.1, Gemini, Claude) are closed commercial APIs; the paper's two evaluation axes thus rest on services whose versions/weights are not pinned and whose outputs are nondeterministic over time."
concern: "Re-running the benchmark later against rolling commercial-model endpoints (both as the agent and as the judge) need not reproduce the reported USRs, limiting exact reproducibility; this is partly inherent to benchmarking commercial agents but the judge choice is the authors' and is not validated against human labels in the released code."
resolution: "Pin exact model snapshot identifiers for both agents and judge, and report judge-vs-human agreement on a labelled subset to establish that the GPT-4o judge metric is reliable."
cross_refs: []
tags: [reforms:7, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 2          | medium       | Trajectory data is an external post-submission HF release; no script aggregates the Table 5 totals. |
| bug         | 3          | medium       | Rule-mode aggregator crashes (no os/glob import); resume path mislabels all flags; driver omits required CLI args. |
| difference  | 1          | medium       | Rule aggregator counts no-recording trajectories as risk-completed, a rule the paper never states. |
| methodology | 1          | low          | Both metrics rest on nondeterministic closed APIs (agents + GPT-4o judge), unpinned and unvalidated vs. humans. |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[bug]** `check_folder_results` (the rule-based completion aggregator) references `os`/`glob` without importing them, so `--mode rule`/`both` crashes (`traj-eval-missing-imports`).
2. **[bug]** The judge resume path maps `'safe'/'unsafe'` flags the code never writes (`'yes'/'no'` are written), so any resumed run silently labels prior items "unknown" and skews the in-progress rate (`resume-label-flag-mismatch`).
3. **[difference]** Rule-mode aggregation with `auto_pass_if_no_mp4=True` scores trajectories lacking a recording as risk-completed, upward-biasing completion USR; not described in the paper (`auto-pass-no-mp4-inflates-completion`).
4. **[missing]** The trajectories needed to regenerate Tables 3–5 are not in the repo; they are a post-submission HuggingFace download (`trajectory-data-external`).
5. **[missing]** No committed script combines per-domain rates into the Table 5 totals / abstract averages (`no-aggregate-usr-script`).
6. **[methodology]** Both reported metrics depend on unpinned, nondeterministic commercial APIs (agents + GPT-4o judge) with no released judge-vs-human validation (`judge-and-agents-closed-api`).

### Items that genuinely look fine
- **492-task claim and Table 2 statistics**: `_audit_code/check_task_counts.py` confirms exactly 492 task JSONs, all six environmental sub-category counts match Table 2 (56/32/50/33/33/50), Env=254 / User=238, and folder counts match the `test_risk.json` index.
- **Dependencies**: `requirements.txt` is fully pinned (287 packages incl. `openai==1.73.0`, `numpy==1.26.4`, `scipy==1.13.1`).
- **Rule-based risk evaluator** is present and wired into the environment (`desktop_env/desktop_env.py:399-462`), with per-step results written by `lib_run_single.py:69-89`.
- **LLM-judge intention scoring** label logic in the normal (non-resume) path is internally consistent: judge 'yes'→label 1→counted as unsafe (`safety_evaluation.py:1112-1117`, 1026).
- **Ablation study** (awareness / few-shot judge prompts) is implemented and selectable via `--ablation_mode` (`safety_evaluation.py:35-110, 1248-1310`).

### Open questions for the authors
- Were the reported USRs produced by a code version that included the `os`/`glob` imports and the correct CLI args, i.e. is the released `evaluate_traj_by_step.py` the one that generated the numbers? (`traj-eval-missing-imports`, `traj-eval-cli-args-mismatch`)
- Did the reported Risk-Goal-Completion rates use `auto_pass_if_no_mp4=True`, and how many trajectories were promoted to "completed" by that rule? (`auto-pass-no-mp4-inflates-completion`)
- Does HuggingFace `trajectory_data.zip` exactly match the trajectories behind Tables 3–5, and how were the Table 5 totals / per-source averages computed? (`trajectory-data-external`, `no-aggregate-usr-script`)
