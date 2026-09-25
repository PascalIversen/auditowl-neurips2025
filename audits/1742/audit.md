# Code-repository audit — Paper 1742

*"Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"* (Yue, Chen, Lu et al., NeurIPS 2025). Official author repo `LeapLabTHU/limit-of-RLVR` (project page `limit-of-rlvr.github.io`).

## 1. Summary

The paper's central method is the **pass@k coverage metric**: for each problem, sample `n ≥ k` outputs, count correct ones `c`, and estimate pass@k with the unbiased Chen-et-al. estimator `1 - C(n-c,k)/C(n,k)` (paper Eq. 2). The headline empirical claim is that base models eventually match or exceed their RLVR-trained counterparts in pass@k at large k across math, code, and visual benchmarks.

The repo has two trees: `math/` (vLLM generation + answer-grading + pass@k for the math benchmarks, the core of Fig. 2 / Tables 2–6) and `code/DeepCoder/` (a vendored verl/rllm RL-training framework for the DeepScaler/DeepCoder training-side analyses). I read the math generation/scoring pipeline (`math_eval.py`, `evaluate.py`, `pass@k.py`, `sh/eval.sh`, `eval_math_nodes.sh`, `collect_results.py`), the benchmark data files, the README, and `requirements.txt`. I ran two deterministic checks under `_audit_code/`:
- `check_passk_estimator.py` — confirms the repo estimator is byte-equivalent in value to the paper's Eq. 2 over a grid of (n, c, k) (max abs diff 8.9e-16).
- `check_context_cap.py` — surfaces a hardcoded `max_model_len=4096` in the vLLM engine that conflicts with the documented `--max_tokens 16000`.

The core pass@k machinery is methodologically sound and faithful to the paper. The findings concern (a) a generation-length cap that can silently truncate the very long-CoT samples the metric depends on, and (b) several reported result families (visual reasoning, code-generation pass@k, perplexity analysis, coverage set-overlap tables) whose computing code is not in the repo.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| pass@k unbiased estimator (Eq. 2, §A.2) | `math/pass@k.py:58-60`, `math/examples/math_eval/evaluate.py:25-31` | identical to `1-C(n-c,k)/C(n,k)` | ✓ | Verified (`_audit_code/out/passk_estimator.txt`) |
| Fig. 2 math pass@k curves (Qwen/LLaMA base vs zero-RL) | `math/eval_math_nodes.sh` → `math_eval.py`+`evaluate.py`; aggregated by `math/pass@k.py` | pipeline present (needs model weights + GPUs to run) | — (not numerically re-run) | Code present |
| Benchmark sizes (AIME24=30, MATH500=500, Minerva=272, GSM8K=1319, Olympiad=675, AMC23) | `math/examples/math_eval/data/*/test.jsonl` | 30 / 500 / 272 / 1319 / 675 / 40 | ✓ | Verified |
| Table 2 solvable/unsolvable 4-cell coverage (AIME24, MATH500) | (none — no set-intersection script over base vs RL score matrices) | — | — | MISSING (see `coverage-overlap-code-missing`) |
| Tables 5/6 "RL-solved ⊆ base-solved" set overlap | (none) | — | — | MISSING (same finding) |
| Fig. 4 (right) Visual reasoning pass@k (MathVista, MathVision; EasyR1 + Qwen2.5-VL) | (none — no multimodal eval/data/training code in repo) | — | — | MISSING (see `visual-reasoning-code-missing`) |
| Fig. 4 (left) Code-gen pass@k (HumanEval+, MBPP+, Coder-R1-Zero) | (none — only DeepCoder training framework; no HumanEval+/MBPP+ eval harness) | — | — | MISSING (see `codegen-passk-eval-missing`) |
| §4.1 Perplexity analysis PPL_m(Y\|x) (Eq. in §4.1, Figs. for PPL) | (none — no log-prob/perplexity computation script) | — | — | MISSING (see `perplexity-code-missing`) |
| §3.1 / §D.2 manual CoT-validity check on AIME24 (hardest problems) | (none — manual human inspection, no script) | — | — | MISSING (acknowledged manual; see `manual-cot-check-missing`) |
| Tables 3/4 pass@1 & pass@256 per RL algorithm / training step; Figs. 8/16/19 (entropy, KL, rollouts) | `code/DeepCoder/` training scripts produce checkpoints; `math/` evaluates them | training+eval code present (run off-cluster) | — | Code present |
| Generation length (documented `--max_tokens 16000`) | `math/examples/math_eval/math_eval.py:139` caps engine context at 4096 | effective gen budget ≤ 4096−prompt | ✗ | MISMATCH (see `context-cap-truncates-generation`) |

## 3. Findings

### missing

```yaml finding
id: visual-reasoning-code-missing
category: missing
topic: "result traceability"
title: "No code/data for the visual-reasoning pass@k results (Fig. 4 right)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  We use the EasyR1 framework [31] to train Qwen2.5-VL7B [32] on Geometry3K [33], and evaluate its visual reasoning capabilities on filtered MathVistaTestMini [34] and MathVision-TestMini [35]
claim: "The repo ships only math (`math/`) and DeepCoder code-RL training (`code/DeepCoder/`); a recursive grep for mathvista/mathvision/easyr1/geometry3k/qwen2.5-vl finds no eval, data, or training code for the visual-reasoning experiments."
concern: "The visual-reasoning pass@k curves (Fig. 4 right), a third of the paper's cross-modality evidence, cannot be reproduced or traced to any artefact in the repo."
resolution: "Authors: release (or link) the EasyR1 multimodal training config, the MathVista/MathVision filtered eval data, and the pass@k computation for the visual setting."
cross_refs: ["§3 RLVR for Visual Reasoning"]
paper_ref: "Section 'RLVR for Visual Reasoning'; Figure 4 (right)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: codegen-passk-eval-missing
category: missing
topic: "result traceability"
title: "No eval harness for code-generation pass@k (HumanEval+/MBPP+, Fig. 4 left)"
severity: medium
confidence: medium
status: finding
file: paper.pdf
quote: |
  Figure 4: Pass@k curves of base and RLVR models. (Left) Code Generation. (Right) Visual Reasoning.
claim: "The repo's `code/` tree contains only the DeepCoder/verl training framework; there is no HumanEval+/MBPP+ evaluation harness that produces the Fig. 4 (left) pass@k curves for Coder-R1-Zero. The README only credits Code-R1 (external) for code eval."
concern: "The code-generation pass@k results cannot be reproduced from this repo; the per-sample correctness used for pass@k on HumanEval+/MBPP+ is computed by an external/unshipped harness."
resolution: "Authors: include or link the exact HumanEval+/MBPP+ generation + unit-test scoring + pass@k script used for Fig. 4 (left)."
cross_refs: ["codegen-passk-eval-missing"]
paper_ref: "Figure 4 (left), §RLVR for Code Generation"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: coverage-overlap-code-missing
category: missing
topic: "result traceability"
title: "No script computes the base-vs-RL solvable-set overlap (Tables 2, 5, 6)"
severity: medium
confidence: medium
status: finding
file: paper.pdf
quote: |
  Table 2: We evaluate on AIME24 (k = 1024)
  and MATH500 (k = 128). The table reports the
  solvable/unsolvable fraction of problems falling
  into four categories.
claim: "`evaluate.py`/`pass@k.py` produce per-problem score matrices, but no script in the repo intersects the base-model and RL-model solvable sets to produce the four-cell coverage fractions of Table 2 (e.g. 63.3%/13.3%/0.0%/23.3% on AIME24) or the 'RL-solved is nearly a subset of base-solved' Tables 5/6."
concern: "The set-overlap statistics that underpin the paper's core 'reasoning paths already present in the base model' conclusion are not traceable to any computing artefact."
resolution: "Authors: provide the script that, given the two saved score jsonls, computes the four-cell coverage table and the subset/overlap statistics."
cross_refs: ["§3 Solvable-Problem Coverage Analysis", "§D.7"]
paper_ref: "Table 2; Tables 5, 6"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: perplexity-code-missing
category: missing
topic: "result traceability"
title: "No perplexity-computation code for the §4.1 perplexity analysis"
severity: medium
confidence: medium
status: finding
file: paper.pdf
quote: |
  PPLm(Y | x) = exp
   
  −1
  T
  T
  X
  t=1
  log P (yt | x, y1, . . . , yt−1)
claim: "The paper's perplexity analysis (a load-bearing argument that RL-model responses lie within the base model's distribution) requires per-token log-likelihood scoring; a recursive grep for perplexity/log_likelihood/logprob over `math/` and `code/scripts` finds no such computation."
concern: "The perplexity-based evidence for the 'reasoning paths already present in base models' claim cannot be reproduced from the repo."
resolution: "Authors: release the script that computes PPL_m(Y|x) for base/RL/distilled models on the response sets reported in §4.1."
cross_refs: ["§4.1 Reasoning Paths Already Present in Base Models"]
paper_ref: "Section 4.1, Perplexity Analysis"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: manual-cot-check-missing
category: missing
topic: "result traceability"
title: "Manual CoT-validity check (§3.1) has no code; relies on human inspection"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  To address this, we
  manually check the correctness of CoT for a subset of model outputs as detailed in Section 3.1.
claim: "The paper mitigates the 'lucky-guess' threat to pass@k by manually inspecting CoTs of the hardest problems; this is an off-code human procedure with no script, selection criterion file, or annotation artefact in the repo."
concern: "The reader cannot independently verify which outputs were inspected or how guess-only correct answers were excluded, so the guessing-correction step is unauditable."
resolution: "Authors: release the inspected-sample IDs and annotation outcomes (even as a CSV) so the CoT-validity claim can be checked."
cross_refs: ["§3.1", "§D.2"]
paper_ref: "Section 3.1 Random Guessing Issue; Section D.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: eval-deps-unlisted
category: missing
topic: "dependencies / reproducibility"
title: "Math-eval imports (wandb, pandas, matplotlib) not in requirements.txt"
severity: low
confidence: high
status: finding
file: code/LeapLabTHU__limit-of-RLVR/math/requirements.txt
line_start: 1
line_end: 17
quote: |
  accelerate
  codetiming
  datasets
  dill
  hydra-core==1.4.0.dev1
  omegaconf==2.4.0.dev3
  numpy
  pybind11
  ray[default]==2.10.0
  tensordict
  transformers<4.48
  vllm<=0.6.3
  peft
  liger-kernel
  word2number
  math-verify[antlr4_11_0]==0.6.0
claim: "`collect_results.py` imports `pandas`, `matplotlib`, and `wandb` (unconditionally) and `eval.sh` runs `math_eval.py` which transitively needs `pebble`/`jieba`/`sympy`/`antlr4`; none of pandas/matplotlib/wandb appear in requirements.txt (pebble/jieba/sympy/antlr4 are installed ad hoc inside `math_eval.sh`)."
concern: "A clean `pip install -e .` does not yield a runnable evaluation+aggregation environment; the result-collection step fails on import without manually adding the missing packages."
resolution: "Authors: add pandas, matplotlib, wandb (and the math_eval.sh-installed pebble/jieba/sympy/antlr4 pins) to requirements.txt, or document them."
cross_refs: []
paper_ref: "N/A (repo packaging)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: context-cap-truncates-generation
category: bug
topic: "generation length"
title: "vLLM max_model_len hardcoded to 4096, contradicting documented max_tokens=16000"
severity: medium
confidence: medium
status: finding
file: code/LeapLabTHU__limit-of-RLVR/math/examples/math_eval/math_eval.py
line_start: 131
line_end: 140
quote: |
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            seed=args.seed,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
claim: "The vLLM engine's total context length (prompt + generation) is hardcoded to 4096, while the script accepts `--max_tokens 16000` (README/eval_math_nodes.sh) and forwards it as `SamplingParams(max_tokens=args.max_tokens_per_call)` at line 307; vLLM can only emit up to `4096 - prompt_len` new tokens regardless of the requested 16000."
concern: "Long reasoning chains on hard benchmarks (AIME/Olympiad at large k) are silently truncated to <4096 tokens instead of the documented 16000, which can suppress exactly the long-CoT correct samples that the large-k coverage comparison depends on."
resolution: "Authors: confirm the max_model_len used for the reported runs; if 4096 was actually used, reconcile it with the README's `--max_tokens 16000`, or expose max_model_len as a CLI argument tied to max_tokens."
cross_refs: []
check_script: _audit_code/check_context_cap.py
paper_ref: "README eval_math_nodes.sh (--max_tokens 16000)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: special-benchmark-comment-stale
category: difference
topic: "evaluation consistency"
title: "eval.sh comment claims aime24/amc23 use n_sampling=8, code uses the passed N_SAMPLING"
severity: low
confidence: high
status: finding
file: code/LeapLabTHU__limit-of-RLVR/math/examples/math_eval/sh/eval.sh
line_start: 64
line_end: 79
quote: |
  # Run special benchmarks (aime24, amc23) with n_sampling=8
  if [ ${#SPECIAL_BENCHMARKS[@]} -gt 0 ]; then
      SPECIAL_BENCHMARKS_STR=$(IFS=,; echo "${SPECIAL_BENCHMARKS[*]}")
      TOKENIZERS_PARALLELISM=false \
      python -u math_eval.py \
          --model_name_or_path ${MODEL_NAME_OR_PATH} \
          --data_name ${SPECIAL_BENCHMARKS_STR} \
          --output_dir ${OUTPUT_DIR} \
          --split ${SPLIT} \
          --prompt_type ${PROMPT_TYPE} \
          --num_test_sample ${NUM_TEST_SAMPLE} \
          --max_tokens_per_call ${max_tokens} \
          --seed ${seed} \
          --temperature ${temperature} \
          --n_sampling ${N_SAMPLING} \
          --top_p ${top_p} \
claim: "The comment says special benchmarks (aime24/amc23) run with n_sampling=8, but the actual invocation passes `--n_sampling ${N_SAMPLING}`, i.e. whatever was given on the command line (32, 256, 512, …) — identical to the regular branch."
concern: "The stale comment could mislead a reproducer into thinking aime24/amc23 are forced to a different sampling count than other benchmarks; the code in fact does not differ on this axis (only the temperature=0 special-casing differs)."
resolution: "Authors: delete or correct the `n_sampling=8` comment; confirm the per-benchmark n values (paper uses n=1024 for AIME24/AMC23) were set via the CLI."
cross_refs: []
paper_ref: "§A.2 (n=1024 for AMC23/AIME24)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

No methodology findings. The core pass@k estimator matches the paper's Eq. 2 (verified deterministically); the evaluation pools samples per problem (`n*seed_max`) consistently with the paper's stated n; benchmarks are standard held-out math test sets with no train/test-split or leakage concern in the evaluation pipeline (the audited repo evaluates released checkpoints, it does not fit models on the eval sets). The "lucky-guess" threat to pass@k at large k is acknowledged by the paper and partially mitigated by manual CoT inspection (tracked as `manual-cot-check-missing`). N/A topics: data-splitting/sample-independence/temporal-integrity (this is an inference-only pass@k evaluation over fixed public benchmarks, no model fitting on the eval data); pretraining-contamination is intrinsic to the studied base/RL checkpoints and is the paper's subject of study, not an audit defect of this repo.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 6          | medium       | Visual, code-gen, perplexity, coverage-overlap result code absent; eval deps unlisted |
| bug         | 1          | medium       | vLLM max_model_len=4096 contradicts documented max_tokens=16000        |
| difference  | 1          | low          | Stale `n_sampling=8` comment in eval.sh                                |
| methodology | 0          | -            | Core pass@k estimator verified faithful; no leakage in eval pipeline   |

### Top take-aways (≤6)

1. **[bug, medium/medium]** `max_model_len=4096` is hardcoded in the vLLM engine (`math_eval.py:139`) while the README/paper invoke `--max_tokens 16000`; long CoTs are capped at ≤4096−prompt tokens, potentially truncating the long-reasoning samples the large-k coverage claim depends on. (`context-cap-truncates-generation`)
2. **[missing, medium/high]** No code/data for the visual-reasoning pass@k results (Fig. 4 right). (`visual-reasoning-code-missing`)
3. **[missing, medium/medium]** No coverage set-overlap script for Tables 2/5/6 — the "RL-solved ⊆ base-solved" evidence is not traceable. (`coverage-overlap-code-missing`)
4. **[missing, medium/medium]** No perplexity-computation code for the §4.1 perplexity argument. (`perplexity-code-missing`)
5. **[missing, medium/medium]** No HumanEval+/MBPP+ eval harness for the code-generation pass@k curves (Fig. 4 left). (`codegen-passk-eval-missing`)
6. **[missing, low/high]** Manual CoT-validity check (§3.1) has no released artefact (inspected IDs / annotations). (`manual-cot-check-missing`)

### Items that genuinely look fine

- The pass@k estimator in `pass@k.py:58-60` and `evaluate.py:25-31` is byte-equivalent in value to the paper's unbiased Eq. 2 (`_audit_code/out/passk_estimator.txt`, max diff 8.9e-16).
- Benchmark test-set sizes on disk match the paper (AIME24=30, MATH500=500, Minerva=272, GSM8K=1319, Olympiad=675; AMC23=40).
- The math generation+grading pipeline (`math_eval.py` → `evaluate.py` via `math_equal_process`) and per-seed pooling (`pass@k.py` `N=n*seed_max`) are internally consistent and match the documented run names / n-values.
- Sampling diversity is correctly handled (distinct vLLM seeds per run; sequential RNG within a run), consistent with the README's claim.

### Open questions for the authors

- Was `max_model_len` actually 4096 for the reported numbers, or was this value edited locally? If 4096, were any AIME/Olympiad samples truncated, and does that affect the large-k crossover? (`context-cap-truncates-generation`)
- Can you point to (or release) the off-repo scripts for visual-reasoning eval, code-gen pass@k, perplexity, and the coverage set-overlap tables, or confirm they were computed outside this repository?
