# Code audit — EVOREFUSE (paper 54)

## 1. Summary

EVOREFUSE is a **dataset-construction / benchmarking** paper. Its contributions are
(i) an evolutionary prompt-optimization algorithm that mutates/recombines seed
instructions and scores them with an ELBO-based fitness function, and (ii) two
released datasets — EVOREFUSE-TEST (582 pseudo-malicious instructions) and
EVOREFUSE-ALIGN (3,000 SFT/DPO pairs). The repo
(`code/FishT0ucher__EVOREFUSE/`, commit `6f9d69c`) contains the released
datasets (`datasets/evo_test.jsonl`, `datasets/evo_align.json`), the framework
script and two ablation variants, LoRA fine-tuning YAMLs, per-model generation
scripts, metric scripts (PRR, CRR, lexical diversity, log-prob, LongPPL), the
tactic-mining pipeline, and gradient/information-flow visualization scripts.

What I did: read every `.py` and config in the repo and the methodology/results
sections of `paper.pdf`. I ran one deterministic check
(`_audit_code/check_undefined_and_paths.py`) that scans every Python file for
(a) `openai.generate(...)` calls vs. any definition/import of an `openai`
symbol, (b) hardcoded placeholder strings (`"path"`, `"file"`, `"file.jsonl"`),
and (c) presence of dependency-specification files. Output is in
`_audit_code/out/checks.json`. I did **not** execute any repo script (no model
weights, no API keys, no real input paths).

The dominant finding is that the repo is **not runnable as shipped**: the core
generation/evaluation pipeline calls an undefined `openai.generate(...)` symbol
in 7 files, every input/output/model path is a literal placeholder (`"path"` /
`"file"`), and there is no dependency specification. Beyond runnability, the
released framework script has a logic bug that silently skips the safety check
on recombined instructions, and several metric scripts do not match the
quantities the paper reports. Because the headline numbers depend on a closed
commercial GPT-4O loop and on data files that are not wired up, none of the
reported tables can be reproduced from the repo end-to-end.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — PRR refusal rates, 9 LLMs × 9 benchmarks | `metric/prr.py` computes PRR over a `*.jsonl` of responses; responses produced by `generation/*.py` | not computable (placeholder paths, undefined `openai`) | — | NOT RUNNABLE (see undefined-openai-symbol, placeholder-paths) |
| Table 2 — MSTTR/HDD/MTLD diversity | `metric/lexical.py` (`ld.msttr` only; HDD/MTLD commented out) | not computable; only MSTTR is wired | partial | PARTIAL (see lexical-only-msttr) |
| Table 2 — Log-Prob(y\|x) | `metric/prob.py` | not computable (placeholder paths) | — | NOT RUNNABLE |
| Table 2 — LongPPL(y\|x) | `metric/longppl.py` | computes thresholded avg token log-prob of the **instruction**, not LongPPL of the response | likely ✗ | MISMATCH (see longppl-not-longppl) |
| Table 2 — Safety (human annotation) | (none — manual annotation by 3 annotators) | — | — | N/A (manual; not code) |
| Table 3 — SFT/DPO mitigation (PRR/CRR) | `finetune/sft.yaml`, `finetune/dpo.yaml` (LlamaFactory); CRR via `metric/crr.py` | not computable (placeholder dataset/model, undefined `openai`) | — | NOT RUNNABLE |
| Table 4 / Fig. 1(left) — gradient & information-flow attribution | `visual/gradient.py`, `visual/information_flow.py` | single hardcoded example string; placeholder model path | qualitative only | PARTIAL (hardcoded single example) |
| Fig. 1(right) — word clouds of high-attribution tokens | `analysis/cloud.py` (+ tactic mining) | not computable (placeholder paths) | — | NOT RUNNABLE |
| Fig. 2 — fitness/PRR trajectories, ablations | `framework/evorefuse.py`, `evorefuse_wofitness.py`, `evorefuse_worecombine.py` | not computable (placeholders, undefined `openai`) | — | NOT RUNNABLE |
| EVOREFUSE-TEST = 582 instructions | `datasets/evo_test.jsonl` | 582 lines | ✓ | Verified |
| EVOREFUSE-ALIGN = 3,000 instances | `datasets/evo_align.json` | 3,085 entries | ≈ (85 extra) | Minor mismatch |

## 3. Findings

## missing

```yaml finding
id: missing-requirements
category: missing
topic: "dependencies / reproducibility"
title: "No dependency specification file; README lists an impossible numpy version"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 13
line_end: 20
quote: |
  # Python Version
  python == 3.8.18

  # Core Dependencies
  numpy == 1.12.5
  transformers == 4.43.1
claim: "The repo ships no requirements.txt / environment.yml / setup.py / pyproject.toml; the only dependency hint is the README, which pins numpy==1.12.5 — a version that does not exist (numpy never released a 1.12.5; 1.12.x stopped at 1.12.1). The scripts additionally import torch, pandas, lexical_diversity, sentence-transformers and openai, none of which are listed."
concern: "The environment cannot be rebuilt from the repo: there is no machine-readable dependency list and the one human-readable version pin is invalid, so reproducing any reported number is blocked at install time."
resolution: "Provide a requirements.txt / environment.yml pinning all actually-used packages (torch, transformers, pandas, lexical_diversity, sentence-transformers, openai client, etc.) with valid versions."
cross_refs: []
check_script: _audit_code/check_undefined_and_paths.py
paper_ref: "README Requirements section"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: placeholder-paths
category: missing
topic: "runnability / data wiring"
title: "Every input/output/model path across the pipeline is a literal placeholder"
severity: high
confidence: high
status: finding
file: framework/evorefuse.py
line_start: 66
line_end: 68
quote: |
  model_path = "path"
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(model_path)
claim: "Model paths, classifier paths, input data paths, and output file paths are hardcoded to placeholder strings ('path', 'file', 'file.jsonl') throughout the repo (23 files flagged by the scan, including framework/, metric/, generation/, mining/, analysis/, visual/, and finetune/*.yaml dataset/model fields)."
concern: "No script can locate its inputs, model weights, or write its outputs as shipped, so the generation/evaluation/fine-tuning pipeline cannot be executed to reproduce any table without the author re-supplying every path by hand."
resolution: "Replace placeholders with real relative paths (or CLI args/config) and document which dataset and model each script consumes; e.g. wire metric scripts to datasets/evo_test.jsonl."
cross_refs: ["undefined-openai-symbol"]
check_script: _audit_code/check_undefined_and_paths.py
paper_ref: "Tables 1-3, Figures 1-2"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: prr-crr-driver-not-self-contained
category: missing
topic: "result traceability"
title: "Headline refusal-rate tables depend on a closed GPT-4O loop with no reproducible driver"
severity: medium
confidence: medium
status: finding
file: metric/crr.py
line_start: 22
line_end: 30
quote: |
  for num, line in enumerate(input_data.readlines()):
      one_data = json.loads(line)
      instruction = one_data["instruction"]
      response = one_data["response"]
      benchmark = one_data["benchmark"]

      prompt = system_prefix + response
      results = openai.generate(prompt)
claim: "CRR (Table 1/Table 3) is computed by sending each response to an undefined `openai.generate` LLM-as-judge; the response files it consumes (`one_data['response']`, `one_data['benchmark']`) are not produced by any wired-up script in the repo and are not shipped, and the judge is a non-deterministic commercial API."
concern: "The CRR/PRR numbers cannot be regenerated: the intermediate response files are absent and the judge is a closed, non-deterministic API with no version pin, so Table 1 and Table 3 are not traceable to runnable code."
resolution: "Ship the intermediate response JSONL files (or a script that produces them with real paths), pin the judge model/version, and document determinism settings."
cross_refs: ["undefined-openai-symbol", "placeholder-paths"]
paper_ref: "Table 1, Table 3"
tags: [reforms:2, reforms:8]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: undefined-openai-symbol
category: bug
topic: "runnability"
title: "`openai.generate(...)` is called in 7 files but `openai` is never defined or imported"
severity: high
confidence: high
status: finding
file: framework/evorefuse.py
line_start: 285
line_end: 285
quote: |
                mut_response = openai.generate(prompt)
claim: "The mutation, recombination, judge, CRR, tactic-mining and several generation scripts all invoke `openai.generate(...)`, but no file imports or defines an `openai` symbol exposing a `.generate` method (the scan finds 0 of 7 such files import/define it; the real OpenAI SDK exposes `OpenAI().chat.completions.create`, not `openai.generate`). The one file that does `from openai import OpenAI` (generation/other_llm.py) then calls an undefined `client` variable."
concern: "Every script that drives mutation/recombination/judging/CRR raises NameError on the first GPT-4O call, so the core EVOREFUSE generation loop and the CRR metric do not run as shipped."
resolution: "Add the missing `openai` wrapper module (or replace with `OpenAI().chat.completions.create`) and define the `client` used in generation/other_llm.py."
cross_refs: ["placeholder-paths"]
check_script: _audit_code/check_undefined_and_paths.py
paper_ref: "Algorithm 1; Section 3.3"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: recombination-skips-safety-judge
category: bug
topic: "safety verification"
title: "Recombined instructions are never safety-checked; code reuses a stale judge verdict"
severity: high
confidence: high
status: finding
file: framework/evorefuse.py
line_start: 318
line_end: 326
quote: |
            cross_instruction, reason = get_instruction_reason(cross_response)


            prompt_judge_suffix = "##Instruction: " + cross_instruction + " " + "##Reason: " + reason
            prompt_judge = prompt_judge_prefix + prompt_judge_suffix


            if judge_response == "safe":
                cross.append(cross_instruction)
claim: "In the recombination loop the code builds `prompt_judge` for the recombined instruction (line 322) but never calls `openai.generate(prompt_judge)`; the `if judge_response == \"safe\"` test on line 325 reads the `judge_response` variable left over from the last iteration of the mutation while-loop above, so the recombined instruction's own safety justification is never evaluated."
concern: "The paper claims every recombined instruction 'passes through the same safety verification process' (Section 3.3, Recombination); in the released code that check is a no-op driven by an unrelated stale verdict, so unsafe recombined instructions can enter the candidate pool — undermining the safety guarantee central to the benchmark's validity."
resolution: "Insert `judge_response = openai.generate(prompt_judge)` before the `if` on line 325, matching the mutation branch (lines 291-296)."
cross_refs: ["undefined-openai-symbol"]
paper_ref: "Section 3.3 'Recombination'; Algorithm 1 line 4"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: longppl-not-longppl
category: difference
topic: "metrics"
title: "metric/longppl.py computes thresholded instruction log-prob, not LongPPL of responses"
severity: medium
confidence: medium
status: question
file: metric/longppl.py
line_start: 6
line_end: 32
quote: |
  def calculate_token_probabilities(model, tokenizer, instruction, threshold):

      inputs = tokenizer(instruction, return_tensors="pt").to("cuda:0")
      with torch.no_grad():
          outputs = model(**inputs)
      logits = outputs.logits[0, :-1, :]
      probs = torch.softmax(logits, dim=-1)
      log_probs = torch.log(probs)
      input_ids = inputs.input_ids[0][1:]
      token_log_probs = []
      valid_log_probs = []
      for i in range(len(input_ids)):
          token_id = input_ids[i].item()
          log_prob = log_probs[i, token_id].item()
          token = tokenizer.convert_ids_to_tokens([token_id])[0]
          token_log_probs.append((token, log_prob))
          if log_prob > threshold:
              valid_log_probs.append(log_prob)
      if len(valid_log_probs) == 0:
          num = 0
          valid_log_probs.append(num)
      if valid_log_probs:
          average_log_prob = -sum(valid_log_probs) / len(valid_log_probs)
      else:
          average_log_prob = None
      return token_log_probs, average_log_prob
claim: "The script named after LongPPL averages per-token negative log-probabilities over the **instruction** text, keeping only tokens with log-prob above a fixed threshold (-2.8). Table 2 reports LongPPL(y|x), i.e. the response y given x, computed by the LongPPL key-token contrastive procedure of ref [43]; this code computes neither the response perplexity nor LongPPL's long-vs-short-context key-token selection."
concern: "If this is the script that produced the Table 2 LongPPL column, the reported quantity is a thresholded instruction-perplexity rather than the cited LongPPL(y|x), so the column does not measure what the paper states."
resolution: "Authors: confirm whether Table 2's LongPPL column was produced by this script; if so, explain the divergence from ref [43]'s LongPPL and from the (y|x) conditioning shown in the table header."
cross_refs: []
paper_ref: "Table 2, LongPPL(y|x) column; ref [43]"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

```yaml finding
id: lexical-only-msttr
category: difference
topic: "metrics"
title: "lexical.py computes only MSTTR; HDD and MTLD (also in Table 2) are commented out"
severity: low
confidence: high
status: finding
file: metric/lexical.py
line_start: 24
line_end: 33
quote: |
    flt = ld.flemmatize(all_instructions)
    # print(file_path)
    # print(ld.ttr(flt))

    # print(file_path)
    # print(ld.root_ttr(flt))


    print(file_path)
    print(ld.msttr(flt,window_length=800))
claim: "Table 2 reports three diversity metrics (MSTTR, HDD, MTLD), but the shipped lexical script only computes and prints `ld.msttr(...)`; no call to `ld.hdd` or `ld.mtld` exists anywhere in the repo (the other ld.* calls present are commented out and are ttr/root_ttr, not HDD/MTLD)."
concern: "The HDD and MTLD columns of Table 2 are not traceable to any computation in the repo, so two of the three reported diversity metrics cannot be reproduced."
resolution: "Add the HDD and MTLD computations (e.g. ld.hdd and ld.mtld) used for Table 2, or point to where they were computed."
cross_refs: []
paper_ref: "Table 2, HDD and MTLD columns"
tags: [reforms:2, reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding rises above a `question`. The fitness objective in
`framework/evorefuse.py` (`fitness_score`, lines 217-231) implements the paper's
Eq. 5 faithfully in sign and structure: `combined = -ppl_weight*log(perplexity)
+ refusal_weight*log(mean refusal indicator)`, where `-log(perplexity)` equals
the average response log-probability (the confidence term) and the refusal term
is the log of a classifier-derived refusal proxy — both matching Eq. 5's two
terms with `λ=ppl_weight=0.1`. The simulated-annealing acceptance and linear
cooling (`tau = 0.1 - 0.005*i`) match the paper's `τ_t = max(τ_f, τ_0 - β·t)`.
Topics N/A: temporal integrity (no time dimension); pretraining contamination
beyond the disclosed use of a public refusal classifier and commercial GPT-4O
(the API non-reproducibility is captured under `prr-crr-driver-not-self-contained`);
data splitting / sample independence (this is benchmark construction, not a
train/test predictive task).

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 3 | high | All paths are placeholders; no dependency file; judge-loop intermediates absent. |
| bug | 2 | high | Undefined `openai.generate` in 7 files; recombination skips its own safety check. |
| difference | 2 | medium | LongPPL script measures instruction-perplexity, not LongPPL(y\|x); HDD/MTLD not computed. |
| methodology | 0 | - | Fitness/annealing implementation matches Eq. 5; no invalid procedure found. |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **[bug] undefined-openai-symbol** — `openai.generate(...)` is called in 7 driver files but never defined/imported; the core generation, judging, and CRR scripts all NameError on first use. (high/high)
2. **[missing] placeholder-paths** — every model/input/output path across 23 files is a literal `"path"`/`"file"`; nothing can locate its data as shipped. (high/high)
3. **[bug] recombination-skips-safety-judge** — recombined instructions are never safety-checked; the code reuses a stale `judge_response` from the mutation loop, contradicting the paper's safety-verification claim. (high/high)
4. **[missing] missing-requirements** — no requirements/env file; README pins a non-existent `numpy==1.12.5`. (medium/high)
5. **[missing] prr-crr-driver-not-self-contained** — Table 1/3 refusal rates depend on absent intermediate response files and a non-deterministic closed GPT-4O judge. (medium/medium)
6. **[difference] longppl-not-longppl** — the LongPPL script computes thresholded instruction perplexity, not LongPPL(y\|x) per ref [43]. (medium/medium, filed as question)

### Items that genuinely look fine
- `datasets/evo_test.jsonl` has exactly 582 instructions (matches the paper).
- `datasets/evo_align.json` has 3,085 entries with `conversations`/`chosen`/`rejected` keys suitable for SFT/DPO (paper says 3,000; ~3% more).
- The `fitness_score` implementation matches Eq. 5 in sign and structure, and the simulated-annealing cooling matches the paper.
- The PRR prefix list (`metric/prr.py`) is the standard AutoDAN refusal-prefix set cited as ref [40].
- LoRA fine-tuning YAMLs (`finetune/sft.yaml`, `dpo.yaml`) match the paper's stated hyperparameters (5 epochs, lr 2e-5 SFT / 1e-5 DPO, warmup 0.03).

### Open questions for the authors
- Was Table 2's LongPPL column produced by `metric/longppl.py`? If so, why does it diverge from ref [43]'s LongPPL and from the (y|x) conditioning? (longppl-not-longppl)
- Where are the HDD and MTLD computations behind Table 2? (lexical-only-msttr)
- Can the authors release the intermediate response JSONL files (or a wired-up driver) so Tables 1 and 3 are reproducible without re-supplying every path and the GPT-4O judge? (prr-crr-driver-not-self-contained)
