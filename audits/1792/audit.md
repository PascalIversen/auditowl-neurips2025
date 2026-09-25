# Code audit — Ada-R1: Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization (paper 1792)

## 1. Summary

The repository `StarDewXXX/AdaR1` is a thin "recipe" wrapper around three external
projects rather than a self-contained reproduction package. The author-authored content
is essentially: a `README.md` "Reproduce Guide [In Progress]"; four LLaMA-Factory training
YAMLs (`LLaMA-Factory-YAMLs/`); one MergeKit config (`mergekit/examples/naive_merge.yml`);
and the single substantive Python script that implements the paper's core contribution,
`Dataset-Construction/deepscaler-release/scripts/constrcut_adaptive_dataset.py` (the
bi-level preference-pair builder). Everything else is bundled third-party source:
`mergekit/` (Arcee MergeKit) and `Dataset-Construction/deepscaler-release/` (the
Light-R1 / DeepScaleR + verl sampling-and-eval harness). No trained models, no DPO trainer
(LLaMA-Factory itself is not bundled, only its YAMLs), and no script that aggregates the
paper's Table 2 / abstract numbers are present.

What I did: read the README, all four YAMLs, the merge config, and the dataset-construction
script line by line; located the sampling driver (`scripts/eval/sample_from_model.sh`,
`eval_model.sh`) and the correctness/accuracy computation it depends on
(`verl/.../trainer/main_generation.py`, `deepscaler/rewards/math_utils/utils.py`); traced
each headline number to whatever computes it; and wrote one deterministic check under
`_audit_code/` (`check_gain_sign.py`) to confirm how the code assigns the group-level
preference. I stayed read-only on `code/`.

Note on fetch health (per `fetch_manifest.json`): the repo was re-cloned with
`--recurse-submodules`; 86 LFS-tracked files (e.g. all `processed_data/*.parquet`,
`deepscaler/data/*.parquet`, `mergekit/_data/*`) remain as ~130-byte git-lfs pointer stubs
because git-lfs is not installed here. I treat those as **not-materialised in this
environment, not authored-and-missing** — they are tracked by the repo's `.gitattributes`
LFS filter and would download with `git lfs pull`. The complete re-clone therefore did not
turn up any author-authored Python beyond the single dataset script above; the previous
(possibly incomplete) clone would have shown the same author surface area.

## 2. Traceability table

Headline numbers come from the abstract and Table 2. None of them are produced by any
script in the repo: there is no trained Ada-R1 checkpoint, no DPO training run, and no
evaluation/aggregation script that reads model outputs and emits these accuracy/length
values. The bundled verl harness (`main_generation.py`) *can* generate samples and compute
per-response correctness, but (a) it requires the trained models, which are absent, and
(b) nothing in the repo turns its per-response outputs into the Table-2 grid or the
abstract's reduction percentages.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Abstract: MATH −58% length, no acc loss | (none) | — | — | MISSING (no eval/aggregation script; no trained model) |
| Abstract: GSM8K −74% length, +acc | (none) | — | — | MISSING |
| Table 2 (7B/1.5B × {Long,Short,Merge,…,Ada-R1} × 5 benchmarks, acc+length) | (none) | — | — | MISSING (no script aggregates these) |
| Table 6 (α ablation on 100 AIME) | (none) | — | — | MISSING (no ablation driver) |
| Fig. 1 (gain proportion; CoT-length vs acc) | `constrcut_adaptive_dataset.py:336-351` prints `acc_counters`, `gain_avg` etc. | partial (prints stats, no figure) | unverifiable (needs LFS parquet + models) | PARTIAL / MISSING-plot |
| Per-response correctness (used by all of the above) | `verl/.../trainer/main_generation.py` + `deepscaler/rewards/math_utils/utils.py:430,481` | grader present (3rd-party) | n/a | Present (third-party) |
| Bi-level preference pair construction (the contribution) | `constrcut_adaptive_dataset.py:94-169,190-361` | builds DPO json | see findings | Present (author) |

Because the entire quantitative results section traces to "(none)", the dominant finding is
a `missing` one (no result-producing pipeline for any reported number), with the bi-level
construction script being the one author artefact that *is* present and therefore the place
where code↔paper and methodology findings live.

## 3. Findings

## missing

```yaml finding
id: no-result-aggregation-pipeline
category: missing
topic: "result traceability"
title: "No script produces any reported number (Table 2, abstract reductions, Table 6)"
severity: high
confidence: high
status: finding
file: README.md
line_start: 44
line_end: 61
quote: |
  ## Reproduce Guide [In Progress]
  To reproduce our method, you need to use MergeKit, LLaMA-Factory and our dataset construction scripts.
claim: "The repo provides a merge config, training YAMLs, and a preference-dataset builder, but contains no trained Ada-R1 model and no evaluation/aggregation script that reads model generations and emits the accuracy/length values in Table 2, the abstract's −58%/−74% length-reduction claims, or the Table 6 α-ablation. The bundled verl harness computes per-response correctness but nothing turns it into the reported grid."
concern: "Every quantitative claim in the paper is untraceable to repo code: a reader cannot regenerate any reported number, only re-run a generic sampler on models they must obtain elsewhere."
resolution: "Provide the evaluation/aggregation script(s) that map model outputs to each Table 2 cell, the abstract reduction percentages, and Table 6; and release (or link) the trained Ada-R1 checkpoints these numbers come from."
cross_refs: ["repro-guide-in-progress"]
paper_ref: "Abstract; Table 2; Table 6"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-trained-models-or-dpo-trainer
category: missing
topic: "expected code completeness"
title: "No trained checkpoints and no DPO training code (only LLaMA-Factory YAMLs)"
severity: high
confidence: high
status: finding
file: README.md
line_start: 59
line_end: 61
quote: |
  ### Step 3: Training an Adaptive Reasoning Model

  After completing all the above steps, you can execute the final training phase using **LLaMA-Factory** or any other framework that **supports DPO**. We provide the configuration file for LLaMA-Factory in `/LLaMA-Factory-YAMLs/`
claim: "Only four LLaMA-Factory YAML configs are shipped; LLaMA-Factory itself is not bundled, and no merged/short-CoT/Ada-R1 weights are present. Reproduction depends on an external, unpinned framework plus models the repo does not provide or link."
concern: "The headline numbers depend on full-parameter DPO training and on the merged and short-CoT checkpoints; without the trainer pin or any released weights, the results cannot be reproduced from this repo."
resolution: "Pin the LLaMA-Factory commit and its dataset_info registration for `ds-*_dpo_bilevel_*`, and release the merged, short-CoT, and final Ada-R1 checkpoints (or HF links)."
cross_refs: ["no-result-aggregation-pipeline"]
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec-author-code
category: missing
topic: "environment / dependencies"
title: "No requirements/environment spec for the author dataset-construction script"
severity: medium
confidence: high
status: finding
file: Dataset-Construction/deepscaler-release/scripts/constrcut_adaptive_dataset.py
line_start: 1
line_end: 18
quote: |
  import json
  # from utils import extract_answer
  # from grader import grade_answer
  from datasets import load_from_disk
  from datasets import load_dataset
  import random
  from transformers import AutoTokenizer
  import numpy as np
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt
  import random
  import sys
  import os
claim: "The core author script depends on datasets, transformers, numpy, seaborn, matplotlib and the bundled deepscaler reward utils, but the repo ships no requirements.txt / environment.yml / pyproject for the author layer (only the vendored verl/mergekit have their own setups)."
concern: "The environment for the contribution's data pipeline cannot be rebuilt deterministically; version skew in transformers/datasets tokenization directly affects the token-length comparisons that drive the preference pairs."
resolution: "Add a pinned dependency spec for the dataset-construction step (at minimum transformers, datasets, numpy versions)."
cross_refs: []
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: short-sft-1b-yaml-points-at-7b
category: bug
topic: "training configuration"
title: "1.5B short-CoT SFT YAML is a verbatim copy of the 7B one (wrong base model + output path)"
severity: medium
confidence: high
status: finding
file: LLaMA-Factory-YAMLs/ds-1b-short-sft.yaml
line_start: 1
line_end: 19
quote: |
  ### model
  model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  ### method
  stage: sft
  do_train: true
  finetuning_type: full
  deepspeed: examples/deepspeed/ds_z3_config.json

  ### dataset
  dataset: Qwen2.5-32B-Instruct_mix_mathematic_problems_ds-format #open_r1_cot_length_6k #qwen-14b-medium-cot #
  template: deepseek3
  cutoff_len: 4096 #2048 #8192
  max_samples: 2000 #10000 #10000
  overwrite_cache: true
  preprocessing_num_workers: 16
  # packing: true

  ### output
  output_dir: ../models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-Short-COT
claim: "`ds-1b-short-sft.yaml` is byte-identical to `ds-7b-short-sft.yaml` (verified with diff): it fine-tunes the 7B Distill model and writes to `Deepseek-Qwen-7B/Deepseek-Qwen-7B-Short-COT`, despite the filename and README Step 0 designating it the 1.5B short-CoT config."
concern: "Running the released 1.5B recipe would produce a 7B short model at the wrong path, so the 1.5B short-CoT model the paper merges (Sec 5.1) is not buildable from the shipped config as written."
resolution: "Provide the actual 1.5B config: set `model_name_or_path` to `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and the output_dir to the 1.5B path."
cross_refs: []
paper_ref: "Section 5.1 Short CoT Models"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-construct-script-path-wrong
category: bug
topic: "documentation / paths"
title: "README points to scripts/eval/constrcut_adaptive_dataset.py but file is in scripts/"
severity: low
confidence: high
status: finding
file: README.md
line_start: 55
line_end: 57
quote: |
  ### Step 2: Construct Training Dataset

  Then we use the scripts provided by **Light-R1** to generate initial samples from the Short-CoT and Long-CoT models. You should run the `Dataset-Construction/deepscaler-release/scripts/eval/sample_from_model.sh` and `Dataset-Construction/deepscaler-release/scripts/eval/constrcut_adaptive_dataset.py`
claim: "README references `scripts/eval/constrcut_adaptive_dataset.py`, but the script actually lives at `Dataset-Construction/deepscaler-release/scripts/constrcut_adaptive_dataset.py` (one directory up from `eval/`)."
concern: "Following the documented path fails with file-not-found; minor but blocks the documented Step 2."
resolution: "Fix the path in the README (drop the `eval/` segment)."
cross_refs: []
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: group-pref-uses-length-normalized-gain
category: difference
topic: "group-level preference rule"
title: "Group preference decided by a length-normalized `gain`, not the paper's accuracy-margin rule"
severity: medium
confidence: high
status: finding
file: Dataset-Construction/deepscaler-release/scripts/constrcut_adaptive_dataset.py
line_start: 271
line_end: 277
quote: |
          relative_accuracy_gain = long_accuracy - short_accuracy - 1/(2*K) #/ short_accuracy if short_accuracy != 0 else (long_accuracy - 1/K) / (1/K)
          relative_length_increnment = (long_avg_length - short_avg_length) / short_avg_length
          
          if relative_accuracy_gain > 0:
              gain = relative_accuracy_gain / relative_length_increnment
          else:
              gain = relative_accuracy_gain * (relative_length_increnment/max_length_inc_ratio)
claim: "The paper (Eq. 3 / group-preference rule, p.5) assigns the group by a bare accuracy margin `Ê[CL]-Ê[CS] > ε`. The code instead computes a length-normalized `gain` and chooses long iff `gain>0`. I verified (`_audit_code/check_gain_sign.py`, 200k random draws, 0 mismatches) that with long>short length the length term never flips the sign, so the rule reduces to `long_acc - short_acc > 1/(2K)`."
concern: "The implemented rule is a valid binary group assignment, but the paper neither states the threshold ε=1/(2K) nor the length-normalized `gain` formulation; the `gain` magnitude (and the `with_weight`/`all_gain_transformed` machinery built on it) is computed but does not affect the released dataset, so the description and code describe different procedures."
resolution: "State ε explicitly (=1/(2K)?) and clarify whether the length-normalized `gain` is part of the method or vestigial; reconcile Eq. 3 with the code's `relative_accuracy_gain` definition."
cross_refs: ["dead-weight-and-transform"]
check_script: _audit_code/check_gain_sign.py
paper_ref: "Stage II, Group-Level Preference (Eq. 3 and the gL>gS rule)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dead-weight-and-transform
category: difference
topic: "preference weighting"
title: "Per-pair `weight` and normalized-gain transform are computed but never used"
severity: low
confidence: high
status: finding
file: Dataset-Construction/deepscaler-release/scripts/constrcut_adaptive_dataset.py
line_start: 156
line_end: 168
quote: |
          all_samples = []
          for chosen_item in chosen_group:
              for rejected_item in rejected_group:
                  if with_weight:
                      weight = abs(gain)
                      all_samples.append(format_pairwise_sample(chosen_item, rejected_item, 1))
                  else:
                      all_samples.append(format_pairwise_sample(chosen_item, rejected_item, 1))

          random.shuffle(all_samples)
          all_samples = all_samples[0:max_pairs]
          output_data += all_samples # inter-group
          output_data += inner_group_samples # inner-group
claim: "Both branches of the `with_weight` conditional call `format_pairwise_sample(..., 1)` with a constant weight of 1; the computed `weight = abs(gain)` is discarded. Likewise `all_gain_transformed` (lines 354-356, via `normalize_gain`/`transform`) is computed at the end of `__main__` and never consumed."
concern: "If the paper's described preference signal involves any gain-based weighting or reward normalization (Appendix B mentions 'we normalized the reward values'), the released code does not apply it — all pairs are unit-weighted."
resolution: "Confirm whether gain-weighting / reward normalization was used for the reported runs; if so, supply the code path that actually applies it."
cross_refs: ["group-pref-uses-length-normalized-gain"]
paper_ref: "Appendix B Training Details ('we normalized the reward values')"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: repro-guide-in-progress
category: difference
topic: "repository provenance"
title: "README labels the reproduction guide '[In Progress]' on an untagged shallow main"
severity: low
confidence: high
status: finding
file: README.md
line_start: 44
line_end: 45
quote: |
  ## Reproduce Guide [In Progress]
  To reproduce our method, you need to use MergeKit, LLaMA-Factory and our dataset construction scripts.
claim: "The repo's own README marks the reproduction instructions as in-progress; the audited code is an untagged `main` (single visible commit `c13c296`) with no release tag matching the NeurIPS 2025 submission."
concern: "There is no commit pinned to the paper's results, and the authors themselves flag the recipe as incomplete, so the audited artefact is not asserted to be the exact state that produced the numbers."
resolution: "Tag the commit corresponding to the submission and complete the reproduction guide."
cross_refs: ["no-result-aggregation-pipeline"]
tags: [forensics:git-archaeology]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: instance-pref-shortest-wrong-when-no-correct
category: methodology
topic: "instance-level preference construction"
title: "Instance-level fallback can prefer a shorter WRONG response over a longer wrong one"
severity: medium
confidence: medium
status: finding
file: Dataset-Construction/deepscaler-release/scripts/constrcut_adaptive_dataset.py
line_start: 132
line_end: 143
quote: |
              elif len(correct_chosen_group) != 0 and len(wrong_chosen_group) == 0:
                  sorted_indices = sorted(range(len(correct_chosen_group_lengths)), key=lambda i: correct_chosen_group_lengths[i])
                  shorest_item = correct_chosen_group[sorted_indices[0]]
                  longest_item = [correct_chosen_group[i] for i in sorted_indices[-M:]]

              elif len(correct_chosen_group) == 0 and len(wrong_chosen_group) != 0:
                  sorted_indices = sorted(range(len(wrong_chosen_group_lengths)), key=lambda i: wrong_chosen_group_lengths[i])
                  shorest_item = wrong_chosen_group[sorted_indices[0]]
                  longest_item = [wrong_chosen_group[i] for i in sorted_indices[-M:]]

              for long_item in longest_item:
                  inner_group_samples.append(format_pairwise_sample(shorest_item, long_item, 1))
claim: "The paper defines the instance-level preferred response as the shortest CORRECT response (yw = argmin over correct responses). The code's third branch (no correct responses in the chosen group) sets the preferred (`shorest_item`) to the shortest WRONG response and rejects longer wrong responses, emitting these as positive training pairs (chosen=short-wrong, rejected=long-wrong)."
concern: "On problems where the chosen group has zero correct samples, DPO is trained to prefer a short but incorrect answer, which optimizes brevity at the expense of correctness — contrary to the paper's stated 'prefer the shortest correct response' and potentially degrading accuracy on hard items."
resolution: "Confirm whether the no-correct-response branch was active in the reported runs (and on how many problems); if so, justify training on short-wrong>long-wrong pairs or exclude such problems from the instance-level set."
cross_refs: []
paper_ref: "Stage II, Instance-Level Preference (yw = argmin_{y in correct} |y|)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: inter-group-pairs-may-be-all-wrong
category: methodology
topic: "group-level pair construction"
title: "Inter-group pairs can be built from all-wrong chosen groups (chosen not guaranteed correct)"
severity: low
confidence: medium
status: finding
file: Dataset-Construction/deepscaler-release/scripts/constrcut_adaptive_dataset.py
line_start: 109
line_end: 113
quote: |
          correct_chosen_group = filter_group(chosen_group, ground_truth_answer, "correct")

          if len(correct_chosen_group) != 0:
              chosen_group = correct_chosen_group
claim: "For inter-group (group-level) pairs, the chosen group is restricted to correct responses only when at least one correct response exists; if the chosen group has zero correct responses, the full (all-wrong) group is kept and paired against the rejected group as the preferred side."
concern: "Group-level preference pairs can then have an incorrect 'chosen' response, weakly training the model to prefer an incorrect reasoning style; the paper's framing assumes the preferred group reflects correct reasoning."
resolution: "Confirm the prevalence of zero-correct chosen groups and whether such problems should be dropped from the group-level set."
cross_refs: ["instance-pref-shortest-wrong-when-no-correct"]
paper_ref: "Stage II, Group-Level Preference"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                |
|-------------|------------|--------------|----------------------------------------------------------------|
| missing     | 3          | high         | No pipeline produces any reported number; no weights/DPO trainer; no dep spec. |
| bug         | 2          | medium       | 1.5B short-SFT YAML is a copy of the 7B one; README script path wrong. |
| difference  | 3          | medium       | Group rule via length-normalized `gain`; dead weighting; "[In Progress]" untagged main. |
| methodology | 2          | medium       | Instance-level fallback can prefer short-but-wrong; chosen group may be all-wrong. |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **[missing] No result-producing pipeline** — every Table 2 / abstract / Table 6 number traces to "(none)"; the repo has no eval-aggregation script and no trained Ada-R1 model (`no-result-aggregation-pipeline`, high/high).
2. **[missing] No trained checkpoints and no DPO trainer** — only LLaMA-Factory YAMLs are shipped (`no-trained-models-or-dpo-trainer`, high/high).
3. **[methodology] Instance-level fallback can prefer a shorter WRONG answer** over a longer wrong one when the chosen group has no correct responses (`instance-pref-shortest-wrong-when-no-correct`, medium/medium).
4. **[difference] Group preference uses a length-normalized `gain`**, not the paper's accuracy-margin rule with threshold ε (`group-pref-uses-length-normalized-gain`, medium/high).
5. **[bug] The 1.5B short-CoT SFT YAML is a verbatim copy of the 7B one** (wrong base model + output path), so the 1.5B short model is not buildable from the released config (`short-sft-1b-yaml-points-at-7b`, medium/high).
6. **[missing] No dependency spec** for the author dataset-construction layer (`no-dependency-spec-author-code`, medium/high).

### Items that genuinely look fine
- M1=4 and M2=2 in the code (`max_pairs=4`, `M=2`) match Table 7's reported M1/M2 hyperparameters.
- Sampling settings match: `--n_samples 12` (sample_from_model.sh) matches Appendix B "we sample 12 times"; the gain threshold reduces to ε=1/(2K), a sensible margin.
- Per-response correctness uses the established DeepScaleR/`grade_answer_sympy` grader; multiple-choice and empty ground-truths are correctly skipped (lines 243-246).
- The merge config (`naive_merge.yml`) implements the paper's Stage-I linear merge with weight 0.85/0.15 (≈ α=0.85), consistent with Sec 4 / Table 6's α study.
- The LFS pointer stubs (`processed_data/*.parquet`, etc.) are tracked via `.gitattributes` LFS filters — not authored-and-missing; they are simply unfetched in this environment.

### Open questions for the authors
- Was the no-correct-response instance-level branch (short-wrong > long-wrong) and the all-wrong group-level case active in the reported runs, and on how many of the 2,500 problems? (supports `instance-pref-shortest-wrong-when-no-correct`, `inter-group-pairs-may-be-all-wrong`)
- Is the length-normalized `gain` / `weight` / `all_gain_transformed` machinery part of the reported method, or vestigial code? Appendix B mentions reward normalization that the released script does not apply. (`dead-weight-and-transform`)
- Where is the evaluation/aggregation code that produces Table 2 and the abstract's −58%/−74% reductions, and where are the trained Ada-R1 checkpoints? (`no-result-aggregation-pipeline`)
