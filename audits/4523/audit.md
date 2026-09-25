# Code Audit — Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations (CoD)

## 1. Summary

The repository (`FaisalHamman/CoD`) implements the CoD framework: a CFE-infused few-shot
knowledge-distillation pipeline. It contains (a) a counterfactual-explanation generator
(`cfx-generator/`, prompts an OpenAI model and builds balanced k-shot subsets), (b) teacher
fine-tuning scripts (`text-classification/teacher_trainer*.py`), (c) a distillation driver
(`text-classification/ted_no_trainer*.py`) that implements KD / LWD / TED with optional CFE
data, and (d) orchestration shell scripts (`scripts_cfx/`). The README maps to the paper and
the audited code is the repo cited in the paper (`https://github.com/FaisalHamman/CoD`).

What I did: read the paper (`paper_text.txt` / `paper.pdf`) and every Python/shell file in the
repo; traced the headline tables (Tables 1–3, 5–9) to code; and wrote two deterministic checks
under `_audit_code/` (`check_eval_split.py`, executed; output in `_audit_code/out/`). I did not
run the training pipeline (requires GPUs, an OpenAI key, and downloaded teacher checkpoints).
Everything under `code/` was treated read-only.

Key conclusions:
- The reported per-cell accuracies are the **maximum-over-epochs accuracy on the validation
  split**, which is also the *only* held-out set used (the test split is deleted). This is
  model/epoch selection on the same data that is reported, and there is no independent test set
  — a methodology concern affecting every results table.
- The paper claims each generated CFE is **validated against the teacher** (kept only if it flips
  the teacher's prediction). No code performs this check; labels are hard-set to `1 - y`.
- The paper claims the metric is computed **"on the test set"** and that input–CFE pairs share a
  mini-batch; the headline code evaluates on **validation** and (in the `cfx` mode actually run)
  concatenates and shuffles original+CFE rows without enforcing pairing.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (DeBERTa-base→small, KD/LWD ±CFE, 6 datasets × 6 k) | `text-classification/ted_no_trainer_qwen.py` via `scripts_cfx/run_seed.sh` / `run.sh` | not re-run | n/a | Code present; metric = max-epoch **validation** acc (see val-eval, save-best findings) |
| Table 2 (TED / TED+CFE) | `learn_filters_glue_no_trainer.py` + `ted_no_trainer.py` via `run_seed_ted.sh`, `learn_ted_filters.sh` | not re-run | n/a | Code present; same eval caveat |
| Table 3 / Table 6 (Qwen2.5-1.5B→0.5B) | `ted_no_trainer_qwen.py` via `run_seed_qwen.sh` | not re-run | n/a | Code present; same eval caveat |
| Table 4 (full teacher accuracies) | `teacher_trainer.py` / `teacher_trainer_qwen.py` | not re-run | n/a | Code present (eval on validation) |
| Table 5 (DeBERTa-xsmall student) | `ted_no_trainer_qwen.py` via `run_seed.sh` | not re-run | n/a | Code present; same eval caveat |
| Table 7 (prompt-template ablation, SST2 variants) | `cfx-generator/prompts.py` (`prompt_templates_sst2_variants`) | — | — | **MISSING**: variants defined but never referenced by any generator code |
| Table 8 (compute / energy via codecarbon) | `ted_no_trainer*.py` `--measure_energy` + `EmissionsTracker` | not re-run | n/a | Code present |
| Table 9 (soft-label ablation: α=0; random soft labels) | α=0 = set `--kl_alpha 0` (supported). Random-soft-label path | — | — | **PARTIAL**: α=0 traceable; "replacing soft labels with random values" has no code |
| Synthetic 2D-moons experiment (Fig. 1, Fig. 3) | (none) | — | — | **MISSING**: no script generates the 2D-moons teacher/student decision boundaries |

## 3. Findings

## missing

```yaml finding
id: cfe-teacher-validation-absent
category: missing
topic: CFE generation
title: Paper's teacher-prediction check on CFEs is not implemented; labels hard-set to 1 - y
severity: high
confidence: high
status: finding
file: code/FaisalHamman__CoD/cfx-generator/utils.py
quote: |
  counterfactual_sentiment = 0 if raw_datasets["train"][i]['label'] == 1 else 1
line_start: 164
line_end: 173
claim: The CFE generator prompts the LLM for a flipped-sentiment sentence and stores it with label `1 - y` (`counterfactual_sentiment`), then concatenates it to the training set. It never loads the teacher model or checks whether the generated sentence actually flips the teacher's prediction; grep for "teacher"/"predict"/"flip"/"validate" in `cfx-generator/` returns nothing.
concern: The paper's CFE definition and method rely on a teacher-validation step ("We then check whether this generated example indeed flips the teacher model's prediction, ensuring its utility as a true CFE"); without it, the "counterfactuals" are unvalidated label-flipped paraphrases, so the central premise (boundary-near, teacher-flipping examples) is not enforced by the code that produced the results.
resolution: Point to the script that filters generated sentences by teacher prediction, or confirm the headline tables were produced from the unvalidated `cfx` datasets in `cfx-generator/utils.py`.
cross_refs: [paper-says-pair-in-minibatch]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: prompt-variant-ablation-no-code
category: missing
topic: ablations
title: Prompt-template ablation (Table 7) variants are defined but never used
severity: low
confidence: high
status: finding
file: code/FaisalHamman__CoD/cfx-generator/prompts.py
quote: |
  prompt_templates_sst2_variants  = {
line_start: 106
line_end: 106
claim: >-
  `prompt_templates_sst2_variants` (the SST2 prompt variants for the Table 7 robustness ablation) is defined in `prompts.py`, but the generator (`utils.py`) always indexes `prompt_templates[self.dataset_name]`; no code path selects a variant, so Table 7 cannot be reproduced from the repo.
concern: The prompt-robustness ablation reported in Table 7 has no driver code, so the claim "COD is robust to prompt choices" is not reproducible from the artefact.
resolution: Add or point to the script that loops over `prompt_templates_sst2_variants` and runs generation + distillation per variant.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: random-softlabel-ablation-no-code
category: missing
topic: ablations
title: Random soft labels ablation (Table 9) has no implementation
severity: low
confidence: medium
status: finding
file: code/FaisalHamman__CoD/text-classification/ted_no_trainer_qwen.py
quote: |
  kl_loss = kl_div(
      outputs.logits, teacher_outputs.logits.detach())
line_start: 963
line_end: 964
claim: Table 9 reports an ablation that replaces the teacher's soft labels with random values. The KD loss always uses the teacher's real logits; there is no flag or code path that randomizes/corrupts the teacher logits (grep for `rand`/`noise`/`corrupt` over the soft-label path finds nothing relevant). The α=0 variant is reproducible (`--kl_alpha 0`), but the random-soft-label variant is not.
concern: One of the two Table-9 conditions cannot be reproduced from the code.
resolution: Provide the script/flag that substitutes random soft labels for the teacher logits.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: synthetic-moons-experiment-no-code
category: missing
topic: synthetic experiment
title: 2D-moons motivating experiment (Fig. 1, Fig. 3) has no code in the repo
severity: low
confidence: high
status: finding
file: code/FaisalHamman__CoD/README.md
quote: |
  Supported datasets: `sst2`, `cola`, `imdb`, `sentiment140`, `amazon`, `yelp`.
line_start: 41
line_end: 41
claim: The repo's datasets and scripts cover only the six text-classification benchmarks. There is no script implementing the synthetic 2D `make_moons` teacher (`[2->64->64->2]`) and CFE-infused student used for Fig. 1 / Fig. 3, which the paper presents as the core intuition-validating experiment.
concern: The illustrative decision-boundary figures motivating the method cannot be reproduced from the artefact.
resolution: Add the synthetic 2D-moons notebook/script, or note it is out of scope for the released code.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone runtime bug was confirmed in the code paths actually invoked by the headline run
scripts (`run_seed.sh`, `run_seed_qwen.sh` call `ted_no_trainer_qwen.py`, whose TED/Qwen-specific
imports are commented out and replaced by `AutoModelForSequenceClassification`; the KD/LWD paths
they use do not reference the removed `filter_states`). Note `text-classification/ted_no_trainer.py`
still imports `TEDDebertaV2Config`/`QwenConfig` at module top (lines 54, 66) which would fail at
import, but the headline scripts run `ted_no_trainer_qwen.py`, not this file — so this is not on the
result-producing path. Reported as a question rather than a finding.

```yaml finding
id: ted-no-trainer-broken-imports
category: bug
topic: repository hygiene
title: ted_no_trainer.py references undefined TED/Qwen config classes at import time
severity: low
confidence: medium
status: question
file: code/FaisalHamman__CoD/text-classification/ted_no_trainer.py
quote: |
  from transformers import QwenConfig, QwenForSequenceClassification 
line_start: 54
line_end: 54
claim: >-
  `ted_no_trainer.py` imports `QwenConfig, QwenForSequenceClassification` (line 54) and builds `CONFIG_MAPPING` from `TEDDebertaV2Config`/`QwenConfig` (lines 66-67) while the `models.ted_deberta_v2` import is commented out (line 51); transformers 4.28 has no `QwenConfig`, so this module would raise ImportError. The headline KD/LWD/Qwen runs invoke `ted_no_trainer_qwen.py` (these imports commented out) instead, so this file may be vestigial.
concern: If any reported number used `ted_no_trainer.py` directly it would not import; it is unclear which TED-table runs used which file.
resolution: Confirm whether `ted_no_trainer.py` is used for any reported table (e.g., the TED stage-2 in `run_seed_ted.sh`) and, if so, how it imports successfully.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

## difference

```yaml finding
id: eval-on-validation-not-test
category: difference
topic: evaluation set
title: Accuracy is computed on the validation split, but the paper says "on the test set"
severity: medium
confidence: high
status: finding
file: code/FaisalHamman__CoD/text-classification/ted_no_trainer_qwen.py
quote: |
  if "test" in raw_datasets:
          del raw_datasets["test"]
line_start: 749
line_end: 750
claim: The distillation script deletes the `test` split (line 749-750) and sets `eval_dataset = processed_datasets["validation"]` (line 793); all reported accuracies come from the validation split. The paper states "Performance is evaluated using accuracy on the test set for each dataset." For SST2/CoLA/IMDB the GLUE/HF test splits are unlabeled (paper Appendix C.1), so the validation split is a legitimate held-out set — but it is not the "test set" the paper names. Confirmed for `ted_no_trainer.py`, `ted_no_trainer_qwen.py`, `learn_filters_glue_no_trainer.py` by `_audit_code/check_eval_split.py`.
concern: The reported metric is computed on a different (validation) split than the paper states, which matters because that same split is also used for checkpoint selection (see model-selection finding).
resolution: Clarify that "test set" in the paper means the validation split actually used, and report which split each dataset's numbers come from.
cross_refs: [save-best-selects-on-eval-set]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: paper-says-pair-in-minibatch
category: difference
topic: training procedure
title: Headline runs concatenate+shuffle original/CFE rows; paper says each input-CFE pair shares a mini-batch
severity: medium
confidence: high
status: finding
file: code/FaisalHamman__CoD/cfx-generator/utils.py
quote: |
  raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], new_dataset])
line_start: 179
line_end: 179
claim: In the `cfx` data mode used by every headline run script (`run_seed.sh`, `run_seed_qwen.sh`, `run_seed_ted.sh` pass `data_type ... cfx`), originals and CFEs are concatenated into one split (`utils.py:179`) and the training DataLoader uses `shuffle=True` (`ted_no_trainer_qwen.py:812`), so an input and its CFE generally land in different mini-batches. The same-mini-batch pairing and the symmetric pair loss are implemented only in the `pair_cfx` branch (`ted_no_trainer_qwen.py:986-995`), which no headline script uses.
concern: The paper states "we ensure that each input-CFE pair is included in the same mini-batch, enabling the student to jointly learn from both examples"; the actually-run pipeline does not enforce this, so the described mechanism differs from the executed one.
resolution: Confirm whether the reported tables used `cfx` (concatenate+shuffle) or `pair_cfx` (paired) mode, and reconcile with the paper's mini-batch-pairing description.
cross_refs: [cfe-teacher-validation-absent]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: save-best-selects-on-eval-set
category: methodology
topic: model selection / held-out test set
title: Reported accuracy is the max-over-epochs validation accuracy, and validation is the only held-out set
severity: high
confidence: high
status: finding
file: code/FaisalHamman__CoD/text-classification/ted_no_trainer_qwen.py
quote: |
  if args.save_best:
      if eval_metric[task_to_metrics[args.task_name]] > best_eval:
          best_eval = eval_metric[task_to_metrics[args.task_name]]
line_start: 1085
line_end: 1087
claim: With `--save_best` (set by every headline run script), the script evaluates on the validation split after each epoch and writes `all_results.json` only when the epoch's validation accuracy exceeds the running best (lines 1085-1100), i.e. the reported number is the maximum validation accuracy over all epochs (`num_train_epochs` = 40-100 in the scripts). The test split was deleted (see eval-on-validation finding), so the same split is used both to pick the best epoch and as the reported result, and no untouched test set exists. `_audit_code/check_eval_split.py` confirms `save_best_takes_max` and `deletes_test_split` for the distillation scripts.
concern: Selecting the best of up to ~100 per-epoch evaluations on the very set whose score is reported is an optimistic, selection-on-the-evaluation-set estimate with no independent held-out test set, inflating all reported accuracies (Tables 1-3, 5-9) by an unknown amount; the bias can differ between methods/k and thus affect the COD-vs-baseline gaps.
resolution: Re-report using a fixed final-epoch (or validation-selected, test-evaluated) protocol with a genuinely held-out test set disjoint from the selection set; quantify how much the "best-epoch on the reported split" inflates each cell.
cross_refs: [eval-on-validation-not-test]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Teacher-validation of CFEs, two ablations, and the synthetic experiment have no code |
| bug         | 0 (1 question) | low      | `ted_no_trainer.py` has bad imports but is off the headline path |
| difference  | 2          | medium       | Eval on validation (paper says test); cfx mode does not pair input+CFE in a mini-batch |
| methodology | 1          | high         | Reported metric = max-epoch validation acc; no independent test set |

## 5. Closing lists

**Top take-aways** (≤6, severity × confidence):
1. [methodology] Reported accuracies are max-over-epochs accuracy on the validation split, which is also the only held-out set (test deleted) — selection-on-the-eval-set with no independent test set. (`save-best-selects-on-eval-set`)
2. [missing] The paper's teacher-prediction validation of CFEs is not implemented; CFE labels are hard-set to `1 - y` with no teacher check. (`cfe-teacher-validation-absent`)
3. [difference] Headline `cfx` runs concatenate+shuffle originals and CFEs; the paper's "same mini-batch pairing" is only in the unused `pair_cfx` branch. (`paper-says-pair-in-minibatch`)
4. [difference] Metric is computed on the validation split though the paper says "on the test set." (`eval-on-validation-not-test`)
5. [missing] The synthetic 2D-moons motivating experiment (Figs. 1, 3) has no code. (`synthetic-moons-experiment-no-code`)
6. [missing] Prompt-template ablation variants (Table 7) and the random-soft-label ablation (Table 9) have no driver code. (`prompt-variant-ablation-no-code`, `random-softlabel-ablation-no-code`)

**Items that genuinely look fine**:
- KD loss is a standard temperature-2 KL on teacher/student logits; LWD adds a per-layer MSE with a learned projection to match dimensions (`ted_no_trainer_qwen.py:957-984`) — methodologically sound.
- k-shot subsets are class-balanced and seeded; the few-shot training data is disjoint from the validation split used for evaluation, so there is no train/eval contamination of the reported metric (separate from the model-selection concern).
- The `--measure_energy` / codecarbon path for Table 8 is present and wired.
- Five seeds are genuinely swept (`SEEDS=(0 1 2 3 4)`), consistent with "averaged over five runs."

**Open questions for the authors**:
- Were the reported tables produced with `--save_best` (max-epoch) or last-epoch checkpoints, and on which split per dataset? This determines the magnitude of the optimistic bias in #1/#4.
- Did the released tables use `cfx` (concatenate) or `pair_cfx` (paired) data mode? (#3)
- Was any teacher-prediction filtering applied to CFEs before training, outside the released generator? (#2)
- Which file (`ted_no_trainer.py` vs `..._qwen.py`) produced the TED rows, given the import issue in the former? (`ted-no-trainer-broken-imports`)
