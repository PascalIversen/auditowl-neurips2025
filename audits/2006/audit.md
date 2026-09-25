# Code audit — *Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs* (NeurIPS 2025, #2006)

## 1. Summary

The repo (`technion-cs-nlp/vlm-circuits-analysis`, audited commit `8bb6218`) is the
authors' mechanistic-interpretability codebase. It contains: a vendored TransformerLens
fork extended for VLMs (`third_party/TransformerLens`), per-task prompt CSVs and
data-generation scripts (`data/`, `data_generation/`), and the experiment drivers:
`script_node_circuit_discovery_and_eval.py` (attribution patching + faithfulness, Fig. 4 /
Tables 5–6), `script_node_intersection.py` (IoU, Fig. 5), `script_node_cross_modality_analysis.py`
(sub-circuit interchange, Fig. 6), `script_backpatching_experiment.py` (back-patching, Table 1)
and `script_backpatch_vqav2.py` (Table 8). `figures_and_results_processing.ipynb` reads the
saved `.pt` outputs to render every figure/table.

This is a `torch.set_grad_enabled(False)` analysis pipeline; running it requires the three
12B-class VLMs and (per Appendix B.4) up to 4 GPUs. I therefore audited statically plus with
three deterministic `_audit_code/` checks (no GPU): file-existence
(`check_missing_files.py`), the random-baseline index off-by-one
(`check_random_subcircuit_oob.py`), and the back-patching model-selection protocol
(`check_backpatching_selection.py`). I verified prompt CSVs match the paper's Table 3 counts
(e.g. Qwen visual counting = 382 rows vs. reported 383), confirmed the 4 `images.tar.gz` are
Git-LFS pointer stubs (a fetch artefact, not a finding), and confirmed no precomputed `.pt`
result artefacts ship with the repo.

## 2. Traceability table

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Fig. 4 / Table 6 — circuit faithfulness | `script_node_circuit_discovery_and_eval.py:37-102`, `evaluation_utils.py:circuit_faithfulness` | yes (needs GPU + models) | not re-run | Present, not re-run |
| Table 5 — circuit sizes | `script_node_circuit_discovery_and_eval.py` + notebook cell 2/14 | yes | not re-run | Present, not re-run |
| Fig. 5 / "18% shared", "12%", "38%" — Normalized IoU | `script_node_intersection.py`, `analysis_utils.get_full_intersection_dict` | yes | not re-run | Present; random baseline has index off-by-one (see `random-baseline-index-oob`) |
| Fig. 6 — sub-circuit interchange faithfulness | `script_node_cross_modality_analysis.py:96-427` | yes | not re-run | Present; random baseline off-by-one + `vl_random_D` position-range bug |
| Fig. 7 — visual↔text cosine similarity | notebook cell 22 (`modality_alignment_utils`, `general_utils.generate_activations`) | yes | not re-run | Present, not re-run |
| Table 1 — back-patched accuracies (point estimates) | `script_backpatching_experiment.py` + notebook cell 8 | yes | not re-run | Present; best config selected on the same eval set (see `backpatching-best-config-on-eval-set`) |
| Table 1 — "± std across resamples", green statistical-significance marks; §5 "bootstrap 1000 iterations, lower bound > baseline" | (none) | — | — | MISSING (no bootstrap / std / resampling code anywhere) |
| "closes ≈32% of the gap" (avg relative diff) | notebook cell 8 | yes | not re-run | Present; averages only relative diffs in (0, 1] |
| Table 8 — VQAv2 / RealWorldQA back-patching (incl. "± std", bootstrap) | `script_backpatch_vqav2.py` | partial | not re-run | Present, but external VQA question/annotation JSONs absent (see `missing-external-vqa-data`); no bootstrap code |
| Tables 9–10 — PaliGemma2 scale results (incl. bootstrap std) | (none located) | — | — | MISSING driver + no bootstrap code |
| Table 4 — baseline task accuracies | `evaluation_utils.model_accuracy`, prompt CSVs | yes | not re-run | Present, not re-run |
| Appendix C / factual-recall baseline acc on back-patching path | `evaluation_utils.py:441-485` (`model_accuracy_for_factual_recall`) | yes | — | Broken: hard-loads absent `qa_raw.json` (see `missing-qa-raw-and-sentiment-json`) |

## 3. Findings

### missing

```yaml finding
id: missing-bootstrap-significance
category: missing
topic: "statistical integrity / result traceability"
title: "No code computes Table 1 bootstrap std or the significance marks claimed in §5"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  To show statistical significance, we use bootstrap re-sampling with 1000 iterations and verify that
  the lower bound of the back-patched accuracy is higher than the baseline accuracy. We report the
  back-patched accuracies, the improvement compared to the baseline model accuracy, and standard
  deviations across resamples in Table 1.
claim: "Table 1 reports per-cell '± std' and green-highlighted statistical significance derived from a 1000-iteration bootstrap whose lower bound must exceed the baseline; a repo-wide grep finds no bootstrap, resample, percentile, or std-computation code in any .py or in the figures notebook (only an unrelated PIL `Image.Resampling` reference)."
concern: "The headline claim of statistically significant improvement (the green marks in Table 1, the '± std' values, and the analogous Tables 8–10) cannot be reproduced or checked because the significance/uncertainty computation is absent from the released code."
resolution: "Authors: please add the bootstrap-resampling script that produces the per-cell standard deviations and the lower-bound-vs-baseline significance test, or clarify where it lives."
cross_refs: ["backpatching-best-config-on-eval-set"]
check_script: _audit_code/check_missing_files.py
paper_ref: "Section 5; Table 1; Tables 8-10"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-dependency-spec
category: missing
topic: "expected code completeness / dependencies"
title: "No dependency specification (requirements.txt / environment.yml / pyproject) in repo"
severity: medium
confidence: high
status: finding
file: out/missing_files.csv
csv_row: 1
quote: |
  dependency_spec_present (requirements/environment/setup/pyproject),False
claim: "The repo ships no pinned (or unpinned) dependency manifest; the code imports torch, transformers (with model-specific classes Qwen2VL/Gemma3/Llava/Mllama), qwen_vl_utils, sklearn, umap, googletrans, PIL, tqdm, plus a vendored TransformerLens fork that itself has unstated transitive requirements."
concern: "transformers model classes and TransformerLens internals are version-sensitive; without a pinned environment the circuit-discovery pipeline cannot be reliably rebuilt, blocking reproduction."
resolution: "Authors: add a requirements.txt / environment.yml pinning torch, transformers, and the other listed libraries (and the TransformerLens fork commit)."
cross_refs: []
check_script: _audit_code/check_missing_files.py
paper_ref: "NeurIPS checklist Q5/Q8 (claims exact environment provided)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-qa-raw-and-sentiment-json
category: missing
topic: "missing data on main execution path"
title: "qa_raw.json and sentiment_vl.json absent but hard-loaded on the back-patching path"
severity: medium
confidence: high
status: finding
file: evaluation_utils.py
line_start: 453
line_end: 453
quote: |
      qa_json = json.load(open("./data/factual_recall/qa_raw.json", "r"))
claim: "`model_accuracy_for_factual_recall` (called by `script_backpatching_experiment.py` to compute the factual-recall baseline accuracy) unconditionally opens `data/factual_recall/qa_raw.json`; likewise `sentiment_analysis_utils.load_sentiment_analysis_parallel_l_prompts` (line 176, used via `get_parallel_l_prompts` on the back-patching path) unconditionally opens `data/sentiment_analysis/sentiment_vl.json`. Neither file exists in the repo (only the per-model CSVs are shipped)."
concern: "The factual-recall back-patching baseline and the sentiment parallel-text prompts crash with FileNotFoundError, so Table 1's factual-recall and sentiment columns are not reproducible from the released artefacts even though the CSVs are present."
resolution: "Authors: add `qa_raw.json` and `sentiment_vl.json` to the data release (the NeurIPS checklist promises the dataset will be published), or refactor these load paths to derive the needed fields from the shipped CSVs."
cross_refs: []
check_script: _audit_code/check_missing_files.py
paper_ref: "Table 1 (Factual Recall, Sentiment Analysis rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-external-vqa-data
category: missing
topic: "missing data / external benchmark"
title: "VQAv2 question JSON path referenced for Table 8 not present and not fetched"
severity: low
confidence: medium
status: question
file: script_backpatch_vqav2.py
line_start: 33
line_end: 33
quote: |
  VQA_QUESTIONS_PATH = r"./vqa/v2_OpenEnded_mscoco_val2014_questions.json"
claim: "Table 8 (VQAv2 / RealWorldQA back-patching) depends on an external VQAv2 questions JSON under `./vqa/` that is not in the repo and has no fetch script or documented download in the README."
concern: "Table 8 cannot be reproduced from the repo alone; the external benchmark dependency is undocumented (VQAv2 is public, so this is a documentation/fetch gap rather than a hard blocker)."
resolution: "Authors: document the VQAv2/RealWorldQA download and expected paths, or add a fetch script."
cross_refs: []
check_script: _audit_code/check_missing_files.py
paper_ref: "Appendix E.4, Table 8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: random-baseline-index-oob
category: bug
topic: "random baseline construction"
title: "Random sub-circuit uses inclusive randint(0, n_layers/n_heads), allowing out-of-range indices"
severity: low
confidence: high
status: finding
file: script_node_cross_modality_analysis.py
line_start: 75
line_end: 92
quote: |
        random_sub_circuit.append(
            Component(
                c.hook_name,
                layer=random.randint(0, model.cfg.n_layers),
                head=(
                    random.randint(0, model.cfg.n_heads)
                    if c.head_idx is not None
                    else None
                ),
                position=random.choice(possible_positions),
                neurons=(
                    random.sample(
                        list(range(0, model.cfg.d_mlp)), k=len(c.neuron_indices)
                    )
                    if c.neuron_indices is not None
                    else None
                ),
            )
        )
claim: "The random-baseline sub-circuit generator draws layer indices from `random.randint(0, model.cfg.n_layers)` and head indices from `random.randint(0, model.cfg.n_heads)`; Python's `random.randint(a,b)` is inclusive of `b`, so it can emit `layer == n_layers` and `head == n_heads`, which are one past the valid 0..n-1 range (the check script shows ~3.5% of sampled layer indices equal n_layers for a 28-layer model)."
concern: "Random-baseline components placed at a non-existent layer/head can never coincide with a real circuit component, slightly deflating the random-baseline interchange faithfulness that Fig. 6's normalization uses as its lower bound; the related `vl_random_D` is also built over `l_D_limits` (textual data positions) instead of `vl_D_limits` (line 217-219), giving the visual random-D baseline wrong position labels."
resolution: "Use `random.randint(0, n-1)` / `random.randrange(n)` for layer and head, and build `vl_random_D` over `vl_D_limits`; re-run the random-baseline normalization to confirm Fig. 6 is unaffected."
cross_refs: []
check_script: _audit_code/check_random_subcircuit_oob.py
paper_ref: "Figure 6 (interchange faithfulness random baseline); Equation (5) normalization"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: relative-diff-filtering-32pct
category: difference
topic: "headline aggregate (gap-closing %)"
title: "The '≈32% gap closed' average discards relative diffs ≤0 or >1 before averaging"
severity: low
confidence: medium
status: finding
file: out/relative_diff_filter.csv
csv_row: 2
quote: |
  relative-diff-filtering-32pct,if 0 < relative_diff <= 1.0:
claim: "The notebook cell that prints 'Average relative diff' (the basis for the paper's '≈32% of the gap' claim) appends a (model,task) relative improvement to the averaged list only when `0 < relative_diff <= 1.0`, silently excluding cases where back-patching did not help (≤0) or where the visual accuracy already met/exceeded textual (denominator issues / >1)."
concern: "Conditioning the headline aggregate on positive, ≤100% gap-closing entries upward-biases the reported '32%'; the paper text does not state that non-positive or saturating cases were excluded from the average."
resolution: "Authors: report the average over all (model,task) pairs (or state the exclusion rule and how many pairs were dropped) so the 32% figure is reproducible without the implicit filter."
cross_refs: ["backpatching-best-config-on-eval-set"]
check_script: _audit_code/check_relative_diff_filter.py
paper_ref: "Section 5 ('closing approximately 32% of the performance gap')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

```yaml finding
id: backpatching-best-config-on-eval-set
category: methodology
topic: "model selection / held-out evaluation"
title: "Best back-patching (src,dst,window) chosen on the same prompts the Table 1 accuracy is reported on"
severity: high
confidence: high
status: finding
file: script_backpatching_experiment.py
line_start: 380
line_end: 389
quote: |
    vl_prompts = load_dataset(
        model=model,
        processor=processor,
        task_name=args.task_name,
        model_name=args.model_name,
        language_only=False,
        seed=args.seed,
        correct_preds_only=False,  # Important because we don't want to have 100% accuracy, but instead want to be able to improve it
        train_test_split_ratio=0.5,  # Doesn't matter, we don't split to discovery and test here
    )[0]
claim: "Back-patching scans a grid of (layer_window_size ∈ {5,3,1}) × src_layer × dst_layer and measures accuracy on `vl_prompts` (= all prompts, index [0]); the baseline accuracy is measured on the same `vl_prompts`; notebook cell 8 then reports the single maximum-accuracy configuration (`topk_2d(...,k=1)` → `sorted(...,reverse=True)[:1]`) as Table 1. There is no held-out split (the inline comment states 'we don't split to discovery and test here')."
concern: "Selecting the best of dozens-to-hundreds of layer/window configurations on the very prompts whose accuracy is then reported (and against which significance is judged) inflates the back-patching gain via selection on the evaluation set; the reported per-task improvement and the '32% gap closed' are upper bounds, not held-out estimates."
resolution: "Authors: select lsrc/ldst/window on a discovery split and report Table 1 accuracies on a disjoint evaluation split (the dataset already supports a 75/25 split used elsewhere), or quantify the optimism by reporting held-out numbers for the selected configuration."
cross_refs: ["missing-bootstrap-significance", "relative-diff-filtering-32pct"]
check_script: _audit_code/check_backpatching_selection.py
paper_ref: "Section 5; Appendix E.2 (Table 7 best layers); Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 4 | high | No bootstrap/std/significance code; no dependency spec; two JSONs hard-loaded on back-patching path absent |
| bug | 1 | low | Random-baseline index off-by-one (inclusive randint) + `vl_random_D` wrong position range |
| difference | 1 | low | "32% gap closed" average filters out non-positive / >1 relative diffs |
| methodology | 1 | high | Back-patching config selected on the same prompts Table 1 is reported on (no held-out split) |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **[methodology]** `backpatching-best-config-on-eval-set` — Table 1 / the "32% gap closed" headline uses the single best of a large layer/window grid, selected and reported on the *same* prompt set with no held-out split; the gains are optimistic upper bounds. (high/high)
2. **[missing]** `missing-bootstrap-significance` — the bootstrap that produces Table 1's "± std" and the green statistical-significance marks (and Tables 8–10) is nowhere in the released code. (high/high)
3. **[missing]** `missing-dependency-spec` — no requirements/environment file for a version-sensitive transformers + TransformerLens pipeline. (medium/high)
4. **[missing]** `missing-qa-raw-and-sentiment-json` — `qa_raw.json` / `sentiment_vl.json` are hard-loaded on the back-patching path (factual-recall baseline, sentiment parallel prompts) but absent. (medium/high)
5. **[difference]** `relative-diff-filtering-32pct` — the "≈32%" average silently drops (model,task) pairs with non-positive or saturating relative diffs. (low/medium)
6. **[bug]** `random-baseline-index-oob` — inclusive `randint` plus a `vl_random_D` position-range mix-up slightly perturb the random-baseline normalization. (low/high)

### Items that genuinely look fine
- Prompt CSVs match the paper's Table 3 counts (e.g. Qwen visual counting = 382 data rows vs. reported 383) and carry GT + model prediction columns; `correct_preds_only` filtering is principled.
- The train/discovery–eval split (`balanced_answers_train_test_split`) is answer-balanced and seeded, and circuit discovery uses discovery prompts while faithfulness uses the disjoint eval prompts (a real held-out split for the circuit analysis — the back-patching experiment is the exception).
- The faithfulness normalization (`circuit_faithfulness`) correctly uses clean and all-ablated baselines per Eq. (2), and counterfactual answers are asserted to differ from the answer.
- The TransformerLens VLM fork is vendored in-repo (`third_party/`), so the custom patching code is self-contained rather than referencing an external unspecified branch.
- The 4 `images.tar.gz` are Git-LFS pointer stubs — a fetch artefact per the manifest, not a missing-code finding.

### Open questions for the authors
- Are Table 1's reported accuracies measured on a held-out split, or on the same prompts used to pick lsrc/ldst/window? If held-out, where is that split made? (`backpatching-best-config-on-eval-set`)
- Where is the bootstrap-resampling / significance code that produced the "± std" and green marks in Tables 1, 8, 9, 10? (`missing-bootstrap-significance`)
- Were any (model, task) pairs excluded from the "32% gap closed" average, and if so how many? (`relative-diff-filtering-32pct`)
