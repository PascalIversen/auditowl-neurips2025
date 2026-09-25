# Code Audit — DePass: Unified Feature Attributing by Simple Decomposed Forward Pass (Paper 4465)

## 1. Summary

DePass is a mechanistic-interpretability method that decomposes Transformer hidden
states into additive components and propagates them through a "decomposed forward
pass" (attention scores and MLP activations frozen), yielding per-component
attribution scores. The repo (`TsinghuaC3I__Decomposed-Forward-Pass`, single commit
`568bd6d`) contains the core engine (`DePass/manager.py`, `DePass/utils.py`) plus four
experiment families matching the paper's structure: token-level output attribution
(Fig. 1, faithfulness), token-level subspace masking (Table 1, factuality),
component-level head/neuron masking (Fig. 3), and subspace-level language/semantic
decoding (Table 2). The headline quantities are attribution-faithfulness curves,
masking-accuracy tables, and decoded-token tables — there are no trained models to
release and no random data splits driving the main claims, so the audit focused on
result traceability (does code produce each reported table/figure?) and on
faithfulness between the implemented procedure and the paper's described method.

What I ran (read-only, under `_audit_code/`):
- `check_artifacts.py` — confirms (a) `result_analysis.ipynb` referenced by the
  README is absent; (b) `get_model_answer.py` (Table 1 driver) contains no
  accuracy/correctness computation; (c) `manager.py` calls an undefined method
  `get_last_layer_attribute_state`; (d) no `requirements.txt`/env file exists.
- Manual data-split intersection checks (inline `python3`) on the truthful probe and
  language probe train/test files — both are essentially disjoint (overlap 2 rows),
  so no leakage finding there.

I could not execute the model pipelines (they require multi-GPU LLM inference:
Llama-2-7B/13B, Llama-3.1-8B/70B, Qwen2), so faithfulness of the numeric outputs was
assessed by reading, not by reproduction.

## 2. Result-traceability table

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Fig. 1 / 5–9 faithfulness (Δp comprehensiveness & sufficiency curves) | `Output-Input-Attribution/get_importance_score.py` + `get_patch_result.py` compute per-sample Δp; `result_analysis.ipynb` would aggregate/plot | Δp computed; final per-K averages over dataset not produced by any present script | — | PARTIAL (aggregation notebook missing; underlying Δp present) |
| Table 1 factuality accuracy (No-Info / Misinfo / +TACS / +DePass), 6 models × 2 datasets | `Subspace-Input-Attribution/subspace-input-experiment/get_model_answer.py` | Generates answer strings only; **no accuracy/correctness scoring** | — | MISSING (no script turns answers into the % accuracies) |
| Fig. 3 / 16–18 component masking accuracy (Top-k / Bottom-k, heads & neurons) | `attention-head-attribution/get_mask_head_answer.py`, `mlp-neuron-attribution/get_mask_neuron_answer.py` | Per-sample `is_correct_masked` booleans saved per (method, mask_type, %) | plausibly (mean over booleans is the only missing step, trivial) | Verified (computation present) |
| Table 2 / 12 language vs semantic decoded tokens | `Subspace-Level-DePass-Evaluation/get_embedding.py` | Decodes top-k tokens from a language/semantic split of hidden state | code uses **mean-of-languages difference**, not the SVD-of-classifier-weights Pt the paper describes (§4.3, App. E.2) | MISMATCH (method differs; see `subspace-projection-mismatch`) |
| Fig. 4 / 27 t-SNE of language subspace | (plotting only; embeddings from `get_embedding.py`) | embeddings computed | — | N/A (visualization) |
| Table 3/4 softmax vs alternative MLP normalizations | `manager.py` `mlp_decomposed_compute_{softmax,linear,relu,max,taylor}` | normalization variants implemented; the patch-top/recover-top % values aggregated by absent notebook | — | PARTIAL (variants present; aggregation absent — same as Fig. 1) |
| Table 11 runtime (DePass vs ablation neuron attribution) | (none) | no timing/benchmark script in repo | — | MISSING (no runtime-comparison script) |
| Fig. 12–14 / 26 probe accuracy curves | `classifier-training/eval_classifier.py`, `language_probing/eval_classifier.py` | per-layer test accuracy computed | — | Verified (computation present) |
| Appendix C.5 DePass+SAE attribution tables 8–10 | (none) | no SAE-attribution script in repo | — | MISSING (case studies not in code) |

## 3. Findings

### missing

```yaml finding
id: table1-accuracy-no-script
category: missing
topic: "result traceability"
title: "No script computes Table 1 factuality accuracies; driver only saves answer strings"
severity: high
confidence: high
status: finding
file: Input-Level-DePass-Evaluation/Subspace-Input-Attribution/subspace-input-experiment/get_model_answer.py
line_start: 282
line_end: 302
quote: |
        for prompt_type in prompt_all:
            prompt = data[prompt_type]
            result[prompt_type] = prompt
            result[f"{prompt_type}_answer"] = generate_answer(model, tokenizer, prompt, device)

            if prompt_type == "prompt_wrong":
                mask_self, mask_decompose, mask_random = mask_token(
                    prompt, AttrStateManager, classifiers,
                    args.classifier_start_layer, args.classifier_end_layer,
                    device, args.max_mask_percent, classifier_bound=args.classifier_bound
                )
                result[f"{prompt_type}_self_based_mask"] = mask_self
                result[f"{prompt_type}_self_based_mask_answer"] = generate_answer(model, tokenizer, mask_self, device)

                result[f"{prompt_type}_DePass_based_mask"] = mask_decompose
                result[f"{prompt_type}_DePass_based_mask_answer"] = generate_answer(model, tokenizer, mask_decompose, device)

                result[f"{prompt_type}_random_based_mask"] = mask_random
                result[f"{prompt_type}_random_based_mask_answer"] = generate_answer(model, tokenizer, mask_random, device)

        results.append(result)
claim: "The only driver for the Table 1 experiment (No Info / Misinfo / +TACS / +Ours masking) generates and stores raw model answer strings (`*_answer`, `*_mask_answer`) but never compares them to the ground-truth target/correct_option, so no accuracy is computed."
concern: "Table 1 is a headline result (e.g. the abstract-level claim that DePass masking raises Llama-2-7b accuracy from 10.16% to 43.13%), yet the repo contains no code that turns the saved answers into the reported percentages, so the central numbers are not reproducible from the released code."
resolution: "Authors: please add (or point to) the scoring script that maps the saved `*_answer` fields to the per-setting accuracies in Table 1, including the exact answer-matching rule used for CounterFact (target) and TruthfulQA (correct_option)."
cross_refs: []
check_script: _audit_code/check_artifacts.py
paper_ref: "Table 1; Section 4.1.2"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: runtime-table-no-script
category: missing
topic: "result traceability"
title: "Table 11 runtime comparison (DePass vs ablation) has no timing script in repo"
severity: low
confidence: high
status: finding
file: README.md
line_start: 127
line_end: 131
quote: |
  ### Model Component-Level DePass Evaluation

  Experiments for **4.2 Model Component-Wise DePass**, decomposing model components such as attention heads and MLP neurons.
  - `Model-Component-Level-DePass-Evaluation/attention-head-attribution/get_mask_head_answer.py`: Evaluates importance of attention heads.  
  - `Model-Component-Level-DePass-Evaluation/mlp-neuron-attribution/get_mask_neuron_answer.py`: Evaluates importance of MLP neurons.  
claim: "The repo lists only head/neuron masking evaluation scripts for §4.2; no script measures or compares the DePass-vs-ablation neuron-attribution runtimes reported in Table 11 (e.g. 321.04s ablation vs 7.22s DePass on Llama-2-7B)."
concern: "The two-orders-of-magnitude speedup is an explicit efficiency claim but cannot be reproduced or checked from the released code."
resolution: "Authors: provide the benchmarking harness used to produce Table 11 (which layer, prompt length, warm-up, repetitions, and the exact ablation baseline timed)."
cross_refs: []
check_script: _audit_code/check_artifacts.py
paper_ref: "Table 11 (Appendix D.2)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: result-analysis-notebook-missing
category: missing
topic: "result traceability"
title: "README-referenced result_analysis.ipynb (aggregates/plots Fig. 1) is absent"
severity: low
confidence: high
status: finding
file: README.md
line_start: 118
line_end: 118
quote: |2
  - `result_analysis.ipynb`: Visualizes attribution and ablation results.  
claim: "The README's Output-Input-Attribution section lists `result_analysis.ipynb` as the script that visualizes attribution/ablation results, but no such file exists anywhere in the repo (only get_importance_score.py and get_patch_result.py are present)."
concern: "The per-K averaging that turns get_patch_result.py's per-sample Δp into the Fig. 1/5–9 comprehensiveness and sufficiency curves is not in the repo; the underlying per-sample Δp values are computed, so this is a plotting/aggregation gap rather than a missing core computation."
resolution: "Authors: add the missing result_analysis.ipynb (or equivalent) that aggregates per-sample Δp across the dataset into the reported curves."
cross_refs: ["table1-accuracy-no-script"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Figure 1; README File Structure"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: requirements-missing
category: missing
topic: "dependencies / environment"
title: "No requirements.txt / environment file; only three pinned packages in README prose"
severity: low
confidence: high
status: finding
file: README.md
line_start: 10
line_end: 24
quote: |2
  Tested with the following major packages:

  -`torch==2.4.1+cu121`

  -`transformers==4.44.2`

  -`numpy==1.26.3`

  Ensure GPU support (CUDA 12.1) is available for best performance.

  ```bash

  pip install torch==2.4.1+cu121 transformers==4.44.2 numpy==1.26.3

  ```
claim: "The repo ships no requirements.txt / environment.yml / pyproject; the README pins only torch, transformers and numpy, while the scripts additionally import scikit-learn, matplotlib and tqdm with no version pins."
concern: "scikit-learn's LogisticRegression default behaviour (e.g. removal of the `multi_class` argument used in `language_probing/classifier.py`) and other unpinned deps can change results or break import across versions, making the environment not fully rebuildable."
resolution: "Authors: add a pinned requirements file covering scikit-learn, matplotlib, tqdm (and any others) at the tested versions."
cross_refs: []
check_script: _audit_code/check_artifacts.py
paper_ref: "README Environment Setup"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: generate-cite-broken-method
category: bug
topic: "API / token-level generation attribution"
title: "model_generate_cite calls undefined self.get_last_layer_attribute_state -> AttributeError"
severity: low
confidence: high
status: finding
file: DePass/manager.py
line_start: 569
line_end: 569
quote: |
        attribute_state, states = self.get_last_layer_attribute_state(generated_text)
claim: "model_generate_cite (the generation-based token-attribution entrypoint, lines 482-580) calls self.get_last_layer_attribute_state, but the class only defines get_last_layer_decomposed_state (line 148); there is no get_last_layer_attribute_state method, so any call to model_generate_cite raises AttributeError."
concern: "The documented generation-attribution API used to produce the multi-output token-attribution examples (e.g. Appendix B.3, 'Tom handed a bag to' -> 'Amy'/'a') cannot run as written; it is never exercised by the main experiment scripts, which is why the headline numbers are unaffected."
resolution: "Rename the call to get_last_layer_decomposed_state (and align return signature), then add a smoke test that runs model_generate_cite end-to-end."
cross_refs: []
check_script: _audit_code/check_artifacts.py
paper_ref: "Appendix B.3"
tags: [lones:stage-4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: subspace-projection-mismatch
category: difference
topic: "subspace-level decomposition (Table 2)"
title: "Language subspace built from mean-of-languages difference, not the SVD classifier projection the paper describes"
severity: medium
confidence: high
status: finding
file: Subspace-Level-DePass-Evaluation/get_embedding.py
line_start: 141
line_end: 159
quote: |
        mean_state = torch.zeros(model.config.hidden_size).to(model.device)
        for lang, hidden_states in hidden_states_all.items():
            mean_state += hidden_states_all[lang][layer_idx][0][-1]
        mean_state /= len(hidden_states_all)
        for lang,prompt_data in data.items():
            if lang == "case_id":
                continue
            prompt = prompt_data["prompt"]
            answer = prompt_data["answer"]
            hidden_states = hidden_states_all[lang]
            attribute_state=torch.zeros(2,hidden_states[layer_idx].shape[1],hidden_states[layer_idx].shape[2]).to(model.device)
            attribute_state[0, :] = hidden_states[layer_idx][0].clone()
            language_embedding=attribute_state[0, -1] - mean_state
            attribute_state[0, :] = attribute_state[0, :] - language_embedding
            attribute_state[1, :] = language_embedding
claim: "The subspace-level decoding experiment (Table 2 language/semantic tokens) defines the 'language' component as the last-token hidden state minus the across-languages mean (`language_embedding = h_last - mean_state`), and the 'semantic' component as the remainder; it does NOT load the trained language classifier nor build the SVD projection matrix Pt = Ur Ur^T from the classifier weights as described in Section 4.3 and Appendix E.2."
concern: "The paper attributes Table 2's clean language/semantic separation to projecting onto the language-classifier subspace, but the released code uses a different (mean-difference) construction; both are individually reasonable, yet the reported result is not produced by the method the paper documents, so a reader reproducing 'the SVD classifier projection' would get different decompositions."
resolution: "Authors: confirm which construction produced Table 2 / Fig. 4, and either release the classifier-projection (Pt) version used for the paper or update §4.3/App. E.2 to describe the mean-difference construction in get_embedding.py."
cross_refs: []
check_script: _audit_code/check_artifacts.py
paper_ref: "Section 4.3; Appendix E.2; Table 2"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

N/A — no methodology finding. The two probe train/test splits that feed any
quantitative claim (truthful probe: `classifier-training/truthful/{train,test}_data.json`;
language probe: `language_probing/data/{train,test}_dataset.json`) are essentially
disjoint (set-intersection of 2 rows in each, verified inline), the probes are trained
on train and evaluated on a held-out test set, and the headline masking/faithfulness
experiments are attribution-quality evaluations on a fixed pretrained model with greedy
decoding (no tuned hyperparameters touching a test set). The masking baselines (TACS,
random, Norm/Coef/AtP) are run under the same split, metric and budget as DePass.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Table 1 accuracies have no scoring script; runtime table, aggregation notebook, and requirements file also absent. |
| bug         | 1          | low          | Unused generation API calls an undefined method (AttributeError). |
| difference  | 1          | medium       | Table 2 language subspace built by mean-difference, not the described SVD classifier projection. |
| methodology | 0          | -            | Probe splits disjoint; baselines run under matched conditions; no leakage/tuning-on-test found. |

## 5. Closing lists

### Top take-aways (ranked)
1. **[missing] `table1-accuracy-no-script`** — the headline factuality table (Table 1) has no accuracy-scoring code; the driver only saves raw answer strings. High severity, high confidence.
2. **[difference] `subspace-projection-mismatch`** — Table 2 / Fig. 4 language–semantic split uses a mean-of-languages difference, not the SVD-of-classifier-weights projection Pt the paper describes. Medium / high.
3. **[missing] `runtime-table-no-script`** — Table 11's 100x speedup claim has no benchmarking script. Low / high.
4. **[missing] `result-analysis-notebook-missing`** — README-referenced aggregation/plotting notebook for Fig. 1 is absent (underlying Δp values are computed). Low / high.
5. **[bug] `generate-cite-broken-method`** — `model_generate_cite` calls a non-existent method; never used by the main experiments. Low / high.
6. **[missing] `requirements-missing`** — no pinned dependency file beyond three README-prose packages. Low / high.

### Items that genuinely look fine
- Core decomposed forward pass (`attn_decomposed_compute`, `mlp_decomposed_compute_softmax`) implements the additive attention/MLP propagation of Eqs. 12–15, including the RMSNorm per-component scaling (Eq. 11) via `get_rmsnorm_scaling`.
- Component-level head/neuron masking scripts (Fig. 3) compute per-sample correctness via `check_answer_match`, only the trivial mean is left to the reader.
- Truthful probe and language probe each use a held-out, essentially disjoint test set (verified: 2-row overlap each); probes trained on train and evaluated on test.
- Baselines (TACS, random masking, Input×Grad/IG/GradSHAP, Mean/Last/Rollout attention, Norm/Coef/AtP) are implemented and run under the same protocol as DePass.
- Multiple MLP-normalization variants (softmax/linear/relu/max/taylor) for the Table 3/4 ablation are all present in `manager.py`.

### Open questions for the authors
- Was Table 1 scored by exact-prefix match (like the component-level `check_answer_match`), and on which generated span? The absence of scoring code leaves the matching rule unspecified.
- Which subspace construction (classifier SVD projection vs the mean-difference in `get_embedding.py`) generated the published Table 2 and Fig. 4?
