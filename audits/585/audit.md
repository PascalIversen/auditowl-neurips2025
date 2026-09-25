# Code-repository audit — ReplaceMe (NeurIPS 2025, paper #585)

## 1. Summary

ReplaceMe is a training-free depth-pruning method for transformers: select a
contiguous block of layers via an activation-distance metric, estimate a linear
transformation (analytic least-squares or numerical cosine/Adam) that
approximates the pruned block from a small calibration set, then fuse that
transform into the preceding MLP down-projection. The repo
(`code/mts-ai__ReplaceMe/`) is a small installable Python package
(`ReplaceMe/`) exposing console scripts for distance profiling
(`distance.py`), LS estimation (`lstsq.py`), numerical estimation
(`cosine_dist.py`), a UIDL baseline (`uidl_method.py`/`UIDL_pipeline.py`), an
end-to-end pipeline (`ReplaceMe_pipeline.py`), and lm-eval-based evaluation
(`evaluator.py` + `utils.py`). Config YAMLs live in `examples/` and `reproduce/`.

I read every Python module and YAML, the paper PDF (method §2, experiments §3,
Tables 1–6, Figs 2–3, reproducibility checklist), and ran deterministic checks
under `_audit_code/` (`check_repo_structure.py`): it confirms the package has no
`__init__.py` (only a misnamed `init.py`), that `adam_method` references an
attribute `triangular_weight` that `LowerTriangularLinear` never defines, that
the calibration-dataset loader has no handler for the paper's `orca_generated`
or Aya/66-language sources, and that no file other than `setup.py` mentions
CLIP/ViT/vision data. The math of the default cosine objective was checked by
hand against Eq. 8 and is correct. The code is GPU-only (`device='cuda'`
hardcoded in the estimators) and depends on gated HF models, so I could not
execute the full pipeline; findings are from static reading plus the
deterministic checks above.

## 2. Traceability table

The repo is a method library, not an experiment-runner: it produces a single
pruned model + benchmark JSON per config invocation. It contains no scripts that
assemble the comparison tables/figures, no baseline numbers (most are cited from
[3] with `*`), and no ViT/CLIP path. "Computed value" below is left "—" because
running requires gated models + GPU, which is out of scope (READ-ONLY, no
execution possible here).

| Paper artefact | Repo location | Computed value | Matches | Status |
|---|---|---|---|---|
| Table 1 — Ours (Cosine)/(LS) on Llama-2-7B, benchmarks C3/CMNLI/CHID/WSC/HellaSwag/PIQA/Race-M/Race-H/MMLU/CMMLU | `ReplaceMe_pipeline.py` produces the model; eval via `utils.py:eval_model` covers only boolq/race/openbookqa/piqa/sciq/lambada/winogrande — the Table-1 benchmark set is **not** wired into any default eval | — | — | PARTIAL — eval of Table-1 tasks only reachable through `eval_model_specific` custom config; no driver provided (see `table1-eval-tasks-not-wired`) |
| Table 2 — Llama-3-8B-Instruct, Avg-acc/ppl, LS & Cosine & Multi_LT_NC | `ReplaceMe_pipeline.py` (lstsq/cosine) + `utils.py:eval_model` (exact task set matches) | — | — | Computation path present; Multi_LT_NC requires manual `num_A`/`merge_consecutive` config (no preset) |
| Table 2 / Table 1 — baseline rows (UIDL, SVD-LLM, LLMPruner, SliceGPT, LaCo, LLM-Streamline) | UIDL only (`uidl_method.py`); others absent (cited from [3]) | — | — | MISSING baseline code (acknowledged via `*`; cross-ref `missing-baseline-implementations`) |
| Table 3 — accuracy at 12.5/25/37.5/50% compression | Achievable by varying `layers_to_skip`; no preset configs/driver | — | — | PARTIAL (no driver) |
| Table 4 — calibration-data ablation (fineweb / slim_orca / orca_generated / mix / 66-lang) | `utils.py:get_calib_dataloader` handles fineweb, SlimOrca, `fineweb_and_orca`; **no** `orca_generated`, **no** 66-language handler | — | — | MISSING data + loader (see `missing-orca-generated-handler`) |
| Table 5 — regularization sweep (α grid; L1/L2; cosine & LS) | LS α via `lstsq.py` `alpha`; cosine L1/L2 reg — **not implemented** in `adam_method` (only mse/cosine/elasticnet losses, no α-scaled L1/L2 term) | — | — | MISSING cosine-reg code (see `missing-cosine-regularization`) |
| Table 6 — CLIP/ViT pruning (MS-COCO, Cifar10, VOC2007, VTAB/EuroSAT) | none (no vision code anywhere; check confirms) | — | — | MISSING (see `missing-vit-clip-code`) |
| Fig. 2 — time/CO2/relative-accuracy vs UIDL | CodeCarbon mentioned in paper; no emission-logging or plotting code in repo | — | — | MISSING (figure data + plotting) |
| Fig. 3 — acc vs #pruned-layers, 3 models | no sweep driver / plotting code | — | — | MISSING (figure data) |
| §A.10 brute-force layer selection; A.16 Multi_LT_NC analysis; A.6 solver ablation; A.3 structured (diag/orthonormal/triangular) LTs | partial: `diag`/`two_vectors`/`thri` flags + `solver` options exist; `thri` path is **broken** (`thri-attribute-bug`); brute-force selection script absent | — | — | PARTIAL / MISSING |
| Checklist claim: "ran with different random seeds … std ≈ 0.08" | `utils.py:seed_all(seed=42)` hardcoded; called at import; no seed CLI/loop | — | — | MISSING multi-seed harness (see `single-fixed-seed-no-sweep`) |

## 3. Findings

## missing

```yaml finding
id: missing-vit-clip-code
category: missing
topic: "result traceability / vision experiments"
title: "No code for the CLIP/ViT pruning experiments (Table 6, §3.4)"
severity: high
confidence: high
status: finding
file: ReplaceMe/utils.py
line_start: 67
line_end: 92
quote: |
  def get_calib_dataloader(
      dataset: str,
      dataset_subset: str,
      dataset_column: str,
      dataset_size: Optional[int],
      batch_size: int,
      tokenizer: PreTrainedTokenizerBase
  ) -> DataLoader:
      """Load and prepare calibration dataset."""
      dataset_handlers = {
          'HuggingFaceFW/fineweb': lambda: datasets.load_dataset(dataset, name='sample-10BT', split=dataset_subset),
          'allenai/c4': lambda: datasets.load_dataset(dataset, 'en', split=dataset_subset),
          'arcee-ai/sec-data-mini': lambda: datasets.load_dataset(dataset, split=dataset_subset),
          'wikitext': lambda: datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=dataset_subset),
          'Open-Orca/SlimOrca': lambda: _load_orca_dataset(dataset_size, tokenizer),
          'fineweb_and_orca': lambda: _load_mixed_dataset(dataset_size, dataset_subset, tokenizer)
      }
claim: "The entire codebase is LLM/decoder-only: model loading uses AutoModelForCausalLM, layer access assumes model.model.layers (or falcon transformer.h), the calibration loader handles only text corpora, and evaluation uses lm-eval text tasks. No module loads a CLIP/ViT model, MIMIC/MS-COCO/Cifar10/EuroSAT/VOC data, or computes retrieval recall@5 / zero-shot accuracy."
concern: "Table 6 and §3.4 report a full set of vision-encoder pruning results (a stated contribution — generality to ViT), but no code in the repo can produce any of those numbers."
resolution: "Authors: please add the CLIP/ViT pruning and evaluation scripts (model loading, MIMIC calibration, MS-COCO/Cifar10/VOC/VTAB evaluation) used for Table 6, or state that vision code is out of scope of the released library."
cross_refs: ["§3.4", "Table 6"]
check_script: _audit_code/check_repo_structure.py
paper_ref: "Table 6, Section 3.4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-orca-generated-handler
category: missing
topic: "calibration data"
title: "No loader for orca_generated (self-generated) calibration data used in Table 4"
severity: medium
confidence: high
status: finding
file: ReplaceMe/utils.py
line_start: 76
line_end: 86
quote: |
      dataset_handlers = {
          'HuggingFaceFW/fineweb': lambda: datasets.load_dataset(dataset, name='sample-10BT', split=dataset_subset),
          'allenai/c4': lambda: datasets.load_dataset(dataset, 'en', split=dataset_subset),
          'arcee-ai/sec-data-mini': lambda: datasets.load_dataset(dataset, split=dataset_subset),
          'wikitext': lambda: datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=dataset_subset),
          'Open-Orca/SlimOrca': lambda: _load_orca_dataset(dataset_size, tokenizer),
          'fineweb_and_orca': lambda: _load_mixed_dataset(dataset_size, dataset_subset, tokenizer)
      }

      if dataset not in dataset_handlers:
          raise ValueError(f"Dataset {dataset} not implemented")
claim: "get_calib_dataloader supports only the six listed dataset keys and raises ValueError otherwise. There is no handler for the paper's 'orca_generated' (model-self-generated) data nor for the Aya / 66-language mix used in Table 4."
concern: "Table 4 reports orca_generated and a 66-language mix as calibration sources, and the reproducibility checklist says orca_generated is 'provided with the supplementary materials', but the released code cannot load it (it is neither bundled in the repo nor wired into the loader), so those rows are not reproducible from the repo."
resolution: "Authors: add a loader entry and ship (or link) the orca_generated and Aya-mix calibration files, or document the exact generation procedure and dataset key."
cross_refs: ["Table 4"]
check_script: _audit_code/check_repo_structure.py
paper_ref: "Table 4; Reproducibility checklist Q5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-cosine-regularization
category: missing
topic: "regularization (Table 5)"
title: "L1/L2 regularization for the cosine objective is not implemented"
severity: medium
confidence: medium
status: finding
file: ReplaceMe/utils.py
line_start: 254
line_end: 260
quote: |
      loss_fn = {
          "cosine": cosine_loss,
          "mse": nn.MSELoss(reduction='mean'),
          "elasticnet": lambda XA, Y: nn.MSELoss(reduction='mean')(XA, Y) + \
                                     0.09 * torch.norm(XA, p=1) + \
                                     0.045 * torch.norm(XA, p=2)**2
      }[loss]
claim: "The Adam (cosine) estimator has no regularization hyperparameter: adam_method takes no alpha, and the only penalized loss ('elasticnet') hardcodes coefficients 0.09/0.045 applied to the activations XA, not to the transform T. The paper's §2.3 states L1/L2 regularization is applied to T for the cosine objective, and Table 5 reports a cosine + L1 (alpha=1e-4) and cosine + L2 (alpha=0.01) sweep."
concern: "Table 5's cosine-objective regularization rows (and the §2.3 claim of L1/L2 penalties on T under the cosine objective) have no corresponding code path; the LS objective's alpha (Ridge) is the only regularizer wired (lstsq.py:221)."
resolution: "Authors: point to the code that applies alpha-scaled L1/L2 to T under the cosine objective, or add it; clarify whether Table 5 cosine rows used a different (unreleased) script."
cross_refs: ["Table 5", "§2.3"]
paper_ref: "Section 2.3; Table 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-baseline-implementations
category: missing
topic: "baselines"
title: "Only UIDL baseline implemented; SVD-LLM/LLMPruner/SliceGPT/LaCo/LLM-Streamline absent"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  * indicates that the numbers are taken from streamline paper [3].
claim: "The repo implements ReplaceMe and a single competing baseline (UIDL, uidl_method.py). All other competitors in Tables 1-2 (LLM-Streamline, LLMPruner, SliceGPT, LaCo, SVD-LLM) have no code in the repo."
concern: "Most baseline numbers are imported from another paper [3] rather than recomputed under the same calibration/eval pipeline, so apples-to-apples comparison cannot be verified from this repo (the authors disclose this with the '*' marker, lowering severity)."
resolution: "Authors: confirm baselines were evaluated under the identical 25%-compression, calibration, and lm-eval harness; if numbers are quoted from [3], note any setup differences (few-shot counts, eval version)."
cross_refs: ["Table 1", "Table 2"]
paper_ref: "Table 1 caption ('* taken from streamline paper [3]')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: single-fixed-seed-no-sweep
category: missing
topic: "statistical stability / reproducibility"
title: "No multi-seed harness despite claimed seed-variation stability study"
severity: low
confidence: high
status: finding
file: ReplaceMe/utils.py
line_start: 26
line_end: 36
quote: |
  def seed_all(seed: int = 42):
      """Seed all major sources of randomness for reproducibility."""
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

      print(f"Seeded with seed: {seed}")
claim: "seed_all hardcodes seed=42 and is called once at module import in every pipeline; there is no CLI/config option to vary the seed and no loop that runs the method across seeds."
concern: "The paper (§3.2 'all experiments are executed multiple times') and the reproducibility checklist ('ran with different random seeds … std ≈ 0.08') describe a multi-seed stability study, but the released code provides no harness to vary the seed or aggregate runs, so that std cannot be reproduced from the repo."
resolution: "Authors: expose the seed as a config parameter and provide the multi-run/aggregation script used to compute the reported standard deviation."
cross_refs: ["§3.2", "Reproducibility checklist Q4"]
paper_ref: "Section 3.2; Reproducibility checklist Q4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-init-py
category: missing
topic: "packaging / repository completeness"
title: "Package has no __init__.py (only a misnamed init.py with a circular import)"
severity: low
confidence: high
status: finding
file: ReplaceMe/init.py
line_start: 1
line_end: 9
quote: |
  __version__ = "0.1.0"

  from ReplaceMe import (
      ReplaceMe_pipeline,
      cosine_dist,
      distance,
      evaluator,
      lstsq
      )
claim: "The package directory ReplaceMe/ contains init.py, not __init__.py (confirmed by check_repo_structure.py). As written it is never imported as the package initializer, and its body 'from ReplaceMe import ...' would be a self-import that fails if it ever were the initializer. The console-script entry points in setup.py reference submodules directly, so they still work, but 'import ReplaceMe' exposes none of these names."
concern: "Anyone using the library as 'import ReplaceMe; ReplaceMe.lstsq(...)' (the documented public surface) gets an empty namespace; the file is dead/broken code, a minor completeness defect."
resolution: "Rename init.py to __init__.py and replace the self-import with relative imports (e.g. 'from .lstsq import lstsq')."
cross_refs: []
check_script: _audit_code/check_repo_structure.py
paper_ref: ""
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: thri-attribute-bug
category: bug
topic: "structured linear transform (triangular)"
title: "adam_method returns model.triangular_weight, an attribute never defined"
severity: medium
confidence: high
status: finding
file: ReplaceMe/utils.py
line_start: 290
line_end: 290
quote: |
      return model.triangular_weight.T.to(torch.float64) if thri else model.weight.T.to(torch.float64)
claim: "When thri=True, adam_method builds a LowerTriangularLinear (which defines only self.weight, line 164) and trains it, but at return time accesses model.triangular_weight, an attribute that LowerTriangularLinear never sets (confirmed by AST scan in check_repo_structure.py: attributes = ['weight'])."
concern: "Any run with the lower-triangular structured-LT option (thri=True, the §A.3 structured-transform ablation) crashes with AttributeError at the end of estimation, so that ablation cannot be produced by this code."
resolution: "Return torch.tril(model.weight).T (or model.weight.T) for the thri branch; confirm whether the A.3 triangular-LT results were produced by this code path."
cross_refs: ["§A.3"]
check_script: _audit_code/check_repo_structure.py
paper_ref: "Appendix A.3 (structured transforms)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-install-wrong-dir
category: bug
topic: "documentation / install instructions"
title: "README install step cd's into nonexistent 'patchme' directory"
severity: low
confidence: high
status: finding
file: readme.md
line_start: 18
line_end: 22
quote: |
  ```bash
  git clone https://github.com/mts-ai/ReplaceMe.git
  cd patchme
  pip install -e .
  ```
claim: "The documented install sequence clones into 'ReplaceMe' but then 'cd patchme', a directory that does not exist; pip install -e . then runs in the wrong/missing directory."
concern: "Following the README verbatim fails at the install step; minor but it is the single documented entry to the whole pipeline."
resolution: "Change 'cd patchme' to 'cd ReplaceMe'."
cross_refs: []
paper_ref: ""
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: table1-eval-tasks-not-wired
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Default evaluator omits the Table-1 benchmark suite (CMNLI/CHID/MMLU/CMMLU/C3/HellaSwag/WSC)"
severity: low
confidence: medium
status: question
file: ReplaceMe/utils.py
line_start: 304
line_end: 316
quote: |
      wino_res = evaluator.simple_evaluate(
          model='hf',
          tasks=['winogrande'],
          model_args=f"pretrained={model_path},dtype=bfloat16,device=auto,parallelize=False,device_map=auto",
          num_fewshot=5
      )['results']
      
      other_res = evaluator.simple_evaluate(
          model='hf',
          tasks=['boolq', 'race', 'openbookqa', 'piqa', 'sciq', 'lambada_openai'],
          model_args=f"pretrained={model_path},dtype=bfloat16,device=auto,parallelize=False,device_map=auto",
          num_fewshot=0
      )['results']
claim: "The default eval_model path evaluates only winogrande/boolq/race/openbookqa/piqa/sciq/lambada_openai — exactly the Table-2 suite. The Table-1 suite (C3, CMNLI, CHID, WSC, HellaSwag, PIQA, Race-M/H, MMLU, CMMLU) is never wired into a default; only the generic eval_model_specific (arbitrary lm-eval task dict) could compute them, and no config preset for the Table-1 tasks is shipped."
concern: "The Table-1 'Ours (Cosine)/(LS)' numbers (the authors' own rows) require a different evaluation configuration than anything the repo ships; the two reported benchmark suites differ and only one is reproducible out of the box."
resolution: "Authors: ship the lm-eval task config (task names, few-shot counts) used for the Table-1 Chinese/HellaSwag/MMLU suite so those self-computed rows are reproducible."
cross_refs: ["Table 1", "missing-baseline-implementations"]
paper_ref: "Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: min-distance-layer-unused
category: difference
topic: "layer selection"
title: "Documented min_distance_layer (cut-start constraint) is never used in selection"
severity: low
confidence: high
status: finding
file: ReplaceMe/utils.py
line_start: 354
line_end: 378
quote: |
  def select_non_overlapping_blocks(
      average_distances: List[float],
      layers_to_skip: int,
      num_blocks: int = 4,
      merge_consecutive: bool = False
  ) -> List[Tuple[int, int]]:
      """Select optimal non-overlapping layer blocks based on distances."""
      blocks = [
          (i + 1, i + layers_to_skip + 1, avg)
          for i, avg in enumerate(average_distances)
      ]
      
      # Sort by distance and select non-overlapping
      selected = []
      used_layers = set()
      
      for block in sorted(blocks, key=lambda x: x[2]):
          start, end, _ = block
          block_layers = set(range(start, end))
          
          if not block_layers & used_layers:
              selected.append(block)
              used_layers.update(block_layers)
              if len(selected) >= num_blocks:
                  break
claim: "min_distance_layer is documented in lstsq/cosine_dist/distance as 'index of the layer to start the cut' and set to 20 in the reproduce configs, but select_non_overlapping_blocks (the actual block chooser) does not accept or use it; selection is purely the global argmin over all candidate start indices. In profile_distances, min_distance_layer is a parameter but is immediately overwritten (distance.py:152) and used only for a log message."
concern: "A reader configuring min_distance_layer=20 to constrain the cut would have no effect; the cut index is chosen automatically, so the parameter's documented behavior does not match the code (the code's own automatic selection is valid, hence difference not bug)."
resolution: "Authors: either honor min_distance_layer as a lower bound in select_non_overlapping_blocks or remove it from the configs/docstrings to avoid implying a constraint that is not applied."
cross_refs: []
paper_ref: "Section 2.1 (cut-index selection)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A as a finding source for the core estimator: the implemented procedure
matches the paper's math. I verified by hand that the default cosine path
(`cosine_dist.py:189-190`, `accurate=False`) optimizes
`cos(M_i·T, L_{i+n} + M_i - L_i) = cos(M_i·T, L_{i+n} - Y_i)` (since
`Y_i = L_i - M_i`), which is exactly Eq. 8; the LS closed form
(`lstsq.py:222`, `(MᵀM + αI)⁻¹ Mᵀ(L_{i+n} - Y_i)`) matches Eq. 6; layer
selection by angular distance is monotone in cosine distance so the argmin cut
matches Eq. 4. Calibration data is used only to fit the linear transform (no
labels, no held-out leakage concern in the conventional sense). No methodology
finding is filed.

The §A.17 task-specific-calibration result (calibrating on SciQ then evaluating
on SciQ) is a potential train/eval-overlap concern, but it is presented as an
auxiliary observation, the calibration is unsupervised (no labels used in the
LT fit), and there is no calibration/eval script for it in the repo to inspect —
so it does not rise to a code-grounded methodology finding here.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 6          | high         | Vision (Table 6) experiments have no code; orca_generated/Aya data + cosine-reg + multi-seed harness absent |
| bug         | 2          | medium       | thri (triangular-LT) path crashes on undefined attribute; README install dir wrong |
| difference  | 2          | low          | Table-1 eval suite not wired into any default; min_distance_layer documented but unused |
| methodology | 0          | -            | Core LS/cosine estimators match the paper's equations; no methodology defect found |

## 5. Closing lists

**Top take-aways** (≤6, by severity × confidence):
1. `missing-vit-clip-code` (missing, high/high) — Table 6 / §3.4 ViT-CLIP pruning results have no code in the repo at all.
2. `missing-orca-generated-handler` (missing, medium/high) — the self-generated calibration data (Table 4) is neither bundled nor loadable.
3. `thri-attribute-bug` (bug, medium/high) — the lower-triangular structured-LT ablation crashes (`model.triangular_weight` undefined).
4. `missing-cosine-regularization` (missing, medium/medium) — §2.3/Table 5 cosine-objective L1/L2 regularization has no code path.
5. `single-fixed-seed-no-sweep` (missing, low/high) — claimed multi-seed stability (std≈0.08) has no harness; seed hardcoded to 42.
6. `missing-baseline-implementations` (missing, low/high) — only UIDL is implemented; other baselines are quoted from [3], so the comparison cannot be re-run here.

**Items that genuinely look fine**:
- Core cosine objective (`cosine_dist.py:189-190`) algebraically equals Eq. 8.
- LS closed form with Ridge α (`lstsq.py:221-222`) matches Eq. 6 and §2.3 (LS regularization).
- Transform fusion into `mlp.down_proj` (`lstsq.py:242-246`, `cosine_dist.py:233-235`) implements the "fuse, no extra parameters" claim.
- Default eval task set (`utils.py:304-316`) exactly matches the Table-2 benchmark suite and few-shot settings stated in the Table-2 caption.
- Dependencies are pinned (`setup.py`) including `lm-eval==0.4.8`, `transformers==4.46.3`.
- Calibration data is used label-free to fit the LT; no conventional train/test leakage in the pruning procedure.

**Open questions for the authors**:
- Were the Table-1 Chinese/HellaSwag/MMLU rows for "Ours" produced with `eval_model_specific`? Please ship that task config (`table1-eval-tasks-not-wired`).
- Which script produced Table 5's cosine + L1/L2 rows, given the released `adam_method` has no T-regularizer (`missing-cosine-regularization`)?
- Was `min_distance_layer=20` in the reproduce configs intended to constrain the cut? It is currently inert (`min-distance-layer-unused`).
