# Audit: Table2LaTeX-RL (NeurIPS 2025, paper 2007)

## 1. Summary

The paper proposes **VSGRPO**, a dual-reward (visual CW-SSIM + structural TEDS-Structure)
GRPO fine-tuning recipe for converting table images to LaTeX, built on Qwen2.5-VL-3B and
InternVL2-1B. Reported metrics are CW-SSIM, compile ratio, TEDS and TEDS-Structure,
stratified into simple/medium/complex table subsets (Tables 1, 2, 5, 6, 7) plus an external
benchmark (Table 3) and a human study (Table 4).

The repository (`code/newLLing__Table2LaTeX-RL/`) is a fork of the **ms-swift** training
framework. Only a handful of files are paper-specific:
- `examples/train/grpo/plugin/plugin.py` — the VSGRPO reward functions (`Table2LatexAcc`,
  `Table2Latexform`).
- `examples/train/grpo/plugin/run_external_rm.sh` — the GRPO training command.
- `qwenvl_test.py` — distributed inference that produces `prediction`/`reference` JSONL.
- `cw_ssim.ipynb` — the evaluation notebook that computes the reported CW-SSIM / TEDS /
  TEDS-Structure numbers from that JSONL.
- `arxiv_papers_get.py`, `table.ipynb` — dataset-construction scripts (download + table
  extraction/cleaning).

What I did: read all paper-specific files; mapped each reported metric to the notebook
function that computes it; verified the model and evaluation dataset links resolve on
Hugging Face (model `LLLHHH/Table2Latex-RL`, dataset `LLLHHH/Table2LaTeX-RL` = 1,211 rows =
496+354+361 test tables); ran a deterministic check (`_audit_code/check_nonzero_averaging.py`)
reproducing the notebook's averaging semantics; and searched the repo for the table-complexity
classification logic described in the paper.

The code is essentially a re-implementation of inference + metrics that consumes
**pre-generated outputs / provided model weights**; it is not a one-command reproduction of the
headline numbers. The two most consequential observations are (a) reported CW-SSIM/TEDS are
averaged over compile-successful predictions only, and (b) the complexity-based stratification
that every results table depends on has no code in the repo.

## 2. Traceability table

| Paper artefact | Repo location | Computes value? | Matches paper | Status |
|---|---|---|---|---|
| Reward functions: CW-SSIM>0.6 → 1, TEDS-Struct>0.9 → 1 (Sec 4, Sec 5.1) | `examples/train/grpo/plugin/plugin.py:698,718,676` | Yes (training reward) | thresholds 0.6 / 0.9 match paper | Verified |
| GRPO config: num_gens=4, bs=4, 1 epoch, ε=0.2, β=0.02 (Sec 5.1) | `examples/train/grpo/plugin/run_external_rm.sh:21,22,34` | Partially | num_gens=4, bs=4, 1 epoch match; ε/β not in script (framework default) | Partial |
| Table 1 CW-SSIM (per complexity) | `cw_ssim.ipynb` cell 0 `calculate_cwssim` + cell 13 `calculate_average_and_ratio` | Yes | computation present; value = mean over non-zero only | See `cwssim-conditional-on-compile` |
| Table 1 Compile ratio | `cw_ssim.ipynb` cell 13 `calculate_average_and_ratio` (`non_zero_ratio`) | Yes | = fraction of non-zero CW-SSIM scores | Verified (definition) |
| Table 2 TEDS / TEDS-Structure | `cw_ssim.ipynb` cell 1/3 `process_file` → `calculate_stats` (`Average (Non-zero)`) | Yes | reported as Average(Non-zero) | See `cwssim-conditional-on-compile` |
| Complexity split: simple/medium/complex (Sec 3, Sec 5.1; basis of Tables 1,2,5,6,7) | (none) | No code classifies tables by `\multirow`/`\multicolumn` count + cell count | — | MISSING |
| Test set construction (496 simple / 354 medium / 361 complex sampled from 101,469) | (none) — eval set provided on HF only | No sampling/split script | n/a | MISSING (script) |
| Training corpus 1,209,986 pairs (Sec 5.1) | `arxiv_papers_get.py`, `table.ipynb` (pipeline only) | Pipeline present; corpus not shipped | n/a | Disclosed (data not shared) |
| RL training set 5,936 complex tables (Sec 4) | (none) | No selection script / file | n/a | MISSING (script/data) |
| Table 3 external benchmark [6] | `cw_ssim.ipynb` cell 3 (`latte_2`/`gt` JSON reader) | Yes (metric only) | LATTE outputs supplied by authors | Verified (metric path) |
| Table 4 human evaluation | (none) | No code (expected) | — | N/A (human study) |
| Ablations Tables 5/6/7 | reuse `run_external_rm.sh` + notebook | reward-component / data-selection variants are config edits | — | Partial (no driver per variant) |

## 3. Findings

## missing

```yaml finding
id: complexity-classification-missing
category: missing
topic: "data splitting / table complexity"
title: "No code classifies tables into simple/medium/complex (basis of all results tables)"
severity: medium
confidence: high
status: finding
file: code/newLLing__Table2LaTeX-RL/table.ipynb
line_start: 1
line_end: 1
quote: |
  source_dir = "arxiv_papers/2406-20000-14000"
claim: "The dataset-prep notebook table.ipynb only extracts and cleans \\begin{table} environments; neither it nor arxiv_papers_get.py nor any other paper-specific file implements the paper's complexity classifier (>=2 \\multirow/\\multicolumn and 100-160 cells = medium, >160 cells = complex, else simple). A repo-wide grep for the threshold 160 in the custom files returns 0 matches."
concern: "Every quantitative table (1, 2, 5, 6, 7) is reported per complexity level, so the headline 'strong on complex tables' claim cannot be reproduced or verified without the classification code that defines those subsets."
resolution: "Authors: please add the script that assigns simple/medium/complex labels (cell counting + \\multirow/\\multicolumn counting) to the 1,211 test tables, or include the per-table labels with the released evaluation set."
cross_refs: ["test-split-script-missing"]
paper_ref: "Section 3 (complexity definition); Section 5.1 (test counts 496/354/361)"
tags: [reforms:1, reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: test-split-script-missing
category: missing
topic: "result traceability / protocol"
title: "No script samples/constructs the 496/354/361 test split or the 5,936-table RL set"
severity: low
confidence: high
status: finding
file: code/newLLing__Table2LaTeX-RL/README.md
line_start: 56
line_end: 58
quote: |
  ### (Step2) Evaluation

  Before running the evaluation, please download the evaluation datasets from [🤗 Table2LaTeX-RL Evaluation](https://huggingface.co/datasets/LLLHHH/Table2LaTeX-RL), and the model form [🤗 Table2LaTeX-RL Model](https://huggingface.co/LLLHHH/Table2Latex-RL)
claim: "The repo ships only inference (qwenvl_test.py) and metric (cw_ssim.ipynb) code plus a HF download of a 1,211-row eval set; there is no code that samples 496 simple / 354 medium / 361 complex tables from the 101,469-table 2024 pool, and no script that selects the 5,936 complex tables used to train VSGRPO."
concern: "The experimental protocol (test-set sampling and RL-set selection) is not reproducible from the repo; only the final eval set and weights are provided, so the upstream selection cannot be audited."
resolution: "Authors: provide the sampling scripts (with seeds) that build the test subsets and the 5,936-table RL training set, or document that the released HF eval set IS the exact frozen test set."
cross_refs: ["complexity-classification-missing"]
paper_ref: "Section 4 (5,936 complex tables); Section 5.1 (test sampling)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-paths-inference
category: bug
topic: "reproducibility / hardcoded paths"
title: "qwenvl_test.py hardcodes private absolute paths for model and data; no CLI"
severity: low
confidence: high
status: finding
file: code/newLLing__Table2LaTeX-RL/qwenvl_test.py
line_start: 101
line_end: 111
quote: |
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/data/shared/Qwen/LLaMA-Factory/qwen/qwenvl-2.5-grpo-5epoch",
            torch_dtype=torch.float16
        )
        model = model.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            "/data/shared/Qwen/LLaMA-Factory/qwen/qwenvl-2.5-grpo-5epoch"
        )
claim: "The evaluation entrypoint loads the model and the input JSONL (line 167) from hardcoded absolute paths under /data/shared/... that do not exist in any clone, with no argument parsing to point at the HF download the README tells users to fetch."
concern: "Running `python qwenvl_test.py` as the README instructs fails immediately with a path-not-found error; the user must edit source to substitute the downloaded model/data paths."
resolution: "Expose model_path / input_path / output_path as CLI args or env vars and default them to the README's HF artefacts."
cross_refs: []
paper_ref: "README Step2"
tags: [reforms:8]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-eval-command-typo
category: bug
topic: "documentation"
title: "README evaluation command misspelled (`pythoh qwenvl_test.py`)"
severity: low
confidence: high
status: finding
file: code/newLLing__Table2LaTeX-RL/README.md
line_start: 62
line_end: 64
quote: |
  ```bash
  pythoh qwenvl_test.py
  ```
claim: "The single documented evaluation command uses `pythoh` instead of `python`."
concern: "Copy-pasting the documented command fails; minor but it is the only reproduction command given for Step 2."
resolution: "Fix `pythoh` -> `python`."
cross_refs: ["hardcoded-paths-inference"]
paper_ref: "README Step2"
tags: [reforms:8]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: missing-dependency-spec
category: difference
topic: "dependencies / environment"
title: "Metric/RL dependencies (table_recognition_metric, pdf2image, poppler, texlive) not in requirements"
severity: low
confidence: high
status: finding
file: code/newLLing__Table2LaTeX-RL/requirements.txt
line_start: 1
line_end: 1
quote: |
  -r requirements/framework.txt
claim: "requirements.txt only pulls the ms-swift framework deps; the paper-specific runtime needs `table_recognition_metric` (TEDS), `pdf2image`+poppler, and a `texlive-full` LaTeX install, which are listed nowhere in requirements/ and are only mentioned as ad-hoc `pip install` lines / a Docker image in the README."
concern: "A reviewer rebuilding the environment from requirements/ alone cannot run plugin.py or cw_ssim.ipynb (ImportError / missing pdflatex); the exact LaTeX/poppler versions affect compile ratio and CW-SSIM."
resolution: "Pin table_recognition_metric, pdf2image, and the LaTeX/poppler stack (the README's Docker tag) in a requirements file or environment.yml."
cross_refs: []
paper_ref: "README Step1/Step3; Section 5.1 (texlive-full Docker)"
tags: [reforms:8, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: cwssim-conditional-on-compile
category: methodology
topic: "evaluation metric aggregation"
title: "Reported CW-SSIM / TEDS averaged over compile-successful (non-zero) predictions only"
severity: medium
confidence: high
status: finding
file: code/newLLing__Table2LaTeX-RL/cw_ssim.ipynb
line_start: 1
line_end: 1
quote: |
  avg = total_score / non_zero_count if non_zero_count > 0 else 0.0
  ratio = non_zero_count / valid_count if valid_count > 0 else 0.0
  return avg, ratio
claim: "calculate_average_and_ratio (cw_ssim.ipynb cell 13) and calculate_stats (cells 1/3, reporting `Average (Non-zero)`) divide the summed score by non_zero_count, i.e. the reported CW-SSIM/TEDS/TEDS-Structure is the mean over only those tables that compiled / parsed successfully; failed cases (score 0) are excluded from the mean and reported separately as `compile ratio` = non_zero_count/valid_count. Confirmed by _audit_code/check_nonzero_averaging.py (Average Non-zero = 0.60 vs Average All = 0.36 on a 3-of-5 toy set)."
concern: "Conditioning the quality metric on compile success makes a model that fails to compile many hard tables look as good (on CW-SSIM/TEDS) as one that compiles them all, so cross-model comparisons in Tables 1-3 are confounded by differing compile/parse rates unless read jointly with the separately-reported ratio."
resolution: "Authors: confirm whether the reported CW-SSIM/TEDS/TEDS-Structure are Average(Non-zero) or Average(All); if Non-zero, also report Average(All) so the quality numbers are not conditioned on each model's own compile rate."
cross_refs: []
check_script: _audit_code/check_nonzero_averaging.py
paper_ref: "Tables 1-3, 5-7; Section 5.1"
tags: [reforms:7, whalen:pitfall-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 2          | medium       | Complexity classifier + test/RL-set selection scripts absent. |
| bug         | 2          | low          | Hardcoded private paths in inference; README `pythoh` typo. |
| difference  | 1          | low          | Metric/LaTeX/poppler deps not pinned in requirements. |
| methodology | 1          | medium       | CW-SSIM/TEDS reported as mean over compile-successful predictions only. |

### Top take-aways (≤6)

1. **[methodology, medium]** Reported CW-SSIM/TEDS/TEDS-Structure are averaged over
   compile-successful (non-zero) predictions only (`cwssim-conditional-on-compile`); quality
   numbers are conditioned on each model's own compile rate and must be read with the
   separately-reported compile ratio.
2. **[missing, medium]** No code implements the simple/medium/complex classifier that defines
   the subsets in every results table (`complexity-classification-missing`).
3. **[missing, low]** No script reproduces the 496/354/361 test sampling or the 5,936-table RL
   training-set selection (`test-split-script-missing`).
4. **[bug, low]** `qwenvl_test.py` hardcodes nonexistent private absolute paths and has no CLI,
   so the documented Step-2 command cannot run as-is (`hardcoded-paths-inference`).
5. **[difference, low]** Metric/runtime dependencies (TEDS lib, pdf2image, poppler, texlive)
   are unpinned and missing from requirements (`missing-dependency-spec`).
6. **[bug, low]** README evaluation command typo `pythoh` (`readme-eval-command-typo`).

### Items that genuinely look fine

- **Temporal split**: training = arXiv Oct 2017–Apr 2023, test = arXiv Jan–Nov 2024
  (Section 5.1). Train/test are temporally disjoint document pools, so no obvious train/test
  table leakage; the "same processing pipeline" reuse is appropriate.
- **Reward thresholds**: plugin.py uses CW-SSIM>0.6 and TEDS-Structure>0.9 (lines 698, 718),
  matching the paper's stated 0.6 / 0.9 reward thresholds.
- **CW-SSIM definition**: the code's simplified single-level Haar-DWT + per-subband SSIM
  (averaged over cA/cH/cV/cD) matches the paper's explicitly "modified CW-SSIM for binary
  table images" description (Section 4); the SSIM constants C1=6.5025, C2=58.5225 equal
  (0.01·255)² and (0.03·255)². Naming differs from canonical CW-SSIM but is disclosed.
- **Artefact availability**: the HF model (`LLLHHH/Table2Latex-RL`, 4B BF16 safetensors) and
  evaluation dataset (`LLLHHH/Table2LaTeX-RL`, 1,211 rows = 496+354+361) both resolve and are
  public.
- **GRPO config**: `run_external_rm.sh` matches the paper's num_generations=4, batch size 4,
  one epoch, full-parameter tuning with frozen ViT.

### Open questions for the authors

- Are the headline CW-SSIM/TEDS values in Tables 1–3 the `Average (Non-zero)` or `Average (All)`
  figure produced by the notebook? (Decides the severity of `cwssim-conditional-on-compile`.)
- Is the released 1,211-row HF dataset the exact frozen test set used in the paper, including the
  simple/medium/complex labels, or were those labels generated by code not in the repo?
- The KL coefficient β=0.02 and clip ε=0.2 from Eq. (2) are not set in `run_external_rm.sh`; were
  framework defaults relied on, and do they equal the paper's values?
