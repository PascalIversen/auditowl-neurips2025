# Code-Repository Audit — Quantization Error Propagation (QEP), NeurIPS 2025 (#4598)

## 1. Summary

The repo `FujitsuResearch/qep` implements **Quantization Error Propagation (QEP)**, a
plug-in correction for layer-wise post-training quantization (PTQ) of LLMs. The core
contribution is a closed-form weight correction (paper Eq. 4/6):
`W*(α) = W + α·W·δ·X̂ᵀ·Ĥ⁻¹`, where `δ = X − X̂` is the upstream quantization error.
This is implemented in `src/gptq.py::Helper.run_weight_correct` (the line
`W += (W @ self.H_delta @ Hinv) * perccorr`), driven by `src/llama.py` (RTN/GPTQ/QuIP)
and `src/qep_awq.py` (AWQ). Evaluation (WikiText-2/PTB/C4 perplexity, and ArcE/PiQA/
StoryCloze zero-shot accuracy via a vendored `lm_eval`) is wired into `src/llama.py`'s
`--ppl` / `--tasks` flags.

What I did:
- Read the paper (Tables 1–5, 11; Figs 1–3; §6; Appendix B/C/D; NeurIPS checklist).
- Read the core implementation (`llama.py`, `qep_awq.py`, `gptq.py`, `quant.py`,
  `datautils.py`, `resultutils.py`, `modelutils.py`, AWQ calib utils).
- Verified the QEP correction matches paper Eq. (6) numerically:
  `_audit_code/check_qep_formula.py` → `out/check_qep_formula.txt` (code reproduces the
  paper closed form to a relative error ~1e-4, the only gap being Hessian damping).
- Statically checked third-party imports against `requirement.txt` (found `scipy`,
  `portalocker` missing).
- Checked git provenance (single commit `344329a`, only `main`, no submission tag).

Headline assessment: the **method itself is faithfully implemented** and the main
perplexity/accuracy tables are reproducible in principle. The most consequential issues
are (a) the code **cannot import/run as shipped** because two hard dependencies are
unlisted; (b) several **paper-reported artefacts have no code** (Fig. 2 error-accumulation
experiment, Table 3 runtime measurement, the OmniQuant comparison in Table 11); (c) the
main results tables report **the best of 5 random seeds per configuration** (admitted only
in Appendix D.3); and (d) the **per-layer / model-specific αl setting** that the paper uses
(αl = 0 for Llama-2-70B MLP) is not implementable from the exposed CLI, which only offers a
single global `--perccorr`.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — WikiText-2 PPL (per-channel INT4/3/2, all methods) | `src/llama.py` (`--ppl wikitext2`) + `llama_sequential`/`run_awq_with_QEP` | not run (needs GPU/weights) | — | Code present; not executed here |
| Table 2 — zero-shot avg acc (ArcE/PiQA/SC) | `src/llama.py` (`--tasks`) → `src/zeroShot/` | not run | — | Code present; not executed |
| Table 5/6/… — group-wise (g32/g64/g128) PPL/acc | `src/llama.py` (`--groupsize`) | not run | — | Code present (groupsize flag) |
| Fig. 1 — INT4/3/2 PPL bars (same numbers as Table 1) | same as Table 1 | not run | — | Computed values present; plotting script absent (acceptable) |
| **Fig. 2 — ∆m error accumulation across blocks (quantize first 10 blocks, n=10, Eq. 2)** | (none) | — | — | **MISSING (no partial-quantization / per-block Frobenius-norm code)** |
| **Table 3 — quantization runtime (GPTQ/AWQ/QEP+RTN, 7B/13B/70B)** | (none in main path) | — | — | **MISSING (no timing in `llama.py`/`qep_awq.py`)** |
| Table 4 — PPL relative to RTN across calib datasets | `src/llama.py` (`--dataset c4/ptb/wikitext2`) | not run | — | Code present (dataset flag) |
| **Table 11 / App. D.4 — OmniQuant baseline row** | (none) | — | — | **MISSING (no OmniQuant implementation/runner)** |
| Fig. 3 / App. D.3 — QuIP ±QEP averaged over 5 seeds (SEM) | `src/llama.py` (`--seed`), repeated runs | not run | — | Reproducible via repeated `--seed`; no aggregation/plot script |
| αl = 1/2 default, αl = 0 for Llama-2-70B MLP | `--perccorr` (global) in `src/llama.py:313-315` | — | partial | Global α only; per-layer/MLP-specific α=0 not exposed (see finding) |

Main metric values (Tables 1–2) trace to executable code; three reported artefacts
(Fig. 2, Table 3, Table 11) have no computing code in the repo.

## 3. Findings

## missing

```yaml finding
id: unlisted-hard-deps-scipy-portalocker
category: missing
topic: "dependencies / environment"
title: "scipy and portalocker are hard imports but absent from requirement.txt"
severity: high
confidence: high
status: finding
file: src/gptq.py
line_start: 16
line_end: 18
quote: |
  import primefac
  import scipy
  import math
claim: "gptq.py imports scipy at module load and resultutils.py imports portalocker at module load; both are imported transitively by the main entrypoint src/llama.py (which does `from gptq import *` and `from resultutils import *`), yet neither scipy nor portalocker appears in requirement.txt."
concern: "A fresh environment built from requirement.txt cannot even import src/llama.py (ImportError on scipy), so no reported result is reproducible without out-of-band dependency installation."
resolution: "Add scipy and portalocker (and confirm numpy/datasets/tqdm) to requirement.txt with the versions used for the paper."
cross_refs: []
check_script: _audit_code/check_imports.py
paper_ref: "Code availability (NeurIPS checklist Q5: 'README file with a list of required packages')"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fig2-error-accumulation-no-code
category: missing
topic: "result traceability"
title: "No code computes Fig. 2 (per-block quantization-error accumulation ∆m)"
severity: medium
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  This experiment sets n = 10. Figure 2 shows an approximately exponential accumulation or
  errors within the quantized layer, as well as an error growth that persists in the unquantized layers.
claim: "Figure 2 quantizes only the first 10 Transformer blocks, keeps the rest in FP, and plots the squared Frobenius norm ∆m (Eq. 2) of original-vs-partially-quantized block outputs; no script in src/ performs partial (first-n-blocks-only) quantization or computes a per-block output-difference norm (grep for the relevant logic returns nothing)."
concern: "Figure 2 is the motivating empirical evidence for the paper's central premise (cross-layer error growth), yet the experiment that produces it is not in the repo and cannot be reproduced."
resolution: "Provide the script that quantizes the first n=10 blocks and evaluates ∆m per block (Eq. 2)."
cross_refs: []
check_script: _audit_code/check_imports.py
paper_ref: "Figure 2, Section 4 / Eq. (2)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: table3-runtime-no-code
category: missing
topic: "result traceability"
title: "No timing code for Table 3 quantization-runtime measurements"
severity: low
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  Runtime
  We examine the impact of computation time
  required for the correction term. Table 3 shows the processing time of each layer-wise PTQ.
claim: "Table 3 reports wall-clock quantization times (GPTQ/AWQ/QEP+RTN at 7B/13B/70B), but the main quantization drivers src/llama.py and src/qep_awq.py contain no time.time()/perf_counter instrumentation (timing exists only in the unused src/zeroShot/models/gptq.py)."
concern: "The runtime claims (e.g. 'QEP correction requires significantly less computation than the quantization process') are not reproducible from the code as the measurement harness is absent."
resolution: "Provide the timing harness used for Table 3, or state how the times were measured."
cross_refs: []
paper_ref: "Table 3, Section 6.1 (Runtime)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: omniquant-baseline-no-code
category: missing
topic: "baselines"
title: "OmniQuant comparison (Table 11) has no code in the repo"
severity: low
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  For completeness, we compare QEP-enhanced layerwise PTQ with block-wise OmniQuant [Shao et al.,
  2023] on LLaMA-2-7B using WikiText-2 perplexity; lower values indicate better performance.
claim: "Appendix D.4 / Table 11 reports an OmniQuant baseline, but the repo contains no OmniQuant implementation or runner (grep for 'omni' in src/ returns nothing); only RTN/GPTQ/AWQ/QuIP are implemented."
concern: "The OmniQuant comparison row cannot be reproduced from this repo; the reader must trust the externally-obtained number."
resolution: "State the OmniQuant code/version used, or include a runner; otherwise mark Table 11's OmniQuant row as reproduced from an external implementation."
cross_refs: []
paper_ref: "Table 11, Appendix D.4"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-missing-run-instructions
category: missing
topic: "reproducibility / documentation"
title: "README has no run commands or results table, contradicting checklist claim"
severity: low
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 15
quote: |
  # Fujitsu Open Source Model Compression Software
  - Release Plan
    - NeurIPS paper version release (December 2, 2025)
    - Official version release (March, 2026)
  - Citation
claim: "The README contains only a release plan and a citation; it has no list of required packages, no example commands, and no results table, whereas the NeurIPS checklist (Q5) states the submission includes 'execution scripts and a README file with a list of required packages'."
concern: "Without exact reproduction commands or a results table, mapping CLI flags to specific paper table cells (bit-width, group size, α per model) is left to the reader to reverse-engineer."
resolution: "Add a README section with the exact commands that reproduce each main table/figure and the package list."
cross_refs: ["unlisted-hard-deps-scipy-portalocker"]
paper_ref: "NeurIPS checklist Q5 (Open access to data and code)"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: save-result-crashes-on-mistral
category: bug
topic: "result logging"
title: "save_experiment_results IndexErrors for non-Llama models (e.g. Mistral-7B)"
severity: low
confidence: medium
status: finding
file: src/resultutils.py
line_start: 22
line_end: 26
quote: |
    pattern_llama = r'llama-(\d+)'
    pattern_B = r'(\d+)b'
    llama_match = re.findall(pattern_llama, args.model)
    B_match = re.findall(pattern_B, args.model)
    model_str = f"llama-{llama_match[0]}-{B_match[0]}b"
claim: "save_experiment_results parses the model name assuming it contains 'llama-<n>' and '<n>b'; for Mistral-7B (a model the paper evaluates) llama_match is empty, so llama_match[0] raises IndexError."
concern: "Running any Mistral-7B experiment with --save-result crashes at result-saving time; the metric is computed but cannot be persisted via this path."
resolution: "Make the model-name parsing robust to non-Llama models (Mistral, Llama-3), or guard the regex match."
cross_refs: []
paper_ref: "Tables 1–2 (Mistral-7B column)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: per-layer-alpha-not-exposed
category: difference
topic: "hyperparameters / propagation strength"
title: "Model/layer-specific αl (α=0 for Llama-2-70B MLP) not implementable from CLI"
severity: medium
confidence: high
status: finding
file: src/gptq.py
line_start: 246
line_end: 269
quote: |
  def run_weight_correct(
      self, layer, percdamp=.01, perccorr=.25
  ):
      W = layer.weight.data.clone()
      if isinstance(layer, nn.Conv2d):
          W = W.flatten(1)
      if isinstance(layer, transformers.Conv1D):
          W = W.t()
      W = W.float()
      H = self.H.clone()

      dead = torch.diag(H) == 0
      H[dead, dead] = 1
      W[:, dead] = 0

      damp = percdamp * torch.mean(torch.diag(H))
      diag = torch.arange(H.shape[0], device=H.device)
      H[diag, diag] += damp
      H = torch.linalg.cholesky(H)
      H = torch.cholesky_inverse(H)
      Hinv = H

      # 重み補正
      W += (W @ self.H_delta @ Hinv) * perccorr
claim: "The propagation strength α (=perccorr) is a single global scalar passed identically to every corrected layer; src/llama.py and src/qep_awq.py call run_weight_correct(..., perccorr=args.perccorr) for all attention/MLP-up/gate layers with no per-layer or per-model branch."
concern: "The paper sets αl = 1/2 for all layers EXCEPT MLP layers in Llama-2-70B which are set to αl = 0; with only a global --perccorr, a single run cannot apply α=0.5 to attention and α=0 to MLP simultaneously, so the exact Llama-2-70B configuration behind Table 1/2 is not reproducible from the CLI as shipped."
resolution: "Expose a per-layer / per-module α schedule (or a flag that zeroes correction for MLP layers), or document the exact command(s) used to obtain the Llama-2-70B numbers."
cross_refs: ["mlp-down-proj-never-corrected"]
paper_ref: "Section 6 (Quantization): 'αl = 1/2 for all layers, except for the MLP layers in the Llama-2 70B model, for which we set αl = 0'"
tags: [reforms:3, forensics:post-hoc-selection]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: best-seed-reported-in-main-tables
category: difference
topic: "selective reporting / statistics"
title: "Main tables report the best of 5 seeds per configuration"
severity: medium
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  The main text lists the best seed per configuration for
  consistency with past work. This appendix confirms that the gains are not seed-specific but robust
claim: "Appendix D.3 states the main-text tables (Table 1/2) report the best seed per configuration; the code (src/llama.py --seed) runs a single seed per invocation and provides no max-over-seeds selection script, so the 'best seed' selection was done manually off-repo."
concern: "Reporting the best of 5 seeds (rather than the mean) optimistically biases the headline perplexity/accuracy numbers; the selection step is not in the repo and is only disclosed in the appendix."
resolution: "Report mean ± SEM over the 5 seeds in the main tables (as Fig. 3 does for QuIP), or clearly mark Table 1/2 values as best-of-5 and provide the selection script."
cross_refs: []
paper_ref: "Appendix D.3 (Stability of QuIP Results Across Random Seeds)"
tags: [stats:excessive-similarity, forensics:post-hoc-selection, reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: mlp-down-proj-never-corrected
category: difference
topic: "method scope / QEP correction"
title: "mlp.down_proj is unconditionally excluded from QEP correction"
severity: low
confidence: high
status: finding
file: src/llama.py
line_start: 127
line_end: 131
quote: |
            for name, module in subset.items():
                if args.qep and name != 'mlp.down_proj':
                    helper.run_weight_correct(
                        module, percdamp=args.percdampqep, perccorr=args.perccorr
                    )
claim: "In both drivers (llama.py and qep_awq.py) the QEP weight correction is skipped for mlp.down_proj for every model (the comment 'down_proj does not perform correction'), so down_proj always behaves as α=0 regardless of the --perccorr setting."
concern: "The paper presents QEP correction as applied across layers with a tunable αl; the unconditional exclusion of down_proj for all models is a fixed design choice the main text does not state, which slightly changes which weights are actually corrected."
resolution: "Document that down_proj is always excluded from correction, or make it configurable; clarify whether the reported gains include this exclusion."
cross_refs: ["per-layer-alpha-not-exposed"]
paper_ref: "Section 5.2 / Eq. (4)–(6)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A — no methodological invalidity found. The QEP correction is a calibration-set-only
weight transform applied before quantization; it does not touch the held-out
perplexity/accuracy test sets. Calibration (C4/Pile, 128 segments) and evaluation
(WikiText-2/PTB/C4 test splits, lm_eval tasks) are standard and separated, matching prior
PTQ work (GPTQ/AWQ/QuIP). The core correction faithfully implements paper Eq. (6) (verified
numerically, see `_audit_code/out/check_qep_formula.txt`).

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 5          | high         | scipy/portalocker unlisted (blocks import); Fig.2/Table3/Table11 have no code; bare README |
| bug         | 1          | low          | result-saving crashes on Mistral-7B model name               |
| difference  | 3          | medium       | per-layer α not exposed; best-of-5-seed reporting; down_proj always uncorrected |
| methodology | 0          | -            | No invalid procedure found; correction matches Eq. (6)       |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing]** `scipy` and `portalocker` are hard imports of the main entrypoint but are
   absent from `requirement.txt`, so the code cannot be imported/run from the specified
   environment (`unlisted-hard-deps-scipy-portalocker`, high/high).
2. **[missing]** Figure 2 (the motivating cross-layer error-accumulation experiment, Eq. 2)
   has no producing code in the repo (`fig2-error-accumulation-no-code`, medium/high).
3. **[difference]** Main Tables 1–2 report the **best of 5 random seeds** per configuration
   (disclosed only in Appendix D.3); no selection script is in the repo
   (`best-seed-reported-in-main-tables`, medium/high).
4. **[difference]** The paper's per-model αl schedule (α=0 for Llama-2-70B MLP) is not
   implementable from the single global `--perccorr` CLI flag
   (`per-layer-alpha-not-exposed`, medium/high).
5. **[missing]** Table 3 runtime numbers and the Table 11 OmniQuant baseline have no
   computing/timing code (`table3-runtime-no-code`, `omniquant-baseline-no-code`, low/high).
6. **[bug]** `--save-result` crashes for Mistral-7B because the result logger assumes a
   Llama model-name format (`save-result-crashes-on-mistral`, low/medium).

### Items that genuinely look fine
- **QEP correction is faithful to Eq. (6).** `gptq.py:269` (`W += (W @ H_delta @ Hinv) * perccorr`)
  reproduces the paper closed form to relative error ~1e-4 (only Hessian damping differs);
  see `_audit_code/out/check_qep_formula.txt`. `perccorr` default 0.5 matches the paper's
  default αl = 1/2.
- **Calibration/evaluation separation is sound.** Calibration uses C4/Pile (128 segments,
  `datautils.py`, `awq/utils/calib_data.py`); perplexity uses the standard WikiText-2/PTB/C4
  test encodings (`llama_eval`), and zero-shot tasks use a vendored `lm_eval`. No test-set
  leakage into the correction or quantization.
- **Baselines present and run under the same harness.** RTN/GPTQ/AWQ/QuIP all run through
  the same `llama.py`/`qep_awq.py` drivers with and without `--qep`, so the with/without-QEP
  comparison uses identical splits, metrics, and preprocessing.
- **Group-wise and multi-dataset settings are supported** via `--groupsize` and `--dataset`
  flags, matching the Appendix D and Table 4 sweeps.

### Open questions for the authors
- Which exact commands (per model, bit-width, group size) reproduce each Table 1/2 cell, and
  how was the αl = 0 MLP setting for Llama-2-70B applied given only a global `--perccorr`?
- Will Fig. 2, Table 3 (timing), and the OmniQuant comparison code be released, or were they
  produced by separate/external harnesses?
- For Table 1/2 best-of-5-seed reporting: can mean ± SEM be provided for all methods (not
  only QuIP in Fig. 3)?
