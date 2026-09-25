# Code audit — Paper 793: *Mixture of Inputs: Text Generation Beyond Discrete Token Sampling*

## 1. Summary

The repository `EvanZhuang/mixinputs` is a **vLLM monkey-patch** that implements the
paper's core method (MOI). It ships:
`mixinputs/` (the patch: `vllm_patch.py`, copied-and-modified `gpu_model_runner.py`
and `gpu_input_batch.py`, a CLI installer `commands.py`), plus `example/` (two
end-to-end eval scripts `aime_moi.py` / `tinyzero_moi.py`, their `.sh` launchers,
a `reward.py`, and two sample result JSONs). The MOI mixing math lives entirely in
`mixinputs/gpu_model_runner.py:1190-1257`.

What I did:
- Read the method (paper §3–4, Eq. 1–5, Alg. 1) and mapped each equation to the code.
- Re-derived the code's effective per-token mixing weight and compared it
  algebraically and numerically to paper Eq. (5) (`_audit_code/check_moi_formula.py`):
  they match to floating-point error (max abs diff 3.3e-16).
- Quantified the effect of the code computing entropy/mixing over the **top-20**
  logprobs rather than the full vocabulary V (same script): the normalized entropy
  used by the code is ~2× the paper's full-vocab value.
- Inventoried which paper tables/figures the repo can actually reproduce.
- This is an inference patch; there is no model training. I did not (and cannot
  without 32–49B models + GPUs) re-run the benchmarks; findings are static-analysis
  based.

The method implementation is faithful and correct in its algebra. The main gaps are
(a) the repo contains only 2 of the paper's ~10 quantitative artefacts (the full
benchmark harness, statistical tests, ablations, throughput, case-study, and
hyperparameter-analysis code are all absent), and (b) a top-k truncation in the
mixing/entropy that the paper's equations do not describe.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Method: mixed embedding `h_t=Σ w_i e_i`, Eq.(5) weights | `mixinputs/gpu_model_runner.py:1216-1228` | weights = paper Eq.(5) (verified, diff 3.3e-16) | ✓ | Verified |
| Direct Mixture baseline (`h_t=Σ p_i e_i`) | `mixinputs/gpu_model_runner.py:1229-1237` | implemented (beta=-100 path) | ✓ | Verified |
| Table 1 (16 model×task accuracies, 4 models) | (none — no GPQA/LiveCodeBench harness, no grid driver) | — | — | MISSING |
| Table 1 AIME (QwQ) ~80, CountDown4 (Nemotron) ~59 | `example/*` + sample JSONs (single config) | sample JSON 0.855 (top_p 0.8) etc. | partial | PARTIAL (2 runs only) |
| Table 2 — prompt-blending case study (3 LLMs × 5 sentiment sets) | (none) | — | — | MISSING |
| Table 3 — throughput (tokens/s, overhead) | (none) | — | — | MISSING |
| Table A1 — single-tuned transfer | (none) | — | — | MISSING |
| Table A2 — MT-Bench scores | (none) | — | — | MISSING |
| Table A3 — McNemar p-values | (none — no significance-test code) | — | — | MISSING |
| Table A4 — 64-run AIME mean/σ | (none — driver runs once, no seed loop) | — | — | MISSING |
| Fig. 2 — hyperparameter importance (best-of-N + RandomForest) | (none) | — | — | MISSING |
| Fig. 3 — β sensitivity | (none — no β-sweep driver) | — | — | MISSING |

## 3. Findings

## missing

```yaml finding
id: benchmark-harness-absent
category: missing
topic: "result traceability"
title: "Repo lacks the harness for almost all paper tables/figures (only 2 of 4 benchmarks, no stats/ablation/throughput code)"
severity: high
confidence: high
status: finding
file: example/README.md
line_start: 66
line_end: 72
quote: |
  ## Results

  Results are stored in the `results/` directory as JSON files with naming convention:
  ```
  {model_name}={dataset_name}={top_p}={temperature}.json
  ```
  We provide results for 2 sample runs.
claim: "The repo provides eval drivers only for AIME and CountDown4 (example/aime_moi.py, example/tinyzero_moi.py) and ships exactly 2 sample result JSONs; there is no code for GPQA-Diamond, LiveCodeBench, the McNemar tests (Table A3), the 64-run variance study (Table A4), the throughput analysis (Table 3), the prompt-blending case study (Table 2 / §7.2), MT-Bench (Table A2), or the hyperparameter-importance analysis (Fig. 2, best-of-N + RandomForestRegressor)."
concern: "Most headline numbers (Table 1's GPQA/LiveCodeBench columns, all appendix tables, both analysis figures, and every reported statistical test) cannot be reproduced from this repository, and there is no grid-search/seed-averaging driver to regenerate Table 1 itself."
resolution: "Authors: please release the GPQA (lm-eval-harness) and LiveCodeBench drivers, the grid-search/seed-averaging script that produces Table 1, and the scripts computing Tables 2–A4 and Figs. 2–3."
cross_refs: ["§5", "§6", "§7", "Appendix A-G"]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: example-scripts-omit-logprobs-rely-on-patch-default
category: missing
topic: "reproducibility / required configuration"
title: "Example drivers never request logprobs; MOI silently depends on the patch's hidden default of 20"
severity: low
confidence: high
status: finding
file: example/aime_moi.py
line_start: 147
line_end: 151
quote: |
      sampling_params = SamplingParams(
          temperature=args.temperature,
          max_tokens=args.max_new_tokens,
          top_p=args.top_p,
      )
claim: "The example SamplingParams omit `logprobs=`, so MOI's mixing support is entirely determined by the patch's hard-coded fallback of 20 logprobs in gpu_input_batch.py:330; the number of candidate tokens mixed is never surfaced as a documented, tunable experimental knob."
concern: "A reader reproducing the paper has no in-script signal that the method only mixes over the top-20 tokens, and changing this number (which materially affects the entropy term, see topk-truncation-vs-paper) requires editing patched vLLM internals."
resolution: "Authors: document the effective top-k used for mixing and expose it as a configurable parameter, or confirm all reported numbers used the default 20."
cross_refs: ["topk-truncation-vs-paper"]
check_script: _audit_code/check_moi_formula.py
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No runtime/wiring bugs found in the MOI computation that the code's own intent
contradicts. (The repo cannot be executed here without 32–49B-parameter models and
GPUs; this is a static finding-free pass on the `bug` category, not a confirmation
that the patch runs against every vLLM build — see `vllm-pin-too-loose`, filed as a
`difference`.)

## difference

```yaml finding
id: topk-truncation-vs-paper
category: difference
topic: "method faithfulness / entropy"
title: "Mixing and entropy use top-20 logprobs, not the full vocabulary V the paper's Eq.(1)(2) define"
severity: medium
confidence: high
status: finding
file: mixinputs/gpu_model_runner.py
line_start: 1202
line_end: 1209
quote: |
            prior_probs = torch.softmax(logprobs_slice, dim=-1)

            n = prior_probs.size(-1)
            # add eps for numerical stability
            logp = torch.log(prior_probs.clamp_min(1e-12))
            H = -(prior_probs * logp).sum(dim=-1)
            H_max = torch.log(torch.tensor(float(n), device=prior_probs.device))
            normalized_entropy = (H / H_max).view(-1, 1, 1)
claim: "`prior_probs` is a softmax over only the top-k retained logprobs (k defaults to 20 via gpu_input_batch.py:327-330), then renormalized; the normalized entropy uses H_max=log(n=k), and the embedding mixture (lines 1218-1228) sums over only these k tokens. The paper's Eq.(1) sums over all V vocabulary tokens and Eq.(2) normalizes entropy by log V."
concern: "Computing H over the renormalized top-20 instead of the full vocab inflates the normalized entropy by ~2× in peaked distributions (verified, _audit_code/out/moi_formula_check.txt), and H is exactly the coefficient that interpolates between the distribution and the sampled token (Eq.5), so the implemented interpolation differs systematically from the one described — even though the truncated procedure is itself a valid, sound approximation."
resolution: "Authors: clarify in the paper that mixing/entropy are computed over the top-k (default 20) tokens, and report sensitivity to k; or compute H over the full vocabulary to match Eq.(2)."
cross_refs: ["example-scripts-omit-logprobs-rely-on-patch-default", "moi-weights-match-paper"]
check_script: _audit_code/check_moi_formula.py
paper_ref: "Eq. (1), (2); Section 4.2"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: vllm-pin-too-loose
category: difference
topic: "dependencies / reproducibility"
title: "setup.py pins vllm>=0.8.5 but the patch copies version-specific private vLLM internals"
severity: low
confidence: high
status: finding
file: setup.py
line_start: 11
line_end: 13
quote: |
  requirements = [
      'vllm>=0.8.5',
  ]
claim: "The package declares `vllm>=0.8.5`, but mixinputs/gpu_model_runner.py and gpu_input_batch.py are verbatim copies of vLLM 0.8.5's private `vllm.v1.worker` classes with the MOI hook spliced in, and vllm_patch.py monkey-patches those exact symbols; the README itself states 'Requires vllm == 0.8.5 or vllm == 0.8.5.post1'."
concern: "An open `>=0.8.5` range will install newer vLLM versions whose refactored internals break the patch, so a naive `pip install` does not reproduce the paper's environment despite the README's tighter (but non-enforced) instruction."
resolution: "Authors: pin `vllm==0.8.5` (or `==0.8.5.post1`) in setup.py to match the README and the copied internals."
cross_refs: []
paper_ref: "Appendix G (vLLM implementation)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodological defect found in the implemented procedure. The Bayesian
Dirichlet–Multinomial mixing is implemented as derived; the Direct-Mixture ablation
is present; per-task β tuning is documented in the paper and the example scripts. The
only soundness-adjacent issue (top-k truncation) is a faithfulness `difference`, not
an invalid procedure — the truncated estimator is itself well-defined. N/A for
data-splitting / leakage / target-leakage / temporal-integrity: MOI is a
training-free decoding intervention with no train/test split, no fitted preprocessing,
and no learned parameters.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 2          | high         | Only 2 of ~12 quantitative artefacts have code; no stats/ablation/throughput/figure harness |
| bug         | 0          | -            | No intent-contradicting wiring bug found (not executed)    |
| difference  | 2          | medium       | Top-20 truncation vs full-V entropy/mix; loose vLLM pin    |
| methodology | 0          | -            | Implemented mixing is sound; method matches Eq.(5)         |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `benchmark-harness-absent`** (high/high) — the repo reproduces at most
   AIME + CountDown4 single configs; GPQA, LiveCodeBench, all appendix tables, both
   analysis figures, and every statistical test (incl. McNemar Table A3, 64-run
   Table A4) have no code.
2. **[difference] `topk-truncation-vs-paper`** (medium/high) — entropy and the
   embedding mixture are computed over the top-20 logprobs, not the full vocabulary V
   that Eq.(1)/(2) specify; this ~2× inflates the entropy term that drives the
   interpolation.
3. **[difference] `vllm-pin-too-loose`** (low/high) — `setup.py` allows `vllm>=0.8.5`
   while the patch hard-copies 0.8.5-specific internals; a fresh install can break.
4. **[missing] `example-scripts-omit-logprobs-rely-on-patch-default`** (low/high) —
   the effective mixing top-k is a hidden patch default (20), never surfaced in the
   eval scripts.

### Items that genuinely look fine
- **MOI weight formula matches paper Eq.(5) exactly** (`gpu_model_runner.py:1216-1228`):
  re-derived and numerically verified, max abs diff 3.3e-16
  (`_audit_code/out/moi_formula_check.txt`). Filed as the fine-status anchor
  `moi-weights-match-paper` via cross-ref.
- **Direct Mixture baseline** (`h_t=Σ p_i e_i`) is implemented (lines 1229-1237,
  `beta=-100` path) and matches the paper's §5.3 ablation definition.
- **β / direct-mixture configuration** is read from `MIXINPUTS_BETA`
  (`gpu_input_batch.py:196`) and documented in the README, consistent with the paper's
  per-task β tuning.
- **Numerical-stability epsilon** (`clamp_min(1e-12)`) on the log — benign, not a finding.

### Open questions for the authors
- Were all reported numbers produced with the default top-20 mixing support, and how
  sensitive are the gains to this k (vs full-vocab mixing)? (relates to
  `topk-truncation-vs-paper`).
- Can the authors release the off-repo drivers for GPQA/LiveCodeBench, the
  grid-search/seed-averaging harness behind Table 1, and the statistical-test scripts
  behind Table A3/A4? (relates to `benchmark-harness-absent`).
