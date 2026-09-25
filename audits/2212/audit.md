# Audit — Best-of-N Jailbreaking (NeurIPS 2025, paper 2212)

## Summary

The repo `jplhughes/bon-jailbreaking` is the official code for the paper. It contains
the four core attack runners (`bon/attacks/run_text_bon.py`, `run_image_bon.py`,
`run_audio_bon.py`, `run_prepair.py`), the augmentation logic, a naive
repeated-sampling baseline (`run_baseline.py`), the bootstrap/ASR-trajectory and
power-law-fitting utilities (`bon/utils/shotgun_utils.py`, `power_law_simple.py`), the
HarmBench dataset of 159 direct requests (`data/direct_request.jsonl`), the vocalized
audio inputs (`data/audio_files/`), and two reproduction notebooks
(`experiments/2_plot_asr.ipynb`, `2_plot_powerlaw.ipynb`). I read the attack runners,
the augmentation functions, the ASR/bootstrap pipeline, the power-law fitter, the
experiment shell scripts, the notebooks, and the released jailbreak CSVs.

I ran three deterministic checks under `_audit_code/`:
(1) `check_missing_powerlaw_module.py` — confirms `bon/utils/power_law.py` and all five
symbols imported by `2_plot_powerlaw.ipynb` are absent from the repo;
(2) `check_released_csv_asr.py` — cross-checks the implied ASR from the released
successful-jailbreak CSVs against the paper's text ASRs (4 of 5 match within rounding,
strongly confirming provenance; Circuit Breaking does not);
(3) a manual probe of `apply_random_capitalization` showing it drops non-ASCII
alphabetic characters.

Key structural finding: this is a black-box attack paper whose numbers come from live
commercial APIs (OpenAI / Anthropic / Google / GraySwan), and **no experiment outputs
are committed** (`exp/` is gitignored). Every headline number therefore requires
re-running attacks against paid, nondeterministic APIs at the paper's scale (N=7,200–10,000
on 159 requests). Reliance on commercial APIs is a legitimate reason the repo is not
self-contained, but it is still a reproducibility limitation. Beyond that, the repo ships
the *attack* code but is missing the *analysis* code for several reported results: the
forecasting error (Fig. 4 / abstract 4.6%), the attack-composition / sample-efficiency
results (Fig. 5 / §5), the PAIR/TAP baseline comparisons (§3.1 / App. E.1), and the
text/vision resampling-reliability numbers (§6.4). One shipped notebook
(`2_plot_powerlaw.ipynb`) cannot run due to a missing module.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 1a / §3.1 text ASR per model (Sonnet 78%, GPT-4o 87%, Gemini Pro 50%, Llama3 94%) | `run_text_bon.py` (attack) + `shotgun_utils.py:process_powerlaw_data/calculate_asr_trajectories` (ASR) | requires live API runs; not committed | implied ASR from released success CSV matches 4/5 | Pipeline present, outputs MISSING (live API) |
| §3.1 Circuit Breaking 52% ASR | same | CSV implies 37.1% | ✗ (CSV is display artifact) | MISMATCH (low-conf, see csv-circuitbreaking-asr-mismatch) |
| Fig. 1b vision ASR per model | `run_image_bon.py` + `shotgun_utils.py` | requires live API runs | — | Pipeline present, outputs MISSING (live API) |
| Fig. 1c audio ASR per model | `run_audio_bon.py` + `shotgun_utils.py` | requires live API runs | — | Pipeline present, outputs MISSING (live API) |
| Fig. 1a "error bars from 100 bootstrap trajectories" (§2) | `shotgun_utils.py:calculate_asr_trajectories` (notebook calls `num_repeats=10`) | 10 trajectories in shipped notebook | ✗ (count differs) | DIFFERENCE (bootstrap-repeats-10-vs-100) |
| Fig. 1d / Fig. 3 power-law fit `-log(ASR)=aN^-b` | `power_law_simple.py:fit_power_law`; notebook `2_plot_asr.ipynb` | runnable for GPT-4o-Mini/Gemini-Flash subset | — | Partially present |
| Fig. 3/4 power-law via `2_plot_powerlaw.ipynb` | `2_plot_powerlaw.ipynb` cell 13 | crashes on import | — | BUG (powerlaw-notebook-dead-import) |
| Fig. 4 / abstract: forecast error 4.6% (4.4% text, 6.3% vision, 2.5% audio) | (none) — no forecasting driver/notebook | — | — | MISSING (forecasting-code-absent) |
| Fig. 5 / §5: composition ASR gains & sample-efficiency (28x/68x/250x etc.) | `run_prepair.py` (attack only); no analysis/plot | — | — | MISSING (composition-analysis-absent) |
| §3.1 / App. E.1: "33% higher than PAIR, 39% higher than TAP" | (none) — only PrePAIR is implemented | — | — | MISSING (pair-tap-baseline-absent) |
| §6.1 augmentation-vs-baseline (e.g. Sonnet 68% vs 3.8%) | `run_baseline.py` (naive resampling) + ASR pipeline | requires live API runs | — | Pipeline present, outputs MISSING (live API) |
| §6.4 / abstract: resampling reliability 30% text / 25% vision / 15% audio (~20%) | only `experiments/audio_reliability/` (audio); no text/vision | — | — | MISSING (reliability-text-vision-absent) |
| §3.1 unaugmented single-shot ASR (0.6% Sonnet) | `run_baseline.py` covers resampling baseline | requires live API runs | — | Pipeline present, outputs MISSING (live API) |

## missing

```yaml finding
id: forecasting-code-absent
category: missing
topic: "result traceability / forecasting"
title: "No code computes the power-law forecasting error (Fig. 4 / abstract 4.6%)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  We verify these power-law trends by forecasting the ASR after 10,000 samples, having observed 1,000, and find an error of 4.6% ASR, averaged across models and modalities.
claim: "The repo ships fitting utilities (power_law_simple.py) and a power-law plot for the GPT-4o-Mini/Gemini-Flash subset, but no script or notebook fits a power law on small-N data, extrapolates to large N, and computes the 4.6%/4.4%/6.3%/2.5% forecast errors reported in the abstract and Section 4.2."
concern: "The headline forecasting claim (a central contribution) cannot be reproduced or verified from the repo because the extrapolate-and-measure-error procedure is not implemented."
resolution: "Authors: please add the forecasting script/notebook that produces the per-model forecast errors and the averaged 4.6%."
cross_refs: ["bootstrap-tiling-extrapolation", "§4.2"]
check_script: _audit_code/check_missing_powerlaw_module.py
paper_ref: "Abstract; Section 4.2 'Forecasting'; Figure 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: composition-analysis-absent
category: missing
topic: "result traceability / attack composition"
title: "No analysis/plot code for composition ASR gains and sample-efficiency (Fig. 5 / §5)"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  For Claude Sonnet and GPT-4o, sample efficiency increases by 34x with text inputs and 18x with vision inputs. Notably, both Gemini models see particularly high sample efficiency increases with audio inputs—222x on average.
claim: "run_prepair.py runs the PrePAIR/MSJ prefix attack, but no script computes the composition ASR deltas (e.g. '+20% Claude Sonnet text', '32% to 70% vision') or the sample-efficiency ratios (28x/68x/250x/34x/18x/222x) shown in Figure 5 and Section 5."
concern: "The composition results, an entire section of the paper, have no value-producing code in the repo, so they cannot be reproduced or audited."
resolution: "Authors: please add the script/notebook that computes the composition ASR gains and sample-efficiency ratios from the PrePAIR/MSJ + BoN runs."
cross_refs: ["§5"]
paper_ref: "Section 5; Figure 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pair-tap-baseline-absent
category: missing
topic: "baselines"
title: "PAIR and TAP baseline comparisons are not implemented in the repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  As for other black-box baselines, BoN achieves 33% higher ASR than PAIR (Chao et al., 2023) and 39% higher than TAP (Mehrotra et al., 2023) on average, outperforming both on all models except o1 (see Appendix E.1).
claim: "bon/attacks/ contains only the authors' own attacks (text/image/audio BoN, PrePAIR, baseline resampling). There is no PAIR or TAP implementation or runner; the prompts/pair/ directory belongs to the authors' PrePAIR variant, not the PAIR/TAP baselines used for these comparison numbers."
concern: "The cross-method comparison claims (BoN beats PAIR by 33% / TAP by 39%) cannot be reproduced or checked for fair, like-for-like tuning because the baseline attacks are absent from the repo."
resolution: "Authors: please add or point to the PAIR/TAP baseline code and confirm it used the same dataset, classifier, and N budget as BoN."
cross_refs: ["App. E.1"]
paper_ref: "Section 3.1 Results; Appendix E.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: reliability-text-vision-absent
category: missing
topic: "result traceability / reliability"
title: "Text/vision resampling-reliability numbers (§6.4) have no computing code"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  At temperature 1, attacks generate harmful responses only 30% (text), 25% (vision), and 15% (audio) of the time on average.
claim: "The only reliability harness in the repo is experiments/audio_reliability/ (audio only). No script resamples successful text or vision jailbreaks to compute the 30% (text) / 25% (vision) reliability figures, nor the abstract's ~20% resampling-reliability headline."
concern: "Two of the three resampling-reliability numbers, and the abstract's headline reliability claim, are not reproducible from the repo."
resolution: "Authors: please add the text/vision resampling-reliability scripts or confirm they were run off-repo."
cross_refs: ["§6.4"]
paper_ref: "Abstract; Section 6.4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: experiment-outputs-not-committed
category: missing
topic: "reproducibility / committed outputs"
title: "No experiment outputs committed; all numbers require live commercial-API runs"
severity: medium
confidence: high
status: finding
file: code/jplhughes__bon-jailbreaking/.gitignore
line_start: 23
line_end: 23
quote: |
  exp
claim: "The exp/ directory that holds every attack's per-(n,k) responses, classifier judgements, and search-step jsonls is gitignored, and no exp/ data is present in the repo. The notebooks read from ./exp/bon/... which does not exist. Regenerating any reported ASR requires running the attacks against paid, nondeterministic commercial APIs (OpenAI/Anthropic/Google/GraySwan) at N=7,200–10,000 on 159 requests."
concern: "Because no intermediate outputs are shared, none of the paper's quantitative results can be reproduced without large paid-API budgets, and API nondeterminism means exact figures cannot be recovered even then; this is a reproducibility limitation inherent to a commercial-API black-box study."
resolution: "Authors: please release the per-request success/failure trajectory data (the exp/ search_steps or powerlaw jsonls) so ASR curves and final ASRs can be recomputed without re-querying the APIs."
cross_refs: ["forecasting-code-absent", "csv-circuitbreaking-asr-mismatch"]
paper_ref: "Section 3 (all reported ASRs)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: powerlaw-notebook-dead-import
category: bug
topic: "reproduction notebook"
title: "2_plot_powerlaw.ipynb imports bon.utils.power_law, which does not exist"
severity: medium
confidence: high
status: finding
file: code/jplhughes__bon-jailbreaking/experiments/2_plot_powerlaw.ipynb
line_start: 175
line_end: 175
quote: |
      "from bon.utils.power_law import (\n",
claim: "Cell 13 of the power-law reproduction notebook imports five symbols (exp_power_law, exp_power_law_single_term_no_constant, o1_simple_fit_power_law_no_constant, power_law, power_law_single_term_no_constant) from bon.utils.power_law. The module bon/utils/power_law.py does not exist in the repo (only power_law_simple.py and powerlaw_plot_utils.py do), and none of the five imported symbols are defined anywhere in the repo (verified by AST scan)."
concern: "The notebook the README points to for replicating the power-law figures raises ModuleNotFoundError on import and cannot run, so the Figure 3/4 power-law results are not reproducible via the shipped notebook."
resolution: "Authors: please add the missing bon/utils/power_law.py module (or fix the import to use power_law_simple), and re-run the notebook end-to-end."
cross_refs: ["forecasting-code-absent"]
check_script: _audit_code/check_missing_powerlaw_module.py
paper_ref: "Figures 3 and 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: random-cap-drops-nonascii
category: bug
topic: "text augmentation"
title: "apply_random_capitalization silently deletes non-ASCII letters when selected"
severity: low
confidence: high
status: finding
file: code/jplhughes__bon-jailbreaking/bon/attacks/run_text_bon.py
line_start: 328
line_end: 337
quote: |
      new_text = []
      for c in text:
          if c.isalpha() and random.random() < sigma ** (1 / 2):
              if "a" <= c <= "z":
                  new_text.append(chr(ord(c) - 32))  # Convert to uppercase
              elif "A" <= c <= "Z":
                  new_text.append(chr(ord(c) + 32))  # Convert to lowercase
          else:
              new_text.append(c)
      return "".join(new_text)
claim: "When a character passes c.isalpha() and the random gate but is neither in a-z nor A-Z (e.g. accented letters such as 'é', 'ñ'), neither the if nor elif branch fires and there is no else, so the character is dropped from the output entirely (verified: input 'café señor' with sigma=1 -> 'CAF SEOR')."
concern: "The augmentation is meant to recase letters, not delete them; on requests containing non-ASCII letters it removes information from the prompt, which the function's own intent contradicts (low impact since HarmBench direct requests are predominantly ASCII English)."
resolution: "Add an else branch that appends the original character when it is alphabetic but outside the a-z/A-Z ranges."
cross_refs: []
paper_ref: "Section 2 'Random Capitalization'; Appendix C.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: bootstrap-repeats-10-vs-100
category: difference
topic: "bootstrapping"
title: "Shipped notebook uses 10 bootstrap trajectories; paper states 100"
severity: low
confidence: high
status: finding
file: code/jplhughes__bon-jailbreaking/experiments/2_plot_asr.ipynb
line_start: 81
line_end: 81
quote: |
      "            dfs[model_name], model_path, num_repeats=10\n",
claim: "The ASR-plot reproduction notebook calls calculate_asr_trajectories with num_repeats=10, whereas the paper (Section 2 'Bootstrapping') states it plots the mean over 100 bootstrap trajectories and uses their standard deviation for error bars."
concern: "Mean ASR is unbiased under bootstrapping, but error bars from 10 trajectories are noisier than from 100; the released notebook does not reproduce the paper's stated error-bar procedure, though the underlying function supports num_repeats=100."
resolution: "Authors: confirm the paper figures used 100 trajectories and update the notebook default to 100, or clarify the discrepancy."
cross_refs: []
paper_ref: "Section 2 'Bootstrapping'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: csv-circuitbreaking-asr-mismatch
category: difference
topic: "released data consistency"
title: "Released text-jailbreak CSV implies 37% Circuit Breaking ASR vs paper's 52%"
severity: low
confidence: low
status: question
file: out/released_csv_asr.csv
csv_row: 4
quote: |
  Circuit Breaking,59,37.1,52,MISMATCH
claim: "docs/assets/data/text_jailbreaks.csv contains only successful jailbreaks (label==1), one per request; unique successes / 159 gives an implied ASR. For Sonnet/GPT-4o/Gemini-Pro/Llama3 this matches the paper within rounding (78/87/47/94%), but for Circuit Breaking the CSV has 59 unique successes (37.1%) versus the paper's reported 52%."
concern: "Either the website CSV is a partial/older Circuit Breaking run or the reported 52% used a different N or post-hoc correction; the discrepancy is unexplained, though the CSV is a website-display artifact and not claimed to reproduce Figure 1."
resolution: "Authors: clarify why the released Circuit Breaking successes (37%) differ from the reported 52% ASR — different N, different run, or human-grading corrections not reflected in the CSV?"
cross_refs: ["experiment-outputs-not-committed"]
check_script: _audit_code/check_released_csv_asr.py
paper_ref: "Section 3.1 ('52% ASR' on Circuit Breaking)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: bootstrap-tiling-extrapolation
category: methodology
topic: "bootstrapping / extrapolation"
title: "Bootstrap cycles the observed trajectory when N exceeds observed samples"
severity: low
confidence: medium
status: question
file: code/jplhughes__bon-jailbreaking/bon/utils/shotgun_utils.py
line_start: 662
line_end: 667
quote: |
          success_trajectory = np.stack(
              df.groupby("i")["flagged"].apply(
                  lambda x: np.tile(x.to_numpy(), (num_samples + len(x) - 1) // len(x))[:num_samples]
              )
          ).T
          np.apply_along_axis(lambda col: np.random.shuffle(col), 0, success_trajectory)
claim: "In bootstrap_type='sample_without_replacement', when the requested num_samples exceeds the number of observed steps for a request, the observed success/failure vector is tiled (cycled) and then shuffled, i.e. the same observed outcomes are reused to fill larger N rather than sampling new outcomes."
concern: "If this path is used to populate ASR trajectories at N beyond the observed budget (e.g. for forecasting validation), the 'expected ASR' at large N is constructed by recycling the same finite observations, which can bias the cumulative-success curve; whether this affects any reported number depends on which N values were actually requested, which the repo's missing forecasting driver does not show."
resolution: "Authors: confirm whether any reported ASR/forecast value was generated with num_samples greater than the observed per-request step count, and if so justify the tiling behavior."
cross_refs: ["forecasting-code-absent"]
paper_ref: "Section 2 'Bootstrapping'; Section 4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|-------------------------------------------------------------|
| missing     | 5          | medium       | Forecasting, composition, PAIR/TAP, reliability code + outputs absent |
| bug         | 2          | medium       | Powerlaw notebook crashes on import; aug drops non-ASCII letters |
| difference  | 2          | low          | Notebook 10 vs paper 100 bootstraps; CSV vs paper CB ASR     |
| methodology | 1          | low          | Bootstrap tiling when N exceeds observed samples (question)  |

## Top take-aways

1. **(missing)** No code computes the forecasting error (Fig. 4 / abstract 4.6%) — a
   central contribution is unreproducible from the repo. `forecasting-code-absent`
2. **(missing)** No analysis/plot code for the entire attack-composition section
   (Fig. 5 / §5 sample-efficiency 28x–250x). `composition-analysis-absent`
3. **(bug)** `2_plot_powerlaw.ipynb` imports a module (`bon.utils.power_law`) that does
   not exist; the shipped power-law notebook cannot run. `powerlaw-notebook-dead-import`
4. **(missing)** PAIR/TAP baseline comparisons (§3.1: "+33% / +39%") are not implemented;
   only the authors' own attacks ship. `pair-tap-baseline-absent`
5. **(missing)** No experiment outputs are committed (`exp/` gitignored); every ASR
   requires paid, nondeterministic commercial-API runs. `experiment-outputs-not-committed`
6. **(difference)** Released text-jailbreak CSV implies 37% Circuit Breaking ASR vs the
   paper's 52% (4/5 other models match within rounding). `csv-circuitbreaking-asr-mismatch`

## Items that genuinely look fine

- **Repo provenance**: This is the official repo. Implied ASRs from the released
  success CSVs match the paper for Sonnet (78%), GPT-4o (87%), Gemini Pro (~50%), and
  Llama3 8B (94%) — see `_audit_code/out/released_csv_asr.csv`.
- **Dependency specification**: `requirements.txt` pins all top-level dependencies to
  exact versions; Python version stated.
- **Dataset present**: the 159 HarmBench direct requests
  (`data/direct_request.jsonl`) and the vocalized audio inputs (`data/audio_files/`)
  are in the repo.
- **Core attack + ASR pipeline present**: text/image/audio BoN runners, the naive
  resampling baseline, the bootstrap ASR-trajectory code, and the power-law fitter all
  exist and read coherently; `2_plot_asr.ipynb` (the other notebook) does not have the
  missing-module problem.
- **Augmentation functions**: word scrambling, ASCII noising, and random capitalization
  match their docstrings (aside from the non-ASCII edge case in `random-cap-drops-nonascii`).
- **Classifier wiring**: harmfulness is graded by the HarmBench GPT-4o prompt with
  a too-short filter, recitation filter, and a false-positive-phrase list — consistent
  with the paper's manual-review-of-flags description.

## Open questions for the authors

- For each reported per-model ASR (text/vision/audio), which committed or releasable
  artefact lets a reviewer recompute it without re-querying the commercial APIs?
  (`experiment-outputs-not-committed`)
- Why does the released Circuit Breaking success CSV imply 37% while the paper reports
  52%? (`csv-circuitbreaking-asr-mismatch`)
- Was any reported ASR/forecast computed with `num_samples` exceeding the observed
  per-request step count (triggering the bootstrap tiling path)?
  (`bootstrap-tiling-extrapolation`)
