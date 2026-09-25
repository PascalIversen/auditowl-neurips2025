# Code Audit — Paper 838: A Difference-of-Convex Functions Approach to Energy-Based Iterative Reasoning

## 1. Summary

The paper proposes **DCAReasoner**, an energy-based iterative-reasoning model whose
inference is a difference-of-convex (DCA) fixed-point routine with finite-step
convergence guarantees. The empirical claims are: **Table 1** (MSE and inference
time of DCAReasoner vs two baselines IRED and IREM on five continuous algorithmic
reasoning tasks, each in "same" and "harder" difficulty, mean ± standard error over
5 runs, ~10000 test problems), **Table 2** (MSE / accuracy / relative inference time
on a symptom-to-diagnosis text-classification task in DistilBERT embedding space),
**Figure 1** (energy landscape t-SNE), **Figure 2** (DCA convergence), and **Figure 3 /
Appendix B.4** (robustness to input noise on QR).

The cloned repo `DanielTschernutter__DCAReasoner` (single commit `ee0be72`,
2025-11-05) contains only the DCAReasoner method: a model (`model/dca_reasoner.py`),
a trainer, an evaluator computing MSE, the five synthetic dataset generators
(`data/datasets.py`, copied from the IREM release), and a single driver `main.py`
that trains+evaluates DCAReasoner once per dataset. Dependencies are pinned in the
README.

I read every Python file in the repo, cross-checked each dataset generator against
the dataset definitions in Appendix B.3, and grepped the entire repo for baseline
code, timing code, standard-error/multi-run aggregation, seeding, and the
text-classification / noise-robustness pipelines. I wrote one check under
`_audit_code/` (`check_eval_steps.py`) that reproduces the evaluator's loop-counting
logic. The repo runs and the implemented DCAReasoner procedure is methodologically
sound, but **large portions of the reported results have no producing code in the
repo**: the two baselines, all inference-time numbers, the 5-run standard errors,
Table 2, Figure 1, and Figure 3 are not reproducible from this repository.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — DCAReasoner MSE, all 5 tasks × 2 difficulties | `main.py` + `evaluation/evaluator.py:31-47` (MSELoss on `preds[-1]`) | code computes a single-run MSE; values not reproduced here (random, no seed) | unknown | Partial: metric computed, but single run only, no SE, no fixed seed |
| Table 1 — DCAReasoner **inference time [s]** (10 cells) | (none — no timing code anywhere) | — | — | MISSING |
| Table 1 — **IRED** MSE + inference time (20 cells) | (none — no IRED code) | — | — | MISSING (baseline absent) |
| Table 1 — **IREM** MSE + inference time (20 cells) | (none — no IREM code) | — | — | MISSING (baseline absent) |
| Table 1 — "mean and standard error … five evaluation runs" | (none — `main.py` runs once, no aggregation) | — | — | MISSING (no multi-run / SE harness) |
| "roughly 10000 test problems per difficulty level" | `evaluation/evaluator.py:31-47`, `main.py:52-55` | 21 batches × 512 = 10752 (see `_audit_code/out/eval_steps.txt`) | ≈ (off by one batch) | MINOR DISCREPANCY |
| Table 2 — text-classification MSE / accuracy / inference time | (none — no DistilBERT, embeddings, or symptom_to_diagnosis pipeline) | — | — | MISSING |
| Figure 1 — energy landscape t-SNE | (none) | — | — | MISSING |
| Figure 2 (App. B.1) — DCA convergence visualization | `model/dca_reasoner.py:150-173` produces energies/iterates; no plotting script for Fig. 2 | — | — | MISSING (figure-producing script) |
| Figure 3 (App. B.4) — robustness to input noise (QR) | (none — no noise-injection / sweep code) | — | — | MISSING |
| Dataset definitions (App. B.3) | `data/datasets.py:10-133` | generators match paper formulas (verified by hand) | ✓ | Verified |

## 3. Findings

## missing

```yaml finding
id: baselines-irem-ired-absent
category: missing
topic: baselines
title: Table 1 and Table 2 baselines (IRED, IREM) have no code in the repository
severity: high
confidence: high
status: finding
file: code/DanielTschernutter__DCAReasoner/main.py
quote: |
  from model import DCAReasoner, DCAReasonerSettings
  from trainer import DCAReasonerTrainer, DCAReasonerTrainerSettings
  from data import MatrixInverse, MatrixCompletion, Parity, QRDecomposition, MatrixMultiplication
  from evaluation import DCAReasonerEvaluatorSettings, DCAReasonerEvaluator
line_start: 1
line_end: 4
claim: The only driver, main.py, imports and runs DCAReasoner exclusively; a full-repo grep for "IREM", "IRED", "diffusion", or any baseline model/training code returns nothing. The two baselines whose MSE and inference-time columns dominate Tables 1 and 2 are entirely absent.
concern: The central empirical claim ("superior or on par but significantly faster than state-of-the-art IRED/IREM") cannot be reproduced or checked, and the baseline numbers (network-size scaling, training budget, fair tuning) cannot be verified from this repository.
resolution: Provide the baseline training/evaluation code (or the exact forks/commits of irem_code_release and ired_code_release with the wrapper used), the network-size scaling used to match parameter counts, and the scripts that produced the baseline columns.
cross_refs: [inference-time-no-timing-code, multi-run-se-absent]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: inference-time-no-timing-code
category: missing
topic: timing / efficiency claim
title: Reported inference times (Table 1, Table 2; "3-27x faster" headline) have no timing code
severity: high
confidence: high
status: finding
file: code/DanielTschernutter__DCAReasoner/evaluation/evaluator.py
quote: |
  def _eval(self) -> float:
      total_number_of_test_steps = 0
      if self.settings.max_number_of_test_steps:
          tqdm_total = self.settings.max_number_of_test_steps
      else:
          tqdm_total = len(self.loader)
      mse_metric = AverageMeter()
line_start: 31
line_end: 37
claim: The evaluator computes only MSE (AverageMeter over MSELoss). A full-repo grep for "time", "perf_counter", "timeit", or any wall-clock measurement returns nothing, yet Table 1 reports an "Inference-Time [s]" column for every cell and the abstract/results claim speedups of factors 3-27.
concern: The headline efficiency advantage — the paper's main selling point over IRED/IREM — is not produced by any code in the repository, so the timing numbers and the speedup factors are unverifiable and non-reproducible.
resolution: Add the timing harness (which device, warmup, batching, what is included in the measured region) used to generate the inference-time columns and the relative percentages in Table 2.
cross_refs: [baselines-irem-ired-absent]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: text-classification-task-absent
category: missing
topic: experiment coverage
title: Section 5.3 / Table 2 / Figure 1 text-classification experiment has no code
severity: high
confidence: high
status: finding
file: code/DanielTschernutter__DCAReasoner/data/datasets.py
quote: |
  from data.datasets import MatrixInverse, MatrixCompletion, Parity, QRDecomposition, MatrixMultiplication
line_start: 1
line_end: 1
claim: The repo exposes only the five synthetic algorithmic-reasoning generators. A full-repo grep for "distilbert", "bert", "embedding", "huggingface", "symptom", or "diagnosis" returns nothing. The entire Section 5.3 pipeline (DistilBERT fine-tuning, CLS-token embedding extraction, symptom_to_diagnosis data loading, training DCAReasoner in embedding space) that produces Table 2 and Figure 1 is absent.
concern: Table 2 (MSE/accuracy/inference time) and Figure 1 (energy landscape) — a full reported experiment — cannot be reproduced from this repository.
resolution: Provide the DistilBERT fine-tuning script, embedding-extraction code, the symptom_to_diagnosis loader, and the training/eval driver for the embedding-space experiment.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: multi-run-se-absent
category: missing
topic: statistical reporting / reproducibility
title: No multi-run or standard-error aggregation; main runs each dataset exactly once
severity: high
confidence: high
status: finding
file: code/DanielTschernutter__DCAReasoner/main.py
quote: |
      evaluator = DCAReasonerEvaluator(model, dataset, evaluator_settings)
      score = evaluator.eval()
      evaluator = DCAReasonerEvaluator(model, ood_datasets[dataset_name], evaluator_settings)
      ood_score = evaluator.eval()

      print(f"Score for {dataset_name}: {score}", flush=True)
      print(f"Score for {dataset_name} (ood): {ood_score}", flush=True)
line_start: 56
line_end: 62
claim: main.py trains one model per dataset and evaluates once, printing a single scalar MSE per difficulty. There is no loop over runs/seeds and no computation of a mean or standard error anywhere in the repo (grep for "std"/"standard error"/"stderr" returns nothing), yet Table 1 reports "mean ± standard error" over "five evaluation runs".
concern: The reported error bars (and therefore any implied stability/significance of the comparisons) are not produced by the repository, and the single-run point estimates cannot be matched to the table without the multi-run protocol.
resolution: Provide the script that performs the five evaluation runs and computes the per-cell mean and standard error reported in Table 1 (and the analogous procedure for Table 2).
cross_refs: [no-seeding-nondeterministic]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-seeding-nondeterministic
category: missing
topic: reproducibility / determinism
title: No random seeding anywhere; datasets and starting points are nondeterministic
severity: medium
confidence: high
status: finding
file: code/DanielTschernutter__DCAReasoner/data/datasets.py
quote: |
      def __getitem__(self, index):
          R_corrupt = np.random.uniform(-1, 1, (self.h, self.w))
          R_corrupt = R_corrupt.dot(R_corrupt.transpose())
line_start: 19
line_end: 22
claim: Every dataset's __getitem__ ignores `index` and freshly samples np.random each call (and DCAReasoner._sample_starting_point also calls np.random.uniform). A full-repo grep for "seed"/"manual_seed" returns nothing, so neither training data, test problems, nor DCA starting points are seeded.
concern: Results (Table 1 point estimates and the standard errors) are not reproducible to fixed values, and there is no fixed held-out test set — each evaluation draws fresh random problems — so reported numbers cannot be regenerated deterministically.
resolution: Add seeding (numpy + torch) for data generation, starting-point sampling, and the multi-run loop, and state which seeds produced the reported tables.
cross_refs: [multi-run-se-absent]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: noise-robustness-absent
category: missing
topic: experiment coverage
title: Appendix B.4 / Figure 3 noise-robustness experiment has no code
severity: low
confidence: high
status: finding
file: code/DanielTschernutter__DCAReasoner/evaluation/evaluator.py
quote: |
      def _init_loss_function(self):
          self.loss_function = torch.nn.MSELoss()
line_start: 28
line_end: 29
claim: The evaluator applies the model to clean inputs only; a full-repo grep for "noise"/"gaussian" returns nothing. The QR noise-injection sweep over scales {1e-4 … 10} that produces Figure 3 / Appendix B.4 is absent.
concern: The noise-robustness claim ("our predictions are robust to noisy input data") cannot be reproduced from this repository.
resolution: Provide the noise-injection evaluation script used for Figure 3.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: eval-step-off-by-one
category: bug
topic: evaluation loop
title: Evaluator runs 21 test batches instead of the configured 20 (off-by-one break)
severity: low
confidence: high
status: finding
file: code/DanielTschernutter__DCAReasoner/evaluation/evaluator.py
quote: |
              total_number_of_test_steps += 1
              if self.settings.max_number_of_test_steps and total_number_of_test_steps >= self.settings.max_number_of_test_steps + 1:
                  break
line_start: 44
line_end: 46
claim: The break condition uses `>= max_number_of_test_steps + 1`, so with max=20 (set in main.py) the loop executes 21 batches (the metric is updated on each), evaluating 21*512 = 10752 problems rather than the intended 20 batches / 10240. Verified by `_audit_code/check_eval_steps.py` (out/eval_steps.txt).
concern: The code evaluates one more batch than the configured `max_number_of_test_steps`, contradicting the code's own intent; impact is small because the metric is an average over many problems and the paper states "roughly 10000".
resolution: Confirm whether 20 or 21 batches were used for the reported numbers, and fix the break to `>= max_number_of_test_steps` if 20 was intended.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: matmul-divide-by-five
category: difference
topic: dataset definition
title: Matrix Multiplication target is scaled by 1/5 in code but described as M^2 in the paper
severity: low
confidence: medium
status: finding
file: code/DanielTschernutter__DCAReasoner/data/datasets.py
quote: |
          R = np.matmul(R_corrupt, R_corrupt).flatten() / 5.
          return R_corrupt.flatten(), R.flatten()
line_start: 128
line_end: 129
claim: The Matrix Multiplication generator returns matmul(A, A)/5, i.e. the target is A^2 divided by 5. Appendix B.3 and Section 5.2 describe the target as "the square, i.e., M^2" with no scaling factor mentioned.
concern: The scaling changes the magnitude of the regression target and therefore the absolute MSE reported for this task, so the paper's stated target definition does not exactly match the code (the divide-by-5 is plausibly inherited from the original IREM dataset and is itself valid, just undocumented).
resolution: State the 1/5 scaling in the dataset description, or confirm it matches the baseline (IREM/IRED) target scaling so the comparison is on identical targets.
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A — the DCAReasoner procedure that *is* implemented is methodologically sound:
the evaluator computes MSE (matching the paper's metric) on the converged DCA
iterate; train and test problems are independent fresh random draws from synthetic
generators (no fixed test set is needed and collision probability is effectively
zero, so this is not leakage); no preprocessing is fitted across a split; and no
tuning decision touches a test set (there is no tuning loop at all). The
methodological concerns about the *comparison* (fair baseline tuning, parameter-count
matching, timing methodology) cannot be evaluated because the baselines and timing
code are absent — those route to `missing` (see `baselines-irem-ired-absent`,
`inference-time-no-timing-code`), not `methodology`.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 6          | high         | Baselines, inference-timing, Table 2 text task, 5-run/SE harness, seeding, noise experiment all absent. |
| bug         | 1          | low          | Evaluator runs 21 batches instead of 20 (off-by-one break). |
| difference  | 1          | low          | Matrix-Multiplication target scaled 1/5, not stated in paper. |
| methodology | 0          | -            | Implemented DCAReasoner procedure is sound; comparison soundness unverifiable (absent baselines). |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `baselines-irem-ired-absent`** — IRED and IREM, the comparison baselines whose numbers fill most of Tables 1 and 2, have no code in the repo; the "superior/on-par but faster" headline is not reproducible. (high/high)
2. **[missing] `inference-time-no-timing-code`** — the inference-time columns and the "3-27x faster" speedup claims have no timing code at all. (high/high)
3. **[missing] `text-classification-task-absent`** — the entire Section 5.3 experiment (Table 2, Figure 1: DistilBERT, embeddings, symptom_to_diagnosis) is absent. (high/high)
4. **[missing] `multi-run-se-absent`** — `main.py` runs each dataset once; no mean/standard-error aggregation exists, yet Table 1 reports SE over five runs. (high/high)
5. **[missing] `no-seeding-nondeterministic`** — no seeding anywhere; data, test problems, and DCA starting points are nondeterministic, so reported values cannot be regenerated. (medium/high)
6. **[bug] `eval-step-off-by-one`** — evaluator runs 21 test batches instead of the configured 20 (low impact). (low/high)

### Items that genuinely look fine
- The five synthetic dataset generators (`data/datasets.py`) match the formulas in Appendix B.3, including the harder/OOD variants (verified by hand against the text).
- The evaluation metric is MSE (`torch.nn.MSELoss`), matching the paper.
- DCAReasoner's hyperparameters in `main.py` (Nx=8, Ny=4000, tol=1e-5, max 30 DCA iters, lr=1e-4, batch 512, 10000 train steps, l/u bounds ±1 or ±5 for QR/MatMul) match Appendix B.2 / Table 3.
- No train/test leakage: train and eval batches are independent fresh random draws; no preprocessing fitted across a split; no tuning loop touches any test set.
- Dependencies are pinned in the README.

### Open questions for the authors
- Which seeds and how many runs produced the exact Table 1 / Table 2 numbers, and where is the aggregation script?
- What is the timing methodology (device, warmup, measured region) behind the inference-time columns?
- Is the 1/5 scaling of the Matrix-Multiplication target applied identically to the baselines, so the MSE comparison is on identical targets?
