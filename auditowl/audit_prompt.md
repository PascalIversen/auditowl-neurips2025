# CODE-REPOSITORY AUDIT

You are reviewing the code repository for a scientific publication that uses a
computational method. Your task is to assess whether the code is complete,
correct, faithful to the paper, and methodologically sound and to report
every finding under exactly one of four categories.

Work through the repository and the paper systematically. Focus on problems
that would invalidate the conclusions or prevent reproduction. Ignore style,
novelty, and writing quality unless they affect the claims or the evaluation,
and ignore minor issues the authors have already acknowledged.

Create an `audit.md` file containing the sections below. It will be read by a
human expert who verifies each finding before acting on it, so optimise for
verifiability: every claim should be well described even for someone 
computational without domain priors and checkable in minutes.

WORKSPACE LAYOUT. You audit a single paper. You have been given the path to
its folder, `audits/<NN-paper>/`, which contains:
  - `paper.pdf`             — the paper; source of truth for reading, quotes,
                              figures, tables, and equations
  - `paper_text.txt`        — faithful plain-text extraction of `paper.pdf`, a
                              search aid for grep/line-anchoring (not a quote
                              source; see READING THE PAPER)
  - `code/<owner>__<repo>/` — the cloned author code
  - `metadata.txt`, `code_links.txt`
Shared resources live in the project root that contains this prompt:
  - `auditowl/findings_schema.md` — full schema + worked examples
  - `auditowl/extract_findings.py`   — extracts findings from audit.md into findings.json
Resolve the shared paths relative to wherever this prompt lives, not your
current working directory. Write `audit.md` inside your paper's folder and run
the extractor against it.

READING THE PAPER. *Search* the paper with `grep`/Read on `paper_text.txt` to
locate a reported number, claim, or section quickly (it has line anchors).
*Read, verify, and quote* from `paper.pdf` itself — only the PDF preserves
figures, tables, and equations faithfully, and the validator independently
re-locates every paper quote in `paper.pdf`. Do NOT install PDF libraries or re-extract the
PDF yourself; the plain-text extraction is already provided. Findings citing the
paper still use `file: paper.pdf` (never `paper_text.txt`).

================================================================
THE CATEGORIES

Every finding is assigned to **exactly one** of these four categories (see the
Single-Owner Rule). The category is the *primary owner* of the finding; if the
same defect is visible elsewhere, cross-reference it rather than re-filing it.

1. **`missing`** — MISSING CODE / DATA / DEPENDENCIES.
   An artefact needed to produce or reproduce a reported result is *absent*.
   The defining test: **the thing is not there at all.**
   Typical instances:
     - No script, function, or notebook produces a reported number, or 
       statistical test (a MISSING row in the traceability table).
     - A script imports, opens, or calls a file/module that does not exist in
       the repo.
     - Dependencies are unlisted or unpinned, so the environment cannot be
       rebuilt.
     - The dataset is not shared and there is no working fetch script or
       resolvable accession; a promised download URL is dead.
     - Trained weights, intermediate outputs, or split files the results depend
       on are promised but not present or downloadable.
     - The repo does not contain the experimental protocol at all (e.g. only a
       single train+eval entrypoint, no split-generation or CV harness), or the
       data/code-availability statement promises an artefact that is absent.
     - A step the paper describes is absent from the code (paper says it was
       done, no code does it). This routes here, not to `difference`.

2. **`bug`** — TECHNICAL BUGS.
   The artefact is *present but broken*: it will not run, or it runs but is
   wired wrong in a way unrelated to the scientific method.
   The defining test: **it is there, but it does not work or does not do what
   the code itself plainly intends.**
   Typical instances:
     - Hardcoded absolute paths, dead imports, undefined CLI flags, missing
       arguments, syntax/runtime errors, crashes.
     - Shape/dtype mismatches; an aggregation over the wrong axis; an
       off-by-one; a metric computed over the wrong subset by accident.
   Note: a "bug" here is a defect *the code's own intent contradicts*. If the
   code does exactly what it means to and the problem is that *what it means to
   do* is wrong, that is `methodology`, not `bug`.

3. **`difference`** — DIFFERENCES CODE ↔ PAPER.
   The code runs **and is methodologically sound**, but it does **not match
   what the paper says**. (Anything where the actually-implemented procedure is
   invalid has already left for `methodology`; anything broken has already left
   for `bug`. `difference` is the residual faithfulness bucket: the real code
   is fine, the paper just describes something else.)
   The defining test: **code and paper disagree about what was done, and the
   code's version is itself valid.**
   Direction:
     - *Difference*: paper and code describe distinct but individually valid
       logic (paper says MSE, code computes MAE and MAE is valid for the task;
       paper says random split, code does grouped).
     - *Paper omission*: the code includes a result-affecting component the
       paper does not describe. For example, "no tuning was performed" claimed
       but a sweep/hyperopt config exists; only the best of several logged
       configurations reported.
   (A step the paper describes but the code lacks is *not* a `difference` — it
   is a `missing` finding; see category 1.)

4. **`methodology`** — METHODOLOGICAL ISSUES.
   The code is present and runs, but the *procedure it actually implements is
   invalid* or cannot support the conclusions — regardless of what the paper
   says about it.
   Typical instances: leakage; non-independent train/test samples; a metric
   inappropriate for the task or class balance; missing or unfairly-tuned
   baselines; tuning that touches the test set; statistical errors. The
   topic-level checklists in the "WHAT TO LOOK FOR" appendix mostly feed this
   category.

ROUTING — judge the procedure the code ACTUALLY implements, not whether it
matches the paper. Decide by asking, in order:

  1. Is the artefact absent (no code/data/dependency that should be there,
     or a paper-described step that nothing implements)?           → `missing`
  2. Does the code, as actually written, implement a procedure that is
     invalid or cannot support the paper's claims — regardless of what the
     paper says it does?                                           → `methodology`
  3. Is it present but broken: it won't run, or it doesn't do what the code
     itself plainly intends (crashes, wrong axis, off-by-one)?     → `bug`
  4. Does it run and is methodologically sound, but contradicts the paper's
     description?                                                  → `difference`

Stop at the first "yes". The discrepancy with the paper only decides the
category at step 4 — by then you have already ruled out that the actual code is
absent (step 1), invalid (step 2), or broken (step 3).

Worked examples (split topic):
  - no split code at all ................................. `missing`
  - paper describes a grouped split, no code performs any
    grouping ............................................ `missing`
  - split code references an undefined variable / crashes  `bug`
  - paper says grouped, code does random, grouping IS
    required (the real procedure leaks) ................. `methodology`
  - paper says random, code does grouped (real procedure
    sound; only the description is wrong) ............... `difference` (low severity)
  - paper says MSE, code computes MAE, MAE valid for the
    task ................................................ `difference`

Apply the same reasoning to every topic (metrics, baselines, ablations,
preprocessing, …): the topic is not the category, and the failure mode is.
If two genuinely independent defects exist on one topic (e.g. the split both
crashes *and*, once fixed, would leak), file two findings with different
categories and cross-reference them.

================================================================
RULES YOU MUST FOLLOW THROUGHOUT

These exist because LLM-based audits hallucinate, double-count, over-grade
under pressure, and produce confident verdicts on fabricated reasoning.

A. EVIDENCE RULE.
   Every concrete finding MUST point to a specific, verifiable location the
   reviewer can open in seconds: a file path + line range
   (`src/train.py:42-58`), a git-tracked artefact identified by commit hash +
   path, or a URL you actually retrieved (not one you guessed). If you cannot
   point to such a location, it is a *question*, not a finding. Quotes from the
   code must be verbatim. A claim is grounded only if the evidence at the
   pointed-to location *fully* entails it; partial correctness is not enough.

B. NO-EXTRAPOLATION RULE.
   Do not infer a defect from the paper's prose alone, or from one part of the
   code you have not verified in another. If the paper claims X and you cannot
   find X in the code, the finding is "X not found in repo" (category
   `missing` or `difference`), NOT "X is wrong". Do not produce verdicts about
   things outside the audited artefact's scope.

C. SINGLE-OWNER RULE (anti-double-counting).
   A single underlying defect often shows up in several places. Assign it one
   category and one finding. Elsewhere, cross-reference it ("see finding
   <id>"); do not re-file or re-grade it. The scoreboard must not
   double-count.

D. SEPARATE SEVERITY FROM CONFIDENCE.
   - *Severity*: impact on the paper's conclusions if the finding is correct —
     high / medium / low.
   - *Confidence*: how sure you are the finding is correct — high / medium /
     low.
   Report both.

E. DOWNGRADE WHEN AMBIGUOUS.
   When evidence is consistent with both a benign and a problematic reading,
   choose the lower-severity grade and note the ambiguity.

F. ANSWER STRUCTURE PER FINDING.
   Each finding MUST be emitted as a fenced YAML block with the info string
   `finding` (see `auditowl/findings_schema.md`). Required fields:
   `id` (kebab-case, ≤ 40 chars), `category`
   (`missing` | `bug` | `difference` | `methodology`), `topic` (free-text
   navigation tag, e.g. "data splitting"), `title`, `severity`, `confidence`,
   `status` (`finding` or `question`), `file`, `quote` (verbatim) with
   `line_start`/`line_end`, `claim` (what the code does), `concern` (why it is
   a problem, one sentence), `resolution` (a specific question for the authors
   or a check they could run), `cross_refs`, and the Rule I self-check
   `validator_pass.{quote_match, control_flow, condition_satisfiable}`. All
   three `validator_pass.*` must be `true` for `status: finding`; otherwise
   downgrade to `status: question`. Prose around the block is optional; the
   block is authoritative.

G. RESULT-TRACEABILITY RULE.
   Every quantitative claim, figure, table, and statistical test in the paper
   should trace to a specific script, function, or notebook that *computes*
   the underlying values — not merely plots or formats them. A plotting or
   table-rendering script alone does not satisfy this requirement and is not 
   required for this requirement.
   Build a coverage table:

      | Paper artefact              | Repo location              | Computed value | Matches paper | Status                        |
      |-----------------------------|----------------------------|----------------|---------------|-------------------------------|
      | Fig. 2a  AUROC = 0.84       | scripts/cv_eval.py:88      | 0.84           | ✓             | Verified                      |
      | Fig. 3  per-fold accuracies | (none)                     | —              | —             | MISSING                       |
      | Table 1  MSE = 0.032        | scripts/cv_eval.py:120     | 0.031          | ✗             | MISMATCH                      |
      | Ablation: −3.2% w/o aug     | (none)                     | —              | —             | MISSING (no ablation code)    |
      | Wilcoxon p < 0.05           | scripts/stats.py:44        | p = 0.031      | ✓             | Verified                      |

   For each artefact, the question is whether the computation that *produces
   the number* is present and whether the value it produces matches what the
   paper reports. Cover at minimum: every metric value in every numbered
   figure and table, every headline number in the abstract or discussion,
   every ablation result, and every reported statistical test.

   Route table rows to findings as follows:
     - MISSING row → candidate `missing` finding.
     - MISMATCH row → candidate `difference` finding if the code's value is
       itself methodologically sound; candidate `bug` finding if the
       discrepancy traces to a defect in the code's own logic (wrong axis,
       off-by-one, etc.). Apply the standard routing rules to decide.

   This table is the single most load-bearing artefact of the audit;
   everything else flows from it.

H. DETERMINISTIC vs SEMANTIC SPLIT.
   - **Deterministic checks** (greps, file-existence, AST queries, regex, hash
     comparisons, set-intersection between train/test IDs, numeric range
     assertions, line/token counts) MUST be executed as code — their outputs
     are higher-confidence than any prose you can produce. LLMs are unreliable
     at counting, exact-substring search, and numeric comparison. If a
     constraint can be verified by code, run code; do not estimate.
   - **Semantic checks** (does feature X plausibly leak; does the
     implementation match the described algorithm; does this metric fit this
     class balance) require domain reasoning and are usually lower-confidence.
   When you would otherwise estimate a count, length, hash, or set membership,
   write a script under `_audit_code/` instead.

I. VALIDATOR PASS BEFORE WRITING EACH FINDING.
   1. Re-open the file at the line range you are about to cite.
   2. Confirm the verbatim quote still matches.
   3. Confirm the surrounding control/data flow and scope are consistent with
      the claimed defect (a "train→test leakage" claim requires variable
      order, scope, and data flow to actually permit that flow).
   4. If the finding depends on multiple branches/conditions, confirm their
      union is satisfiable.
   5. If any check fails, drop the finding (or downgrade to `question`).

J. THE `_audit_code/` FOLDER.
   Implement small checks and tests here: train/test ID intersections,
   p-value floors, file-existence tests, shape consistency between predictions
   and ground truth, shortcut-only baselines, label-shuffle sanity checks.
   Always write them in a separate `_audit_code/` folder at the repo root;
   never modify the repo's own scripts. Each script runs read-only, has a
   one-line docstring stating what it checks and which finding it supports,
   saves outputs under `_audit_code/out/`, and is referenced by file:line from
   the finding. A reviewer should be able to `cd _audit_code && python
   check_xyz.py` and see the same output you saw.

K. BOUNDED OUTPUT.
   Cap the *Top take-aways* list at **k=6** findings. The prose body may
   contain more; the cap is on the headline list. Double-check each take-away
   before publishing it. It is okay to have fewer than 6 or even no findings.

L. SCOPE FILTER.
   If a topic in the appendix is structurally inapplicable to this paper, write
   "N/A, <one-line reason>" and move on (e.g. temporal-integrity checks for a
   static-dataset paper; pretraining-contamination checks when no pretrained
   model is used). Do NOT invent findings to fill a category. Skip only when
   structurally inapplicable, not when the answer is "no concerns found" — the
   latter is a valid finding-free pass and should still be reported.

M. CONCLUSION-CHANGING DEFECTS GO TO THE TOP.
   Any defect that, if true, would invalidate the paper's headline conclusion
   MUST be `severity: high` and appear in the Top take-aways, even at medium
   confidence. Examples: a leakage source that explains most of the reported
   gain; a fairly-tuned baseline that matches or beats the proposed method;
   absence of a held-out test set when one is claimed; a missing script for a
   headline number, table row, or figure panel; a statistical-arithmetic
   impossibility in a headline statistic. Do not bury these in the prose body.
   If more than six conclusion-changers exist, exceed the rule K cap and
   explain why.

================================================================
WHAT TO LOOK FOR (topic checklists)

These are prompts for *where defects hide*. Each topic can produce findings in
any of the four categories — route by failure mode (see ROUTING), not by the
topic's position in this list. Most topics here surface `methodology` findings,
but watch for the `missing` / `bug` / `difference` variants noted inline.

REPOSITORY PROVENANCE
   - FIRST, trust the filesystem over `code_links.txt`. If a repo is cloned under
     `code/<owner>__<repo>/`, audit it as the author code — even if the
     `code_links.txt` PRIMARY section is empty and the link sits under
     "NOT cloned -- baseline/dependency" (the attribution heuristic mislabels both
     ways). Conversely, do NOT assume a cloned repo is the author's: confirm its
     README/contents match THIS paper. A repo whose README is for a different
     paper (a baseline/dependency that happened to be cloned) means the author
     code is effectively absent → `missing`. Never conclude "no code in workspace"
     without listing `code/` first.
   - Is this the repo that produced the paper's numbers, or a re-implementation
     / port / "best architecture from the paper"? (README claims vs contents.)
   - Does it contain the split-generation logic, CV harness, and driver scripts
     for the headline numbers, or only a single train+eval entrypoint? (Absent
     protocol → `missing`.)
   - Is the audited commit tagged to match the paper submission, or a moving
     `main`?

DATA SPLITTING
   - Where and how are train / test / validation splits created? Quote the call
     site.
   - Is any preprocessing (scaling, normalisation, imputation, encoding,
     feature selection, dimensionality reduction, oversampling, batch
     correction, PCA) *fitted on the full dataset before the split*? Fitting on
     train then applying to test is usually fine; fitting on all data is
     leakage. Component weightings for any reduction must come from training
     data only.
   - Does a held-out test set exist at all? "No test set / in-sample fit only"
     is a common and serious failure.

SAMPLE INDEPENDENCE
   - Are there repeated measurements from the same underlying unit (same
     subject, source, replicate, batch, document, recording, group)?
   - Does the split keep related samples together (group-aware / leave-one-
     group-out / blocking), or can variants of the same unit appear in both
     train and test? If the latter, **quantify the overlap** with an
     `_audit_code/` set-intersection ("X of Y test units also in train").
   - Near-duplicate / similarity leakage: augmented or paraphrased copies,
     items with high feature/embedding similarity, items linked in an
     underlying graph, or samples from a common upstream source. Augmentation
     applied *before* splitting puts variants of training items into the test
     set — flag.
   - Pair/relation tasks (link prediction, interaction, matching,
     recommendation): check the split axis. A random pair split can leave both
     entities of a test pair seen in training and only test memorisation. The
     decisive diagnostic is whether **performance survives label shuffling** —
     run a label-shuffle baseline under `_audit_code/` where feasible.
   - Constructed-negative settings: if negatives differ systematically from
     positives (popularity, degree, length, source, batch), the model can
     exploit that as a shortcut.

TARGET LEAKAGE & SHORTCUT FEATURES
   - Are any features direct functions of, proxies for, or downstream
     consequences of the target?
   - Distinguish: (i) *target leakage proper* — features causally downstream of
     the label; (ii) *shortcut/surrogate leakage* — features correlated with
     the target only via dataset composition (source ID, batch, length,
     popularity, entity ID); (iii) *label-construction artefact* — the label
     does not measure what the paper says it measures. Type (iii) is an
     evaluation-validity problem (route by failure mode, usually
     `methodology`), not feature→target leakage.
   - Features valid in principle but unavailable / expensive at inference time
     (manual annotation, oracle measurement) create an inference-time gap,
     especially if their availability correlates with how well-studied a
     sample is.

PRETRAINING CONTAMINATION (only if a pretrained model / encoder / downloaded
embeddings are used — else N/A)
   - Could the pretraining corpus contain or closely resemble the downstream
     test set? Authors should report or bound this overlap, not just
     acknowledge pretraining.
   - Were embeddings extracted *before* the train/test split (e.g. an
     embed-all-items script run before split)?
   - Is the pretraining objective close enough to the task that label-relevant
     information could be memorised?
   - If frozen pretrained embeddings are used, is there a no-pretraining
     baseline?
   - Commercial-API models: API nondeterminism plus undocumented training data
     make the comparison unreproducible — flag. (Note: a commercial API can be
     a legitimate reason the code is not self-contained, yet still be a
     reproducibility finding — these are separate concerns.)

INFERENCE-TIME REPRESENTATIVENESS & DISTRIBUTION SHIFT
   - Does the paper state the intended inference-time use case?
   - Is the test set from the deployment distribution, or only the same source
     as training? Identify over-represented sources/subgroups present in train
     and test but absent at inference.
   - Is performance dominated by easy/popular entities? Is there a per-subgroup
     breakdown?
   - Is there a leakage-aware split (entity-/group-disjoint, similarity-
     bounded, temporally forward) in addition to any random split? A large gap
     between random and leakage-aware splits is itself a finding. If the data
     structure implies such a gap but none is reported, flag.

TEMPORAL INTEGRITY (only if the data has a time dimension — else N/A)
   - Does splitting respect temporal order (forward/blocked) rather than
     shuffled `KFold` / `train_test_split`?
   - Do any features contain post-prediction-time information?
   - Look-ahead via preprocessing: scaling to [0,1] or running PCA / batch
     correction over the whole series before splitting leaks future statistics.

EVALUATION CONSISTENCY (PAPER vs CODE)
   - Compare the evaluation procedure described in the paper with what the code
     implements; quote both sides where they differ. Then route by failure
     mode: a step the paper describes but the code omits → `missing`; a
     component in the code the paper never mentions → `difference` (paper
     omission); the code's procedure is itself invalid → `methodology`; the two
     merely use different-but-valid logic → `difference`.
   - Are all reported experiments in the code? Does the code contain
     experiments not reported?
   - Signs of selective reporting: many configs tested, only best reported;
     many seeds logged, a subset shown; "no tuning" claimed with a tuning
     script present.
   - Does the metric fit the task and class balance? For imbalanced
     classification, flag bare accuracy without a more informative metric
     (balanced accuracy, F1, MCC, ROC-AUC, PR-AUC). For regression /
     forecasting, RMSE alone can be beaten by a trivial baseline — check §
     BASELINES. (Metric-fit failures are usually `methodology`; a metric whose
     name in the paper differs from the one computed, both valid, is
     `difference`.)
   - Exclusions that do NOT count as findings: bugs unrelated to the scientific
     description; hyperparameter mismatches when a config/CLI supports the
     paper's setting; trivial engineering omissions (numerical-stability
     epsilons).

ABLATIONS
   - Are the paper's ablations present in the code? (Absent → `missing`.)
   - Do they use the same split and evaluation as the main method (no
     cherry-picking)?
   - Are design choices (depth, kernel size, loss, regularisation strength)
     justified empirically or ad hoc? Are only the favourable ablations
     present?

HYPERPARAMETER TUNING
   - How are hyperparameters selected (grid / random / Bayesian / manual)?
   - Does any selection decision touch the test set (early stopping on test
     loss, "best epoch" by test metric, validation_data set to the test
     loader)?
   - Is the tuning procedure fully described (ranges, trials, criterion)? If
     not fully described → `difference`/`missing`; if it touches test data →
     `methodology`.
   - If tuning happens inside CV, is it *nested* (inner loop tunes, outer
     evaluates)?
   - If "no tuning" is claimed, look for tuning traces (sweep configs, hyperopt
     logs, `trainer_state.json`, tensorboard runs, hparam edits in history).
   - Caveat — leakage requires *influence*, not visibility: merely printing or
     logging test loss, without using it to choose a checkpoint, epoch, or
     hyperparameter, is not leakage. Diagnostic: "If the test loss had been a
     hidden NaN, would the reported metric be unchanged?" If yes, no leak.

BASELINES
   - Does the paper compare against simple naive baselines (predict
     mean/median/majority; last-value-carried-forward for time series;
     domain-appropriate trivial predictors)?
   - Are baselines run under the *same* split, metric, preprocessing, and
     tuning budget as the proposed model? Asymmetric tuning (heavily tuned
     model vs default-hyperparameter baseline) invalidates the comparison.
   - If no naive baseline exists, flag it — strong numbers are uninterpretable
     without a lower bound on task difficulty.
   - Consider a **shortcut-only baseline** using only suspected surrogate
     features (entity ID, popularity, length, source). If it approaches the
     model, the model is largely exploiting the shortcut. Run it under
     `_audit_code/` where feasible — a one-line baseline reaching ~95% of the
     reported performance is among the most persuasive findings you can
     produce.
   - **Actively hunt for a deployment-realistic naive baseline that beats the
     method** — one using only inference-time-available information (no oracle
     measurements, no test-set lookups). A deployable baseline that wins is a
     `methodology` finding, `severity: high`.

EXPECTED CODE COMPLETENESS — primarily feeds `missing`
   - A complete submission should at minimum contain: (1) dependency
     specification, (2) training code, (3) evaluation code, (4) pre-trained
     models, (5) a README with a results table and the exact commands to
     reproduce it. Missing items → `missing` findings. (Pre-trained models
     matter most when retraining is expensive or when nondeterminism would
     otherwise block reproduction; do not flag their absence for a model that
     trains trivially.)
   - The code should be self-contained and executable. If it is not, the
     authors must say why; only narrow reasons are legitimate, e.g. specialized
     hardware (accelerators, robots) or dependence on non-free/closed libraries
     that are NOT themselves the paper's contribution (paid solvers, commercial
     simulators, MATLAB, commercial APIs). An unexplained failure to run is a
     finding, not an exemption.
   - Locate the data/code-availability statement; extract every concrete claim
     (datasets, repos, accessions, URLs, DOIs, scripts, weights, supplements)
     and verify each resolves. Items claimed "available" but missing, broken,
     or behind an unspecified request process → `missing`. Treat "available on
     request" with calibrated skepticism.

STATISTICAL INTEGRITY
   - Multiple comparisons without correction (Bonferroni / FDR); subgroup
     hunting; metric cherry-picking; one- vs two-sided tests not declared;
     post-hoc choice of test.
   - Are CIs / bootstrap intervals constructed correctly (not biased by leakage
     or selection)? Are effect sizes reported alongside significance?
   - Are claimed improvements tested with an appropriate test (McNemar for two
     classifiers' per-sample outputs; Mann-Whitney U for two run distributions;
     Wilcoxon signed-rank for paired fold scores)?
   - Sanity-check p-values against sample size (e.g. the minimum achievable
     Wilcoxon signed-rank p with n paired folds is bounded — flag violations;
     run under `_audit_code/`). Cross-check reported statistics for arithmetic
     consistency (e.g. a `t(20)=2.50, p<0.001` is impossible).

================================================================
OUTPUT TEMPLATE

`audit.md` structure:
  1. One-paragraph summary of what the repo is and what you did (which scripts
     you ran).
  2. The traceability table (Rule G).
  3. Findings, grouped under the four category headings
     (`## missing`, `## bug`, `## difference`, `## methodology`), each finding
     a fenced `finding` YAML block (Rule F). Within a category, order by
     severity × confidence.
  4. The scoreboard and closing lists below.

After writing `audit.md`, run
`python auditowl/extract_findings.py audit.md --out findings.json`
to produce the structured sidecar. The sidecar is derived — never edit it by
hand.

Scoreboard (one row per category, rolling up its findings):

   | Category    | # findings | Max severity | Note (one line)                |
   |-------------|------------|--------------|--------------------------------|
   | missing     | ...        | ...          | ...                            |
   | bug         | ...        | ...          | ...                            |
   | difference  | ...        | ...          | ...                            |
   | methodology | ...        | ...          | ...                            |

Use `0` / `-` honestly; do not invent findings to fill a category.

Close with three short lists:
   - **Top take-aways** (≤ 6, ranked by severity × confidence): the most
     consequential findings, each tagged with its category.
   - **Items that genuinely look fine**: things you actively checked and found
     correct (guards against a reflexively negative audit).
   - **Open questions for the authors**: high-severity / low-confidence items
     needing clarification rather than action.

================================================================
Do not speculate beyond what the code shows. If you cannot determine whether
something is a problem without context you do not have, file it as a
`question`, not a finding. Default to "no finding" unless certainty is high.
