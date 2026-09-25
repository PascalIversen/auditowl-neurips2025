# AuditOwl: agentic code auditing of NeurIPS 2025 papers

Code and data for *"AuditOwl: evidence-trace agentic code auditing"*.

An LLM agent audits a paper against its released code, tracing each headline
claim to the code that produces it and recording every discrepancy with a
re-runnable check. A second agent adversarially re-verifies each finding. We
apply this to a random sample of NeurIPS 2025 main-track papers and validate the
output four ways: expert human review, seeded-defect recall, cross-model-family
re-verification, and repeated-run reliability.

## Run AuditOwl on your own paper

The audit is a prompt that an agent follows, so all you need is an agentic
coding tool (we used [Claude Code](https://claude.com/claude-code) with Claude
Opus 4.8, 1M context) and this repository.

1. **Make a folder for your paper** with the PDF, a plain-text copy of it, and
   your code:

   ```bash
   pip install -r requirements.txt pymupdf requests beautifulsoup4
   mkdir -p audits/mypaper/code
   cp /path/to/paper.pdf audits/mypaper/paper.pdf
   git clone https://github.com/<owner>/<repo> audits/mypaper/code/<owner>__<repo>
   python -c "
   import sys; sys.path.insert(0, 'auditowl')
   from fetch_paper_inputs import read_pdf_text
   open('audits/mypaper/paper_text.txt', 'w', encoding='utf-8').write(
       read_pdf_text('audits/mypaper/paper.pdf')[2])"
   ```

2. **Run the audit.** Start the agent in the repository root and ask it:

   > Audit the paper in `audits/mypaper/` by following `auditowl/audit_prompt.md`
   > exactly. The code is under `code/<owner>__<repo>/`. Write `audit.md`, then
   > run `python auditowl/extract_findings.py audits/mypaper/audit.md --out
   > audits/mypaper/findings.json`. Stay read-only on `code/`.

   The agent writes its own checks to `audits/mypaper/_audit_code/`.

3. **Verify the findings (recommended).** In a fresh session, ask a second
   agent to follow `auditowl/verifier_prompt.md` on the same folder. It
   re-checks every finding at its cited location and writes
   `findings_verified.json`. `auditowl/launch_verify.md` describes the full
   two-pass setup we used (Sonnet 4.6, with Opus 4.8 re-judging any change).

4. **Read `audit.md`.** Every finding quotes the paper or the code at a file and
   line, so each one is quick to confirm or dismiss. Do not form a verdict
   based on the results without checking the accuracy and relevance of the
   findings. The AI can make mistakes.

`auditowl/launch_audit.md` and `auditowl/launch_verify.md` show how we ran
batches of papers in parallel.

## Install

```bash
pip install -r requirements.txt      # Python 3.9+, developed on 3.10
```

## Layout

```
auditowl/     the method: prompts, finding schema, sampling, input fetch
audits/       the dataset: one directory per audited paper
analysis/     the cross-paper data layer most figures read
studies/      the four validation studies, each self-contained
figures/      one script per paper figure, named after the figure
```

## Reproducing the figures

Every paper figure has exactly one script, named after it. Each writes PDF + PNG
to `figures/out/`.

```bash
python analysis/aggregate.py                  # build the data layer first

python figures/fig02_finding_categories.py    # Fig 2  discrepancy categories
python figures/fig03_code_coverage.py         # Fig 3  per-paper code coverage
python figures/fig04_scorecard.py             # Fig 4  reproducibility scorecard
python figures/fig05_reliability.py           # Fig 5  test-retest stability
python figures/fig06_expert_review.py         # Fig 6  expert review precision
python figures/fig07_recall.py                # Fig 7  recall assessment
python figures/figS1_severity_confidence.py   # Fig S1 severity x confidence
python figures/figS2_calibration.py           # Fig S2 manual-judgement calibration
python figures/figS3_cost_runtime.py          # Fig S3 cost and runtime
python figures/figS4_reviewer_time.py         # Fig S4 reviewer time
```

Figure 1 is a hand-drawn schematic and has no script.

Other entry points:

```bash
python analysis/print_funnel.py                          # code-availability funnel
python analysis/print_fig2b_provenance.py                # Fig 2b cards vs. the findings they summarize
python studies/expert_review/analyze_ratings.py          # Table S2 + agreement stats
python studies/expert_review/table_s1_by_category.py     # Table S1
python studies/expert_review/gpt_reviewer_verdicts.py    # the GPT reviewer's verdicts
python studies/expert_review/analyze_overlap.py          # author vs. human-eval agreement on shared findings
python studies/cross_model/compare_verdicts.py           # raw cross-model agreement
python studies/cross_model/paper_agreement.py            # both readings + the adjudicated outcome
python studies/recall/real/compute_recall.py             # real-discrepancy recall
python studies/recall/real/scicoqa_judge.py               # re-judge AuditOwl's side (GPT-5.5)
python studies/recall/real/judge_baselines_gpt55.py       # re-judge the baselines' side (GPT-5.5)
```

The committed JSON under `analysis/data/` and `studies/*/data/` is exactly what
these scripts produce; re-running them leaves those files unchanged.

## What is where

### `auditowl/`: the method

| File | What |
|---|---|
| `audit_prompt.md` | the audit prompt; this *is* the method |
| `verifier_prompt.md` | the adversarial verification prompt |
| `launch_audit.md`, `launch_verify.md` | how a run is launched |
| `findings_schema.md` | the schema every finding conforms to |
| `extract_findings.py` | `audit.md` → `findings.json` |
| `sample_papers.py` | draws the paper sample (fixed seed) |
| `scrape_accepted_papers.py` | builds the accepted-paper population |
| `fetch_paper_inputs.py` | fetches each `paper.pdf` and clones its repo |

### `audits/<paper_id>/`: the dataset

Each folder is named by the paper's number in our sampling frame.

**What an audit describes.** Every audit reflects the authors' released code as
we retrieved it between 29 May and 4 June 2026. The exact commit (or archive)
of each repository is recorded in `analysis/data/repo_revisions.json`. Several
authors have updated their repositories since, so a finding can describe code
that no longer exists in the current version. Findings are machine-generated
candidate discrepancies, not judgments of the work, and the authors' own
responses to them are not part of this release (see *Author correspondence*
below).

> **These reports are LLM output and individual findings can be wrong.** They
> are published so the aggregate statistics in the paper can be checked, and
> because a finding that cites a file and line is cheap for that paper's authors
> to confirm or dismiss. They are not a verdict on any paper.
>
> A finding is also not evidence of bad science. Most of them describe code that
> is incomplete, undocumented, or out of step with the text, which is the
> ordinary state of research code released under current community standards. Almost none of it
> speaks to whether the underlying result is true, and none of it speaks to
> whether the work was done honestly. Treat any individual finding as a prompt
> to look at the code, not as an established defect, and do not judge a paper
> on this output without a human checking the specific claim. See the paper for
> our measured error rate.

| File | What |
|---|---|
| `audit.md` | the agent's full report |
| `findings.json` | findings extracted from it |
| `findings_verified.json` | after adversarial verification |
| `findings_verified_codex*.json` | after cross-model-family verification |
| `_audit_code/`, `_verifier_code*/` | the re-runnable checks both agents wrote |
| `metadata.txt`, `code_links.txt`, `fetch_manifest.json` | inputs and provenance |
| `repo_provenance.json` | is the cloned repo the authors' own, or a baseline? |

The exact revision each repository was audited at is recorded in
`analysis/data/repo_revisions.json` (a git commit where available, otherwise
the archive/ZIP source). That is what lets a reader re-resolve a `file:line`
citation against the tree the auditor actually read, even though the clones
themselves are not redistributed.

### `analysis/`: the cross-paper layer

`aggregate.py` reads every paper's findings and writes `data/figure_data.json`,
which most figures consume. `print_funnel.py` prints the code-availability
funnel. `all_findings.md` lists every finding in readable form.

### `studies/`: the four validation studies

| Directory | Study |
|---|---|
| `expert_review/` | expert raters judge findings for correctness and relevance |
| `reliability/` | the same audit repeated 10× on each of 5 papers |
| `recall/synthetic/` | defects deliberately seeded into real repositories |
| `recall/real/` | recall on real, dated paper–code discrepancies |
| `cross_model/` | re-verification by a different model family |

## What is deliberately not here

**The audited papers' code.** `audits/<paper>/code/` is absent: we cannot
redistribute third-party repositories. `code_links.txt` and
`fetch_manifest.json` record exactly what was cloned, and
`auditowl/fetch_paper_inputs.py` re-fetches it. The analysis does not need it:
`analysis/data/repo_fingerprints.json` caches the derived per-repo facts, so
every figure regenerates from this repository alone. Same policy for
`studies/recall/real/judge_baselines_gpt55.py`'s input: SciCoQA's own raw
baseline predictions are not redistributed here; the script expects them
fetched into a local `_scicoqa_raw_cache/` from SciCoQA's own release
(`github.com/UKPLab/scicoqa`, `out/inference_discrepancy_detection_real.tar.gz`).
The committed `baselines_gpt55_judge.json`/`baselines.json` already carry the
resulting verdicts, so re-fetching is only needed to regenerate them.

**Excluded papers.** 6 of the 106 drawn papers were excluded before the
empirical sample of 100; `audits/theory/` and `audits/excluded_justified_no_code/`
are not redistributed, same policy as the audited papers' code above.
`analysis/data/sampling_frame_exclusions.json` records each excluded paper's
number, exclusion category and the checklist answer/justification that
triggered it.

**Author correspondence.** We contacted the authors of every sampled paper and
report their verdicts only in aggregate, per the paper's Ethics Statement. The
aggregate is released: `studies/expert_review/author_response_verdicts.csv` has
one row per judged discrepancy carrying its category and the two author
verdicts, with no paper identifier, no issue id and no free text, sorted so the
ordering carries no grouping. Section S1.3's category-level correctness/relevance
counts recompute from it; the 27/100 response rate and the rejection-cause
breakdown do not, since the csv carries no paper identifier and no free text.
`overlap_comparison.csv` is the same kind of
release for the findings both authors and human reviewers judged: one row per
overlapping finding carrying both sides' verdicts, same anonymisation.

**Reviewer identities.** Expert reviewers appear as `R1`–`R11`, matching the
roster table in the appendix.

**Agent session transcripts.** `figures/figS3_cost_runtime.py` derives its
numbers from local agent logs that are not part of the release. Those numbers
are committed in `analysis/data/compute_cost.json`, so the script re-plots the
figure from them; only re-deriving requires the raw logs.

## Two funnels

`analysis/print_funnel.py` reports the funnel used in the paper: papers binned
by the fraction of *traced result artifacts* whose producing code was found.
`analysis/aggregate.py` also computes a stricter scorecard funnel that layers
install, entrypoint and completeness gates on top, so it ends lower. They answer
different questions; the paper cites the former.
