# Findings schema — structured output for `audit.md`

Every finding in `audit.md` MUST be expressed as a fenced YAML block with the
info string `finding`:

~~~markdown
```yaml finding
id: pair-split-leakage
category: methodology
topic: "data splitting"
title: "Random pair-split leaks every cell line and drug into test"
severity: high
confidence: high
status: finding
file: prog/run_DeepCDR.py
line_start: 128
line_end: 136
quote: |
  def DataSplit(data_idx,ratio = 0.95):
      data_train_idx,data_test_idx = [], []
      for each_type in TCGA_label_set:
          data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
          train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
          test_list = [item for item in data_subtype_idx if item not in train_list]
          data_train_idx += train_list
          data_test_idx += test_list
      return data_train_idx,data_test_idx
claim: "Stratified 95/5 random split over (cell-line, drug) pairs."
concern: "Every cell line and drug in test also appears in train paired with a different counterpart; the model is allowed to memorise per-cell-line and per-drug IC50 means rather than generalise."
resolution: "Provide cell-line-blind and drug-blind split implementations, or quantify per-entity overlap."
cross_refs: ["§4.2"]
check_script: _audit_code/check_pair_split_overlap.py
paper_ref: "Section 'Data Splitting Strategy'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```
~~~

The YAML block is authoritative. Prose around the block is for human readers
and may add narrative; tools consume the YAML. Keep them consistent.

## Why a structured block

- A VS Code reviewer extension can list findings, jump to `file:line`, and
  show a verify / reject button per finding without parsing prose.
- The cited `quote` can be re-checked against current file contents at the
  cited `file:line` (drift detection) by re-reading the source.
- The summary scoreboards in `analysis/` can be regenerated from
  `findings.json` instead of regrepping prose.
- Findings can be deduplicated across audits by `id` (extractor prepends the
  audit folder name to make IDs globally unique).

## Field reference

### Required fields

| Field | Type | Notes |
|---|---|---|
| `id` | string | Defect slug, aim for kebab-case and ≤ 40 chars (a style convention, not mechanically enforced by the extractor). Stable across re-runs of the same audit. Globally namespaced as `<audit>/<id>` by the extractor. |
| `category` | enum | Owning category per the Single-Owner Rule (Rule C). One of `missing` \| `bug` \| `difference` \| `methodology`. |
| `topic` | string | Free-text navigation tag for where the defect lives (e.g. `"data splitting"`, `"baselines"`, `"statistical integrity"`). Not an enum. |
| `title` | string | Short title, ≤ 80 chars. |
| `severity` | enum | `high` \| `medium` \| `low`. Impact on the paper's conclusions (Rule D). |
| `confidence` | enum | `high` \| `medium` \| `low`. How sure you are. Independent of severity (Rule D). |
| `status` | enum | `finding` \| `question`. Use `question` when evidence is incomplete (Rule A / B). |
| `file` | string | Repo-relative path (`prog/train.py`), or `paper.pdf`, or a script-output CSV (`out/check_auc.csv`), or a URL you actually retrieved. |
| `quote` | string \| block | Verbatim quote from the evidence location. For code, include the full cited range. |
| `claim` | string | What the code / artefact does (Rule F). |
| `concern` | string | Why it is a concern (Rule F). One sentence preferred. |
| `resolution` | string | A specific question for the authors or a specific check (Rule F). |

### Optional fields

| Field | Type | Notes |
|---|---|---|
| `line_start` | int \| null | First line of the cited range. Null for paper / URL evidence. |
| `line_end` | int \| null | Last line of the cited range. Equal to `line_start` for a single line. |
| `cross_refs` | list[string] | Ids of related findings the same defect touches (`["random-pair-split", "shortcut-baseline"]`); free-form paper-section pointers (`"§4.2"`) are also allowed. Empty list if none. |
| `check_script` | string | Path to a `_audit_code/` script that supports the finding. |
| `paper_ref` | string | Free-form pointer into the paper (`"Table 2, row 3"`, `"Methods §3.2"`). |
| `validator_pass` | object | Rule I self-check. Keys: `quote_match`, `control_flow`, `condition_satisfiable`, all booleans. All `true` for findings; if any are `false`, downgrade to `status: question`. |
| `csv_row` | int | If `file` is a CSV from `scripts/check_*.py`, the row index of the finding within that CSV. |
| `url_retrieved_at` | ISO 8601 string | If `file` is a URL, when you actually retrieved it. |

### Evidence kinds

The `file` field disambiguates by prefix / extension:

- **Code** — repo-relative path (`prog/run.py`). `line_start` and `line_end` required.
- **Script output** — `out/check_*.csv`. `csv_row` required.
- **Paper** — `paper.pdf` or `supplement.pdf`. `paper_ref` required; line numbers null.
- **URL** — `https://...`. `url_retrieved_at` required; line numbers null.

If you cannot point to one of these, the item is not a finding — drop it or
file as a question with `status: question` (Rule A / B).

## Schema-level invariants

1. **Single-owner**: a defect appears as a finding in exactly one category
   (the owner). Other findings cross-reference it via `cross_refs`. The
   scoreboard must not double-count.
2. **Quote verbatim**: `quote` must be byte-equal to the cited range in the
   evidence location. Whitespace and indentation preserved, so the quote can be
   re-checked against the current file at `file:line` to detect drift.
3. **Validator gate**: a `finding` requires `validator_pass.*` all `true`.
   A `question` does not.
4. **Severity / confidence independence**: do not collapse a high-severity
   low-confidence item into a medium-severity entry. Report both
   dimensions; the VS Code extension surfaces them separately.
5. **ID stability**: an `id` should be stable across re-runs of the audit
   against the same repo and the same paper. Use a slug derived from the
   defect itself (`pair-split-leakage`), not from line numbers
   (`finding-128-136`), so that minor refactors of the audited code do not
   re-key findings.

## Worked examples

### Code finding (most common)

```yaml finding
id: random-seed-not-propagated
category: methodology
topic: "reproducibility / seeding"
title: "Only random.seed set — numpy / TF / cuDNN unseeded"
severity: medium
confidence: high
status: finding
file: prog/run_DeepCDR.py
line_start: 266
line_end: 266
quote: |
  random.seed(0)
claim: "Only Python's random module is seeded; numpy, TensorFlow, Keras dropout, and cuDNN are not."
concern: "Reported 'five independent runs' differ only in NN init noise; the train/test split itself is identical across the five runs because it is determined by random.seed(0)."
resolution: "Set numpy.random.seed, tf.random.set_seed, and use unique seeds across runs; document which sources of randomness were varied across the five replicates."
cross_refs: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### Question (low evidence)

```yaml finding
id: tcga-patient-eval-script-missing
category: missing
topic: "result traceability"
title: "TCGA patient-validation script not in repo"
severity: medium
confidence: medium
status: question
file: README.md
line_start: 100
line_end: 114
quote: |
  ## Patient data format
  Each row of patient_*.csv is (patient_id, drug, predicted_IC50)...
claim: "README describes the patient-validation data format but no script in prog/ consumes it."
concern: "The paper reports a TCGA patient validation (54 records / 31 patients / 12 drugs) but the script that produces those predictions is not in the repo."
resolution: "Authors: please point to the patient-evaluation script, or confirm it was run off-repo."
cross_refs: ["§11"]
```

### Script-output finding

```yaml finding
id: auc-ci-impossibly-narrow
category: methodology
topic: "statistical integrity"
title: "Reported AUC SE = 0.003 incompatible with (n+=120, n-=180)"
severity: high
confidence: high
status: finding
file: out/check_auc.csv
csv_row: 3
quote: |
  paper_auc=0.93,paper_se=0.003,nplus=120,nminus=180,hanley_mcneil_se_floor=0.018,verdict=IMPOSSIBLE
claim: "Hanley-McNeil distribution-independent lower bound on AUC SE for (k=0.93, n+=120, n-=180) is 0.018; the paper reports SE=0.003."
concern: "Reported SE is below the mathematical floor by a factor of 6; either the AUC, the SE, or the sample sizes are misreported."
resolution: "Authors: clarify how SE was computed (DeLong with paired data? bootstrap? per-fold?) and confirm sample sizes."
cross_refs: []
check_script: scripts/check_auc.py
paper_ref: "Table 2, ROC-AUC column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### Paper-only finding (statistical-integrity)

```yaml finding
id: grim-table1-impossible-mean
category: methodology
topic: "statistical integrity"
title: "Table 1 mean 3.42 impossible for n=23 on integer Likert 1–5"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  "Group A (n = 23): mean = 3.42, SD = 1.12"
claim: "On an integer 1–5 scale with n=23, no integer sum produces a mean of 3.42 to two decimal places. Nearest plausible means are 3.39 (sum=78) and 3.43 (sum=79)."
concern: "GRIM failure: the reported mean cannot be produced by any combination of integer responses at the stated sample size."
resolution: "Authors: provide the per-respondent data or clarify whether n includes fractional weights."
cross_refs: []
check_script: scripts/check_grim.py
paper_ref: "Table 1, Group A row"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Side-car artefacts

After the agent writes `audit.md`, run:

```
python <AUDITOWL>/auditowl/extract_findings.py audit.md --out findings.json
```

This produces a flat `findings.json` consumable by:

- the VS Code reviewer extension (click-through verification);
- the cross-audit summary scripts in `analysis/`;
- any downstream dedup / search / aggregation.

If you do not run the extractor, the `audit.md` is still self-contained — the
YAML blocks render as code blocks in a normal markdown viewer.
