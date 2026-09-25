# Merge-by-defect re-clustering prompt (`recluster-robustness`)

The adversarial LLM pass that produced the merged-by-defect clustering in
`data/merged_clusters.json` — the matcher behind every "merged" reliability
number (detection rate, Jaccard, Krippendorff α, Gwet AC1).

The workflow ran **one agent per paper** (`PAPERS = ['1829','1333','2657','2578','2371']`)
in parallel, each forced to return structured output against the schema below.
Input per paper: `/tmp/recluster/<num>.json` =
`{paper, n_runs, findings:[{fid, run, category, loc, title, claim}]}` (the original
`/tmp` inputs were not retained, but they are fully determined by the released
run findings: `build_recluster_inputs.py` regenerates them, and every `fid` the
committed partition references resolves against the result). `fid` strings are `r{run}#{idx}` indices into
each run's `findings.json`.

---

## Agent prompt (verbatim)

```
You are de-duplicating reproducibility-audit findings produced by 10 independent repeat audits of the SAME paper and SAME frozen codebase. Each run surfaced several findings; the same real defect is often described at a different file/line, under a different category, or in different words across runs. Your job: group findings that describe the SAME underlying defect into one cluster, so we can measure how reliably each TRUE defect is re-detected.

Read the file /tmp/recluster/${num}.json — it has {paper, n_runs, findings:[{fid, run, category, loc, title, claim}]}.

CLUSTERING RULE (be precise and conservative):
- Put two findings in the SAME cluster IFF they are about the SAME underlying defect — the same thing wrong with the same artifact/behaviour — EVEN IF they differ in:
    • category (one run files it 'missing', another 'methodology'/'bug'/'difference'),
    • location (one cites paper.pdf, another cites run.py; or different line ranges of the same mechanism),
    • wording / which specific file or table they name as the example.
  Example of SAME defect: "benchmark datasets + Q-matrices not in repo" cited at README.md, at a .sh script, and at model/OLinear_C.py are ONE defect (the data/Q-matrices aren't shipped). "best-on-test selection" cited in run.py and in exp_forecast.py is ONE defect.
- Keep findings in DIFFERENT clusters if they are genuinely distinct defects, even if same category or same file (e.g. "requirements unpinned" vs "wrong PyPI name 'pywt'" are arguably distinct — use judgment; if one clearly subsumes/duplicates the other across runs, merge; if they're independent problems, don't).
- Do NOT merge just because two findings share a topic or category. Merge only genuine same-defect duplicates.
- EVERY fid must be assigned to exactly one cluster. Singleton clusters (a defect only one run raised) are expected and correct — do not force-merge them.

Return: paper, n_findings_assigned (= total fids you placed; must equal the input findings count), and clusters (each = {label, fids:[exact fid strings]}). Use the exact fid strings from the file. Add a one-line notes on how aggressively you merged.
```

(`${num}` is substituted per paper.)

## Output schema (forced structured output)

```json
{
  "type": "object", "additionalProperties": false,
  "required": ["paper", "n_findings_assigned", "clusters", "notes"],
  "properties": {
    "paper": { "type": "string" },
    "n_findings_assigned": { "type": "integer",
      "description": "total fids placed into clusters (must equal the input count)" },
    "clusters": { "type": "array", "items": {
      "type": "object", "additionalProperties": false,
      "required": ["label", "fids"],
      "properties": {
        "label": { "type": "string", "description": "short description of the single underlying defect" },
        "fids":  { "type": "array", "items": { "type": "string" },
                   "description": "exact fid strings from the input file" }
      }
    }},
    "notes": { "type": "string" }
  }
}
```

The `n_findings_assigned == input count` requirement + downstream coverage check
in `make_worksheets.py` (every raw fid in exactly one cluster) is the
"complete + exact partition" validation referenced in `recluster_comparison.json`.

## Adversarial verification of the partition

Paper 2578's 16-defect partition was independently re-checked by a second
workflow, `verify-2578-clustering` (session `150ce47a-…jsonl`, line 198): three
adversarial reviewers under a merge-lens, split-lens, and assignment-lens, each
returning structured `should_merge / should_split / misassigned_fid / wrong_label`
issues.

## Note on the "upper bound" wording

`recluster_comparison.json` labels merged the "upper bound on reliability." The
prompt is explicitly **conservative** (merge only genuine duplicates; never
force-merge singletons; keep distinct defects separate), so it under-merges by
design relative to a coarse by-problem grouping. It is the *more lenient of the
two computed matchers* (vs. the strict anchor matcher), but relative to true
defect-level grouping it is a **conservative** estimate, not a ceiling — see the
methods text in `validation_section.tex`.
