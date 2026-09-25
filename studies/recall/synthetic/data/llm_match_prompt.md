# Fixed-cluster finding assignment (the synthetic recall arm, stage-B matcher)

You are matching reproducibility-audit findings from repeated blind audits against a
PRE-REGISTERED answer key of planted defects. Stage A already assigned the easy
cases by file/line anchor; you get the leftovers.

Read data/match_inputs/<pid>.json:
  {paper, seeds:[{sid, category, sites:[{file,line}], title, description}],
   baseline:[{id, category, file, title}],
   unassigned:[{fid, run, category, loc, title, claim, concern}]}

For EACH unassigned finding, output exactly one assignment:
  - "seed:<sid>"     — the finding describes the SAME underlying defect the seed
                       planted (same thing wrong with the same artefact/behaviour),
                       even if category, file, line, or wording differ.
                       Also set mechanism_correct: true iff the finding identifies
                       the defect's actual mechanism (not merely "something wrong
                       near that file").
  - "baseline:<id>"  — it re-detects a pre-existing (non-planted) finding.
  - "emergent"       — genuinely neither: a new observation about the repo.

BE CONSERVATIVE, exactly like a defect-level deduplicator: do not credit a seed
just because a finding touches the same file; the finding must be about the
planted problem. When torn between seed and emergent, choose emergent — recall
must not be inflated by generous matching.

Return JSON: {"paper": "<pid>", "assignments": [{"fid": "...", "run": "...",
"assigned_to": "...", "mechanism_correct": true|false|null}], "notes": "one line"}
Every input fid appears exactly once. mechanism_correct is null unless assigned_to
is a seed.
