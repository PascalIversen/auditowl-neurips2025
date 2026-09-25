# START AUDIT

1. Read `audit_done.txt` (create it empty if missing). It lists the paper
   folders already audited, one per line.

2. Pick the next **10** empirical paper folders under `audits/` that are NOT in
   `audit_done.txt` (skip `audits/theory/`).

3. Spawn **one subagent per paper, all 10 in a single message** (parallel),
   each on **Opus 4.8 with 1M context**. Give each subagent:
   - its paper folder path,
   - "Audit this paper by following `auditowl/audit_prompt.md` exactly. The
     cloned repo is under `code/<owner>__<repo>/`, metadata in `metadata.txt`.
     Write `audit.md`, then run
     `python auditowl/extract_findings.py <folder>/audit.md --out <folder>/findings.json`.
     Stay read-only on `code/`."

4. When a subagent finishes, append its folder to `audit_done.txt`.
