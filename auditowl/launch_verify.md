# START VERIFY

1. Read `verify_done.txt` (create it empty if missing). It lists the paper
   folders already verified, one per line.

2. Pick the next **20** paper folders that have been audited but NOT yet
   verified: folders listed in `audit_done.txt` and NOT in `verify_done.txt`
   (each must contain `audit.md` + `findings.json`).

3. **First pass — Sonnet.** Spawn **one subagent per paper, all 20 in a single
   message** (parallel), each on **Sonnet 4.6**. Give each subagent:
   - its paper folder path,
   - "Verify this audit by following `auditowl/verifier_prompt.md` exactly.
     The original audit is `audit.md` + `findings.json`; the cloned repo is under
     `code/<owner>__<repo>/` (a finding's `file` is relative to that repo root).
     Re-check every finding adversarially — assume each is wrong until the
     evidence at the cited spot proves it right.
     **Stay cheap:** budget ~3–5 tool calls per finding; open the cited file at
     the cited lines only; do NOT inventory the repo or grep repo-wide unless the
     finding's claim is that something is *absent*; do not re-read files.
     Decide every verdict first, then write `findings_verified.json` **once at the
     very end** (the original findings with a `verdict` / `reason` / `changed`
     field added to each). Stay read-only on `code/`, `_audit_code/`, `audit.md`,
     and `findings.json`; write only `findings_verified.json` and
     `_verifier_code/`."

4. **Second pass — Opus escalation (judgment only).** After the Sonnet pass, for
   each paper read `findings_verified.json` and collect every finding whose
   `verdict` is `reject`, `lowered`, or `cannot-verify`. For each paper that has
   at least one such finding, spawn **one Opus 4.8 subagent (1M context)** and
   give it ONLY those findings to re-check, with the same cited-location-only
   discipline from `auditowl/verifier_prompt.md`. It confirms or corrects each
   escalated verdict and rewrites only those entries in `findings_verified.json`
   (leave `keep` findings untouched). Papers where the Sonnet pass marked every
   finding `keep` skip this pass entirely — no Opus cost.

5. When a paper is fully done (Sonnet pass + any Opus escalation), append its
   folder to `verify_done.txt`.
