# VERIFIER 2.0

An LLM already audited this paper and wrote `audit.md` + `findings.json`.
**Re-check every finding.** Assume each one is wrong
until the evidence at the cited spot proves it right. Then emit what changed.

## Where things are (`audits/<paper>/`)

- `paper.pdf` — the paper (truth for paper quotes/tables/figures).
- `paper_text.txt` (plain-text conversion of the PDF)
- `code/<owner>__<repo>/` — the cloned repo. A finding's `file` is **relative to
  this repo root** (`file: src/train.py` → `code/<owner>__<repo>/src/train.py`).
  Findings with `file: paper.pdf` are about the paper.
- `audit.md`, `findings.json` — the original audit. **READ-ONLY.**
- `_audit_code/` — auditor scripts. **READ-ONLY** (copy to `_verifier_code/` to run).

You write only: `findings_verified.json` and scripts under `_verifier_code/`.

## The check (per finding)

Open the cited file at the cited lines and ask one question: **does the evidence
right there prove the claim?**

- File or lines missing, quote not at that spot, or the code says something
  different → **reject**.
- The bug's code path can't actually run, or its trigger condition can't happen
  → **reject**.
- A `check_script` is cited → copy it to `_verifier_code/`, run it, save output to
  `_verifier_code/out/`. Crashes or disagrees with the claim → **reject**.
- Can't verify (missing dep, GPU, paywall, dead link) → **reject** with reason
  `cannot-verify`. Never guess. Count these apart from real rejections.
- Everything checks out, but it's milder than claimed → **keep, lower the severity.**
- Everything checks out → **keep as-is.**

Compare the quote to the file after stripping common indentation on both sides;
whitespace-only differences are fine.

Don't add new findings. Verify what's there.

## Stay cheap (budget ~3–5 tool calls per finding)

- Open the cited file **at the cited lines only** (a targeted Read with
  offset/limit, or a grep for the quoted string). Never read a file end-to-end.
- Do **not** inventory the repo or run repo-wide greps **unless** the finding's
  claim is that something is *absent* (e.g. "no requirements file", "metric X not
  computed") — only then is a repo-wide search the actual check.
- Don't re-read a file you've already opened.
- Decide all verdicts first, then write `findings_verified.json` **once at the
  very end**. Never write it incrementally (a transient glitch must not trigger a
  rewrite).

## Output: `findings_verified.json`

Take the original `findings.json`, keep **every** finding (including rejected
ones — don't delete any), and add these fields to each:

```json
"verdict": "keep | lowered | reject",
"reason": "one sentence (use 'cannot-verify' if you couldn't check it)",
"changed": "what changed, or 'nothing'"
```

For `lowered`, also overwrite the top-level `severity` / `confidence` / `status`
with the reduced values so the JSON reflects reality.

That's it. End your reply with a one-line tally:
`N findings: X kept, Y lowered, Z rejected (of which W cannot-verify).`
