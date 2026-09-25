#!/usr/bin/env python3
"""Cross-model re-verification of the 610 findings via the OpenAI Codex CLI.

Stage 1: for each of the 87 audited papers, run `codex exec` (default codex
model, high reasoning effort) with the repo's auditowl/verifier_prompt.md,
blinded to the existing Claude verdicts, writing
`audits/<paper>/findings_verified_codex_stage1.json`.

Stage 2: escalation, mirroring auditowl/launch_verify.md — findings whose stage-1
verdict is reject/lowered/cannot-verify are re-checked in a fresh session on a
stronger general GPT model; the driver merges the corrections into the final
`audits/<paper>/findings_verified_codex.json` (per-finding `escalated` flag +
top-level `verifier_meta`).

Never touches any pre-existing file; check_manifest.py runs after every
session. Resume via codex_verify_done.txt (lines: "stage1 <paper>" /
"stage2 <paper>").
"""
import argparse
import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
def _find_root(start: Path) -> Path:
    """Repo root = the ancestor holding `audits/`. Found by content rather than a
    fixed number of `.parent` hops, so this survives the file moving depth."""
    for candidate in (start, *start.parents):
        if (candidate / "audits").is_dir():
            return candidate
    raise SystemExit(f"no `audits/` directory found above {start}")


ROOT = _find_root(HERE)
AUDITS = ROOT / "audits"
LOGS = HERE / "logs"
DONE = HERE / "codex_verify_done.txt"
VERIFIER_PROMPT = ROOT / "auditowl/verifier_prompt.md"

STAGE1_OUT = "findings_verified_codex_stage1.json"
FINAL_OUT = "findings_verified_codex.json"
ESCALATION_OUT = "findings_verified_codex_escalation.json"
SCRATCH = "_verifier_code_codex"

VALID_VERDICTS = {"keep", "lowered", "reject"}

BLINDING_HEADER = """\
# INDEPENDENT RE-VERIFICATION (cross-model run)

You are working inside a single paper's audit folder (your current directory).

**BLINDING — read this first.** A previous verifier's verdicts already exist
in this folder. You must NOT open, read, grep, or list the contents of
`findings_verified.json`, `_verifier_code/`, or any other file whose name
contains `verified` (except the output file you yourself write). Your verdicts
must be fully independent of the previous verifier's.

Follow the protocol below exactly. Where it mentions `audits/<paper>/`, that
is your current directory.

---

"""


def build_stage1_prompt() -> str:
    protocol = VERIFIER_PROMPT.read_text()
    protocol = protocol.replace("findings_verified.json", STAGE1_OUT)
    protocol = protocol.replace("_verifier_code/", SCRATCH + "/")
    protocol = protocol.replace("_verifier_code`", SCRATCH + "`")
    return BLINDING_HEADER + protocol


def build_stage2_prompt(flagged: list[dict]) -> str:
    findings_json = json.dumps(flagged, indent=2)
    return (
        BLINDING_HEADER
        + f"""\
# ESCALATION PASS (judgment only)

A first-pass verifier re-checked this paper's audit findings and flagged the
{len(flagged)} finding(s) below as `reject`, `lowered`, or `cannot-verify`.
Re-check ONLY these findings, with the same discipline as
`auditowl/verifier_prompt.md`: open the cited file at the cited lines
(finding `file` paths are relative to the repo root under
`code/<owner>__<repo>/`; `file: paper.pdf` means the paper, see
`paper_text.txt`), assume the finding is wrong until the evidence at the cited
spot proves it right, budget ~3-5 tool calls per finding. Repo-wide search is
allowed only when the claim is that something is *absent*. If a `check_script`
is cited, copy it into `{SCRATCH}/` and run it. Confirm or correct each
first-pass verdict — reverting an over-eager reject back to `keep` is a valid
and expected outcome.

The flagged findings, including the first-pass `verdict` / `reason` /
`changed`:

```json
{findings_json}
```

Write EXACTLY one new file, `{ESCALATION_OUT}`, once at the very end:
a JSON object `{{"findings": [...]}}` containing one entry per flagged finding
with fields `id`, `verdict` ("keep" | "lowered" | "reject"), `reason` (one
sentence; "cannot-verify" if uncheckable), `changed` (what changed vs the
first pass, or "nothing"), and — only if the verdict is `lowered` — the
reduced `severity` / `confidence` / `status`.

Everything else in this folder is READ-ONLY (you may also write scripts under
`{SCRATCH}/`). End with a one-line tally.
"""
    )


def all_papers() -> list[str]:
    return sorted(p.parent.name for p in AUDITS.glob("*/findings_verified.json"))


def done_set() -> set[str]:
    if not DONE.exists():
        return set()
    return {line.strip() for line in DONE.read_text().splitlines() if line.strip()}


def mark_done(tag: str) -> None:
    with DONE.open("a") as f:
        f.write(tag + "\n")


def load_finding_ids(paper: str) -> set[str]:
    data = json.loads((AUDITS / paper / "findings.json").read_text())
    return {f["id"] for f in data["findings"]}


def validate_verdict_fields(finding: dict) -> str | None:
    if finding.get("verdict") not in VALID_VERDICTS:
        return f"bad verdict {finding.get('verdict')!r}"
    if not isinstance(finding.get("reason"), str) or not finding["reason"].strip():
        return "missing reason"
    if "changed" not in finding:
        return "missing changed"
    return None


def validate_stage1(paper: str) -> str | None:
    """Return an error string, or None if the stage-1 output is valid."""
    out = AUDITS / paper / STAGE1_OUT
    if not out.is_file():
        return f"{STAGE1_OUT} not written"
    try:
        data = json.loads(out.read_text())
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"
    findings = data.get("findings")
    if not isinstance(findings, list):
        return "no findings list"
    got = {f.get("id") for f in findings}
    want = load_finding_ids(paper)
    if got != want:
        return f"id mismatch: missing {sorted(want - got)[:3]}, extra {sorted(got - want)[:3]}"
    for f in findings:
        err = validate_verdict_fields(f)
        if err:
            return f"{f.get('id')}: {err}"
    return None


def check_manifest() -> None:
    r = subprocess.run(
        [sys.executable, str(HERE / "check_manifest.py")], capture_output=True, text=True
    )
    if r.returncode != 0:
        sys.exit(f"FATAL: protected files changed!\n{r.stdout}{r.stderr}")


def run_codex(paper: str, prompt: str, log_path: Path, model: str | None,
              effort: str, timeout: int) -> int:
    cmd = [
        "codex", "exec",
        "-C", str(AUDITS / paper),
        "--sandbox", "workspace-write",
        "--skip-git-repo-check",
        "--color", "never",
        "--json",
        "-c", f'model_reasoning_effort="{effort}"',
        "-o", str(log_path.with_suffix(".last.md")),
        "-",
    ]
    if model:
        cmd[2:2] = ["-m", model]
    with log_path.open("w") as log:
        try:
            r = subprocess.run(cmd, input=prompt, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout)
            return r.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n[driver] TIMEOUT after {timeout}s\n")
            return -1


def stage1_paper(paper: str, args) -> tuple[str, str]:
    prompt = build_stage1_prompt()
    for attempt in (1, 2):
        log = LOGS / f"{paper}.stage1{'' if attempt == 1 else '.retry'}.jsonl"
        rc = run_codex(paper, prompt, log, args.model, args.effort, args.timeout)
        check_manifest()
        err = validate_stage1(paper)
        if err is None:
            mark_done(f"stage1 {paper}")
            return paper, "ok"
        bad = AUDITS / paper / STAGE1_OUT
        if bad.is_file():
            bad.rename(bad.with_suffix(f".bad{attempt}.json"))
        if attempt == 1:
            prompt = build_stage1_prompt() + (
                f"\n\n---\nNOTE: a previous attempt produced invalid output ({err}). "
                f"Make sure {STAGE1_OUT} contains ALL original findings with their "
                f"original `id` fields unchanged, each with verdict/reason/changed."
            )
    return paper, f"FAILED after retry (rc={rc}): {err}"


def flagged_findings(stage1: dict) -> list[dict]:
    out = []
    for f in stage1["findings"]:
        if f["verdict"] in ("reject", "lowered") or "cannot-verify" in f["reason"].lower():
            out.append(f)
    return out


def write_final(paper: str, stage1: dict, corrections: dict[str, dict],
                stage1_model: str | None, stage2_model: str | None, effort: str) -> None:
    final = copy.deepcopy(stage1)
    for f in final["findings"]:
        corr = corrections.get(f["id"])
        f["escalated"] = corr is not None
        if corr:
            f["stage1_verdict"] = f["verdict"]
            f["stage1_reason"] = f["reason"]
            for k in ("verdict", "reason", "changed", "severity", "confidence", "status"):
                if k in corr:
                    f[k] = corr[k]
    final["verifier_meta"] = {
        "run": "cross-model codex",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model if corrections else None,
        "reasoning_effort": effort,
        "cli": codex_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_escalated": len(corrections),
    }
    (AUDITS / paper / FINAL_OUT).write_text(json.dumps(final, indent=2) + "\n")


def codex_version() -> str:
    try:
        return subprocess.run(["codex", "--version"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"


def stage2_paper(paper: str, args) -> tuple[str, str]:
    stage1_path = AUDITS / paper / STAGE1_OUT
    stage1 = json.loads(stage1_path.read_text())
    stage1_model = args.model
    flagged = flagged_findings(stage1)

    if not flagged:
        write_final(paper, stage1, {}, stage1_model, None, args.effort)
        mark_done(f"stage2 {paper}")
        return paper, "ok (nothing flagged)"

    esc_path = AUDITS / paper / ESCALATION_OUT
    prompt = build_stage2_prompt(flagged)
    err = "unknown"
    for attempt in (1, 2):
        log = LOGS / f"{paper}.stage2{'' if attempt == 1 else '.retry'}.jsonl"
        rc = run_codex(paper, prompt, log, args.stage2_model, args.effort, args.timeout)
        check_manifest()
        err = validate_escalation(esc_path, {f["id"] for f in flagged})
        if err is None:
            corrections = {f["id"]: f for f in json.loads(esc_path.read_text())["findings"]}
            write_final(paper, stage1, corrections, stage1_model,
                        args.stage2_model, args.effort)
            mark_done(f"stage2 {paper}")
            return paper, f"ok ({len(flagged)} escalated)"
        if esc_path.is_file():
            esc_path.rename(esc_path.with_suffix(f".bad{attempt}.json"))
        if attempt == 1:
            prompt = build_stage2_prompt(flagged) + (
                f"\n\n---\nNOTE: a previous attempt produced invalid output ({err})."
            )
    return paper, f"FAILED after retry (rc={rc}): {err}"


def validate_escalation(esc_path: Path, want_ids: set[str]) -> str | None:
    if not esc_path.is_file():
        return f"{esc_path.name} not written"
    try:
        data = json.loads(esc_path.read_text())
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"
    findings = data.get("findings")
    if not isinstance(findings, list):
        return "no findings list"
    got = {f.get("id") for f in findings}
    if got != want_ids:
        return f"id mismatch: missing {sorted(want_ids - got)[:3]}, extra {sorted(got - want_ids)[:3]}"
    for f in findings:
        err = validate_verdict_fields(f)
        if err:
            return f"{f.get('id')}: {err}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--papers", nargs="*", help="paper folder names (default: all pending)")
    ap.add_argument("--limit", type=int, help="cap number of papers this invocation")
    ap.add_argument("--jobs", type=int, default=3, help="parallel codex sessions")
    ap.add_argument("--model", default="gpt-5.5",
                    help="stage-1 model (gpt-5.5 mirrors the original weaker-first pass)")
    ap.add_argument("--stage2-model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="high", choices=("medium", "high", "xhigh"))
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per codex session")
    ap.add_argument("--dry-run", action="store_true", help="print prompt + paper list, run nothing")
    args = ap.parse_args()

    papers = args.papers or all_papers()
    unknown = [p for p in papers if not (AUDITS / p / "findings.json").is_file()]
    if unknown:
        sys.exit(f"unknown papers: {unknown}")
    done = done_set()
    if args.stage == 2:
        not_ready = [p for p in papers if f"stage1 {p}" not in done]
        if not_ready:
            sys.exit(f"stage 1 not done for: {not_ready}")
    papers = [p for p in papers if f"stage{args.stage} {p}" not in done]
    if args.limit:
        papers = papers[: args.limit]

    if args.dry_run:
        print(build_stage1_prompt() if args.stage == 1 else "(stage-2 prompts are per-paper)")
        print(f"\n--- would run stage {args.stage} on {len(papers)} papers ---")
        for p in papers:
            print(p)
        return

    LOGS.mkdir(exist_ok=True)
    if not papers:
        print(f"stage {args.stage}: nothing pending")
        return
    missing_code = [p for p in papers if not (AUDITS / p / "code").is_dir()]
    if missing_code:
        print(f"WARNING: no code/ dir for {missing_code} (paper-only findings still verifiable)")

    worker = stage1_paper if args.stage == 1 else stage2_paper
    print(f"stage {args.stage}: {len(papers)} papers, {args.jobs} parallel")
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(worker, p, args): p for p in papers}
        for fut in as_completed(futs):
            paper, status = fut.result()
            print(f"  {paper}: {status}", flush=True)
            if status.startswith("FAILED"):
                failures.append(paper)
    if failures:
        sys.exit(f"{len(failures)} paper(s) failed: {failures}")
    print("all done; run check_manifest.py + compare_verdicts.py")


if __name__ == "__main__":
    main()
