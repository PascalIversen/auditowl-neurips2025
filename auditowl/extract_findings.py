#!/usr/bin/env python3
"""Extract structured findings from audit.md into findings.json.

Reads an audit.md (Appendix B / findings-schema.md format), pulls every
fenced ```yaml finding``` block, validates it against the schema, and emits
a flat findings.json on stdout (or to --out).

The extractor:
- Prepends the audit's folder name to each `id`, so ids are globally unique
  across the audits/ tree (used for cross-audit dedup).
- Always emits JSON (invalid blocks are dropped from `findings`, valid ones kept);
  prints all validation errors to stderr and exits 1 if `--strict` was passed
  or if zero findings survived, exits 0 otherwise.

Usage:
    python extract_findings.py path/to/audit.md
    python extract_findings.py path/to/audit.md --out path/to/findings.json
    python extract_findings.py path/to/audit.md --strict   # exit 1 on any warning
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "extract_findings.py requires PyYAML. Install with `pip install pyyaml`.\n"
    )
    sys.exit(2)


REQUIRED_FIELDS = (
    "id",
    "category",
    "topic",
    "title",
    "severity",
    "confidence",
    "status",
    "file",
    "quote",
    "claim",
    "concern",
    "resolution",
)
OPTIONAL_FIELDS = (
    "line_start",
    "line_end",
    "cross_refs",
    "check_script",
    "paper_ref",
    "validator_pass",
    "csv_row",
    "url_retrieved_at",
)
ENUM_CATEGORY = {"missing", "bug", "difference", "methodology"}
ENUM_SEVERITY = {"high", "medium", "low"}
ENUM_CONFIDENCE = {"high", "medium", "low"}
ENUM_STATUS = {"finding", "question"}

FENCE_RE = re.compile(
    r"^```yaml\s+finding\s*\n(.*?)\n```",
    re.DOTALL | re.MULTILINE,
)


class ExtractionError(Exception):
    pass


def extract_blocks(text: str) -> list[tuple[int, str]]:
    """Return [(line_number, yaml_text)] for each ```yaml finding``` block."""
    blocks: list[tuple[int, str]] = []
    for match in FENCE_RE.finditer(text):
        line = text[: match.start()].count("\n") + 1
        blocks.append((line, match.group(1)))
    return blocks


def validate(finding: dict, audit_md_line: int) -> list[str]:
    errors: list[str] = []
    where = f"audit.md:{audit_md_line}"

    for field in REQUIRED_FIELDS:
        if field not in finding or finding[field] in (None, ""):
            errors.append(f"{where}: missing required field `{field}`")

    if finding.get("category") not in ENUM_CATEGORY:
        errors.append(
            f"{where}: category={finding.get('category')!r} not in {sorted(ENUM_CATEGORY)}"
        )
    if finding.get("severity") not in ENUM_SEVERITY:
        errors.append(
            f"{where}: severity={finding.get('severity')!r} not in {sorted(ENUM_SEVERITY)}"
        )
    if finding.get("confidence") not in ENUM_CONFIDENCE:
        errors.append(
            f"{where}: confidence={finding.get('confidence')!r} not in {sorted(ENUM_CONFIDENCE)}"
        )
    if finding.get("status") not in ENUM_STATUS:
        errors.append(
            f"{where}: status={finding.get('status')!r} not in {sorted(ENUM_STATUS)}"
        )

    ls, le = finding.get("line_start"), finding.get("line_end")
    if ls is not None and le is not None and ls > le:
        errors.append(f"{where}: line_start ({ls}) > line_end ({le})")

    file_str = str(finding.get("file", ""))
    is_paper = file_str.endswith(".pdf")
    is_url = file_str.startswith(("http://", "https://"))
    is_csv = file_str.startswith("out/") and file_str.endswith(".csv")
    is_code = not (is_paper or is_url or is_csv)

    if is_code and (ls is None or le is None):
        errors.append(
            f"{where}: code evidence (`file: {file_str}`) requires line_start and line_end"
        )
    if is_csv and finding.get("csv_row") is None:
        errors.append(
            f"{where}: csv evidence (`file: {file_str}`) requires `csv_row`"
        )
    if is_url and not finding.get("url_retrieved_at"):
        errors.append(
            f"{where}: url evidence (`file: {file_str}`) requires `url_retrieved_at`"
        )

    if finding.get("status") == "finding":
        vp = finding.get("validator_pass") or {}
        for key in ("quote_match", "control_flow", "condition_satisfiable"):
            if vp.get(key) is not True:
                errors.append(
                    f"{where}: status=finding requires validator_pass.{key}=true (got {vp.get(key)!r})"
                )

    if cross_refs := finding.get("cross_refs"):
        if not isinstance(cross_refs, list):
            errors.append(f"{where}: cross_refs must be a list")

    return errors


def normalize(
    raw: dict, audit_name: str, audit_md_line: int
) -> dict:
    finding = dict(raw)
    section_local_id = str(finding["id"]).strip()
    finding["id_local"] = section_local_id
    finding["id"] = f"{audit_name}/{section_local_id}"
    finding["category"] = str(finding["category"])
    finding["audit"] = audit_name
    finding["audit_md_line"] = audit_md_line
    # `tags` was a deprecated taxonomy field; drop it if an older audit.md still
    # carries one so it never propagates into findings.json.
    finding.pop("tags", None)
    finding.setdefault("cross_refs", [])
    finding.setdefault("line_start", None)
    finding.setdefault("line_end", None)
    return finding


def extract(audit_path: Path) -> tuple[list[dict], list[str]]:
    text = audit_path.read_text(encoding="utf-8")
    audit_name = audit_path.parent.name
    blocks = extract_blocks(text)

    findings: list[dict] = []
    errors: list[str] = []
    seen_ids: dict[str, int] = {}

    for line, body in blocks:
        try:
            raw = yaml.safe_load(body)
        except yaml.YAMLError as exc:
            errors.append(f"audit.md:{line}: YAML parse error: {exc}")
            continue
        if not isinstance(raw, dict):
            errors.append(f"audit.md:{line}: finding block must be a YAML mapping")
            continue

        block_errors = validate(raw, line)
        errors.extend(block_errors)
        if block_errors:
            continue

        finding = normalize(raw, audit_name, line)
        prev_line = seen_ids.get(finding["id_local"])
        if prev_line is not None:
            errors.append(
                f"audit.md:{line}: duplicate id `{finding['id_local']}` "
                f"(first seen at audit.md:{prev_line})"
            )
            continue
        seen_ids[finding["id_local"]] = line
        findings.append(finding)

    return findings, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audit_md", type=Path, help="Path to audit.md")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write findings.json here (default: stdout)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any block fails validation (default: 0 if any blocks succeed)",
    )
    args = parser.parse_args()

    if not args.audit_md.exists():
        sys.stderr.write(f"audit file not found: {args.audit_md}\n")
        return 2

    findings, errors = extract(args.audit_md)

    for err in errors:
        sys.stderr.write(f"[error] {err}\n")

    payload = {
        "schema_version": "1",
        "audit": args.audit_md.parent.name,
        "audit_md": str(args.audit_md),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "findings": findings,
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        sys.stderr.write(
            f"wrote {len(findings)} findings to {args.out} "
            f"({len(errors)} validation errors)\n"
        )
    else:
        sys.stdout.write(output)

    if errors and (args.strict or not findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
