#!/usr/bin/env python3
"""Freeze SHA-256 hashes of the canonical audit outputs before the cross-model run.

Writes studies/cross_model/protect_manifest.json covering every protected file under
audits/*/. Run once before the Codex/GPT verification; check_manifest.py then
detects any accidental modification with a single fast pass, run automatically
after every session rather than relying on a manual `git diff` review.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def _find_root(start: Path) -> Path:
    """Repo root = the ancestor holding `audits/`. Found by content rather than a
    fixed number of `.parent` hops, so this survives the file moving depth."""
    for candidate in (start, *start.parents):
        if (candidate / "audits").is_dir():
            return candidate
    raise SystemExit(f"no `audits/` directory found above {start}")


ROOT = _find_root(Path(__file__).resolve().parent)
MANIFEST = Path(__file__).resolve().parent / "protect_manifest.json"

PROTECTED = [
    "audit.md",
    "findings.json",
    "findings_verified.json",
    "repo_provenance.json",
    "token_cost.json",
    "fetch_manifest.json",
    "code_links.txt",
    "metadata.txt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = {}
    for audit_dir in sorted((ROOT / "audits").iterdir()):
        if not audit_dir.is_dir():
            continue
        for name in PROTECTED:
            p = audit_dir / name
            if p.is_file():
                entries[str(p.relative_to(ROOT))] = sha256(p)
    MANIFEST.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_files": len(entries),
                "files": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(entries)} files")


if __name__ == "__main__":
    main()
