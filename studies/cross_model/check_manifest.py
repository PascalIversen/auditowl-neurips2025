#!/usr/bin/env python3
"""Verify protect_manifest.json: no protected audit file changed or vanished.

Exit 0 = clean, exit 1 = violations (listed on stdout). Run by the driver after
every codex session and manually at milestones.
"""
import hashlib
import json
import sys
from pathlib import Path

def _find_root(start: Path) -> Path:
    """Repo root = the ancestor holding `audits/`. Located by content so this
    keeps working if the file moves to a different depth."""
    for candidate in (start, *start.parents):
        if (candidate / "audits").is_dir():
            return candidate
    raise SystemExit(f"no `audits/` directory found above {start}")


ROOT = _find_root(Path(__file__).resolve().parent)
MANIFEST = Path(__file__).resolve().parent / "protect_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    bad = []
    for rel, want in manifest["files"].items():
        p = ROOT / rel
        if not p.is_file():
            bad.append(f"MISSING  {rel}")
        elif sha256(p) != want:
            bad.append(f"MODIFIED {rel}")
    if bad:
        print("\n".join(bad))
        print(f"check_manifest: {len(bad)} violation(s) out of {manifest['n_files']} files")
        return 1
    print(f"check_manifest: clean ({manifest['n_files']} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
