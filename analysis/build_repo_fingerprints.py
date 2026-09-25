#!/usr/bin/env python3
"""Snapshot the code-dependent booleans aggregate.py needs, WITHOUT publishing
any third-party code. The cloned `code/<owner>__<repo>/` trees are git-ignored
(we cannot redistribute the audited authors' repositories), so a reader who
clones only this repo cannot regenerate `inst_ok` / `run_ok` / `n_code_files`
for Figure 4 and the funnel. This writes just the three derived facts per
paper -- no file names, no file contents, no repo structure -- so those
figures are regeneratable from the public release. aggregate.py uses these
as a fallback only when a paper's `code/` directory isn't present locally.

Run: python analysis/build_repo_fingerprints.py
"""
import json
from pathlib import Path

from aggregate import repo_has_install, repo_has_runnable, repo_code_file_count

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"


def main():
    out = {}
    for d in sorted((ROOT / "audits").iterdir()):
        if not d.is_dir():
            continue
        repos = list((d / "code").glob("*/")) if (d / "code").exists() else []
        if not repos:
            continue
        out[d.name] = {
            "has_code": True,
            "inst_ok": any(repo_has_install(r) for r in repos),
            "run_ok": any(repo_has_runnable(r) for r in repos),
            "n_code_files": sum(repo_code_file_count(r) for r in repos),
        }
    path = DATA / "repo_fingerprints.json"

    # This snapshot exists precisely so the figures work WITHOUT the cloned
    # trees. Running here without them would therefore overwrite it with an
    # empty dict and break the very thing it is for -- silently, since finding
    # no repositories looks identical to there being none.
    if not out:
        raise SystemExit(
            f"found no cloned repositories under {ROOT / 'audits'}/*/code/ -- refusing "
            f"to overwrite {path.name}, which is what lets the figures regenerate "
            "without them. Re-run this only in a working tree that has the audited "
            "repositories cloned."
        )
    if path.exists():
        prev = json.loads(path.read_text())
        if len(out) < len(prev):
            raise SystemExit(
                f"would shrink {path.name} from {len(prev)} to {len(out)} papers -- "
                "refusing. Some repositories are missing locally; re-clone them or "
                "delete the file deliberately if the corpus really did shrink."
            )

    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {path} ({len(out)} papers)")


if __name__ == "__main__":
    main()
