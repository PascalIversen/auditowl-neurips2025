#!/usr/bin/env python3
"""Record which revision of each audited repository was actually audited.

The audits cite evidence as `file:line`, which only resolves against the exact
tree the auditor read. The clones under `audits/*/code/` are git-ignored and not
redistributed, so without this the citations cannot be re-resolved by a reader.
This reads the commit straight out of each local clone's `.git` and commits the
result, so the evidence trail stays checkable after the clones are gone.

Sources that are not git repositories -- Zenodo archives, OpenReview/NeurIPS
supplementary ZIPs -- have no commit; they are recorded with their origin and
`commit: null` rather than omitted, so the file accounts for every clone.

Run: python analysis/build_repo_revisions.py   (needs the clones present)
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "repo_revisions.json"


def git(repo: Path, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main():
    out, n_git, n_archive = {}, 0, 0
    for repo in sorted(ROOT.joinpath("audits").glob("*/code/*")):
        if not repo.is_dir():
            continue
        paper, name = repo.parts[-3], repo.name
        if (repo / ".git").exists() and (sha := git(repo, "rev-parse", "HEAD")):
            out.setdefault(paper, {})[name] = {
                "commit": sha,
                "committed_at": git(repo, "log", "-1", "--format=%cI"),
                "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                "remote": git(repo, "config", "--get", "remote.origin.url"),
            }
            n_git += 1
        else:
            # not a git checkout: supplementary ZIP, Zenodo archive, or a clone
            # whose .git was not retained
            out.setdefault(paper, {})[name] = {"commit": None, "source": "non-git artifact"}
            n_archive += 1

    if not n_git:
        raise SystemExit(
            f"no git checkouts found under {ROOT / 'audits'}/*/code/ -- refusing to "
            f"overwrite {OUT.name}. Run this in a tree that has the clones."
        )
    if OUT.exists():
        prev = json.loads(OUT.read_text())
        prev_pinned = sum(1 for r in prev.values() for v in r.values() if v.get("commit"))
        if n_git < prev_pinned:
            raise SystemExit(
                f"would reduce pinned revisions from {prev_pinned} to {n_git} -- refusing. "
                "Some clones are missing locally."
            )

    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} — {n_git} pinned to a commit, {n_archive} non-git artifacts, "
          f"{len(out)} papers")


if __name__ == "__main__":
    main()
