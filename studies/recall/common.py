#!/usr/bin/env python3
"""Shared helpers for the recall experiment. Everything writes ONLY inside this folder."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root
AUDITS = ROOT / "audits"

# papers.json configures the pilot arm only. Load it lazily so the shared
# helpers below (dump_json / load_json / wilson_ci) can be imported by the
# other arms, which have their own paper lists and do not ship this file.
_CONFIG = None


def config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        path = HERE / "papers.json"
        if not path.exists():
            raise SystemExit(f"{path.name} not found -- this arm's paper list is "
                             "not part of the release.")
        _CONFIG = json.loads(path.read_text())
    return _CONFIG


def __getattr__(name):
    """Resolve CONFIG / PAPERS on first access (PEP 562).

    Keeps `from common import CONFIG, PAPERS` working for the pilot scripts,
    while letting the other arms import the shared helpers without needing
    papers.json present.
    """
    if name == "CONFIG":
        return config()
    if name == "PAPERS":
        return config()["papers"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
CATEGORIES = ("missing", "difference", "bug", "methodology")

# files an audit run needs as inputs (mirrors studies/reliability/build_sandboxes.py)
INPUT_FILES = ["paper.pdf", "paper_text.txt", "metadata.txt", "code_links.txt"]

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
               ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}


def assert_inside_here(path: Path) -> Path:
    """Guard: refuse to write anywhere outside studies/recall/."""
    p = Path(path).resolve()
    if HERE not in p.parents and p != HERE:
        raise SystemExit(f"SANDBOX VIOLATION: refusing to write outside {HERE}: {p}")
    return p


def paper_src(paper: dict) -> Path:
    d = AUDITS / paper["dir"]
    if not d.is_dir():
        raise SystemExit(f"missing source audit folder: {d}")
    return d


def _ignore(_dir, names):
    return [n for n in names if n in IGNORE_DIRS]


def copy_tree(src: Path, dst: Path, read_only: bool = False) -> None:
    assert_inside_here(dst)
    if dst.exists():
        return
    shutil.copytree(src, dst, ignore=_ignore, symlinks=False,
                    ignore_dangling_symlinks=True)
    if read_only:
        for dirpath, _dirnames, filenames in os.walk(dst):
            for f in filenames:
                fp = Path(dirpath) / f
                try:
                    fp.chmod(fp.stat().st_mode & ~0o222)
                except OSError:
                    pass


def make_writable(root: Path) -> None:
    assert_inside_here(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            try:
                (Path(dirpath) / f).chmod(0o644)
            except OSError:
                pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root: Path, rel_to: Path | None = None) -> dict[str, str]:
    """{relpath: sha256} over all files under root (sorted, ignore dirs skipped)."""
    rel_to = rel_to or root
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for f in sorted(filenames):
            p = Path(dirpath) / f
            out[str(p.relative_to(rel_to))] = sha256_file(p)
    return out


def norm_file(f: str | None) -> str:
    """Mirror of studies/reliability/analyze.py:norm_file — strip the code/<owner>__<repo>/ prefix."""
    if not f:
        return ""
    f = str(f).strip().replace("\\", "/")
    parts = f.split("/")
    if parts and parts[0] == "code":
        parts = parts[1:]
        if parts and "__" in parts[0]:
            parts = parts[1:]
    return "/".join(parts)


def norm_seed_file(f: str | None) -> str:
    """Seed/suspect paths are relative to code/ — prepend it so norm_file strips
    both the code/ segment and a leading <owner>__<repo>/ clone dir, giving the
    same normal form as finding paths."""
    return norm_file("code/" + str(f)) if f else ""


def lines_overlap(als, ale, bls, ble, window: int = 10):
    if als is None or bls is None:
        return None
    ale = ale if ale is not None else als
    ble = ble if ble is not None else bls
    return not (ale + window < bls or ble + window < als)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def dump_json(obj, path: Path) -> None:
    assert_inside_here(path)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")
