#!/usr/bin/env python3
"""Build audit folders for papers drawn in list.csv.

For each paper this script:
  1. downloads the paper PDF (paper.pdf)
  2. classifies it as a *theory* paper (no experiments / NeurIPS checklist Q5 = NA)
     or an *empirical* paper
  3. finds AUTHOR / PRIMARY code repositories only -- baseline and dependency
     links (e.g. notears, flask, huggingface/pytorch-image-models) are skipped
  4. git-clones the author repo(s) into code/<owner>__<repo>/

Folder layout:
  audits_new/<num>/                   empirical papers
  audits_new/theory/<num>/            theory papers (no code expected)

The script walks list.csv in order and keeps going until it has collected
--nontheory empirical papers (theory papers are set aside, not counted).

Author-code detection uses four independent signals, requiring at least one:
  * an author-code cue phrase near the URL ("our code is available at",
    "we open-source", "publicly accessible at", ...)
  * repo-name / paper-title token or acronym overlap
  * repo-name / extracted-method-name overlap
  * a DOI/Zenodo reference adjacent to the link
...and rejects links under well-known library orgs, or vetoes a repo whose only
signal is a baseline-introduction cue ("based on the X package", "its
implementation is available at") with no strong author signal alongside it.

Usage:
    pip install requests pymupdf beautifulsoup4
    python fetch_paper_inputs.py                 # collect 20 empirical papers
    python fetch_paper_inputs.py --nontheory 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "neurips-audit/1.0"}
CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")
URL_RE = re.compile(r"https?://[^\s)>\]}\"',]+", re.IGNORECASE)

# Owners that host libraries/frameworks/models, not a paper's own code.
KNOWN_ORGS = {
    "huggingface", "pytorch", "tensorflow", "google", "google-research",
    "googleresearch", "facebookresearch", "facebook", "microsoft", "openai",
    "nvidia", "deepmind", "google-deepmind", "salesforce", "pallets",
    "scikit-learn", "scipy", "numpy", "pandas-dev", "black-forest-labs",
    "stability-ai", "open-mmlab", "rwightman", "pytorch-labs", "allenai",
}

# Phrases that mark a link as the authors' own released code.
AUTHOR_CUES = [
    r"\bour (?:code|implementation|source ?code|repo(?:sitory)?|codebase)\b",
    r"\bwe (?:open[- ]?source|release|publicly release)\b",
    r"\b(?:source )?codes?\b[^.]{0,40}\b(?:available|accessible|released|public|"
    r"coming soon|published|can be found|found (?:at|on)|on github)\b",
    r"\b(?:code|implementation)s? (?:is|are|will be|can be found)\b[^.]{0,30}"
    r"\b(?:available|accessible|released|public|github|found)\b",
    r"\bpublicly (?:available|accessible)\b",
    r"\breproducing\b[^.]{0,60}\b(?:published|available)\b",
    r"\bproject (?:page|website|site)\b",
]
AUTHOR_CUE_RE = re.compile("|".join(AUTHOR_CUES), re.IGNORECASE)

# Phrases that mark a link as a baseline / dependency / backbone (suppress it).
BASELINE_CUE_RE = re.compile(
    r"\b(?:based on|package|its (?:python )?implementation|baseline|backbone|"
    r"we (?:use|adopt|employ|build upon|follow|borrow)|borrowed from|"
    r"pre-?trained|foundation model|taken from|provided by|toolbox|toolkit|"
    r"library)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "and", "for", "with", "via", "your", "own", "all", "once", "from",
    "into", "using", "this", "that", "data", "model", "models", "learning",
    "general", "based", "method", "methods", "neural", "deep", "network",
    "networks", "improving", "understanding", "role", "high", "low",
}


def slugify(title: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return s[:maxlen] or "untitled"


def get_pdf_url(abstract_url: str) -> str | None:
    html = requests.get(abstract_url, headers=HEADERS, timeout=60).text
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    return meta["content"] if meta and meta.get("content") else None


def download(url: str, dest: str) -> None:
    with requests.get(url, headers=HEADERS, timeout=180, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


def read_pdf_text(pdf_path: str) -> tuple[str, list[str], str]:
    """Return (collapsed text, link-annotation URIs, newline-preserving text).

    The collapsed text is one long line (whitespace squashed) for URL mining
    and substring matching. The newline-preserving text keeps line breaks so it
    can be written as a greppable sidecar with stable line anchors.
    """
    doc = fitz.open(pdf_path)
    pages, annots = [], []
    for page in doc:
        pages.append(page.get_text())
        for link in page.get_links():
            if link.get("uri"):
                annots.append(link["uri"])
    doc.close()
    raw = "\n".join(pages)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")  # normalize line endings
    raw = re.sub(r"-\n", "", raw)             # de-hyphenate line breaks
    # Drop C0/C1 control glyphs (NUL etc.) that math/symbol fonts emit; they
    # make the text register as binary and break plain grep. Keep \t and \n.
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw)
    collapsed = re.sub(r"\s+", " ", raw)      # one long line for URL mining
    readable = re.sub(r"[ \t]+", " ", raw)    # squash spaces, keep newlines
    readable = re.sub(r"\n{3,}", "\n\n", readable).strip()
    return collapsed, annots, readable


# Matches a repo path with optional scheme and whitespace around the slashes,
# so line-wrapped/space-broken URLs in PDFs ("github.com/ owner/repo") are caught.
CODE_PATH_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org)"
    r"\s*/\s*([A-Za-z0-9_.-]+)\s*/\s*([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
NON_REPO_OWNERS = {"orgs", "about", "features", "sponsors", "settings", "search"}


def normalize_repo(raw: str) -> str | None:
    m = CODE_PATH_RE.search(raw)
    if not m:
        return None
    host, owner, repo = m.group(1).lower(), m.group(2), m.group(3)
    repo = repo.rstrip(".,;:)]}>\"'")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not repo or owner.lower() in NON_REPO_OWNERS:
        return None
    return f"https://{host}/{owner}/{repo}"


def tokens(text: str) -> set[str]:
    # split on non-alnum and camelCase boundaries
    parts = re.split(r"[^A-Za-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text))
    return {p.lower() for p in parts if len(p) >= 3 and p.lower() not in STOPWORDS}


def acronym(title: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", title)
    return "".join(w[0] for w in words if w.lower() not in STOPWORDS).lower()


def title_match(repo: str, title: str) -> bool:
    # Match on the repo NAME only -- owners are often generic org/group names
    # ("Graph-Machine-Learning-Group", "TorchSpatiotemporal") that collide with
    # common title words and pull in dependencies/prior work.
    name = repo.rstrip("/").split("/")[-1]
    if tokens(name) & tokens(title):
        return True
    name_flat = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    title_flat = re.sub(r"[^A-Za-z0-9]", "", title).lower()
    # repo name appears verbatim in the title (e.g. "bayesqp" in "BayeSQP: ...")
    if len(name_flat) >= 4 and name_flat in title_flat:
        return True
    acr = acronym(title)
    return len(name_flat) >= 3 and (name_flat in acr or acr in name_flat)


def extract_methods(text: str) -> set[str]:
    """Flattened names the paper gives its own method (\"...named DCAReasoner\")."""
    methods = set()
    for m in re.finditer(
        r"\b(?:named|dubbed|call(?:ed)?|termed|coined|denoted(?: as)?)\s+"
        r"\(?([A-Z][A-Za-z0-9-]{2,})", text):
        methods.add(re.sub(r"[^a-z0-9]", "", m.group(1).lower()))
    return methods


def find_occurrences(text: str, repo: str) -> list[int]:
    """Start positions of a repo path in text, tolerant of spaces around slashes."""
    host, owner, name = re.match(r"https://([^/]+)/([^/]+)/(.+)", repo).groups()
    pat = re.escape(host) + r"\s*/\s*" + re.escape(owner) + r"\s*/\s*" + re.escape(name)
    return [m.start() for m in re.finditer(pat, text, re.IGNORECASE)]


def classify_repos(text: str, annots: list[str], title: str) -> tuple[list[dict], list[dict]]:
    """Return (author_repos, other_repos) as lists of {repo, reason}."""
    # Candidate repos: bare/space-broken paths in text + link annotations.
    candidates: set[str] = set()
    for m in CODE_PATH_RE.finditer(text):
        repo = normalize_repo(m.group(0))
        if repo:
            candidates.add(repo)
    for uri in annots:
        repo = normalize_repo(uri)
        if repo:
            candidates.add(repo)

    methods = extract_methods(text)
    author, other = [], []

    for repo in sorted(candidates):
        owner, name = repo.split("/")[-2:]
        if owner.lower() in KNOWN_ORGS:
            other.append({"repo": repo, "reason": f"known-org:{owner.lower()}"})
            continue

        # Inspect every occurrence: cue phrase before, DOI/Zenodo just after.
        has_author_cue = has_baseline_cue = doi_adjacent = False
        for pos in find_occurrences(text, repo):
            before = text[max(0, pos - 150):pos]
            after = text[pos:pos + 120]
            if AUTHOR_CUE_RE.search(before):
                has_author_cue = True
            if BASELINE_CUE_RE.search(before):
                has_baseline_cue = True
            if re.search(r"\b(doi|zenodo)\b", after, re.IGNORECASE):
                doi_adjacent = True

        matches_title = title_match(repo, title)
        matches_method = re.sub(r"[^a-z0-9]", "", name.lower()) in methods

        strong = matches_title or matches_method or doi_adjacent
        reason = None
        if has_author_cue:
            reason = "author-cue"
        elif matches_title:
            reason = "title-match"
        elif matches_method:
            reason = "method-name"
        elif doi_adjacent:
            reason = "doi-archived"

        # A baseline cue vetoes only when no strong author signal is present.
        if reason and not (has_baseline_cue and not (strong or has_author_cue)):
            author.append({"repo": repo, "reason": reason})
        else:
            why = "baseline-cue" if has_baseline_cue else "no-author-signal"
            other.append({"repo": repo, "reason": why})
    return author, other


def is_theory(text: str) -> bool:
    """True if the NeurIPS checklist marks the paper as having no experiments.

    Only the author's own *justification* is inspected (the span between
    "Justification:" and "Guidelines:"). The boilerplate Guidelines text always
    contains "does not include experiments requiring code", so reading past it
    would wrongly flag empirical papers that merely answered NA (e.g. authors
    who say "code will be released after acceptance").
    """
    signals = (
        "does not include experiment", "no experiment",
        "does not involve any experiment", "theory work", "theoretical work",
        "purely theoretical", "this is a theory",
    )
    for q in (r"open access to (?:the )?data and code",
              r"experimental result reproducibility"):
        m = re.search(
            q + r".*?answer:\s*\[([^\]]*)\]\s*justification:\s*(.*?)\s*guidelines:",
            text, re.I)
        if m:
            ans = m.group(1).strip().lower()
            just = m.group(2).lower()
            if ans in ("na", "n/a") and any(s in just for s in signals):
                return True
    return False


def git_clone(url: str, dest: str) -> dict:
    """Shallow-clone url into dest, pulling submodules and LFS content.

    Returns a status dict {ok, status, error}. A plain --depth 1 clone leaves
    submodules empty and LFS files as pointer stubs, which makes released code
    look "missing" to the audit; we recurse submodules and best-effort `git lfs
    pull` so the clone reflects what the authors actually shipped.
    """
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--recurse-submodules",
             "--shallow-submodules", url, dest],
            check=True, capture_output=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("      clone FAILED: timed out after 600s", file=sys.stderr)
        return {"ok": False, "status": "timeout", "error": "600s timeout"}
    except subprocess.CalledProcessError as e:
        err = getattr(e, "stderr", b"")
        msg = err.decode(errors="replace") if isinstance(err, bytes) else str(e)
        last = msg.strip().splitlines()[-1] if msg.strip() else str(e)
        print(f"      clone FAILED: {last}", file=sys.stderr)
        return {"ok": False, "status": "failed", "error": last}
    # Best-effort LFS materialisation (no-op if git-lfs absent or no LFS files).
    try:
        subprocess.run(["git", "lfs", "pull"], cwd=dest,
                       capture_output=True, timeout=600)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return {"ok": True, "status": "ok", "error": ""}


def _is_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 300:
            return False
        with open(path, "rb") as fh:
            return fh.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def scan_clone_health(repo_dir: str) -> dict:
    """Count LFS pointer stubs and empty/absent submodules under repo_dir."""
    lfs_stubs, empty_subs, size_bytes = [], [], 0
    for dp, dns, fs in os.walk(repo_dir):
        if os.sep + ".git" + os.sep in dp + os.sep or dp.endswith(os.sep + ".git"):
            continue
        for f in fs:
            p = os.path.join(dp, f)
            try:
                size_bytes += os.path.getsize(p)
            except OSError:
                pass
            if _is_lfs_pointer(p):
                lfs_stubs.append(os.path.relpath(p, repo_dir))
    for gm in glob.glob(os.path.join(repo_dir, "**", ".gitmodules"), recursive=True):
        base = os.path.dirname(gm)
        for line in open(gm, errors="replace"):
            line = line.strip()
            if line.startswith("path"):
                sub = line.split("=", 1)[1].strip()
                d = os.path.join(base, sub)
                if not (os.path.isdir(d) and os.listdir(d)):
                    empty_subs.append(os.path.relpath(d, repo_dir))
    return {
        "size_mb": round(size_bytes / 1e6, 1),
        "lfs_pointer_stubs": len(lfs_stubs),
        "lfs_examples": lfs_stubs[:5],
        "empty_submodules": empty_subs,
    }


# ---- NeurIPS supplemental .zip ---------------------------------------------
# Many papers ship their only runnable code in the conference *supplemental*
# zip, never on a git host. The URL-mining pass above never sees it, so the
# audit would wrongly read "no code". We materialise the supplement's CODE here
# so the on-disk view matches what the authors actually released. A supplement
# that is only a PDF / dataset (no source files) is deliberately NOT unpacked
# into code/ — that would fake "source present" for a paper that released no code.
SUPP_JUNK_DIRS = {".venv", "venv", "env", "site-packages", "__pycache__",
                  "node_modules", "__MACOSX", ".git", ".ipynb_checkpoints",
                  ".mypy_cache", ".pytest_cache"}
SUPP_CODE_EXT = {".py", ".ipynb", ".m", ".cpp", ".cc", ".cxx", ".c", ".h",
                 ".hpp", ".cu", ".cuh", ".java", ".js", ".ts", ".jsx", ".tsx",
                 ".go", ".rs", ".sh", ".bash", ".r", ".jl", ".scala", ".lua",
                 ".f90", ".f", ".mlx", ".pyx"}


def supplement_url(abstract_url: str) -> str | None:
    """Map an abstract page URL to its Supplemental-Conference.zip on the
    proceedings site (same hash as the abstract)."""
    m = re.search(r"hash/([0-9a-f]+)-Abstract", abstract_url or "")
    if not m:
        return None
    return ("https://papers.nips.cc/paper_files/paper/2025/file/"
            f"{m.group(1)}-Supplemental-Conference.zip")


def _supp_is_junk(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return any(p in SUPP_JUNK_DIRS or p.endswith((".dist-info", ".egg-info"))
               for p in parts)


def fetch_supplement(abstract_url: str, code_dir: str) -> dict:
    """Download the supplemental zip; if it carries source code, unpack the code
    (minus bundled virtualenvs / caches) into ``code_dir/supplement``.

    Returns a status dict for fetch_manifest.json. Only creates code/supplement
    when the zip actually contains code files — so a PDF-only or dataset-only
    supplement leaves the paper correctly recorded as having no released source.
    """
    url = supplement_url(abstract_url)
    if not url:
        return {"status": "no-hash"}
    dest = os.path.join(code_dir, "supplement")
    if os.path.isdir(dest) and os.listdir(dest):
        return {"status": "cached", "url": url}
    try:
        r = requests.get(url, headers=HEADERS, timeout=300)
        if r.status_code == 404:
            return {"status": "absent-404", "url": url}
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
    except Exception as e:  # noqa: BLE001 - record any network/zip failure
        return {"status": f"error:{str(e)[:80]}", "url": url}

    members, n_code = [], 0
    for info in zf.infolist():
        if info.is_dir() or _supp_is_junk(info.filename):
            continue
        members.append(info)
        if os.path.splitext(info.filename)[1].lower() in SUPP_CODE_EXT:
            n_code += 1
    if n_code == 0:
        return {"status": "no-code", "url": url, "n_files": len(members),
                "size_mb": round(len(r.content) / 1e6, 1)}

    dest_abs = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    written = 0
    for info in members:
        rel = info.filename.replace("\\", "/")
        out_path = os.path.normpath(os.path.join(dest, rel))
        if not os.path.abspath(out_path).startswith(dest_abs + os.sep):
            continue  # zip-slip guard
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            with zf.open(info) as src, open(out_path, "wb") as out:
                shutil.copyfileobj(src, out)
            written += 1
        except (OSError, zipfile.BadZipFile):
            continue
    print(f"      supplement: unpacked {n_code} code file(s) into code/supplement",
          file=sys.stderr)
    return {"status": "ok", "url": url, "n_code_files": n_code,
            "n_files_written": written, "size_mb": round(len(r.content) / 1e6, 1)}


def process(num: int, paper: dict, outdir: str) -> dict:
    title = paper["title"]
    print(f"\n[{num}] {title}")

    # Cache PDFs by paper number so re-runs don't re-download.
    cache_dir = os.path.join(outdir, ".pdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    pdf_path = os.path.join(cache_dir, f"{num}.pdf")

    if not os.path.exists(pdf_path):
        try:
            pdf_url = get_pdf_url(paper["url"])
            if not pdf_url:
                raise RuntimeError("no citation_pdf_url meta tag")
            print(f"  downloading {pdf_url}")
            download(pdf_url, pdf_path)
        except Exception as e:
            print(f"  PDF download failed: {e}", file=sys.stderr)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return {"num": num, "title": title, "ok": False}
    else:
        print("  (using cached PDF)")

    text, annots, readable = read_pdf_text(pdf_path)
    theory = is_theory(text)
    author, other = classify_repos(text, annots, title)

    # Final folder location.
    base = os.path.join(outdir, "theory") if theory else outdir
    folder = os.path.join(base, str(num))   # folders are named by paper number only
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(pdf_path, os.path.join(folder, "paper.pdf"))

    # Faithful plain-text sidecar so audit agents can grep the paper for
    # reported numbers and cite line anchors, without re-extracting the PDF
    # themselves. The PDF stays the source of truth for figures/tables/quotes.
    with open(os.path.join(folder, "paper_text.txt"), "w", encoding="utf-8") as f:
        f.write(readable)

    with open(os.path.join(folder, "metadata.txt"), "w", encoding="utf-8") as f:
        f.write(f"paper_number: {num}\n")
        f.write(f"title: {title}\n")
        f.write(f"authors: {paper['authors']}\n")
        f.write(f"abstract_url: {paper['url']}\n")
        f.write(f"category: {'theory' if theory else 'empirical'}\n")

    with open(os.path.join(folder, "code_links.txt"), "w", encoding="utf-8") as f:
        f.write("# AUTHOR / PRIMARY repositories (cloned):\n")
        for r in author:
            f.write(f"{r['repo']}\t[{r['reason']}]\n")
        f.write("\n# Other code links found (NOT cloned -- baseline/dependency):\n")
        for r in other:
            f.write(f"{r['repo']}\t[{r['reason']}]\n")

    cat = "THEORY" if theory else "empirical"
    print(f"  category: {cat} | author repos: {[r['repo'] for r in author]}"
          f" | skipped: {[r['repo'] for r in other]}")

    cloned = []
    code_dir = os.path.join(folder, "code")
    author_dests = {f"{r['repo'].split('/')[-2]}__{r['repo'].split('/')[-1]}": r["repo"]
                    for r in author}
    # Prune any clone that is no longer classified as author code (idempotent
    # reruns). The unpacked supplement (code/supplement) is kept, not a clone.
    if os.path.isdir(code_dir):
        for existing in os.listdir(code_dir):
            if existing not in author_dests and existing != "supplement":
                print(f"    pruning stale clone {existing}")
                path = os.path.join(code_dir, existing)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
    manifest = {}
    if author:
        os.makedirs(code_dir, exist_ok=True)
        for dest_name, repo in author_dests.items():
            dest = os.path.join(code_dir, dest_name)
            if os.path.exists(dest):
                cloned.append(repo)
                manifest[dest_name] = {"repo": repo, "clone_status": "cached",
                                       **scan_clone_health(dest)}
                continue
            print(f"    cloning {repo}")
            res = git_clone(repo, dest)
            entry = {"repo": repo, "clone_status": res["status"],
                     "clone_error": res["error"]}
            if res["ok"]:
                cloned.append(repo)
                entry.update(scan_clone_health(dest))
                if entry["lfs_pointer_stubs"] or entry["empty_submodules"]:
                    print(f"      incomplete: {entry['lfs_pointer_stubs']} LFS "
                          f"stub(s), {len(entry['empty_submodules'])} empty "
                          f"submodule(s)", file=sys.stderr)
            manifest[dest_name] = entry
    elif os.path.isdir(code_dir) and not os.listdir(code_dir):
        os.rmdir(code_dir)

    # Supplemental-zip code: fetch + unpack into code/supplement when the zip
    # carries source (closes the blind spot where code lives only in the
    # supplement, never a git host). PDF/dataset-only supplements are skipped.
    supp = fetch_supplement(paper["url"], code_dir)
    manifest["__supplement__"] = supp
    if supp.get("status") in ("ok", "cached"):
        if supp["url"] not in cloned:
            cloned.append(supp["url"])
        with open(os.path.join(folder, "code_links.txt"), "a") as f:
            f.write(f"\n# Supplemental-Conference.zip (code unpacked into code/supplement):\n")
            f.write(f"{supp['url']}\t[supplement-code]\n")

    # Record fetch health so the audit can tell "author shipped nothing" apart
    # from "our clone was incomplete" (timeout / LFS stubs / empty submodules).
    with open(os.path.join(folder, "fetch_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return {"num": num, "title": title, "ok": True, "theory": theory,
            "author": author, "other": other, "cloned": cloned}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="neurips_2025_main_track.csv")
    parser.add_argument("--list", default="list.csv")
    parser.add_argument("--outdir", default="audits_new")
    parser.add_argument("--nontheory", type=int, default=20,
                        help="collect this many empirical (non-theory) papers")
    parser.add_argument("--max", type=int, default=200,
                        help="safety cap on how many list rows to walk")
    args = parser.parse_args()

    papers = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    numbers = [int(r["paper_number"]) for r in csv.DictReader(open(args.list, encoding="utf-8"))]

    os.makedirs(args.outdir, exist_ok=True)
    results = []
    n_empirical = 0

    for num in numbers[: args.max]:
        if n_empirical >= args.nontheory:
            break
        res = process(num, papers[num - 1], args.outdir)
        results.append(res)
        if res.get("ok") and not res.get("theory"):
            n_empirical += 1

    print("\n" + "=" * 74)
    print("SUMMARY")
    theory = [r for r in results if r.get("theory")]
    empirical = [r for r in results if r.get("ok") and not r.get("theory")]
    no_code = [r for r in empirical if not r["cloned"]]
    for r in empirical:
        status = f"{len(r['cloned'])} repo(s)" if r["cloned"] else "NO CODE"
        print(f"  [{r['num']:>4}] {status:11} {r['title'][:54]}")
    print("  --- theory (set aside) ---")
    for r in theory:
        print(f"  [{r['num']:>4}] {'THEORY':11} {r['title'][:54]}")
    print(f"\nEmpirical papers: {len(empirical)} (target {args.nontheory})")
    print(f"  with author code: {len(empirical) - len(no_code)}")
    print(f"  no author code:   {len(no_code)}")
    print(f"Theory papers set aside: {len(theory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
