#!/usr/bin/env python3
"""Shared helpers for `scicoqa_judge.py` and `judge_baselines_gpt55.py`:
the verbatim SciCoQA `discrepancy_evaluation_v2` judge prompt, and the
`codex exec -m gpt-5.5` calling mechanism used as the single, neutral judge
for both sides of the head-to-head (AuditOwl's own findings and the
baselines' raw predictions).
"""
import json
import pathlib
import re
import subprocess
import threading
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent

# Scratch working area for codex subprocess calls -- never the repo itself.
# Codex runs with sandbox=read-only anyway, but this is belt-and-suspenders.
CODEX_CWD = pathlib.Path("/tmp/scicoqa_gpt55_judge/codex_workdir")
CODEX_OUT_DIR = pathlib.Path("/tmp/scicoqa_gpt55_judge/codex_out")
CODEX_CWD.mkdir(parents=True, exist_ok=True)
CODEX_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5.5"

# Verbatim copy of the discrepancy_evaluation_v2 judge prompt
# (scicoqa/inference/discrepancy_eval.py in github.com/UKPLab/scicoqa).
PROMPT = """Your task is to evaluate whether a reference paper - code discrepancies matches a predicted paper - code discrepancy. Follow these steps:

1. Analyze which part of the paper or code each discrepancy is describing. Extract the core claims and issues from the reference and predicted discrepancies.
2. Analyze whether the core claims are about the same issue, i.e. if they describe the same or different paper-code discrepancies. The two discrepancies might use different wording or one might be more detailed than the other. Focus on whether the issue is the same, even if minor details are different. However, if they describe different issues (even about the same topic or part of the paper or code) they do not match.
3. Provide a brief explanation of your reasoning.

## Reference Paper-Code Discrepancy
{reference_discrepancy}

## Predicted Paper-Code Discrepancy
{predicted_discrepancy}

## Answer Format
Provide your answer in the following format:
```yaml
core_claim_reference: <core claim from reference discrepancy>
core_claim_predicted: <core claim from predicted discrepancy>
reasoning: <explanation of why the core claims concern the same issue>
match: <yes | no>
```"""


def render(f: dict) -> str:
    """One finding, rendered the way SciCoQA's prompt asks predictions to read."""
    body = " ".join(p.strip() for p in (f.get("title", ""), f.get("claim", ""),
                                        f.get("concern", "")) if p and p.strip())
    loc = f.get("file")
    return f"{body} (cited location: {loc})" if loc else body


_MATCH_RE = re.compile(r"match:\s*[\"\']?\s*(yes|no)", re.IGNORECASE)
_print_lock = threading.Lock()


def judge_codex(ref: str, pred: str, tag: str, max_retries: int = 6, timeout_s: int = 240):
    """Call `codex exec -m gpt-5.5` with the discrepancy_evaluation_v2 prompt.

    Returns (verdict, raw_text) where verdict is 'yes' / 'no' / '?'. Retries
    with exponential backoff on subprocess error, timeout, or unparseable
    output rather than silently dropping the pair.
    """
    prompt = PROMPT.format(reference_discrepancy=ref, predicted_discrepancy=pred)
    last_text = ""
    for attempt in range(max_retries):
        out_file = CODEX_OUT_DIR / f"{tag}_{uuid.uuid4().hex[:8]}.txt"
        try:
            r = subprocess.run(
                ["codex", "exec", "-m", MODEL, "-s", "read-only",
                 "--skip-git-repo-check", "-o", str(out_file), prompt],
                capture_output=True, text=True, timeout=timeout_s,
                stdin=subprocess.DEVNULL, cwd=str(CODEX_CWD),
            )
        except subprocess.TimeoutExpired:
            with _print_lock:
                print(f"  [retry {attempt+1}/{max_retries}] TIMEOUT on {tag}", flush=True)
            time.sleep(min(60, 2 ** attempt))
            continue

        if out_file.exists():
            last_text = out_file.read_text()
            try:
                out_file.unlink()
            except OSError:
                pass

        if r.returncode != 0 or not last_text.strip():
            with _print_lock:
                print(f"  [retry {attempt+1}/{max_retries}] rc={r.returncode} "
                      f"empty_out={not last_text.strip()} on {tag}: "
                      f"{r.stderr[-300:] if r.stderr else ''}", flush=True)
            time.sleep(min(60, 2 ** attempt))
            continue

        m = _MATCH_RE.findall(last_text)
        if m:
            return m[-1].lower(), last_text

        with _print_lock:
            print(f"  [retry {attempt+1}/{max_retries}] unparseable output on {tag}", flush=True)
        time.sleep(min(60, 2 ** attempt))

    with _print_lock:
        print(f"  GIVING UP after {max_retries} retries on {tag}; marking '?'", flush=True)
    return "?", last_text


def load_headtohead():
    return json.loads((HERE / "baselines.json").read_text())


def rowmap_1_to_10():
    """item number -> row_id, from baselines.json's per_item list.

    baselines.json holds the 10 most-recently-filed items WITH baseline
    predictions available: items 1,2,3,4,6,7,8,9,10,11. Item 5 is excluded --
    zero baseline predictions exist for it in SciCoQA's own release, for any
    model -- and item 11 (the next-most-recent item with real predictions)
    backfills it rather than reporting n=9.
    """
    hh = load_headtohead()
    rm = {int(it["item"].split()[0]): it["row_id"] for it in hh["per_item"]}
    expected = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)
    assert sorted(rm.keys()) == list(expected), f"unexpected item keys: {sorted(rm.keys())}"
    return rm


def run_dirs():
    """item number -> FINAL/runs/<dir> Path (item 5's directory is unused)."""
    return {int(p.name.split("_")[0]): p for p in (HERE / "runs").iterdir() if p.is_dir()}


def refs_for_item(row_id: str):
    row = json.loads((HERE / f"answer_keys/row_{row_id}.json").read_text())
    return [(k.replace("discrepancy_description_", ""), row[k])
            for k in ("discrepancy_description_gemini", "discrepancy_description_gpt")
            if row.get(k)]


def kept_findings(item_dir: pathlib.Path, run: str):
    data = json.loads((item_dir / run / "findings_verified.json").read_text())
    return [f for f in data.get("findings", []) if f.get("verdict") == "keep"]


# --- baseline raw predictions, extracted from SciCoQA's own release ---
#
# `_scicoqa_raw_cache/` is a gitignored, re-fetchable cache of SciCoQA's own
# out/inference_discrepancy_detection_real.tar.gz (github.com/UKPLab/scicoqa,
# main branch) -- not ours to redistribute, same policy as audited papers'
# `code/`. Re-download and extract it before running judge_baselines_gpt55.py
# if this cache is absent.
SCICOQA_ARCHIVE_ROOT = (
    HERE / "_scicoqa_raw_cache" / "extracted" / "out" / "inference"
    / "discrepancy_detection" / "real" / "full"
)

# model name (as used in baselines.json) -> archive subdirectory
BASELINE_MODEL_DIRS = {
    "gpt-5": "discrepancy_gen-013-gpt-5",
    "gpt-5-mini": "discrepancy_gen-003-gpt-5-mini",
    "gemini-2.5-pro": "discrepancy_gen-007-gemini-2.5-pro",
}


def _load_eval_v2_by_id(model_dir_name: str) -> dict:
    path = SCICOQA_ARCHIVE_ROOT / model_dir_name / "eval_v2" / "generations.jsonl"
    by_id = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            by_id.setdefault(d["discrepancy_id"], []).append(d)
    return by_id


_BASELINE_CACHE = {}


def get_baseline_candidates(model_name: str, row_id: str):
    """Distinct raw candidate-discrepancy texts a baseline model predicted for
    the paper identified by `row_id` (SciCoQA discrepancy_id), extracted from
    out/inference_discrepancy_detection_real.tar.gz's real/full/*/eval_v2
    generations.jsonl ("generation" field = the model's raw predicted
    discrepancy text, already exploded to one candidate per row by SciCoQA's
    own eval pipeline). Returns a sorted list (possibly empty if the model
    was not evaluated on / did not predict anything for this paper -- treated
    as "not predicted", matching baselines.json's
    baselines_full_condition_nonpred_miss convention).
    """
    if model_name not in _BASELINE_CACHE:
        _BASELINE_CACHE[model_name] = _load_eval_v2_by_id(BASELINE_MODEL_DIRS[model_name])
    by_id = _BASELINE_CACHE[model_name]
    rows = by_id.get(row_id, [])
    return sorted(set(r["generation"] for r in rows))
