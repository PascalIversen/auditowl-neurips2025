#!/usr/bin/env python3
"""Compare Claude vs Codex/GPT verification verdicts over the 610 findings.

Joins audits/*/findings_verified.json (Claude: Sonnet 4.6 + Opus 4.8
escalation) against audits/*/findings_verified_codex.json (cross-model:
gpt-5.5 + gpt-5.6-sol escalation) on the stable finding `id`. Writes
studies/cross_model/data/comparison.json and a human-readable
studies/cross_model/data/comparison.md with agreement matrix, strictness rates,
Cohen's kappa, breakdowns, and the full disagreement list.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_root(start: Path) -> Path:
    """Walk up to the repo root -- the ancestor holding `audits/`.

    Located by content rather than by a fixed number of `.parent` hops so the
    script keeps working if this file is moved to a different depth.
    """
    for candidate in (start, *start.parents):
        if (candidate / "audits").is_dir():
            return candidate
    raise SystemExit(f"no `audits/` directory found above {start}")


ROOT = _find_root(HERE)
AUDITS = ROOT / "audits"
DATA = HERE / "data"

VERDICTS = ["keep", "lowered", "reject"]


def is_cannot_verify(f: dict) -> bool:
    return "cannot-verify" in f.get("reason", "").lower()


def load(paper: str) -> dict | None:
    claude_p = AUDITS / paper / "findings_verified.json"
    codex_p = AUDITS / paper / "findings_verified_codex.json"
    stage1_p = AUDITS / paper / "findings_verified_codex_stage1.json"
    if not codex_p.is_file():
        return None
    original = {f["id"]: f for f in json.loads((AUDITS / paper / "findings.json").read_text())["findings"]}
    claude = {f["id"]: f for f in json.loads(claude_p.read_text())["findings"]}
    codex = {f["id"]: f for f in json.loads(codex_p.read_text())["findings"]}
    stage1 = {}
    if stage1_p.is_file():
        stage1 = {f["id"]: f for f in json.loads(stage1_p.read_text())["findings"]}
    assert set(claude) == set(codex) == set(original), f"{paper}: id sets differ"
    return {"original": original, "claude": claude, "codex": codex, "stage1": stage1}


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[v] * cb[v] for v in VERDICTS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def rates(findings: list[dict]) -> dict:
    n = len(findings)
    c = Counter(f["verdict"] for f in findings)
    cv = sum(is_cannot_verify(f) for f in findings)
    return {
        "n": n,
        **{v: c[v] for v in VERDICTS},
        "cannot_verify": cv,
        "strictness_pct": round(100 * (c["lowered"] + c["reject"]) / n, 2) if n else None,
        "reject_pct": round(100 * c["reject"] / n, 2) if n else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--papers", nargs="*", help="restrict to these paper folders")
    args = ap.parse_args()

    papers = args.papers or sorted(
        (p.parent.name for p in AUDITS.glob("*/findings_verified.json")),
        key=lambda n: n + "_",  # "<id>_" keeps the original "<id>_<slug>" order
    )
    rows, skipped = [], []
    for paper in papers:
        d = load(paper)
        if d is None:
            skipped.append(paper)
            continue
        for fid, orig in d["original"].items():
            cl, cx = d["claude"][fid], d["codex"][fid]
            s1 = d["stage1"].get(fid, {})
            rows.append({
                "paper": paper,
                "id": fid,
                "title": orig["title"],
                "category": orig["category"],
                "severity": orig["severity"],
                "confidence": orig["confidence"],
                "claude_verdict": cl["verdict"],
                "claude_reason": cl["reason"],
                "codex_verdict": cx["verdict"],
                "codex_reason": cx["reason"],
                "codex_stage1_verdict": s1.get("verdict"),
                "codex_escalated": cx.get("escalated", False),
                "codex_cannot_verify": is_cannot_verify(cx),
                "claude_cannot_verify": is_cannot_verify(cl),
                "agree": cl["verdict"] == cx["verdict"],
            })

    matrix = {a: {b: 0 for b in VERDICTS} for a in VERDICTS}
    for r in rows:
        matrix[r["claude_verdict"]][r["codex_verdict"]] += 1
    pairs = [(r["claude_verdict"], r["codex_verdict"]) for r in rows]
    s1_pairs = [(r["claude_verdict"], r["codex_stage1_verdict"]) for r in rows
                if r["codex_stage1_verdict"]]

    def breakdown(key: str) -> dict:
        out = {}
        for val in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == val]
            out[val] = {
                "n": len(sub),
                "agree_pct": round(100 * sum(r["agree"] for r in sub) / len(sub), 2),
                "codex_strictness_pct": round(
                    100 * sum(r["codex_verdict"] != "keep" for r in sub) / len(sub), 2),
                "claude_strictness_pct": round(
                    100 * sum(r["claude_verdict"] != "keep" for r in sub) / len(sub), 2),
            }
        return out

    disagreements = [r for r in rows if not r["agree"]]
    summary = {
        "n_papers": len(papers) - len(skipped),
        "n_skipped_papers": len(skipped),
        "skipped": skipped,
        "n_findings": len(rows),
        "agreement_pct": round(100 * sum(r["agree"] for r in rows) / len(rows), 2) if rows else None,
        "cohen_kappa": round(kappa(pairs), 4) if rows else None,
        "cohen_kappa_vs_stage1": round(kappa(s1_pairs), 4) if s1_pairs else None,
        "claude": rates([{"verdict": r["claude_verdict"],
                          "reason": r["claude_reason"]} for r in rows]),
        "codex_final": rates([{"verdict": r["codex_verdict"],
                               "reason": r["codex_reason"]} for r in rows]),
        "codex_stage1": rates([{"verdict": r["codex_stage1_verdict"], "reason": ""}
                               for r in rows if r["codex_stage1_verdict"]]),
        "n_escalated": sum(r["codex_escalated"] for r in rows),
        "matrix_claude_x_codex": matrix,
        "by_category": breakdown("category"),
        "by_severity": breakdown("severity"),
        "by_confidence": breakdown("confidence"),
        "n_disagreements": len(disagreements),
    }

    if not rows:
        raise SystemExit(
            f"joined 0 findings under {AUDITS} -- refusing to overwrite the "
            "committed comparison.json/.md with an empty result. Every paper "
            "needs both findings_verified.json and findings_verified_codex.json; "
            "check that AUDITS points at the audit tree."
        )

    DATA.mkdir(exist_ok=True)
    (DATA / "comparison.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    md = ["# Cross-model verdict comparison (Claude vs Codex/GPT)", ""]
    md += [f"- Findings joined: **{summary['n_findings']}** across "
           f"{summary['n_papers']} papers"
           + (f" ({len(skipped)} papers pending)" if skipped else "")]
    md += [f"- Verdict agreement: **{summary['agreement_pct']}%**, "
           f"Cohen's κ = **{summary['cohen_kappa']}**"]
    for name in ("claude", "codex_stage1", "codex_final"):
        r = summary[name]
        if r["n"]:
            md += [f"- {name}: {r['keep']} keep / {r['lowered']} lowered / "
                   f"{r['reject']} reject ({r['cannot_verify']} cannot-verify) → "
                   f"strictness {r['strictness_pct']}%"]
    md += ["", "## Agreement matrix (rows: Claude, cols: Codex final)", "",
           "| | " + " | ".join(VERDICTS) + " |", "|---|" + "---|" * 3]
    for a in VERDICTS:
        md += [f"| **{a}** | " + " | ".join(str(matrix[a][b]) for b in VERDICTS) + " |"]
    for key in ("by_category", "by_severity", "by_confidence"):
        md += ["", f"## {key[3:].capitalize()} breakdown", "",
               "| value | n | agree % | Claude strict % | Codex strict % |",
               "|---|---|---|---|---|"]
        for val, s in summary[key].items():
            md += [f"| {val} | {s['n']} | {s['agree_pct']} | "
                   f"{s['claude_strictness_pct']} | {s['codex_strictness_pct']} |"]
    md += ["", f"## Disagreements ({len(disagreements)})", ""]
    for r in disagreements:
        md += [f"### {r['id']}",
               f"- **{r['title']}** _{r['category']} · severity {r['severity']} · "
               f"confidence {r['confidence']}_",
               f"- Claude: **{r['claude_verdict']}** — {r['claude_reason']}",
               f"- Codex: **{r['codex_verdict']}**"
               + (" (escalated)" if r["codex_escalated"] else "")
               + f" — {r['codex_reason']}", ""]
    (DATA / "comparison.md").write_text("\n".join(md) + "\n")
    print(f"wrote {DATA / 'comparison.json'} and comparison.md")
    print(f"joined {summary['n_findings']} findings, agreement "
          f"{summary['agreement_pct']}%, kappa {summary['cohen_kappa']}")
    if skipped:
        print(f"pending papers (no codex final yet): {len(skipped)}")


if __name__ == "__main__":
    main()
