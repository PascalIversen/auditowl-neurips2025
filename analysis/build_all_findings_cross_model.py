#!/usr/bin/env python3
"""
build_all_findings_cross_model.py — the double-verified rebuttal issue list.

Like build_all_findings.py, but an issue makes the main list only if it survived
BOTH verifications: the Claude verifier kept it (findings_verified.json,
verdict == "keep") AND the independent cross-model verifier (OpenAI Codex CLI,
gpt-5.5 -> gpt-5.6-sol; audits/*/findings_verified_codex.json) did not reject
it. GPT "lowered" survivors are annotated inline. The GPT-rejected findings
are not dropped silently: they appear in an appendix with the evidence-level
re-check verdict recorded in ADJUDICATION below.

  python analysis/build_all_findings_cross_model.py   -> all_findings_cross_model.md
"""
from __future__ import annotations
import json, glob, re, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.normpath(os.path.join(HERE, "..", "audits"))
SEV = ["high", "medium", "low"]
CONF = ["high", "medium", "low"]
SEVR = {s: i for i, s in enumerate(SEV)}
CONFR = {c: i for i, c in enumerate(CONF)}

# Evidence-level re-check of every cross-model reject: each was re-opened against
# the cited files and the verdict below recorded. Shown inline in the appendix.
ADJUDICATION = {
    "numpy-random-integers-removed": ("REJECT_CORRECT", "finding wrong: numpy 2.2.6 retains random_integers (verified empirically)"),
    "multiseed-std-harness-absent": ("REJECT_CORRECT", "finding wrong: 33 robust/ scripts run --itr 7 --fix_seed 0"),
    "auto-pass-no-mp4-inflates-completion": ("REJECT_CORRECT", "inflation path unreachable: module never imports os -> NameError first (the crash itself is a different, real defect)"),
    "readme-generalization-script-commented-out": ("REJECT_CORRECT", "README launches the active condition/generalization.py, not the dead duplicate"),
    "table1-label-mismatch-mcca-vs-dsvd": ("REJECT_CORRECT", "code emits both mCCA and max-dSVD columns; no mismatch"),
    "pde-guidance-disabled-in-notebook": ("REJECT_CORRECT", "guidance active as-run (zeta_obs 5000 / zeta_pde 0.1)"),
    "reported-numbers-not-reproducible": ("REJECT_CORRECT", "seeds set before net construction; stored outputs match Table 1 after rounding"),
    "hf-dataset-only-one-variable": ("REJECT_CORRECT", "HF dataset has held all six variables since 2025-11-07"),
    "trained-models-off-repo-dropbox": ("TECHNICALITY", "substance confirmed; rejected only on non-verbatim quote"),
    "deps-unpinned": ("TECHNICALITY", "substance confirmed; rejected only on non-verbatim quote"),
    "reliability-text-vision-absent": ("FINDING_STANDS", "GPT wrong: the harness it cites is a disabled-by-default stub; finding holds"),
}


def title_of(dpath: str) -> str:
    p = os.path.join(dpath, "metadata.txt")
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8", errors="replace"):
            if ln.startswith("title:"):
                return ln.split("title:", 1)[1].strip()
    return "(title unavailable)"


def fence(q: str, maxlines: int = 12) -> str:
    q = (q or "").rstrip()
    if not q:
        return ""
    lines = q.split("\n")
    extra = ""
    if len(lines) > maxlines:
        extra = f"\n... (+{len(lines) - maxlines} more lines)"
        lines = lines[:maxlines]
    body = ("\n".join(lines) + extra).replace("```", "``​`")
    return "```\n" + body + "\n```"


def finding_block(L: list, i: int, x: dict, star: bool) -> None:
    loc = x.get("file", "") or ""
    ls, le = x.get("line_start"), x.get("line_end")
    if ls:
        loc += f":{ls}" + (f"-{le}" if le and le != ls else "")
    L.append(f"#### {'★ ' if star else ''}F{i} · [{x['_s']} sev · {x['_c']} conf] · "
             f"{x.get('category', '?')} · {(x.get('title') or '').strip()}")
    if x.get("topic"):
        L.append(f"*topic: {x['topic'].strip()}*")
    if x.get("claim"):
        L.append(f"- **Claim:** {x['claim'].strip()}")
    if x.get("concern"):
        L.append(f"- **Concern:** {x['concern'].strip()}")
    if x.get("resolution"):
        L.append(f"- **Ask:** {x['resolution'].strip()}")
    ev = []
    if loc:
        ev.append(f"`{loc}`")
    if x.get("paper_ref"):
        ev.append(f"paper: {x['paper_ref'].strip()}")
    if ev:
        L.append("- **Evidence:** " + " · ".join(ev))
    if x.get("_xm"):
        L.append(f"- **Cross-model:** {x['_xm']}")
    fq = fence(x.get("quote"))
    if fq:
        L.append(fq)
    L.append("")


def main():
    cell = Counter()
    papers, rejected = [], []
    n_lowered = 0
    for fj in sorted(glob.glob(os.path.join(AUD, "*", "findings_verified.json"))):
        dpath = os.path.dirname(fj)
        cx_path = os.path.join(dpath, "findings_verified_codex.json")
        d = json.load(open(fj))
        cx = {x["id"]: x for x in json.load(open(cx_path))["findings"]} if os.path.exists(cx_path) else {}
        aud = d.get("audit", "")
        m = re.match(r"(\d+)_", aud)
        pid = m.group(1) if m else aud.split("_")[0]
        kept, rej = [], []
        for x in d.get("findings", []):
            if x.get("verdict", "keep") != "keep":
                continue
            s = (x.get("severity") or "low").lower()
            c = (x.get("confidence") or "low").lower()
            x["_s"] = s if s in SEV else "low"
            x["_c"] = c if c in CONF else "low"
            g = cx.get(x["id"])
            gv = g["verdict"] if g else None
            if gv == "reject":
                verdict, note = ADJUDICATION.get(x.get("id_local", ""), ("UNADJUDICATED", ""))
                x["_xm"] = f"REJECTED by cross-model verifier — adjudication: **{verdict}** ({note}). GPT: {g['reason']}"
                x["_adj"] = verdict
                rej.append(x)
                continue
            if gv == "lowered":
                n_lowered += 1
                x["_xm"] = (f"kept, severity lowered by cross-model verifier "
                            f"(→ {g.get('severity', '?')}/{g.get('confidence', '?')}) — {g['reason']}")
            else:
                x["_xm"] = ""
            cell[(x["_s"], x["_c"])] += 1
            kept.append(x)
        for x in rej:
            rejected.append((pid, title_of(dpath), x))
        if kept:
            kept.sort(key=lambda x: (SEVR[x["_s"]], CONFR[x["_c"]]))
            n_hh = sum(1 for x in kept if x["_s"] == "high" and x["_c"] == "high")
            papers.append({"pid": pid, "title": title_of(dpath), "f": kept,
                           "n": len(kept), "hh": n_hh})

    papers.sort(key=lambda p: (-p["hh"], -p["n"], int(p["pid"]) if p["pid"].isdigit() else 0))
    TOTAL = sum(p["n"] for p in papers)
    NPAP = len(papers)
    HH = cell[("high", "high")]
    HH_PAPERS = sum(1 for p in papers if p["hh"])

    L = []
    L.append("# NeurIPS audit — findings surviving BOTH verifications\n")
    L.append("_Every issue below was (1) adversarially verified by the primary Claude pipeline "
             "(Sonnet 4.6 first pass, Opus 4.8 escalation; `findings_verified.json`, verdict `keep`) "
             "AND (2) independently re-verified by a cross-model verifier from a different vendor "
             "(OpenAI Codex CLI agent, gpt-5.5 at high reasoning effort with gpt-5.6-sol escalation, "
             "blinded to the primary verdicts; `findings_verified_codex.json`, verdict `keep` or "
             "`lowered`). Cross-model `lowered` survivors are annotated inline. The "
             f"{len(rejected)} primary-kept findings the cross-model verifier rejected are NOT in the "
             "main list — they appear in the appendix with the evidence-level adjudication of each "
             "reject shown inline._\n")
    L.append(f"**{TOTAL} double-verified findings across {NPAP} papers** "
             f"({n_lowered} with cross-model-lowered severity).  ★ marks high-severity + "
             f"high-confidence findings (**{HH} findings / {HH_PAPERS} papers**).\n")
    L.append("## Severity × confidence (double-verified set)\n")
    L.append("| severity ↓ / confidence → | high | medium | low | total |")
    L.append("|---|--:|--:|--:|--:|")
    for s in SEV:
        row = [cell[(s, c)] for c in CONF]
        L.append(f"| **{s}** | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    col = [sum(cell[(s, c)] for s in SEV) for c in CONF]
    L.append(f"| **total** | {col[0]} | {col[1]} | {col[2]} | **{TOTAL}** |")
    L.append("")

    L.append("## Papers (most high/high findings first)\n")
    L.append("| paper | ★ high/high | total |")
    L.append("|---|--:|--:|")
    for p in papers:
        t = p["title"].replace("|", "\\|")
        L.append(f"| #{p['pid']} · {t} | {p['hh']} | {p['n']} |")
    L.append("\n---\n")

    for p in papers:
        L.append(f"## #{p['pid']} · {p['title']}")
        L.append(f"**{p['n']} findings** ({p['hh']} ★ high/high)\n")
        for i, x in enumerate(p["f"], 1):
            finding_block(L, i, x, star=(x["_s"] == "high" and x["_c"] == "high"))
        L.append("---\n")

    L.append("# Appendix — cross-model-rejected findings (excluded above)\n")
    L.append("_Primary-kept findings the cross-model verifier rejected, each re-opened "
             "against the cited files: REJECT_CORRECT = the finding is genuinely wrong "
             "(do not send); TECHNICALITY = substance confirmed, rejected only on strict "
             "verbatim-quote matching; FINDING_STANDS = the cross-model reject is itself "
             "wrong._\n")
    order = {"FINDING_STANDS": 0, "TECHNICALITY": 1, "REJECT_CORRECT": 2}
    rejected.sort(key=lambda r: (order.get(r[2].get("_adj", ""), 9), r[0]))
    for pid, title, x in rejected:
        L.append(f"## [{x.get('_adj', '?')}] #{pid} · {title}")
        finding_block(L, 1, x, star=False)
        L.append("---\n")

    out = os.path.join(HERE, "all_findings_cross_model.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    adjc = Counter(x.get("_adj", "?") for _, _, x in rejected)
    print(f"{TOTAL} double-verified findings / {NPAP} papers  ({HH} high/high across {HH_PAPERS} papers; "
          f"{n_lowered} cross-model-lowered)")
    print(f"appendix: {len(rejected)} cross-model rejects  {dict(adjc)}")
    print(f"wrote {out}  ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
