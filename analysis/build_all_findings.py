#!/usr/bin/env python3
"""
build_all_findings.py — dump verifier-kept findings, grouped by paper, to markdown.

Reads audits/<id>/findings_verified.json (verdict == "keep"), pulls each paper's
title from its metadata.txt, and writes a reference markdown:

  - header with totals (+ severity×confidence matrix & filter sizes in full mode)
  - a per-paper index (most high/high findings first)
  - one section per paper, each finding as a compact block
    (severity/confidence/category, claim, concern, ask, evidence, code quote)

Modes:
  python analysis/build_all_findings.py              -> all_findings.md (all kept findings)
  python analysis/build_all_findings.py send-out     -> send_out.md  (high-sev & high-conf only)

High-severity + high-confidence findings (the recommended send-out) are marked ★.
"""
from __future__ import annotations
import json, glob, re, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.normpath(os.path.join(HERE, "..", "audits"))
SEV = ["high", "medium", "low"]
CONF = ["high", "medium", "low"]
SEVR = {s: i for i, s in enumerate(SEV)}
CONFR = {c: i for i, c in enumerate(CONF)}


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
    body = ("\n".join(lines) + extra).replace("```", "``​`")  # neutralize stray fences
    return "```\n" + body + "\n```"


def main():
    send_out = len(sys.argv) > 1 and sys.argv[1] in ("send-out", "--send-out", "high-high")
    cell = Counter()
    catc = Counter()
    papers = []
    for fj in glob.glob(os.path.join(AUD, "*", "findings_verified.json")):
        d = json.load(open(fj))
        dpath = os.path.dirname(fj)
        aud = d.get("audit", "")
        m = re.match(r"(\d+)_", aud)
        pid = m.group(1) if m else aud.split("_")[0]
        kept = [x for x in d.get("findings", []) if x.get("verdict", "keep") == "keep"]
        if not kept:
            continue
        for x in kept:
            s = (x.get("severity") or "low").lower()
            c = (x.get("confidence") or "low").lower()
            x["_s"] = s if s in SEV else "low"
            x["_c"] = c if c in CONF else "low"
            cell[(x["_s"], x["_c"])] += 1
        if send_out:
            kept = [x for x in kept if x["_s"] == "high" and x["_c"] == "high"]
            if not kept:
                continue
        for x in kept:
            catc[x.get("category", "?")] += 1
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
    if send_out:
        L.append("# NeurIPS audit — high-severity + high-confidence findings\n")
        L.append("_The human-validation send-out set: `audits/*/findings_verified.json`, "
                 "verifier-kept findings filtered to **severity = high AND confidence = high**._\n")
        L.append(f"**{TOTAL} findings across {NPAP} papers.**\n")
        L.append("## By category\n")
        L.append("| category | findings |")
        L.append("|---|--:|")
        for c, n in catc.most_common():
            L.append(f"| {c} | {n} |")
        L.append("")
    else:
        L.append("# NeurIPS audit — all findings, by paper\n")
        L.append("_Generated from `audits/*/findings_verified.json` (verifier-kept findings only; "
                 "the verifier's `lowered`/`reject` findings are excluded)._\n")
        L.append(f"**{TOTAL} findings across {NPAP} papers.**  ★ marks **high-severity + high-confidence** "
                 f"findings — the recommended human-validation send-out "
                 f"(**{HH} findings / {HH_PAPERS} papers**).\n")
        L.append("## Severity × confidence\n")
        L.append("| severity ↓ / confidence → | high | medium | low | total |")
        L.append("|---|--:|--:|--:|--:|")
        for s in SEV:
            row = [cell[(s, c)] for c in CONF]
            L.append(f"| **{s}** | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
        col = [sum(cell[(s, c)] for s in SEV) for c in CONF]
        L.append(f"| **total** | {col[0]} | {col[1]} | {col[2]} | **{TOTAL}** |")
        L.append("")
        L.append("## Send-out size by filter\n")
        L.append("| filter | findings | papers |")
        L.append("|---|--:|--:|")
        filt = [
            ("high-sev & high-conf  ★", lambda s, c: s == "high" and c == "high"),
            ("high-sev, any conf", lambda s, c: s == "high"),
            ("sev ≥ med & conf ≥ med", lambda s, c: SEVR[s] <= 1 and CONFR[c] <= 1),
            ("all findings", lambda s, c: True),
        ]
        for name, pred in filt:
            f = sum(cell[(s, c)] for s in SEV for c in CONF if pred(s, c))
            pap = sum(1 for p in papers if any(pred(x["_s"], x["_c"]) for x in p["f"]))
            L.append(f"| {name} | {f} | {pap} |")
        L.append("")

    L.append("## Papers (most send-out findings first)\n")
    if send_out:
        L.append("| paper | findings |")
        L.append("|---|--:|")
    else:
        L.append("| paper | ★ high/high | total |")
        L.append("|---|--:|--:|")
    for p in papers:
        t = p["title"].replace("|", "\\|")
        if send_out:
            L.append(f"| #{p['pid']} · {t} | {p['n']} |")
        else:
            L.append(f"| #{p['pid']} · {t} | {p['hh']} | {p['n']} |")
    L.append("\n---\n")

    for p in papers:
        L.append(f"## #{p['pid']} · {p['title']}")
        if send_out:
            L.append(f"**{p['n']} findings** (all ★ high-sev / high-conf)\n")
        else:
            L.append(f"**{p['n']} findings** ({p['hh']} ★ high/high)\n")
        for i, x in enumerate(p["f"], 1):
            star = "" if send_out else ("★ " if (x["_s"] == "high" and x["_c"] == "high") else "")
            loc = x.get("file", "") or ""
            ls, le = x.get("line_start"), x.get("line_end")
            if ls:
                loc += f":{ls}" + (f"-{le}" if le and le != ls else "")
            L.append(f"#### {star}F{i} · [{x['_s']} sev · {x['_c']} conf] · "
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
            fq = fence(x.get("quote"))
            if fq:
                L.append(fq)
            L.append("")
        L.append("---\n")

    out = os.path.join(HERE, "send_out.md" if send_out else "all_findings.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    tag = (f"{TOTAL} high/high findings / {NPAP} papers" if send_out
           else f"{TOTAL} findings / {NPAP} papers  ({HH} high/high across {HH_PAPERS} papers)")
    print(tag)
    print(f"wrote {out}  ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
