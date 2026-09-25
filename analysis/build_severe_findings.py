#!/usr/bin/env python3
"""Emit `analysis/severe_findings.md` — every high-severity finding that survived
adversarial re-verification, grouped by category.

Population matches the headline figures: severity == "high" and NOT dropped
(verdict != "reject" and supplement_verdict != "false_positive"). Questions are
included (flagged) so the total reconciles with `figure_data.json`'s
`severity_breakdown.n_high` count.

Run:  python analysis/build_severe_findings.py
"""
from __future__ import annotations
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "severe_findings.md"

CAT_ORDER = ["missing", "difference", "bug", "methodology"]
CATLAB = {"missing": "Missing code / data", "difference": "Paper–code mismatch",
          "bug": "Technical bug", "methodology": "Methodology / validity"}
CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def is_dropped(f):
    return (f.get("verdict") == "reject"
            or f.get("supplement_verdict") == "false_positive"
            or f.get("provenance_verdict") == "false_positive")


def pretty_title(audit: str) -> str:
    slug = audit.split("_", 1)[1] if "_" in audit else audit
    return slug.replace("_", " ").strip()


# ---- collect --------------------------------------------------------------
by_cat = defaultdict(list)
for fp in sorted(glob.glob(str(ROOT / "audits" / "*" / "findings_verified.json"))):
    d = json.loads(Path(fp).read_text())
    audit = d.get("audit", "")
    pid = audit.split("_")[0]
    title = pretty_title(audit)
    for f in d.get("findings", []):
        if is_dropped(f):
            continue
        if (f.get("severity") or "").lower() != "high":
            continue
        by_cat[f.get("category")].append({"pid": pid, "paper": title, **f})

total = sum(len(v) for v in by_cat.values())
papers = {c: len({x["pid"] for x in v}) for c, v in by_cat.items()}

# ---- render ---------------------------------------------------------------
lines = []
lines.append("# Severe findings — NeurIPS 2025 reproducibility audit\n")
lines.append(
    "All **high-severity** findings that survived the adversarial re-verification "
    "pass (verdict ≠ *reject*, not a supplement false-positive, not a "
    "repo-provenance false-positive), across the 100 "
    "audited papers. Grouped by category, then ordered by confidence and paper id. "
    "Generated from `audits/*/findings_verified.json` by "
    "`build_severe_findings.py`.\n")
lines.append(f"**{total} high-severity findings** in total.\n")

lines.append("| Category | Findings | Papers |")
lines.append("|---|--:|--:|")
for c in CAT_ORDER:
    lines.append(f"| {CATLAB[c]} | {len(by_cat.get(c, []))} | {papers.get(c, 0)} |")
lines.append(f"| **Total** | **{total}** | |\n")

for c in CAT_ORDER:
    items = by_cat.get(c, [])
    if not items:
        continue
    items.sort(key=lambda x: (CONF_RANK.get((x.get("confidence") or "").lower(), 3),
                              int(x["pid"]) if x["pid"].isdigit() else 0))
    lines.append(f"\n## {CATLAB[c]} — {len(items)} findings across {papers[c]} papers\n")
    for x in items:
        q = "  *(question — evidence incomplete)*" if x.get("status") == "question" else ""
        lines.append(f"### #{x['pid']} · {x['paper']}{q}")
        lines.append(f"**{x.get('title', '').strip()}**\n")
        meta = [f"confidence: {x.get('confidence', '?')}"]
        if x.get("topic"):
            meta.append(f"topic: {x['topic']}")
        if x.get("verdict") and x["verdict"] != "keep":
            meta.append(f"verdict: {x['verdict']}")
        lines.append("_" + " · ".join(meta) + "_\n")
        if x.get("claim"):
            lines.append(f"- **Claim:** {x['claim'].strip()}")
        if x.get("concern"):
            lines.append(f"- **Concern:** {x['concern'].strip()}")
        if x.get("resolution"):
            lines.append(f"- **Ask:** {x['resolution'].strip()}")
        # evidence
        ev = x.get("file", "")
        ls, le = x.get("line_start"), x.get("line_end")
        if ls:
            ev += f":{ls}" + (f"-{le}" if le and le != ls else "")
        ev_bits = [f"`{ev}`"] if ev else []
        if x.get("paper_ref"):
            ev_bits.append(f"paper: {x['paper_ref']}")
        if x.get("check_script"):
            ev_bits.append(f"check: `{x['check_script']}`")
        if ev_bits:
            lines.append(f"- **Evidence:** " + " · ".join(ev_bits))
        lines.append("")

OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}  ({total} findings: "
      + ", ".join(f"{c} {len(by_cat.get(c, []))}" for c in CAT_ORDER) + ")")
