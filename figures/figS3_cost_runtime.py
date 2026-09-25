#!/usr/bin/env python3
"""Per-paper compute figure: token count and wall-clock, audit + verification.

Two panels, one bar per paper (sorted by total tokens, descending):
  a · tokens (millions)     bar = audit  (base)  +  verification (stacked on top)
  b · wall-clock (minutes)  bar = audit  (base)  +  verification (stacked on top)

Everything is recovered from the raw Claude Code sub-agent transcripts
(~/.claude/projects/<proj>/*/subagents/*.jsonl), classified audit vs verification
by the run's .meta.json description (see classify()) and mapped to a paper by the
audits/<folder> path in the run's first user prompt. AUDIT is the pure-audit pass
ONLY (468M) — deliberately NOT read from token_cost.json, which mislabels the
superseded ~109M "Verify audit X" Opus batch as audit (write_token_costs.py is
being fixed to match). The superseded batch is dropped from both series.

Verification is two-stage (Sonnet validator -> Opus escalation), so every message
is priced by ITS OWN model, not a single table (cost is a side field; bars = tokens).

Run:     python figures/figS3_cost_runtime.py
Outputs: figures/out/figS3_cost_runtime.{png,pdf,svg}
         analysis/data/compute_cost.json
"""
from __future__ import annotations
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures" / "out"
DATA = ROOT / "analysis" / "data"
# Claude Code stores each session under ~/.claude/projects/<slug>, where the
# slug is the project's absolute path with "/" and "_" replaced by "-". Derive
# it from this repo's own location rather than hard-coding it: a literal path
# both leaks the author's username and breaks on every other machine.
PROJ = (Path.home() / ".claude" / "projects" /
        str(ROOT).replace("/", "-").replace("_", "-"))
FP_PATH = DATA / "repo_fingerprints.json"
REPO_FINGERPRINTS = json.loads(FP_PATH.read_text()) if FP_PATH.exists() else {}

# ---- model-aware pricing (USD per 1M tokens) ------------------------------
# cache_write = 1.25x input, cache_read = 0.1x input (Anthropic standard ratios).
PRICE = {
    "opus":   {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    "sonnet": {"input":  3.0, "output": 15.0, "cache_write":  3.75, "cache_read": 0.30},
    "haiku":  {"input":  1.0, "output":  5.0, "cache_write":  1.25, "cache_read": 0.10},
}


def price_for(model: str) -> dict:
    m = (model or "").lower()
    if "sonnet" in m:
        return PRICE["sonnet"]
    if "haiku" in m:
        return PRICE["haiku"]
    return PRICE["opus"]            # opus + <synthetic>/unknown fall back to Opus


FOLDER_RE = re.compile(r"audits/([0-9A-Za-z_]+)")


def token_type_breakdown() -> dict:
    """Cache-read share of total tokens, from the per-paper audits/*/token_cost.json
    files (unlike the raw session transcripts above, these ARE part of the release,
    so this number stays reproducible without the author's local logs).

    Coverage is uneven: most files only tally the audit pass (write_token_costs.py's
    original scope); a few re-audited papers also carry a "stages" breakdown that
    adds the verification pass. Both are summed together, and how many papers
    contributed each is reported alongside the fraction for transparency.
    """
    tok = defaultdict(int)
    n_files = n_with_verify = 0
    for path in glob.glob(str(ROOT / "audits" / "*" / "token_cost.json")):
        try:
            j = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        n_files += 1
        if "stages" in j:
            n_with_verify += 1
            for s in j["stages"]:
                for k in ("input", "output", "cache_write", "cache_read"):
                    tok[k] += s.get(k, 0)
        else:
            for k in ("input", "output", "cache_write", "cache_read"):
                tok[k] += j.get("tokens", {}).get(k, 0)
    total = sum(tok.values())
    return {
        "source": "audits/*/token_cost.json",
        "n_papers": n_files,
        "n_papers_incl_verify_stage": n_with_verify,
        "tokens": dict(tok),
        "cache_read_fraction": round(tok["cache_read"] / total, 4) if total else None,
    }


def first_user_text(path: Path) -> str:
    for line in path.open():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message")
        if isinstance(msg, dict) and msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                return c
            return " ".join(d.get("text", "") for d in c if isinstance(d, dict))
    return ""


# Inter-message gaps longer than this are idle (an agent sitting queued while
# other parallel agents hold the concurrency slots) — not compute. Cap each gap
# at this many seconds so raw transcript span (which can be ~50 min of idle for a
# ~3 min run) does not masquerade as wall-clock.
IDLE_CAP_S = 120.0


def run_metrics(path: Path):
    """Model-aware cost ($), total tokens, and *active* wall-seconds (idle-capped)."""
    cost = 0.0
    toks = 0
    stamps = []
    for line in path.open():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = obj.get("timestamp")
        if ts:
            try:
                stamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        us = msg.get("usage")
        if not isinstance(us, dict):
            continue
        p = price_for(msg.get("model"))
        u = {
            "input": us.get("input_tokens", 0) or 0,
            "output": us.get("output_tokens", 0) or 0,
            "cache_write": us.get("cache_creation_input_tokens", 0) or 0,
            "cache_read": us.get("cache_read_input_tokens", 0) or 0,
        }
        toks += sum(u.values())
        cost += sum(u[k] * p[k] / 1e6 for k in p)
    stamps.sort()
    wall = sum(min((stamps[i + 1] - stamps[i]).total_seconds(), IDLE_CAP_S)
               for i in range(len(stamps) - 1))
    return cost, toks, wall


def classify(desc: str) -> str | None:
    """audit (initial pass) / verify (production re-check) / None (drop).

    Production verification = the two-stage Sonnet validator + Opus escalation
    pipeline: "Validate audit X", "Verify NNNN (Sonnet)", "Verify audit X (Sonnet)",
    "Re-verify X ...", "Opus escalation X".

    DROPPED: the superseded heavy-Opus full re-verification batch — "Verify audit X"
    with no "(Sonnet)" tag, one early session, 5-24M tokens each. It was replaced by
    the cheap Sonnet "Verify NNNN" pass, so counting it would roughly double the
    verification total (and every paper still has a production verification without it).
    """
    d = (desc or "").lower()
    if "validat" in d or "escalat" in d or "re-verify" in d or "reverify" in d:
        return "verify"
    if "verif" in d:
        if "audit" in d and "sonnet" not in d:      # superseded heavy-Opus batch
            return None
        return "verify"
    if "audit" in d:
        return "audit"
    return None


# ---- walk transcripts ------------------------------------------------------
# Exclude the two non-paper buckets — theory/ (out-of-scope theory papers) and
# excluded_justified_no_code/ (papers set aside as justified no-code). Neither is
# a single paper in the 100-paper sample, so neither gets a bar.
_BUCKETS = {"theory", "excluded_justified_no_code"}
on_disk = {os.path.basename(p) for p in glob.glob(str(ROOT / "audits" / "*"))
           if os.path.isdir(p) and os.path.basename(p) not in _BUCKETS}


def released_no_code(folder: str) -> bool:
    """Same rule aggregate.py uses (has_code = repo present AND author core code):
    a paper released no author code if it has no cloned repo under code/, OR its
    repos are all non-core — repo_provenance.json core_present == "no" (a baseline
    / dependency clone, not the authors' own, e.g. #543 reasoning-gym, #2254 notears)."""
    d = ROOT / "audits" / folder
    if not any(os.path.isdir(p) for p in glob.glob(str(d / "code" / "*"))):
        fp = REPO_FINGERPRINTS.get(folder, {})
        if not fp.get("has_code", False):
            return True
    pjp = d / "repo_provenance.json"
    if pjp.exists():
        try:
            return json.load(open(pjp)).get("core_present") == "no"
        except (json.JSONDecodeError, OSError):
            pass
    return False


# Auditing a no-code paper was nonsensical — no author code to read — so its
# token/wall is zeroed below, but it stays in the figure as an empty bar.
NO_CODE = {f for f in on_disk if released_no_code(f)}
agg = defaultdict(lambda: {"audit_tokens": 0, "audit_cost": 0.0, "audit_wall": 0.0, "audit_runs": 0,
                           "verify_tokens": 0, "verify_cost": 0.0, "verify_wall": 0.0, "verify_runs": 0})

for meta in glob.glob(str(PROJ / "*" / "subagents" / "*.meta.json")):
    try:
        desc = json.load(open(meta)).get("description", "")
    except (json.JSONDecodeError, OSError):
        continue
    kind = classify(desc)
    if kind is None:
        continue
    jsonl = Path(meta[:-len(".meta.json")] + ".jsonl")
    if not jsonl.exists():
        continue
    folder = None
    m = FOLDER_RE.search(first_user_text(jsonl))
    if m:
        # transcripts predate the rename to bare ids: audits/<id>_<slug> -> <id>
        folder = re.sub(r"^(\d+)_.*", r"\1", m.group(1))
    if folder not in on_disk:
        continue
    cost, toks, wall = run_metrics(jsonl)
    a = agg[folder]
    a[f"{kind}_tokens"] += toks
    a[f"{kind}_cost"] += cost
    a[f"{kind}_wall"] += wall
    a[f"{kind}_runs"] += 1

# AUDIT is computed straight from the pure-audit transcripts (468M) — NOT from the
# committed token_cost.json, which mislabels the superseded ~109M "Verify audit X"
# Opus batch as audit. write_token_costs.py is being fixed to match this split.

# The transcripts are the author's local agent session logs and are not part of
# the release. Without them we cannot RE-DERIVE the numbers -- but the derived
# result is committed, so the figure itself stays reproducible: fall back to
# plotting `data/compute_cost.json` rather than overwriting it with zeros.
REPLOT_ONLY = not agg

if REPLOT_ONLY:
    cached = json.loads((DATA / "compute_cost.json").read_text())
    rows, tot = cached["per_paper"], cached["totals"]
    print(f"no agent transcripts under {PROJ}")
    print(f"replotting {len(rows)} papers from the committed data/compute_cost.json "
          "(re-deriving requires the author's local session logs)")
    ttb = token_type_breakdown()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "compute_cost.json").write_text(json.dumps(
        {"per_paper": rows, "totals": tot, "pricing_usd_per_mtok": PRICE,
         "n_papers": len(rows), "token_type_breakdown": ttb}, indent=2))
    print(f"cache-read fraction {ttb['cache_read_fraction']:.1%} "
          f"({ttb['n_papers']} papers, {ttb['n_papers_incl_verify_stage']} incl. verify stage)")
else:
    papers = sorted(on_disk)                           # all 100 sample papers
    rows = []
    for f in papers:
        a = agg[f]
        coded = f not in NO_CODE
        z = (lambda v: v) if coded else (lambda v: 0)   # zero out no-code papers
        rows.append({
            "folder": f,
            "has_code": coded,
            "audit_tokens": z(a["audit_tokens"]),
            "verify_tokens": z(a["verify_tokens"]),
            "audit_wall_min": round(z(a["audit_wall"]) / 60, 2),
            "verify_wall_min": round(z(a["verify_wall"]) / 60, 2),
            "audit_cost_usd": round(z(a["audit_cost"]), 2),  # kept for reference only
            "verify_cost_usd": round(z(a["verify_cost"]), 2),
            "audit_runs": a["audit_runs"],                   # run counts (informational)
            "verify_runs": a["verify_runs"],
        })

    tot = {
        "audit_tokens": sum(r["audit_tokens"] for r in rows),
        "verify_tokens": sum(r["verify_tokens"] for r in rows),
        "audit_wall_min": sum(r["audit_wall_min"] for r in rows),
        "verify_wall_min": sum(r["verify_wall_min"] for r in rows),
        "audit_cost_usd": round(sum(r["audit_cost_usd"] for r in rows), 2),
        "verify_cost_usd": round(sum(r["verify_cost_usd"] for r in rows), 2),
    }
    ttb = token_type_breakdown()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "compute_cost.json").write_text(json.dumps(
        {"per_paper": rows, "totals": tot, "pricing_usd_per_mtok": PRICE,
         "n_papers": len(rows), "token_type_breakdown": ttb}, indent=2))
    print(f"papers: {len(rows)} ({len(NO_CODE)} no-code, zeroed)")
    print(f"cache-read fraction {ttb['cache_read_fraction']:.1%} "
          f"({ttb['n_papers']} papers, {ttb['n_papers_incl_verify_stage']} incl. verify stage)")
print(f"audit  tokens {tot['audit_tokens']/1e6:,.0f}M | verify tokens {tot['verify_tokens']/1e6:,.0f}M "
      f"| total {(tot['audit_tokens']+tot['verify_tokens'])/1e6:,.0f}M")
print(f"audit  wall {tot['audit_wall_min']:,.0f} min | verify wall {tot['verify_wall_min']:,.0f} min "
      f"(active compute, idle-capped; agents ran in parallel)")

# ---- the per-audit figures quoted in the text (Section 4.6 / Figure S3) ----
# Means are over the papers that actually ran an audit; the no-code papers are
# zeroed above and would drag both means down if included.
_ran = [r for r in rows if r["has_code"]]
_tok = [r["audit_tokens"] + r["verify_tokens"] for r in _ran]
_wall = [r["audit_wall_min"] + r["verify_wall_min"] for r in _ran]
print(f"per audited paper (n={len(_ran)}): mean {sum(_tok)/len(_tok)/1e6:.1f}M tokens, "
      f"mean {sum(_wall)/len(_wall):.1f} min end-to-end, longest {max(_wall):.1f} min")

# ---- figure ---------------------------------------------------------------
_avail = {fnt.name for fnt in fm.fontManager.ttflist}
_FONT = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
              if f in _avail), "sans-serif")
INK, MUTE = "#333333", "#767676"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [_FONT, "DejaVu Sans"],
    "font.size": 14, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.edgecolor": "#4D4D4D", "axes.linewidth": 1.0,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "legend.frameon": False, "svg.fonttype": "none",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
AUDIT_C = "#3775BA"        # sky blue — the audit pass
VERIFY_C = "#E1A33C"       # amber — verification stacked on top

# one shared x-order: by total tokens (audit+verify), descending → lightest right
order = sorted(range(len(rows)),
               key=lambda i: rows[i]["audit_tokens"] + rows[i]["verify_tokens"],
               reverse=True)
at = np.array([rows[i]["audit_tokens"] for i in order]) / 1e6      # millions
vt = np.array([rows[i]["verify_tokens"] for i in order]) / 1e6
aw = np.array([rows[i]["audit_wall_min"] for i in order])
vw = np.array([rows[i]["verify_wall_min"] for i in order])
x = np.arange(len(order))

# No-code papers are zeroed above, so they sort to the trailing end; count them
# from has_code for the "N no code" bracket under the empty region.
n_nocode = sum(1 for i in order if not rows[i]["has_code"])
n_code = len(order) - n_nocode

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.0))
fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.10, hspace=0.32)


def panel(ax, base, top, ylabel, unit):
    ax.bar(x, base, width=0.9, color=AUDIT_C, edgecolor="none", label="Audit")
    ax.bar(x, top, bottom=base, width=0.9, color=VERIFY_C, edgecolor="none",
           label="Verification")
    ax.set_xlim(-1, len(order))
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks([])
    ax.set_ylim(0, (base + top).max() * 1.14)
    mean = (base + top)[:n_code].mean() if n_code else 0      # mean over audited papers
    ax.axhline(mean, color=MUTE, lw=1.0, ls="--", zorder=0)
    ax.text(len(order) - 1, mean, f"mean {mean:,.1f}{unit} ", ha="right",
            va="bottom", fontsize=14, color=MUTE)
    # bracket grouping the zeroed no-code papers at the trailing end
    if n_nocode:
        y0, y1 = ax.get_ylim()
        tick = (y1 - y0) * 0.05
        xl, xr = n_code - 0.5, len(order) - 0.5
        yb = -tick * 1.3
        ax.plot([xl, xl, xr, xr], [yb + tick, yb, yb, yb + tick],
                color=MUTE, lw=1.0, clip_on=False, zorder=5)
        ax.text((xl + xr) / 2, yb - tick * 0.6, f"{n_nocode} no code",
                ha="center", va="top", fontsize=14, color=MUTE)


panel(ax1, at, vt, "Tokens (M)", "M")
panel(ax2, aw, vw, "Time (min)", "m")
ax1.legend(loc="upper right", ncol=2, fontsize=14, handlelength=1.1,
           labelspacing=0.3, columnspacing=1.4)

for ext in ("png", "pdf", "svg"):
    fig.savefig(FIGS / f"figS3_cost_runtime.{ext}")
plt.close(fig)
print("wrote figures/figS3_cost_runtime.{png,pdf,svg} and data/compute_cost.json")
