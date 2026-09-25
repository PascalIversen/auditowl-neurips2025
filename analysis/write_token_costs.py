#!/usr/bin/env python3
"""Write per-audit token usage + cost into each audit folder.

For every audit subagent run recorded in the Claude Code session transcripts,
sum its token usage, match it to the paper folder named in its first prompt,
and write `<folder>/token_cost.json` (plus a one-line `token_cost.txt`).

Needs the local Claude Code session transcripts under PROJ below, which are
not part of this release (agent session transcripts, see README "What is
deliberately not here"). The committed `analysis/data/compute_cost.json`
already carries the resulting totals.

Usage:
    python analysis/write_token_costs.py            # write into audits/
    python analysis/write_token_costs.py --dry-run  # print, don't write
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.expanduser(
    "~/.claude/projects/<this repo's path with / replaced by ->"
)

# Opus 4.8 pricing, USD per 1M tokens.
PRICE = {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.5}


def sum_usage(path):
    u = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            continue
        u["input"] += usage.get("input_tokens", 0)
        u["output"] += usage.get("output_tokens", 0)
        u["cache_write"] += usage.get("cache_creation_input_tokens", 0)
        u["cache_read"] += usage.get("cache_read_input_tokens", 0)
    return u


def first_user_text(path):
    for line in open(path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message")
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            return " ".join(
                d.get("text", "") for d in content if isinstance(d, dict)
            )
    return ""


def cost(u):
    return sum(u[k] * PRICE[k] / 1e6 for k in PRICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folder_re = re.compile(r"audits/([0-9A-Za-z_]+)")
    # Accumulate per folder (handles retries: multiple runs sum together).
    per_folder = defaultdict(
        lambda: {"runs": 0, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    )

    for meta_path in glob.glob(f"{PROJ}/*/subagents/*.meta.json"):
        try:
            desc = json.load(open(meta_path)).get("description", "").lower()
        except (json.JSONDecodeError, OSError):
            continue
        # Count the PURE initial-audit pass only. Exclude every verification run --
        # "Validate audit X", "Verify audit X" (incl. the superseded heavy-Opus
        # batch, which says "audit"), "Re-verify X", "Opus escalation X" -- so this
        # is the true audit cost, not audit+verification. (cf. figures/figS3_cost_runtime.py)
        if "audit" not in desc or any(w in desc for w in ("verif", "validat", "escalat")):
            continue
        jsonl = meta_path[: -len(".meta.json")] + ".jsonl"
        if not os.path.exists(jsonl):
            continue
        m = folder_re.search(first_user_text(jsonl))
        if not m:
            continue
        # transcripts predate the rename to bare ids: audits/<id>_<slug> -> <id>
        folder = re.sub(r"^(\d+)_.*", r"\1", m.group(1))
        u = sum_usage(jsonl)
        agg = per_folder[folder]
        agg["runs"] += 1
        for k in ("input", "output", "cache_write", "cache_read"):
            agg[k] += u[k]

    written = 0
    grand = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    for folder, agg in sorted(per_folder.items()):
        dest = os.path.join(WORK, "audits", folder)
        if not os.path.isdir(dest):
            continue  # e.g. older-batch folder no longer present
        usage = {k: agg[k] for k in ("input", "output", "cache_write", "cache_read")}
        total_tokens = sum(usage.values())
        usd = cost(usage)
        payload = {
            "folder": folder,
            "model": "claude-opus-4-8",
            "audit_runs": agg["runs"],
            "tokens": usage,
            "total_tokens": total_tokens,
            "pricing_usd_per_mtok": PRICE,
            "estimated_cost_usd": round(usd, 2),
        }
        for k in usage:
            grand[k] += usage[k]
        line = (
            f"{folder}: {total_tokens:,} tokens "
            f"(in {usage['input']:,} / out {usage['output']:,} / "
            f"cache_w {usage['cache_write']:,} / cache_r {usage['cache_read']:,}) "
            f"-> ${usd:,.2f}  [{agg['runs']} run(s), claude-opus-4-8]"
        )
        if args.dry_run:
            print(line)
        else:
            with open(os.path.join(dest, "token_cost.json"), "w") as f:
                json.dump(payload, f, indent=2)
            written += 1

    gt = sum(grand.values())
    print(
        f"\n{'DRY-RUN ' if args.dry_run else ''}"
        f"{'matched' if args.dry_run else 'wrote'} {written or len(per_folder)} folders | "
        f"total {gt:,} tokens -> ${cost(grand):,.2f}"
    )


if __name__ == "__main__":
    main()
