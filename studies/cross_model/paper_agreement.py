#!/usr/bin/env python3
"""Section 4.4 - the cross-model verdict-agreement figures, both readings.

`compare_verdicts.py` reports agreement over *every* verdict pair, so a
severity-only difference (the cross-family verifier keeps a finding but lowers
its severity) counts as a disagreement. Section 4.4 asks the narrower question --
how often does the cross-family verifier reject a finding the same-family
pipeline kept -- and so counts rejections only.

All three readings are printed below: Section 4.4 quotes the second, its
footnote the third.

Run: python studies/cross_model/paper_agreement.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "comparison.json"
SEVERITIES = ("high", "medium", "low")


def gwet_ac1(a, b, categories):
    """Chance-corrected agreement, robust to skewed marginals.

    Same estimator the human-evaluation study uses (Section S1.2.1). Cohen's
    kappa is near-degenerate here for the reason documented there: both
    verifiers keep almost everything, so the expected-agreement term approaches
    the observed one and kappa collapses even though the two disagree on very
    few items.
    """
    n, k = len(a), len(categories)
    po = np.mean([x == y for x, y in zip(a, b)])
    pi = {c: (sum(x == c for x in a) + sum(y == c for y in b)) / (2 * n) for c in categories}
    pe = sum(p * (1 - p) for p in pi.values()) / (k - 1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_ci(a, b, categories, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = [gwet_ac1([a[j] for j in idx], [b[j] for j in idx], categories)
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    d = json.loads(DATA.read_text())
    rows = d["rows"]
    disagreements = [r for r in rows if not r["agree"]]

    print(f"{len(rows)} findings compared across {d['summary']['n_papers']} papers\n")

    print("A. every verdict difference (compare_verdicts.py)")
    print(f"   disagreements {len(disagreements)}  "
          f"agreement {d['summary']['agreement_pct']}%  "
          f"Cohen's kappa {d['summary']['cohen_kappa']}")
    kinds = Counter((r["claude_verdict"], r["codex_verdict"]) for r in disagreements)
    for (a, b), n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"     claude={a:<8} codex={b:<8} {n:>3}")

    # Section 4.4 counts a verdict as changed when either verifier rejected a
    # finding the other kept, in both directions: swapping the judge flips a
    # verdict whichever way it goes.
    rejects = [r for r in disagreements
               if "reject" in (r["codex_verdict"], r["claude_verdict"])]
    agreement = 100 * (len(rows) - len(rejects)) / len(rows)
    sev = Counter(r["severity"] for r in rejects)
    breakdown = ", ".join(f"{sev[s]} {s}" for s in SEVERITIES if sev[s])

    one_way = [r for r in disagreements if r["codex_verdict"] == "reject"]
    print("\nB. rejections, either direction (Section 4.4)")
    print(f"   => {len(rejects)} discrepancies ({breakdown}), agreement {agreement:.0f}%")
    print(f"      of these, {len(one_way)} are cross-family rejections and "
          f"{len(rejects) - len(one_way)} the reverse")

    # Chance-corrected agreement, reported the same way as the human study.
    VERDICTS = ["keep", "lowered", "reject"]
    a = [r["claude_verdict"] for r in rows]
    b = [r["codex_verdict"] for r in rows]
    ac1 = gwet_ac1(a, b, VERDICTS)
    lo, hi = bootstrap_ci(a, b, VERDICTS)
    print("\n   chance-corrected: Gwet's AC1 "
          f"{ac1:.3f} [95% CI {lo:.3f}, {hi:.3f}]; "
          f"Cohen's kappa {d['summary']['cohen_kappa']}")
    print("   marginals (why kappa collapses, cf. Section S1.2.1):")
    for label, r in (("same-family ", a), ("cross-family", b)):
        c = Counter(r)
        print(f"     {label} " + "  ".join(f"{v}={c.get(v, 0)}" for v in VERDICTS))

    # Section 4.4's footnote widens this to include severity downgrades, but not
    # the two cases where our own verifier was the stricter of the pair.
    downgrades = [r for r in disagreements
                  if r["claude_verdict"] == "keep" and r["codex_verdict"] == "lowered"]
    both = len(rejects) + len(downgrades)
    print(f"\nC. rejections plus the {len(downgrades)} severity downgrades (Section 4.4, footnote)")
    print(f"   => {both} differences, agreement {100*(len(rows)-both)/len(rows):.1f}%")


if __name__ == "__main__":
    main()
