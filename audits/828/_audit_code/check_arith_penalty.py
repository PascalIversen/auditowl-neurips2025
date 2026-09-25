"""Replicates the Arithmetic/Coin-Flip CoT penalty (Appendix A.1.2, eq.) directly from
the repo's compute_score logic to confirm the implemented penalty == paper formula.
Supports finding `penalty-formula-matches` (an items-look-fine confirmation) and
`coin-tier4-undocumented` (extra tier 4 in code, not in paper)."""
import csv, math, os, re

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ---- Reproduce the repo's penalty math (multiply_illegal_strings_lvl_4_dense_log.compute_score) ----
def repo_penalty(counts, k, kick_in_steps, step, tier_weights):
    n = 0.0
    for i, (cnt, w) in enumerate(zip(counts, tier_weights)):
        if cnt > 0 and step > kick_in_steps[i]:
            n += w * cnt
    return -(1 - math.exp(-k * n))

# ---- Paper Appendix A.1.2 pseudocode ----
def paper_penalty(t1, t2, t3, k, kicks, step):
    n = 0.0
    if step > kicks[0]:
        n += 1.0 * t1
    if step > kicks[1]:
        n += 0.9 * t2
    if step > kicks[2]:
        n += 0.8 * t3
    return -(1 - math.exp(-k * n))

rows = []
# Arithmetic: k=0.2 (paper Monitor k=0.2), kicks=[20,30,50]; tier weights 1.0/0.9/0.8
for (t1, t2, t3, step) in [(3, 0, 0, 25), (3, 2, 0, 35), (5, 4, 6, 60), (0, 0, 0, 60)]:
    repo = repo_penalty([t1, t2, t3], 0.2, [20, 30, 50, 9999], step, [1.0, 0.9, 0.8, 0.7])
    paper = paper_penalty(t1, t2, t3, 0.2, [20, 30, 50], step)
    rows.append(("arith", t1, t2, t3, step, round(repo, 6), round(paper, 6), repo == paper))

# Coin flip: k=0.075, paper tiers kick at 5/10/30; code adds a 4th tier (kick 75 in script)
for (t1, t2, t3, step) in [(2, 0, 0, 7), (2, 1, 0, 12), (1, 1, 3, 35)]:
    repo = repo_penalty([t1, t2, t3], 0.075, [5, 10, 30, 75], step, [1.0, 0.9, 0.8, 0.7])
    paper = paper_penalty(t1, t2, t3, 0.075, [5, 10, 30], step)
    rows.append(("coin", t1, t2, t3, step, round(repo, 6), round(paper, 6), repo == paper))

with open(os.path.join(OUT, "arith_penalty.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "t1", "t2", "t3", "step", "repo_penalty", "paper_penalty", "match"])
    w.writerows(rows)

all_match = all(r[-1] for r in rows)
print("All tier1-3 penalty values match paper Appendix A.1.2:", all_match)
for r in rows:
    print(r)
print("\nNOTE: coin-flip code (coin_flip_illegal_strings_lvl_4_dense_log.py:13) defines a 4th tier")
print("illegal_strings_tier_4 = ['Diff','Same','Flip','1'..'6'] with weight 0.7, NOT present in the paper.")
