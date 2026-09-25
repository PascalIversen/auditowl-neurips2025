"""Checks whether baseline models (GraphAny, end-to-end MeanGNN/GAT, DSS/SGC/GCNII) reported in
Tables 1/5/6 and Fig 3 have any implementation in the repo. Supports finding `missing-baselines`."""
import os, re
ROOT = os.path.join(os.path.dirname(__file__), '..', 'code', 'benfinkelshtein__EquivarianceEverywhere')
terms = ['GraphAny', 'graph_any', 'DSS', 'TS-SGC', 'SGCConv', 'GCNIIConv', 'GCN2Conv', 'end-to-end', 'end_to_end', 'baseline']
hits = {t: [] for t in terms}
for dp, _, fs in os.walk(ROOT):
    if '.git' in dp: continue
    for f in fs:
        if not f.endswith('.py'): continue
        p = os.path.join(dp, f)
        with open(p, errors='ignore') as fh:
            for i, line in enumerate(fh, 1):
                for t in terms:
                    if t.lower() in line.lower():
                        hits[t].append(f"{os.path.relpath(p, ROOT)}:{i}")
out = os.path.join(os.path.dirname(__file__), 'out', 'baselines.txt')
with open(out, 'w') as o:
    for t, h in hits.items():
        o.write(f"{t}: {len(h)} hits\n")
        for x in h: o.write(f"   {x}\n")
print(open(out).read())
