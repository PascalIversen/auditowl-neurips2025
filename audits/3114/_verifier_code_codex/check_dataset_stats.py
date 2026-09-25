"""Checks repo's committed graph_statistic.csv n/m against paper Table 2 values.
Supports finding: dataset-stats-mismatch (difference). Read-only."""
import csv, os

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "Rick7117__aesp-local-pagerank")
CSV = os.path.join(REPO, "tables", "graph_statistic.csv")

# Paper Table 2 (Dataset Statistics), transcribed from paper.pdf (paper_text.txt lines ~3998-4078)
paper_table2 = {
    "as-skitter": (1694616, 11094209),
    "cit-patent": (3764117, 16511740),
    "com-dblp": (317080, 1049866),
    "com-friendster": (65608366, 1806067135),
    "com-lj": (3997962, 34681189),
    "com-orkut": (3072441, 117185083),
    "com-youtube": (1134890, 2987624),
    "ogb-mag240m": (244160499, 1728364232),
    "ogbl-ppa": (576039, 21231776),
    "ogbn-arxiv": (169343, 1157799),
    "ogbn-mag": (1939743, 21091072),
    "ogbn-papers100M": (111059433, 1615685450),
    "ogbn-products": (2385902, 61806303),
    "ogbn-proteins": (132534, 39561252),
    "soc-lj1": (4843953, 42845684),
    "soc-pokec": (1632803, 22301964),
    "sx-stackoverflow": (2584164, 28183518),
    "wiki-en21": (6216199, 160823797),
    "wiki-talk": (2388953, 4656682),
}

repo = {}
with open(CSV) as f:
    for row in csv.DictReader(f):
        repo[row["name"]] = (int(row["n"]), int(row["m"]))

out = os.path.join(os.path.dirname(__file__), "out", "dataset_stats.csv")
mismatches = []
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "repo_n", "paper_n", "n_match", "repo_m", "paper_m", "m_match"])
    for name, (pn, pm) in paper_table2.items():
        if name not in repo:
            w.writerow([name, "MISSING", pn, "?", "MISSING", pm, "?"])
            continue
        rn, rm = repo[name]
        nm = (rn == pn)
        mm = (rm == pm)
        w.writerow([name, rn, pn, nm, rm, pm, mm])
        if not nm or not mm:
            mismatches.append((name, rn, pn, nm, rm, pm, mm))

print("Mismatches between repo graph_statistic.csv and paper Table 2:")
for m in mismatches:
    print(" ", m)
print(f"\nTotal mismatches: {len(mismatches)}")
print(f"Output written to {out}")
