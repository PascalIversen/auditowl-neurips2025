"""Checks abstract/intro vs Section-3 vs Table-derived pass@1 improvement deltas (supports headline-delta-inconsistency).

No repo code produces these numbers (ProRL training/eval code is unreleased); this is a
paper-internal arithmetic cross-check of values printed in the PDF tables vs prose.
"""
import csv, os

# Table values (pass@1 averages) from paper.pdf Tables 1-3.
base = {  # DeepSeek-R1-Distill-Qwen-1.5B
    "math_avg": 44.45,
    "code_avg": 23.08,
    "gpqa": 15.86,
    "ifeval": 44.05,
    "reasoning_gym": 4.24,
}
ours = {  # Nemotron-Research-Reasoning-Qwen-1.5B
    "math_avg": 60.14,
    "code_avg": 37.49,
    "gpqa": 41.78,
    "ifeval": 66.02,
    "reasoning_gym": 59.06,
}

# Deltas claimed in the abstract/introduction (p.1-2) vs Section 3 prose (p.4).
abstract_intro = {"math": 14.7, "code": 13.9, "logic": 54.8, "stem": 25.1, "ifeval": 18.1}
section3 = {"math": 15.7, "code": 14.4, "logic": 54.8, "stem": 25.9, "ifeval": 22.0}

table_delta = {
    "math": round(ours["math_avg"] - base["math_avg"], 2),
    "code": round(ours["code_avg"] - base["code_avg"], 2),
    "logic": round(ours["reasoning_gym"] - base["reasoning_gym"], 2),
    "stem": round(ours["gpqa"] - base["gpqa"], 2),
    "ifeval": round(ours["ifeval"] - base["ifeval"], 2),
}

os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "out", "headline_deltas.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["domain", "abstract_intro", "section3", "table_derived",
                "abstract_matches_table", "section3_matches_table"])
    for d in ["math", "code", "logic", "stem", "ifeval"]:
        w.writerow([d, abstract_intro[d], section3[d], table_delta[d],
                    abs(abstract_intro[d] - table_delta[d]) <= 0.1,
                    abs(section3[d] - table_delta[d]) <= 0.1])
        print(f"{d:8s} abstract={abstract_intro[d]:5} sec3={section3[d]:5} "
              f"table={table_delta[d]:6} | abs_ok={abs(abstract_intro[d]-table_delta[d])<=0.1} "
              f"sec3_ok={abs(section3[d]-table_delta[d])<=0.1}")
print("wrote", out)
