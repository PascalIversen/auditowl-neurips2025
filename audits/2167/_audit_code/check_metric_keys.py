#!/usr/bin/env python3
"""Extract the metric keys actually produced by CFMetrics.calc_all_metrics via AST.

Supports finding hypervolume-not-computed: the paper's Table 1/3/6/8 report a
'Hypervol.' diversity column, but calc_all_metrics must list it for the repo to
compute it. Read-only AST parse; writes out/metric_keys.csv.
"""
import ast
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.abspath(os.path.join(
    HERE, "..", "code", "ofurman__DiCoFlex", "counterfactuals", "metrics", "metrics.py"))

def main():
    src = open(METRICS, encoding="utf-8").read()
    tree = ast.parse(src)
    keys = []
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            methods.append(node.name)
        if isinstance(node, ast.FunctionDef) and node.name == "calc_all_metrics":
            for n in ast.walk(node):
                if isinstance(n, ast.Dict):
                    for k in n.keys:
                        if isinstance(k, ast.Constant):
                            keys.append(k.value)
    out = os.path.join(HERE, "out", "metric_keys.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["calc_all_metrics_key"])
        for k in keys:
            w.writerow([k])
    print("Methods defined on CFMetrics:", sorted(methods))
    print("\ncalc_all_metrics returns these keys:")
    for k in keys:
        print("  ", k)
    has_hv = any("hyperv" in (k or "").lower() or "diversity" in (k or "").lower() for k in keys)
    print(f"\nhypervolume/diversity key present in calc_all_metrics: {has_hv}")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
