#!/usr/bin/env python3
"""Checks which weighting schemes in computeweight.calculate_weight run; sqrt/arccos/cosine raise NameError because `math` is never imported. Supports finding computeweight-missing-math-import."""
import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent / "code" / "Darcyddx__graph-prompt"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("computeweight", REPO / "computeweight.py")
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)

lines = []
for m in ["linear", "sqrt", "squared", "equal", "arccos", "cosine", "adaptive"]:
    try:
        r = cw.calculate_weight(2, 6, m, num_active_graphs=3)
        lines.append(f"{m:10s} -> OK ({r})")
    except Exception as e:  # noqa: BLE001
        lines.append(f"{m:10s} -> {type(e).__name__}: {e}")

report = "\n".join(lines) + "\n"
print(report, end="")
(OUT / "check_weight_methods.txt").write_text(report)
