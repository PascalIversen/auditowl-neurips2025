"""Deterministic check: which third-party modules the src/ entrypoint imports are
NOT listed in requirement.txt. Supports finding unlisted-hard-deps-scipy-portalocker.
Also greps for code producing Fig.2/Table3/OmniQuant. Read-only."""
import os
import re

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "FujitsuResearch__qep")
SRC = os.path.join(REPO, "src")

# third-party (non-local) top-level imports across src/*.py
local = {"datautils", "gptq", "modelutils", "quant", "quant_quip", "qep_awq",
         "resultutils", "vector_balance", "awq", "zeroShot"}
stdlib = {"os", "sys", "re", "json", "csv", "math", "gc", "copy", "time",
          "datetime", "functools", "collections", "logging"}
imports = set()
for f in os.listdir(SRC):
    if f.endswith(".py"):
        for line in open(os.path.join(SRC, f)):
            m = re.match(r"\s*(?:import|from)\s+([a-zA-Z0-9_]+)", line)
            if m:
                imports.add(m.group(1))
third_party = sorted(imports - local - stdlib)

req = open(os.path.join(REPO, "requirement.txt")).read().lower()
req_pkgs = set(re.split(r"[<>=\n]", req))
missing = [p for p in third_party if p.lower() not in req and p.lower() not in req_pkgs]

print("third-party imports in src/*.py:", third_party)
print("NOT found in requirement.txt   :", missing)

# Fig.2 / Table3 / OmniQuant producing-code probes
def grep(pattern):
    hits = []
    for root, _, files in os.walk(SRC):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                for i, line in enumerate(open(p, errors="ignore"), 1):
                    if re.search(pattern, line, re.I):
                        hits.append(f"{os.path.relpath(p, REPO)}:{i}")
    return hits

print("\nFig2 partial-quant/TransBlock-delta code:", grep(r"transblock|first.?10|partial.?quant|delta_m"))
print("Table3 timing in main drivers (llama/qep_awq):",
      [h for h in grep(r"time\.time|perf_counter") if "zeroShot" not in h])
print("OmniQuant code:", grep(r"omniquant"))
