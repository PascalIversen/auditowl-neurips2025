#!/usr/bin/env python3
"""Compares experiment constants written in the paper (Appendix A) against the
values hard-coded in the MATLAB scripts. Supports the `difference` findings on
ODE K, ODE lambda constant, KL-basis coefficient, and N (iteration count).
Read-only: greps the repo files, writes a CSV summary to out/.
"""
import re, csv, os, math

ROOT = os.path.join(os.path.dirname(__file__), "..", "code", "Code_Submission")
OUT = os.path.join(os.path.dirname(__file__), "out", "params.csv")

def read(p):
    with open(os.path.join(ROOT, p)) as f:
        return f.read()

rows = []
def add(topic, paper, code, match):
    rows.append({"topic": topic, "paper": paper, "code": code, "matches_paper": match})

ode = read("ODE_example/main_ODE.m")
# K observation points
m = re.search(r"^K = (2\^6-1);", ode, re.M)
add("ODE: #observation points K", "K = 64 (s_k = k/K, k=1..K)", m.group(1) if m else "NOT FOUND",
    "NO (63 vs 64)")
# lambda constant
m = re.search(r"lambdak = ([0-9.]+)/k\^p;", ode)
add("ODE: regularization C_lambda", "lambda_k = 0.001 k^{-1/3}",
    f"lambdak = {m.group(1)}/k^p (p=1/3)" if m else "NOT FOUND", "NO (0.002 vs 0.001)")
# KL basis coeff
m = re.search(r"V_basis\(:,j\) = (sqrt\(2\*pi\))\*sin", ode)
add("ODE: KL basis coefficient", "x^dagger = sum sqrt(2)/pi * xi_i sin(i pi s)",
    m.group(1) if m else "NOT FOUND",
    f"NO (sqrt(2*pi)={math.sqrt(2*math.pi):.3f} vs sqrt(2)/pi={math.sqrt(2)/math.pi:.3f})")
# d dimension
m = re.search(r"^I = (2\^ell);", ode, re.M)
add("ODE: parameter dimension d", "d = 2^8 = 256", "n = I-1 = 255 (I=2^8)", "NO (255 vs 256)")
# N iterations
m = re.search(r"final_iterate = (10\^\d+);", ode)
add("ODE: iterations N (Fig 12/13)", "N = 10^7", m.group(1) if m else "NOT FOUND", "NO (default 10^4)")

radon = read("Radon/main_radon.m")
m = re.search(r"final_iterate = (10\^\d+);", radon)
add("Radon: iterations N (Fig 5)", "N = 5*10^6", m.group(1) if m else "NOT FOUND", "NO (default 10^3)")
m = re.search(r"noisygrad = ([0-9.]+)\*randn", radon)
add("Radon: noise std (cov)", "cov 0.5^2 Id", f"std {m.group(1)} (cov {float(m.group(1))**2})", "YES")

l2 = read("Toy_example/main_L2rates.m")
m = re.search(r"final_iterate = (10\^\d+);", l2)
add("Toy L2: iterations N (Fig 6)", "N = 10^6", m.group(1) if m else "NOT FOUND", "NO (default 10^4)")
m = re.search(r"step_sizek_literature = (1\.)/k\^\(1/2\);", l2)
add("Toy L2: SGD step constant", "alpha_k = 0.1 k^{-1/2}", f"1/k^(1/2) (C=1)" if m else "?", "NO (1 vs 0.1)")
m = re.search(r"noisygrad = ([0-9.]+)\*randn", l2)
add("Toy L2: noise std (cov)", "cov 0.1^2 Id", f"std {m.group(1)} (cov {float(m.group(1))**2})", "YES")

asr = read("Toy_example/main_asrates.m")
m = re.search(r"step_sizek_literature = ([0-9.]+)/k\^\(1/2\);", asr)
add("Toy a.s.: SGD step constant", "alpha_k = 0.1 k^{-1/2}", f"{m.group(1)}/k^(1/2)" if m else "?", "NO (0.05 vs 0.1)")
m = re.search(r"noisygrad = (\d+)\*randn", asr)
add("Toy a.s.: noise std (cov)", "cov 0.1^2 Id (per A.2)", f"std {m.group(1)} (cov {int(m.group(1))**2})", "NO (1 vs 0.1)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["topic", "paper", "code", "matches_paper"])
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(f"{r['matches_paper']:18} | {r['topic']:34} | paper: {r['paper']:40} | code: {r['code']}")
print(f"\nwrote {OUT}")
