"""Report the VE_diffuser sigma_max (=> T = sigma_max**2) used in each notebook.

Supports finding: circle-nll-T-mismatch (paper App. G.1 states T=9 for the circle NLL,
but circle2d_log_likelihood.ipynb's active diffuser uses sigma_max=sqrt(5.0) => T=5).
Read-only. Run: cd _audit_code && python check_T_settings.py
"""
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "samuel-howard__log_smoothing"))

pat = re.compile(r"VE_diffuser\(sigma_min=([^,]+),\s*sigma_max=([^)]+)\)")
rows = []
for nb in sorted(f for f in os.listdir(REPO) if f.endswith(".ipynb")):
    data = json.load(open(os.path.join(REPO, nb)))
    for ci, c in enumerate(data["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for m in pat.finditer(src):
            smax = m.group(2).strip()
            # evaluate simple forms
            T = None
            if smax.replace(".", "").isdigit():
                T = float(smax) ** 2
            elif smax == "jnp.sqrt(5.0)" or "sqrt(5" in smax:
                T = 5.0
            elif smax == "sigma_max":
                T = "varies"
            rows.append({"notebook": nb, "cell": ci, "sigma_max": smax, "T": T})

out = os.path.join(os.path.dirname(__file__), "out", "T_settings.json")
json.dump(rows, open(out, "w"), indent=2)
for r in rows:
    print(f"{r['notebook']:35s} cell {r['cell']:<3} sigma_max={r['sigma_max']:<14} T={r['T']}")
print("\nPaper App. G.1 states T=9 for the circle experiment.")
print("Wrote", out)
