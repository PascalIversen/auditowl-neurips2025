#!/usr/bin/env python3
"""Reproduce paper Tables 2/3 and Figure 4 Vgap from the repo's Data/Tests JSON,
replicating DataCollection.ipynb's get_data (latest file by lexicographic sort).
Supports traceability-table rows for findings on result reproduction.
"""
import json, os, sys

TESTS = os.path.join(os.path.dirname(__file__), "..", "code",
                     "zenodo-17424410__RPOMDP_Benchmark", "Data", "Tests")
TESTS = os.path.abspath(TESTS)
PREFIX = "RPolicyTest"

# env code-name -> paper display name
ENVS = ["Toy", "Machine", "ChainInf", "Chain10", "Tiger",
        "HeavenOrHell5", "HeavenOrHell10", "MiniHallway", "Aloha10",
        "Replacement", "CNC_Detection"]
PAPER = {"Toy":"TOY*","Machine":"ECHO","ChainInf":"PARITY(inf)","Chain10":"PARITY(10)",
         "Tiger":"TIGER","HeavenOrHell5":"HoH(5)","HeavenOrHell10":"HoH(10)",
         "MiniHallway":"MINIHALLWAY","Aloha10":"ALOHA(10)","Replacement":"REPLACEMENT",
         "CNC_Detection":"HEALTHDETECTION"}
RTYPES = ["full", "mid", "maxent", "rmdp"]


def get_data(env, alg, rtype):
    start = f"{PREFIX}_{env}_{alg}_{rtype}_"
    matches = sorted(f for f in os.listdir(TESTS) if f.startswith(start))
    if not matches:
        return None
    with open(os.path.join(TESTS, matches[-1])) as fh:
        return json.load(fh)


def vals(d):
    return float(d["value_sol"]), float(d["value_adv"]), float(d["std_adv"])


def main():
    out = []
    out.append("=== Table 2 (RHSVI on M_Center=mid, M_Ent=maxent, M_RMDP=rmdp) ===")
    out.append("env\tmid(min,std)\tmaxent(min,std)\trmdp(min,std)")
    for env in ENVS:
        row = [PAPER[env]]
        for rt in ["mid", "maxent", "rmdp"]:
            d = get_data(env, "RHSVI", rt)
            if d is None:
                row.append("MISSING")
            else:
                _, va, sd = vals(d)
                row.append(f"{va:.2f},{sd:.2f}")
        out.append("\t".join(row))

    out.append("")
    out.append("=== Table 3 (RHSVI=full, RQMDP=full, RFIB=full) ===")
    out.append("env\tRHSVI(min,std)\tRQMDP(min,std)\tRFIB(min,std)")
    for env in ENVS:
        row = [PAPER[env]]
        for alg in ["RHSVI", "RQMDP", "RFIB"]:
            d = get_data(env, alg, "full")
            if d is None:
                row.append("MISSING")
            else:
                _, va, sd = vals(d)
                row.append(f"{va:.2f},{sd:.2f}")
        out.append("\t".join(row))

    out.append("")
    out.append("=== Figure 4 Vgap = (value_adv - Vtilde)/|Vtilde|, Vtilde = RHSVI full value_sol ===")
    out.append("env\tVtilde(sol)\tleft:M\tCenter(mid)\tEnt(maxent)\tRMDP(rmdp)\tright:RHSVI\tRQMDP\tRFIB")
    for env in ENVS:
        full = get_data(env, "RHSVI", "full")
        if full is None:
            out.append(f"{PAPER[env]}\tMISSING")
            continue
        vtilde = float(full["value_sol"])  # D[:,0] in notebook
        row = [PAPER[env], f"{vtilde:.3f}"]
        # left panel: M (full value_adv), Center(mid), Ent(maxent), RMDP(rmdp)
        for alg, rt in [("RHSVI","full"),("RHSVI","mid"),("RHSVI","maxent"),("RHSVI","rmdp")]:
            d = get_data(env, alg, rt)
            if d is None:
                row.append("NA")
            else:
                gap = (float(d["value_adv"]) - vtilde)/abs(vtilde)
                row.append(f"{gap:+.3f}")
        # right panel: RHSVI(full), RQMDP, RFIB
        for alg in ["RQMDP","RFIB"]:
            d = get_data(env, alg, "full")
            if d is None:
                row.append("NA")
            else:
                gap = (float(d["value_adv"]) - vtilde)/abs(vtilde)
                row.append(f"{gap:+.3f}")
        out.append("\t".join(row))

    text = "\n".join(out)
    print(text)
    outdir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "tables.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
