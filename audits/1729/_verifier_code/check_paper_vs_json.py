#!/usr/bin/env python3
"""Compare paper Table 2/3 reported min. values against the values produced by the
shipped Data/Tests JSON (latest file per env/alg/rtype, as DataCollection.ipynb does).
Quantifies the reproduction gap; supports a difference/reproducibility finding."""
import json, os, glob

TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
        "zenodo-17424410__RPOMDP_Benchmark", "Data", "Tests"))


def latest(env, alg, rt):
    fs = sorted(glob.glob(os.path.join(TESTS, f"RPolicyTest_{env}_{alg}_{rt}_*.json")))
    if not fs:
        return None
    return json.load(open(fs[-1]))


def vadv(env, alg, rt):
    d = latest(env, alg, rt)
    return None if d is None else float(d["value_adv"])


# (display, code). Paper min values from Table 2 (Center=mid, Ent=maxent, RMDP=rmdp)
T2 = {
 "TOY*":("Toy",(37.48,37.48,32.49)),
 "ECHO":("Machine",(19.31,19.30,21.12)),
 "PARITY(inf)":("ChainInf",(9.25,9.07,13.65)),
 "PARITY(10)":("Chain10",(59.92,62.40,55.19)),
 "TIGER":("Tiger",(19.35,19.35,13.12)),
 "HoH(5)":("HeavenOrHell5",(-24.04,-21.55,-24.04)),
 "HoH(10)":("HeavenOrHell10",(-37.35,-36.71,-37.36)),
 "MINIHALLWAY":("MiniHallway",(0.76,0.76,0.76)),
 "ALOHA(10)":("Aloha10",(62.41,59.70,56.49)),
 "REPLACEMENT":("Replacement",(-47.64,-46.69,-46.58)),
 "HEALTHDETECTION":("CNC_Detection",(-5718.42,-5661.49,-5554.34)),
}
# Paper Table 3 min values (RHSVI, RQMDP, RFIB) all rtype=full
T3 = {
 "TOY*":("Toy",(69.99,32.49,62.47)),
 "ECHO":("Machine",(31.10,25.45,25.44)),
 "PARITY(inf)":("ChainInf",(20.00,7.98,20.00)),
 "PARITY(10)":("Chain10",(62.71,38.89,62.71)),
 "TIGER":("Tiger",(19.36,-20.04,14.49)),
 "HoH(5)":("HeavenOrHell5",(-21.55,-63.76,-63.76)),
 "HoH(10)":("HeavenOrHell10",(-36.71,-63.76,-63.76)),
 "MINIHALLWAY":("MiniHallway",(0.76,0.25,0.25)),
 "ALOHA(10)":("Aloha10",(46.96,57.32,53.39)),
 "REPLACEMENT":("Replacement",(-46.16,-51.96,-70.87)),
 "HEALTHDETECTION":("CNC_Detection",(-5596.72,-5671.29,-5660.60)),
}

out = []
def reldiff(p, j):
    if j is None: return None
    return abs(p - j) / (abs(p) if abs(p) > 1e-9 else 1.0)

out.append("=== Table 2: paper min vs JSON value_adv (rtypes mid/maxent/rmdp) ===")
maxrel2 = 0.0
for disp,(code,paper) in T2.items():
    js = [vadv(code,"RHSVI",rt) for rt in ["mid","maxent","rmdp"]]
    rels = [reldiff(p,j) for p,j in zip(paper,js)]
    maxrel2 = max([maxrel2]+[r for r in rels if r is not None])
    out.append(f"{disp:16s} paper={paper} json={tuple(round(x,2) if x is not None else None for x in js)} reldiff={tuple(round(r,3) if r is not None else None for r in rels)}")
out.append(f"max rel diff (T2) = {maxrel2:.3f}")

out.append("")
out.append("=== Table 3: paper min vs JSON value_adv (RHSVI/RQMDP/RFIB, full) ===")
maxrel3 = 0.0
for disp,(code,paper) in T3.items():
    js = [vadv(code,alg,"full") for alg in ["RHSVI","RQMDP","RFIB"]]
    rels = [reldiff(p,j) for p,j in zip(paper,js)]
    maxrel3 = max([maxrel3]+[r for r in rels if r is not None])
    out.append(f"{disp:16s} paper={paper} json={tuple(round(x,2) if x is not None else None for x in js)} reldiff={tuple(round(r,3) if r is not None else None for r in rels)}")
out.append(f"max rel diff (T3) = {maxrel3:.3f}")

text="\n".join(out)
print(text)
od=os.path.join(os.path.dirname(__file__),"out"); os.makedirs(od,exist_ok=True)
open(os.path.join(od,"paper_vs_json.txt"),"w").write(text+"\n")
