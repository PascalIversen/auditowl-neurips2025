"""Checks whether Base MARL baseline configs share hyperparameters with PSI configs.
Supports finding: baseline-asymmetric-hparams. Read-only; parses YAML configs."""
import os, glob, re, json

ROOT = os.path.join(os.path.dirname(__file__), "..", "code",
                    "iamlilAJ__Pre-Strategy-Intervention", "baselines")
KEYS = ["SEED","EPS_START","EPS_FINISH","MAX_GRAD_NORM","LR","ENT_COEF"]

def parse(path):
    d = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*"?([A-Z_]+)"?\s*:\s*([^#,\n]+)', line)
            if m and m.group(1) in KEYS:
                d.setdefault(m.group(1), m.group(2).strip().strip('",'))
    return d

pairs = [
    ("QLearning/config/alg/iql.yaml","QLearning/config/alg/base_marl_iql.yaml"),
    ("QLearning/config/alg/vdn.yaml","QLearning/config/alg/base_marl_vdn.yaml"),
    ("QLearning/config/alg/qmix.yaml","QLearning/config/alg/base_marl_qmix.yaml"),
    ("QLearning/config/alg/pqn.yaml","QLearning/config/alg/base_marl_pqn.yaml"),
    ("IPPO/config/alg/ippo.yaml","IPPO/config/alg/base_marl_ippo.yaml"),
    ("MAPPO/config/alg/mappo.yaml","MAPPO/config/alg/base_marl_mappo.yaml"),
]
out = []
for psi, base in pairs:
    p = parse(os.path.join(ROOT, psi)); b = parse(os.path.join(ROOT, base))
    diffs = {k: (p.get(k), b.get(k)) for k in KEYS if k in p and k in b and p[k] != b[k]}
    out.append({"psi": psi, "base": base, "differing_hparams": diffs})
    print(f"{psi}  vs  {base}")
    for k,(pv,bv) in diffs.items():
        print(f"    {k}: PSI={pv}  BASE={bv}")
    print()
with open(os.path.join(os.path.dirname(__file__),"out","hparam_symmetry.json"),"w") as f:
    json.dump(out, f, indent=2)
print("any_differs:", any(o["differing_hparams"] for o in out))
