"""Checks whether any notebook sets an RNG seed (supports 'no-seed-control' finding)."""
import json, glob, os, re
root = os.path.join(os.path.dirname(__file__), '..', 'code', 'neurips2025-las__LAS-implementation')
nbs = sorted(glob.glob(os.path.join(root, 'LAS-mains', '*', 'code', 'PINN_training.ipynb')))
seed_pat = re.compile(r'(manual_seed|np\.random\.seed|random\.seed|cuda\.manual_seed|set_seed)')
out = []
for nb in nbs:
    src = '\n'.join(''.join(c['source']) for c in json.load(open(nb))['cells'] if c['cell_type']=='code')
    hits = seed_pat.findall(src)
    has_repeat = 'repeat = [0, 1, 2, 3, 4]' in src
    out.append((os.path.relpath(nb, root), len(hits), has_repeat))
with open(os.path.join(os.path.dirname(__file__),'out','seeds.txt'),'w') as f:
    for r in out:
        line=f"{r[0]}: seed_calls={r[1]} repeat5={r[2]}"
        print(line); f.write(line+'\n')
