"""Checks which arch/model/loss names the configs reference vs which are
registered in the repo (supports findings: stage2-vae-type-unregistered).
Pure static parsing (regex) -- does not import torch. Read-only."""
import os
import re
import json

REPO = os.path.join(os.path.dirname(__file__), '..', 'code', 'lyd-2022__Latent-Harmony')
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUT, exist_ok=True)

# 1. Collect registered class names: a @<X>_REGISTRY.register() decorator
#    immediately followed by `class Name(...)`.
registered = {'arch': set(), 'model': set(), 'loss': set(), 'dataset': set()}
reg_map = {
    'ARCH_REGISTRY': 'arch', 'MODEL_REGISTRY': 'model',
    'LOSS_REGISTRY': 'loss', 'DATASET_REGISTRY': 'dataset',
}
for root, _, files in os.walk(os.path.join(REPO, 'basicsr')):
    for fn in files:
        if not fn.endswith('.py'):
            continue
        path = os.path.join(root, fn)
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            m = re.search(r'@(\w+_REGISTRY)\.register\(', line)
            if not m:
                continue
            kind = reg_map.get(m.group(1))
            # find next class line
            for j in range(i + 1, min(i + 4, len(lines))):
                cm = re.match(r'\s*class\s+(\w+)', lines[j])
                if cm and kind:
                    registered[kind].add(cm.group(1))
                    break

# 2. Collect 'type:' values referenced in configs.
referenced = []
cfg_dir = os.path.join(REPO, 'configs')
for fn in sorted(os.listdir(cfg_dir)):
    if not fn.endswith('.yml'):
        continue
    with open(os.path.join(cfg_dir, fn), encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            m = re.search(r'^\s*type:\s*([A-Za-z_]\w*)\s*$', line)
            if m:
                referenced.append((fn, ln, m.group(1)))

# 3. Report unresolved arch/model/loss type references.
all_registered = set().union(*registered.values())
# basicsr builtins that resolve via io_backend/optimizer/scheduler, not registry:
non_registry = {'disk', 'PairedImageDataset', 'L1Loss', 'KlLoss', 'GANLoss',
                'Adam', 'CosineAnnealingRestartLR'}
unresolved = []
for fn, ln, t in referenced:
    if t in non_registry:
        continue
    if t not in all_registered:
        unresolved.append({'config': fn, 'line': ln, 'type': t})

result = {
    'registered': {k: sorted(v) for k, v in registered.items()},
    'config_type_refs': [{'config': f, 'line': l, 'type': t} for f, l, t in referenced],
    'unresolved_arch_model_loss_types': unresolved,
}
with open(os.path.join(OUT, 'registry_vs_configs.json'), 'w') as f:
    json.dump(result, f, indent=2)

print("Registered arch classes:", sorted(registered['arch']))
print("Registered model classes:", sorted(registered['model']))
print("Registered loss classes:", sorted(registered['loss']))
print()
print("UNRESOLVED type references in configs (not registered, not builtin):")
for u in unresolved:
    print(f"  {u['config']}:{u['line']}  type: {u['type']}")
if not unresolved:
    print("  (none)")
