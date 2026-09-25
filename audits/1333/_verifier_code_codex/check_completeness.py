"""File-existence check for expected reproduction artefacts and for files
referenced by hardcoded absolute paths / imports (supports findings:
missing-deps-readme-eval, hardcoded-abs-paths). Read-only."""
import os
import json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'lyd-2022__Latent-Harmony'))
OUT = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUT, exist_ok=True)

expected = {
    'dependency_spec': ['requirements.txt', 'environment.yml', 'setup.py',
                        'setup.cfg', 'pyproject.toml', 'Pipfile'],
    'license': ['LICENSE', 'LICENSE.txt', 'LICENSE.md'],
    'eval_or_test_entrypoint': ['test.py', 'eval.py', 'evaluate.py',
                                'inference.py', 'infer.py',
                                'basicsr/test.py'],
    'restoration_net_arch': ['basicsr/archs/sfhformer_arch.py',
                             'basicsr/archs/nafnet_arch.py'],
    'pretrained_weights_dir': ['weights', 'pretrained', 'checkpoints',
                               'experiments'],
    'datasets_dir': ['datasets'],
}

report = {}
for category, candidates in expected.items():
    found = [c for c in candidates if os.path.exists(os.path.join(REPO, c))]
    report[category] = {'candidates': candidates, 'found': found,
                        'present': bool(found)}

# Hardcoded absolute paths referenced in source.
abs_path_refs = []
for root, _, files in os.walk(os.path.join(REPO, 'basicsr')):
    for fn in files:
        if not fn.endswith('.py'):
            continue
        path = os.path.join(root, fn)
        with open(path, encoding='utf-8', errors='replace') as f:
            for ln, line in enumerate(f, 1):
                if '/fs-computility' in line or '/home/' in line.lower():
                    abs_path_refs.append({
                        'file': os.path.relpath(path, REPO),
                        'line': ln, 'text': line.strip()})

report['hardcoded_abs_path_refs'] = abs_path_refs

with open(os.path.join(OUT, 'completeness.json'), 'w') as f:
    json.dump(report, f, indent=2)

print("Expected-artefact presence:")
for cat, info in report.items():
    if cat == 'hardcoded_abs_path_refs':
        continue
    print(f"  {cat:28s} present={info['present']}  found={info['found']}")
print()
print(f"Hardcoded absolute path references: {len(abs_path_refs)}")
for r in abs_path_refs:
    print(f"  {r['file']}:{r['line']}  {r['text']}")
