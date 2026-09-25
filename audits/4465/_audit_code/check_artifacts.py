"""Checks: (1) result_analysis.ipynb missing (README-referenced); (2) no Table-1 accuracy scoring
in get_model_answer.py; (3) broken method call get_last_layer_attribute_state; (4) requirements file absent.
Supports findings: table1-accuracy-no-script, result-analysis-notebook-missing,
generate-cite-broken-method, requirements-missing."""
import os, re, json
REPO = os.path.join(os.path.dirname(__file__), '..', 'code', 'TsinghuaC3I__Decomposed-Forward-Pass')
out = {}

# 1 result_analysis.ipynb
found = []
for root,_,files in os.walk(REPO):
    for f in files:
        if 'result_analysis' in f:
            found.append(os.path.join(root,f))
out['result_analysis_ipynb_found'] = found

# 2 accuracy scoring for Table 1 (get_model_answer in subspace-input-experiment)
gma = os.path.join(REPO,'Input-Level-DePass-Evaluation/Subspace-Input-Attribution/subspace-input-experiment/get_model_answer.py')
txt = open(gma).read()
out['get_model_answer_has_check_answer_match'] = 'check_answer_match' in txt
out['get_model_answer_has_accuracy'] = bool(re.search(r'accuracy|is_correct|correct_count', txt))

# 3 broken method call
mgr = open(os.path.join(REPO,'DePass/manager.py')).read()
out['calls_get_last_layer_attribute_state'] = mgr.count('get_last_layer_attribute_state')
out['defines_get_last_layer_attribute_state'] = bool(re.search(r'def get_last_layer_attribute_state', mgr))
out['defines_get_last_layer_decomposed_state'] = bool(re.search(r'def get_last_layer_decomposed_state', mgr))

# 4 requirements
out['requirements_files'] = [f for f in ['requirements.txt','environment.yml','setup.py','pyproject.toml'] if os.path.exists(os.path.join(REPO,f))]

print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(os.path.dirname(__file__),'out','artifacts.json'),'w'), indent=2)
