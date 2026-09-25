import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "findings.json"
OUTPUT = ROOT / "findings_verified_codex_stage1.json"

verdicts = {
    "4465_DePass_Unified_Feature_Attributing_by_Simple_Decomposed_Forw/table1-accuracy-no-script": {
        "verdict": "keep",
        "reason": "The cited driver only records generated answers and masks, and the checker plus targeted search found no Table 1 factuality accuracy computation for this input-level experiment.",
        "changed": "nothing",
    },
    "4465_DePass_Unified_Feature_Attributing_by_Simple_Decomposed_Forw/runtime-table-no-script": {
        "verdict": "keep",
        "reason": "The README lists only the component-level answer scripts, and targeted source searches found no runtime or timing script for Table 11.",
        "changed": "nothing",
    },
    "4465_DePass_Unified_Feature_Attributing_by_Simple_Decomposed_Forw/result-analysis-notebook-missing": {
        "verdict": "keep",
        "reason": "The README cites result_analysis.ipynb at the specified line, and the checker found no matching notebook in the repository.",
        "changed": "nothing",
    },
    "4465_DePass_Unified_Feature_Attributing_by_Simple_Decomposed_Forw/requirements-missing": {
        "verdict": "keep",
        "reason": "The cited README block lists only three pinned packages, and the checker plus targeted file search found no requirements or environment file.",
        "changed": "nothing",
    },
    "4465_DePass_Unified_Feature_Attributing_by_Simple_Decomposed_Forw/generate-cite-broken-method": {
        "verdict": "keep",
        "reason": "The cited method calls self.get_last_layer_attribute_state, while the class defines no such method and repo search found no definition.",
        "changed": "nothing",
    },
    "4465_DePass_Unified_Feature_Attributing_by_Simple_Decomposed_Forw/subspace-projection-mismatch": {
        "verdict": "keep",
        "reason": "The cited code constructs a per-example language vector by subtracting the mean hidden state, while the paper describes classifier-weight and SVD-based projection.",
        "changed": "nothing",
    },
}

data = json.loads(INPUT.read_text())
for finding in data["findings"]:
    finding.update(verdicts[finding["id"]])

OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
