import json
from pathlib import Path


VERDICTS = {
    "ud-metric-missing": {
        "verdict": "keep",
        "reason": "The cited QA lines print only ISR and ASR, and a source search found no explicit UD or clean-vs-poisoned utility-drop computation.",
        "changed": "nothing",
    },
    "retrieval-noise-ablation-missing": {
        "verdict": "keep",
        "reason": "The cited RAP embedding function only encodes retrieved memory and a Python-source search found no Gaussian or random-normal embedding-noise injection.",
        "changed": "nothing",
    },
    "embedding-sweep-missing": {
        "verdict": "keep",
        "reason": "The cited EHRAgent code hardcodes all-MiniLM-L6-v2, while the other named models appear only as comments and no EHRAgent model-selection driver was found.",
        "changed": "nothing",
    },
    "qa-missing-pairs-config": {
        "verdict": "keep",
        "reason": "QA/victim.json contains one entry for victim food with no target field, and the QA driver reads that file directly.",
        "changed": "nothing",
    },
    "qa-inject-flag-indentation": {
        "verdict": "keep",
        "reason": "The cited if block is dedented outside the loop and therefore marks only the final loaded record, which currently works because generation appends the inject record last.",
        "changed": "nothing",
    },
    "qa-retrieval-method": {
        "verdict": "keep",
        "reason": "The cited QA retrieval code ranks stored questions by Levenshtein distance, and no live QA embedding/cosine retrieval path was found.",
        "changed": "nothing",
    },
    "qa-model-hardcoded": {
        "verdict": "keep",
        "reason": "The live QA llm call hardcodes model='gpt-4o', and the checker confirms --core_model is only defined or used in commented code.",
        "changed": "nothing",
    },
    "qa-nshots-default": {
        "verdict": "lowered",
        "reason": "The code default is n_shots=3 and Section 5.1 says 3/4/5 records respectively, but Appendix B also says QA retrieves three stored questions.",
        "changed": "Lowered confidence/status because the paper itself contains a conflicting QA retrieval-count statement matching the code.",
        "severity": "low",
        "confidence": "low",
        "status": "question",
    },
    "asr-substring-success-criterion": {
        "verdict": "keep",
        "reason": "The cited EHR checker returns success from victim-absent and target-present substring tests, and RAP uses an analogous target-present/victim-absent search-action test.",
        "changed": "nothing",
    },
}


def main() -> None:
    src = Path("findings.json")
    dst = Path("findings_verified_codex_stage1.json")
    data = json.loads(src.read_text())
    for finding in data["findings"]:
        verdict = VERDICTS[finding["id_local"]]
        finding["verdict"] = verdict["verdict"]
        finding["reason"] = verdict["reason"]
        finding["changed"] = verdict["changed"]
        if verdict["verdict"] == "lowered":
            finding["severity"] = verdict["severity"]
            finding["confidence"] = verdict["confidence"]
            finding["status"] = verdict["status"]
    dst.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
