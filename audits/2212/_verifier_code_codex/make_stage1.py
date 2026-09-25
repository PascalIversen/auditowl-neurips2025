import json
from pathlib import Path


verdicts = {
    "2212_Best_of_N_Jailbreaking/forecasting-code-absent": {
        "verdict": "keep",
        "reason": "Paper quote is present and targeted searches plus the check script found no repo code that computes the reported forecast-error values.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/composition-analysis-absent": {
        "verdict": "keep",
        "reason": "Paper quote is present and focused searches found PrePAIR/MSJ runners but no code computing the cited composition ASR deltas or sample-efficiency ratios.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/pair-tap-baseline-absent": {
        "verdict": "keep",
        "reason": "Searches found PAIR/TAP dataset references and PrePAIR prompt config, but no PAIR/TAP baseline runner or implementation for the reported comparison.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/reliability-text-vision-absent": {
        "verdict": "reject",
        "reason": "The repo contains a text reliability check implementation in bon/attacks/run_text_bon.py, so the claim that the only reliability harness is audio-only is not proven.",
        "changed": "rejected because the absence claim was too broad",
    },
    "2212_Best_of_N_Jailbreaking/experiment-outputs-not-committed": {
        "verdict": "keep",
        "reason": ".gitignore ignores exp, the exp directory is absent, and experiment notebooks/scripts point to ./exp/bon outputs that are not present.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/powerlaw-notebook-dead-import": {
        "verdict": "keep",
        "reason": "The cited notebook imports bon.utils.power_law and the check script confirms that module and imported symbols are absent.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/random-cap-drops-nonascii": {
        "verdict": "keep",
        "reason": "The cited function has no append path for non-ASCII alphabetic characters that pass the random gate, so those characters are dropped.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/bootstrap-repeats-10-vs-100": {
        "verdict": "keep",
        "reason": "The cited notebook uses num_repeats=10 while the paper text says it generated 100 bootstrapped trajectories.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/csv-circuitbreaking-asr-mismatch": {
        "verdict": "keep",
        "reason": "The cited check script ran successfully and reproduced the Circuit Breaking 37.1% versus paper 52% mismatch.",
        "changed": "nothing",
    },
    "2212_Best_of_N_Jailbreaking/bootstrap-tiling-extrapolation": {
        "verdict": "keep",
        "reason": "The cited sample_without_replacement branch tiles each group's flagged vector to num_samples and shuffles it, so num_samples greater than observed steps reuses outcomes.",
        "changed": "nothing",
    },
}


data = json.loads(Path("findings.json").read_text())
for finding in data["findings"]:
    finding.update(verdicts[finding["id"]])

Path("findings_verified_codex_stage1.json").write_text(json.dumps(data, indent=2) + "\n")
