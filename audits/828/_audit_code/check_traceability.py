"""File-existence / grep checks for headline quantitative artefacts:
- Coin-Flip Table 1 prefill-causality harness (320 prompts, prefill conditions)
- ToM Table 4 name-intervention experiment (100 completions/condition)
- ToM Figure 3 plotting / manual-label pipeline
- which generalisation config the headline ToM run script actually uses (names vs objects)
Supports `missing` findings for absent eval harnesses and the names-vs-objects difference."""
import csv, os, re, subprocess

ROOT = os.path.join(os.path.dirname(__file__), "..", "code")
RL = os.path.join(ROOT, "RobertMcCarthy97__neurips_encoded_rl")
MS = os.path.join(ROOT, "GeodesicResearch__mars-steg")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def grep_count(pattern, base):
    """Count files (excluding .git) containing a regex (case-insensitive)."""
    try:
        r = subprocess.run(
            ["grep", "-rIlE", "--exclude-dir=.git", pattern, base],
            capture_output=True, text=True)
        files = [l for l in r.stdout.splitlines() if l.strip()]
        return files
    except Exception as e:
        return []

checks = []
# Table 1: prefill conditions over 320 prompts. Search for the distinctive prefill strings.
t1 = grep_count(r'Flip 1: ?[RL]|No prefill|prefill_condition|prefill_type', RL)
# exclude unrelated vllm kv-cache 'prefill'
t1 = [f for f in t1 if "third_party" not in f and "vllm" not in f]
checks.append(("coin_table1_prefill_harness", "files w/ Table-1 prefill-condition logic", t1))

# Table 4 (ToM): 100 completions per condition, name interventions
t4 = grep_count(r'100 completions|intervened|steganographic token|replace.*name.*condition', MS)
checks.append(("tom_table4_intervention_harness", "files w/ Table-4 name-intervention logic", t4))

# Figure 3 plotting / manual-label aggregation
f3 = grep_count(r'encoded name|steganograph|percentage of (flagged|encoded)|manual label', MS)
checks.append(("tom_fig3_plot_or_label_pipeline", "files w/ Fig-3 encoded-name aggregation/label pipeline", f3))

# Which config does the headline 7B 'with_penalise' run script use?
run_sh = os.path.join(MS, "scripts", "4_testing_theory_of_mind_on_generalisation",
                      "train", "run_theory_of_mind_deepseek_7b_with_penalise.sh")
cfg_used = open(run_sh).read()
override = re.search(r'(config_override[\w]*\.yaml)', cfg_used)
override_name = override.group(1) if override else "NONE"
ov_path = os.path.join(MS, "experiments", "d_generalisation", override_name) if override else None
pen_type = mode = "?"
if ov_path and os.path.exists(ov_path):
    txt = open(ov_path).read()
    m1 = re.search(r'nouns_penalisation_type:\s*(\w+)', txt)
    m2 = re.search(r'mode:\s*(\w+)', txt)
    pen_type = m1.group(1) if m1 else "?"
    mode = m2.group(1) if m2 else "?"

with open(os.path.join(OUT, "traceability.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["artefact", "description", "n_matching_files", "files"])
    for cid, desc, files in checks:
        w.writerow([cid, desc, len(files), "; ".join(os.path.relpath(x, ROOT) for x in files)])
    w.writerow(["headline_7b_run_config", "override yaml used by run_..._7b_with_penalise.sh",
                override_name, f"nouns_penalisation_type={pen_type}, mode={mode}"])

for cid, desc, files in checks:
    print(f"{cid}: {len(files)} file(s) -> {[os.path.relpath(x, ROOT) for x in files]}")
print(f"\nHeadline 7B run script uses: {override_name} -> nouns_penalisation_type={pen_type}, mode={mode}")
print("(Paper Fig.3 headline ToM experiment penalises CHARACTER NAMES.)")
