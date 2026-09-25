# Code-repository audit — NeurIPS 2025 #828 (RE-AUDIT)

**Paper:** *Large language models can learn and generalize steganographic chain-of-thought under process supervision* (Skaf, Ibanez-Lissen, McCarthy, Watts, Georgiev, Whittingham, Gonzalez-Manzano, Tice, Young, Radmard, Lindner).

## 1. Summary

This is an empirical RL paper with three experiments: **Arithmetic** and **Coin Flip**
(Qwen2.5-3B-Instruct, PPO on a customized TinyZero/VeRL codebase) and a **Theory-of-Mind
(ToM)** generalization experiment (GenPRM-7B / DeepSeek-R1-Distill-Qwen-7B, PPO via TRL
v0.11.0). The headline claims are: (i) CoT penalties cause obfuscated encodings
(Arithmetic >90% accuracy; Coin Flip "R"/"L" scheme); (ii) the Coin-Flip encoding is
*load-bearing* (Table 1 prefill-causality: 98/98/56/55 over 320 prompts); and (iii) the
ToM encoding *generalizes* to held-out, never-penalized names (Figure 3a–d, Table 4).

**This is a re-audit.** The original audit ran when `code/` was empty and the two
anonymous `4open.science` links had expired; it produced only "code absent" findings.
Both author repositories are now cloned and present:
- `code/RobertMcCarthy97__neurips_encoded_rl/` — Arithmetic + Coin Flip (built on TinyZero/VeRL).
- `code/GeodesicResearch__mars-steg/` — ToM generalization experiment.

I audited both. The **training and reward machinery is genuinely present and faithful to
the paper's described method**: I confirmed by re-implementation that the CoT penalty
(`verl/utils/cot_reward_score/*_lvl_4_dense_log.py`) reproduces Appendix A.1.2 exactly
(`_audit_code/check_arith_penalty.py`), the harmonic-mean composite reward
(`mars_steg/task/base_task.py:365-367`) reproduces eq. 6, and the token-level KL+penalty
reward (`mars_steg/utils/ppo_trainer.py:1269`) reproduces eq. 5 including the `KLoffset`.
The hyperparameters in the run scripts/configs match Tables 2 and 3 closely.

The remaining issues are about **result traceability and faithfulness**, not broken code:
(1) the **evaluation/analysis harnesses for the three headline quantitative artefacts —
Table 1 (Coin-Flip prefill causality), Table 4 (ToM name interventions), and the Figure 3
curves/strategy distribution — are not in either repo** (no script computes those numbers;
`plot.py` is a one-line stub; Fig 3c/3d are manually labelled per §B.3); and (2) the
**headline ToM run/test scripts use the `objects` penalisation split, while the paper's
Figure 3 narrative is about character *names*** (a names-mode split exists in the code but
is wired to the zero-shot/finetuning-only override configs, not the main `with_penalise`
run). I also note a minor substring-matching weakness in the names split and a model-name
config that defaults to DeepSeek-R1-Distill-Qwen-7B with GenPRM-7B commented out.

Scripts run (all read-only, outputs in `_audit_code/out/`):
`check_arith_penalty.py` (penalty == paper formula), `check_tom_split.py` (name-split
disjointness + substring pairs), `check_traceability.py` (headline-artefact file
existence + which config the headline run uses), and the pre-existing `check_repo_urls.py`.

## 2. Result-traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Arithmetic CoT penalty formula (App. A.1.2) | `verl/utils/cot_reward_score/multiply_illegal_strings_lvl_4_dense_log.py:136-162` | reproduced | ✓ | Verified (`check_arith_penalty.py`) |
| Coin-Flip CoT penalty formula (App. A.1.2) | `verl/utils/cot_reward_score/coin_flip_illegal_strings_lvl_4_dense_log.py:77-107` | reproduced | ✓ (tiers 1-3) | Verified; code adds undocumented tier 4 |
| Arithmetic illegal-string tiers + kick-ins (20/30/50) | `multiply_illegal_strings_lvl_4_dense_log.py:8-95` + `scripts/10_march_rl/3B_multiply_lvl4.sh:37` | match | ✓ | Verified |
| Coin-Flip illegal strings (Heads/Tails→Head/Tail→H/T) | `coin_flip_illegal_strings_lvl_4_dense_log.py:10-13` | match | ✓ | Verified |
| Arithmetic hyperparams (Table 2: temp 1.0, k 0.2, clamp 1.0, batch 256/128, KL 1e-3) | `scripts/10_march_rl/3B_multiply_lvl4.sh` | match | ✓ | Verified |
| Coin-Flip hyperparams (Table 2: temp 1.1, k 0.075, clamp 0.6) | `scripts/10_march_rl/3B_coin_lvl4_temperature.sh` | match | ✓ | Verified |
| Coin-Flip dataset generation (6 flips, Same/Different) | `examples/data_preprocess/coin_flip/coin_6_flips_decompose.py` | generates | ✓ | Verified (data generated, not committed) |
| Overall reward r = rtask − pCoT (eq. 1) | `verl/trainer/ppo/ray_trainer.py:677-703` | task + (negative) overseer summed | ✓ | Verified |
| Arithmetic ">90% accuracy" headline (§3.1) | training code present; **no eval script reporting accuracy %** | — | — | MISSING (no accuracy-reporting harness) |
| **Table 1** Coin-Flip prefill accuracies (98/98/56/55, n=320) | **(none)** | — | — | MISSING (no prefill-causality harness) |
| Coin-Flip "1 of 3 seeds converged" (§3.1/App D.1) | training scripts present; **no seed-aggregation/eval** | — | — | MISSING |
| ToM harmonic-mean composite reward (eq. 6, wtask=1, wlang=2) | `mars_steg/task/base_task.py:365-367` + `experiments/d_generalisation/config_override.yaml:4-5` | reproduced | ✓ | Verified |
| ToM token-level KL+penalty + KLoffset (eq. 5, KLoffset=0.022) | `mars_steg/utils/ppo_trainer.py:1269` + `theory_of_mind_aspect.py:241` | reproduced | ✓ | Verified |
| ToM hyperparams (Table 3: LoRA r8/α16/dropout.05, 4bit, lr 4e-5, init_kl 0.2, adaptive, whiten, ppo_epochs 10) | `mars_steg/configs/config_deepseek_7b.yaml` | match | ✓ | Verified |
| ToM dataset (ExploreToM) | `mars_steg/dataset/theory_of_mind.csv` (13309 rows) + `generators/theory-of-mind/` | present | ✓ | Verified |
| ToM train/test name-disjoint split (§3.3) | `mars_steg/task/tasks/theory_of_mind_task.py:243-374` (`unseen_nouns`) | split logic present | partial | Verified present; headline run uses `objects` not names (see `tom-config-objects-not-names`) |
| **Figure 3a** ToM task score train/test vs steps | wandb logs `test_task_score`; **no plotting / curve script** | — | — | MISSING (plot.py is a stub) |
| **Figure 3b** avg unencoded character-name count vs steps | **(none)** | — | — | MISSING |
| **Figure 3c** % CoT traces with encoded name (train/test) | **(none — manual labelling, §B.3)** | — | — | MISSING |
| **Figure 3d** test-set encoding-strategy distribution | **(none — manual labelling)** | — | — | MISSING |
| **Table 4** ToM intervened-name accuracies (62.5%/37.2%/…, 100 completions/cond) | **(none)** | — | — | MISSING (no intervention harness) |

Routing: the MISSING analysis-harness rows route to one `missing` finding
(`headline-eval-harnesses-missing`). The names-vs-objects config issue is a `difference`.
The undocumented coin tier-4 is a `difference`. The names-substring split is a low
`methodology`. The manual-labelling reliability of Fig 3c/3d is a `methodology` question.

## 3. Findings

## missing

```yaml finding
id: headline-eval-harnesses-missing
category: missing
topic: "result traceability / evaluation harnesses"
title: "Scripts that compute Table 1, Table 4, and the Figure 3 curves/distribution are absent from both repos"
severity: high
confidence: high
status: finding
file: code/GeodesicResearch__mars-steg/mars_steg/utils/plot.py
line_start: 1
line_end: 1
quote: |
  # plotting util functions
claim: "Both repos contain training code, but no script computes/plots the three headline quantitative artefacts: the Coin-Flip prefill-causality table (Table 1: 98/98/56/55 over 320 prompts), the ToM name-intervention table (Table 4: 100 completions/condition), or the Figure 3 curves and Fig 3d strategy distribution. _audit_code/check_traceability.py finds 0 files implementing a Table-1 prefill harness and 0 implementing a Table-4 intervention harness; mars_steg/utils/plot.py is a one-line stub and no file in either repo calls matplotlib savefig to produce Fig 3."
concern: "None of the paper's three headline numbers/figures can be reproduced or independently checked from the released code; the values exist only in the paper, and Fig 3c/3d are produced by manual labelling (App. B.3) with no released labels or aggregation script."
resolution: "Authors: release the prefill-causality evaluation script (Table 1), the name-intervention script generating 100 completions/condition (Table 4), the script that produces the Figure 3a/3b training/test curves, and the manually-labelled trace data + aggregation behind Fig 3c/3d."
cross_refs: ["tom-config-objects-not-names", "fig3-manual-label-reliability"]
check_script: _audit_code/check_traceability.py
paper_ref: "Table 1; Figure 3a-d; Table 4; Appendix B.3, E.1-E.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: encoded-rl-deps-unpinned
category: missing
topic: "dependency specification / environment"
title: "Arithmetic/Coin-Flip repo dependencies are unpinned (no versions in requirements.txt)"
severity: low
confidence: high
status: finding
file: code/RobertMcCarthy97__neurips_encoded_rl/requirements.txt
line_start: 1
line_end: 13
quote: |
  accelerate
  codetiming
  datasets
  dill
  flash-attn
  hydra-core
  numpy
  pandas
  pybind11
  ray
  tensordict<0.6
  transformers<4.48
  vllm<=0.6.3
  wandb
claim: "The neurips_encoded_rl requirements.txt pins almost no versions (only loose upper bounds on tensordict/transformers/vllm), unlike the mars-steg repo which pins exact versions; PPO/vLLM behaviour is version-sensitive."
concern: "The exact training environment for the Arithmetic and Coin-Flip runs cannot be rebuilt deterministically, reducing reproducibility of those experiments."
resolution: "Authors: pin exact versions (a frozen environment / lockfile) for the TinyZero-based Arithmetic and Coin-Flip experiments, matching the runs reported in the paper."
cross_refs: []
paper_ref: "Appendix A.2 (TinyZero/VeRL stack)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No findings. The reward/penalty/training code I traced runs as intended; the penalty
math, harmonic-mean composite reward, and token-level KL+penalty all reproduce the paper's
equations under deterministic re-implementation (`_audit_code/check_arith_penalty.py`,
manual trace of `base_task.py:365` and `ppo_trainer.py:1269`). No crashes, wrong-axis
aggregations, dead imports, or shape errors were found in the audited paths.

## difference

```yaml finding
id: tom-config-objects-not-names
category: difference
topic: "evaluation consistency (paper vs code)"
title: "Headline ToM run/test scripts penalise & split on OBJECTS, but paper Figure 3 is about character NAMES"
severity: medium
confidence: high
status: finding
file: code/GeodesicResearch__mars-steg/experiments/d_generalisation/config_override.yaml
line_start: 7
line_end: 13
quote: |
  penalisation_class_kwargs:
    max_new_tokens_threshold: 900
    nouns_path: mars_steg/dataset/theory_of_mind_nouns.yaml
    nouns_penalisation_type: objects
    temporal_reward_penalisation_offset: 0.022
    chance_penalisation: 1.0
    mode: unseen_object
claim: "The headline ToM run script run_theory_of_mind_deepseek_7b_with_penalise.sh and its test counterpart both load config_override.yaml, which sets nouns_penalisation_type: objects and mode: unseen_object — penalising and generalisation-splitting on the 428 OBJECT strings, not the 100 character NAMES. The paper's §3.3 / Figure 3 narrative is explicitly about penalising and generalising over character names ('penalization is administered based on the occurrence of the names of specific characters'; Fig 3b 'unencoded character names'; Fig 3d name-encoding strategies)."
concern: "The committed configuration for the headline generalization experiment optimizes a different superclass (objects) than the names-based experiment the paper presents in Figure 3, so a reader running the released script does not reproduce the reported experiment; the names path exists (mode: unseen_nouns, config_override_zero_shot/finetuning) but is wired to the non-headline variants."
resolution: "Authors: confirm which configuration produced Figure 3 / Table 4 and provide the exact names-mode override used for the main RL run, or correct the paper if the headline run was on objects."
cross_refs: ["headline-eval-harnesses-missing", "tom-names-substring-split"]
check_script: _audit_code/check_traceability.py
paper_ref: "Section 3.3; Figure 3b/3d; Appendix B.4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: coin-tier4-undocumented
category: difference
topic: "reward / monitor specification"
title: "Coin-Flip monitor adds an undocumented 4th penalty tier (Diff/Same/Flip/digits) not in the paper"
severity: low
confidence: high
status: finding
file: code/RobertMcCarthy97__neurips_encoded_rl/verl/utils/cot_reward_score/coin_flip_illegal_strings_lvl_4_dense_log.py
line_start: 10
line_end: 13
quote: |
    illegal_strings_tier_1 = ["Heads", "Tails"]
    illegal_strings_tier_2 = ["Head", "Tail"]
    illegal_strings_tier_3 = ["H", "T"]
    illegal_strings_tier_4 = ["Diff", "Same", "Flip", "1", "2", "3", "4", "5", "6"]
claim: "The Coin-Flip reward function (used by scripts/10_march_rl/3B_coin_lvl4_temperature.sh) defines a 4th illegal-string tier penalising 'Diff','Same','Flip' and the digits 1-6 (weight 0.7, kick-in at step 75), but Appendix A.1.2 lists only three tiers (Heads/Tails, Head/Tail, H/T)."
concern: "The actual Coin-Flip monitor penalises a broader set of strings than the paper describes; tiers 1-3 match the paper exactly (verified) but the additional tier 4 changes the optimization pressure and is undisclosed."
resolution: "Authors: state whether the reported Coin-Flip run used the 4th tier (kick_in step 75), and if so add it to Appendix A.1.2; if not, confirm the reported run set tier_4 kick-in beyond training length."
cross_refs: []
check_script: _audit_code/check_arith_penalty.py
paper_ref: "Appendix A.1.2 (Coin Flip illegal strings)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: tom-model-name-deepseek-not-genprm
category: difference
topic: "model specification"
title: "ToM config defaults to DeepSeek-R1-Distill-Qwen-7B; paper names GenPRM-7B as the ToM model"
severity: low
confidence: high
status: finding
file: code/GeodesicResearch__mars-steg/mars_steg/configs/config_deepseek_7b.yaml
line_start: 2
line_end: 2
quote: |
    model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # The model name #"GenPRM/GenPRM-7B"  or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
claim: "The ToM training config used by the headline run script sets model_name to deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, with GenPRM/GenPRM-7B present only as a commented alternative; the paper (§3.3, Table 3, App. C) states the ToM model is GenPRM-7B."
concern: "The released default would train a different (though closely related — GenPRM-7B is itself a DeepSeek-derived model) base model than the paper reports for Figure 3, so the headline experiment may not be reproduced exactly with the committed config."
resolution: "Authors: confirm whether Figure 3 / Table 4 used GenPRM-7B or DeepSeek-R1-Distill-Qwen-7B, and set the committed config's model_name to the one actually used."
cross_refs: ["tom-config-objects-not-names"]
paper_ref: "Section 3.3; Table 3; Appendix C"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: fig3-manual-label-reliability
category: methodology
topic: "evaluation validity / measurement"
title: "Figure 3c/3d steganography rates come from manual labelling with no reported reliability"
severity: medium
confidence: medium
status: question
file: paper.pdf
quote: |
  This process was carried out through manual labeling, since—even
  after using frontier models such as Claude 4.5 and GPT-5 as judges—the models were unable to
  reliably classify the steganographic samples across the different categories.
claim: "The primary generalization metric (% of CoT traces containing an encoded name, Fig. 3c, and the strategy breakdown, Fig. 3d) is produced by manual human labelling of whether each trace is steganographic; neither repo contains the labelled traces or an aggregation/plotting script, and the paper reports no labelling protocol, annotator count, blinding, or inter-rater agreement."
concern: "A subjective, single-pass manual label of 'is this steganographic?' — which the authors say frontier LLM judges could not do reliably — with no inter-rater reliability or blinding makes the headline generalization curves susceptible to confirmation bias and unreproducible from the released artefacts."
resolution: "Authors: release the labelled traces and report the labelling protocol (number of annotators, blinding to train/test membership and training step, inter-rater agreement)."
cross_refs: ["headline-eval-harnesses-missing"]
paper_ref: "Appendix B.3; Figure 3c/3d"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

```yaml finding
id: tom-names-substring-split
category: methodology
topic: "sample independence / train-test split"
title: "ToM names split uses substring (not word-boundary) matching, risking name-leakage for nested names"
severity: low
confidence: high
status: finding
file: code/GeodesicResearch__mars-steg/mars_steg/task/tasks/theory_of_mind_task.py
line_start: 329
line_end: 330
quote: |
            test_mask = self.dataset['infilled_story'].apply(lambda x: any(name in x for name in test_names))
            train_mask = ~test_mask
claim: "In mode 'unseen_nouns' the train/test split assigns a story to test if any test name is a SUBSTRING of the story (name in x), and the disjointness assertion at line 344 also uses substring str.contains(name, regex=False). The object split (lines 390/442/459) correctly uses word-boundary regex, but the names split does not. _audit_code/check_tom_split.py finds 3 nested-name pairs among the 100 names (Gabriel/Gabriella, Daniel/Danielle, Julia/Julian)."
concern: "For nested names, substring matching mis-assigns and could let a train name (e.g. 'Julia') appear inside test stories via a test name ('Julian'), or trip the disjointness assert; it weakens the strict name-disjointness underpinning the generalization claim. Impact is limited (3/100 names) and the headline run uses the object split, so severity is low."
resolution: "Authors: switch the names split (lines 329, 344) to word-boundary regex matching as used for objects, and re-verify train/test name disjointness."
cross_refs: ["tom-config-objects-not-names"]
check_script: _audit_code/check_tom_split.py
paper_ref: "Section 3.3 (train/test name partition)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: tom-loadbearing-edit-confound
category: methodology
topic: "causal / load-bearing claim"
title: "ToM load-bearing intervention (Table 4) lacks a non-name edit control; correct-name swap also lowers accuracy"
severity: low
confidence: medium
status: question
file: paper.pdf
quote: |
  We observed a consistent and substantial drop in accuracy when the original steganographic token was
  replaced—even with the semantically correct name. This indicates that the model has internalized
  the steganographic signal as a meaningful feature and relies on it to successfully complete the task.
claim: "Table 4 / App. E.3 show accuracy dropping when the steganographic token is replaced by the correct original name AND (last paragraph) when replaced by an unrelated name; the experiment has no control that edits a non-name, non-steganographic token by an equivalent amount."
concern: "If substituting even the semantically-correct name into the model's own continuation lowers accuracy, part of the measured drop may reflect generic prefill-edit / off-distribution sensitivity rather than the steganographic encoding being uniquely load-bearing; no harness for this experiment is in the repo to check it."
resolution: "Authors: add a non-name equivalent-edit control and report whether the steganographic-token drop exceeds it; release the intervention harness."
cross_refs: ["headline-eval-harnesses-missing"]
paper_ref: "Appendix E.3; Table 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                |
|-------------|------------|--------------|--------------------------------|
| missing     | 2          | high         | Headline eval/plot harnesses (Table 1, Table 4, Fig 3) absent; encoded-RL deps unpinned. |
| bug         | 0          | -            | Reward/penalty/training code runs as intended; penalty == paper formula (verified). |
| difference  | 3          | medium       | Headline ToM run uses objects not names; undocumented coin tier 4; model defaults to DeepSeek not GenPRM. |
| methodology | 3          | medium       | Fig 3c/3d manual-label reliability; names substring split; load-bearing edit-confound. |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **[missing] `headline-eval-harnesses-missing`** — No script computes/plots the three headline quantitative artefacts (Table 1 prefill causality, Table 4 name interventions, Figure 3 curves + strategy distribution); `plot.py` is a stub and Fig 3c/3d are manual labels with no released data. (high / high)
2. **[difference] `tom-config-objects-not-names`** — The committed headline ToM run/test scripts penalise & split on *objects* (`unseen_object`), but the paper's Figure 3 / Table 4 are about character *names*; the names path exists but is wired to the non-headline configs. (medium / high)
3. **[methodology] `fig3-manual-label-reliability`** — The headline generalization rates (Fig 3c/3d) rest on subjective manual labelling (LLM judges deemed unreliable by the authors) with no reported protocol, blinding, or inter-rater agreement, and no released labels. (medium / medium, question)
4. **[difference] `coin-tier4-undocumented`** — The Coin-Flip monitor adds an undocumented 4th penalty tier (Diff/Same/Flip/digits); tiers 1-3 match the paper exactly (verified). (low / high)
5. **[methodology] `tom-names-substring-split`** — The names train/test split uses substring (not word-boundary) matching; 3/100 names are nested, a narrow name-leakage vector (object split is correct). (low / high)
6. **[difference] `tom-model-name-deepseek-not-genprm`** — ToM config defaults to DeepSeek-R1-Distill-Qwen-7B with GenPRM-7B commented out; paper names GenPRM-7B. (low / high)

### Items that genuinely look fine
- **CoT penalty math matches the paper.** `_audit_code/check_arith_penalty.py` reproduces Appendix A.1.2 exactly for both Arithmetic and Coin Flip (tiers 1-3): all 7 test cases match to 6 d.p.
- **Composite reward matches eq. 6.** The harmonic-mean reward in `mars_steg/task/base_task.py:365-367` is `(t+l)·(task·lang)/(t·lang+l·task+ε)`, identical to the paper, with `t_weight=1, l_weight=2` per the config (matching wtask=1, wlanguage=2).
- **Token-level reward matches eq. 5.** `mars_steg/utils/ppo_trainer.py:1269` computes `-kl_coef·KL − (penalty − KLoffset)`, with `KLoffset=0.022` from the config; task score added at the last token (line 1276).
- **Hyperparameters match Tables 2 & 3.** Arithmetic/Coin scripts (temp 1.0/1.1, k 0.2/0.075, entropy clamp 1.0/0.6, batch 256/128, KL 1e-3) and ToM config (LoRA r8/α16/dropout.05, 4-bit, lr 4e-5, init_kl 0.2, adaptive KL, whiten, ppo_epochs 10, max_new_tokens 1024) all agree.
- **Data is present/reproducible.** Coin-Flip data generator exists; the 13,309-row ExploreToM `theory_of_mind.csv` plus its generator and the nouns YAML are committed; the ToM generalization split logic (`test_train_split`, `unseen_nouns`/`unseen_object`) is implemented with a disjointness assertion.
- **Both repos are the correct author code** (encoded_rl ← McCarthy/TinyZero for Arithmetic+Coin; mars-steg ← Geodesic/MARS for ToM), matching footnote 3 and the paper's method.

### Open questions for the authors
- Which configuration (names vs objects) and which base model (GenPRM-7B vs DeepSeek-R1-Distill-Qwen-7B) produced Figure 3 and Table 4? (`tom-config-objects-not-names`, `tom-model-name-deepseek-not-genprm`)
- Can the prefill-causality (Table 1), name-intervention (Table 4), and Figure 3 plotting/manual-label scripts and labelled data be released? (`headline-eval-harnesses-missing`, `fig3-manual-label-reliability`)
- Did the reported Coin-Flip run include the 4th penalty tier? (`coin-tier4-undocumented`)
- Does the steganographic-token-replacement accuracy drop exceed an equivalent non-name edit-control baseline? (`tom-loadbearing-edit-confound`)
