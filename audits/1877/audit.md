# Code-repository audit — GLOBE (Recognition through Reasoning, NeurIPS 2025, #1877)

## 1. Summary

GLOBE is a reasoning-based image geo-localization framework: a Qwen2.5-VL-7B
LVLM is fine-tuned with GRPO using three task-specific rewards (localizability,
visual-grounding consistency, geo-localization accuracy). The released repo
`lingli1996/GLOBE` is a fork of ModelScope's **ms-swift** RL training framework;
the paper-specific contribution lives entirely in
`examples/train/grpo/globe/` (5 shell scripts plus `plugin.py`, `dataset.py`,
`eval.py`). Dependencies are pinned in `requirements_globe.txt`. The
MP16-Reason dataset and two trained GLOBE checkpoints are published on
HuggingFace and resolve (verified via the HF API).

What I did: read every file under `examples/train/grpo/globe/` and the README;
located the paper's reported metrics, rewards, ablations, and availability
statement via `paper_text.txt` and quoted from `paper.pdf`; verified all
README/availability URLs with `curl` (HTTP 200 + HF API file listings); and ran
two deterministic checks under `_audit_code/` — `check_plugin_paths.py` (do the
plugin/register paths in the train scripts exist) and `check_eval_metric.py`
(does the eval code compute the paper's distance metric). Both outputs are under
`_audit_code/out/`.

The dominant finding: the paper's *only* reported metric is distance-threshold
accuracy (% of predictions within 1/25/200/750/2500 km of the ground-truth
coordinate), obtained by geocoding the model's predicted place names through
**Microsoft Azure Maps** and computing geographic distance. The released
`eval.py` does neither — it computes a substring-based country/city *string-match*
accuracy and never queries any geocoder or computes any distance. No script in
the repo produces any number in any table of the paper.

## 2. Traceability table

All paper headline numbers are "% @ km" (distance-threshold) accuracies (Tables
2–5, Appendix). The repo's `eval.py` emits only city/country string-match
accuracy; the geocoding + distance step that converts place names to the
reported metric is absent everywhere in the repo.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 GLOBE-7B MP16-Reason-Test (17.99/62.85/73.83/86.68/92.52 @ km) | eval.py computes string-match acc only; no geocode/distance | — | — | MISSING (no distance metric) |
| Table 2 GLOBE-7B IM2GPS3K (9.84/40.18/56.19/71.45/82.38 @ km) | (none) | — | — | MISSING |
| Table 2 all baseline rows (ISNs, GeoCLIP, LVLMs…) | (none) | — | — | MISSING (no baseline code) |
| Table 3 reward-ablation rows (Loc/VGC/GA configs, % @ km) | train_*.sh configs exist; eval metric absent | — | — | MISSING (metric) |
| Table 4 backbone ablation (Qwen vs InternVL3, % @ km) | (none) | — | — | MISSING |
| Table 5 distillation-dataset ablation (% @ km) | (none) | — | — | MISSING |
| OSV-5M mini-3K results (Appendix A.2.2) | eval.py lists `osv-test-mini-3k` config, no distance metric | — | — | MISSING (metric) |
| Efficiency: 0.44 examples/sec throughput (§4.2) | (none) | — | — | MISSING |
| Rgeo reward (Eq. 3): I[c]·(α·I[t]+(1−α)) | plugin.py:159-209 GeoLocAccuracyV2ORM | substring match, 1.0/0.2/0.0 | ✗ | MISMATCH (see rgeo-* finding) |
| Reward weights README (1.0,0.2,0.5)=acc,loc,vm | train_all_rewards.sh:13 `1 0.5 0.2` | acc=1,loc=0.5,vm=0.2 | ✗ | MISMATCH (see reward-weight finding) |

## 3. Findings

## missing

```yaml finding
id: eval-distance-metric-missing
category: missing
topic: "result traceability / evaluation metric"
title: "No code computes the paper's distance-threshold (% @ km) metric"
severity: high
confidence: high
status: finding
file: examples/train/grpo/globe/eval.py
line_start: 152
line_end: 163
quote: |
  def match_location(pred, ground_truth, match_city_or_country_threshold=0.7):
      len1 = len(pred)
      len2 = len(ground_truth)
      pred = pred.lower()
      ground_truth = ground_truth.lower()
      if len1 == 0:
          return 0
      if (pred in ground_truth and len1 / len2 >= match_city_or_country_threshold) or (
          ground_truth in pred and len2 / len1 >= match_city_or_country_threshold
      ):
          return 1
      return 0
claim: "eval.py's only accuracy computation is a case-insensitive substring match between predicted and ground-truth city/country strings; it reports city-accuracy and country-accuracy, and never geocodes predictions or computes geographic distance."
concern: "Every number in the paper is a distance-threshold accuracy (% within 1/25/200/750/2500 km) obtained by geocoding predicted place names via Azure Maps and computing distance to the ground-truth coordinate (paper §4.1); none of those numbers can be produced by the released code, so no table in the paper is reproducible from this repo."
resolution: "Authors: please release the Azure-Maps geocoding + haversine-distance evaluation script (and the place-name→coordinate cache) that produces the % @ km tables, for both GLOBE and the baselines."
cross_refs: ["azure-geocode-step-missing", "baseline-eval-code-missing"]
check_script: _audit_code/check_eval_metric.py
paper_ref: "§4.1 Evaluation Metrics; Tables 2-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: azure-geocode-step-missing
category: missing
topic: "evaluation pipeline"
title: "Azure Maps place-name→coordinate geocoding step absent from repo"
severity: high
confidence: high
status: finding
file: examples/train/grpo/globe/eval.py
line_start: 224
line_end: 235
quote: |
        if "LAT" not in fieldnames:
            fieldnames.append("LAT")
        if "LON" not in fieldnames:
            fieldnames.append("LON")
        
        if "IMG_ID" not in fieldnames:
            fieldnames.append("IMG_ID")

        if "pred_LAT" not in fieldnames:
            fieldnames.append("pred_LAT")
        if "pred_LON" not in fieldnames:
            fieldnames.append("pred_LON")
claim: "eval.py adds `pred_LAT`/`pred_LON` output columns but no code in the repo ever populates them; there is no Azure Maps query, geocoder call, or coordinate lookup anywhere in the GLOBE code (verified by grep: 0 occurrences of azure/geocode/haversine/great_circle/geopy)."
concern: "The paper states predicted country+city are concatenated and sent to Microsoft Azure Maps to obtain a representative GPS coordinate for distance evaluation; that conversion is the bridge between the model's text output and every reported metric, and it is entirely missing — the `pred_LAT/pred_LON` columns are declared but left empty."
resolution: "Authors: provide the geocoding code (or coordinate lookup table) that fills pred_LAT/pred_LON from predicted place names; clarify how Azure-Maps nondeterminism/region-center choices were handled for reproducibility."
cross_refs: ["eval-distance-metric-missing"]
check_script: _audit_code/check_eval_metric.py
paper_ref: "§4.1 'we concatenate the predicted city and country ... query Microsoft Azure Maps'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baseline-eval-code-missing
category: missing
topic: "baselines"
title: "No code for any of the baselines compared in Tables 2-5"
severity: medium
confidence: high
status: finding
file: examples/train/grpo/globe/eval.py
line_start: 37
line_end: 50
quote: |
  dataset_info = {
      "img2gps3k": {
          "prefix": "data/im2gps3ktest",
          "gt_file": "data/img2gps3k_gt.csv"
      },
      "mp16-reason-test-12k": {
          "prefix": "data/eval_images/",
          "gt_file": "data/mp16-reason-test-12k.csv"
      },
      "osv-test-mini-3k": {
          "prefix": "data/osv-test-mini-3k",
          "gt_file": "data/test_mini.csv"
      }
  }
claim: "The repo contains only an inference/eval harness for the authors' own deployed GLOBE model (via a vLLM OpenAI endpoint); there is no code to run or evaluate the ~15 baselines (ISNs, GeoCLIP, Translocator, PIGEOTTO, G3, GeoReasoner, the LVLM family, etc.) reported in Tables 2-5."
concern: "Without baseline code run under the same metric/split, the headline 'state-of-the-art' comparison cannot be reproduced or checked from the repo."
resolution: "Authors: release (or cite exact commands/configs for) the baseline evaluation runs, especially for the open-source models reported under the same % @ km protocol."
cross_refs: ["eval-distance-metric-missing"]
paper_ref: "Tables 2-5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: rm-training-dataset-missing
category: missing
topic: "data availability"
title: "Reward-model training dataset (geo_loc_rm_20w, 200K) not published"
severity: medium
confidence: high
status: finding
file: examples/train/grpo/globe/dataset.py
line_start: 22
line_end: 27
quote: |
  register_dataset(
      DatasetMeta(
          ms_dataset_id='geo_loc_rm_20w',
          hf_dataset_id='geo_loc_rm_20w',
          preprocess_func=GeoLocRMmodelPreprocessor(),
          tags=['geo', 'vision', 'reward']))
claim: "train_rm.sh trains the localizability reward model on `data/geo_loc_rm_20w` (README: 'geoloc_rm_20w', LoRA r=16); the dataset is registered with hf_dataset_id='geo_loc_rm_20w' but the only dataset published under the globe-project HF org is MP16-Reason (verified via HF API; globe-project/geo_loc_rm_20w returns 401/not-found)."
concern: "The localizability reward (Rloc, Eq. 1) is a trained component contributing to GLOBE's reported gains, but the data needed to retrain its reward model is neither in the repo nor on HuggingFace, blocking reproduction of the full reward pipeline."
resolution: "Authors: publish geo_loc_rm_20w (or document how it is derived from MP16-Reason) and the trained localizability reward-model weights."
cross_refs: ["locatability-hardcoded-url"]
check_script: _audit_code/check_eval_metric.py
paper_ref: "§3.2 Localizability Reward; README 'Reward Model Training'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: train-scripts-broken-plugin-path
category: bug
topic: "runnability"
title: "All train scripts reference examples/train/grpo/geo/ which does not exist"
severity: high
confidence: high
status: finding
file: examples/train/grpo/globe/train_all_rewards.sh
line_start: 11
line_end: 12
quote: |
    --external_plugins examples/train/grpo/geo/plugin.py \
    --custom_register_path examples/train/grpo/geo/dataset.py \
claim: "Every GRPO/SFT/RM training script passes `--external_plugins examples/train/grpo/geo/plugin.py` and `--custom_register_path examples/train/grpo/geo/dataset.py`, but no `geo/` directory exists — the actual plugin.py/dataset.py live under `examples/train/grpo/globe/` (verified by _audit_code/check_plugin_paths.py: all 8 referenced paths exist=False)."
concern: "Running any training script as documented fails immediately: the reward functions (globe_accuracy, globe_locatablity, globe_visual_match) and the dataset preprocessor are never registered, so swift cannot resolve --reward_funcs or the dataset; the repo's headline training command is non-functional out of the box."
resolution: "Authors: change all `examples/train/grpo/geo/` references to `examples/train/grpo/globe/` (or rename the directory) so the documented commands run."
cross_refs: []
check_script: _audit_code/check_plugin_paths.py
paper_ref: "README Training section"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: locatability-hardcoded-url
category: bug
topic: "reward computation"
title: "Localizability reward hardcoded to an unreachable private vLLM URL"
severity: medium
confidence: high
status: finding
file: examples/train/grpo/globe/plugin.py
line_start: 98
line_end: 101
quote: |
        # Locatablity reward model is servered remote by vLLM
        # Place your reward model url here
        reward_url = "http://29.163.178.251:8083/v1"
        openai_api_key = "EMPTY"
claim: "GeoLocatablityORM (the Rloc reward) sends every completion to a hardcoded internal IP (29.163.178.251:8083); on any failure the except branch sets reward = 0.0 and continues."
concern: "Without the authors' private reward-model server (and its weights, which are not released — see rm-training-dataset-missing) this reward silently returns 0 for every sample, so 'train_all_rewards.sh' does not actually reproduce the Loc-reward configuration; the hardcoded address makes the headline training run non-portable."
resolution: "Authors: expose the reward URL as a CLI/env argument and release the localizability reward-model weights, or document how to stand the server up."
cross_refs: ["rm-training-dataset-missing"]
paper_ref: "§3.2 Localizability Reward"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: rgeo-reward-substring-vs-exact
category: difference
topic: "reward formulation"
title: "Geo-accuracy reward uses substring match + 0.2 partial, not Eq. 3's exact-match indicators"
severity: low
confidence: high
status: finding
file: examples/train/grpo/globe/plugin.py
line_start: 201
line_end: 208
quote: |
            if match_location(city, gt_city) and match_location(country, gt_country):
                reward = 1.0
            elif match_location(country, gt_country):
                reward = 0.2
            else:
                reward = 0.0

            rewards.append(reward)
claim: "GeoLocAccuracyV2ORM grants 1.0 (city+country), 0.2 (country only), or 0.0, where `match_location` is a 0.7-ratio substring comparison; the paper's Eq. 3 uses exact-match indicators I[c=c],I[t=t] with a partial reward of (1−α)."
concern: "The implemented reward differs from the paper's stated formula: matching is fuzzy substring rather than exact, and the country-only partial reward is fixed at 0.2 (implying α=0.8) rather than an exposed weight α; both are individually valid reward designs but do not match the equation as written."
resolution: "Authors: confirm α=0.8 and that fuzzy substring matching (threshold 0.7) was used during training, and reconcile Eq. 3 with the code."
cross_refs: []
paper_ref: "Eq. (3), §3.2 Geo-localization Accuracy Reward"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: reward-weight-order-mismatch
category: difference
topic: "reward weights"
title: "README reward weights (loc=0.2, vm=0.5) disagree with train script (loc=0.5, vm=0.2)"
severity: low
confidence: high
status: finding
file: examples/train/grpo/globe/train_all_rewards.sh
line_start: 13
line_end: 15
quote: |
    --reward_funcs globe_accuracy globe_locatablity globe_visual_match external_math_format \
    --reward_weights 1 0.5 0.2 1 \
    --use_vllm true \
claim: "train_all_rewards.sh assigns weights accuracy=1, locatability=0.5, visual_match=0.2; the README's 'All Reward Functions' section lists weights 1.0, 0.2, 0.5 for accuracy, locatability, visual_match respectively (locatability and visual_match swapped)."
concern: "The two documented sources disagree on which auxiliary reward is weighted more heavily; readers cannot tell which configuration produced the reported results."
resolution: "Authors: state the exact reward weights used for the reported all-rewards model and make README and script consistent."
cross_refs: []
paper_ref: "README 'Using All Reward Functions'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No standalone methodology finding is filed. The procedure the code *actually*
implements (GRPO on Qwen2.5-VL with the three rewards) is methodologically
reasonable as written; the evaluation-validity problem is owned by
`eval-distance-metric-missing` / `azure-geocode-step-missing` (the metric code is
absent, not present-but-invalid). Topic scan:

- Data splitting / sample independence: N/A as a leakage check — train
  (MP16-Reason-train) and test (MP16-Reason-test) are released as separate
  splits of MP-16; the paper itself flags test–train overlap for some baselines
  ("Underlined results indicate test–train overlap", Table 2 note). No split
  code is in the repo, but the splits are provided as data, so this is a data
  artefact rather than a code defect.
- Pretraining contamination: Qwen2.5-VL-7B is a pretrained backbone; the paper
  does not bound overlap between Qwen pretraining and the test set. This is a
  general LVLM-evaluation caveat, not specific to this repo; noted, no finding.
- Tuning touching test set: not observed — eval is a separate offline script.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 4          | high         | No code computes the paper's distance metric; no geocoding, baselines, or RM data |
| bug         | 2          | high         | Train scripts point to a non-existent `geo/` dir; Rloc URL hardcoded   |
| difference  | 2          | low          | Rgeo reward and README weights differ from paper/script               |
| methodology | 0          | -            | Implemented training procedure is sound; eval issues owned by `missing` |

## 5. Closing lists

### Top take-aways (ranked by severity × confidence)
1. **[missing] eval-distance-metric-missing** — the released `eval.py` computes
   only string-match country/city accuracy; the paper's only metric is
   distance-threshold (% @ km) accuracy. No paper table is reproducible.
2. **[missing] azure-geocode-step-missing** — the Azure-Maps place-name→coordinate
   geocoding that bridges text predictions to the distance metric is entirely
   absent (`pred_LAT/pred_LON` columns declared but never filled).
3. **[bug] train-scripts-broken-plugin-path** — all 5 training scripts reference
   `examples/train/grpo/geo/` which does not exist; documented training commands
   fail immediately (deterministically verified).
4. **[missing] baseline-eval-code-missing** — no code for any of the ~15
   baselines in the SOTA comparison tables.
5. **[bug] locatability-hardcoded-url** — Rloc reward hardcoded to a private IP
   and silently returns 0 on failure; reward server/weights not released.
6. **[missing] rm-training-dataset-missing** — the 200K reward-model training set
   (geo_loc_rm_20w) is neither in the repo nor on HuggingFace.

### Items that genuinely look fine
- Dependencies are pinned (`requirements_globe.txt`: trl/ms_swift/transformers/vllm versions).
- MP16-Reason dataset and both GLOBE checkpoints (Qwen2.5VL-7B, InternVL3-8B)
  are published and resolve on HuggingFace (verified via HF API file listings).
- The GRPO training configuration in the scripts is internally coherent and
  matches the paper's stated hyperparameters (lr 1e-6, 1 epoch, num_generations
  16, temperature 1.0, beta 0.001).
- The accuracy reward's solution parsing (`sol.split("\t")` → country, city) is
  consistent with the dataset preprocessor that writes `solution = "country\tcity"`.

### Open questions for the authors
- How were Azure-Maps geocoding results cached/made deterministic for the
  reported % @ km numbers, and can that script + cache be released?
- What exact reward weights and α value produced the all-rewards GLOBE model
  (README vs train script disagree; Eq. 3 vs code disagree)?
- Will the localizability reward-model weights and its training data be released
  so the full three-reward configuration is reproducible?
