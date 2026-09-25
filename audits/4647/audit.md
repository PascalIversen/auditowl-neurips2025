# Code audit — Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation (NeurIPS 2025, #4647)

## 1. Summary

The repository (`github.com/MrZihan/Dynam3D`, cloned at commit `ca7295e`) is a fork of the
g3D-LF / ETPNav / VLN-CE Habitat codebase. It contains three parts: `Dynam3D_Pretrain/`
(large-scale 3D-language pretraining of the representation model, ~2.3k-line trainer
implementing the merging-discriminator / distillation / subspace-contrastive losses of
Eqs. 4-7), `Dynam3D_VLN/` (the navigation policy `Policy_Dynam3D_VLN` wrapping
LLaVA-Phi-3-mini plus the imitation-learning trainer that runs episodes in the Habitat
simulator and computes SR/OSR/SPL/NE/NDTW), and `discrete_to_CE/` (scripts that convert
discrete VLN datasets — REVERIE, NavRAG, ScaleVLN — into continuous-environment format).

The paper is an empirical SOTA paper whose headline numbers (Tables 1-7) come from running
navigation episodes in the simulator with trained checkpoints and aggregating per-episode
metrics. I read the paper (PDF + text extraction), the README, both run scripts
(`scripts/main.bash`, `run_3dff/3dff.bash`), the eval/inference rollout
(`vlnce_baselines/ss_trainer_Dynam3D.py`), the metric definitions
(`habitat_extensions/measures.py`), the eval driver (`vlnce_baselines/common/base_il_trainer.py`),
the VLN policy (`models/Policy_Dynam3D_VLN.py`), and the pretraining trainer
(`src_3dff/ss_trainer_3DFF.py`). I ran one deterministic coverage check
(`_audit_code/check_artifacts_and_experiment_coverage.py`, output
`_audit_code/out/experiment_coverage.csv`) over the whole tree and fetched the cited
HuggingFace dataset page.

Main conclusions: the *evaluation procedure that is implemented is methodologically sound*
(canonical R2R-CE protocol: geodesic distance to goal, 3 m success threshold, standard SPL/NDTW),
so there are no leakage / metric-fit / baseline-fairness findings. The defects are all about
**completeness / reproducibility**: the navigation checkpoints that produce every headline number
are not in the repo (and the README itself marks them unreleased), and the code for several
reported experiments (REVERIE-CE / NavRAG-CE evaluation, pre-exploration, lifelong memory, the
Table 7 SLAM/depth-noise robustness study, the Table 6 ablation switches, and the real-world
robot protocol) is absent from the released code.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 R2R-CE Val/Test SR/OSR/SPL/NE (Dynam3D) | `Dynam3D_VLN/vlnce_baselines/ss_trainer_Dynam3D.py:725-748` computes per-episode metrics; `:391-396` averages them | — (needs `ckpt.iter*.pth`, not in repo) | unverifiable | MISSING checkpoint (see `missing-vln-checkpoints`) |
| Table 2 REVERIE-CE Val SR/OSR/SPL/NE (Dynam3D, NaVid, g3D-LF) | only `discrete_to_CE/discrete_to_CE_reverie_*.py` (data conversion); no eval wiring | — | unverifiable | MISSING eval (see `missing-reverie-navrag-eval`) |
| Table 2 NavRAG-CE Val SR/OSR/SPL/NE | only `discrete_to_CE/discrete_to_CE_navrag_*.py` (data conversion); no eval wiring | — | unverifiable | MISSING eval (see `missing-reverie-navrag-eval`) |
| Table 3 Pre-exploration / Lifelong-Memory rows | (none) | — | — | MISSING (see `missing-preexplore-lifelong`) |
| Table 4 / Table 5 real-world static & dynamic SR/OSR/NE | (none; only `demos/*.mp4`) | — | — | MISSING protocol (see `missing-realworld-protocol`) |
| Table 6 ablation rows (single-view, w/o instance, w/o zone, w/o subspace) | no config/model switch in `Dynam3D_VLN/` (verified by coverage check) | — | — | MISSING (see `missing-ablation-toggles`) |
| Table 7 robustness w/ simulated SLAM noise + depth noise | (none; no noise-injection code) | — | — | MISSING (see `missing-noise-robustness-code`) |
| §4.5 step-time breakdown (455 ms train / 649 ms infer) | (none; no timing/profiling harness) | — | — | MISSING (timing not instrumented) |
| Eqs. 4-7 pretraining losses (merging discriminator, distillation, subspace contrastive) | `Dynam3D_Pretrain/src_3dff/ss_trainer_3DFF.py:841-1064` | code present | plausibly | Present (not numerically verifiable; needs data+compute) |
| R2R-CE success criterion (3 m, geodesic) | `Dynam3D_VLN/habitat_extensions/measures.py:52-58`, `ss_trainer_Dynam3D.py:738` | matches standard R2R-CE | ✓ | Verified (looks fine) |

Note on "Computed value" being blank: every navigation number requires running the Habitat
simulator on multi-GB scene datasets with a trained checkpoint for tens of GPU-hours; none of
these artefacts ship in the repo, so values cannot be recomputed here. The traceability question
that *can* be answered deterministically — is the producing code present, and are the required
weights/wiring present — is answered in the findings.

## 3. Findings

## missing

```yaml finding
id: missing-vln-checkpoints
category: missing
topic: "result traceability / released weights"
title: "VLN checkpoints producing all headline navigation numbers are not released"
severity: high
confidence: high
status: finding
file: README.md
line_start: 16
line_end: 16
quote: |
  * [ ] Release the checkpoints of vision-language navigation.
claim: "The navigation eval/inference scripts load a trained policy from `data/logs/checkpoints/release/ckpt.iter8000.pth` (scripts/main.bash:31) / `ckpt.iter12000.pth` (:43), but no .pth/.pt/.ckpt weight file exists anywhere in the repo (coverage check: released_weight_files_in_repo=0), and the README TODO list explicitly marks the VLN checkpoints as not yet released."
concern: "Every headline number in Tables 1-7 is produced by running the policy in the simulator; without the trained checkpoint none of the reported R2R-CE / REVERIE-CE / NavRAG-CE results can be reproduced, only retrained from scratch at large GPU cost."
resolution: "Authors: release the Dynam3D-VLN checkpoint(s) used for Tables 1-3 (e.g. on the existing HuggingFace repo), or state the exact training command, seed, and compute needed to reproduce them."
cross_refs: ["missing-reverie-navrag-eval", "§4.1"]
check_script: _audit_code/check_artifacts_and_experiment_coverage.py
paper_ref: "Tables 1-7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-reverie-navrag-eval
category: missing
topic: "result traceability / evaluation coverage"
title: "No evaluation wiring for REVERIE-CE or NavRAG-CE; only data-conversion scripts"
severity: high
confidence: high
status: finding
file: Dynam3D_VLN/scripts/r2r_vlnce.yaml
line_start: 1
line_end: 5
quote: |
  ENVIRONMENT:
    MAX_EPISODE_STEPS: 5000

  SIMULATOR:
    ACTION_SPACE_CONFIG: v0
claim: "The only task config / run script shipped for the VLN model is R2R-CE (`scripts/r2r_vlnce.yaml`, `scripts/iter_train.yaml` with `task_type: r2r`); REVERIE/NavRAG appear in the repo only inside `discrete_to_CE/` data-conversion scripts (which contain no metric or eval code) and the README. The coverage check finds 0 REVERIE/NavRAG references outside conversion scripts and the README."
concern: "Table 2 (REVERIE-CE, NavRAG-CE) and the REVERIE-CE column of Table 3 are headline results, but the repo provides no config, dataset wiring, or evaluation entry point to reproduce them; the eval metric in ss_trainer_Dynam3D.py is hardcoded to the R2R success rule."
resolution: "Authors: release the REVERIE-CE and NavRAG-CE task configs, ground-truth path files, and the eval command used for Table 2 / Table 3."
cross_refs: ["missing-vln-checkpoints", "§4.1", "§4.2"]
check_script: _audit_code/check_artifacts_and_experiment_coverage.py
paper_ref: "Table 2; Table 3 REVERIE-CE columns"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-preexplore-lifelong
category: missing
topic: "result traceability / experiment coverage"
title: "Pre-exploration and Lifelong-Memory experiments (Table 3) have no code"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  Table 3: Evaluation of VLN for Pre-exploration and Lifelong Memory. Pre-exploration allows agents to scan and encode environmental representations before evaluation, while Lifelong Memory enables agents to retain the environmental representations of previous episodes for subsequent episodes.
claim: "The paper reports a full Table 3 (Pre-exploration: scan/encode the whole scene before evaluation; Lifelong Memory: group episodes by scene and reuse 3D representations across episodes), but the coverage check finds no code implementing pre-exploration or lifelong memory (preexplore_or_lifelong_code=0); the eval rollout resets the representation per episode (`policy_net.feature_fields.reset(batch_size)`)."
concern: "The +5-8% pre-exploration and +2.7-4.9% lifelong-memory gains in Table 3 cannot be reproduced because the protocol that produces them (pre-scan, cross-episode memory retention, scene grouping) is absent from the released code."
resolution: "Authors: release the pre-exploration scan/encode script and the scene-grouped lifelong-memory evaluation harness used for Table 3."
cross_refs: ["§4.2"]
check_script: _audit_code/check_artifacts_and_experiment_coverage.py
paper_ref: "Table 3, §4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-ablation-toggles
category: missing
topic: "ablations"
title: "Ablation Study (Table 6) component switches are not present in the released model/config"
severity: medium
confidence: high
status: finding
file: Dynam3D_VLN/scripts/iter_train.yaml
line_start: 50
line_end: 61
quote: |
  MODEL:
    task_type: r2r

    policy_name: Policy_Dynam3D_VLN
    NUM_ANGLES: 12
    pretrained_path: pretrained/r2r_ce/mlm.sap_habitat_depth/ckpts/model_step_100000.pt
    fix_lang_embedding: False
    fix_pano_embedding: False
    use_depth_embedding: True
    use_sprels: True
    merge_ghost: True
    consume_ghost: True
claim: "Table 6 ablates Single-View vs Rendered-Pano patches, presence of Instance tokens, presence of Zone tokens, and the Subspace-Alignment loss, but the released VLN config/model exposes no switch for any of these (coverage check vln_ablation_toggles=0 after excluding an unrelated `single_view` comment in graph_utils.py); `Policy_Dynam3D_VLN` always assembles the full patch+instance+zone input."
concern: "The five ablation rows of Table 6 (used to argue Patch>Instance>Zone importance and the value of Subspace Alignment) cannot be reproduced without the configuration toggles that disable each component."
resolution: "Authors: release the ablation configs / flags that turn off instance tokens, zone tokens, panoramic rendering, and the subspace-alignment loss."
cross_refs: ["§4.6"]
check_script: _audit_code/check_artifacts_and_experiment_coverage.py
paper_ref: "Table 6, §4.6"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-noise-robustness-code
category: missing
topic: "result traceability / experiment coverage"
title: "SLAM-noise / depth-noise robustness study (Table 7) is not implemented in the repo"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  To simulate SLAM noise, we introduce localization noise, uniformly sampled from -5 cm to +5 cm, and orientation noise, uniformly sampled from -3 degrees to +3 degrees, to the agent's position at each simulation step. To simulate depth camera noise, we add random noise to the depth maps acquired from the simulator.
claim: "Table 7 reports navigation robustness under simulated SLAM-pose noise and depth-map noise injected into the simulator, but no code applies such localization/orientation/depth noise (coverage check slam_or_depth_noise_injection_code=0); the only `noise` references in the model are unrelated depth-zero filtering in feature_fields.py."
concern: "The Table 7 robustness numbers cannot be reproduced because the noise-injection procedure (a purely simulator-side, reproducible-in-principle manipulation) is absent from the released code."
resolution: "Authors: release the noise-injection code (or config flags) used to produce Table 7."
cross_refs: ["§4.6"]
check_script: _audit_code/check_artifacts_and_experiment_coverage.py
paper_ref: "Table 7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-realworld-protocol
category: missing
topic: "result traceability / real-world experiments"
title: "Real-world robot experiments (Tables 4-5) have no released code/protocol"
severity: low
confidence: medium
status: finding
file: paper.pdf
quote: |
  As shown in Tables 4, 5 and Figure 4, we evaluate our Dynam3D on both real-world static and dynamic environments using the Hello Robot Stretch 3. Each setting includes 20 test cases, and navigation is deemed successful if the robot stops within 1 meter of the target.
claim: "Tables 4-5 report real-world navigation on a Hello Robot Stretch 3 (20 cases each, 1 m success), but the repo contains no robot-deployment code (coverage check realworld_robot_code=0); only three demo videos under demos/."
concern: "The real-world numbers cannot be reproduced from the repo; this is partly excused by the hardware dependency, but the deployment/control protocol and the success-judgement procedure are not documented in the code either."
resolution: "Authors: provide the robot-side deployment code or a written protocol (sensor pipeline, success-judgement procedure, the 20 test cases) for Tables 4-5; confirm these were run off-repo."
cross_refs: ["§4.3"]
check_script: _audit_code/check_artifacts_and_experiment_coverage.py
paper_ref: "Tables 4, 5, §4.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-relative-ckpt-load
category: bug
topic: "reproducibility / hardcoded paths"
title: "Policy hardcodes torch.load(\"dynam3d.pth\") relative to CWD; convert_ckpt also CWD-relative"
severity: low
confidence: high
status: finding
file: Dynam3D_VLN/vlnce_baselines/models/Policy_Dynam3D_VLN.py
line_start: 80
line_end: 80
quote: |
        self.feature_fields.load_state_dict(torch.load("dynam3d.pth"),strict=True)
claim: "The VLN policy loads the pretrained representation weights from the bare relative filename `dynam3d.pth` with `strict=True`, while the run scripts launch `run.py` from the `Dynam3D_VLN/` directory; `convert_ckpt.py` likewise reads `ckpt.iter100000.pth` and writes `dynam3d.pth` in the CWD. The README never states where `dynam3d.pth` must be placed."
concern: "Initialization will raise FileNotFoundError unless `dynam3d.pth` happens to sit in the launch directory, and `strict=True` means any key mismatch from the convert step aborts loading; the path is not configurable, hindering a clean reproduction run."
resolution: "Make the pretrained-weights path a config/CLI option and document the expected location, instead of a CWD-relative literal."
cross_refs: ["missing-vln-checkpoints"]
paper_ref: "§3.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: duplicated-nested-package-tree
category: difference
topic: "repository hygiene / packaging"
title: "Repo ships a duplicated nested vlnce_baselines/vlnce_baselines/ source tree"
severity: low
confidence: high
status: finding
file: Dynam3D_VLN/vlnce_baselines/vlnce_baselines/ss_trainer_Dynam3D.py
line_start: 83
line_end: 83
quote: |
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
claim: "The VLN package contains a full second copy of itself at `Dynam3D_VLN/vlnce_baselines/vlnce_baselines/` (entire trainer/models/config tree duplicated), in addition to the canonical `Dynam3D_VLN/vlnce_baselines/`. The two copies are byte-similar but separate files."
concern: "Two coexisting copies of result-producing code create ambiguity about which trainer/model actually ran the experiments and risk silent divergence; this is a packaging artefact rather than a logic defect, so it does not change any number, but it should be removed for a clean reproduction."
resolution: "Authors: delete the inner `vlnce_baselines/vlnce_baselines/` duplicate and confirm which copy was used for the reported results."
cross_refs: []
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The implemented navigation evaluation is the canonical R2R-CE
protocol and is sound (see "Items that genuinely look fine"). Topics assessed:

- **Data splitting / held-out test set**: standard VLN-CE splits (`train` / `val_unseen` /
  `test`); evaluation uses `val_unseen` / `test` via `EVAL.SPLIT` / `INFERENCE.SPLIT`. No
  in-sample fitting. No leakage observed in the eval path.
- **Sample independence**: VLN episodes are drawn from the standard benchmark splits; scene-
  disjoint unseen splits are the field convention and the config uses them. No finding.
- **Target/shortcut leakage**: N/A — the model predicts low-level actions from RGB-D
  observations; no tabular features that could leak the label.
- **Pretraining contamination**: The 3D representation model and LLaVA-Phi-3-mini are
  pretrained; the paper uses standard VLN-CE unseen splits whose scenes are held out from
  training. Not separately bounded, but this is the field norm; no concrete contamination
  evidence in the code. No finding.
- **Hyperparameter tuning / test-set touching**: the eval rollout computes metrics only;
  it does not select checkpoints or hyperparameters on the eval split. `main.bash`
  hardcodes which iteration checkpoint to evaluate (`iter8000`/`iter12000`), which could
  reflect checkpoint selection, but there is no in-code evidence of selection-on-test — filed
  as an open question, not a finding.
- **Baselines**: NaVid and g3D-LF are retrained on the authors' training set for the REVERIE/
  NavRAG comparison (paper §4.1); that retraining code is not in this repo (covered by
  `missing-reverie-navrag-eval`). No evidence of asymmetric tuning in-repo.
- **Statistical integrity**: no significance tests, CIs, or error bars are reported; nothing
  to check for arithmetic consistency.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 6          | high         | VLN checkpoints + REVERIE/NavRAG eval + pre-explore/lifelong/ablation/noise/real-world code absent |
| bug         | 1          | low          | CWD-relative hardcoded `dynam3d.pth` load |
| difference  | 1          | low          | Duplicated nested `vlnce_baselines/vlnce_baselines/` tree |
| methodology | 0          | -            | Implemented R2R-CE evaluation is sound; no leakage/metric/baseline defect found |

### Top take-aways (≤6, ranked by severity × confidence)

1. **[missing] `missing-vln-checkpoints`** (high/high): the trained navigation checkpoints
   behind every headline number (Tables 1-7) are not in the repo and the README marks them
   unreleased — Tables 1-3 are not reproducible without retraining.
2. **[missing] `missing-reverie-navrag-eval`** (high/high): no config or eval wiring for
   REVERIE-CE / NavRAG-CE (only data-conversion scripts); Table 2 and the REVERIE column of
   Table 3 cannot be reproduced.
3. **[missing] `missing-preexplore-lifelong`** (medium/high): no code implements the
   pre-exploration scan or the cross-episode lifelong-memory protocol of Table 3.
4. **[missing] `missing-ablation-toggles`** (medium/high): no switches exist to disable
   instance/zone/panoramic/subspace components, so Table 6 ablations are not reproducible.
5. **[missing] `missing-noise-robustness-code`** (low/high): the SLAM/depth-noise injection
   for the Table 7 robustness study is absent.
6. **[bug] `hardcoded-relative-ckpt-load`** (low/high): the policy loads `dynam3d.pth` from a
   CWD-relative literal with `strict=True`, blocking a clean run.

### Items that genuinely look fine

- **R2R-CE success metric**: `success = distances[-1] <= 3.` with `distances` being geodesic
  distance-to-nearest-goal (`measures.py:52-58`) is the canonical R2R-CE 3 m success rule;
  SPL, oracle-success, NDTW, SDTW are computed in the standard way (`ss_trainer_Dynam3D.py:738-746`).
- **Eval split handling**: `eval()` sets the split from `EVAL.SPLIT` (default `val_unseen`)
  and only computes metrics — no test-set-based model selection in the eval path
  (`base_il_trainer.py:774-853`).
- **Dependency specification**: `environment.yml` is a fully version-pinned conda export
  (312 lines), so the environment is rebuildable (modulo the external CUDA-only deps
  `torch_kdtree`, `tinycudann` documented in the README).
- **Pretraining losses present**: the merging-discriminator / distillation / subspace-
  contrastive losses (Eqs. 4-7) are implemented in `src_3dff/ss_trainer_3DFF.py` rather than
  merely described.

### Open questions for the authors

- Does the HuggingFace repo (`MrZihanWang/Dynam3D`) actually contain the pretraining
  checkpoint `dynam3d.pth` / `ckpt.iter100000.pth`? The dataset viewer errors out, so I could
  confirm only that 164 GB of navigation data is present, not the weights. If the VLN
  checkpoints are there despite the unchecked README TODO, that would downgrade
  `missing-vln-checkpoints`.
- `scripts/main.bash` hardcodes evaluating `ckpt.iter8000.pth` (eval) and `ckpt.iter12000.pth`
  (inference). How were these specific iterations chosen — fixed in advance, or selected by
  val/test performance? (Selection on the reported split would be a methodology concern; no
  in-code evidence either way.)
