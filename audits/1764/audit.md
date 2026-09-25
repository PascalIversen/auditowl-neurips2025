# Audit — AlphaZero Neural Scaling and Zipf's Law: a Tale of Board Games and Power Laws (paper 1764)

## 1. Summary

The paper is an **empirical analysis** of AlphaZero agents (OpenSpiel) on four board
games (Connect Four, Pentago, Oware, Checkers). Its claims are descriptive: (1) board
states played by AlphaZero follow Zipf's law; (2) the Zipf exponent correlates with the
size-scaling exponent (modulated via inference temperature); (3) value loss increases
with state rank (states optimised in descending frequency order); (4) inverse scaling in
Oware/Checkers coincides with a late-game frequency anomaly. There is no train/test
classifier, no held-out predictive benchmark, and no proposed model that is itself
evaluated — the artefacts are figures of frequencies, losses and Elo curves.

The repository (`github.com/OrenNeumann/alphazero_zipfs_law`) is a **plotting/analysis
pipeline**. `src/generate_all_plots.py` regenerates every figure; by default
(`load_data = True`) it reads pre-computed intermediate data from a sibling
`../plot_data/` directory, distributed as a separate GitHub "Experiment data" release
(confirmed to exist, 11 assets). With `load_data = False` it recomputes the
intermediates from raw OpenSpiel game logs and trained model checkpoints, which are
**not** in the repo (the raw data is "available on request"; models come from a separate
repo, `AlphaZero-scaling-laws`). External tools (Pascal Pons Connect-Four solver,
Bayeselo) are required for some panels and are clearly documented as third-party.

Two repos were cloned. `OrenNeumann__alphazero_zipfs_law` (commit `01dd606`, 2025-10-13)
is the canonical author repo and is more complete (it contains the `solver_loss` /
`solver_loss_from_dataset` functions and the error-bar plotting variants);
`AlphaZeroZipf__AlphaZero-Zipf-s-Law` (commit `301dc01`, 2025-05-19) is an older,
strictly-subset anonymised copy. **This audit targets the OrenNeumann repo.** All
file:line references below are to that repo.

What I did: read the paper (PDF + text) and every analysis/plotting module; mapped each
figure to the function that computes its values (Rule G table below); ran read-only
checks under `_audit_code/` (`check_artifacts.py`) verifying the missing `paths.yaml`,
the undocumented `../matches/` dependency, and that no `plot_data`/`models` are shipped
in the repo. I could not execute the pipeline (no `plot_data` release downloaded, no
OpenSpiel models, no GPU/solver), so dynamic claims are confidence-bounded accordingly.

## 2. Traceability table (Rule G)

Because this is an analysis paper, "Computed value" is the *quantity the script
produces* (a curve / exponent / loss array), not a single scalar; "Matches paper" asks
whether the producing computation is present and consistent with the described
procedure. Numeric exactness cannot be re-verified without the release data and models.

| Paper artefact | Repo location (computes values) | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 1 Zipf curves + α (Connect4, Pentago) | `zipf_curves.py:65` `plot_main_zipf_curves`; gen `_generate_zipf_curves:14`; fit `_fit_power_law:29` | freq-vs-rank + α via weighted log-log polyfit | ✓ (computes) | Verified (code present) |
| Fig. 9 / App C.1 per-game & per-agent Zipf + α | `zipf_curves.py:93` `plot_appendix_zipf_curves` | same | ✓ | Verified |
| Fig. 2 toy-model / random-game Zipf | `zipf_law_theory.py` `plot_zipf_law_theory` | plateau distribution | ✓ | Verified |
| Fig. 10 / App C.2 non-uniform-policy Zipf | `zipf_law_theory.py` `plot_appendix_theory_zipf` | toy curves | ✓ | Verified |
| Fig. 3A inference Zipf vs T | `temperature.py:37` `plot_temperature_curves` (axs[0]); gen `_generate_temperature_zipf_curves:18` | freq curves per T | ✓ | Verified |
| Fig. 3B Connect4 size-scaling Elo vs T | `temperature.py:75-110` | Elo via BayesElo from `../matches/temperature_scaling/.../matrix.npy` | data path undocumented | MISSING (matches dir) — see `matches-dir-undocumented` |
| Fig. 3C / Fig. 12 exponent correlation | `temperature.py:118-161` | (zipf_exp, elo_exp) pairs | ✓ (derived from 3A/3B) | Verified (depends on 3B data) |
| Fig. 4A Connect4 value loss (train set) | `value_loss.py:138` `connect4_loss_plots` (i=0); gen `_generate_loss_curves:19`, core `value_loss.value_loss:8` | (z−v)² sorted by rank | ✓ | Verified (code present) |
| Fig. 4B Connect4 value loss (ground truth) | `connect4_loss_plots` (i=1) loads `../plot_data/solver/loss_curves.pkl`; gen `generate_common_solver_loss_curves:64` + `solver_calc.solver_loss:94` | solver-labelled (z−v)² | ✓ | Verified |
| Fig. 4C α-β pruning CPU time vs rank | `alpha_beta_pruning_time.py` `save_pruning_time` (via `value_loss.py:191`) | geo mean/std CPU time | ✓ | Verified |
| Fig. 5A Oware/Checkers Elo size scaling | `elo_curves.py:16` `_oware_size_scaling` (`../matches/oware_base/...`), `:139` (`../matches/checkers/matrix.npy`) | Elo via BayesElo | data path undocumented | MISSING (matches dir) — see `matches-dir-undocumented` |
| Fig. 5B/5C Oware/Checkers Zipf bump | `zipf_curves.py:93` (appendix curves reused) | freq curves | ✓ | Verified |
| Fig. 6 turn-distribution anomaly | `game_turns.py` `plot_game_turns` (loads `../plot_data/turns/...`) | turn# vs rank, late-state fraction | ✓ | Verified |
| Fig. 7 Oware loss split early/late | `value_loss.py:470` `oware_value_loss`; gen `_gereate_oware_loss_curves:397`, mask `state_counter.late_turn_mask:179` (thr=40) | loss curves split by turn mask | ✓ | Verified |
| Fig. 8 large-agent loss over checkpoints | `value_loss.py:536` `oware_checkpoint_value_loss`; gen `_gereate_oware_checkpoint_loss_curves:415` | loss per checkpoint | ✓ | Verified |
| Fig. 11 / App D p(optimal) vs T | `policy_degradation.py` `plot_policy_degradation` (`../plot_data/solver/...`) | optimal-move prob | ✓ | Verified |
| Figs. 14/15 / App G board positions | `appendix/board_positions.py` `plot_oware`/`plot_checkers` | sampled states | ✓ | Verified |
| Fig. 16 / App I capture-difference freq | `appendix/capture_difference.py` `plot_capture_differences` | freq vs capture diff | ✓ | Verified |
| App J ideal-game distribution (Eqs 13–26) | analytic (no code needed) | — | N/A | N/A (closed-form derivation) |

No numbered tables and no reported statistical tests (Wilcoxon/t/etc.) appear in the
paper; error bars are std-dev across agents/seeds (computed in the loss/Elo scripts).
GRIM/SPRITE/p-value-floor checks are therefore **N/A** (no integer-mean summary stats,
no reported significance tests).

## 3. Findings

## missing

```yaml finding
id: matches-dir-undocumented
category: missing
topic: "result traceability / data availability"
title: "Elo match matrices for Figs 3B & 5A loaded from undocumented '../matches/' dir"
severity: high
confidence: high
status: finding
file: src/plotting/plot_scripts/elo_curves.py
line_start: 16
line_end: 30
quote: |
  def _oware_size_scaling() -> tuple[list[int], list[float], list[float], list[int], list[float], list[float]]:
      """ Load match matrices and calculate Elo ratings. """
      dir_name = '../matches/oware_base/'
      r = BayesElo()
      agents = PlayerNums()

      ######## load fixed-size models ########
      fixed_size_models = []
      # Enumerate self-matches models
      for i in range(6):
          for j in range(4):
              fixed_size_models.append('q_' + str(i) + '_' + str(j))
      
      for model in tqdm(fixed_size_models, desc='Loading fixed-size matches'):
          matches = np.load(dir_name + 'fixed_size/' + str(model) + '/matrix.npy')
claim: "The size-scaling Elo curves of Fig. 5A (Oware/Checkers) and Fig. 3B (Connect Four, via temperature.py:83) are computed from tournament match matrices read from a sibling directory '../matches/...'; the README documents only a '../plot_data/' release and never mentions a 'matches' folder, and _audit_code/check_artifacts.py confirms README has no 'matches' reference."
concern: "The inverse-scaling Elo curve (Fig. 5A) and the size-scaling exponents in the exponent-correlation result (Figs 3B/3C) — both headline claims — depend on a data artefact (the match matrices) that is neither shipped nor documented in the data-availability instructions, so these figures cannot be regenerated from the released material."
resolution: "Authors: please confirm whether the '../matches/' match matrices are included in the 'Experiment data' release (and under what name), and add the 'matches/' layout to the README, or provide the script + raw data needed to regenerate them."
cross_refs: ["fig3b-matches", "load-data-false-needs-models-and-paths"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Fig. 5A; Fig. 3B; README data layout"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-paths-yaml
category: missing
topic: "reproducibility / configuration"
title: "Code opens src/config/paths.yaml which is absent; only example_paths.yaml shipped"
severity: medium
confidence: high
status: finding
file: src/general/general_utils.py
line_start: 6
line_end: 15
quote: |
  def models_path() -> str:
      with open("src/config/paths.yaml", "r") as f:
          config = yaml.load(f, Loader=yaml.FullLoader)
      return config['paths']['models_dir']


  def game_path(env: str) -> str:
      with open("src/config/paths.yaml", "r") as f:
          config = yaml.load(f, Loader=yaml.FullLoader)
      return config['game_paths'][env]
claim: "models_path() and game_path() open 'src/config/paths.yaml', but the repo ships only 'src/config/example_paths.yaml' (verified by _audit_code/check_artifacts.py: paths.yaml exists=False, example_paths.yaml exists=True), and the README never mentions either file or any rename step."
concern: "Any from-scratch run (load_data=False) — and the load_data=True paths that call gather_data/zipf_curves/policy_degradation/value_loss generation — raises FileNotFoundError on src/config/paths.yaml until the user manually creates it, with no documented instruction to do so."
resolution: "Authors: either commit a default paths.yaml, or add a README step instructing users to copy example_paths.yaml to paths.yaml; confirm the intended models_dir layout."
cross_refs: ["load-data-false-needs-models-and-paths"]
check_script: _audit_code/check_artifacts.py
paper_ref: "README; src/config/"
tags: [reforms:2, bug-adjacent]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: load-data-false-needs-models-and-paths
category: missing
topic: "expected code completeness / data"
title: "From-scratch reproduction needs unshared models and raw logs ('on request' only)"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 36
line_end: 41
quote: |
  ## Generating all results from scratch

  Some experiments require importing or installing 3rd party data and software manually:
  - Almost all experiments require agent training/inference data, and the trained agents. 
      These can be generated by OpenSpiel (Apache License 2.0) with the code available [here](https://github.com/OrenNeumann/AlphaZero-scaling-laws) (Apache License 2.0).
claim: "The trained agents and raw game logs needed for load_data=False are not in this repo; the README points to a separate training repo, and the paper's data-availability statement says the full raw dataset is only 'available on request' (paper checklist Q5)."
concern: "Only the curated intermediate '../plot_data/' release lets one regenerate figures; the model checkpoints required to recompute losses/Zipf curves from scratch (and the raw logs) are not directly downloadable, so independent end-to-end reproduction is gated on author cooperation."
resolution: "This is an acknowledged limitation; authors could mitigate by archiving the trained model weights (e.g. a Zenodo DOI) rather than 'on request'. Confirm whether the weights are publicly downloadable."
cross_refs: ["missing-paths-yaml", "matches-dir-undocumented"]
paper_ref: "README 'Generating all results from scratch'; Checklist Q5 (paper.pdf)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone runtime bug was confirmed beyond the missing `paths.yaml`
(filed under `missing` as the primary owner, since the file is simply absent rather than
mis-wired). The pipeline could not be executed, so latent runtime bugs cannot be ruled
out, but none is evident from static reading. `0` findings.

## difference

```yaml finding
id: fig3b-matches
category: difference
topic: "result traceability / data layout"
title: "Temperature-scaling Elo (Fig 3B) reads '../matches/temperature_scaling' not in plot_data layout"
severity: low
confidence: high
status: finding
file: src/plotting/plot_scripts/temperature.py
line_start: 78
line_end: 91
quote: |
      for ind in sorted_t:
          n+=1
          print(f'({n}/{len(TEMPERATURES)}) Temperature: {TEMPERATURES[ind]}')
          r = BayesElo()
          agents = PlayerNums()
          matches = np.load(f'../matches/temperature_scaling/connect_four_temp_num_{ind}/matrix.npy')
          sizes = 7
          copies = 4
          for i, j in combinations(range(len(matches)), 2):
              r.add_match(i, j, p=matches[i, j])
          for i in range(sizes):
              for j in range(copies):
                  agents.add(f'q_{i}_{j}', 10000)
          elo = r.extract_elo(agents)
claim: "Fig. 3B's Elo-vs-parameters curves at each temperature are computed by BayesElo from match matrices at '../matches/temperature_scaling/...'; the path is sound and the code valid, but it points outside the documented '../plot_data/' release tree (same undocumented 'matches' tree as Fig 5A)."
concern: "A reader following the README's data instructions (which only describe plot_data zips) cannot locate this input, even though the computation itself is correct."
resolution: "Document the 'matches/' directory and its contents alongside the plot_data release; this is a documentation/layout gap rather than an analysis error."
cross_refs: ["matches-dir-undocumented"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Fig. 3B"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: repo-not-tagged-to-submission
category: difference
topic: "repository provenance"
title: "Audited commit is a moving main dated after submission, not a tagged release"
severity: low
confidence: medium
status: finding
file: README.md
line_start: 1
line_end: 3
quote: |
  # AlphaZero Neural Scaling and Zipf's Law: a Tale of Board Games and Power Laws

  We provide here all resources needed to reproduce the paper's results.
claim: "The canonical repo HEAD is commit 01dd606 dated 2025-10-13 ('update plots'); there is no git tag pinning the state to the NeurIPS 2025 submission/camera-ready, and a second anonymised copy (AlphaZeroZipf, 2025-05-19) is a strict subset, so the exact code version that produced the paper figures is ambiguous."
concern: "Without a tagged commit, a reviewer cannot be certain the audited code is the same version that generated the published figures (plots in the repo were updated post-submission)."
resolution: "Authors: tag the commit corresponding to the camera-ready figures, or state which commit produced the paper's plots."
cross_refs: []
paper_ref: "metadata.txt; git log"
tags: [forensics:git-archaeology, reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: value-label-player-perspective
category: methodology
topic: "value-loss label construction"
title: "Training value labels use player-0 return while model/solver values are current-player relative"
severity: medium
confidence: low
status: question
file: src/data_analysis/state_frequency/state_counter.py
line_start: 147
line_end: 152
quote: |
          if self.save_value:
              game_return = board.player_return(0)
              if game_return == 0:
                  self.draws += 1
              for key in keys:
                  self.values[key] = self.values.get(key, 0) + game_return
claim: "The train-set value label z for each state is the mean of board.player_return(0) (always player 0's outcome). In value_prediction.py:38 the model is queried with state.observation_tensor() (OpenSpiel returns the to-move player's perspective), and in solver_calc.py:81 the solver value is explicitly converted to player-0 perspective via (1-2*current_player). The model value and the player-0 label may therefore use opposite sign conventions on states where player 1 is to move, in Fig. 4A (and the Oware/Fig.7 train-loss curves)."
concern: "If the model's value head output is current-player-relative but z is player-0-relative, (z-v)^2 would be inflated on roughly half the states (player-1-to-move), distorting the absolute loss magnitudes (though the monotone loss-vs-rank trend, the paper's actual claim, could survive)."
resolution: "Authors: confirm the sign convention of the AlphaZero value head vs board.player_return(0); if they differ, state whether the train-loss curves (Fig 4A, Fig 7) account for it. The solver path (Fig 4B) already converts perspective, so this question concerns the train-label path only."
cross_refs: []
paper_ref: "Fig. 4A; Fig. 7; Eq. 6 (paper.pdf)"
tags: [whalen:pitfall-2]
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | Elo match matrices (Figs 3B/5A) undocumented; paths.yaml absent; from-scratch needs unshared models. |
| bug         | 0          | -            | No standalone runtime bug confirmed (pipeline not executed). |
| difference  | 2          | low          | Fig 3B data path & untagged moving-main provenance — documentation/provenance gaps, code valid. |
| methodology | 1          | medium       | Open question on value-label player perspective (low confidence). |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `matches-dir-undocumented`** (high/high): Fig 5A inverse-scaling Elo and
   Fig 3B size-scaling Elo are computed from `../matches/` match matrices that the README
   data-availability instructions never mention and that are not in the repo — these
   headline figures may not be regenerable from the released "Experiment data".
2. **[missing] `missing-paths-yaml`** (medium/high): the code unconditionally opens
   `src/config/paths.yaml`, which is absent (only `example_paths.yaml` ships) and is not
   referenced in the README, blocking the data-generation paths.
3. **[missing] `load-data-false-needs-models-and-paths`** (medium/high): from-scratch
   reproduction depends on trained-agent weights / raw logs that are only "available on
   request", limiting independent end-to-end verification.
4. **[methodology] `value-label-player-perspective`** (medium/low, filed as *question*):
   possible sign-convention mismatch between player-0 training labels and
   current-player model/solver values in the train-loss curves.
5. **[difference] `repo-not-tagged-to-submission`** (low/medium): no tag pins the audited
   code to the published figures; repo plots were updated after submission.
6. **[difference] `fig3b-matches`** (low/high): documentation gap for the Fig 3B input
   path (cross-refs #1).

### Items that genuinely look fine
- Every numbered figure traces to a concrete value-computing function (not just a
  renderer) — see the traceability table; the Zipf, value-loss, turn-distribution,
  capture-difference, and policy-degradation panels all have generating code present.
- Dependencies are specified and pinned (`pyproject.toml` + committed `poetry.lock`).
- Third-party dependencies (Pascal Pons Connect-Four solver, Bayeselo, OpenSpiel) are
  clearly documented with URLs and licenses; their non-inclusion is legitimate.
- The "Experiment data" release exists and (per its description) covers ab_pruning,
  board_positions, capture_difference, elo_curves, solver labels, turns, zipf_theory,
  temperature and value_loss — i.e. the `../plot_data/` tree for the load_data=True path.
- The ground-truth value loss (Fig 4B) correctly converts solver values to a consistent
  player-0 perspective (`solver_calc.py:81`).
- No reported statistical tests or integer-mean summary tables exist, so GRIM/SPRITE/
  p-value-floor integrity checks are structurally N/A.

### Open questions for the authors
- Is the `../matches/` directory part of the public release, and what is its layout?
  (Blocks `matches-dir-undocumented`.)
- Are the trained model weights publicly archived, or only available on request?
- What is the sign convention of the AlphaZero value head relative to
  `board.player_return(0)`, and does the train-loss path (Fig 4A/Fig 7) need a
  perspective conversion like the solver path? (`value-label-player-perspective`.)
- Which commit produced the camera-ready figures?
