# Audit — Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning (NeurIPS 2025, #2585)

## 1. Summary

This is primarily a **theoretical** sample-complexity paper. Its core contributions are
analytical (instance-dependent bounds for DR-DMDP and two a-priori-knowledge-free algorithms
for DR-AMDP); the proofs are in Appendices A–G and are out of scope for a code audit. The paper
also reports a small **empirical** supplement that is presented as validating the theory:
Section 5 / **Figure 2** (four log–log error-vs-`n` panels for Algorithm 2 and 3 under KL and
χ² uncertainty, with a claimed regression slope of **−1/2**), and **Appendix H** (Table 2 — a
baseline comparison against DR RVI Q-learning; Table 3 / Figure 3 — a 20-state/30-action
large-scale experiment with δ = 0.4).

**Re-audit context.** The original audit ran before the NeurIPS Supplemental ZIP was fetched
and therefore filed three "no code" `missing` findings (since marked supplement false-positives).
The supplement is now unpacked under `code/supplement/`, so this re-audit assesses the actual
code. The supplement contains the algorithm cores (`dr_rl_empirical_kl.py`,
`dr_rl_empirical_chi_square.py` — robust Bellman + KL/χ² `dual_opt`), the Hard-MDP model
(`model/hard_mdp_unichain.py`, Def. 5.1), the large-scale MDP (`model/large_scale.py`), the
Figure-2 drivers (`Reduction_DMDP_parallel.py` = Alg 2, `Anchored_AMDP_parallel.py` = Alg 3),
the Appendix-H baseline (`dr_q_learning.py` + `add_baseline.py` / `dr_q_learning_experiment.py`),
the large-scale drivers (`add_large_scale_dmdp.py`, `add_large_scale_anchored.py`), and two
plotting scripts (`plot.py`, `plot_parallel.py`, log-log `np.polyfit`).

**What I did (read-only).** I read every script and re-ran the inner computation of the Figure-2
drivers on a small `n`-grid (`_audit_code/out/repro_slopes.txt`). The KL Algorithm 2 path
reproduces a log-log slope of **−0.52**, the KL Algorithm 3 path **−0.54**, and the χ² Algorithm 2
class **−0.48** — all consistent with the paper's headline −1/2 claim. So the released code does
substantively reproduce the central empirical behaviour. The remaining findings are
**reproducibility/wiring** issues, not invalidations of the result:

1. Every driver that saves results writes to a subdirectory it never creates, so `pickle.dump`
   crashes with `FileNotFoundError` before any output is written (deterministic, `bug`).
2. No released driver instantiates the χ² algorithm class, so Figure-2 panels (b) and (d) are
   not produced by the shipped entry points without a one-line edit (`difference`).
3. The Appendix-H "DR RVI Q-learning" baseline (Table 2) is run with `empirical=False`, i.e. on
   the **true** transition kernel rather than sampled transitions, while the two proposed
   methods it is compared against use empirical samples (`methodology`/question).
4. No `requirements`/README and Windows-only `data\...` paths in the plotting scripts (`missing`,
   low).

Two paper-level procedure observations carried over from the original audit (single run per
data point; δ=0.4 large-scale violates Assumption 2) remain valid `status: question` items.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2a — KL, Algorithm 2, slope ≈ −1/2 | `Reduction_DMDP_parallel.py:20-66` → `dr_rl_empirical_kl.py value_iteration` | re-ran: slope −0.52 (`out/repro_slopes.txt`) | ✓ (slope) | Verified (computation present & runs; see save-dir bug) |
| Fig. 2c — KL, Algorithm 3, slope ≈ −1/2 | `Anchored_AMDP_parallel.py:21-67` → `anchored_relative_value_iteration` | re-ran: slope −0.54 | ✓ (slope) | Verified |
| Fig. 2b — χ², Algorithm 2, slope ≈ −1/2 | `dr_rl_empirical_chi_square.py` (class present) | re-ran class manually: slope −0.48 | ✓ (slope) | Computation present, **no driver instantiates it** (see `chi2-panels-driver-missing`) |
| Fig. 2d — χ², Algorithm 3, slope ≈ −1/2 | `dr_rl_empirical_chi_square.py` (class present) | not run | — | Computation present, no driver (see `chi2-panels-driver-missing`) |
| Exact (per-`p`, full-grid) Fig-2 error numbers | drivers above, but `pickle.dump` crashes | — | — | Not saved by released code (`save-subdir-not-created`); no committed pkl |
| Table 2 — DR RVI Q-learning baseline (9 sample sizes) | `add_baseline.py:19-32` → `dr_q_learning.relative_value_iteration_q(empirical=False)` | — | — | Baseline uses TRUE kernel, not samples (`baseline-uses-true-kernel`) |
| Table 2 — Reduction / Anchored columns | same drivers as Fig 2 (KL) | (slope verified above) | partial | Computation present |
| Table 3 / Fig. 3 — large-scale 20×30, δ=0.4 | `add_large_scale_dmdp.py`, `add_large_scale_anchored.py` → `model/large_scale.py` | not run (save crashes) | — | Computation present; save-dir bug; δ=0.4 outside Assumption 2 |
| Ground-truth average reward g* | `(relative_)value_iteration` on `model.transition_map` | g*≈0.4265 (Hard MDP) | n/a | Verified present |
| Theory: Theorems 4.3–4.5 bounds | Appendices A–G (proofs) | n/a | n/a | Out of scope (proof, not code) |

Per Result-Traceability (Rule G): the *computations* that produce the Figure-2 / Appendix-H
numbers are now present and, where re-runnable, reproduce the −1/2 behaviour. The defects below
concern wiring (save dirs), completeness (χ² driver, deps), and one baseline-fairness question.

## 3. Findings

## missing

```yaml finding
id: no-deps-no-readme-no-results
category: missing
topic: "code completeness / reproducibility"
title: "Supplement ships no requirements file, no README, and no committed result data"
severity: low
confidence: high
status: finding
file: code/supplement/plot.py
line_start: 9
line_end: 12
quote: |
    filename = r"data\Anchored_AMDP\kl\.pkl"

    with open(filename, 'rb') as f:
        sample_error_data = pickle.load(f)
claim: "The supplement contains only .py source (verified by find: no requirements.txt/.cfg/.toml, no README/.md, and no committed .pkl/.png/.npy result files); the plotting scripts read result pickles from data\\Anchored_AMDP\\kl\\ that are not shipped and must be regenerated by the drivers."
concern: "There is no dependency specification or run instructions, and none of the Figure-2/Table-2/Table-3 numerical outputs are committed, so a reader must reconstruct the environment and re-run every experiment (which the save-dir bug then blocks) to obtain any reported number."
resolution: "Authors: add a requirements file (numpy/matplotlib versions), a short README mapping each script to the figure/table it produces, and ideally commit the result pickles used for the paper's figures."
cross_refs: ["save-subdir-not-created", "windows-only-data-paths"]
check_script: _audit_code/check_save_dirs.py
paper_ref: "Section 5 (Figure 2); Appendix H (Tables 2-3, Figure 3); NeurIPS Checklist Q5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: save-subdir-not-created
category: bug
topic: "result traceability / runnability"
title: "All experiment drivers crash on save: pickle.dump targets a subdir that is never created"
severity: medium
confidence: high
status: finding
file: code/supplement/Reduction_DMDP_parallel.py
line_start: 60
line_end: 65
quote: |
        os.makedirs("data", exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/Reduction_DMDP/kl/p_{p:.2f}_{current_time}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(sample_error_data, f)
claim: "The driver creates only the top-level 'data' directory but writes the pickle to the nested 'data/Reduction_DMDP/kl/' which does not exist; the same pattern repeats in Anchored_AMDP_parallel.py (data/Anchored_AMDP/kl/), add_baseline.py & dr_q_learning_experiment.py (rebuttal/baseline/, no makedirs at all), and add_large_scale_*.py (rebuttal/large_scale/, no makedirs)."
concern: "open(filename,'wb') raises FileNotFoundError after the entire (multi-hour at the paper's n up to ~1e5) computation finishes, so the released scripts cannot persist the Figure-2 / Table-2 / Table-3 results as shipped — verified by a sandbox simulation in _audit_code/check_save_dirs.py."
resolution: "Authors: create the full nested output directory (e.g. os.makedirs(os.path.dirname(filename), exist_ok=True)) in each of the six drivers, or commit the result pickles."
cross_refs: ["no-deps-no-readme-no-results"]
check_script: _audit_code/check_save_dirs.py
paper_ref: "Section 5, Figure 2; Appendix H, Tables 2-3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: windows-only-data-paths
category: bug
topic: "runnability / portability"
title: "Plotting scripts use backslash Windows paths that do not resolve on POSIX"
severity: low
confidence: high
status: finding
file: code/supplement/plot_parallel.py
line_start: 23
line_end: 23
quote: |
    data_files = glob.glob(r"data\Anchored_AMDP\kl\p_*.pkl")
claim: "plot_parallel.py (lines 23, 91) and plot.py (lines 9, 51) hardcode backslash-separated paths (data\\Anchored_AMDP\\kl\\...); on Linux/macOS the backslashes are literal filename characters, so glob matches nothing and the input/output paths are wrong."
concern: "On any non-Windows machine the plotting scripts silently find no data (empty figure) or write to a malformed filename, so the Figure-2 plot cannot be regenerated cross-platform as shipped."
resolution: "Authors: use os.path.join / forward slashes for the plot input/output paths so the scripts are portable."
cross_refs: ["no-deps-no-readme-no-results"]
paper_ref: "Section 5, Figure 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: chi2-panels-driver-missing
category: difference
topic: "evaluation consistency (paper vs code)"
title: "No released driver runs the χ² algorithm; Figure-2 panels (b) and (d) need a manual class swap"
severity: low
confidence: high
status: finding
file: code/supplement/Reduction_DMDP_parallel.py
line_start: 22
line_end: 26
quote: |
    # Initialize the empirical DR-RL algorithm
    # If using KL divergence, use DR_RL_empirical_kl
    # If using Chi-square divergence, use DR_RL_empirical_chi_square
    dr_rl_empirical = DR_RL_empirical_kl(mdp, delta, gamma)
    target_g_star = dr_rl_empirical.g_star
claim: "Every Figure-2 / large-scale driver imports DR_RL_empirical_chi_square but instantiates only DR_RL_empirical_kl (Reduction_DMDP_parallel.py:25, Anchored_AMDP_parallel.py:26, add_large_scale_dmdp.py:21, add_large_scale_anchored.py:21,41 — verified by _audit_code/check_chi_square_driver.py), so as shipped the drivers compute only the KL panels (Fig 2a, 2c); the χ² panels (Fig 2b, 2d) reported in the paper are produced only by hand-editing the class on the commented line."
concern: "Half of Figure 2 (the χ² panels) is not reproducible by running the released entry points; the comment shows the intended swap and the χ² class is present and works (re-ran it: slope −0.48), so this is a packaging/faithfulness gap, not an invalid or absent computation."
resolution: "Authors: parameterise the divergence (e.g. a CLI/argument selecting the KL vs χ² class) so each of the four Figure-2 panels can be produced from the released scripts without source edits."
cross_refs: ["save-subdir-not-created"]
check_script: _audit_code/check_chi_square_driver.py
paper_ref: "Figure 2 (b) χ² Algorithm 2 and (d) χ² Algorithm 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: baseline-uses-true-kernel
category: methodology
topic: "baselines / evaluation fairness"
title: "Appendix-H DR RVI Q-learning baseline (Table 2) runs on the TRUE kernel, not on samples"
severity: medium
confidence: medium
status: question
file: code/supplement/add_baseline.py
line_start: 26
line_end: 31
quote: |
    for trajectory in range(0, n_trajectory):
        dr_q_learning.reset()
        dr_q_learning.relative_value_iteration_q(empirical = False, n_sample = n_sample)
        g_star_baseline = dr_q_learning.g_star
        error = max([abs(g_star_baseline[s] - target_g_star[s]) for s in mdp.states])
        average_error += error
claim: "The Table-2 baseline calls relative_value_iteration_q with empirical=False (same in dr_q_learning_experiment.py:26), which sets dist_of_sa/dist_of_r to model.transition_map/reward_map (dr_q_learning.py:118-120) — i.e. the exact transition kernel — and only uses n_sample as the iteration cap and step size (1/n_sample)**0.95; the two proposed methods it is compared against (Reduction/Anchored) use empirical=True and draw n_sample transitions per (s,a)."
concern: "If the baseline never consumes sampled transitions while the proposed methods do, the Table-2 'sample-size' axis is not comparable across methods and the claim that the proposed methods 'significantly outperform the previous baseline' may reflect a different (sample-free) experimental setting for the baseline rather than a fair head-to-head; filed as a question because it is plausible the authors intend n_sample only as an iteration budget for the Q-learning-style update and I have not confirmed which configuration produced Table 2's exact numbers."
resolution: "Authors: confirm whether the DR RVI Q-learning baseline in Table 2 was run with empirical samples (empirical=True) on the same n_sample budget as the proposed methods, or clarify why empirical=False is the intended/fair baseline setting; if empirical=False was used, re-run with sampled transitions."
cross_refs: ["save-subdir-not-created"]
paper_ref: "Appendix H, Table 2 (DR RVI Q-learning row)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

```yaml finding
id: single-run-no-error-bars
category: methodology
topic: "statistical integrity"
title: "Each Figure-2 / Appendix-H data point is a single run; no error bars"
severity: low
confidence: medium
status: question
file: code/supplement/Reduction_DMDP_parallel.py
line_start: 41
line_end: 41
quote: |
  def run_experiment_for_p(p_values, delta=0.1, n_min=10, grain=10, trajectory_for_each_par=1):
claim: "All drivers default trajectory_for_each_par=1 (Reduction_DMDP_parallel.py:41, Anchored_AMDP_parallel.py:41) and average over that single trajectory, matching the paper's statement that 'each data point ... corresponds to a single estimate generated by one independent run'; no repetition / std / SEM is computed and the only 'significance' is the regression-line fit."
concern: "A single estimate per (n, p) gives no run-to-run variance, so the fitted −1/2 slope carries no uncertainty quantification; this is an evaluation-robustness limitation the paper acknowledges, not a defect in the computation."
resolution: "Authors: run several trajectories per configuration and report error bars (std/SEM) around the Figure-2 points, or justify why one run suffices for the rate claim."
cross_refs: ["chi2-panels-driver-missing"]
paper_ref: "Section 5; NeurIPS Checklist Q7 (statistical significance)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

```yaml finding
id: large-scale-violates-assumption
category: methodology
topic: "evaluation validity"
title: "Large-scale experiment uses δ=0.4 that knowingly violates Assumption 2"
severity: low
confidence: medium
status: question
file: code/supplement/add_large_scale_dmdp.py
line_start: 37
line_end: 37
quote: |
    delta = 0.4
claim: "The large-scale drivers hardcode delta=0.4 (add_large_scale_dmdp.py:37, add_large_scale_anchored.py:57), which the paper (Appendix H) acknowledges 'violates Assumption 2', yet cites the still-near-(−1/2) slope as supporting the theorems."
concern: "Using an uncertainty size outside the regime the theorems assume means the large-scale experiment cannot strictly validate those theorems; whether the near-(−1/2) slope is meaningful here is a design question the paper itself flags."
resolution: "Authors: clarify what claim the δ=0.4 large-scale experiment supports given it lies outside Assumption 2, and whether an in-assumption large-scale run shows the same slope."
cross_refs: ["baseline-uses-true-kernel"]
paper_ref: "Appendix H, large-scale MDP paragraph (Table 3, Figure 3)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 1 | low | No requirements/README; no committed result pickles (figures must be regenerated). |
| bug | 2 | medium | Every driver crashes on save (uncreated subdir); plotting scripts use Windows-only paths. |
| difference | 1 | low | χ² algorithm present but no driver instantiates it; Fig-2 (b)/(d) need a manual class swap. |
| methodology | 1 | medium | Table-2 baseline runs on the true kernel, not samples (question); 2 carried-over questions. |

(The three `methodology`-topic items are all `status: question`, so the methodology *finding*
count is 0; the row's "1" counts the highest-severity question, `baseline-uses-true-kernel`.)

## 5. Closing lists

**Top take-aways** (≤6, ranked by severity × confidence):
1. `[bug]` Every experiment driver writes its pickle to a subdirectory it never creates, so the
   run crashes with `FileNotFoundError` and persists no output as shipped
   (`save-subdir-not-created`, medium/high; `_audit_code/check_save_dirs.py`).
2. `[methodology]` The Appendix-H DR RVI Q-learning baseline (Table 2) is run with
   `empirical=False`, i.e. on the exact transition kernel rather than sampled transitions, while
   the proposed methods use samples — a possible fairness gap in the only head-to-head comparison
   (`baseline-uses-true-kernel`, medium/medium, question).
3. `[difference]` No released driver instantiates the χ² algorithm class, so Figure-2 panels (b)
   and (d) require a manual one-line class swap to reproduce (`chi2-panels-driver-missing`,
   low/high; `_audit_code/check_chi_square_driver.py`).
4. `[bug]` The plotting scripts hardcode Windows backslash paths, so they find no data on
   Linux/macOS (`windows-only-data-paths`, low/high).
5. `[missing]` No requirements file, no README, and no committed result data
   (`no-deps-no-readme-no-results`, low/high).

**Items that genuinely look fine** (actively checked):
- The headline −1/2 slope IS reproducible: re-running the released KL Algorithm-2 path gave
  slope −0.52, the KL Algorithm-3 path −0.54, and the χ² class −0.48 (`_audit_code/out/repro_slopes.txt`).
  The empirical conclusion of the paper is supported by the released code.
- The Hard-MDP model (`model/hard_mdp_unichain.py`) faithfully implements Definition 5.1
  (0-indexed: state 0 ↔ paper state 1 with reward 1, state 1 ↔ paper state 2 with reward 0); the
  ground-truth g* is computed by (relative) value iteration on the true kernel, exactly as the
  paper describes.
- `γ(n)=1−1/√n` and the `n` grid (`np.logspace(log10(10), log10(131342), 20)`) are wired
  consistently with the −1/2-rate setup; the χ² and KL `dual_opt` routines implement the
  divergence duals the paper's algorithms call for.

**Open questions for the authors:**
- Was the Table-2 DR RVI Q-learning baseline run on sampled transitions (`empirical=True`) on
  the same `n_sample` budget as the proposed methods, or on the true kernel (`empirical=False`)?
  (see `baseline-uses-true-kernel`)
- Can a divergence-selecting argument be added so all four Figure-2 panels run from the released
  scripts without editing the source? (see `chi2-panels-driver-missing`)
- Were multiple trajectories per configuration ever run, and can error bars be added to Figure 2?
  (see `single-run-no-error-bars`)
- What claim is the δ=0.4 large-scale experiment intended to support given it lies outside
  Assumption 2? (see `large-scale-violates-assumption`)
