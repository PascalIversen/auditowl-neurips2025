# Audit: On Evaluating Policies for Robust POMDPs (NeurIPS 2025, #1729)

## 1. Summary

This is the canonical author code (Julia / POMDPs.jl) for "On Evaluating Policies
for Robust POMDPs", extracted from Zenodo deposit 17424410 (paper reference [29]);
its README confirms it is the codebase for this NeurIPS 2025 paper, so per the
repository-provenance rule it is audited as the author code. The paper is an
empirical/methods paper on *robust POMDP* evaluation: it proposes benchmarks
(TOY*, ECHO, PARITY), a robust policy-evaluation pipeline (build a "nature MDP"
adversary and solve it with MCTS to find the worst-case value), and two efficient
baselines (RQMDP, RFIB). There is no learning, no dataset, and no train/test split;
the reported quantities are computed POMDP values. The experiment is driven by
`RunAll.sh → run.jl` (RHSVI on M and on the simplified models M_Center/M_Ent/M_RMDP,
plus RQMDP and RFIB on M), writing one JSON per (env, solver, rtype) into
`Data/Tests/`; `DataCollection.ipynb` renders Tables 1–3 and Figure 4 from those JSONs.

I could not execute the Julia code (needs Julia 1.11.5 + a Gurobi license + GNU
parallel), so I did static analysis plus deterministic checks against the *shipped*
result JSONs. I read the driver (`run.jl`), the reference solver (`RHSVI.jl` +
the robust backup in `RobustAlphaVectors.jl`), the baselines (`RQMDP.jl`), the
adversary construction (`adversarialnature.jl`), the model simplifications
(`approximate.jl`, `makeuncertain.jl`), the plotting notebook, and `Project.toml`/
`Manifest.toml`. I wrote two scripts under `_audit_code/`:
`check_tables.py` re-derives Tables 2/3 and the Figure-4 Vgap values from the JSONs
using the notebook's own file-selection logic, and `check_paper_vs_json.py`
quantifies the gap between the paper's reported numbers and the shipped JSONs. A
third deterministic check (in `check_tables.py`) confirmed Table 1 dimensions.
Headline conclusion: the pipeline is present and traceable end-to-end — every
figure panel and table cell maps to a producing script and a shipped data file, the
model-simplification ↔ nature-heuristic mapping is correct, and Table 1 reproduces
exactly. The main caveat is reproducibility: the MCTS evaluation is unseeded *and*
wall-clock-time-bounded, and the shipped JSONs differ from the paper by up to ~32%
in the high-variance MCTS cells (RQMDP on PARITY/TIGER, RHSVI(mid/maxent) on
PARITY(∞)), though the qualitative conclusions are unaffected.

## 2. Traceability table (Rule G)

Mapping uses the code↔paper env names (`Toy`=TOY*, `Machine`=ECHO, `ChainInf`=PARITY(∞),
`Chain10`=PARITY(10), `CNC_Detection`=HEALTHDETECTION; `mid`=M_Center, `maxent`=M_Ent,
`rmdp`=M_RMDP). "Computed value" = value re-derived by `_audit_code/check_tables.py`
from the latest shipped JSON (the file the notebook itself would load).

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 (dims) all 11 benchmarks | `run.jl` envs + `Data/Tests/*RHSVI_full*.json` states/obs/actions | all 11 (|S|,|Ω|,|A|) | ✓ (11/11) | Verified |
| Table 2 RHSVI(M_Center/M_Ent/M_RMDP) min/std | `run.jl:282-353` + `RHSVI.jl`; JSON `value_adv`,`std_adv` | e.g. TOY* 37.48/37.46/32.46 | ✓ for most; ✗ PARITY(∞) | Mostly verified, see `repro-nondeterminism` |
| Table 3 RHSVI/RQMDP/RFIB min/std | `run.jl:282-353` + `RHSVI.jl`/`RQMDP.jl`; JSON | e.g. PARITY(10) RQMDP 26.51 (paper 38.89) | ✗ for high-var cells | Mostly verified, see `repro-nondeterminism` |
| Figure 4 left Vgap (M, M_Center, M_Ent, M_RMDP) | `DataCollection.ipynb` cell17; `(value_adv−Ṽ)/|Ṽ|`, Ṽ=RHSVI full `value_sol` | reproduced all 11 rows | ✓ qualitatively | Verified (formula matches caption) |
| Figure 4 right Vgap (RHSVI, RQMDP, RFIB) | `DataCollection.ipynb` cell17 | reproduced all 11 rows | ✓ qualitatively | Verified |
| Q1 "value gap ≈ 0 for smaller envs, positive for ALOHA/HEALTHDETECTION" | Figure 4 data | ALOHA/HEALTHDETECTION gaps > 0 | ✓ | Verified |
| Q2/Q3 "RFIB optimal on PARITY; TOY*/ECHO not naively solvable" | Figure 4 right (RFIB), left | RFIB gap≈0 on PARITY; naive gaps<0 on TOY*/ECHO | ✓ | Verified |
| RHSVI robust-backup ad-hoc fix (App. D.1) | `RobustAlphaVectors.jl:396-612` (2-stage LP, exploitability penalty `:572`) | n/a (qualitative) | ✓ present | Verified present |
| Nature heuristics θ_Center/θ_Ent/θ_RMDP (Sec. 3) | `approximate.jl:75-86` | n/a | ✓ mapping correct | Verified |
| MCTS adversary / nature MDP (Sec. 4) | `adversarialnature.jl:21-85`, `run.jl:303-336` | n/a | ✓ present | Verified |
| RQMDP / RFIB baselines (Sec. 5) | `RQMDP.jl:35-82` (RQMDP), `:160-228` (RFIB) | n/a | ✓ present | Verified |

No artefact is fully MISSING: every reported figure panel and table cell has both a
producing code path and a shipped JSON data file.

## 3. Findings

## missing

_No `missing` findings._ All reported artefacts (Tables 1–3, both Figure 4 panels,
the Q1–Q3 headline claims) trace to a producing script plus a shipped result file.
Dependencies are fully declared in `Project.toml` with a complete pinned
`Manifest.toml` (1571 lines); the only non-free dependency is Gurobi, which the
README documents and which is a legitimate paid-solver exemption.

## bug

_No `bug` findings._ The one flag-name mismatch I checked (`RunAll.sh` passes
`--solver` while `run.jl` declares `--solvers`) is resolved by ArgParse.jl's default
unambiguous-prefix abbreviation (`--solver` uniquely abbreviates `--solvers`), and
the shipped JSON filenames confirm the runs produced correctly-named output, so it
is not a defect that breaks the pipeline. See "Items that look fine".

## difference

```yaml finding
id: repro-nondeterminism
category: difference
topic: "reproducibility / result traceability"
title: "MCTS evaluation is unseeded and time-bounded; shipped data differs from paper by up to ~32%"
severity: medium
confidence: high
status: finding
file: run.jl
line_start: 317
line_end: 336
quote: |
            t_eval = @elapsed begin
                vals_adv = []
                for i in 1:evalnmbr
                    this_val_adv = 0.0
                    for (sinit_adv, prob) in weighted_iterator(binit_adv)
                        this_max_time = evaltime*prob
                        this_n_iterations = Int(ceil(total_iterations*prob))
                        solver_adv = MCTSSolver(max_time=this_max_time, n_iterations=this_n_iterations, depth=100, exploration_constant=25.0, estimate_value=f_leaf, init_Q=f_init, init_N=0)
                        # solver_adv = MCTSSolver(max_time=this_max_time, n_iterations=this_n_iterations, depth=100, exploration_constant=5.0, estimate_value=f_leaf, init_Q=f_init, init_N=0, enable_tree_vis=true)
                        policy_adv = solve(solver_adv, env_adv)
                        a = action(policy_adv, sinit_adv)
                        this_val_adv -= prob * value(policy_adv, sinit_adv)
                        # inchrome(D3Tree(policy_adv, sinit_adv))
                    end
                    push!(vals_adv, this_val_adv)
                    println(this_val_adv)
                end
            end
            val_adv, std_adv = minimum(vals_adv), Statistics.std(vals_adv)
claim: "The worst-case value reported in Tables 2/3 and Figure 4 is the minimum over evalnmbr=5 MCTS runs; the MCTSSolver is constructed with no rng= argument, no Random.seed! is set anywhere in the repo (grep finds none), and each run is bounded by wall-clock time (max_time=evaltime*prob, evaltime=300s), so the number of completed MCTS iterations depends on machine speed."
concern: "Re-deriving Tables 2/3 from the shipped Data/Tests JSONs with the notebook's own file-selection logic reproduces most cells to within ~2%, but the high-variance MCTS cells diverge from the paper by up to ~32% (PARITY(10) RQMDP: paper 38.89 vs JSON 26.51; TIGER RQMDP: -20.04 vs -26.21; PARITY(inf) RHSVI mid/maxent: 9.25/9.07 vs 7.02/7.05); without a seed and with time-bounded search the exact reported numbers cannot be reproduced, and the paper does not disclose this."
resolution: "Authors: please set and report an RNG seed (pass rng= to MCTSSolver), or report results as means +/- std over independent seeds with a fixed iteration budget rather than a wall-clock budget; and confirm whether the shipped JSONs are the exact ones behind Tables 2/3 (several envs have multiple re-run timestamps and the notebook silently picks the latest)."
cross_refs: ["§6.1", "Appendix D.3"]
check_script: _audit_code/check_paper_vs_json.py
paper_ref: "Tables 2 and 3; Figure 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: figure4-machine-vs-echo-label
category: difference
topic: "figure/table labeling"
title: "Figure 4 labels the ECHO benchmark as 'Machine' (code label never renamed)"
severity: low
confidence: high
status: finding
file: DataCollection.ipynb
line_start: 0
line_end: 0
quote: |
    elif name.startswith("Machine"):
        return r"\textsc{Machine}"
claim: "The plotting notebook's get_latex_name maps the env code-name 'Machine' to the figure label \\textsc{Machine}, whereas the same benchmark is named ECHO in Table 1, Appendix D.2, and Section 3; the figure was generated by this code so its left panel carries the label 'Machine' (visible in the paper text extraction of Figure 4), inconsistent with the ECHO name used in the tables/prose."
concern: "A reader cross-referencing Figure 4 (row 'Machine') against Table 1/3 (row ECHO) may not realise they are the same benchmark; it is a naming inconsistency only, the underlying environment and computed values are identical."
resolution: "Authors: update get_latex_name('Machine') to return ECHO so the figure label matches Table 1 and the running text."
cross_refs: ["repro-nondeterminism"]
paper_ref: "Figure 4 (left panel y-axis labels) vs Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: reference-value-from-uncertified-solver
category: methodology
topic: "evaluation validity / reference value"
title: "Vgap reference value comes from the same RHSVI whose backup has no correctness guarantee"
severity: medium
confidence: medium
status: question
file: RPOMDPs/Algorithms/RobustAlphaVectors.jl
line_start: 571
line_end: 575
quote: |
    # println(model)
    @objective(model, Max, Q - 0.01 * sum(alpha_diff))
    # @objective(model, Max, Q)
    # println(model)
    optimize!(model)
claim: "The Figure-4 metric Vgap(pi)=(V^{pi,theta*}-Vtilde)/|Vtilde| uses Vtilde = the RHSVI value of the RPOMDP M (value_sol of the RHSVI-full run); Vtilde is produced by the same ad-hoc robust backup (this second-stage LP with the exploitability penalty) that Appendix D.1 explicitly says it has 'no theoretical basis to claim the approach is correct'."
concern: "Both the reference value Vtilde and the RHSVI policy being evaluated derive from a backup the authors acknowledge is unverified, so a near-zero Vgap could reflect two consistent approximation errors rather than an exactly-correct evaluation; the paper does disclose this limitation, which is why this is filed as a question rather than a finding."
resolution: "Authors: for the small benchmarks where exact values are known analytically (TOY*, ECHO, PARITY - Section 3/Appendix A), report Vtilde against the analytic optimum to bound the RHSVI reference error independently of the MCTS evaluation."
cross_refs: ["§6.1 Metric", "Appendix D.1"]
paper_ref: "Figure 4 caption (Vtilde); Appendix D.1 (robust backup, 'no theoretical basis')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 0          | -            | All figures/tables trace to code + shipped data; deps pinned. |
| bug         | 0          | -            | Flag-name mismatch is saved by ArgParse prefix-abbreviation. |
| difference  | 2          | medium       | Unseeded/time-bounded MCTS → ≤~32% repro gap; ECHO mislabelled "Machine". |
| methodology | 1 (question)| medium      | Vgap reference from the unverified RHSVI backup (paper-acknowledged). |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[difference] `repro-nondeterminism`** — The MCTS worst-case evaluation is
   unseeded and wall-clock-time-bounded; re-deriving Tables 2/3 from the shipped
   JSONs reproduces most cells within ~2% but diverges up to ~32% on the
   high-variance cells (RQMDP on PARITY/TIGER, RHSVI(mid/maxent) on PARITY(∞)).
   Qualitative conclusions hold, but exact numbers are not reproducible and the lack
   of seeding is undisclosed.
2. **[methodology, question] `reference-value-from-uncertified-solver`** — The Vgap
   denominator/reference Ṽ is produced by the same RHSVI ad-hoc backup the paper
   itself says is theoretically uncertified; near-zero gaps could reflect matched
   approximation errors. Paper-acknowledged, hence a question.
3. **[difference] `figure4-machine-vs-echo-label`** — Figure 4 labels the ECHO
   benchmark "Machine" because the notebook's name map was never updated; cosmetic.

### Items that genuinely look fine
- **Table 1 dimensions**: all 11 benchmarks' (|S|,|Ω|,|A|) reproduce exactly from the
  shipped JSON state/obs/action counts (`_audit_code/check_tables.py`).
- **Figure 4 metric formula**: the plot (`DataCollection.ipynb` cell17) computes
  `(value_adv − Ṽ)/|Ṽ|`, exactly matching the caption's Vgap definition; the
  differently-signed `value_diff = (value_sol−value_adv)/|value_adv|` in `getvalues`
  is dead code (`Data[...,2]` is never used in any rendered table or figure).
- **Nature-heuristic mapping**: `mid→to_mid_POMDP` (centroid = θ_Center),
  `maxent→to_maxent_POMDP` (max-entropy = θ_Ent), `rmdp→to_rmdp_POMDP` (θ_RMDP)
  map correctly to the paper's Section-3 heuristics; the Table-2 column order
  (M_Center, M_Ent, M_RMDP) matches the notebook's insertion-ordered `stds` dict.
- **Worst-case aggregation**: `val_adv = minimum(vals_adv)`, `std_adv = std(vals_adv)`
  over `evalnmbr=5` runs faithfully implements "run MCTS five times and report the
  lowest value" (run.jl:335); the sign handling (nature reward `−reward`, negated
  back) is consistent.
- **RHSVI ad-hoc backup**: the two-stage LP and the state-exploitability penalty
  described in Appendix D.1 are actually implemented (`RobustAlphaVectors.jl:396-612`),
  and the paper openly flags the missing correctness guarantee.
- **RQMDP/RFIB baselines**: implemented as robust (worst-case-over-interval) value
  iteration via LP, consistent with the Section-5 contribution.
- **Dependencies**: `Project.toml` + complete pinned `Manifest.toml`; Gurobi is the
  only non-free dependency and is documented in the README (legitimate exemption).
- **`--solver` vs `--solvers`**: `RunAll.sh` uses the singular flag; ArgParse.jl's
  default unambiguous-prefix matching resolves it to `--solvers`, and the shipped
  output filenames confirm the runs ran as intended.

### Open questions for the authors
- Are the JSONs currently in `Data/Tests/` (several envs have multiple re-run
  timestamps; the notebook silently loads the latest) the exact ones behind the
  published Tables 2/3 and Figure 4, or were the published numbers from an earlier
  run? (Relates to `repro-nondeterminism`.)
- Can the RHSVI reference value Ṽ be cross-checked against the analytic optima for
  the small benchmarks (TOY*, ECHO, PARITY) to bound the reference error
  independently of MCTS? (Relates to `reference-value-from-uncertified-solver`.)

### Scope notes (N/A topics)
Data splitting, sample independence, target/shortcut leakage, pretraining
contamination, temporal integrity, and most statistical-integrity checks are
structurally N/A: this is a planning/solver-benchmark paper with no dataset, no
train/test split, no learned model, and no significance tests — the reported
quantities are computed POMDP values, not statistics over samples.
