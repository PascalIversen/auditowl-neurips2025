# Code-repository audit — Paper 2963

**"Learning Equilibria from Data: Provably Efficient Multi-Agent Imitation Learning"** (MURMAIL)
Repo: `github.com/tfreihaut/Murmail`, cloned at `code/tfreihaut__Murmail/`.

## 1. Summary

This is a primarily theoretical NeurIPS paper. The only empirical content is a
"Numerical Validation" (§5) consisting of **Figure 2, panels (a)–(d)** —
exploitability ("Nash-Gap") vs. queries/dataset-size curves for MURMAIL (Algorithm 2)
and multi-agent Behavior Cloning (BC) on the |S|=7 lower-bound game of Fig. 1, for four
expert mixtures controlling the concentrability coefficient C(µE,νE) ∈ {≤2, ≤1000,
≤10000, ∞} — plus **one additional plot in Appendix J.4** for a second, randomly
generated |S|=10 environment with C(µE,νE) < ∞. There are no tables and no numeric
statistics in the paper; all empirical claims are qualitative (BC fails at C=∞, MURMAIL
succeeds; BC can be more efficient when C is finite and small).

The repo is small and self-contained: `murmail.py` (Algorithm 2), `behavior_cloning.py`
(BC baseline), `innerloop_rl.py` (UCBVI inner loop), `game_solver.py` (a zero-sum
value-iteration NE solver), `utils.py` (exact exploitability + plotting), and
`experiment.ipynb` (the driver that builds the 7-state environment, runs both methods
for four hardcoded expert configs, and saves `Experiment_C_*.png`). The four committed
PNGs correspond to the Fig. 2 panels.

**What I did.** I read every module and the notebook (including its saved cell outputs,
which confirm `num_runs=10` and `learning_rate=50.0`). I traced each figure panel to the
code that produces it; grepped the entire repo for any random seeding (found none); and
wrote one deterministic check under `_audit_code/` comparing the single-sample
mirror-descent update actually used by every Fig. 2 run against (i) the paper's
Algorithm-2 update rule and (ii) the repo's own batch update
(`_audit_code/check_exp_grad_update.py`, output `_audit_code/out/exp_grad_update.txt`).

**Headline findings.** (1) The `batch_size=1` mirror-descent update used by all four
Fig. 2 MURMAIL runs only exponentiates the gradient at the single sampled action, not
the full action vector the paper's update specifies, and disagrees with the repo's own
(paper-consistent) batch update — a `bug`. (2) The driver hard-codes a tuned learning
rate (η=50) and `num_runs=10`, whereas the paper (App. J.2) states an "optimal" η and
**1000** seeds, and a notebook comment admits the plots differ from the paper — a
`difference` (paper-vs-code). (3) Nothing is seeded, so the figures are not exactly
reproducible — `missing` (reproducibility). (4) The second (|S|=10 random) environment
of App. J.1/J.4 has no generating code — `missing`.

Because the paper reports no numbers, none of these is a numeric mismatch; the empirical
section is a qualitative proof-of-concept, so severities are bounded by that.

## 2. Traceability table

The paper contains no tables, no statistical tests, and no numeric values to verify; the
only artefacts are the four Fig. 2 panels and the App. J.4 plot. "Computed value" is the
qualitative curve shape, since no numbers are reported.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2(a) C(µE,νE)≤2, MURMAIL vs BC curves | `experiment.ipynb` cell `8224f896` (expert mix 0.5/0.5) + `murmail.py`, `behavior_cloning.py`, `utils.calc_exploitability_true` | committed `Experiment_C_2.png` | qualitative ✓ (no numbers in paper) | Verified (driver present); update rule differs (see `single-sample-md-update-one-action`) |
| Fig. 2(b) C(µE,νE)≤1000 | `experiment.ipynb` cell `38df9571` (mix 0.001/0.999) | committed `Experiment_C_1000.png` | qualitative ✓ | Verified |
| Fig. 2(c) C(µE,νE)≤10000 | `experiment.ipynb` cell `d2edf299` (mix 0.0001/0.9999) | committed `Experiment_C_10000.png` | qualitative ✓ | Verified |
| Fig. 2(d) C(µE,νE)=∞ | `experiment.ipynb` cell `610bf313` (pure a3b3 expert) | committed `Experiment_C_inf.png` | qualitative ✓ | Verified |
| Fig. 2 x-axis "Queries/Dataset Size" up to 10·10⁴ | `utils.plot_results:107-117` plots raw `queries`/`iterations` (max ≈ 1000) | x reaches ≈1000, not 10⁵ | — | Note: notebook comment states plots differ from paper; see `tuned-lr-and-runs-vs-paper` |
| App. J.4 additional plot, random \|S\|=10 env, C<∞ | (none) | — | — | MISSING (no random-env generator; see `random-env-missing`) |
| App. J.2 "optimal learning rate η" + "1000 seeds" | `experiment.ipynb:300,357-359` uses η=50, `num_runs=10` | η=50, 10 runs | ✗ (config differs) | MISMATCH → `tuned-lr-and-runs-vs-paper` |
| App. J.2 expert distributions via zero-sum value iteration | `game_solver.py` present but never called; experts hardcoded in notebook | hardcoded mixtures | qualitative ✓ | Verified (hardcoded mixtures equal described NE mixtures) |

## 3. Findings

## missing

```yaml finding
id: no-seeding-not-reproducible
category: missing
topic: "reproducibility / seeding"
title: "No random seeding anywhere; Fig. 2 curves not exactly reproducible"
severity: low
confidence: high
status: finding
file: experiment.ipynb
line_start: 337
line_end: 337
quote: |
      "num_runs = 10\n",
claim: "BC, MURMAIL, the UCBVI inner loop, and expert sampling all call np.random.* (behavior_cloning.py:113-122, murmail.py:204-216, 305-312, 369-372, innerloop_rl.py:75,81), but no np.random.seed / default_rng is set in any module or in experiment.ipynb (grep for 'seed' over *.py and the notebook returns 0 hits), while each panel runs num_runs=10 unseeded repetitions."
concern: "With num_runs=10 and no seeding, the exact Fig. 2 curves and their error bands cannot be regenerated and the run-to-run variability is undocumented."
resolution: "Add a seed (or per-run seed list) before each run and report it, so the committed figures can be reproduced."
cross_refs: ["tuned-lr-and-runs-vs-paper"]
paper_ref: "Figure 2; Appendix J.2"
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: random-env-missing
category: missing
topic: "result traceability"
title: "Random |S|=10 environment (App. J.1/J.4 plot) has no generating code"
severity: low
confidence: high
status: finding
file: paper.pdf
quote: |
  For this we consider two environments. For the first environment, we generate a random Zero-Sum Markov Game with |S| = 10, |A| = |B| = 3 and a
  reward between −1 and 1.
claim: "Appendix J.1 describes a randomly generated |S|=10 zero-sum Markov Game whose results are shown in the App. J.4 'additional plot'; the only environment built in the repo is the |S|=7 lower-bound game (experiment.ipynb cell e2cab221, num_states=7), and no code generates a random |S|=10 game or its expert policies."
concern: "The App. J.4 result has no traceable computation in the repo, so that plot cannot be reproduced from the provided code."
resolution: "Authors: please add the script that generates the random |S|=10 Markov Game, its expert policies, and the App. J.4 plot."
cross_refs: ["experts-hardcoded-not-solver"]
paper_ref: "Appendix J.1 (Environments with C<∞) and J.4 (Additional plots)"
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: single-sample-md-update-one-action
category: bug
topic: "mirror descent update"
title: "batch_size=1 update exponentiates only the sampled action, not the full action vector"
severity: medium
confidence: high
status: finding
file: murmail.py
line_start: 390
line_end: 397
quote: |
        new_policy = policy.copy()
        # Following the gradient formula in the algorithm
        g = policy[s, a] - (1 if a == a else 0)  # Indicator is 1 when a equals the sampled action
        # Apply the exponentiated gradient update
        new_policy[s, a] = policy[s, a] * np.exp(-self.eta * g)
        # Re-normalize
        new_policy[s] /= new_policy[s].sum()
        return new_policy
claim: "_exp_grad_update (the batch_size=1 path taken by every Fig. 2 run, murmail.py:144-149) updates only the sampled action a via policy[s,a]*exp(-eta*(policy[s,a]-1)) and leaves all other actions a'!=a unchanged before renormalising; Algorithm 2's update mu_{k+1}(a|s) ∝ mu_k(a|s) exp(-eta g(s,a)) with g(s,a)=mu_k(a|S_k)1{S_k=s}-1{A_k=a} multiplies every action a' in the sampled state by exp(-eta*mu_k(a'|s)) (the indicator is 0 for a'!=a). _audit_code/check_exp_grad_update.py confirms single != paper while the repo's own _batch_exp_grad_update == paper."
concern: "The mirror-descent step that drives every MURMAIL curve in Fig. 2 does not implement the paper's Algorithm 2 update and disagrees with the repo's own correct batch update, so the displayed MURMAIL behaviour reflects a different update rule than the one analysed."
resolution: "Replace _exp_grad_update with the full-row update (g = policy[s,:]; g[a]-=1; new_policy[s] = policy[s]*exp(-eta*g); renormalise), matching _batch_exp_grad_update, and re-generate Fig. 2."
cross_refs: ["tuned-lr-and-runs-vs-paper"]
check_script: _audit_code/check_exp_grad_update.py
paper_ref: "Algorithm 2, 'Update policies' block; gradient g^mu_k(s,a)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: tuned-lr-and-runs-vs-paper
category: difference
topic: "hyperparameters / experimental setup"
title: "Driver uses tuned eta=50 and 10 runs; paper states optimal eta and 1000 seeds"
severity: low
confidence: high
status: finding
file: experiment.ipynb
line_start: 300
line_end: 300
quote: |
      "# Note that we have tuned the learning rate and MURMAIL performs better with this setting (compared to plots in the paper)\n",
claim: "The notebook hard-codes learning_rate=50.0 and num_runs=10 (experiment.ipynb:357-369, repeated for all four panels) and a comment states the learning rate was tuned and 'MURMAIL performs better with this setting (compared to plots in the paper)'; Appendix J.2 instead states 'We run the experiments for each environments 1000 times over different seeds' and 'we compute the optimal learning rate eta' (theory sets eta = 2|S|log|Amax|/K ~ 0.015 for |S|=7, K=1000)."
concern: "The committed code reproduces curves the authors themselves note differ from the paper's Fig. 2, at a far smaller run count (10 vs 1000) and an un-derived hand-tuned step size, so the repo does not regenerate the published figures as plotted."
resolution: "Authors: please commit the exact eta and seed/run configuration used for the published Fig. 2 (and App. J.4), or update the figures to match the committed configuration."
cross_refs: ["single-sample-md-update-one-action", "no-seeding-not-reproducible"]
paper_ref: "Appendix J.2 (Experimental Setup); Theorem 4.2"
tags: [forensics:hidden-iteration, reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: experts-hardcoded-not-solver
category: difference
topic: "expert generation"
title: "Expert policies hardcoded in notebook; described value-iteration solver unused"
severity: low
confidence: medium
status: finding
file: experiment.ipynb
line_start: 311
line_end: 320
quote: |
      "expert1 = np.array([\n",
      "    # s0, s1, s2, s3, Sxplt1, Sxplt2, Scopy\n",
      "    [0, 0, 1],  # s0\n",
      "    [0, 0, 1],  # s1\n",
      "    [0, 0, 1],  # s2\n",
      "    [0, 0, 1],  # s3\n",
      "    [0, 0, 1],  # Sxplt1\n",
      "    [0, 0, 1],  # Sxplt2\n",
      "    [0, 0, 1],  # Scopy\n",
      "])\n",
claim: "Appendix J.2 says expert distributions are generated 'using a Value Iteration algorithm for Two Player Zero-Sum Games' (game_solver.MarkovGameValueIteration), but the notebook hardcodes each expert policy as a constant np.array (one block per panel) and never imports or calls game_solver; for the |S|=7 game the hardcoded mixtures (a3 prob 0.5 / 0.999 / 0.9999 / 1.0) equal the NE mixtures the paper describes, so the values are individually valid."
concern: "The committed experts bypass the described NE solver, so the only link between the paper's 'value-iteration-generated experts' and the figures is the notebook's hardcoded constants; the solver path is exercised by no committed code."
resolution: "Authors: confirm the hardcoded |S|=7 experts equal the value-iteration NE mixtures, and provide the solver-driven expert generation used for the random |S|=10 environment."
cross_refs: ["random-env-missing"]
paper_ref: "Appendix J.2, 'To generate the expert distributions, we use a Value Iteration algorithm ...'"
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The empirical section is a deliberate proof-of-concept on a
synthetic, fully-known Markov Game: there is no dataset, no train/test split, no learned
features, no pretrained model, and no statistical test — the standard leakage / split /
baseline-tuning failure modes are structurally inapplicable (see Scope notes below).
Exploitability is computed exactly against the true reward/transition model
(`utils.calc_exploitability_true`), which is the correct Nash-Gap metric for this
setting, and the BC baseline is run under the same environment, metric, and evaluation as
MURMAIL.

## 4. Scope notes (structurally inapplicable topics)

- **Data splitting / sample independence / target leakage / inference-time shift**: N/A —
  no dataset; data are sampled on-the-fly from a fully-specified synthetic Markov Game.
- **Pretraining contamination**: N/A — no pretrained model, encoder, or embeddings.
- **Temporal integrity**: N/A — no real time-series; the only "time" is RL episode steps
  within the synthetic MDP.
- **Statistical integrity**: N/A — the paper reports no p-values, CIs, or numeric
  statistics; Fig. 2 shows mean ± std over runs (qualitative). The related note on run
  count (10 vs 1000) is filed under `tuned-lr-and-runs-vs-paper`.
- **Baselines**: BC is the intended comparison baseline and is run identically to MURMAIL
  (same env, same exact-exploitability metric); no asymmetry found.

## 5. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 2          | low          | No seeding (not reproducible); random \|S\|=10 env of App. J.4 has no code. |
| bug         | 1          | medium       | batch_size=1 mirror-descent update updates only the sampled action, unlike paper / repo's own batch path. |
| difference  | 2          | low          | Tuned η=50 & 10 runs vs paper's optimal η & 1000 seeds (notebook admits plots differ); experts hardcoded, solver unused. |
| methodology | 0          | -            | Synthetic proof-of-concept; standard leakage/split/baseline failure modes inapplicable. |

### Top take-aways (≤6, ranked by severity × confidence)

1. **[bug] `single-sample-md-update-one-action`** (med/high): the `batch_size=1`
   mirror-descent update driving every Fig. 2 MURMAIL curve exponentiates only the sampled
   action, contradicting both Algorithm 2 and the repo's own batch update (verified by
   `_audit_code/check_exp_grad_update.py`).
2. **[difference] `tuned-lr-and-runs-vs-paper`** (low/high): driver uses hand-tuned η=50 and
   10 runs vs the paper's "optimal η" and 1000 seeds, with a notebook comment admitting the
   plots differ from the paper.
3. **[missing] `random-env-missing`** (low/high): the random |S|=10 environment behind the
   App. J.4 plot has no generating code.
4. **[missing] `no-seeding-not-reproducible`** (low/high): nothing is seeded, so the exact
   Fig. 2 curves cannot be regenerated.
5. **[difference] `experts-hardcoded-not-solver`** (low/med): experts are hardcoded
   constants; the described value-iteration NE solver (`game_solver.py`) is never called.

### Items that genuinely look fine

- `utils.calc_exploitability_true` computes exact Nash-Gap via exact policy evaluation +
  value iteration against the true model — the correct, unbiased exploitability metric.
- The |S|=7 lower-bound environment (`experiment.ipynb` cell `e2cab221`) matches the Fig. 1
  / App. J.1 description (7 states, |A|=|B|=3, rewards on the exploitable states, δ_{s0}
  start, γ=0.9).
- The reward construction `_state_reward` matches Algorithm 2's stochastic reward
  `1{A_E=A'_E} − 2µ(A_E|s) + ‖µ(·|s)‖²`.
- The BC baseline is run under the same environment, metric, and evaluation cadence as
  MURMAIL (no tuning asymmetry).
- Dependencies are fully pinned (`requirements.txt`) and the repo is self-contained
  (no external data, no closed APIs).

### Open questions for the authors

- Was the published Fig. 2 generated with the committed `_exp_grad_update` (single-action)
  path, the `_batch_exp_grad_update` path, or off-repo code? This determines whether
  `single-sample-md-update-one-action` affects the published figure or only the committed
  reproduction.
- What exact η, seed list, and run count produced the published Fig. 2 and the App. J.4
  plot, and where is the random |S|=10 environment generator?
