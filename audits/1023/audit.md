# Audit — Over-squashing in Spatiotemporal Graph Neural Networks (paper 1023)

## 1. Summary

This is a **theory paper** (NeurIPS 2025) that formalizes spatiotemporal
over-squashing in convolutional STGNNs and validates it empirically. The core
contributions are theoretical (Theorems 4.1, 5.1, Propositions 4.2/A.2); the
empirical section comprises four quantitative artefacts: success-rate curves on
synthetic COPYFIRST/COPYLAST tasks (Fig. 3) and on ROCKETMAN (Fig. 4 / Fig. 7),
real-world forecasting MAE (Table 1), and a spatial+temporal rewiring ablation
on EngRAD (Table 2).

The repo (`marshka__spatiotemporal-oversquashing`, audited at commit
`a166b499`) is a Hydra/PyTorch-Lightning/`tsl` codebase: synthetic dataset
generators (`lib/datasets/`), an MPTCN model with standard / row-normalized /
dilated temporal convolutions and a GWNet TTS variant (`lib/nn/`), two driver
scripts (`experiments/run_prediction.py`, `experiments/run_realworld.py`), and
sweep configs for each table/figure. I read every source file and config, and
ran three read-only checks under `_audit_code/`:
`check_artifacts.py` (file inventory + grep for FOSR/RGCN, success-rate /
aggregation code, the multi-seed sweep line),
`check_row_norm.py` (numerically confirms the row-normalization weights match
the paper's `RN` description), and inspected the metrics wired into both
drivers. The model architecture, synthetic-task definitions, and temporal
topologies (`R`, `R_N`, `R_D`) are faithfully implemented and match the paper.
The principal gaps are in **result traceability**: the FOSR+RGCN rewiring of
Table 2 is entirely absent, no code converts per-run metrics into the reported
"success rate (%)" or assembles any table/figure, and the committed sweeps omit
the multi-seed replication and the COPYFIRST half of Fig. 3.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig. 2 (R, R_D, R_N matrix powers, illustration) | (none — diagnostic plot of R powers; no script) | — | — | MISSING (illustrative, low impact) |
| Fig. 3 COPYLAST success-rate vs L_T (R, R_N, R_D) | `config/run_copy_time.yaml` + `run_prediction.py` (trains/evals; per-run MSE only) | per-run MSE (not success-rate) | n/a | PARTIAL (no success-rate aggregation; see succrate-aggregation-missing) |
| Fig. 3 COPYFIRST success-rate vs L_T | dataset supports it (`memorize=source,dump=none`) but config fixes COPYLAST only | — | — | PARTIAL (config omits COPYFIRST; see copyfirst-config-missing) |
| Fig. 4 / Fig. 7 ROCKETMAN success-rate (Ring/Lollipop, P=2/3, TTS+T&S) | `config/run_space_vs_time.yaml` + `run_prediction.py` | per-run MSE only | n/a | PARTIAL (no success-rate aggregation) |
| Table 1 MAE METR-LA/PEMS-BAY/EngRAD (MPTCN R, R_N, L∈{1,3,6}) | `config/run_realworld{,_engrad}.yaml` + `run_realworld.py` | per-run MAE (logged) | unverified (training required) | PARTIAL (no table-assembly / aggregation code) |
| Table 1 GWNet (orig.) and GWNet TTS rows | `config/run_realworld_gwnet.yaml` + `gwnet_tts.py` | per-run MAE (logged) | unverified | PARTIAL (no aggregation) |
| Table 2 EngRAD MAE with FOSR rewiring × {RGCN, DCNN} | (none) | — | — | MISSING (no FOSR, no RGCN; see fosr-rgcn-missing) |
| Table 5 TEMPORALTREENEIGHBOURSMATCH accuracy (appendix) | (none — no tree-neighboursmatch dataset/task in repo) | — | — | MISSING (appendix-only, low impact) |
| "±" error bars (Tab 1/2) and "success rate over multiple runs" (Fig 3/4) | requires `+task` multi-seed sweep — commented out in all sweep configs | — | — | PARTIAL (see multi-run-seed-commented) |
| Row-normalized RN = divide t−i by min(i+1,P) (Sec 4) | `convolution_disjoint.py:99-103,149-154` | matches (check_row_norm.py) | ✓ | Verified |

## 3. Findings

### missing

```yaml finding
id: fosr-rgcn-missing
category: missing
topic: "result traceability / Table 2"
title: "No FOSR rewiring or RGCN model code; Table 2 cannot be reproduced"
severity: high
confidence: high
status: finding
file: _audit_code/out/checks.txt
line_start: 52
line_end: 53
quote: |
  === FOSR / RGCN search (Table 2 rewiring) ===
    hits: 0
claim: "A repo-wide search (all .py/.yaml/.md) finds zero references to FOSR, graph rewiring, or RGCN; the file inventory contains no rewiring module and no RGCN model class."
concern: "Table 2 reports EngRAD MAE 'w/ FOSR rewiring' for both RGCN and DCNN spatial layers (and the narrative that temporal rewiring contributes the largest marginal gain), but no code applies FOSR or implements RGCN, so the entire Table 2 experiment is unreproducible from this repo."
resolution: "Provide the FOSR rewiring code and the RGCN model/config used for Table 2, or point to the external script that produced those numbers."
cross_refs: ["succrate-aggregation-missing"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Table 2 (EngRAD, FOSR rewiring w/ RGCN and w/ DCNN)"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: succrate-aggregation-missing
category: missing
topic: "result traceability"
title: "No code computes success-rate (MSE<0.001) or assembles any table/figure"
severity: high
confidence: high
status: finding
file: _audit_code/out/checks.txt
line_start: 55
line_end: 61
quote: |
  === Success-rate / 0.001 threshold / table-building search (Fig3/4,Tab1/2) ===
    raw hits: 5
    ('config/optimizer/adam.yaml', 6, '    lr: 0.001')
    ('config/dataset/la.yaml', 11, '    threshold: 0.1')
    ('config/dataset/engrad.yaml', 19, '    threshold: 0.1')
    ('config/dataset/bay.yaml', 11, '    threshold: 0.1')
    ('experiments/run_prediction.py', 91, '        stopping_threshold=1e-3,')
claim: "The drivers train one model and log per-run scalar metrics (test MSE/MAE) to W&B/Lightning; the only `1e-3` in the repo is an early-stopping threshold, not a success-criterion, and there is no notebook/CSV/plot/aggregation script that converts per-run metrics into the reported 'success rate (%)' (test MSE < 0.001) of Fig. 3/4 or assembles Tables 1-2."
concern: "Every headline empirical number (success-rate curves in Fig. 3/4 and the MAE means±std in Tables 1-2) is produced off-repo, so the computation that yields the reported values is not traceable to any artefact in the repository (Rule G)."
resolution: "Add the aggregation/plotting scripts (or notebooks) that read the logged runs, apply the MSE<0.001 success criterion, average over seeds, and emit the figures/tables; or document the exact W&B queries used."
cross_refs: ["fosr-rgcn-missing", "multi-run-seed-commented"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Fig. 3, Fig. 4, Table 1, Table 2; success criterion in Sec 4 ('task solved when test MSE < 0.001')"
tags: [reforms:5, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: copyfirst-config-missing
category: missing
topic: "result traceability / Fig. 3"
title: "Committed copy-time config produces only COPYLAST, not the COPYFIRST half of Fig. 3"
severity: low
confidence: high
status: finding
file: config/run_copy_time.yaml
line_start: 22
line_end: 26
quote: |
  n_nodes: 1
  spatial_order: 0
  graph_generator: "identity"
  memorize: none  # none, source
  dump: source  # source, none
claim: "run_copy_time fixes memorize=none, dump=source (= predict the LAST value, COPYLAST) and the sweeper does not vary memorize/dump, so the single provided config reproduces only the COPYLAST curves of Fig. 3."
concern: "Fig. 3 reports both COPYFIRST and COPYLAST, but the committed sweep generates only COPYLAST; reproducing the COPYFIRST curves requires manually overriding memorize=source, dump=none, which is not documented in the README or config."
resolution: "Add a sweep dimension (or a second config) toggling memorize/dump so both COPYFIRST and COPYLAST in Fig. 3 are reproduced out of the box, or document the required override."
cross_refs: ["succrate-aggregation-missing"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Fig. 3 (CopyFirst and CopyLast curves)"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

No technical bugs that prevent the intended computation were found in the audited
scope. (`run_prediction.py:81` references the module-global `exp` inside `run`,
but `exp` is bound in `__main__` before `exp.run()` invokes `run`, so this is not
a defect.)

### difference

```yaml finding
id: multi-run-seed-commented
category: difference
topic: "experimental setting / replication"
title: "Multi-seed replication (`+task`) is commented out in every sweep config"
severity: medium
confidence: high
status: finding
file: config/run_realworld.yaml
line_start: 39
line_end: 40
quote: |
      #      +task: 1,2,3,4,5,6,7,8,9,10  # number of seeds per config
      model.hparams.tcn_kwargs.row_normalization: False, True
claim: "In all five sweep configs the `+task: 1..10` grid dimension that replicates each configuration across seeds is commented out, and the global `seed` is `null` (config/default.yaml:30, config/realworld.yaml:27), so running a config as-committed executes exactly one run per setting."
concern: "The paper reports 'success rate across multiple runs' (Fig. 3/4) and mean±std error bars (Tables 1-2), all of which require multiple replicates; as committed the configs produce a single run, so the reported variability/success-rate cannot be reproduced without uncommenting `+task` (the procedure is otherwise valid, hence a difference, not a methodology defect)."
resolution: "Uncomment/enable the `+task` multi-seed dimension by default, and state in the README how many seeds were used for each figure/table and how success rate is computed across them."
cross_refs: ["succrate-aggregation-missing"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Fig. 3/4 ('success rate across multiple runs'); Tables 1-2 (±)"
tags: [reforms:7, forensics:hidden-iteration]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

N/A on the standard leakage/baseline/split checklist — this is a worst-case
sensitivity-bound theory paper validated on **synthetic tasks with
deterministically generated labels** and on standard public forecasting
benchmarks with the conventional sequential 70/10/20 (or year-based EngRAD)
splits used by prior work. No held-out-set, sample-independence, target-leakage,
or pretraining-contamination issue is structurally applicable:

- Data splitting: synthetic splits are sequential index ranges over i.i.d.
  generated sequences (`syntethic.py:340-350`); real-world splits are the
  standard temporal splits via `dataset.get_splitter(...)`
  (`run_realworld.py:94`). Scaling is fit on the train slice only
  (`run_realworld.py:79-108`). No leakage found.
- Baselines: the paper's claims are *relative* (TTS vs T&S, R vs R_N, with/without
  rewiring), and a trivial constant baseline is referenced for the synthetic
  success criterion (MSE≈0.083 for predicting 0.5); `baseline_metrics()` exists
  (`memory.py:96-98`). No naive-baseline omission that would change a conclusion.
- The proofs were read for internal consistency; no statistical-arithmetic
  impossibility (no p-values, t-tests, or CIs are reported).

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 3          | high         | Table 2 (FOSR+RGCN) entirely absent; no success-rate/table aggregation; Fig.3 COPYFIRST config omitted |
| bug         | 0          | -            | None in scope (the `exp` global resolves before `run`)        |
| difference  | 1          | medium       | Multi-seed replication commented out in all sweep configs     |
| methodology | 0          | -            | N/A — synthetic + standard temporal splits; no leakage/baseline issue |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `fosr-rgcn-missing`** (high/high): no FOSR rewiring and no RGCN
   model anywhere in the repo — the entire Table 2 ablation is unreproducible.
2. **[missing] `succrate-aggregation-missing`** (high/high): no code computes the
   "success rate (MSE<0.001)" of Fig. 3/4 or assembles Tables 1-2; all headline
   numbers are produced off-repo.
3. **[difference] `multi-run-seed-commented`** (medium/high): the `+task`
   multi-seed dimension is commented out in every sweep, so the success rates and
   ± error bars cannot be reproduced as committed.
4. **[missing] `copyfirst-config-missing`** (low/high): the committed copy-time
   config reproduces only COPYLAST; COPYFIRST (Fig. 3) needs an undocumented
   override.

### Items that genuinely look fine
- **Row-normalized convolution `R_N`** matches the paper's "divide input at t−i
  by min(i+1, P)" exactly (`_audit_code/check_row_norm.py` → MATCH).
- **MPTCN architecture** (encoder = fixed semi-orthogonal upscale, per-horizon
  linear readout on the last time step, decoupled TMP→MP STMP layers, TTS = L=1)
  faithfully implements the framework of Eq. 6-10 (`stgnn_base.py`,
  `convolution_disjoint.py`).
- **Synthetic task labels** are generated correctly and deterministically:
  COPYFIRST/COPYLAST (`memory.py:68-94`) and ROCKETMAN k-hop average at distance
  i (`memory.py:132-143`).
- **GWNet TTS variant** shares parameter count and adds a non-trainable skip
  connection to mimic the original, as the appendix describes
  (`gwnet_tts.py:54-105`).
- **Dependencies** are specified and pinned (`conda_env.yml`: python/pytorch/
  lightning/pyg versioned, `tsl` pinned to a commit hash).
- **Real-world preprocessing/scaling** is fit on the train slice only; no
  look-ahead leakage (`run_realworld.py:79-108`).

### Open questions for the authors
- How is "success rate (%)" computed across runs (which seeds, how many,
  pass criterion applied to which split's MSE), and which script produced
  Fig. 3/4 and Tables 1-2? (relates to `succrate-aggregation-missing`)
- Where is the FOSR rewiring and RGCN code used for Table 2?
  (relates to `fosr-rgcn-missing`)
- Were the reported runs produced with the `+task` seeds enabled and the global
  `seed` left `null` (init-only variation, fixed synthetic data)?
  (relates to `multi-run-seed-commented`)
