# Code-repository audit — *Accelerated Evolving Set Processes for Local PageRank Computation* (NeurIPS 2025, #3114)

## 1. Summary

The repository (`code/Rick7117__aesp-local-pagerank/`) is a Python + numba
implementation of the paper's local PageRank solvers. `ppr_solver.py` contains
the actual numerical routines (`appr`, `apprOpt`, `locGD`, `aespAPPR`,
`aespLocGD`, plus baselines `ista`, `fista`, `aspr`, `cheby`), each of which
computes the convergence metrics the paper plots (`errs` = `log‖D⁻¹(π̂−π)‖∞`,
`opers` = operation counts, `runtime`, `grad_norms`, `vols`, `gammas`).
`main.py` is the driver: it loads a graph, runs the requested method(s), and
writes a per-(method,node,α,ε) `.npz` into `datasets/<graph>/results/`. The
~14 `plot_*.py` scripts then read those `.npz` files and render the paper's
figures. `tables/` holds five committed CSVs (graph statistics, R-vs-α,
runtime/operation data).

The paper is primarily theoretical (the headline is a complexity bound and the
resolution of an open conjecture); the empirical section validates the
algorithm via convergence plots (Figs. 2–9) and operation/runtime tables
(Tables 2–4) on up to 19 real-world graphs.

What I did: read `paper.pdf`/`paper_text.txt` (esp. §4 Experiments and Appendix
C), read all `.py` files, and ran deterministic checks under `_audit_code/`:
`check_dataset_stats.py` (compares committed `graph_statistic.csv` against paper
Table 2), `check_artefact_inputs.py` (inventories committed datasets / result
`.npz` files / plot-script inputs), plus an interactive smoke test that loads
the only committed graph (`as-skitter`) and runs all five solvers to confirm
they execute and produce sane PPR estimates. The smoke test reproduced
operation counts of the right order of magnitude vs Table 3 for as-skitter
(AESP-LocAPPR ≈1.1e6 vs paper 1.00e6; AESP-LocGD ≈1.6e6 vs paper 1.26e6),
so the solver code itself appears faithful to the method.

The dominant issues are **reproduction artefacts**: only 1 of 19 datasets is
shipped, no `results/*.npz` are committed, the headline Figure 4 reads dead
absolute `/mnt/...` paths, there is no dependency specification, the README is a
generic template describing a non-existent repo layout, and no
download/preprocessing script exists for the 18 missing graphs. None of these
overturn the theory, but they block reproduction of every reported number.

## 2. Traceability table

| Paper artefact | Repo location (computes value) | Reproducible as shipped? | Status |
|---|---|---|---|
| Fig. 2 (init strategies, com-dblp, ops & runtime) | compute `aespAPPR_init` `ppr_solver.py:944-1010` → plotted `plot_init_strategy.py` | needs `datasets/com-dblp/results/{x,y,zero}init_node_20_*.npz` (absent) | MISSING inputs |
| Fig. 3 (C⁰ₕ/ε, vol(S)/γ, R on 19 graphs) | compute in `aespAPPR`; plotted `plot_vol_gamma.py` | needs results for 19 graphs (absent); 18/19 datasets absent | MISSING inputs |
| Fig. 4 (HEADLINE: error vs ops, 4 large graphs) | `plot_from_old_four_plot.py:97-102` reads `/mnt/data/binbin/git/ICML_2025_code_review/results/*` | dead absolute paths; results & datasets absent | MISSING / dead path |
| Fig. 5 (speedup vs α, com-dblp) | `plot_aesp_accelerate_{lineplot,boxplot}.py` ← `datasets/com-dblp/results/` | com-dblp dataset + results absent | MISSING inputs |
| Fig. 6/7 (ASPR/LOCCH/FISTA vs AESP, com-dblp) | `plot_fast_method.py`, `com_dblp_plot_convergence.py` ← com-dblp results | com-dblp dataset + results absent | MISSING inputs |
| Fig. 8 (4 graphs incl. as-skitter) | `plot_convergence.py:20` reads `./datasets/method/results` | wrong/absent path; results absent | MISSING inputs |
| Table 2 (dataset stats n, m) | `graph_statistic.py` → `tables/graph_statistic.csv` | computed; matches paper for 17/19, but m mismatches for ogb-mag240m & ogbn-papers100M | MISMATCH (2 rows) |
| Table 3 (operations, 19 graphs) | `ppr_solver.py` `opers`; no aggregation/table-builder script | per-method `opers` computed; no script assembles Table 3; 18/19 datasets absent | MISSING (table builder + inputs) |
| Table 4 (runtimes, 19 graphs) | `ppr_solver.py` `runtime`; `tables/runtime_operation_data.csv` is com-dblp-only raw rows | no script builds Table 4; 18/19 datasets absent | MISSING (table builder + inputs) |
| R-vs-α (Fig. 3 R column / Table) | `plot_R_with_alpha.py` → `tables/R_alpha_results_com-dblp.csv` | computed for com-dblp (committed CSV); regen needs com-dblp results (absent) + hardcoded `/mnt` write path | committed values present; regen blocked |

Deterministic supports: `_audit_code/out/dataset_stats.csv`,
`_audit_code/out/artefact_inputs.txt`.

## 3. Findings

## missing

```yaml finding
id: results-npz-not-committed
category: missing
topic: "result traceability"
title: "No results/*.npz committed; every figure script reads files that do not exist"
severity: high
confidence: high
status: finding
file: main.py
line_start: 36
line_end: 41
quote: |
      save_dir = os.path.join(g.dataset_path, 'results')
      os.makedirs(save_dir, exist_ok=True)
      
      filename = f'{method}_node_{node}_alpha_{alpha}_eps_{eps}.npz'
      save_path = os.path.join(save_dir, filename)
      np.savez(save_path, **result_data)
claim: "main.py writes per-run results to datasets/<graph>/results/*.npz, and every plotting script loads those .npz files to render Figs. 2-9 and the R/operation tables; the repo commits zero such .npz files (check_artefact_inputs.py: 0 result .npz)."
concern: "None of the paper's figures or per-graph numbers can be regenerated as shipped because the intermediate result artefacts they consume are absent and the only dataset that could regenerate them is as-skitter (1 of 19)."
resolution: "Authors: commit (or provide a download for) the results/*.npz used to produce Figs. 2-9 and Tables 3-4, or a single driver that regenerates them end-to-end."
cross_refs: ["datasets-18-of-19-missing", "fig4-dead-mnt-paths"]
check_script: _audit_code/check_artefact_inputs.py
paper_ref: "Figs. 2-9, Tables 3-4"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: datasets-18-of-19-missing
category: missing
topic: "data availability / preprocessing"
title: "18 of 19 datasets absent and no download/preprocessing script provided"
severity: high
confidence: high
status: finding
file: utils.py
line_start: 21
line_end: 23
quote: |
  class graph:
      def __init__(self, name, path):
          self.adj_m = sp.load_npz(path)
claim: "The graph loader reads a preprocessed *_csr-mat.npz for each dataset, but only datasets/as-skitter/ is shipped (check_artefact_inputs.py: count=1); the paper evaluates on 19 graphs (Table 2), and no script downloads the raw SNAP/OGB graphs or performs the paper's stated preprocessing (treat as undirected, remove self-loops, keep largest connected component) to build the *_csr-mat.npz inputs (grep for download/preprocessing/save_npz code finds none)."
concern: "The headline experiments (incl. Fig. 4 on ogb-mag240m / ogbn-papers100M / com-friendster / wiki-en21) cannot be reproduced because neither the preprocessed graphs nor the code to construct them is present."
resolution: "Authors: add a preprocessing/download script that produces the *_csr-mat.npz files from the public sources, or host the preprocessed graphs."
cross_refs: ["results-npz-not-committed"]
check_script: _audit_code/check_artefact_inputs.py
paper_ref: "Appendix C.1 (Datasets and Preprocessing); Table 2"
tags: [reforms:2, missing:data, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec
category: missing
topic: "expected code completeness"
title: "No requirements.txt / environment file; README references a non-existent one"
severity: medium
confidence: high
status: finding
file: readme.md
line_start: 44
line_end: 49
quote: |
  ```bash
  git clone https://github.com/username/NIPS2025.git
  cd NIPS2025
  pip install -r requirements.txt
  ```
claim: "README instructs `pip install -r requirements.txt`, but no requirements.txt, environment.yml, setup.py, or pyproject.toml exists in the repo (directory listing confirms none)."
concern: "The runtime depends on a specific numba/numpy/scipy stack (numba njit with objmode), and unpinned dependencies mean the environment that produced the paper's timings cannot be rebuilt, which directly affects the runtime claims in Table 4 / Fig. 5."
resolution: "Authors: add a pinned dependency specification (requirements.txt or environment.yml) including numba, numpy, scipy versions."
cross_refs: ["readme-generic-template"]
paper_ref: "Appendix C.2 (implemented in Python with numba)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: tables34-no-builder-script
category: missing
topic: "result traceability"
title: "No script aggregates per-run outputs into Tables 3 and 4"
severity: medium
confidence: medium
status: finding
file: main.py
line_start: 15
line_end: 16
quote: |
      x, errs, opers, runtime, runtime_acc, grad_norms, vols, gammas, eps_t = g.run_method(method, node, alpha, eps)

claim: "Per-run operation counts and runtimes are computed and saved per graph, but no script reads across the 19 graphs to assemble the operation table (Table 3) or runtime table (Table 4); tables/runtime_operation_data.csv holds only raw com-dblp rows, not the per-graph 5-method summary of Tables 3/4."
concern: "The reader cannot trace the specific cell values reported in Tables 3 and 4 to a computation that produces them, so those tables are not independently checkable from the repo."
resolution: "Authors: provide the aggregation script (and inputs) that turns per-run .npz outputs into the exact Table 3 / Table 4 values."
cross_refs: ["results-npz-not-committed", "datasets-18-of-19-missing"]
paper_ref: "Appendix C.3, Tables 3 and 4"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: fig4-dead-mnt-paths
category: bug
topic: "reproducibility / hardcoded paths"
title: "Headline Figure 4 plotting script reads dead absolute /mnt paths"
severity: medium
confidence: high
status: finding
file: plot_from_old_four_plot.py
line_start: 97
line_end: 102
quote: |
  path = '/mnt/data/binbin/git/ICML_2025_code_review/results/ogb-mag240m'
  paths = [
      '/mnt/data/binbin/git/ICML_2025_code_review/results/ogb-mag240m',
      '/mnt/data/binbin/git/ICML_2025_code_review/results/ogbn-papers100M',
      '/mnt/data/binbin/git/ICML_2025_code_review/results/com-friendster',
      '/mnt/data/binbin/git/ICML_2025_code_review/results/wiki-en21',
claim: "The script that renders Figure 4 (the headline four-large-graph convergence comparison) loads results from absolute paths under /mnt/data/binbin/git/ICML_2025_code_review/results/ that exist only on an author machine."
concern: "Figure 4 cannot be regenerated by any user of the repo: the input directories are author-local and not portable, and the same absolute write path is hardcoded in plot_R_with_alpha.py:227 (tables_dir = '/mnt/data/binbin/git/NIPS2025/tables')."
resolution: "Authors: replace the hardcoded /mnt absolute paths with repo-relative paths and ship the corresponding results, so Fig. 4 is reproducible."
cross_refs: ["results-npz-not-committed"]
paper_ref: "Figure 4"
tags: [reforms:2, lones:stage-7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: dataset-stats-m-mismatch
category: difference
topic: "dataset statistics"
title: "Committed edge counts for ogb-mag240m and ogbn-papers100M differ from paper Table 2"
severity: low
confidence: high
status: finding
file: out/dataset_stats.csv
csv_row: 8
quote: |
  ogb-mag240m,244160499,244160499,True,1727235912,1728364232,False
claim: "graph_statistic.py recomputes n and m via nnz//2 and writes tables/graph_statistic.csv; the committed CSV gives m(ogb-mag240m)=1727235912 and m(ogbn-papers100M)=1614061934, whereas paper Table 2 reports 1728364232 and 1615685450 respectively (n matches for both; 17/19 graphs match exactly)."
concern: "The two largest graphs' edge counts in the committed statistics file disagree with the paper's Table 2, indicating the committed npz/CSV was produced from a slightly different preprocessing of those graphs than the version used in the paper (the procedure itself is valid)."
resolution: "Authors: confirm which preprocessing (self-loop removal / LCC extraction) produced the Table 2 values and align the committed graph_statistic.csv, or note the source of the discrepancy."
cross_refs: []
check_script: _audit_code/check_dataset_stats.py
paper_ref: "Table 2, ogb-mag240m and ogbn-papers100M rows"
tags: [forensics:git-archaeology]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: readme-generic-template
category: difference
topic: "repository provenance"
title: "README is a generic template describing a repo layout and API that do not exist"
severity: low
confidence: high
status: finding
file: readme.md
line_start: 55
line_end: 65
quote: |
  ```python
  from src.accelerated_ppr import AcceleratedPPR

  # Load your graph
  G = load_graph('path/to/graph')

  # Initialize the accelerated PPR solver
  solver = AcceleratedPPR(damping_factor=0.85, epsilon=1e-6)

  # Compute PPR vector
  ppr_vector = solver.compute_ppr(G, source_node)
claim: "The README documents a src/ / experiments/ / data/ / results/ layout, a `from src.accelerated_ppr import AcceleratedPPR` API, and entrypoints experiments/run_synthetic.py and experiments/run_realworld.py; none of these directories, modules, or scripts exist (the real entrypoint is main.py + ppr_solver.py + plot_*.py)."
concern: "A user following the README cannot run anything, and the mismatch obscures what the actual reproduction commands are (these live only in experiment_setting.md)."
resolution: "Authors: rewrite the README to match the actual file layout and provide the exact reproduction commands and a results table."
cross_refs: ["no-dependency-spec"]
paper_ref: "n/a"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A — no methodological-validity defect found. This is a theory paper whose
empirical section is a convergence/operation-count comparison of deterministic
local solvers (no train/test split, no learned model, no labels, no statistical
test). Topics that would normally feed `methodology` are structurally
inapplicable: data splitting, sample independence, target/shortcut leakage,
pretraining contamination, temporal integrity, hyperparameter tuning on a test
set, and multiple-comparison statistics are all N/A (deterministic numerical
algorithm, exact ground-truth `opt_x` computed to high precision per source
node). The implemented solver code I smoke-tested on as-skitter produces PPR
estimates and operation counts consistent in order of magnitude with the
paper's Table 3, i.e. no evidence the procedure is invalid.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 4          | high         | 18/19 datasets + all result .npz absent; no deps; no table builder |
| bug         | 1          | medium       | Fig. 4 script reads dead author-local /mnt paths           |
| difference  | 2          | low          | 2 edge-count mismatches vs Table 2; README is a stale template |
| methodology | 0          | -            | N/A — deterministic solver, no learning/eval-validity surface |

## Top take-aways (≤6)

1. **(missing, high)** No `results/*.npz` are committed, so none of Figs. 2-9 or
   the per-graph tables can be regenerated as shipped — `results-npz-not-committed`.
2. **(missing, high)** Only 1 of 19 datasets is shipped and there is no
   download/preprocessing script for the rest, blocking the headline Fig. 4
   experiments — `datasets-18-of-19-missing`.
3. **(bug, medium)** The headline Figure 4 plotting script reads dead absolute
   `/mnt/data/binbin/...` paths and cannot run for any user — `fig4-dead-mnt-paths`.
4. **(missing, medium)** No dependency specification despite a numba-version-
   sensitive runtime and a README that says `pip install -r requirements.txt`
   — `no-dependency-spec`.
5. **(missing, medium)** No script assembles the per-graph Tables 3/4 from
   per-run outputs — `tables34-no-builder-script`.
6. **(difference, low)** Committed edge counts for ogb-mag240m and
   ogbn-papers100M disagree with paper Table 2 — `dataset-stats-m-mismatch`.

## Items that genuinely look fine

- The five solver routines in `ppr_solver.py` run and produce sane PPR
  estimates; on the one shipped graph (as-skitter) the AESP operation counts are
  the same order of magnitude as paper Table 3.
- The only shipped graph (`as-skitter`) loads cleanly, is symmetric, self-loop-
  free, has no isolated nodes, and its (n, m) match Table 2 exactly.
- Ground truth (`opt_x`) is computed per source node by a high-precision APPR
  run (`utils.py:42-57`, `opt_eps = 1e-5/n`), so the reported errors are against
  a legitimate reference, not a leaked/in-sample target.
- Randomness is seeded (`main.py:7` `np.random.seed(2025)`, `main.py:78`
  `random.seed(2025)`); the algorithms themselves are deterministic.

## Open questions for the authors

- Were the Fig. 4 results for the four large graphs produced from the
  `/mnt/.../ICML_2025_code_review/results/` directory referenced in
  `plot_from_old_four_plot.py`, and can those `.npz` (or a regeneration path) be
  shared? (`fig4-dead-mnt-paths`)
- What preprocessing produced the Table 2 edge counts for ogb-mag240m and
  ogbn-papers100M, given the committed `graph_statistic.csv` reports different
  values? (`dataset-stats-m-mismatch`)
