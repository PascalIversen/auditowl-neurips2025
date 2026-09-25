# Code audit — GC-xLSTM (NeurIPS 2025, paper 570)

## Summary

The repository `code/harpoonix__GC-xLSTM/` is the authors' own code for *GC-xLSTM*
(github.com/harpoonix/GC-xLSTM). It contains the proposed model
(`GC-xLSTM/models/clstm.py` → `componentXLSTM` + `train_model_ista`), a vendored
copy of the NX-AI xLSTM library (`xlstm/`), the synthetic-data generators
(`GC-xLSTM/synthetic.py` for Lorenz-96 and VAR), a data loader
(`GC-xLSTM/prepare_data.py`), a single train+eval driver
(`GC-xLSTM/xlstm_neural_gc.py`), nine per-dataset configs
(`GC-xLSTM/configs/`), and on-disk fMRI / MoCap / Moléne data
(`GC-xLSTM/datasets/`). Dependency specs are present
(`environment_pt220cu121.yaml`, `pyproject.toml`).

I read the paper (Tables 1-4, ablation Table 3, the algorithm in Appendix A, and
the dataset table in Appendix C), then traced every headline number to the code
that would compute it. I read the driver, the trainer/model
(`models/clstm.py`), the data loaders, the LR/penalty schedulers, the configs,
and the plotting utilities. I wrote one read-only deterministic check,
`_audit_code/check_artefacts.py` (output `_audit_code/out/artefacts.csv`), which
verifies: (a) which imported modules are absent, (b) whether any AUROC is
computed anywhere in the repo, (c) whether the Table-3 ablation forecaster/
optimizer have a runnable path, and (d) hardcoded absolute paths. I did not run
the model (it needs CUDA + an A6000-class GPU and crashes at import as shipped;
see `missing-amcparser-plot-module`).

Headline result: this **is** the author code (replacing the earlier "no code"
verdict, which was wrong). The single train+eval entrypoint exists and the
Acc./BA metrics for Lorenz-96 and the synthetic VAR trace to code. However three
reproducibility gaps stand out: (1) the **AUROC** column of Table 1 has no
producing code anywhere in the repo; (2) the **Table-3 ablations** (LSTM
forecaster; Group-Lasso optimization) have no config or driver path — the driver
hardwires the full xLSTM + alpha-loss method; and (3) the driver/loader crash on
import because of two missing modules. Methodologically, GC recovery is scored
in-sample against the same ground-truth graph that also drives per-dataset λ
tuning and a "best-accuracy" checkpoint selection.

## Traceability table (Rule G)

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| Table 1, Lorenz F=10/40 — **Acc.** (e.g. 99.1 / 96.3) | `models/clstm.py:661` (`np.mean(predicted_gc==true_GC)`), driver `xlstm_neural_gc.py:275` | value computed in-sample | not numerically re-run | Verified (code present) |
| Table 1, Lorenz F=10/40 — **BA** (e.g. 98.5 / 96.6) | `models/clstm.py:23-33,662`; driver `:276` | value computed in-sample | not numerically re-run | Verified (code present) |
| Table 1, Lorenz F=10/40 — **AUROC** (e.g. 99.3 / 88.0) | (none — only per-λ `tpr,fpr` rows written to a hardcoded CSV at driver `:285-288`; nothing computes AUROC) | — | — | MISSING (no AUROC code) |
| Table 1 — baseline rows (VAR, cLSTM, cMLP, GC-KAN, TCDF, eSRU, GVAR) | (none; "taken as reported in the original papers", paper §4.1) | — | — | N/A (cited, not recomputed) |
| Table 2, fMRI **BA** (GC-xLSTM 73.3) | `models/clstm.py:662`; driver `:276`; data `datasets/fMRI/*` via `prepare_data.py:7-28` | value computed in-sample | not re-run | Verified (code present) |
| Table 2 — fMRI baseline rows | (none; reported from prior work) | — | — | N/A |
| Table 3 ablation (I): **LSTM** forecaster + Joint opt | (none; driver always builds `componentXLSTM`, `xlstm_neural_gc.py:81`; no config selects the `LSTM`/`cLSTM` class) | — | — | MISSING (no ablation path) |
| Table 3 ablation (II): xLSTM + **Group Lasso** opt | (none; `regularize()` group-lasso fn `clstm.py:363-370` is dead code, only in commented line `:640`; no config) | — | — | MISSING (no ablation path) |
| Table 4, VAR BA per (T, V) | `synthetic.py:20-47` (VAR + GC truth); `models/clstm.py:662`; driver `:276`; `use_lags` path `clstm.py:271-275` | value computed in-sample | not re-run | Verified (code present) |
| Fig. 3 (Moléne GC maps, λ=8/10) | `prepare_data.py:30-84`; driver `:294-312` (`create_weather_map`) | qualitative graph | n/a (no metric) | Verified (plot code present) |
| Fig. 4 (MoCap GC graphs) | driver `:314-319` calls `plot_graph_from_GC` | qualitative graph | n/a | MISSING plotter (`missing-amcparser-plot-module`) |
| Fig. 5 (Lorenz F=40 GC vs truth) | `show_gc_util.py:59-89` (`show_gc_actual_and_estimated`) | qualitative graph | n/a | Verified (plot code present) |
| Fig. 6 / Fig. 7 (Company Fundamentals) | `prepare_data.py:115-118` calls `fetch_acatis_data` from a module absent from the repo | — | — | MISSING data+module (ACATIS proprietary, declared) |
| Fig. 8 (scaling time/memory vs V) | driver `:101-126` logs train time + `max_memory_reserved`; no aggregation/plot script in repo | partial | — | MISSING (plot/aggregation absent) |
| Fig. 9 (loss components / variable usage) | driver `:152-198` plots `train/pred/alpha_loss_list`, `var_usage_list` from `clstm.py:646-660` | values computed | n/a | Verified (plot code present) |

## missing

```yaml finding
id: missing-auroc-computation
category: missing
topic: "result traceability"
title: "No code computes the AUROC reported in Table 1"
severity: high
confidence: high
status: finding
file: GC-xLSTM/xlstm_neural_gc.py
line_start: 284
line_end: 288
quote: |
    # save tpr and fpr to a file /home/harsh/xlstm/Neural-GC/exp/lorenz-aucroc/lorenz-tpr-fpr.csv
    with open(f"/home/harsh/xlstm/Neural-GC/exp/{cfg.dataset.name}/tpr-fpr.csv", "a") as f:
        f.write(f"{cfg.training.lam},{tpr},{fpr}\n")
    with open(f"/home/harsh/xlstm/Neural-GC/exp/{cfg.dataset.name}/accuracy.csv", "a") as f:
        f.write(f"{cfg.training.lam},{acc},{bal_acc}\n")
claim: "The driver appends one (lam, tpr, fpr) row per run to a CSV, but no script in the repository reads those rows and computes an AUROC; a repo-wide grep for roc_auc/auroc/trapz/metrics.auc/roc_curve returns nothing (see _audit_code/out/artefacts.csv, row auroc_computation_found=False)."
concern: "Table 1 reports AUROC for every model on Lorenz-96 (e.g. GC-xLSTM 99.3 at F=10, 88.0 at F=40) but the value that produces those numbers — sweeping lambda in {5..15} and integrating tpr/fpr into an ROC area, as the paper describes in Section 4 — is not present, so the AUROC column cannot be reproduced from the repo."
resolution: "Authors: please add the script that reads the per-lambda tpr/fpr CSVs and computes/aggregates AUROC (with the threshold/sweep convention used), or point to where it lives."
cross_refs: ["hardcoded-abs-output-paths"]
check_script: _audit_code/check_artefacts.py
paper_ref: "Table 1, AUROC columns; §4 'we compute all AUROC scores by sweeping over lambda in {5,...,15}'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-table3-ablations
category: missing
topic: "ablations"
title: "Table 3 ablations (LSTM forecaster; Group-Lasso) have no runnable path"
severity: high
confidence: high
status: finding
file: GC-xLSTM/xlstm_neural_gc.py
line_start: 81
line_end: 83
quote: |
  xlstm: componentXLSTM = componentXLSTM(
      X.shape[-1], hidden=cfg.model.embedding_dim, config=config
  ).cuda(device=device)
claim: "The driver unconditionally instantiates componentXLSTM (xLSTM forecaster) and trains it with train_model_ista, which only uses the alpha-loss / proximal optimisation; no config flag or code path selects the plain LSTM/cLSTM forecaster (ablation I) or the standard Group-Lasso optimiser (ablation II). _audit_code/out/artefacts.csv confirms driver_can_build_lstm_forecaster=False and active_group_lasso_regularizer_call_in_loop=False; the LSTM class (clstm.py:35) and the group-lasso regularize() (clstm.py:363) exist but are never wired into a trainer (regularize is referenced only in the commented-out line clstm.py:640)."
concern: "Table 3 attributes GC-xLSTM's gains to the xLSTM architecture (row I) and the joint optimisation (row II) via two ablations, but neither ablated configuration can be produced by the shipped code, so the central design-justification claims are not reproducible."
resolution: "Authors: please provide the configs/scripts that run the LSTM-forecaster and Group-Lasso variants used for Table 3, or confirm they were run from off-repo code."
cross_refs: []
check_script: _audit_code/check_artefacts.py
paper_ref: "Table 3, rows (I) LSTM/Joint and (II) xLSTM/Group Lasso"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-amcparser-plot-module
category: missing
topic: "missing module / runnability"
title: "Driver's top-level import targets a MoCap plotting module not in the repo"
severity: high
confidence: high
status: finding
file: GC-xLSTM/xlstm_neural_gc.py
line_start: 15
line_end: 15
quote: |
  from datasets.mocap.all_asfamc.AMCParser.plot_motion_gc import plot_graph_from_GC
claim: "This is an unconditional top-level import, but datasets/mocap/all_asfamc/AMCParser/ (and plot_motion_gc.py) do not exist in the repo — only the two .npz files exist under all_asfamc/ (verified: _audit_code/out/artefacts.csv amcparser_plot_motion_gc_module_present=False)."
concern: "Because the import is at module top level, the train+eval driver raises ModuleNotFoundError before any training starts for EVERY dataset (not just MoCap), so the repository as shipped does not run."
resolution: "Authors: please add the AMCParser/plot_motion_gc.py module (or move the import inside the 'mocap' branch at line 315) so the driver runs for the other datasets."
cross_refs: ["missing-acatis-prepare-module"]
check_script: _audit_code/check_artefacts.py
paper_ref: "Figure 4 (MoCap GC plots)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-acatis-prepare-module
category: missing
topic: "missing module / runnability"
title: "prepare_data top-level import targets an ACATIS module not in the repo"
severity: medium
confidence: high
status: finding
file: GC-xLSTM/prepare_data.py
line_start: 5
line_end: 5
quote: |
  from datasets.acatis.prepare_acatis_data import fetch_acatis_data
claim: "prepare_data.py imports fetch_acatis_data at module top level, but datasets/acatis/ does not exist in the repo (_audit_code/out/artefacts.csv acatis_prepare_module_present=False); prepare_data is imported by the driver at xlstm_neural_gc.py:14, so this import must succeed for any dataset."
concern: "The ACATIS data itself is legitimately withheld (README: 'all datasets ... except ACATIS, which is not open-source'), but the unconditional import of its loader breaks prepare_data — and hence the driver — for ALL datasets, not just ACATIS; the loader stub should be guarded or the module shipped."
resolution: "Authors: please ship a stub datasets/acatis/prepare_acatis_data.py (raising a clear 'data not available' error), or move the import inside the 'acatis' branch (line 115), so the open datasets remain runnable."
cross_refs: ["missing-amcparser-plot-module"]
check_script: _audit_code/check_artefacts.py
paper_ref: "Figure 6/7 (Company Fundamentals); README 'Datasets'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-abs-output-paths
category: bug
topic: "hardcoded paths"
title: "Hardcoded /home/harsh absolute paths in loader, configs, and CSV writes"
severity: medium
confidence: high
status: finding
file: GC-xLSTM/prepare_data.py
line_start: 32
line_end: 35
quote: |
    df = pd.read_csv("/home/harsh/xlstm/Neural-GC/datasets/molene/Original_Data/aggregated_data.csv")

    # Load the weather stations data
    weather_stations_df = pd.read_csv("/home/harsh/xlstm/Neural-GC/datasets/molene/Original_Data/weather_stations.csv")
claim: "fetch_molene_data reads the Moléne CSVs from absolute paths /home/harsh/xlstm/Neural-GC/... rather than the in-repo relative path datasets/molene/Original_Data/ (the files DO exist in the repo). The same /home/harsh hardcoding appears in fmri.yaml:44-45, mocap-run.yaml:44, mocap-salsa.yaml:44, and in the driver's CSV writes xlstm_neural_gc.py:285-288 (full list in _audit_code/out/artefacts.csv, hardcoded_home_harsh_paths)."
concern: "On any machine other than the authors', the Moléne run raises FileNotFoundError (and the driver's per-run accuracy/tpr-fpr CSV writes fail), even though the data ships with the repo; these paths must be edited by hand before anything runs."
resolution: "Authors: please replace the absolute /home/harsh/... paths with repo-relative paths (e.g. datasets/molene/Original_Data/...) in prepare_data.py, the configs, and the CSV-write sinks."
cross_refs: ["missing-auroc-computation"]
check_script: _audit_code/check_artefacts.py
paper_ref: "Moléne dataset (§4.1, Fig. 3)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: molene-per-timestep-normalization
category: difference
topic: "preprocessing"
title: "Moléne min-max normalisation is across stations per timestep, not per series"
severity: low
confidence: medium
status: finding
file: GC-xLSTM/prepare_data.py
line_start: 69
line_end: 72
quote: |
    # Normalize the time series along each row using min-max scaling
    min_vals = np.min(time_series, axis=1, keepdims=True)
    max_vals = np.max(time_series, axis=1, keepdims=True)
    time_series = (time_series - min_vals) / (max_vals - min_vals)
claim: "time_series has shape (T, p) (rows = timesteps, cols = stations); axis=1 takes the min/max ACROSS the p stations at each timestep, so every row is rescaled by that timestep's spatial min/max rather than each station being normalised over time."
concern: "This couples all stations through a shared per-timestep denominator (a cross-variate transform applied before GC estimation), which is an unusual choice for a method meant to learn inter-station Granger structure from raw temperature; the paper does not describe this normalisation, and it is used only for the qualitative Moléne figures (no metric)."
resolution: "Authors: confirm whether per-timestep cross-station min-max was intended for Moléne, and whether it affects the learned GC maps in Fig. 3 versus per-series normalisation."
cross_refs: []
paper_ref: "§4.1 Moléne; Appendix C"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: tuning-and-selection-against-eval-graph
category: methodology
topic: "hyperparameter tuning / model selection"
title: "Per-dataset lambda and best-accuracy checkpoint selected against the same GC graph used for scoring"
severity: high
confidence: medium
status: finding
file: GC-xLSTM/models/clstm.py
line_start: 661
line_end: 669
quote: |
                accuracy = 100 * np.mean(predicted_gc.cpu().data.numpy() == true_GC)
                bal_acc = 100 * calculate_balanced_accuracy(predicted_gc.cpu().data.numpy(), true_GC)
                logger.info("Accuracy = %.2f%%" % accuracy)
                logger.info("Balanced accuracy = %.2f%%" % bal_acc)
                balanced_accuracy_list.append(bal_acc)
                accuracy_list.append(accuracy)
                if (best_accuracy_gc is None) or (accuracy == np.max(accuracy_list)):
                    best_accuracy_gc = clstm.GC(threshold=False)
                    best_accuracy_model = deepcopy(clstm)
claim: "During training, every check_every steps the code compares the current learned graph to the ground-truth true_GC and keeps the checkpoint (best_accuracy_gc / best_accuracy_model) whose graph best matches true_GC; the driver also reports Acc./BA from a learned graph (xlstm_neural_gc.py:275-276) and the paper states the sparsity hyperparameter lambda 'was tuned specifically for each setting' (§4) against this same single realisation. There is no held-out series or independent selection criterion — the same ground-truth graph is the tuning target and the evaluation target."
concern: "Selecting lambda per dataset and tracking a 'best-accuracy' checkpoint by direct comparison to the evaluation ground truth optimises the reported metric on the answer key, which can inflate Acc./BA relative to a protocol where the selection criterion is independent of the scored labels; the gap is unquantified."
resolution: "Authors: clarify whether the reported Table 1/2/4 numbers come from the best-loss checkpoint (restored at clstm.py:688) or the best-accuracy checkpoint, and report lambda chosen by a criterion that does not use the scored ground-truth graph (or quantify the difference)."
cross_refs: ["missing-table3-ablations"]
paper_ref: "Table 1/2/4; §4 'Only the sparsity hyperparameter lambda was tuned specifically for each setting'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 4 | high | AUROC code, Table-3 ablations, and two imported modules all absent |
| bug | 1 | medium | Hardcoded /home/harsh paths break loader, configs, and CSV writes |
| difference | 1 | low | Moléne normalised across stations per timestep (paper silent) |
| methodology | 1 | high | lambda + best-accuracy checkpoint selected against the scored GC graph |

## Top take-aways

1. **No AUROC code** (`missing-auroc-computation`, missing/high) — Table 1's
   AUROC column has no producing script; only per-λ tpr/fpr are dumped to a
   hardcoded CSV and never aggregated.
2. **Table-3 ablations not runnable** (`missing-table3-ablations`,
   missing/high) — the LSTM-forecaster and Group-Lasso variants that justify the
   two key design choices have no config or driver path.
3. **Driver crashes at import** (`missing-amcparser-plot-module`, missing/high) —
   a top-level import of a MoCap plotting module that is absent stops every run
   before training, regardless of dataset.
4. **Tuning/selection on the answer key** (`tuning-and-selection-against-eval-graph`,
   methodology/high, medium confidence) — per-dataset λ and the "best-accuracy"
   checkpoint are chosen by direct comparison to the same ground-truth graph the
   metric scores, with no independent selection set.
5. **ACATIS loader import breaks all datasets** (`missing-acatis-prepare-module`,
   missing/medium) — the proprietary data is legitimately withheld, but its
   unconditional top-level import breaks `prepare_data` for the open datasets too.
6. **Hardcoded absolute paths** (`hardcoded-abs-output-paths`, bug/medium) —
   `/home/harsh/...` paths in the loader, three configs, and the CSV sinks must
   be hand-edited before anything runs.

## Items that genuinely look fine

- **The Acc./BA metric is implemented and traceable** for Lorenz-96, fMRI, and
  VAR (`models/clstm.py:23-33,661-662`; driver `:267-276`); the BA formula
  `(TPR + (1−FPR))/2` is consistent between the trainer and the driver.
- **Synthetic ground truth is generated by code**, not hand-specified:
  Lorenz-96 GC (`synthetic.py:71-78`) and the 2-step VAR GC
  (`synthetic.py:24-34`) match the paper's dataset description (Appendix C).
- **Dependencies are specified** (`environment_pt220cu121.yaml`, `pyproject.toml`);
  the earlier "dependencies absent" concern does not hold.
- **The variable-number-of-lags extension (Table 4) is implemented**
  (`clstm.py:271-275` `use_lags` path; `Projection`/rank-3 weights at
  `clstm.py:93-119`), matching §4.2.
- **fMRI / MoCap / Moléne raw data ship with the repo** under `datasets/`;
  only their wiring (paths) and one plotter are broken.

## Open questions for the authors

- Which checkpoint produced the reported Table 1/2/4 numbers — best-loss
  (restored, `clstm.py:688-690`) or best-accuracy (`clstm.py:668-669`)? This
  determines the severity of `tuning-and-selection-against-eval-graph`.
- Where is the AUROC aggregation script and the exact λ-sweep / ROC-area
  convention used for Table 1?
- Can the Table-3 ablation configurations (LSTM forecaster; Group Lasso) be
  released so the architecture- and optimisation-attribution claims are
  reproducible?
