# Code Audit — MIGN: Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting (NeurIPS 2025, paper 760)

## 1. Summary

The repository `compasszzn__MIGN` is the official PyTorch-Lightning implementation of MIGN.
Its core consists of: an entrypoint `main.py`; the MIGN Lightning module `models/model.py` +
network `models/hgnn_gcn_edge_wo_sh.py`; a baseline Lightning module `models/model_st.py`
wrapping ~25 spatiotemporal GNN baselines; DGL/PyG datasets in `dataset/dataset.py`; a 4-step
data-processing pipeline in `data/process_data/`; and two vendored third-party libraries under
`baseline/` (`locationencoder`, `pytorch_geometric_temporal`) that I treated as out-of-scope
dependencies. I read the core training/eval/data code, the configs, the loss/criterion code, and
the data-processing scripts; cross-checked them against the paper (Tables 1–5, §4, Appendix A.6/A.7);
verified the Hugging Face dataset link; and ran two read-only static checks under `_audit_code/`
(`check_missing_protocol.py`, `check_climatology_scope.py`).

Overall the main per-variable forecasting result (Table 1) traces to runnable code. The two main
gaps are: (a) the generalization experiment (Table 4 / Fig 3), which the paper describes as a random
half-station disjoint split, has no implementation in the repo, and (b) the Persistence baseline that
appears in every results table has no code. A low-severity normalization-leakage issue (climatology
fit over all years including test) was also confirmed by static check.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1: MIGN per-variable MSE/MAE (2017-22 train / 2023 val / 2024 test) | `main.py` + `models/model.py` (`test_step`, MSE/MAE/RMSE) + `models/hgnn_gcn_edge_wo_sh.py`; split via `configs/hgnn_gcn_edge.yaml` train/val/test years | not run (needs 14 GB data + GPU) | code present | Verified (code path exists) |
| Table 1: 13 baselines (STGCN…DualCast) | `models/model_st.py` (model registry) + `dataset/dataset.py:STDataset` | not run | code present | Verified (code path exists) |
| Tables 1,2,4 & Figs 4,6: **Persistence** baseline | (none) | — | — | MISSING (no persistence code) |
| Table 2: multistep (3-in/4-out) MSE | `data/process_data/step3_generate_graph_dgl_multi_step.py` + same modules with `--input_length 3 --output_length 4`; per-day logging `model.py:162-164` | not run | code present | Verified (code path exists) |
| Table 3: ablations (w/o mesh, w/o SH, w/o enc/dec SH) | `main.py` flags `--sh_before/--sh_after`; `hgnn_gcn_edge_wo_sh.py:32-44,49-57` branch on SH | not run | code present | Verified (toggles present) |
| Table 4 / Fig 3: **global generalization** (random half stations 2017-23 train, unseen half 2024 test) | (none) | — | — | MISSING (no half-station split code) |
| Table 5: SH degree analysis (0,1,2,3) | `--sh_level` arg; `hgnn_gcn_edge_wo_sh.py:33-44` per-degree param dims | not run | code present | Verified |
| Fig 5: refinement-level / neighbor analysis | `--refinement_level`, `--neighbor` via `step3` regen; `main.py:39-40` | not run | code present | Verified |
| "Improvements 13/15/15%" headline (abstract/§4.1) | derived from Table 1 numbers; no script computes the % | — | — | derived (not separately scripted; not required) |
| Normalization (climatology mean/std) | `data/process_data/step2_climatology.py` | computed over all years | — | leakage (see methodology) |

## 3. Findings

## missing

```yaml finding
id: generalization-split-missing
category: missing
topic: "result traceability / data splitting"
title: "Generalization experiment (Table 4 / Fig 3) random half-station split not in repo"
severity: high
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  We randomly sample half of the stations from the year 2017-2022/2023 for training and validation, while using the remaining stations from 2024 as the test set.
claim: "The paper's headline generalization result (Table 4, Fig 3, and a third contribution bullet in the intro) is produced by randomly partitioning stations into two disjoint halves and training on one half (2017-2023) while testing on the unseen half (2024). The shipped code's only split is by year (train_years 2017-2022, val 2023, test 2024 in configs/hgnn_gcn_edge.yaml); no script samples or holds out half the stations, and a static scan of all core .py files (see check script) finds no station-sampling / disjoint-half logic."
concern: "A central claim of the paper (generalization to unseen stations) cannot be reproduced because the procedure that creates the unseen-station split is absent from the repository."
resolution: "Authors: please add the script that builds the random half-station train/test partition for the generalization experiment, including the seed used, so Table 4 and Fig 3 can be reproduced."
cross_refs: []
check_script: _audit_code/check_missing_protocol.py
paper_ref: "Section 4.3 Global Generalization Analysis; Table 4; Figure 3"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: persistence-baseline-missing
category: missing
topic: "baselines / result traceability"
title: "Persistence baseline reported in every results table has no code"
severity: low
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  Persistence
claim: "A 'Persistence' baseline is reported in Tables 1, 2 and 4 and Figures 4 and 6, but no code computes it: a case-insensitive scan of all core .py files (see check script) finds no 'persistence' / last-value predictor, and model_st.py's model registry (lines 41-85) contains no persistence entry."
concern: "The naive lower-bound baseline cannot be reproduced from the repo, and a reader cannot verify the Persistence numbers that anchor task difficulty."
resolution: "Authors: please provide the script that computes the Persistence (last-day-carried-forward) baseline used in Tables 1/2/4 and Figs 4/6."
cross_refs: []
check_script: _audit_code/check_missing_protocol.py
paper_ref: "Table 1 'Persistence' row; Table 2; Table 4; Figures 4 and 6"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: baseline-tuning-procedure-missing
category: missing
topic: "hyperparameter tuning"
title: "Per-variable baseline hyperparameter tuning (Table 7) has no tuning code"
severity: low
confidence: medium
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  For a fair comparison, we tune different hyperparameters for all baselines, finding the best setting for each.
claim: "The paper reports per-variable optimal learning rate / batch size / hidden size for all 13 baselines in Table 7 and states each baseline was tuned, but the repo contains no tuning/sweep driver for the core spatiotemporal baselines: configs/spatial_temporal.yaml hard-codes a single hidden_size/lr, and a scan of core .py files finds no sweep/optuna/ray/hyperopt harness (the only 'tune' matches are the literal filename 'unique_primary_station_ids_<var>_tune.csv')."
concern: "Without the tuning harness and search ranges, the fairness of the baseline comparison (whether baselines were tuned with the same budget as MIGN) cannot be verified or reproduced."
resolution: "Authors: please provide the sweep configuration / search ranges and the script used to select the Table 7 baseline hyperparameters."
cross_refs: []
paper_ref: "Section 4 Implementation; Appendix A.7 / Table 7"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dependencies-unpinned
category: missing
topic: "reproducibility / dependencies"
title: "No requirements/environment file; core deps only loosely pinned in README prose"
severity: low
confidence: high
status: finding
file: README.md
line_start: 22
line_end: 33
quote: |
  # Install PyTorch (CUDA 12.1)
  pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

  # Install DGL (CUDA 12.1 compatible)
  pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html

  # Install additional dependencies
  pip install numpy==1.26.0
  pip install healpy
  pip install lightning
  pip install -U 'wandb>=0.12.10'
  pip install jupyter
claim: "There is no requirements.txt / environment.yml for the core MIGN code; dependencies are listed only as README pip commands, and several packages the code imports (e.g. lightning, healpy, einops, thop, torchstat, torch_geometric, tsl) are unpinned or unlisted (torchstat/thop/einops/tsl appear in imports but not in the README install list)."
concern: "The exact environment cannot be rebuilt deterministically, which can change numerical results and block reproduction."
resolution: "Authors: please add a pinned requirements.txt (or environment.yml) covering all imported packages including einops, thop, torchstat, torch_geometric, torch-spatiotemporal."
cross_refs: []
paper_ref: "README ⚙️ Requirements"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-absolute-paths
category: bug
topic: "reproducibility / hardcoded paths"
title: "Data dirs and run commands hardcode the authors' machine paths"
severity: medium
confidence: high
status: finding
file: configs/hgnn_gcn_edge.yaml
line_start: 7
line_end: 9
quote: |
    data_dir: '/data/zzn/insitu/data'
    embedding_dir: '/data/zzn/insitu/data'
    climatology_dir: '/data/zzn/insitu/data/climatology.npy'
claim: "The data/embedding/climatology directories are hardcoded to the authors' absolute paths (`/data/zzn/insitu/data`) in both configs/hgnn_gcn_edge.yaml and configs/spatial_temporal.yaml, and the README run commands invoke `python /home/zinanzheng/project/MIGN/main.py ...`; there is no CLI flag or env var to override the data root."
concern: "Out of the box the code cannot locate the downloaded data on any other machine; a user must hand-edit multiple config files, and step0-step3 scripts similarly hardcode `/data/zzn/insitu/data` and `/mnt/hda/zzn/realtime/Dataset`."
resolution: "Authors: parameterize the data root (CLI flag / env var) instead of hardcoding it in the YAML configs and the README commands."
cross_refs: []
paper_ref: "README run commands (lines 129-132)"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: hf-dataset-only-one-variable
category: difference
topic: "data availability"
title: "Hugging Face 'processed data' provides only DEWP, not all 6 reported variables"
severity: low
confidence: medium
status: question
file: README.md
line_start: 57
line_end: 66
quote: |
  ## Processed Data

  We provide **processed datasets** for both **baseline models** and **MIGN** at  
  - **spherical level 3**  
  - **HEALPix level 3**

  These datasets are designed for **one-step input → one-step output** prediction tasks.

  You can directly download the processed data from our Hugging Face repository:  
  👉 [compasszzn/MIGN](https://huggingface.co/datasets/compasszzn/MIGN)
claim: "The README presents the Hugging Face repo as the ready-to-use processed dataset, but the HF repo (retrieved 2026-05-30) contains only `pyg_DEWP_20_graph_step_1.zip`, `dgl_neighbor_10_step_1_refine_3.zip`, `DEWP_embeddings_3.pt`, and `3_healpix_embeddings_level_3.pt` — i.e. processed PyG baseline data and SH station-embeddings only for the DEWP variable, whereas the paper reports 6 variables (MAX/MIN/DEWP/SLP/WDSP/MXSPD)."
concern: "Reproducing the baseline tables for the other five variables requires re-running the full raw-data pipeline (download + step0-step4), not just the advertised download; the 'safely skip Data Preparation' note holds only for DEWP."
resolution: "Authors: confirm whether processed PyG data and SH embeddings for the remaining five variables are intended to be on Hugging Face, or document that step4 must be re-run per variable."
cross_refs: []
paper_ref: "README Processed Data section"
url_retrieved_at: "2026-05-30T00:00:00Z"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: climatology-fit-on-all-years
category: methodology
topic: "data splitting / normalization leakage"
title: "Normalization mean/std fit over all years 2017-2024 including val and test"
severity: low
confidence: high
status: finding
file: data/process_data/step2_climatology.py
line_start: 37
line_end: 57
quote: |
        file_paths = Path(os.path.join(args.base_path, node_type))
        csv_files = sorted([
            f for f in file_paths.glob('*.csv') 
        ])
        feature_matrix = np.full((num_stations, len(csv_files)), np.nan)

        for date_idx,file in enumerate(tqdm(csv_files)):

            daily_df = pd.read_csv(file, dtype={'station_id': str})
            daily_df.set_index('station_id', inplace=True)

            for station_idx, station_id in enumerate(station_ids):
                if station_id in daily_df.index:
                    feature_matrix[station_idx, date_idx] = daily_df.loc[station_id, 'observation_value']
                        

        node_data[node_type]['mean'] = np.nanmean(feature_matrix)
        node_data[node_type]['std'] = np.nanstd(feature_matrix)
claim: "step2_climatology.py globs ALL `*.csv` in each variable folder with no year filter and takes np.nanmean/np.nanstd over the whole matrix. step0_filter_feature.py (loop `range(2017, 2025)`) writes every year 2017-2024 into that same folder, so the per-variable mean/std used to z-normalize inputs/labels (dataset.py:46,115,169,250) are computed over the validation (2023) and test (2024) years as well as train."
concern: "The normalization statistics see the test distribution (a form of preprocessing-before-split leakage); because it is a single global scalar mean/std per variable the practical effect is small, but it technically violates the train-only-statistics principle and applies identically to MIGN and all baselines."
resolution: "Authors: recompute climatology using only the 2017-2022 training files and confirm Table 1/2 numbers are unchanged, or clarify that the leak is negligible."
cross_refs: []
check_script: _audit_code/check_climatology_scope.py
paper_ref: "Section 4 Dataset; data/process_data/step2_climatology.py"
tags: [leakage:L1.1, reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Generalization half-station split and Persistence baseline have no code; deps/tuning unspecified. |
| bug         | 1          | medium       | Data roots and run commands hardcode the authors' machine paths. |
| difference  | 1          | low          | HF "processed data" ships only DEWP, not all 6 reported variables. |
| methodology | 1          | low          | Normalization mean/std fit over all years incl. val/test (minor leakage). |

## 5. Closing lists

### Top take-aways
1. `generalization-split-missing` (missing, high): the random half-station disjoint split behind the paper's central generalization claim (Table 4 / Fig 3) is not implemented anywhere in the repo.
2. `hardcoded-absolute-paths` (bug, medium): configs and README commands hardcode `/data/zzn/insitu/data` and `/home/zinanzheng/...`; the code cannot find data on another machine without manual edits.
3. `persistence-baseline-missing` (missing, low): the Persistence baseline reported in every results table has no code.
4. `baseline-tuning-procedure-missing` (missing, low): Table 7 per-variable baseline tuning is claimed but no sweep harness is shipped, so comparison fairness is unverifiable.
5. `climatology-fit-on-all-years` (methodology, low): normalization statistics are fit over all years including val/test (minor, symmetric leakage).
6. `dependencies-unpinned` (missing, low): no requirements/environment file; several imported packages unlisted/unpinned.

### Items that genuinely look fine
- Year-based main split is consistent between paper (§4: 2017-22 train, 2023 val, 2024 test) and code (configs/hgnn_gcn_edge.yaml lines 3-5; STDataset index splits 0:2192 / 2192:2557 / 2557:2922 = 6yr/1yr/1yr).
- MSE/MAE/RMSE in utils/criterion.py correctly mask NaN targets before averaging; the test_step de-normalizes predictions and labels symmetrically (model.py:142,151; model_st.py:152-153) before computing reported metrics, so metrics are in physical units.
- Checkpoint selection and early stopping monitor `val_loss` (main.py:49-58), not test loss — no test-set model selection leakage.
- The decoder skip term `space_2` (hgnn_gcn_edge_wo_sh.py:295-297) uses the station's own INPUT value `t{input_length-1}` (a residual from available input), not a future/target value — not leakage.
- SH degree default in code is 3 (`--sh_level` default) while the paper's headline uses degree 2, but `--sh_level 2` is supported via CLI, so this is a tunable setting, not a defect.

### Open questions for the authors
- For the generalization experiment: what seed and exact procedure produced the random half-station split, and where is that script (needed for Table 4 / Fig 3)?
- Are processed baseline data and SH embeddings for the five non-DEWP variables meant to be on Hugging Face, or must step4 be re-run per variable?
- Does recomputing climatology on training years only leave Tables 1/2 unchanged (confirming the normalization leak is negligible)?
