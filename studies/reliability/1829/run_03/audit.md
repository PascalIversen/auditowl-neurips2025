# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (#1829)

## 1. Summary

The repository `code/jackyue1994__OLinear/` is the official implementation of OLinear, a
linear multivariate time-series forecaster built on the Time-Series-Library / iTransformer
framework. Its central contribution is **OrthoTrans**: each input window is projected onto the
eigenvectors of the *temporal Pearson correlation matrix* (a precomputed orthogonal matrix `Q`),
encoded/decoded by linear layers (a "NormLin" cross-series layer + an intra-series learner), then
projected back with an output `Q`. A variant **OLinear-C** replaces the learnable NormLin weights
with `Softmax(CorrMatv)`, the precomputed across-channel correlation matrix.

The repo contains training code (`run.py`, `experiments/exp_forecast.py`), the model
(`model/OLinear.py`, `model/OLinear_C.py`, ablation variants), the data loaders
(`data_provider/data_loader.py`), per-dataset run scripts (`scripts/`), 11 of the 24 datasets'
raw CSVs plus their precomputed `Q` matrices under `dataset/`, and a notebook
(`dataset/Generate_corrmat.ipynb`) that regenerates the `Q` and channel-correlation matrices.

What I did / ran (all read-only, under `_audit_code/`):
- `check_q_train_only.py` — verified a shipped temporal `Q` matrix is orthonormal and is
  **reproduced exactly from the first-70% (train) slice only**, confirming no test data enters
  the OrthoTrans basis (the paper's `Xtrain` claim).
- `check_channel_files_present.py` — counted the `*_COV_channel_*.npy` files required by the
  OLinear-C scripts (24) vs shipped (1).
- `check_qmat_for_shipped_datasets.py` — confirmed every shipped dataset has its full set of
  temporal `Q` matrices.
- Manually traced the train/val/test splits, the scaler-fit scope, the early-stopping / checkpoint
  selection, the loss function, and the multi-stage test-time evaluation block.

Headline conclusion: the core method (OrthoTrans + NormLin) is implemented faithfully and the
preprocessing/splitting is leakage-free for the datasets shipped. The findings below are
reproducibility / completeness gaps (unpinned deps; most OLinear-C channel matrices absent), not
defects that invalidate the main OLinear numbers.

## 2. Result-traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2/3 long/short-term MSE/MAE for the 11 shipped datasets (cars, covid, DowJones, ILI, nasdaq, power, SP500, unemployment, weather, website, wiki) | `run.py` + `experiments/exp_forecast.py` + `scripts/OLinear/**`, `scripts/OLinear_C/**`; data+Q matrices in `dataset/<name>/` | not re-run (needs GPU training) | code present, data present | Computable (code+data+Q present) |
| Table 2/3 MSE/MAE for ECL, ETT(×4), Traffic, Solar, PEMS(×4), Exchange, METR-LA | code present; raw CSVs and their `Q`/channel `.npy` **not in repo** (README → Google Drive / Tsinghua Cloud / Appendix B) | — | — | Needs external data download + Q regen (see avail-statement-external-data) |
| Table 4 / Table 20 transformation-basis ablation (Fourier, wavelet, Chebyshev, Laguerre, Legendre, Identity) | `model/orthoLinear_basis/OLinear_{FFT,wavelet,wavelet2,cheby,Laguerre,Legendre}.py`, `scripts/ablation/basis/**` | not re-run | code present | Computable |
| Table (var/temp) model ablation | `model/OLinear_ablation_var_temp.py`, `scripts/ablation/var_temp/**` | not re-run | code present | Computable |
| Attention-variant comparison (NormLin vs self-attention) | `model/OLinear_attn_var.py`, `scripts/ablation/attn_var/**` | not re-run | code present | Computable |
| Table 5 OrthoTrans-as-plugin (iTransformer/PatchTST/RLinear) | no iTransformer/PatchTST/RLinear baseline code in repo (only OLinear variants) | — | — | MISSING (plugin-on-baselines code) |
| OLinear-C results (Appendix H) on 23 of 24 datasets | `model/OLinear_C.py` present, but 23/24 `*_COV_channel_*.npy` absent | — | — | MISSING (olinear-c-channel-mats-missing) |
| Table 13 robustness (std over 7 seeds) | `scripts/OLinear/robust/**`, `scripts/OLinear_C/robust/**` use `--itr 7 --fix_seed 0` | not re-run | harness present | Computable (seeds random, not pinned) |
| Figure 3 / Table 41 efficiency (FLOPs, time, memory) | `thop`/`fvcore` imports in `requirements.txt`; `utils/` | not re-run | tooling present | Computable |
| OrthoTrans basis is train-only / orthonormal | `dataset/Generate_corrmat.ipynb` cell 0; verified in `_audit_code/check_q_train_only.py` | orthonormal err 3.9e-8; train-only reproduced exactly | ✓ (matches "Xtrain") | Verified |

## 3. Findings

### missing

```yaml finding
id: requirements-unpinned
category: missing
topic: "dependencies / reproducibility"
title: "requirements.txt lists 13 packages with no version pins (incl. torch)"
severity: medium
confidence: high
status: finding
file: requirements.txt
line_start: 1
line_end: 13
quote: |
  pandas
  scikit-learn
  numpy
  matplotlib
  torch
  fvcore
  einops
  thop
  timm
  reformer_pytorch
  openpyxl
  seaborn
  pywt
claim: "The dependency specification contains only bare package names with no version constraints for any package, including the deep-learning framework torch."
concern: "The runtime environment cannot be deterministically rebuilt; future versions of torch / numpy / scikit-learn may change defaults (e.g. StandardScaler behaviour, RNG, eigendecomposition sign conventions) and silently shift results, undermining exact reproduction of the reported metrics."
resolution: "Authors: pin exact versions (e.g. torch==2.x, numpy==1.26.x, scikit-learn==1.x) used to produce the paper, ideally with a lockfile or the Python version."
cross_refs: []
paper_ref: "Appendix D (Implementation details); README 'Usage' step 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: olinear-c-channel-mats-missing
category: missing
topic: "result traceability / OLinear-C"
title: "23 of 24 channel-correlation matrices required by OLinear-C scripts are absent"
severity: medium
confidence: high
status: finding
file: model/OLinear_C.py
line_start: 42
line_end: 45
quote: |
        assert configs.q_channel_file is not None, 'configs.q_channel_file should not be None in orthoLinear'
        q_channel_file = os.path.join(configs.root_path, configs.q_channel_file)
        if os.path.isfile(q_channel_file):
            self.channel_corr_mat = torch.from_numpy(np.load(q_channel_file)).to(torch.float32).to(device)
claim: "model/OLinear_C.py loads a precomputed channel-correlation matrix via --q_channel_file (lines 42-45), but only weather_COV_channel_ratio0.70.npy is shipped; the 23 other *_COV_channel_*.npy files named by the OLinear_C scripts (including for datasets whose raw CSV IS in the repo: cars, covid, ILI, power, SP500, etc.) are not present (see _audit_code/out/check_channel_files_present.txt: 'required by OLinear_C scripts: 24 / present in dataset/: 1 / MISSING (23)')."
concern: "The OLinear-C variant (reported in Appendix H) cannot be run out-of-the-box for 23 of 24 datasets; on a missing file OLinear_C.py:47 does not raise (see olinear-c-missing-file-assert), so self.channel_corr_mat is left undefined."
resolution: "Authors: ship the precomputed *_COV_channel_*.npy matrices for all OLinear-C datasets, or document that Generate_corrmat.ipynb (cell 1) must be re-run first (and provide the raw CSVs it needs)."
cross_refs: ["olinear-c-missing-file-assert"]
check_script: _audit_code/check_channel_files_present.py
paper_ref: "Appendix H (OLinear-C); §E.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: plugin-on-baselines-code-missing
category: missing
topic: "result traceability / generality"
title: "Table 5 OrthoTrans-as-plugin (iTransformer/PatchTST/RLinear) has no code in repo"
severity: medium
confidence: medium
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  Table 5: Applying OrthoTrans (O.Trans) to iTransformer, PatchTST and RLinear. Average MSEs are
claim: "The paper reports (Table 5) that adding OrthoTrans to iTransformer, PatchTST and RLinear improves their average MSE; the repo's model/ directory contains only OLinear and its own ablation variants, with no iTransformer / PatchTST / RLinear implementations into which OrthoTrans was inserted."
concern: "The 'OrthoTrans as a plug-in' generality claim (a headline contribution) cannot be reproduced from this repo because the modified-baseline code that produced Table 5 is not included."
resolution: "Authors: release the OrthoTrans-augmented iTransformer/PatchTST/RLinear code used for Table 5, or point to the exact branch/commit where the plug-in was applied."
cross_refs: []
paper_ref: "Table 5; §5.2 'OrthoTrans as a plug-in'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: avail-statement-external-data
category: missing
topic: "data availability"
title: "Headline benchmark datasets (ECL/ETT/Traffic/Solar/PEMS/Exchange/METR) not in repo"
severity: low
confidence: high
status: question
file: README.md
line_start: 48
line_end: 50
quote: |
  2. Some datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).  The others can be obtained from the links in Appendix B. 
  Some datasets are already in the folder ```./dataset```. 
  The python script ```Generate_corrmat.ipynb``` for generating Pearson correlation matrices and Q matrices is also provided in ```./dataset```.
claim: "The repo ships raw CSVs + Q matrices for only 11 of the 24 datasets; the standard long-term-forecasting benchmarks (ECL/electricity, ETT×4, Traffic, Solar-Energy, PEMS×4, Exchange, METR-LA) are not present and must be fetched from Google Drive / Tsinghua Cloud / Appendix B links, then their Q and channel matrices regenerated via Generate_corrmat.ipynb."
concern: "Reproducing the main long-term-forecasting tables for the most prominent benchmarks requires external downloads and a manual Q-matrix regeneration step; these are standard public TSLib datasets and the sources are documented, so this is a completeness gap rather than an invalidation (filed as a question since I cannot resolve the external URLs in this sandbox)."
resolution: "Authors: confirm the external links remain live and that Generate_corrmat.ipynb reproduces the exact Q/channel matrices used for the benchmark datasets (the notebook currently only iterates time_lag in {96,192,336,720}, not the {3..60} horizons used by the smaller datasets)."
cross_refs: []
paper_ref: "Appendix B (datasets); §5 Datasets"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

### bug

```yaml finding
id: olinear-c-missing-file-assert
category: bug
topic: "error handling"
title: "OLinear_C.py uses `assert ValueError(...)` which never raises on a missing channel file"
severity: low
confidence: high
status: finding
file: model/OLinear_C.py
line_start: 42
line_end: 47
quote: |
        assert configs.q_channel_file is not None, 'configs.q_channel_file should not be None in orthoLinear'
        q_channel_file = os.path.join(configs.root_path, configs.q_channel_file)
        if os.path.isfile(q_channel_file):
            self.channel_corr_mat = torch.from_numpy(np.load(q_channel_file)).to(torch.float32).to(device)
        else:
            assert ValueError('self.channel_corr_mat should not be None in orthoLinear_corr_only')
claim: "When the channel-correlation file is absent, the else branch runs `assert ValueError(...)`; a ValueError instance is always truthy, so the assert passes silently instead of raising, and self.channel_corr_mat is never assigned."
concern: "Instead of a clear 'file missing' error, the model proceeds and later fails with an obscure AttributeError on self.channel_corr_mat; combined with the 23 missing channel files, this hides the real cause from anyone running OLinear-C."
resolution: "Replace `assert ValueError(...)` with `raise ValueError(...)` (or `raise FileNotFoundError(q_channel_file)`)."
cross_refs: ["olinear-c-channel-mats-missing"]
paper_ref: "Appendix H (OLinear-C)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: corrmat-notebook-hardcoded-windows-path
category: difference
topic: "preprocessing reproducibility"
title: "Q-matrix notebook uses hardcoded Windows paths and only lags {96,192,336,720}"
severity: low
confidence: medium
status: question
file: dataset/Generate_corrmat.ipynb
line_start: 182
line_end: 182
quote: |
      "file_path = r'D:\\SystemData\\xunn\\Desktop\\codes\\iTransformer-main\\dataset\\Solar\\solar_AL.xlsx'\n",
claim: "Generate_corrmat.ipynb (the only artefact that reproduces the OrthoTrans Q and channel matrices) hardcodes per-dataset Windows file paths (cell 2 even points at an absolute D:\\ user-desktop path) and its temporal-Q cell only iterates time_lag in {96,192,336,720}, whereas the shipped small-dataset matrices use lags {3,6,9,12,24,36,48,60}; the notebook must be hand-edited per dataset to regenerate the matrices actually used."
concern: "The provided regeneration script does not, as committed, reproduce the full set of Q matrices used in the paper without manual edits, so anyone needing to regenerate matrices for a non-shipped dataset must reverse-engineer the lag list and fix the paths."
resolution: "Authors: parameterise the notebook (or provide a .py) over dataset + lag list, and replace the absolute D:\\ path with a relative one, so all shipped Q/channel matrices can be regenerated as-is."
cross_refs: ["avail-statement-external-data"]
paper_ref: "§4.2 OrthoTrans; README step 2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

### methodology

No methodology findings. The two most likely leakage vectors were checked and found sound
(see "Items that genuinely look fine"): the OrthoTrans basis and the channel-correlation matrix
are both computed from the training split only, and model selection / early stopping use the
validation loss, not the test set.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | medium       | Unpinned deps; 23/24 OLinear-C channel matrices absent; plug-in-baseline code absent; main benchmark data external (documented). |
| bug         | 1          | low          | `assert ValueError(...)` never raises on a missing channel file. |
| difference  | 1          | low          | Q-matrix regeneration notebook hardcodes Windows paths / partial lag list. |
| methodology | 0          | -            | OrthoTrans basis and channel matrix verified train-only; no leakage found. |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] requirements-unpinned** — no version pins on any dependency (incl. torch); exact reproduction is not guaranteed. (medium / high)
2. **[missing] olinear-c-channel-mats-missing** — 23 of 24 channel-correlation matrices for the OLinear-C variant are absent, including for datasets whose CSV is shipped. (medium / high)
3. **[missing] plugin-on-baselines-code-missing** — the OrthoTrans-as-plug-in code for iTransformer/PatchTST/RLinear (Table 5, a headline generality claim) is not in the repo. (medium / medium)
4. **[bug] olinear-c-missing-file-assert** — `assert ValueError(...)` silently passes, masking the missing-file error in OLinear-C. (low / high)
5. **[missing] avail-statement-external-data** — main benchmark datasets must be downloaded externally and their Q matrices regenerated (documented; filed as question). (low / high)
6. **[difference] corrmat-notebook-hardcoded-windows-path** — the Q-regeneration notebook needs manual edits to reproduce the shipped matrices. (low / medium)

### Items that genuinely look fine (actively checked)
- **OrthoTrans basis is train-only and orthonormal.** `_audit_code/check_q_train_only.py` reproduces `dataset/cars/cars_12_ratio0.7.npy` exactly (sign-free diff = 0) from the first 70% of the series (the train split), with the test rows excluded; `max|QQᵀ−I| = 3.9e-8`. Matches the paper's "Let Xtrain … denote the training set" (§4.2) and "Qi and Qo are pre-computed".
- **Channel-correlation matrix is train-only.** The shipped `weather_COV_channel_ratio0.70.npy` reproduces `np.corrcoef` of the first 70% of weather to within 7e-8 (Generate_corrmat.ipynb cell 1 uses `A_ori[0:int(len_a*ratio)]`, ratio=0.7).
- **No test-set model selection.** `experiments/exp_forecast.py:542-556` computes a test loss each epoch but only *prints* it (line 546); early stopping (line 547) and the reloaded best checkpoint (line 554-556) use the validation loss only — the "visibility, not influence" case.
- **Train/val/test splits are temporal and scaler is train-fit.** Custom 70/10/20 (`data_loader.py:268-272`, scaler fit on `border1s[0]:border2s[0]`), ETT 12/4/4-month, PEMS 0.6/0.2/0.2 (`data_loader.py:563-580`), all with train = first portion; StandardScaler fitted on train only.
- **The "best stage" min-MSE+MAE selection (`exp_forecast.py:852-858`) does NOT affect OLinear.** Both `model/OLinear.py` and `model/OLinear_C.py` `forward` return a single tensor, so `isinstance(outputs_list, tuple)` is False, the per-stage block (line 799) is skipped, and `min_index` is forced to 0 (only the final metric is in the list).
- **Loss matches the paper.** `WeightedL1Loss` (`exp_forecast.py:155-191`) implements the position-decaying weighted L1 loss "following CARD" described at paper lines 618 / 2318.
- **Multi-seed robustness harness exists.** `scripts/**/robust/*.sh` run `--itr 7 --fix_seed 0`, supporting Table 13's 7-seed std (seeds are random, not pinned).
- **Temporal Q matrices present for all 11 shipped datasets** (`_audit_code/check_qmat_for_shipped_datasets.py`), so their main results are reproducible locally.

### Open questions for the authors
- Do the external dataset links (Google Drive / Tsinghua Cloud / Appendix B) still resolve, and does `Generate_corrmat.ipynb` reproduce the exact Q/channel matrices for the non-shipped benchmark datasets (the notebook's temporal cell only iterates lags {96,192,336,720})? (avail-statement-external-data, corrmat-notebook-hardcoded-windows-path)
- Will the OrthoTrans-as-plug-in baseline code (Table 5) and the full set of OLinear-C channel matrices be released? (plugin-on-baselines-code-missing, olinear-c-channel-mats-missing)
