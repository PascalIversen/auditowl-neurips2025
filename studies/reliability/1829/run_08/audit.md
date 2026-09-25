# Audit — OLinear (NeurIPS 2025, paper #1829)

## Summary

The repo `code/jackyue1994__OLinear/` is the official implementation of "OLinear:
A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain"
(README line 3 states this explicitly, and its model files match the paper's
OrthoTrans + NormLin design). I read the paper's Method (§4) and Experiments (§5)
from `paper.pdf` (locating numbers via `paper_text.txt`), then traced the
implementation: the core model (`model/OLinear.py`, `model/OLinear_C.py`), the
NormLin / CSL layer (`layers/Transformer_EncDec.py:102-150`, `layers/newLinear.py`),
the training/eval loop (`experiments/exp_forecast.py`), the data split and scaling
(`data_provider/data_loader.py`), the orthogonal-matrix generation
(`dataset/Generate_corrmat.ipynb`), the driver (`run.py`), and the reproduction
scripts (`scripts/`). I ran two read-only checks under `_audit_code/`:
`check_baselines_and_data.py` (which baseline models / datasets / dependencies are
present) — outputs in `_audit_code/out/check_baselines_and_data.json`.

Headline finding profile: the *proposed* method is implemented faithfully and the
evaluation protocol is sound (training-only correlation matrices, training-only
StandardScaler, early stopping on the validation set, single fixed-seed runs for
the main tables). The reproducibility gaps are around *what is shipped*: no
baseline-model code (so the SOTA comparison Tables 2/3/5 cannot be regenerated
end-to-end from this repo), the seven large long-term benchmarks are not bundled
(download links only), and a couple of minor packaging defects (dependency name,
dead `Exp_*` references).

## Traceability table

| Paper artefact | Repo location | Computed? | Matches paper | Status |
|---|---|---|---|---|
| OrthoTrans: Q from eigendecomp of temporal Pearson corr of **training** series (§4.2) | `dataset/Generate_corrmat.ipynb` cell 0 (`A = data[train_length-...:train_length]`), loaded `model/OLinear.py:24-42` | yes | ✓ (train-only) | Verified |
| NormLin = RowNormL1(Softplus(W)) x  (Eq. 3) | `layers/Transformer_EncDec.py:134-138` (`A=F.softplus(weight_mat); A=F.normalize(A,p=1,dim=-1)`) | yes | ✓ | Verified |
| ISL = LayerNorm(H + Linear(GELU(Linear(H)))) (Eq. 5) | `layers/Transformer_EncDec.py` EncoderLayer + `model/OLinear.py:51-77` | yes | ✓ (GELU MLP) | Verified |
| Weighted L1 loss from CARD (§5) | `experiments/exp_forecast.py:155-191` (`(i+1)**(-alpha)` weights); scripts set `--loss_mode L1 --lossfun_alpha 0.5` | yes | ✓ | Verified |
| Table 2 — OLinear long-term MSE/MAE (ECL/Traffic/ETT/Solar/Exchange/PEMS/Weather) | `run.py`+`scripts/OLinear/*.sh`, `scripts/OLinear_C/*.sh`; **data not shipped** except weather | partially (code present, data needs download) | not run here | PARTIAL (see longterm-data-not-shipped) |
| Table 2/3 — **baseline** rows (TimeMixer, iTrans, PatchTST, …, 11 models) | (none — only OLinear-family models in `model/`) | no | — | MISSING (see baselines-not-in-repo) |
| Table 3 — OLinear short-term S1/S2 (ILI/COVID/METR-LA/NASDAQ/Wiki/SP500/DowJones/Cars/Power/Website/Unemp) | `scripts/OLinear_C/*_S2.sh` etc. + shipped CSVs + precomputed Q | yes (data+code present) | not run here | Reproducible |
| Table 4 — transformation-basis ablation (Fourier/Wavelet/Cheby/Laguerre/Legendre/Identity) | `model/orthoLinear_basis/*.py`, `scripts/ablation/basis/*.sh` | code present (wavelet needs `pywt`/PyWavelets) | not run here | Reproducible w/ dep fix |
| Table 5 — OrthoTrans plug-in into iTrans/PatchTST/RLinear | flags `--iTrans_ortho_trans` etc. in `run.py:78-80`; **no iTrans/PatchTST/RLinear model code** | no | — | MISSING (covered by baselines-not-in-repo) |
| Table 1 — NormLin O(N²D+2ND²) vs MHSA O(2N²D+4ND²) | analytic; consistent with `LinearEncoder` (pre-Linear ND², NormLin N²D, post-Linear ND²) | analytic check | ✓ | Verified (analytic) |
| Table 14 — robustness to random seeds | `scripts/*/robust/*.sh` (`--itr 7 --fix_seed 0`) | yes (code present) | not run here | Reproducible |
| Table 41 / Fig 3 — efficiency (time/memory) | `experiments/exp_forecast.py:32-152` `compute_model_stats` (`--model_stats_mode`) | yes (OLinear only; baselines absent) | not run here | PARTIAL |

## missing

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "baselines / result traceability"
title: "No baseline-model code; SOTA comparison tables not reproducible from repo"
severity: medium
confidence: high
status: finding
file: experiments/exp_basic.py
line_start: 13
line_end: 25
quote: |
        self.model_dict = {
            'OLinear': OLinear,
            'OLinear_C': OLinear_C,
            'OLinear_attn_var': OLinear_attn_var,
            'OLinear_ablation_var_temp': OLinear_ablation_var_temp,
            'OLinear_ablation_lin_design': OLinear_ablation_lin_design,
            'OLinear_no_Q_neither': OLinear_no_Q_neither,
            'OLinear_FFT': OLinear_FFT,
            'OLinear_wavelet': OLinear_wavelet,
            'OLinear_wavelet2': OLinear_wavelet2,
            'OLinear_Legendre': OLinear_Legendre,
            'OLinear_Laguerre': OLinear_Laguerre,
            'OLinear_cheby': OLinear_cheby,
claim: "The only registered/implemented models are OLinear and its own variants; none of the 11 baselines (TimeMixer, FilterNet, FITS, DLinear, TimeMixer++, Leddam, CARD, Fredformer, iTransformer, PatchTST, TimesNet) nor the Table-5 plug-in hosts (iTransformer/PatchTST/RLinear) have code in model/."
concern: "The paper's central claim is SOTA over these baselines (Tables 2,3,5; '1st Count' rows); without baseline code or a stated provenance, the comparison and the headline 'consistently achieves state-of-the-art' cannot be regenerated or verified from this repo."
resolution: "Authors: ship the baseline configs/code (or scripts that call the cited repos), or state explicitly that baseline numbers are quoted from prior papers and give the source per row, confirming identical splits/lookback/metrics."
cross_refs: ["§5.1", "longterm-data-not-shipped"]
check_script: _audit_code/check_baselines_and_data.py
paper_ref: "Tables 2, 3, 5 (baseline columns and '1st Count')"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: longterm-data-not-shipped
category: missing
topic: "data availability"
title: "Seven long-term benchmarks (ECL/Traffic/ETT/Solar/Exchange/PEMS/METR-LA) not bundled"
severity: low
confidence: high
status: finding
file: README.md
line_start: 48
line_end: 49
quote: |
  2. Some datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).  The others can be obtained from the links in Appendix B. 
  Some datasets are already in the folder ```./dataset```. 
claim: "Only weather plus the short-term CSVs ship in dataset/ (verified by _audit_code/check_baselines_and_data.py: shipped data = weather, cars, covid, DowJones, ILI, nasdaq, power, SP500, unemployment, website, wiki_mini); the headline long-term datasets in Table 2 are obtained only via external Google Drive / Tsinghua Cloud / Appendix-B links."
concern: "Table 2 (the long-term SOTA table) cannot be reproduced from the repo alone; reproduction depends on third-party download links that an auditor cannot verify offline and that may rot."
resolution: "Authors: confirm the Google Drive / Tsinghua Cloud links remain live and contain ECL, Traffic, ETTh1/2, ETTm1/2, Solar, Exchange, PEMS03/04/07/08, METR-LA in the expected CSV/NPZ format; ideally pin a DOI/Zenodo mirror."
cross_refs: ["§5"]
check_script: _audit_code/check_baselines_and_data.py
paper_ref: "Table 2; Appendix B (dataset links)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: pywt-wrong-pypi-name
category: missing
topic: "dependencies / environment"
title: "requirements.txt lists 'pywt' (no such PyPI package; correct name is PyWavelets), and versions are unpinned"
severity: low
confidence: high
status: finding
file: requirements.txt
line_start: 13
line_end: 13
quote: |
  pywt
claim: "The wavelet-basis ablation imports `import pywt` (model/orthoLinear_basis/OLinear_wavelet.py:10, OLinear_wavelet2.py:10), but the PyPI distribution is `PyWavelets` (import name pywt); `pip install -r requirements.txt` fails on the line `pywt`. All entries are also unversioned, so the environment is not pin-reproducible."
concern: "`pip install -r requirements.txt` errors out on `pywt`, and even after fixing the name the lack of version pins (incl. torch/numpy) makes exact reproduction unreliable."
resolution: "Replace `pywt` with `PyWavelets` and pin versions (especially torch, numpy, scikit-learn, fvcore) used to produce the paper's numbers."
cross_refs: ["wavelet-ablation"]
check_script: _audit_code/check_baselines_and_data.py
paper_ref: "Table 4 (Wavelet1/Wavelet2 rows)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: undefined-exp-classes
category: bug
topic: "driver / entrypoint"
title: "run.py references Exp_Short_Term_Forecast / Exp_Long_Term_Forecast_Partial / Exp_Imputation which are never imported"
severity: low
confidence: high
status: finding
file: run.py
line_start: 441
line_end: 449
quote: |
    Exp = Exp_Forecast
    if args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
        args.m4_folder = os.path.join('./m4_results', args.model + '_' + args.m4_result_path_str)
        #  + '_' + datetime.now().strftime('%y%m%d_%H%M%S')
    elif args.exp_name == 'partial_train':  # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
claim: "Only `Exp_Forecast` is imported (run.py:8); `Exp_Short_Term_Forecast`, `Exp_Long_Term_Forecast_Partial`, and `Exp_Imputation` are undefined names, so any of these branches raises NameError. None of the shipped scripts set `task_name`/`exp_name` (verified: no `task_name`/`exp_name` token in scripts/), so all reproduction scripts hit the default `forecasting` path and avoid the crash."
concern: "A user following Appendix-described settings (e.g. the partial-training experiment of 'Figure 8', or imputation) would hit a NameError; the partial-train / imputation experiments are effectively unrunnable through this entrypoint as shipped."
resolution: "Import the referenced Exp_* classes (or remove the dead branches). Confirm whether the Figure-8 partial-training and any imputation results were produced with a different, unshipped driver."
cross_refs: []
paper_ref: "run.py entrypoint; Figure 8 (partial training)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

No confirmed code↔paper differences in the proposed method. The implemented
OrthoTrans (training-only eigendecomposition of the temporal Pearson correlation
matrix), the NormLin layer (Softplus + row-wise L1 normalization, Eq. 3), the ISL
(GELU MLP with residual+LayerNorm, Eq. 5), and the weighted-L1 loss all match the
paper's descriptions at the locations cited in the traceability table.

## methodology

No methodology finding. I specifically checked the leakage vectors that matter for
this benchmark family and found them clean:

- **Correlation/Q matrices are training-only.** `dataset/Generate_corrmat.ipynb`
  cell 0 slices `A = data[train_length-int(...*ratio):train_length, ...]` with
  `train_length=int(N*train_ratio)`, and the channel matrix (cell 1) uses
  `A = A_ori[0:int(len_a*ratio)]` (first 70%). Scripts request files named
  `*_ratio0.7.npy` / `*_ratio0.6.npy`, and the 0.6 cases coincide with the 60%-train
  datasets (ETT via `Dataset_ETT_*`/`custom2`, PEMS, unemployment), so Q never sees
  validation/test. This satisfies the paper's stated "Xtrain" construction (§4.2).
- **Scaler is fit on train only.** `data_provider/data_loader.py` fits
  `StandardScaler` on `data[border1s[0]:border2s[0]]` (the train segment) for the
  forecasting datasets (e.g. lines 307-311, 580). (Imputation-only Physio/Air use
  train+valid for scaling, but those are not the forecasting headline tasks.)
- **Early stopping uses the validation set, not test.** `exp_forecast.py:547`
  calls `early_stopping(vali_loss, ...)`; the test loss at line 543/546 is only
  printed, never used to pick the checkpoint (the "hidden-NaN" diagnostic passes).
- **No test-set hyperparameter selection in the headline runs.** Main scripts use
  `--itr 1 --fix_seed 1` and a single `--test_batch_size`, so the
  best-`test_batch_size` loop (`run.py:568-576`) and the multi-seed best
  (`run.py:578-580`) are no-ops. The "best stage by min(MSE+MAE)" selection in
  `exp_forecast.py:849-858` only triggers when the model returns a tuple of
  intermediate stages; `OLinear.py:152` and `OLinear_C.py:162` both `return out`
  (a single tensor), so for the headline models this selection collapses to the
  single final value.
- **Temporal integrity.** Splits are forward/contiguous (train→valid→test by index),
  not shuffled K-fold.

## Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | medium       | No baseline code; long-term data via download links; bad dep name |
| bug         | 1          | low          | run.py refs 3 undefined Exp_* classes (dead branches as shipped) |
| difference  | 0          | -            | Proposed method matches paper |
| methodology | 0          | -            | Leakage vectors checked clean (train-only Q & scaler, val early-stop) |

## Top take-aways

1. **[missing] baselines-not-in-repo** (med/high): the SOTA comparison (Tables 2,3,5)
   has no baseline code or per-row provenance — the headline claim is not
   regenerable from this repo.
2. **[missing] longterm-data-not-shipped** (low/high): Table 2's large benchmarks
   require external download links, not bundled data.
3. **[missing] pywt-wrong-pypi-name** (low/high): `pip install -r requirements.txt`
   fails on `pywt` (should be `PyWavelets`); all deps unpinned.
4. **[bug] undefined-exp-classes** (low/high): `run.py` references three Exp_*
   classes that are never imported (partial-train / imputation branches crash).

## Items that genuinely look fine

- OrthoTrans Q matrices and the channel correlation matrix are computed from the
  **training** portion only (`Generate_corrmat.ipynb`); no future leakage into the
  transform basis.
- StandardScaler fit on train only; forward (non-shuffled) train/valid/test split.
- Early stopping driven by validation loss; test loss only logged.
- NormLin (Eq. 3), ISL (Eq. 5), and the weighted-L1 loss faithfully implemented.
- Headline tables come from single fixed-seed runs; the various "best-of" selection
  loops are no-ops for the OLinear / OLinear-C models as configured.

## Open questions for the authors

- Are the Table 2/3/5 baseline numbers reproduced by the authors or quoted from the
  baselines' papers? If quoted, under identical lookback (T=96), split, metric, and
  loss? (Asymmetric tuning between a tuned OLinear and quoted baselines would weaken
  the comparison.)
- Do the Google Drive / Tsinghua Cloud dataset links remain live and contain all
  long-term benchmarks in the format the loaders expect?
- Which driver produced the Figure-8 partial-training and any imputation results,
  given the undefined Exp_* references in `run.py`?
