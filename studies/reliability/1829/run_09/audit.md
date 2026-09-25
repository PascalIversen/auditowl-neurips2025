# Audit — OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain (paper 1829)

## Summary

The repo (`code/jackyue1994__OLinear/`) is the official implementation of OLinear.
Its README header and figures match this paper, so it is the author code. It contains
the model (`model/OLinear.py` and variants), the OrthoTrans/NormLin layers
(`layers/Transformer_EncDec.py`, `layers/newLinear.py`), a full training/eval harness
(`run.py`, `experiments/exp_forecast.py`, `data_provider/`), per-dataset reproduction
scripts (`scripts/OLinear/*`, `scripts/OLinear_C/*`, `scripts/ablation/*`,
`scripts/.../robust/*`), and a notebook (`dataset/Generate_corrmat.ipynb`) that builds
the orthogonal `Q` matrices used by OrthoTrans.

What I did:
- Read the paper's method (§4.1–4.3) and implementation details (Appendix D), and the
  data-availability / dependency claims.
- Traced the split logic for every dataset loader (`data_provider/data_loader.py`):
  all loaders use a chronological train-first split (0.7/0.1/0.2 or 0.6/0.2/0.2) and
  fit the `StandardScaler` on the **train** portion only — no scaling leakage.
- Verified model selection: `EarlyStopping` is driven by **validation** loss
  (`exp_forecast.py:547`, `utils/tools.py:58-112`); test loss is only printed
  (`exp_forecast.py:543,546`) and never used to pick a checkpoint/epoch — no test leakage.
- Reproduced the OrthoTrans `Q`-matrix construction from `Generate_corrmat.ipynb` and
  compared the shipped npy files against train-only vs full-data computations
  (`_audit_code/check_q_train_only.py`): the shipped matrices match the **train-only**
  computation to ~1e-14 and differ from the full-data computation by ~1e-2 — confirming
  OrthoTrans is fit on training data only (no leakage of test statistics into the basis).
- Inspected the dependency list and the import graph that every experiment pulls in.

The implementation is faithful and methodologically sound on the points that most often
break this class of paper (splitting, scaler leakage, basis-from-train-only, model
selection). The findings below are: one unlisted dependency on the import-time critical
path (`missing`), and one undocumented learnable additive term inside OrthoTrans
(`difference`). I could not re-run training (no GPU), so headline numeric values are
traced to the scripts that compute them but not independently reproduced.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 long-term MSE/MAE (24 benchmarks) | `run.py` → `experiments/exp_forecast.py:560-795` (`metric()` in `utils/metrics.py:107`); driven by `scripts/OLinear/*.sh`, `scripts/OLinear_C/*.sh` | code present | not re-run (GPU) | Code present, values not re-run |
| Table 3 short-term (S1/S2) MSE/MAE | same harness; `scripts/.../*_S2.sh` | code present | not re-run | Code present, values not re-run |
| Table 4 attention-vs-NormLin ablation | `model/OLinear_attn_var.py`; `scripts/ablation/attn_var/*` | code present | not re-run | Code present |
| Table 5 OrthoTrans-as-plugin (iTrans/PatchTST/RLinear) | `scripts/ablation/basis/*` + plugin code | code present | not re-run | Code present |
| Var/temp ablation | `model/OLinear_ablation_var_temp.py`, `OLinear_ablation_lin_design.py`; `scripts/ablation/var_temp/*` | code present | not re-run | Code present |
| Basis ablation (Fourier/wavelet/Legendre/...) | `model/orthoLinear_basis/*`; `scripts/ablation/basis/*` | code present | not re-run | Code present |
| Table 13 std over 7 seeds | `scripts/.../robust/*.sh` (`itr 7`, `fix_seed 0`) | code present | not re-run | Code present |
| Fig. 2 / Fig. 7 transformed-domain & corr-matrix viz | `dataset/Generate_corrmat.ipynb` (cells 0,2,5) | code present | partially reproduced (Q build matches) | Code present |
| OrthoTrans Q matrices (electricity/traffic/ETT/PEMS/Solar/exchange/METR) | (none in repo) | — | — | MISSING from repo; regenerable via `Generate_corrmat.ipynb` after downloading CSVs |
| Standard-benchmark CSVs (electricity/traffic/ETT/PEMS/Solar/exchange/METR) | (none in repo) | — | — | Not shipped; README points to Google Drive / Tsinghua Cloud + Appendix B |
| OrthoTrans basis is fit on TRAIN only | `dataset/Generate_corrmat.ipynb` cell 0; verified `_audit_code/check_q_train_only.py` | meanabsdiff(shipped, train)=6e-14 vs (shipped, full)=2.5e-2 | ✓ (matches paper "Let X_train denote the training set") | Verified |
| Baseline numbers (11 baselines) | (none — taken from original papers / official code) | — | — | Out of scope (disclosed in Appendix D) |

## missing

```yaml finding
id: unlisted-deps-patoolib-tqdm
category: missing
topic: "dependencies / environment"
title: "patoolib and tqdm imported at startup of every run but absent from requirements.txt"
severity: medium
confidence: high
status: finding
file: data_provider/m4.py
line_start: 26
line_end: 27
quote: |
  import patoolib
  from tqdm import tqdm
claim: "data_provider/m4.py unconditionally imports `patoolib` and `tqdm` at module top; m4.py is imported at the top of data_provider/data_loader.py (line 9: `from data_provider.m4 import M4Dataset, M4Meta`), which data_provider/data_factory.py imports, which experiments/exp_forecast.py and run.py import. Neither `patoolib` nor `tqdm` appears in requirements.txt (verified: grep -iE 'tqdm|patool|pyunpack' returns nothing)."
concern: "Because the import is on the startup path of run.py for EVERY dataset (not just M4), a fresh environment built from requirements.txt raises ModuleNotFoundError before any experiment can begin, so the published reproduction commands fail out of the box."
resolution: "Add `tqdm` and `patoolib` (and its unpacker backend) to requirements.txt, or move the M4-only imports inside Dataset_M4 so the main forecasting path does not require them. (Also note: requirements.txt lists `pywt`, whose PyPI name is `PyWavelets`, and pins no versions.)"
cross_refs: []
check_script: _audit_code/check_imports.py
paper_ref: "Appendix D (implementation details); README 'Usage' step 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No findings. The training/eval harness wiring I checked (split borders, scaler fit on
train, early-stopping on validation, metric axes, RevIN denorm) is consistent with the
code's own intent.

## difference

```yaml finding
id: orthotrans-undocumented-delta
category: difference
topic: "model architecture / OrthoTrans"
title: "OrthoTrans adds undocumented learnable additive terms (delta1, delta2) not in the paper's equations"
severity: low
confidence: high
status: finding
file: model/OLinear.py
line_start: 104
line_end: 118
quote: |
        else:
            x_trans = torch.einsum('bndt,tv->bndv', x, self.Q_mat.transpose(-1, -2)) + self.delta1
            # added on 25/1/30
            # x_trans = F.gelu(x_trans)
            # [B, N, D, T]
        assert x_trans.shape[-1] == self.seq_len

        # ########## transformer ####
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.pred_len)

        # [B, N, D, tau]; orthogonal transformation
        if self.Q_chan_indep:
            x = torch.einsum('bndt,ntv->bndv', x_trans, self.Q_out_mat)
        else:
            x = torch.einsum('bndt,tv->bndv', x_trans, self.Q_out_mat) + self.delta2
claim: "The default OLinear path (Q_chan_indep=0, the value used by all main scripts) adds learnable parameters self.delta1 (shape [1,N,1,seq_len]) and self.delta2 (shape [1,N,1,pred_len]; declared at OLinear.py:80-81) to the input and output orthogonal projections. The paper's Eqs. (1)-(2) describe the transform as a pure orthogonal projection Z=(RevINNorm(X)⊗φd)Qi^T and ˜Y=LinearDecode(˜HL)·Qo with no additive term."
concern: "A result-affecting learnable component present in the code is absent from the paper's formal description of OrthoTrans, so the published equations under-specify the model that actually produced the numbers; the code is still methodologically valid (the deltas are ordinary learnable biases)."
resolution: "Authors: state in the paper that OrthoTrans includes per-position learnable bias terms (delta1/delta2), or run the ablation with delta1=delta2=0 to confirm they do not materially change the reported results."
cross_refs: []
paper_ref: "§4.1 Eqs. (1)-(2); paper_text.txt lines 254-281"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No findings. I specifically checked the leakage surfaces that matter for this paper:

- **OrthoTrans basis from train only** — the paper says "Let X_train denote the training
  set" and "Qi and Qo are pre-computed". `_audit_code/check_q_train_only.py` confirms the
  shipped npy matrices reproduce exactly from the train portion and not from full data.
- **Scaler fit on train only** — `data_loader.py:308` (`scaler_data = data[border1s[0]:border2s[0]]`), PEMS at `:580`. The Physio/Air imputation datasets fit on train+val (`:306,464`), which is the standard convention for those imputation benchmarks and does not affect the forecasting tables.
- **Model selection on validation, not test** — `exp_forecast.py:547` calls
  `early_stopping(vali_loss, ...)`; test loss is only printed.

No leakage, no inappropriate metric, and no test-set-touching tuning was found.

## Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 1 | medium | tqdm/patoolib imported on every run's startup path but not in requirements.txt |
| bug | 0 | - | harness wiring consistent with its own intent |
| difference | 1 | low | OrthoTrans uses undocumented learnable additive deltas not in the paper's equations |
| methodology | 0 | - | split/scaler/basis are train-only; model selected on validation |

## Top take-aways
1. (`missing`, medium) `tqdm` and `patoolib` are imported at startup of every experiment
   (via `data_provider/m4.py` → `data_loader.py` → `data_factory.py`) but are not in
   `requirements.txt`, so a fresh environment fails before training starts.
2. (`difference`, low) OrthoTrans adds learnable `delta1`/`delta2` bias terms not described
   in the paper's Eqs. (1)-(2).

## Items that genuinely look fine
- OrthoTrans `Q` matrices are computed from the training portion only (verified numerically).
- All dataset splits are chronological, train-first; `StandardScaler` fit on train only.
- Early stopping / checkpoint selection uses validation loss; test loss is only logged.
- NormLin matches Eq. (3): `F.normalize(F.softplus(W), p=1, dim=-1)` (`layers/Transformer_EncDec.py:136-138`, `layers/newLinear.py:27`).
- Dimension extension is the outer product with a learnable vector, as described (`model/OLinear.py:84-91`).
- Weighted-L1 training loss `(i+1)^(-alpha)` matches the CARD-style loss; test metrics are plain MSE/MAE.
- Reproduction scripts exist for the main tables, ablations, and the 7-seed robustness runs.

## Open questions for the authors
- The standard-benchmark CSVs (electricity, traffic, ETT, PEMS, Solar, exchange, METR-LA)
  and their OrthoTrans `Q` matrices are not in the repo; they rely on an external Google
  Drive / Tsinghua Cloud link (and on re-running `Generate_corrmat.ipynb`). I could not
  verify those links are live (no network). Please confirm the download links resolve and
  that the bundle includes the `*_ratio0.7.npy` Q matrices the scripts expect, or document
  the exact regeneration steps. (Filed as a question, not a finding, because the notebook
  that regenerates the matrices is present and verified to work.)
- Paper Appendix D states early-stopping patience of 10 epochs, but the released scripts
  use `--patience 5`; the CLI supports the paper's value, so this is a configuration note
  rather than a finding.
