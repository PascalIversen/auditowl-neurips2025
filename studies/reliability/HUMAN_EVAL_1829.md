# Human-eval worksheet — #1829 · 1829

**16 distinct defects** (the 10 PDF+text audit runs' findings, merged by defect). Detection = how many of the 10 runs surfaced the defect (high = robust; 1 = one run only). Severity & confidence are the auditor's own labels (spread shown where runs disagreed); the wording/quote is taken from the highest-confidence run that cited code.

Tick **one** box per defect (put an `x`):

- **correct & relevant** — true *and* a substantive reproducibility issue worth raising
- **correct but wrong severity** — true and worth raising, but the severity label is miscalibrated (e.g. an out-of-the-box crash with a trivial fix tagged high that's really low/medium)
- **correct but not relevant** — technically true but trivial / nitpick / already acknowledged
- **unsure** — can't decide without resources beyond the frozen repo + paper
- **false** — the claim misreads the code/paper and does not hold

Frozen code: `1829/code_frozen/`  ·  paper: `audits/1829/paper.pdf`

---

### F01 · requirements.txt fully unpinned (no versions) and lists 'pywt' — wrong PyPI name (should be PyWavelets); install fails

_category: Missing code / data · topic: dependencies / environment_

**severity: medium  (varied: medium, low)  ·  confidence: high  ·  detection: 8/10 runs**

- **Claim:** Every dependency is listed without a version pin (no ==, >=, or ~=).
- **Concern:** The exact runtime environment cannot be reconstructed; future versions of torch/numpy/scikit-learn may change numerics or break the code, hampering reproduction of the reported numbers.
- **Ask:** Authors: pin versions (e.g. a frozen pip freeze / environment.yml) for at least torch, numpy, scikit-learn, pandas, fvcore.
- **Evidence:** `requirements.txt:1-13` · paper: Appendix D
- **Found in runs:** r02, r03, r04, r05, r06, r07, r08, r10  (representative: r02#3)
- **Quoted at `requirements.txt:1-13`:**
```
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
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
This requirements.txt was probably not generated correctly and just written by hand and never tested. Version numbers also missing. The guidelines state "the instructions should contain the exact command and environment needed to run to reproduce the results".
---

### F02 · Baseline & OrthoTrans/NormLin plug-in model code absent — Tables 1/5/6 not reproducible (no iTransformer/PatchTST/RLinear/Timer etc. in repo)

_category: Missing code / data · topic: generality / plug-in experiments_

**severity: low  (varied: high, medium)  ·  confidence: high  (varied: high, medium)  ·  detection: 6/10 runs**

- **Claim:** The model registry contains only OLinear and its ablation variants; no iTransformer, PatchTST, RLinear, or DLinear implementation exists in model/, and no script sets --iTrans_ortho_trans 1 / --PatchTST_ortho_trans 1 / --DLinear_ortho_trans 1 (the flags exist in run.py:78-80 but are dead).
- **Concern:** Table 5's generality result ('OrthoTrans yields average MSE improvements of 5.1% and 10.1% for iTransformer and PatchTST') cannot be reproduced because the baseline forecasters OrthoTrans is plugged into are not in the repo.
- **Ask:** Authors: please add the iTransformer/PatchTST/RLinear model implementations and the plug-in run scripts used to generate Table 5 (and Table 6's plug-in rows).
- **Evidence:** `experiments/exp_basic.py:13-26` · paper: Section 5.2 'OrthoTrans as a plug-in'; Table 5
- **Found in runs:** r01, r02, r03, r07, r08, r10  (representative: r01#0)
- **Quoted at `experiments/exp_basic.py:13-26`:**
```
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
}
```

**Verdict:**   correct & relevant `[X]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
minor point: its table 2 not table 1, but the general framing is true. Not shipping baselines is standard, but they adapted the baseline with their plug in module and claim that this improved it. This should be shipped. The code existed locally but was not pushed for some reason. Not reproducible
---

### F03 · Headline benchmark datasets (ECL/ETT/Traffic/Solar/PEMS/Exchange/METR) and their precomputed Q-matrices are not shipped

_category: Missing code / data · topic: expected code completeness / data availability_

**severity: medium  (varied: medium, low)  ·  confidence: high  (varied: high, medium)  ·  detection: 6/10 runs**

- **Claim:** The model requires a precomputed Q-matrix .npy on disk (assert os.path.isfile). For 14 of 23 OLinear scripts — including the core benchmarks ECL/electricity, ETT(h1/h2/m1/m2), Traffic, Solar, PEMS03/04/07/08, Exchange, METR-LA — neither the data CSV/NPZ nor the referenced Q-matrix file (e.g. electricity_96_ratio0.7.npy) is present in the repo.
- **Concern:** The headline tables (Table 2/17 etc.) cannot be reproduced out-of-the-box: the scripts crash at model construction because the Q-matrix files are missing, and the underlying datasets are not shipped.
- **Ask:** Ship the precomputed Q-matrices for the main benchmarks (or a one-command script that regenerates them from the downloaded data), since the data CSVs are linked in the README but the Q-matrices are not provided for these datasets.
- **Evidence:** `model/OLinear.py:24-32` · paper: Table 2 / Table 17 (long-term forecasting)
- **Found in runs:** r02, r03, r04, r05, r08, r10  (representative: r05#2)
- **Quoted at `model/OLinear.py:24-32`:**
```
q_mat_dir = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
if not os.path.isfile(q_mat_dir):
    q_mat_dir = os.path.join(configs.root_path, q_mat_dir)
assert os.path.isfile(q_mat_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
self.Q_mat = torch.from_numpy(np.load(q_mat_dir)).to(torch.float32).to(device)

assert (self.Q_mat.ndim == 3 if self.Q_chan_indep else self.Q_mat.ndim == 2)
assert (self.Q_mat.shape[0] == self.enc_in if self.Q_chan_indep else self.Q_mat.shape[0] == self.seq_len)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
There is a notebook to generate the Q matrices but it has hardcoded paths and some hardcoded parameters. The Q matrices are quite hard to reproduce. So it is borderline irreproducible but possible.
---

### F04 · No code computes the reported std dev / 99% CIs / Student's t-test (Tables 13/14/15/26)

_category: Missing code / data · topic: statistical integrity / result traceability_

**severity: medium  ·  confidence: high  ·  detection: 4/10 runs**

- **Claim:** run.py writes each run's mse/mae to a text log, but no script in the repo aggregates the 7 per-seed runs into a standard deviation, a 99% confidence interval, or a Student's t-test p-value. A whole-repo grep (_audit_code/check_stats_and_selection.py) finds zero genuine std/CI/t-test computations (all keyword hits are filenames like 'model_stats.txt' or LayerNorm/positional-encoding '.std()' calls).
- **Concern:** Tables 13, 14, 15, and 26 (standard deviations, 99% CIs, and the OLinear-vs-iTransformer Student's t-test with p<0.05) are headline robustness/significance artefacts, yet nothing in the released code produces them, so they cannot be reproduced or checked from the repo.
- **Ask:** Authors: please add the script that reads the per-seed logs and computes the standard deviations, the 99% CIs, and the Student's t-test p-values reported in Tables 13/14/15/26, and state whether the per-method central value entering the t-test is the mean over the 7 seeds.
- **Evidence:** `run.py:582-594` · paper: Tables 13, 14, 15, 26; App F, H.2
- **Found in runs:** r02, r05, r06, r10  (representative: r06#0)
- **Quoted at `run.py:582-594`:**
```
# log into txt
mse_mse_string = (f'mse:{mse:.5f}, mae:{mae:.5f}, lamda1:{lamda1:.2f}, '
                  f'git_multi_stage:{args.git_multi_stage}, decoder_cat:{args.decoder_cat_num}, '
                  f'alpha1:{args.alpha}, loss_fun_alpha1:{args.lossfun_alpha}')
print(mse_mse_string)
with open(log_txt, 'a') as f:
    f.write(f'------------ {setting} -------------' + '\n' + '\n')
    args_dict = vars(args)
    for k, v in sorted(args_dict.items()):
        f.write(f'{k}: {v}, ')
    f.write('\n\n')
    f.write('\t' + mse_mse_string + '\n\n')
    f.write('--------------------------------- Ends -----------------------------\n\n')
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
Neurips guidelines explicitly ask for evaluation code. So this is relevant (and also correct, I could not find the statistical evaluations.)
---

### F05 · Reported metric is selected on the test set (best-of-N runs / min over test-batch-sizes & decoder stages by test MSE+MAE)

_category: Methodology / validity · topic: selective reporting / test-set selection_

**severity: medium  (varied: medium, low)  ·  confidence: medium  (varied: medium, low)  ·  detection: 4/10 runs**

- **Claim:** exp.test() returns the TEST-set MSE/MAE; across the itr loop (and any lamda1 sweep / multiple test_batch_size) the driver keeps the run with the lowest test MSE+MAE as best_mse/best_mae, which is what gets logged to best_log.txt / best_results/. The robust/ scripts run this with --itr 7 --fix_seed 0 (e.g. scripts/OLinear/robust/Weather_orthoLinear.sh:51,57), i.e. 7 seeds, reporting the best of 7 by test score.
- **Concern:** Selecting the best of multiple training runs (or hyper-parameter settings) by test-set MSE/MAE is test-set selection; if any reported number is taken from best_results/ rather than a fixed single seed or a mean over seeds, it overstates performance — and the per-seed values needed for the Table 13 std are not aggregated by any script in the repo.
- **Ask:** Authors: confirm the main Tables 2/3 numbers come from the single-seed itr=1 main scripts (not the itr=7 robust best-of-7), and that Table 13's mean/std is computed over all 7 logged per-seed runs rather than the 'global best' the driver prints. Consider selecting by validation, not test.
- **Evidence:** `run.py:568-580` · paper: run.py driver loop; Appendix F Table 13 (std over 7 seeds)
- **Found in runs:** r01, r04, r05, r06  (representative: r01#2)
- **Quoted at `run.py:568-580`:**
```
for test_bs in sorted(test_batch_size_list):
    print('>>>>>>>testing : {} (test_batch_size: {})<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.
          format(setting, test_bs))
    mse0, mae0 = exp.test(setting, test=args.test_mode, test_batch_size=test_bs)

    if mse0 < mse:
        mse, mae = mse0, mae0
        best_batch_size = test_bs
print(f'\tbest_test_batch_size: {best_batch_size}, best_mse: {mse:.5f}, best_mae: {mae:.5f}')

if mse + mae <= best_mse + best_mae:
    best_lamda1 = lamda1
    best_mse, best_mae, best_ii = mse, mae, ii
```

**Verdict:**   correct & relevant `[]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[]`   false `[x]`

**Notes:**
code reading is correct. But the code is not used. It is unclear if the hyperparameters were selected with this leaking procedure. There is no clear evidence in the code but also no evidence of a correct procedure on a validation set. The paper gives only the HP ranges and I coud not find a criterion for picking them. As such the claim that the reported metric is test-set-selected does not clearly hold. Borderline false because of the clear framing "Reported metric is selected on the test set"
---

### F06 · Basis-ablation scripts request an unregistered model key 'OLinear_wavelet_concat' → KeyError

_category: Technical bug · topic: ablation scripts_

**severity: low  ·  confidence: high  ·  detection: 3/10 runs**

- **Claim:** Three of seven basis-ablation scripts (ECL, PEMS03, Solar) list model names 'OLinear_wavelet_concat' / 'OLinear_wavelet2_concat' that are not keys of experiments/exp_basic.py:model_dict (only 'OLinear_wavelet' and 'OLinear_wavelet2' exist) and are not defined anywhere in model/orthoLinear_basis/.
- **Concern:** Running these scripts crashes with a KeyError at self.model_dict[self.args.model] when it reaches the wavelet entry, so the wavelet-basis comparison for those datasets cannot be reproduced as scripted.
- **Ask:** Authors: rename 'OLinear_wavelet_concat'/'OLinear_wavelet2_concat' to the shipped 'OLinear_wavelet'/'OLinear_wavelet2' in the ECL/PEMS03/Solar basis scripts, or add the missing model files.
- **Evidence:** `scripts/ablation/basis/ECL_orthoLinear_basis.sh:6` · paper: Section 5.2 'Comparison with other transformation bases'
- **Found in runs:** r01, r07, r10  (representative: r01#1)
- **Quoted at `scripts/ablation/basis/ECL_orthoLinear_basis.sh:6`:**
```
model_names=(OLinear_FFT OLinear_wavelet_concat OLinear_wavelet2 OLinear_cheby OLinear_Laguerre OLinear_Legendre)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
can be trivially fixed but affects reproducibility of section 5.2 ablations.
---

### F07 · Early-stopping patience / epoch count in shipped scripts (patience 5–8, 30 epochs) disagree with the paper (patience 10, 50 epochs)

_category: Paper–code mismatch · topic: training schedule_

**severity: medium  ·  confidence: high  ·  detection: 3/10 runs**

- **Claim:** Most OLinear scripts set --train_epochs 30 and --patience 5 (counts across scripts/: 23 use 30 epochs, 5 use 50; 73 use patience 5, 31 use patience 10).
- **Concern:** Appendix D states 'Training is performed for up to 50 epochs with early stopping ... if the validation performance does not improve for 10 consecutive epochs', which the released configs do not match; reported numbers may come from a different schedule than documented.
- **Ask:** Authors: reconcile the per-dataset script schedules with the App. D description, or note that the schedule was tuned per dataset.
- **Evidence:** `scripts/OLinear/Weather_orthoLinear.sh:65-66` · paper: Appendix D, Implementation details
- **Found in runs:** r02, r04, r06  (representative: r04#3)
- **Quoted at `scripts/OLinear/Weather_orthoLinear.sh:65-66`:**
```
--train_epochs 30 \
--patience 5 \
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F08 · OLinear-C per-dataset channel-correlation matrices (COV_channel.npy) missing for every dataset except weather

_category: Missing code / data · topic: result traceability / OLinear-C_

**severity: low  ·  confidence: high  ·  detection: 2/10 runs**

- **Claim:** model/OLinear_C.py loads a precomputed channel-correlation matrix via --q_channel_file (lines 42-45), but only weather_COV_channel_ratio0.70.npy is shipped; the 23 other *_COV_channel_*.npy files named by the OLinear_C scripts (including for datasets whose raw CSV IS in the repo: cars, covid, ILI, power, SP500, etc.) are not present (see _audit_code/out/check_channel_files_present.txt: 'required by OLinear_C scripts: 24 / present in dataset/: 1 / MISSING (23)').
- **Concern:** The OLinear-C variant (reported in Appendix H) cannot be run out-of-the-box for 23 of 24 datasets; on a missing file OLinear_C.py:47 does not raise (see olinear-c-missing-file-assert), so self.channel_corr_mat is left undefined.
- **Ask:** Authors: ship the precomputed *_COV_channel_*.npy matrices for all OLinear-C datasets, or document that Generate_corrmat.ipynb (cell 1) must be re-run first (and provide the raw CSVs it needs).
- **Evidence:** `model/OLinear_C.py:42-45` · paper: Appendix H (OLinear-C); §E.2
- **Found in runs:** r03, r07  (representative: r03#1)
- **Quoted at `model/OLinear_C.py:42-45`:**
```
assert configs.q_channel_file is not None, 'configs.q_channel_file should not be None in orthoLinear'
q_channel_file = os.path.join(configs.root_path, configs.q_channel_file)
if os.path.isfile(q_channel_file):
    self.channel_corr_mat = torch.from_numpy(np.load(q_channel_file)).to(torch.float32).to(device)
```

**Verdict:**   correct & relevant `[X]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
can be reproduced but the scripts for that are broken (hardcoded paths etc). One could argue for correct but not relevant
---

### F09 · 'Seven random seeds' claimed but code uses a single hardcoded seed / robustness runs never explicitly seeded

_category: Missing code / data · topic: reproducibility / seeding_

**severity: low  (varied: medium, low)  ·  confidence: high  (varied: high, medium)  ·  detection: 2/10 runs**

- **Claim:** When fix_seed is set (all released scripts use --fix_seed 1), the seed is the constant 2023; there is no mechanism that iterates over seven distinct seeds, and --itr defaults to 1 (the only itr-loop varies a loss hyperparameter, not the seed).
- **Concern:** The paper repeatedly reports robustness 'over seven random seeds' (Tables 13/14/15/26, §F), but the provided code only ever runs a single fixed seed, so the seven-seed runs that underpin those tables cannot be reproduced.
- **Ask:** Add a seed argument or a seed-list loop (and document the seven seed values used), so reviewers can reproduce the per-seed metrics behind Tables 13/14/15/26.
- **Evidence:** `run.py:467-471` · paper: Tables 13, 14, 15, 26; Appendix F
- **Found in runs:** r05, r10  (representative: r05#1)
- **Quoted at `run.py:467-471`:**
```
if args.fix_seed:
    fix_seed = 2023  # 2023  # if args.task_name == 'forecasting' else 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[x]`

**Notes:**
bad/incomplete search by the agent. 7-seed mechanism is in the code: many scripts use the flag --fix_seed 0 --itr 7. The agent probably only saw the fix_seed=1 path.
---

### F10 · OLinear-C missing-channel-file guard uses `assert ValueError(...)` (always truthy; never raises)

_category: Technical bug · topic: error handling_

**severity: low  ·  confidence: high  ·  detection: 2/10 runs**

- **Claim:** When the channel-correlation file is absent, the else branch runs `assert ValueError(...)`; a ValueError instance is always truthy, so the assert passes silently instead of raising, and self.channel_corr_mat is never assigned.
- **Concern:** Instead of a clear 'file missing' error, the model proceeds and later fails with an obscure AttributeError on self.channel_corr_mat; combined with the 23 missing channel files, this hides the real cause from anyone running OLinear-C.
- **Ask:** Replace `assert ValueError(...)` with `raise ValueError(...)` (or `raise FileNotFoundError(q_channel_file)`).
- **Evidence:** `model/OLinear_C.py:42-47` · paper: Appendix H (OLinear-C)
- **Found in runs:** r03, r07  (representative: r03#4)
- **Quoted at `model/OLinear_C.py:42-47`:**
```
assert configs.q_channel_file is not None, 'configs.q_channel_file should not be None in orthoLinear'
q_channel_file = os.path.join(configs.root_path, configs.q_channel_file)
if os.path.isfile(q_channel_file):
    self.channel_corr_mat = torch.from_numpy(np.load(q_channel_file)).to(torch.float32).to(device)
else:
    assert ValueError('self.channel_corr_mat should not be None in orthoLinear_corr_only')
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F11 · OrthoTrans adds undocumented learnable additive terms delta1/delta2 absent from the paper's transform equations

_category: Paper–code mismatch · topic: model architecture / OrthoTrans_

**severity: low  ·  confidence: high  ·  detection: 2/10 runs**

- **Claim:** The default OLinear path (Q_chan_indep=0, the value used by all main scripts) adds learnable parameters self.delta1 (shape [1,N,1,seq_len]) and self.delta2 (shape [1,N,1,pred_len]; declared at OLinear.py:80-81) to the input and output orthogonal projections. The paper's Eqs. (1)-(2) describe the transform as a pure orthogonal projection Z=(RevINNorm(X)⊗φd)Qi^T and ˜Y=LinearDecode(˜HL)·Qo with no additive term.
- **Concern:** A result-affecting learnable component present in the code is absent from the paper's formal description of OrthoTrans, so the published equations under-specify the model that actually produced the numbers; the code is still methodologically valid (the deltas are ordinary learnable biases).
- **Ask:** Authors: state in the paper that OrthoTrans includes per-position learnable bias terms (delta1/delta2), or run the ablation with delta1=delta2=0 to confirm they do not materially change the reported results.
- **Evidence:** `model/OLinear.py:104-118` · paper: §4.1 Eqs. (1)-(2); paper_text.txt lines 254-281
- **Found in runs:** r04, r09  (representative: r09#1)
- **Quoted at `model/OLinear.py:104-118`:**
```
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
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F12 · Q-matrix generation notebook is single-file with hardcoded Windows paths, fixed lags, and is not wired into the pipeline

_category: Missing code / data · topic: reproducibility / Q-matrix generation_

**severity: low  ·  confidence: high  (varied: high, medium)  ·  detection: 2/10 runs**

- **Claim:** The only way to produce the Q matrices for the (absent) benchmark datasets is to manually edit hardcoded per-file Windows paths in Generate_corrmat.ipynb (file_path, save_path, train_ratio comment '0.6 for ETT and PEMS') and rerun each cell once per dataset; there is no automated batch generation.
- **Concern:** Regenerating the missing Q matrices requires error-prone manual edits (and the right train_ratio per dataset, which is implicit), making faithful reproduction of OrthoTrans inputs fragile.
- **Ask:** Authors: convert the notebook into a parameterised script driven by the dataset config so Q matrices are generated automatically with the correct train ratio per dataset.
- **Evidence:** `dataset/Generate_corrmat.ipynb:1` · paper: §4.2; README step 2
- **Found in runs:** r02, r03  (representative: r02#4)
- **Quoted at `dataset/Generate_corrmat.ipynb:1`:**
```
file_path = r'dataset\Solar\solar_AL.xlsx'  # replace with csv or excel file;
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
different view on the same issue mentioned earlier.
---

### F13 · patoolib and tqdm imported at startup but absent from requirements.txt (undeclared dependencies)

_category: Missing code / data · topic: dependencies / environment_

**severity: low  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** data_provider/m4.py unconditionally imports `patoolib` and `tqdm` at module top; m4.py is imported at the top of data_provider/data_loader.py (line 9: `from data_provider.m4 import M4Dataset, M4Meta`), which data_provider/data_factory.py imports, which experiments/exp_forecast.py and run.py import. Neither `patoolib` nor `tqdm` appears in requirements.txt (verified: grep -iE 'tqdm|patool|pyunpack' returns nothing).
- **Concern:** Because the import is on the startup path of run.py for EVERY dataset (not just M4), a fresh environment built from requirements.txt raises ModuleNotFoundError before any experiment can begin, so the published reproduction commands fail out of the box.
- **Ask:** Add `tqdm` and `patoolib` (and its unpacker backend) to requirements.txt, or move the M4-only imports inside Dataset_M4 so the main forecasting path does not require them. (Also note: requirements.txt lists `pywt`, whose PyPI name is `PyWavelets`, and pins no versions.)
- **Evidence:** `data_provider/m4.py:26-27` · paper: Appendix D (implementation details); README 'Usage' step 1
- **Found in runs:** r09  (representative: r09#0)
- **Quoted at `data_provider/m4.py:26-27`:**
```
import patoolib
from tqdm import tqdm
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[x]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
a nuisance but can be fixed very easily. The authors did not use a proper way to extract the environment and did not test their requirement file. tqdm is actually pulled via fvcore/timm, so patoolib is the only runtime break. Severity should be low
---

### F14 · Reported 'MASE' scales by the test-target 1-step naive error, not the in-sample seasonal naive denominator

_category: Paper–code mismatch · topic: metrics_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** MASE is computed with the denominator = mean |y_true[t] - y_true[t-1]| over the *test prediction horizon itself* (y_naive defaults to y_true), rather than the conventional in-sample (training-period) seasonal/naive forecast error.
- **Concern:** The quantity labelled 'MASE' in Table 35 is computed differently from the standard MASE definition, so the absolute values are not comparable to MASE reported elsewhere; the denominator depends on the test target window.
- **Ask:** Authors: state the exact MASE definition used (denominator source and seasonality), and confirm the baseline MASE values in Table 35 use the identical definition.
- **Evidence:** `utils/metrics.py:81-102` · paper: Table 35
- **Found in runs:** r02  (representative: r02#6)
- **Quoted at `utils/metrics.py:81-102`:**
```
if y_naive is None:
    y_naive = y_true
...
# naive MAE [batch, channel]
mae_naive = np.mean(np.abs(y_true_flat[:, 1:, :] - y_naive_flat[:, :-1, :]), axis=1)

# avoid potential error
mae_naive[mae_naive < 1e-5] = np.nan

# MASE
mase = np.nanmean(mae_model / mae_naive)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
This is not a paper code mismatch. It is a non-standard mase implementation but thats also defined like this in the paper. So rather a methodological problem. Deflates the MASEs but the relative comparisons of the method is unaffected.
---

### F15 · Per-dataset hyperparameters are tuned over ranges but the selection criterion is left unstated

_category: Methodology / validity · topic: hyperparameter tuning_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** The paper selects learning rate, model dimension D, batch size and block number L from discrete ranges per dataset (the chosen values are hardcoded in each script), but neither the paper nor the code states whether selection used the validation set or the test set.
- **Concern:** If the per-dataset hyperparameter choice were made on the test split, the comparison would be optimistic; the repo's early-stopping uses validation loss, but the outer HP choice criterion is not documented.
- **Ask:** Authors: confirm that learning rate / D / L / batch size were selected on the validation set (not the test set), and describe the selection criterion.
- **Evidence:** `paper.pdf` · paper: Appendix D
- **Found in runs:** r02  (representative: r02#7)
- **Quoted at `paper.pdf`:**
```
"with the initial learning rate selected from {10−4, 2 × 10−4, 5 × 10−4}. The model dimension D is chosen from {128, 256, 512}, ... The block number L is chosen from {1, 2, 3}."
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
This was already found in earlier findings. Interestingly the learning rate chosen for Weather (1e03) is not in the range they tested according to the paper {1e-4,2e-4,5e-4}.
---

### F16 · run.py references Exp_Short_Term_Forecast / Exp_Long_Term_Forecast_Partial / Exp_Imputation classes that are never imported

_category: Technical bug · topic: driver / entrypoint_

**severity: low  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** Only `Exp_Forecast` is imported (run.py:8); `Exp_Short_Term_Forecast`, `Exp_Long_Term_Forecast_Partial`, and `Exp_Imputation` are undefined names, so any of these branches raises NameError. None of the shipped scripts set `task_name`/`exp_name` (verified: no `task_name`/`exp_name` token in scripts/), so all reproduction scripts hit the default `forecasting` path and avoid the crash.
- **Concern:** A user following Appendix-described settings (e.g. the partial-training experiment of 'Figure 8', or imputation) would hit a NameError; the partial-train / imputation experiments are effectively unrunnable through this entrypoint as shipped.
- **Ask:** Import the referenced Exp_* classes (or remove the dead branches). Confirm whether the Figure-8 partial-training and any imputation results were produced with a different, unshipped driver.
- **Evidence:** `run.py:441-449` · paper: run.py entrypoint; Figure 8 (partial training)
- **Found in runs:** r08  (representative: r08#3)
- **Quoted at `run.py:441-449`:**
```
Exp = Exp_Forecast
if args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast
    args.m4_folder = os.path.join('./m4_results', args.model + '_' + args.m4_result_path_str)
    #  + '_' + datetime.now().strftime('%y%m%d_%H%M%S')
elif args.exp_name == 'partial_train':  # See Figure 8 of our paper, for the detail
    Exp = Exp_Long_Term_Forecast_Partial
elif args.task_name == 'imputation':
    Exp = Exp_Imputation
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
Yes the code would crash, but it does not reach this code with any of the paper-relevant flags. Correct but not relevant to the paper. The dead code is because they forked from TSLib and did not clean up unused elements. It is harmless and only affects code quality and not the result reproducibility.
---

