# Human-eval worksheet — #2371 · 2371

**15 distinct defects** (the 10 PDF+text audit runs' findings, merged by defect). Detection = how many of the 10 runs surfaced the defect (high = robust; 1 = one run only). Severity & confidence are the auditor's own labels (spread shown where runs disagreed); the wording/quote is taken from the highest-confidence run that cited code.

Tick **one** box per defect (put an `x`):

- **correct & relevant** — true *and* a substantive reproducibility issue worth raising
- **correct but wrong severity** — true and worth raising, but the severity label is miscalibrated (e.g. an out-of-the-box crash with a trivial fix tagged high that's really low/medium)
- **correct but not relevant** — technically true but trivial / nitpick / already acknowledged
- **unsure** — can't decide without resources beyond the frozen repo + paper
- **false** — the claim misreads the code/paper and does not hold

Frozen code: `2371/code_frozen/`  ·  paper: `audits/2371/paper.pdf`

---

### F01 · No code for the five generative baselines (TimeVAE/TimeVQVAE/RTSGAN/SDEGAN/LS4) — the Table 1 & 2 comparison rows (and the normalized-average / rank columns built on them) cannot be reproduced

_category: Missing code / data · topic: result traceability_

**severity: high  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** The TSG evaluation writes one CSV row of raw metrics per (dataset, model) invocation; the repo contains no script that produces the 'Norm. Avg.' (Feat./Dist.) or 'Rank' columns of Table 1, and no code that generates the five baseline competitors (TimeVAE, TimeVQVAE, RtsGAN, SdeGAN, LS4) whose rows fill most of Tables 1 and 2.
- **Concern:** The headline comparison ('SDForger outperforms existing generative models … average rank') depends on normalization and ranking across baselines, none of which can be reproduced from the shipped code — the NeurIPS checklist item 5 explicitly asks for scripts to reproduce results 'for the new proposed method and baselines'.
- **Ask:** Provide the aggregation/normalization/ranking script that turns per-run metric CSVs into Table 1's Norm. Avg. and Rank columns, and the baseline-generation code (or the baseline-generated .npy outputs) so each Table 1/2 row is reproducible.
- **Evidence:** `utils/evaluation/utils_evaluation.py:31-46` · paper: Table 1 (Norm. Avg., Rank columns); Table 2
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r02#0)
- **Quoted at `utils/evaluation/utils_evaluation.py:31-46`:**
```
def initialize_results_file_tsg(csv_file_path):

    if os.path.exists(csv_file_path):
        df_results = pd.read_csv(csv_file_path)
    else:
        df_results = pd.DataFrame(columns=[
            'key',
            'seed', 'data', 'target', 'augmentation',
            'n points train', 'n samples train', 'min train windows',
            'train windows', 'windows length', 'overlap', 'period',
            'sdforge llm', 'learning rate', 'train splitting',
            'permute', 'init value', 'embedding type',
            'embedding dim', 'generated samples', 'var requested',
            'var explained', 'MDD', 'ACD', 'SD', 'KD', 'ED', 'DTW', 'SHAP-RE'
        ])
    return df_results
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
The performance of the baseline was not copied over from the publications but they were run. One baselien seems to even have been modified "For LS4, we modified the architecture to support multivariate generation"  but this is not pushed or documented. According to the Neurips Guidelines, this should be pushed. They also did not state why they ommit the baseline (The guidelines ask for this.) Otherweise it is very hard to reproduce the headline improvement claim. The aggreagation is also not reproducible because it is underspecified in the paper. Therefore I agree with "high". The headline claim is very hard to reproduce.
---

### F02 · Section 6 / Figure 2 text-conditioning kNN-classifier accuracy (0.81) has no classifier or evaluation code in the released package

_category: Missing code / data · topic: result traceability_

**severity: medium  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** The conditional-generation notebook (notebook/conditional_generation.ipynb, 19 cells) only generates and plots the channel-conditioned curves of Figure 2; no cell trains or scores a k-nearest-neighbour classifier, and no skfda/accuracy/score call exists anywhere in either repo.
- **Concern:** The reported 0.81 channel-identification accuracy — the quantitative evidence for the textual-conditioning claim of Section 6 — cannot be reproduced from the released code.
- **Ask:** Authors: add the longitudinal-kNN evaluation code (classifier construction, train/test split on real vs. generated curves, accuracy computation) that yields the 0.81 figure.
- **Evidence:** `paper.pdf:939-940` · paper: Section 6, Figure 2
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r03#1)
- **Quoted at `paper.pdf:939-940`:**
```
For instance, using a longitudinal k-nearest neighbor classifier (Ramos-Carreño et al., 2024) trained
on real data, we achieve an accuracy of 0.81 in identifying the generated curves (see Figure 2). These
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
Easy to reimplement but should be shipped. conditional_generation.ipynb does split generated curves by channel label the model predicts but there is no kNN traiend on any real data there.
---

### F03 · ED and DTW pair the i-th real window with the i-th synthetic sample by raw array index (no distributional matching) and only compare min(n) samples, silently discarding surplus generated curves

_category: Methodology / validity · topic: similarity metrics / evaluation validity_

**severity: medium  (varied: high, medium)  ·  confidence: medium  (varied: high, medium)  ·  detection: 8/10 runs**

- **Claim:** Both distance metrics loop i over the real samples and compute the distance between real row i and generated row i (gen_data[i]); my probe (_audit_code/check_ed_dtw_pairing.py) reproduces this and shows (a) reordering the synthetic block changes ED from 0 to 5.64 even when the two sets are identical, and (b) with 30 real vs 100 generated rows the result equals using only the first 30 generated rows.
- **Concern:** Synthetic instances are generated independently and have no correspondence to specific real instances, so pairing real[i] with gen[i] measures the distance between arbitrary unrelated curves (a distributional/closest-match metric is intended); additionally the paper generates 100 synthetic instances but ED/DTW silently use only the first 30.
- **Ask:** Authors: confirm the intended pairing; ED/DTW for set-vs-set generation should use a matching (e.g. nearest-neighbour / optimal assignment) rather than index alignment, and should not discard generated rows beyond the real count.
- **Evidence:** `code/SDForger__neurips_supplemental/utils/evaluation/distance_based_measures.py:13-37` · paper: Table 1 (ED, DTW columns); Appendix B.2
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r09  (representative: r03#6)
- **Quoted at `code/SDForger__neurips_supplemental/utils/evaluation/distance_based_measures.py:13-37`:**
```
def calculate_ed(ori_data,gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0
        for j in range(n_series):
            distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_distance_eu += distance
        distance_eu.append(total_distance_eu / n_series)

    distance_eu = np.array(distance_eu)
    average_distance_eu = distance_eu.mean()
    return average_distance_eu

def calculate_dtw(ori_data,comp_data):
    distance_dtw = []
    n_samples = ori_data.shape[0]
... (+7 more lines)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[x]`

**Notes:**
"additionally the paper generates 100 synthetic instances but ED/DTW silently use only the first 30." This is irrelevant. The paper also never sells this as distributional matching. There is a problem: the metric rewards mode collapse, but it seems like the metric faithfully implements what the paper states.
---

### F04 · No driver / aggregation script computes the Table 1 & 2 normalized-average and average-rank columns across the 12-dataset benchmark (and no per-dataset configs are shipped)

_category: Missing code / data · topic: result traceability / experiment protocol_

**severity: medium  (varied: medium, low)  ·  confidence: high  (varied: high, medium)  ·  detection: 5/10 runs**

- **Claim:** The TSG evaluation writes one row of raw metrics for a single (dataset,config) run; only two example configs exist (`config.yaml`, `config_ttm.yaml`) for bikesharing/ecl, and no script normalizes scores to [0,1] per metric group or computes the average-rank column that Table 1/Table 2 report.
- **Concern:** The reported aggregated tables (12 datasets, normalized averages, ranks) and Appendix Tables D.1–D.14 cannot be reproduced because the per-dataset drivers and the aggregation/normalization/ranking step are absent.
- **Ask:** Authors: add the per-dataset config set (or a manifest) and the script that aggregates raw CSV rows into the normalized-average and average-rank columns of Tables 1–2 and Appendix D.
- **Evidence:** `code/SDForger__neurips_supplemental/sources/run_TSG_evaluation.py:149-157` · paper: Table 1 'Norm. Avg.'/'Avg. Rank'; Appendix D
- **Found in runs:** r05, r06, r08, r09, r10  (representative: r06#1)
- **Quoted at `code/SDForger__neurips_supplemental/sources/run_TSG_evaluation.py:149-157`:**
```
print(f'\nTSG Evaluation')
result = tsg_evaluation( original_dataset, generated_dataset )

# result_shivani = shivani_evaluation( original_dataset, generated_dataset, device, result_path_visualization = os.path.join(OUTPUT_PATH, str(target)),
#                         combined_name = generated_data_path.split("/")[-2] + suffix )

print('\nSaving sdforger')
save_results(df_results, csv_file_path, values_to_save +
            [result['MDD'], result['ACD'], result['SD'], result['KD'], result['ED'], result['DTW'], result['SHAP-RE']])
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
Near duplicate of F01, except for the dataset configs, which is a fair critique. There is only the run config for bikesharing not for the 14 other datasets. The others can be guessed but it is a reprod. nuisance.
---

### F05 · Table 2 'WQL' is computed from a single point forecast (~0.5*MAE / un-normalised mean pinball loss), not a true Weighted Quantile Loss over quantile forecasts

_category: Methodology / validity · topic: utility metrics / WQL_

**severity: medium  (varied: medium, low)  ·  confidence: medium  (varied: high, medium)  ·  detection: 5/10 runs**

- **Claim:** The Weighted Quantile Loss in Table 2 is computed by applying the pinball loss at quantiles 0.1/0.5/0.9 to the SAME single point forecast `pred` (TTM's deterministic prediction), rather than to per-quantile forecasts.
- **Concern:** WQL is meant to score a probabilistic (multi-quantile) forecast; evaluating each quantile against one point prediction degenerates to a fixed reweighting of the absolute error and does not measure calibration, so the Table 2 'WQL' column is not the standard quantity, though it is applied identically across all training sources.
- **Ask:** Authors: clarify whether TTM produces quantile outputs; if not, state that 'WQL' here is a point-forecast pinball average so it is not over-interpreted as a probabilistic-forecast score.
- **Evidence:** `utils/evaluation/utils_ttm.py:253-259` · paper: Table 2 (WQL columns)
- **Found in runs:** r01, r03, r05, r06, r09  (representative: r01#4)
- **Quoted at `utils/evaluation/utils_ttm.py:253-259`:**
```
def compute_wql(true, pred, quantiles=[0.1, 0.5, 0.9]):
    """Compute Weighted Quantile Loss (WQL)"""
    total_loss = 0
    for q in quantiles:
        errors = true - pred
        total_loss += np.mean(np.maximum(q * errors, (q - 1) * errors))
    return total_loss / len(quantiles)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
Identical for all models, so not an inflation. But as such the metric doesnt measure calibration. The metric is not defined in the paper.
---

### F06 · SHAP-RE (SHR) reshape collapses the channel axis — valid only for univariate input; silently wrong / crashes for multivariate (C>1)

_category: Technical bug · topic: similarity metrics / SHAP-RE_

**severity: low  ·  confidence: high  (varied: high, medium)  ·  detection: 4/10 runs**

- **Claim:** calculate_shapelet_recons_err reshapes a (n_samples, length, n_channels) array to (n_samples, length), which is only size-consistent when n_channels==1; for C>1 numpy raises 'cannot reshape array of size ... into shape (n,length)'.
- **Concern:** The SHAP-RE column is reported in Table 1 only for the single-channel similarity settings, so the reported numbers are unaffected, but the metric silently cannot be applied to multivariate data and would crash if used there.
- **Ask:** Authors: confirm SHAP-RE was computed per channel for single-channel data only, or generalise the reshape to handle C>1 (e.g. flatten or loop over channels).
- **Evidence:** `utils/evaluation/shapelet_based_measures.py:20-24` · paper: Table 1, SHR column
- **Found in runs:** r02, r05, r06, r10  (representative: r10#3)
- **Quoted at `utils/evaluation/shapelet_based_measures.py:20-24`:**
```
train_data  = orig_data.reshape(orig_data.shape[0], orig_data.shape[1])
test_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1])
train_y  = np.random.rand(orig_data.shape[0])
test_y = np.random.rand(gen_data.shape[0])
n_train, p = train_data.shape
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
doesnt affect the papers run in any way. Only called for univariate inputs.
---

### F07 · MASE denominator uses the forecast window's own naive error, not the in-sample / training-set (seasonal) naive error of the paper's definition

_category: Paper–code mismatch · topic: evaluation metrics_

**severity: medium  ·  confidence: medium  (varied: high, medium)  ·  detection: 4/10 runs**

- **Claim:** MASE here divides the mean absolute forecast error by the mean absolute first difference of the *future (test) values being forecast*, rather than by the in-sample one-step naive error of the training series as in the standard MASE definition (Hyndman & Koehler 2006).
- **Concern:** The denominator is computed from the same window being scored, so the reported 'MASE' is a non-standard scaling; it is applied uniformly to all methods so within-table rankings are still internally comparable, but the absolute MASE values are not the conventional metric.
- **Ask:** Authors: confirm whether MASE was intended to scale by the training-series naive error; if the horizon-based scaling is intended, label it as a scaled MAE / custom MASE.
- **Evidence:** `utils/evaluation/utils_ttm.py:247-251` · paper: Table 2 (MASE columns)
- **Found in runs:** r02, r06, r07, r09  (representative: r07#2)
- **Quoted at `utils/evaluation/utils_ttm.py:247-251`:**
```
def compute_mase(true, pred):
    """Compute MASE (Mean Absolute Scaled Error)"""
    numerator = np.mean(np.abs(true - pred))
    denominator = np.mean(np.abs(true[1:] - true[:-1]))  # Naive one-step ahead forecast
    return numerator / denominator if denominator != 0 else np.nan
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
very slight deviation form standard mase definition. Nitpick: Maybe not really a paper code mismatch because the paper never defines it. So maybe a technical bug...
---

### F08 · SHAP-RE (SHR) uses an unseeded RNG and is nondeterministic (varies ~13% run-to-run on identical inputs)

_category: Methodology / validity · topic: evaluation metrics / reproducibility_

**severity: low  ·  confidence: high  ·  detection: 3/10 runs**

- **Claim:** calculate_shapelet_recons_err uses np.random (A_rand_init, Offsets_test, plus S/A/Offsets in USIDL and np.random.permutation inside update_A_par) without any seeding, and sources/run_TSG_evaluation.py never calls set_seed; calling the repo's own function five times on identical inputs (check_shapre_nondeterminism.py) gives SHAP-RE values [1.62, 1.75, 1.81, 1.72, 1.59] with relative spread ~0.13.
- **Concern:** The SHR column of Table 1 is a single reported number per model, but the metric is non-deterministic with ~13% run-to-run variation on fixed inputs, so the reported SHR values (and the resulting model ranking on that metric) are not reproducible and could change the comparison.
- **Ask:** Authors: seed all RNGs before the SHAP-RE computation (or average over multiple seeds and report variance), and state how many SIDL restarts / which seed produced the reported SHR values.
- **Evidence:** `utils/evaluation/shapelet_based_measures.py:6-39` · paper: Appendix B.2 'Shapelet-based Reconstructions'; Table 1 SHR column
- **Found in runs:** r03, r04, r07  (representative: r04#3)
- **Quoted at `utils/evaluation/shapelet_based_measures.py:6-39`:**
```
def calculate_shapelet_recons_err(orig_data, gen_data, K=20, lambdas =0.1, r=0.25):

    c = 100
    epsilon = 1e-5
    maxIter = 1e3
    maxInnerIter = 5

    # Reverse the transpose operation
    # orig_data = orig_data.transpose(2, 0, 1)
    # gen_data = gen_data.transpose(2, 0, 1)
    orig_data = orig_data
    gen_data = gen_data


    train_data  = orig_data.reshape(orig_data.shape[0], orig_data.shape[1])
    test_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1])
    train_y  = np.random.rand(orig_data.shape[0])
    test_y = np.random.rand(gen_data.shape[0])
... (+16 more lines)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
trivial fix. 13% different outcome on identical inputs. There is no seed, no reported variance. The paper prints it as point estimate.  It should either be seeded or a measure of variability should be stated.
---

### F09 · Config `seed` is not propagated to the SDForger LLM (RNG hardcoded to 42), undermining the paper's '5 random seeds' averaging

_category: Methodology / validity · topic: reproducibility / seeding_

**severity: low  (varied: medium, low)  ·  confidence: high  ·  detection: 3/10 runs**

- **Claim:** sdforger_augmentation.py:224 constructs `SDForger(model_path=llm, text_template=..., float_type=...)` with no seed kwarg, so SDForger.__init__ sets self.seed=42 and immediately calls self.set_seed() (sdforger.py:834-840), which runs random.seed(42)/np.random.seed(42)/torch.manual_seed(42) — overwriting the set_seed(SEED=54) the driver had called in run_data_augmentation.py:73. The LLM train/val split (sdforger.py:273) and column permutations then use 42 regardless of config; FastICA is additionally fixed at random_state=0 (sdforger_augmentation.py:165). My check reproduces the override.
- **Concern:** Table D.2 reports statistics 'averaged across 5 seeds' and the abstract leans on robustness, but the dominant generation randomness (LLM finetuning split, sampling, permutation) is pinned to a constant 42, so the 5 seeds vary far less than implied and the multi-seed averages may understate variance.
- **Ask:** Authors: pass the config seed into SDForger (or remove the hardcoded 42) and confirm which randomness sources actually differ across the 5 seeds reported in Table D.2.
- **Evidence:** `utils/augmentation/sdforger.py:86-87` · paper: Appendix Table D.2 ('averaged across 5 seeds')
- **Found in runs:** r05, r08, r09  (representative: r05#5)
- **Quoted at `utils/augmentation/sdforger.py:86-87`:**
```
self.seed = kwargs['seed'] if 'seed' in kwargs else 42
self.set_seed()
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
very minor issue. Trivial fix. But config.yaml seed looks like it controls SDForger, but then does not
---

### F10 · TTM utility evaluation depends on tsfm_public / granite-tsfm, which is absent from the env files and installed from an unpinned main / moving model revision

_category: Missing code / data · topic: dependencies / reproducibility_

**severity: low  (varied: medium, low)  ·  confidence: high  (varied: high, medium)  ·  detection: 3/10 runs**

- **Claim:** The TTM utility evaluation imports tsfm_public, but neither sdforgerpy310cuda.yaml nor sdforgerpy310mps.yaml lists tsfm/granite-tsfm; the README installs it from an unpinned `main`, and config_ttm.yaml sets TTM_MODEL_REVISION: main, so both the library and the pretrained TTM checkpoint are moving targets.
- **Concern:** Table 2's utility numbers depend on an unpinned dependency and an unpinned model revision, so the exact environment that produced them cannot be rebuilt.
- **Ask:** Authors: pin a granite-tsfm commit and a concrete TTM model revision (not `main`), and add tsfm to the environment file.
- **Evidence:** `code/SDForger__neurips_supplemental/README.md:37-42` · paper: Table 2
- **Found in runs:** r03, r08, r09  (representative: r03#2)
- **Quoted at `code/SDForger__neurips_supplemental/README.md:37-42`:**
```
To run TTM evaluation:
```shell
git clone "https://github.com/ibm-granite/granite-tsfm.git"
cd granite-tsfm
pip install ".[notebooks]"
```
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
library side changes constantly (actively developed IBM repo); pip install from an unpinned clone gives you whatever main is that day. Could even be medium severity
---

### F11 · ED is computed on StandardScaler (zero-mean/unit-variance) data, but Appendix B.2 defines it on [0,1]-scaled inputs

_category: Paper–code mismatch · topic: evaluation consistency / preprocessing_

**severity: low  ·  confidence: high  ·  detection: 3/10 runs**

- **Claim:** All series are standard-scaled per timestamp (zero mean, unit variance) before generation and evaluation; the Euclidean Distance metric is then computed on these standard-scaled values.
- **Concern:** Appendix B.2 defines ED stating 'the input time series has been preprocessed to fit within the range of [0,1]', whereas the code (and the protocol in Appendix C.2) use StandardScaler, so the ED values are not on the [0,1] basis the metric definition assumes (the protocol and metric-definition sections of the paper are themselves inconsistent).
- **Ask:** Authors: reconcile the ED definition (B.2, '[0,1]') with the actual standard-scaling used (C.2 and code); confirm which scaling produced the Table 1 ED column.
- **Evidence:** `utils/augmentation/utils_preprocess_data.py:234-237` · paper: Appendix B.2 (ED), Appendix C.2 (protocol)
- **Found in runs:** r01, r02, r06  (representative: r01#2)
- **Quoted at `utils/augmentation/utils_preprocess_data.py:234-237`:**
```
arr_train = np.array(preprocessed_train[var])
scaler = StandardScaler()
scaled_arr_train = scaler.fit_transform(arr_train)
scaled_preprocessed_train.append(scaled_arr_train)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**the paper is not consistent:
"the input time series has been preprocessed to fit within the range of [0, 1]"
"Standard scaling per timestamp is applied"
---

### F12 · TTM RMSE/MASE/WQL are averaged over all forecast channels, not just the stated target channel

_category: Methodology / validity · topic: utility evaluation / metric scope_

**severity: medium  (varied: medium, low)  ·  confidence: low  ·  detection: 2/10 runs**

- **Claim:** evaluate_ttm_model loops over every channel in pred_val (range(0, pred_val.shape[2])) and averages the per-channel RMSE/MASE/WQL uniformly, whereas Table 2 frames the task as forecasting a single target (e.g. bikesharing target: count, controls: temperature, humidity).
- **Concern:** If pred_val carries all input channels (target + controls), the reported per-dataset metric mixes the target's error with control-channel errors, so the Table-2 numbers would not be the pure target-forecast metric the caption implies; whether pred_val.shape[2]==1 (target-only) depends on tsfm internals I could not execute, hence a question.
- **Ask:** Authors: confirm whether pred_val contains only the target channel or all channels; if all channels, restrict the metric to prediction_channel_indices (the target) so it matches the Table-2 caption.
- **Evidence:** `utils/evaluation/utils_ttm.py:335-357` · paper: Table 2 (target/control framing)1
- **Found in runs:** r03, r10  (representative: r10#5)
- **Quoted at `utils/evaluation/utils_ttm.py:335-357`:**
```1
for sample in range(0, true_val.shape[0]):
      list_rmse_per_channel = []
      list_mase_per_channel = []
      list_wql_per_channel = []
      list_h1_per_channel = []
      for channel in range(0, pred_val.shape[2]):
          rmse = compute_rmse(true_val[sample, :, channel], pred_val[sample, :, channel])
          mase = compute_mase(np.array(true_val[sample, :, channel]), np.array(pred_val[sample, :, channel]))
          wql = compute_wql(np.array(true_val[sample, :, channel]), np.array(pred_val[sample, :, channel]))
          h1 = compute_h1_distance(np.array(true_val[sample, :, channel]), np.array(pred_val[sample, :, channel]))
          list_rmse_per_channel.append(rmse)
          list_mase_per_channel.append(mase)
          list_wql_per_channel.append(wql)
          list_h1_per_channel.append(h1)
      rmse_list.append(list_rmse_per_channel)
      mase_list.append(list_mase_per_channel)
      wql_list.append(list_wql_per_channel)
      h1_list.append(list_h1_per_channel)
... (+3 more lines)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**
The finding is true only for the 0-shot baseline. Problematic is: the 0-shot baseline and the three finetuned rows don't measure the same quantity. Finetuned is cnt only
---

### F13 · TrainingArguments hardcodes bf16=True, contradicting the float32 config and breaking the documented MPS run path

_category: Technical bug · topic: reproducibility / training config_

**severity: medium  (varied: medium, low)  ·  confidence: medium  ·  detection: 2/10 runs**

- **Claim:** `bf16=True` is passed unconditionally to HF `TrainingArguments`, while on darwin `self.train_args` also sets `use_mps_device=True` (lines 47-50); the README documents a local MPS run path (`sdforgerpy310mps.yaml`).
- **Concern:** bf16 mixed-precision training is not supported on Apple MPS / CPU in this configuration, so the documented MPS reproduction command is likely to raise an error before training; it works on the A100/CUDA path used for the paper but not on the alternative path the README advertises.
- **Ask:** Authors: gate `bf16` on CUDA availability (e.g. set it from device, as in the commented-out line 50) so the documented MPS/CPU path runs.
- **Evidence:** `utils/augmentation/sdforger.py:287-306` · paper: README 'Setup' (MPS); Parameter settings §4
- **Found in runs:** r03, r08  (representative: r08#4)
- **Quoted at `utils/augmentation/sdforger.py:287-306`:**
```
training_args = TrainingArguments(
    self.output_dir,
    **self.train_args,
    adam_epsilon=1e-04,
    logging_strategy="steps",
    logging_steps=10,
    # weight_decay=0.0001,
    # optim='adafactor',
    # max_grad_norm=1,
    # max_grad_norm=5,
    evaluation_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    # metric_for_best_model="loss",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
... (+2 more lines)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[x]`

**Notes:**
The finding says the Mac (MPS) reproduction command will crash before training because bf16 is turned on. I checked the actual pinned versions and it doesn't. Nothing I found (the training config, accelerate, or PyTorch itself) throws an error on a Mac. I think on older macs it just falls back to regular float32 on newer Macs it just works with bf16. The hardcoding is not super nice and this could maybe break in some hardware settings, but I think its false, maybe borderline not relevant.

### F14 · Table D.5 generation-time comparison has no timing / benchmark script

_category: Missing code / data · topic: result traceability / efficiency_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** Table D.5 reports per-model wall-clock generation times for SDForger and all five baselines, but no timing/benchmark harness exists in either repo (the drivers do not record generation time, and there is no baseline code to time).
- **Concern:** The efficiency claim ('SDForger is substantially faster than all competitors, often by one to two orders of magnitude') cannot be reproduced from the code.
- **Ask:** Authors: release the timing harness and the baseline runners used for Table D.5.
- **Evidence:** `paper.pdf:1865-1868` · paper: Table D.5; §5.3 Generation efficiency
- **Found in runs:** r03  (representative: r03#3)
- **Quoted at `paper.pdf:1865-1868`:**
```
Table D.5: Average generation time: baselines Average time (in seconds) required to generate synthetic
univariate time series for the bikesharing dataset across three targets: count, temperature, and humidity.
We report results for two input sequence lengths: 250 and 500. All models were evaluated under the same
computational constraints (-mem 20G -cores 1+1 -gpu v100) using a single NVIDIA V100 GPU.
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
Paper: "faster than all competitors, often by one to two orders of magnitude. TimeVAE is the closest competitor but remains over 4× slower." Should be backed up by code.
---

### F15 · Shipped univariate config diverges from the paper's reported settings (k=5 and L0=2000 with 1 sample vs the paper's k=3)

_category: Paper–code mismatch · topic: hyperparameters / configs_

**severity: low  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** The default univariate example config sets sdforger_embedding_dim: 5, whereas the paper fixes k=3 for the multisample and univariate settings (Section 4 'Parameter settings'). The repo's README is consistent with the paper text (k=3), so the example config differs from the paper's reported setting.
- **Concern:** A user running the shipped config out-of-the-box reproduces a k=5 run, not the k=3 setting that the paper's main univariate results use; the discrepancy is a config default, and the code path supports k=3 via the same field.
- **Ask:** Authors: set the example config defaults to the paper's reported k (3) for the main settings, or document that the shipped configs are illustrative rather than the exact Table-1 settings.
- **Evidence:** `sources/config/config.yaml:4-15` · paper: Section 4 'Parameter settings' (k=3); README
- **Found in runs:** r10  (representative: r10#4)
- **Quoted at `sources/config/config.yaml:4-15`:**
```
data_train_channels:
- cnt
data_train_params:
- 2000
- 1
evaluation:
  generated_data_path: output/generated_data.npy
  train_data_path: output/train_data.npy
save_results: true
sdforger_augmentation_strategy: univariate
sdforger_batch: 32
sdforger_embedding_dim: 5
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**
slight nitpick: the readme states embedding dimension per channel (3, 5, 7, auto). So it doesnt state the exactoption
---

