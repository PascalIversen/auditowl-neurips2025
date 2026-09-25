# Code Audit — Paper 768: "A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning" (OTQMS)

## 1. Summary

The repository `zqy0126__OTQMS` is the author code for OTQMS, a multi-source
transfer-learning sampling method. It contains the full training/evaluation
pipeline: source-model pretraining (`sources_model_pretrain.py`), the main joint
training/evaluation driver (`main.py`), the OTQMS optimal-quantity computation
(`misc.py:fisher_and_get_multis_optimal_m`, `solve_quadratic_problem`), the
per-class train/eval split (`misc.py:get_trainset_and_evalset_ratioway`), dataset
construction (`datasets.py`), the model (`algorithms.py`, `networks.py`), configs
(`config_json/`), and a dataset downloader (`download.py`). The repo is a single
moving-`main` commit (`77aa8ca OTQMS end`), not a submission-tagged commit.

I read every Python file, the four JSON configs, the README, and `requirements.txt`,
and cross-read the paper (`paper.pdf`, located via `paper_text.txt`). I did not run
the training pipeline (it requires DomainNet/Office-Home downloads, pretrained
ViT-S checkpoints, and an A800 GPU). I wrote two static read-only checks under
`_audit_code/` that confirm: (a) the implemented S-objective contains a `+ 1/S`
term absent from the paper's Eq. (14)/(72); (b) the Fisher matrix is built from
min-max-normalized gradients, a transformation not described in the paper; and
(c) the reported "test" accuracy is the maximum over training epochs of the
accuracy on the same 20% per-class holdout that serves as the test set.

The procedure is broadly internally consistent and the baselines (AllSources∪Target,
Target-Only) go through the identical pipeline, so several concerns are symmetric
across methods. The main substantive findings are (i) two undocumented components
in the core OTQMS objective/Fisher computation that diverge from the paper's stated
math, (ii) model selection (best epoch) performed on the test set with no separate
validation set, (iii) the pretrained-checkpoint path consumed by `main.py` does not
match where `sources_model_pretrain.py` writes them, and (iv) a paper/code learning-
rate mismatch (paper 1e-5, code default 5e-5; no config overrides it).

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 per-domain accuracies (OTQMS, AllSources∪Target, Target-Only, Single-Source) | `main.py:384` (`misc.accuracy`) via `test∈{m,all,none}`; Single-Source not wired in driver | per-epoch eval acc | not runnable here | Pipeline present (OTQMS/all/none); Single-Source-Avg/Best not in driver → see `single-source-baselines-missing` |
| Table 2 unsupervised rows (MSFDA, DATE, M3SDA) | (none) | — | — | Acknowledged: "report their original results from the respective papers" (not a finding) |
| Table 2 baselines MADA, WADN, H-ensemble, MCW | (none in this repo) | — | — | MISSING from this repo → see `external-baseline-code-missing` |
| Table 3 Static-vs-Dynamic (Static-Under/Exact/Over) | (none) | — | — | MISSING (no static-mode branch in code) → see `static-ablation-missing` |
| Fig. 3 increasing-shots curves | `main.py` (loop over `target_perclass_list`) | per-shot acc | not runnable here | Pipeline present; plotted PDFs in `output_images/` |
| Fig. 4 data-efficiency (sample usage / time) | `main.py:290,343` (`final_train_all_samples_num`), `batch_time` logging | counts/time | not runnable here | Pipeline present |
| Fig. 5 domain-preference heatmaps | `main.py:409` writes `alpha`/`m` per epoch | α, m values | not runnable here | Pipeline present |
| Table 6 LoRA (ViT-B) | (none) | — | — | MISSING (no LoRA code) → see `lora-code-missing` |
| Table 7 Digits/WADN (3Conv) | (none) | — | — | MISSING (no Digits dataset class / 3Conv backbone wired) → see `digits-code-missing` |
| Core: optimal (s*, α*) = argmin Eq. (14)/(72) | `misc.py:321-345` | grid search; objective `+1/S` and min-max Fisher | NO (extra term + undocumented normalization) | MISMATCH → see `objective-extra-term`, `fisher-minmax-normalization` |
| lr = 1e-5 (paper §5.1) | `config_json/*.json` (no `lr`) → `hparams_registry.py:43` default 5e-5 | 5e-5 | NO | MISMATCH → see `lr-mismatch` |

## 3. Findings

## missing

```yaml finding
id: pretrained-ckpt-path-mismatch
category: missing
topic: "result traceability / pretrained weights"
title: "main.py reads pretrained source models from a path the pretrain script never writes"
severity: high
confidence: high
status: finding
file: main.py
line_start: 302
line_end: 308
quote: |
                        src_epoch_info = pd.read_csv(f"{args.output_dir}/opti_pre_models_vits_850batch/combined_allofthem.csv") 
                        Acc_srcenv_list = [src_epoch_info[f'Acc_{i}'].values for i in src_envs]
                        max_acc_epoch_list = [np.nanargmax(Acc_srcenv) for Acc_srcenv in Acc_srcenv_list]
                        logger.info(f"Max acc epoch list: {max_acc_epoch_list}")
                        pretrained_model =[
                            f"{args.output_dir}/opti_pre_models_vits_850batch/env{i}_best_checkpoint.pth" for i in src_envs
                        ]
claim: "The OTQMS path of main.py loads source-model checkpoints and a combined CSV from `{output_dir}/opti_pre_models_vits_850batch/`, but sources_model_pretrain.py writes its checkpoints to `{output_dir}/ckp_file/{n_filtered_args}/pretest none/env{env}/` and its combined CSV to `{output_dir}/table_file/{n_filtered_args}/pretest none/env0/combined_allofthem.csv`."
concern: "The directory `opti_pre_models_vits_850batch/` is never created or populated by any script in the repo and is not present, so main.py (the OTQMS run) cannot find the source models without an undocumented manual copy/rename step, breaking reproduction."
resolution: "Authors: add the script/step that assembles `opti_pre_models_vits_850batch/` (renaming env{i}_best_checkpoint.pth and combined_allofthem.csv from the pretrain outputs), or change main.py to read from the actual pretrain output paths."
cross_refs: []
check_script: _audit_code/check_objective_and_split.py
paper_ref: "Algorithm 1; README 'Initialize the source parameters'"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: static-ablation-missing
category: missing
topic: "ablations"
title: "Static-vs-Dynamic ablation (Table 3) has no Static branch in the code"
severity: medium
confidence: medium
status: finding
file: datasets.py
line_start: 176
line_end: 185
quote: |
            if test == "m": # OTQMS transfer
                pass
            elif test == "all": # all sources transfer
                add_env_nums_list = most_train_size_list
                add_env_nums_list[args.test_envs[0]] = 0
            elif test == "none": # target only
                add_env_nums_list = [0 for _ in range(len(most_train_size_list))]
            # elif test == 1/2: 
            #     add_env_nums_list = [ (i+j)/2 for i,j in zip(add_env_nums_list, most_train_size_list)]
            #     add_env_nums_list[args.test_envs[0]] = 0
claim: "The dataset builder only supports test modes m (OTQMS/dynamic), all, and none; there is no 'Static-Under/Exact/Over' mode (compute optimal m once and freeze it), which is what Table 3 reports."
concern: "Table 3's Static-Under/Exact/Over results cannot be reproduced because the repo contains no code path that computes the transfer quantity once and trains to convergence on a fixed resampled set."
resolution: "Authors: provide the Static-* implementation, or confirm Table 3 was produced by a separate (un-shared) configuration."
cross_refs: []
paper_ref: "Table 3, §5.3"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-experiments-code-missing
category: missing
topic: "result completeness"
title: "LoRA (Table 6), Digits/WADN (Table 7), ResNet/ViT-B variants not wired into the runnable code"
severity: medium
confidence: medium
status: question
file: datasets.py
line_start: 16
line_end: 23
quote: |
  DATASETS = [
      # Small images
      "ColoredMNIST",
      "RotatedMNIST",
      # Big images
      "OfficeHome",
      "DomainNet",
  ]
claim: "Only DomainNet and OfficeHome dataset classes exist; there is no Digits dataset class, no 3-layer ConvNet backbone, and no LoRA / ViT-B integration, although the configs include res18/res50/res101 backbones."
concern: "The supplementary results in Tables 6 (LoRA ViT-B) and 7 (Digits, 3Conv) have no corresponding code path in the repo, so those experiments are not reproducible from the released artefact."
resolution: "Authors: add the Digits dataset class + 3Conv backbone and the LoRA/ViT-B integration, or state they were run from a separate codebase."
cross_refs: ["static-ablation-missing"]
paper_ref: "Table 6 (App. D), Table 7 (App. D.2)"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: false
  condition_satisfiable: true
```

```yaml finding
id: external-baseline-code-missing
category: missing
topic: "baselines"
title: "Sample/model-weighting baselines (MADA, WADN, H-ensemble, MCW) absent; only OTQMS/AllSources/Target-Only runnable"
severity: low
confidence: high
status: finding
file: algorithms.py
line_start: 24
line_end: 26
quote: |
  ALGORITHMS = [
      'Computheta',
  ]
claim: "The repo implements a single algorithm (Computheta). The driver only supports test modes m/all/none (OTQMS, AllSources∪Target, Target-Only). No code reproduces the MADA, WADN, H-ensemble, MCW, or Single-Source-Avg/Best baseline numbers in Table 2."
concern: "The competing baselines that OTQMS is claimed to beat by 1.5%/1.0% are not in this repo, so the head-to-head comparison cannot be reproduced from the released code (the in-repo baselines AllSources∪Target and Target-Only are reproducible)."
resolution: "Authors: point to the baseline implementations / commands, or confirm baselines were run from their original repos under the adapted settings described in App. D.2."
cross_refs: ["single-source-baselines-missing"]
paper_ref: "Table 2; §5.1 Baselines"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: cross-entropy-reduce-string
category: bug
topic: "Fisher computation"
title: "F.cross_entropy called with reduce='sum' (a string), which is truthy and does not select sum reduction"
severity: low
confidence: medium
status: finding
file: algorithms.py
line_start: 139
line_end: 139
quote: |
            env_loss = F.cross_entropy(logits, y, reduce='sum')
claim: "update_for_fisher (used to build the empirical-Fisher gradient) passes the deprecated `reduce` flag as the string 'sum'; PyTorch interprets `reduce` as a bool, and any non-empty string is truthy, so this yields the default mean reduction, not a sum."
concern: "The intended per-sample gradient-sum scaling for the empirical Fisher is not applied as written; downstream the gradient is divided by len(x)**0.5 and min-max normalized, so the numerical effect is largely washed out, but the code does not do what it plainly intends."
resolution: "Authors: replace `reduce='sum'` with `reduction='sum'` and confirm the reported Fisher / optimal-m values are unaffected."
cross_refs: ["fisher-minmax-normalization"]
paper_ref: "Algorithm 1, line 10"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: objective-extra-term
category: difference
topic: "core method / objective"
title: "Implemented S-selection objective adds a +1/S term absent from paper Eq. (14)/(72)"
severity: high
confidence: high
status: finding
file: misc.py
line_start: 331
line_end: 331
quote: |
        equation_result = (1/(n0+S) + (S**2 * trace_term)/(d*(n0+S)**2)) * d/2 + 1/S
claim: "The objective minimized over the total transfer quantity S is `(d/2)*(1/(N0+S) + S^2*t/(d*(N0+S)^2)) + 1/S`, i.e. the paper's measure plus an extra `+ 1/S` term."
concern: "The paper's measure (Eq. 14 / Appendix-E Eq. 72) is exactly `(d/2)*(1/(N0+s) + s^2/(N0+s)^2 * t)` with no `1/S` term; the extra monotone-decreasing-in-S term biases the selected S upward relative to the stated theory, so the implemented selection rule is not the one derived in the paper."
resolution: "Authors: explain the `+ 1/S` term (regularizer? typo?) and report how the selected (s*, α*) and the Table-2/3 results change when it is removed."
cross_refs: ["fisher-minmax-normalization"]
check_script: _audit_code/check_objective_and_split.py
paper_ref: "Eq. (14); Appendix E Eq. (72)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: fisher-minmax-normalization
category: difference
topic: "core method / Fisher information"
title: "Fisher matrix A built from per-batch min-max-normalized gradients, not the empirical Fisher described"
severity: high
confidence: high
status: finding
file: misc.py
line_start: 294
line_end: 306
quote: |
            layer_minmax_fisher_info_vector = []
            for fisher_value in fisher_info_vector:
                x_min = torch.min(fisher_value)
                x_max = torch.max(fisher_value)
                normalized_fisher_value = 2 * (fisher_value - x_min) / (x_max - x_min) - 1
                layer_minmax_fisher_info_vector.append(normalized_fisher_value.double())
            layer_minmax_fisher_info_vector_tensor = torch.cat([norm_p.view(-1,1) for norm_p in layer_minmax_fisher_info_vector]).double()

            # if j==0 or j==len(train_loader)-1:
            #     logger.info(f"batch_th: {j}, all_minmax_norm_fisher_vector: {layer_minmax_fisher_info_vector_tensor}")
            temp_gtg = (layer_minmax_fisher_info_vector_tensor.t() @ layer_minmax_fisher_info_vector_tensor)/d_dimension
            # logger.info(f"target_fisher_info_vector_temp_gtg: {temp_gtg}")
            A.add_(model_param_diffs @ layer_minmax_fisher_info_vector_tensor @ layer_minmax_fisher_info_vector_tensor.t() @ model_param_diffs.t())
claim: "Each layer's per-batch Fisher gradient is min-max rescaled to [-1, 1] before the outer product that accumulates A = Θ^T J Θ, so A is built from normalized gradients rather than the raw empirical-Fisher gradients."
concern: "The paper states J(θ0) is the empirical Fisher = mean of raw gradient outer products (Algorithm 1 line 10; Appendix E uses Θ^T J(θ0) Θ); the code instead linearly rescales each layer's gradient to [-1,1] before forming the outer product, an undocumented transformation that changes A=Θ^TJΘ and hence the chosen α*/s*."
resolution: "Authors: document the min-max normalization and justify why the selection still corresponds to minimizing the theoretical measure (14), or report results using the un-normalized empirical Fisher."
cross_refs: ["objective-extra-term"]
check_script: _audit_code/check_objective_and_split.py
paper_ref: "Algorithm 1 line 10; Appendix E Eq. (72)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: lr-mismatch
category: difference
topic: "hyperparameters"
title: "Paper states lr = 1e-5 but no config sets lr, so default 5e-5 is used"
severity: medium
confidence: high
status: finding
file: hparams_registry.py
line_start: 43
line_end: 47
quote: |
    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
claim: "For DomainNet/OfficeHome the default learning rate is 5e-5, and none of config_json/{vits,res18,res50,res101}.json override `lr` (verified in _audit_code/out/checks.txt), so the runs use 5e-5."
concern: "Paper §5.1 states 'The Adam optimizer is employed with a learning rate of 1e−5'; the released configuration trains at 5e-5, a 5x difference that would change the reported accuracies."
resolution: "Authors: confirm which learning rate produced Table 2 and add an explicit `lr` entry to the configs."
cross_refs: []
check_script: _audit_code/out/checks.txt
paper_ref: "§5.1 Implementation Details"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: single-source-baselines-missing
category: difference
topic: "baselines"
title: "Single-Source-Avg/Best baselines reported in Table 2 are not produced by the driver"
severity: low
confidence: medium
status: question
file: main.py
line_start: 155
line_end: 167
quote: |
        for test in args.otherpoint_list:
            if test == "none":
                args.ckp_frequency = 10
                filtered_nargs_dict = {k: v for k, v in vars(args).items() if k in impor_args_n}
                filtered_args = "_".join([f"{k}v{v}" for k, v in filtered_nargs_dict.items()])
            elif test == "all":
                args.ckp_frequency = 1
                filtered_nargs_dict = {k: v for k, v in vars(args).items() if k in impor_args_a}
                filtered_args = "_".join([f"{k}v{v}" for k, v in filtered_nargs_dict.items()])
            else:
                args.ckp_frequency = 1
                filtered_nargs_dict = {k: v for k, v in vars(args).items() if k in impor_args_m}
                filtered_args = "_".join([f"{k}v{v}" for k, v in filtered_nargs_dict.items()])
claim: "The driver only iterates over otherpoint values m/all/none; there is no mode that trains on a single source domain at a time to produce Single-Source-Avg / Single-Source-Best (Table 2)."
concern: "Two of the Source-Ablation baseline rows in Table 2 have no corresponding code path, so they cannot be reproduced from the released driver as-is."
resolution: "Authors: confirm whether Single-Source results were obtained by running the pipeline once per source (and how), or add the corresponding mode."
cross_refs: ["external-baseline-code-missing"]
paper_ref: "Table 2 Source-Ablation Methods"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

```yaml finding
id: model-selection-on-test
category: methodology
topic: "evaluation / model selection"
title: "Reported accuracy is the best-epoch accuracy on the same 20% holdout used as the test set (no validation split)"
severity: medium
confidence: high
status: finding
file: misc.py
line_start: 21
line_end: 36
quote: |
  def check_epoch(models_acc, patience=5, all_epoch_threshold=40):
      
      max_acc = max(models_acc)  
      max_acc_epoch = models_acc.index(max_acc)  
      current_epoch = len(models_acc) - 1 

      if current_epoch >= all_epoch_threshold:
          return True
      
      if len(models_acc) < patience + 1:
          return False
      
      if current_epoch - max_acc_epoch >= patience:
          return True
      
      return False
claim: "Each epoch, accuracy is measured on `eval_loader` (the 20% per-class holdout built in get_trainset_and_evalset_ratioway), early stopping triggers `patience` epochs after the MAX of that accuracy curve, and the number written to the results CSV / Table 2 is that maximum (paper: 'report the highest accuracies within 5 epoch early stops'). There is no separate validation set."
concern: "Selecting the best epoch by the test-set accuracy and reporting that maximum is model selection on the test set, which positively biases every reported number; with no validation split the early-stopping criterion is computed on the test set itself."
resolution: "Authors: report accuracy at an epoch chosen on a held-out validation set (disjoint from test), or report the final-epoch / fixed-epoch accuracy, to remove the optimistic max-over-epochs selection."
cross_refs: []
check_script: _audit_code/check_objective_and_split.py
paper_ref: "§5.1 ('report the highest accuracies within 5 epoch early stops')"
tags: [leakage:L2.1, reforms:6, whalen:pitfall-2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Pretrained-ckpt path mismatch (high); Static/LoRA/Digits/external-baseline code absent |
| bug         | 1          | low          | `reduce='sum'` string in Fisher loss is a no-op (effect washed out by later normalization) |
| difference  | 4          | high         | Core objective adds `+1/S`; Fisher min-max normalized; lr 1e-5 vs 5e-5; Single-Source mode absent |
| methodology | 1          | medium       | Best-epoch model selection on the test set; no validation split |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **`objective-extra-term`** (difference, high/high): the OTQMS S-selection objective in code is the paper's measure **plus an extra `+1/S`** not in Eq. (14)/(72), changing the selected transfer quantity.
2. **`fisher-minmax-normalization`** (difference, high/high): the Fisher matrix A is built from **min-max-normalized gradients** ([-1,1] per layer), an undocumented transformation that alters the core α*/s* computation.
3. **`pretrained-ckpt-path-mismatch`** (missing, high/high): `main.py` loads source checkpoints from `opti_pre_models_vits_850batch/`, which **no script creates** and which the pretrain script does not write to — reproduction blocked without an undocumented manual step.
4. **`lr-mismatch`** (difference, medium/high): paper says lr = 1e-5 but **no config sets lr**, so runs use the 5e-5 default (5x).
5. **`model-selection-on-test`** (methodology, medium/high): reported accuracy is the **maximum over epochs on the 20% test holdout**, with no validation set — optimistic selection (applied symmetrically to all in-repo methods).
6. **`static-ablation-missing`** (missing, medium/medium): the Static-Under/Exact/Over ablation (Table 3) has **no Static code path** in the repo.

### Items that genuinely look fine
- The per-class train/eval split (`get_trainset_and_evalset_ratioway`) is deterministic (first 80% / last 20% per class by index) and **disjoint** — no train/test sample overlap; preprocessing (ImageNet normalization) uses fixed constants, not data-fitted, so no preprocessing leakage.
- The in-repo baselines AllSources∪Target ("all") and Target-Only ("none") run through the **same pipeline, same split, same selection** as OTQMS, so that subset of the comparison is fair/symmetric.
- Seeding is comprehensive (`random`, `numpy`, `torch`, cuDNN deterministic) in both drivers; the lack of error bars is explicitly acknowledged in the NeurIPS checklist (Q7).
- The QP solver (`solve_quadratic_problem`) correctly encodes the simplex + per-source upper-bound constraints `0 ≤ α_i ≤ N_i/s, Σα_i = 1` matching Appendix E Eq. (74).
- ViT-S backbone (`vit_small_patch16_224.augreg_in21k`, ImageNet-21k) matches the paper's stated backbone.

### Open questions for the authors
- Are the `+1/S` term and the gradient min-max normalization intentional parts of OTQMS (and if so, why are they not in the paper's math)? These two together fully define the released selection rule and are the highest-impact items.
- Which learning rate (1e-5 vs 5e-5) produced the Table 2 numbers?
- How were the Single-Source-Avg/Best, Static-*, LoRA, and Digits results produced, given no code path exists for them in this repo?
