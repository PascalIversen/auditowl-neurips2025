# Audit: ACAVP — Enhancing Visual Prompting through Expanded Transformation Space and Overfitting Mitigation (paper 2758)

## 1. Summary

The repository (`code/s-enmt__ACAVP/`, single shallow commit `cd97d91`, dated 2026-03-11) is
a PyTorch implementation of the proposed visual-prompting method **ACAVP**. It contains a single
training/eval entrypoint (`main.py`), the ACAVP prompter (`models/prompters.py`), a TrivialAugment/
RandAugment augmentation module (`augmentations/strategy_augment.py`), data loaders
(`ILM_Dataloader.py`, `utils.py`), an OpenCLIP wrapper, a results aggregator (`collect_results.py`),
and one config (`configs/ACAVP.yaml`). The paper reports a large empirical study: 12 datasets ×
2 backbones, comparisons against VP/EVP/AutoVP/Coordinator baselines and Linear Probing/Fine-Tuning,
a CIFAR-10-C/CIFAR-100-C robustness study (Table 4), a component ablation (Table 5), an
overfitting-mitigation comparison (Table 6, incl. Dropout/MSE/Weight-decay/IPMix), a
TrivialAugment-on-baselines study (Table 7), an inference-time study (Table 8), and a
hyperparameter-sensitivity study (Fig. 3).

I read every code file and the relevant paper sections (Experiments §4, Appendices C and D). I ran
two read-only deterministic checks under `_audit_code/`: `check_repo_completeness.py` (AST-enumerates
the prompter classes, configs, and dependency-spec files actually present) — outputs in
`_audit_code/out/repo_completeness.csv`. The dominant findings are that the repo ships only the
proposed method (no baseline implementations, no ablation/overfitting-study harness, no robustness
evaluation, no label-mapping code) and has no dependency specification, so most reported numbers
cannot be reproduced from this code. The core ACAVP training/eval path itself is methodologically
sound (proper train/val/test separation, validation-based checkpoint selection, no obvious leakage).

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 — ACAVP per-dataset acc (CLIP ViT-B/32), e.g. CIFAR10 96.27, Flowers 89.36, Avg 83.18 | `main.py` + `configs/ACAVP.yaml` (ACAVP only) | code runs ACAVP only | n/a | PARTIAL (only ACAVP path present; values not regenerable without data/weights) |
| Table 2 — VP / EVP / AutoVP columns | (none) | — | — | MISSING (no VP/EVP/AutoVP prompter or config) |
| Table 2 — ZS / LP / FT columns | (none; LP/FT cited from prior work) | — | — | MISSING for ZS; LP/FT explicitly cited from other papers |
| Table 3 — ResNet50 results, incl. † label-mapping columns | `main.py` (`--evp_output_mapping`) | partial | — | PARTIAL: EVP output-mapping present (`utils.get_index`); "random label mapping" (App. D.2) MISSING; baselines MISSING |
| Table 4 — CIFAR10-C / CIFAR100-C robustness | (none) | — | — | MISSING (no corruption dataset / eval code) |
| Table 5 — component ablation (Affine / Color / Affine+Color / Resize+Additive / …) | (none) | — | — | MISSING (only full ACAVP prompter; no ablation switches) |
| Table 6 — overfitting mitigation (Dropout / MSE / Weight decay / TrivialAugment / RandAugment / IPMix) | partial: `weight_decay` arg, TrivialAugment/RandAugment in `augmentations/` | partial | — | PARTIAL: Dropout, MSE-loss, IPMix MISSING |
| Table 7 — TrivialAugment on VP/EVP/AutoVP baselines | (none) | — | — | MISSING (baselines + their TrivialAugment variants absent) |
| Table 8 — inference-time comparison incl. Coordinator | (none) | — | — | MISSING (timing harness + Coordinator absent; eval logs times but no script aggregates Table 8) |
| Fig. 3 — hyperparameter sensitivity (Rθ, Rsh, Rt, Rσ) | `configs/ACAVP.yaml` exposes the knobs; no sweep script | — | — | MISSING sweep harness (knobs settable, but no script reproduces Fig. 3) |
| Table 1 — preliminary padding vs image-sized noise overfitting | (none) | — | — | MISSING (no padding-only / image-sized-noise prompter) |
| Mean ± standard-error aggregation across 3 runs | `collect_results.py:102-108` | mean, std/√n | ✓ (formula correct) | Verified (aggregation logic only; inputs not present) |

## 3. Findings

## missing

```yaml finding
id: baselines-not-implemented
category: missing
topic: "baselines / result traceability"
title: "Repo implements only ACAVP; VP, EVP, AutoVP, Coordinator baselines absent"
severity: high
confidence: high
status: finding
file: models/prompters.py
line_start: 14
line_end: 15
quote: |
  class ACAVP(nn.Module):
      def __init__(self, args):
claim: "models/prompters.py defines exactly one prompter class, ACAVP; main.py instantiates the prompter via prompters.__dict__[args.method] and only configs/ACAVP.yaml exists, so the VP, EVP, AutoVP, and Coordinator baselines reported throughout Tables 2-4, 7, and 8 have no implementation or config in the repo."
concern: "Every headline comparison ('state-of-the-art accuracy among VP methods', 'surpasses linear probing', Tables 2/3/4/7/8) depends on baseline numbers that no code in this repo can produce, so the central claims are not reproducible from the artefact."
resolution: "Authors: provide the VP/EVP/AutoVP/Coordinator prompter implementations and their configs, or state where the baseline numbers were computed."
cross_refs: ["only-acavp-method-crash", "missing-ablation-harness", "missing-robustness-eval"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Tables 2, 3, 4, 7, 8; Appendix D.3"
tags: [reforms:8, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-ablation-harness
category: missing
topic: "ablations"
title: "Table 5 ablation variants (Affine, Color, Affine+Color, Resize+Additive…) not implementable"
severity: high
confidence: high
status: finding
file: models/prompters.py
line_start: 83
line_end: 119
quote: |
  def forward(self, x):
        # denrom 
        x = self.denormalize(x) # => 0 ~ 1

        # Affine VP
        angle, s, sh, t = self.limit_transform()
        self.angle = angle.expand(x.shape[0])
        self.s = s.expand(x.shape[0], 2)
        self.sh = sh.expand(x.shape[0], -1)
        self.t = t.expand(x.shape[0], -1)
        center = self.center.expand(x.shape[0], -1).to(x.device)
        affine_matrix = get_affine_matrix2d(
            self.t, 
            center, 
            self.s, 
            -self.angle, 
            sx=self.sh[..., 0], 
            sy=self.sh[..., 1],
            )
        x = kornia.geometry.transform.affine(x, 
                   affine_matrix[..., :2, :3], 
                   self.affine_mode, 
                   self.affine_padding_mode, 
                   self.affine_align_corners,
                   )   

        # Multiplicative VP
        self.multiplicative_prompt = self._multiplicative_prompt.repeat(x.size(0), 1, 1, 1)
        multiplicative_prompt = self.sigmoid(self.multiplicative_prompt) * self.sigma_range
        x = x * multiplicative_prompt    

        # Padding VP
        self.mask = torch.all(x==0, dim=(0,1)).long()
        self.padding_prompt = self.prompt.repeat(x.size(0), 1, 1, 1)
        padding_prompt = self.mask.to(x.device) * self.padding_prompt.to(x.device)
        x = x + padding_prompt
        x = self.normalize(x)

        return x
claim: "ACAVP.forward always applies affine + multiplicative(color) + additive(padding) transforms together; there are no flags or config keys to disable individual components, and no separate prompter classes for the Table 5 variants (Affine only, Color only, Affine+Color, Resize+Additive+Color, Affine+Additive)."
concern: "The ablation in Table 5 that 'validates our design choices' cannot be reproduced because the code provides no way to run any component subset."
resolution: "Authors: provide the component-toggling flags or the prompter variants used to produce Table 5."
cross_refs: ["baselines-not-implemented"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 5, §4.3"
tags: [reforms:8]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-robustness-eval
category: missing
topic: "result traceability"
title: "CIFAR-10-C / CIFAR-100-C robustness evaluation (Table 4) absent from code"
severity: medium
confidence: high
status: finding
file: main.py
line_start: 281
line_end: 354
quote: |
    if args.dataset == "cifar10":
        train_dataset = CIFAR10(args.dataset_root, transform=train_preprocess,
                                    download=True, train=True)
        val_dataset = CIFAR10(args.dataset_root, transform=preprocess,
                                    download=True, train=True)
        test_dataset = CIFAR10(args.dataset_root, transform=preprocess,
                            download=True, train=False)
claim: "The dataset dispatch in main.py handles cifar10, cifar100, the CoOp LMDB datasets, svhn, gtsrb, and clevr only; there is no branch, loader, or evaluation path for the corrupted CIFAR-10-C / CIFAR-100-C benchmarks reported in Table 4."
concern: "The Table 4 robustness numbers (a headline 'superior robustness to distribution shifts' claim) cannot be regenerated because no corruption dataset loader or evaluation script exists in the repo."
resolution: "Authors: add the CIFAR-10-C/CIFAR-100-C loading and evaluation script used for Table 4."
cross_refs: ["baselines-not-implemented"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 4, §4.2"
tags: [reforms:8]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-overfit-and-labelmap-code
category: missing
topic: "result traceability"
title: "Overfitting-mitigation variants (Dropout/MSE/IPMix) and random label mapping not in repo"
severity: medium
confidence: high
status: finding
file: main.py
line_start: 442
line_end: 451
quote: |
    else:      
        if args.evp_output_mapping:
            class_indices = get_index(
                train_loader, 
                model, 
                len(class_names), 
                device)
        else:
            class_indices = list(range(len(class_names)))
claim: "For the ResNet path the only label-mapping options are EVP's output mapping (utils.get_index) and identity; the 'random label mapping' strategy described in Appendix D.2 is not implemented. Table 6's Dropout, MSE-loss, and IPMix overfitting techniques also have no code path (only weight_decay arg and TrivialAugment/RandAugment exist)."
concern: "Table 3's †/non-† comparison relies on a random-label-mapping baseline absent from the code, and the Table 6 overfitting-technique comparison (motivating the paper's second contribution) is only partially implementable."
resolution: "Authors: provide the random-label-mapping implementation and the Dropout/MSE/IPMix variants used for Tables 3 and 6."
cross_refs: ["baselines-not-implemented"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 3, Table 6, Appendix D.2"
tags: [reforms:8]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-requirements
category: missing
topic: "dependencies / reproducibility"
title: "No dependency specification; many non-trivial imports unpinned"
severity: medium
confidence: high
status: finding
file: main.py
line_start: 26
line_end: 33
quote: |
  import clip
  import open_clip
  from models.openclip_wrapper import OpenCLIPWrapper
  from torch.amp import GradScaler, autocast
  from models import prompters
  from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint, refine_classname, convert_models_to_fp32, get_index, CLEVRCountingDataset
  from augmentations.strategy_augment import TrivialAugmentDataset, RandAugmentDataset
  from ILM_Dataloader import COOPLMDBDataset
claim: "The repo contains no requirements.txt, environment.yml, pyproject.toml, setup.py, Pipfile, or conda file (verified by _audit_code/check_repo_completeness.py), yet it imports clip, open_clip, kornia, lmdb, six, wandb, sklearn, matplotlib, and (for some backbones) big_transfer and torch.hub models without version pins."
concern: "The environment cannot be reliably rebuilt; APIs such as torch.amp.autocast, open_clip.create_model_and_transforms, and kornia.get_affine_matrix2d are version-sensitive, so reproduction is not guaranteed."
resolution: "Authors: add a requirements/environment file pinning torch, torchvision, kornia, open_clip, clip, lmdb, and other dependencies."
cross_refs: []
check_script: _audit_code/check_repo_completeness.py
paper_ref: "n/a (repository completeness)"
tags: [reforms:8, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: only-acavp-method-crash
category: bug
topic: "CLI / method dispatch"
title: "--method other than ACAVP raises KeyError (no other prompter registered)"
severity: medium
confidence: high
status: finding
file: main.py
line_start: 224
line_end: 224
quote: |
    prompter = prompters.__dict__[args.method](args).to(device)
claim: "The prompter is selected by name from prompters.__dict__, but models/prompters.py defines only the ACAVP class (verified via AST in _audit_code/check_repo_completeness.py); any --method VP/EVP/AutoVP/Coordinator therefore raises KeyError, and parse_option additionally requires a matching configs/<method>.yaml which exists only for ACAVP."
concern: "The single advertised entrypoint can only run the proposed method; the --method flag implies baseline support that does not exist, so a reviewer cannot reproduce the comparison rows by varying --method."
resolution: "Authors: register the baseline prompter classes and add their configs, or remove the implication that --method supports baselines."
cross_refs: ["baselines-not-implemented"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "README 'Running Experiments'; Tables 2-4"
tags: [reforms:8]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: dataset-list-stanfordcars
category: difference
topic: "datasets"
title: "Code supports stanfordcars dataset not in paper's 12-dataset list"
severity: low
confidence: medium
status: finding
file: main.py
line_start: 295
line_end: 295
quote: |
    elif args.dataset in ["dtd", "oxfordpets", "food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
claim: "The dataset dispatch accepts 'stanfordcars', but the paper's enumerated benchmark (Appendix D.1, Table 9) lists 12 datasets that do not include Stanford Cars."
concern: "Minor faithfulness gap: a dataset path exists in code that is not part of the reported benchmark; benign on its own but indicates the code may not exactly match the reported experiment set."
resolution: "Authors: confirm Stanford Cars was not used in the reported tables, or clarify its role."
cross_refs: []
paper_ref: "Appendix D.1, Table 9"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The implemented ACAVP training/evaluation procedure is sound: train/val/test
are kept separate, the held-out validation set is used only for checkpoint selection (`is_best = val_acc
> best_val_acc`, `main.py:513`), the final number is computed on the test set after reloading the best
checkpoint (`main.py:553-560`), and the 10% validation carve-out for non-CoOp datasets uses
`train_test_split(..., random_state=args.seed)` over the *training* split only (`main.py:357-372`), so no
test data leaks into validation or training. See "Items that genuinely look fine" below.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                              |
|-------------|------------|--------------|--------------------------------------------------------------|
| missing     | 5          | high         | Only ACAVP shipped: baselines, ablation, robustness, label-mapping, and deps all absent. |
| bug         | 1          | medium       | `--method` other than ACAVP raises KeyError.                 |
| difference  | 1          | low          | Code supports `stanfordcars`, not in paper's 12-dataset set. |
| methodology | 0          | -            | Implemented ACAVP train/val/test path is sound; no leakage found. |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing] `baselines-not-implemented`** — Only the ACAVP prompter/config exists; VP/EVP/AutoVP/Coordinator (the entire comparison basis for Tables 2-4, 7, 8) are absent. (high/high)
2. **[missing] `missing-ablation-harness`** — Table 5 component ablation cannot be run; ACAVP.forward has no component toggles and no variant classes. (high/high)
3. **[missing] `missing-robustness-eval`** — Table 4 CIFAR-10-C/CIFAR-100-C robustness evaluation has no dataset loader or eval script. (medium/high)
4. **[missing] `missing-overfit-and-labelmap-code`** — Random label mapping (Table 3 †) and Dropout/MSE/IPMix (Table 6) are not implemented. (medium/high)
5. **[missing] `no-requirements`** — No dependency specification for version-sensitive deps (kornia, open_clip, torch.amp). (medium/high)
6. **[bug] `only-acavp-method-crash`** — `--method VP/EVP/...` raises KeyError; advertised baseline support does not exist. (medium/high)

### Items that genuinely look fine
- **Train/val/test separation and checkpoint selection**: validation set drives `is_best`, final metric is computed on the test loader after reloading `model_best.pth.tar` (`main.py:507-560`). No test-set tuning.
- **Validation carve-out has no leakage**: 10% validation is split from the *training* set only via `train_test_split(random_state=args.seed)`; CoOp datasets use predefined val splits (`main.py:356-372`, `ILM_Dataloader.py`).
- **Mean ± standard-error aggregation is correct**: `collect_results.py:102-108` computes `np.mean` and `np.std/√n`, matching the paper's "average accuracy and standard error".
- **Seeding of the split is reproducible** across the three reported runs because `train_test_split` uses `random_state=args.seed` (each run uses a distinct `--seed`).

### Open questions for the authors
- Where were the VP/EVP/AutoVP/Coordinator baseline numbers and the Table 5 ablation produced — in a separate (unreleased) branch, or off-repo? (relates to `baselines-not-implemented`, `missing-ablation-harness`)
- Was Stanford Cars used in any reported result, given the code path exists but the dataset is not in Table 9? (`dataset-list-stanfordcars`)
