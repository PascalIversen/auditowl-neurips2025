# Audit — Paper #1171: "Evolving and Regularizing Meta-Environment Learner for Fine-Grained Few-Shot Class-Incremental Learning" (MEL)

## 1. Summary

The repo `code/Legenddddd__MEL/` is the authors' official implementation
(github.com/Legenddddd/MEL; README BibTeX and title match this paper). It is a
small, single-entrypoint PyTorch codebase: `train.py` drives a `FSCILTrainer`
(`models/mel/fscil_trainer.py`) over a ResNet-12 backbone
(`models/resnet12_encoder.py`) with the two method components — the meta-category
vector ("Evolving", `self.attention` updated by attention in
`fscil_trainer.py:134-145`, matching Eq. 4/7-9) and the dual-branch
mutual-KL regularizer ("Regularizing", `base_train` in `helper.py:8-53`,
matching Eq. 10-14). I read the paper (PDF + `paper_text.txt`), mapped every
table/figure/headline number to code, and ran three read-only deterministic
checks under `_audit_code/`:

- `check_dataset_modules.py` → `out/dataset_modules.txt`: which of the four
  Table-1 dataset loader modules referenced by `data_utils.py` actually exist.
- `check_repro_artifacts.py` → `out/repro_artifacts.txt`: dependency spec,
  pretrained weights, README run commands, unlisted deps.
- `check_lmutual_temperature.py` → `out/lmutual_temp.txt`: the temperature/scale
  gap between paper Eq. 14 and the code's `DistillKL(T=4.0)`.

The core method matches the equations and uses standard, published FSCIL splits
(seed fixed to 1, following CEC/PFR). The dominant problem is **reproducibility
completeness**: only 1 of the 4 headline datasets can actually be run, the
ablation and the auxiliary (conventional-benchmark / alternate-backbone)
experiments have no code, and there is no dependency specification or pretrained
checkpoint.

## 2. Traceability table

Verification here is **static** (no GPU / datasets available): "code present"
means the computation that would produce the number exists and matches the
described method; values themselves were not re-run.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 / Table 4 — Stanford Dogs "Ours" (AccT 44.06, AccAvg 52.38) | `train.py` + `dataloader/StanfordDog/StanfordDog.py` + `models/mel/*` | not re-run | n/a (code present) | Present, runnable |
| Table 1 / Table 3 — CUB200 "Ours" (36.00 / 45.44) | loader `dataloader/cub200/cub200.py` **absent** | — | — | MISSING (loader module) |
| Table 1 / Table 5 — Stanford Cars "Ours" (57.58 / 68.43) | loader `dataloader/StanfordCar/StanfordCar.py` **absent** | — | — | MISSING (loader module) |
| Table 1 / Table 6 — FGVCAircraft "Ours" (40.72 / 54.17) | loader `dataloader/Aircraft/Aircraft.py` **absent** | — | — | MISSING (loader module) |
| Figure 3 — per-session accuracies (4 datasets) | as above (3/4 loaders absent) | — | partial | MISSING for 3/4 datasets |
| Table 2 — ablation M0–M5 (Stanford Dogs, 8 rows) | no component toggles in `train.py` args; `both_mlp`, meta-vector, incr. opt. always on | — | — | MISSING (no ablation harness) |
| Figure 4 — meta-category opt. strategies (Avg/Avg+Sim/Avg+Weight/BP/Ours) | only "Ours" rule in `fscil_trainer.py:137-145`; other 4 strategies absent | — | — | MISSING (no variants) |
| Figure 5/6/7 — γ, β, Lmutual direction, N, τ sweeps | `complex_weight`,`part_weight`,`part_num`,`temperature` args exist; Lmutual-direction toggle absent | — | partial | Partly present (no driver/sweep) |
| Figure 8 — miniImageNet / CIFAR100 conventional FSCIL | no mini/CIFAR loader or dataset key | — | — | MISSING |
| Figure 9 — ResNet-18/50, ViT-B/16(+adaptor), C-FeCAM/C-RanPAC | only `ResNet12()`; no alt-backbone code | — | — | MISSING |
| Eq. 11/14 — Lmutual definition | `helper.py:29,33` via `DistillKL(T=4.0)` (`fscil_trainer.py:8-19,59`) | extra temp T=4, ×T² scale | ✗ (formula differs) | MISMATCH (difference) |
| Theorem 3.1 / proof (Appendix A.1) | analytical; no code expected | — | — | N/A (theory) |
| Error bars / significance | none (seed=1, single run) | — | — | Disclosed (checklist Q7 = No) |

## 3. Findings

## missing

```yaml finding
id: missing-dataset-loaders
category: missing
topic: "result traceability / data loading"
title: "3 of 4 Table-1 dataset loaders (CUB200, Stanford Cars, FGVCAircraft) absent from repo"
severity: high
confidence: high
status: finding
file: dataloader/data_utils.py
line_start: 5
line_end: 38
quote: |
  def set_up_datasets(args):
      if args.dataset == 'cub200':
          import dataloader.cub200.cub200 as Dataset
  ...
      if args.dataset == 'StanfordCar':
          import dataloader.StanfordCar.StanfordCar as Dataset
  ...
      if args.dataset == 'Aircraft':
          import dataloader.Aircraft.Aircraft as Dataset
claim: "set_up_datasets imports dataloader.cub200.cub200, dataloader.StanfordCar.StanfordCar and dataloader.Aircraft.Aircraft, but only dataloader/StanfordDog/StanfordDog.py exists; the cub200/, StanfordCar/, Aircraft/ packages contain no loader module."
concern: "Three of the four datasets in the headline Table 1 (CUB200, Stanford Cars, FGVCAircraft) cannot be loaded — running `train.py -dataset cub200` (etc.) raises ModuleNotFoundError at set_up_datasets, so 3/4 of the main results and Figure 3 panels are not reproducible from the released code."
resolution: "Authors: please add the missing dataloader modules dataloader/cub200/cub200.py, dataloader/StanfordCar/StanfordCar.py, dataloader/Aircraft/Aircraft.py (the split CSVs and index_list/ files for these datasets are present, only the .py loaders are missing)."
cross_refs: []
check_script: _audit_code/check_dataset_modules.py
paper_ref: "Table 1; Appendix A.4 Tables 3,5,6; Figure 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-ablation-harness
category: missing
topic: "ablations"
title: "Ablation Table 2 (M0–M5) cannot be reproduced — no component toggles in code"
severity: medium
confidence: high
status: finding
file: train.py
line_start: 6
line_end: 57
quote: |
  def get_command_line_parser():
      parser = argparse.ArgumentParser()
      # about dataset and network
      parser.add_argument('-project', type=str, default='mel')
      parser.add_argument('-dataset', type=str, default='StanfordDog')
claim: "The released code implements only the full method. There is no CLI flag or code path to (a) replace the meta-category vector so that F_hat = F_high (ablation M1), (b) disable incremental meta-category optimization (M3, fscil_trainer.py:134-145 always runs), or (c) disable feature-space transformation (M4; both_mlp/both_mlp1 in Network.py:105-106 are always applied)."
concern: "The 8-row ablation Table 2 (M0–M5 on Stanford Dogs), which is the paper's evidence that each component contributes, cannot be regenerated from the repo without modifying the source — only the M5/full configuration is runnable."
resolution: "Authors: please release the ablation switches (or the per-row configs/scripts) used to produce Table 2 rows M0–M5."
cross_refs: []
paper_ref: "Table 2 (Section 4.3)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-aux-experiments
category: missing
topic: "result traceability / auxiliary experiments"
title: "Figs 4, 8, 9 experiments (opt-strategy variants, conventional benchmarks, alt backbones) absent"
severity: medium
confidence: high
status: finding
file: dataloader/data_utils.py
line_start: 5
line_end: 41
quote: |
  def set_up_datasets(args):
      if args.dataset == 'cub200':
          import dataloader.cub200.cub200 as Dataset
          args.base_class = 100
          args.num_classes = 200
claim: "The repo supports only four fine-grained dataset keys (cub200, StanfordDog, StanfordCar, Aircraft) and only the ResNet12 backbone; there is no miniImageNet/CIFAR100 loader (Fig. 8), no ResNet-18/50/ViT-B-16 or adaptor backbone and no C-FeCAM/C-RanPAC code (Fig. 9), and only the 'Ours' meta-category update rule — not the Avg / Avg+Sim / Avg+Weight / BP variants of Fig. 4."
concern: "The auxiliary results that support generality claims (conventional FSCIL benchmarks Fig. 8; cross-backbone robustness Fig. 9; the meta-category optimization-strategy comparison Fig. 4) have no code in the repo and cannot be reproduced."
resolution: "Authors: please add the conventional-benchmark loaders, the alternate-backbone configurations, and the meta-category optimization-strategy variants used for Figures 4, 8 and 9 (or state which are out of scope for the release)."
cross_refs: ["missing-dataset-loaders"]
check_script: _audit_code/check_dataset_modules.py
paper_ref: "Figures 4, 8, 9"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec
category: missing
topic: "code completeness / dependencies"
title: "No requirements/env file; unlisted deps (torch_dct, timm); no pretrained weights or run command for 3/4 datasets"
severity: medium
confidence: high
status: finding
file: models/mel/helper.py
line_start: 1
line_end: 5
quote: |
  from utils import *
  from tqdm import tqdm
  import torch.nn.functional as F
  import torch_dct as dct
claim: "There is no requirements.txt / environment.yml / setup.py / pyproject.toml anywhere in the repo; helper.py imports torch_dct (a non-standard package, and `dct` is imported but never used) and Network.py imports timm — neither is pinned or listed. No pretrained base checkpoint is shipped (model_dir defaults to None, so the 400-epoch base session must be retrained), and the README gives a single StanfordDog-style command with no results table."
concern: "The environment cannot be reconstructed deterministically (unpinned, partly unlisted dependencies), and with no pretrained weights and a single example command the reported numbers require a full retrain whose exact environment is unspecified — a code-completeness gap relative to the checklist's 'Yes' on open access with instructions."
resolution: "Authors: please add a pinned dependency specification (including torch_dct and timm versions, or remove the unused torch_dct import), and either ship base-session checkpoints or document the exact per-dataset commands and environment."
cross_refs: []
check_script: _audit_code/check_repro_artifacts.py
paper_ref: "Checklist Q5 (Open access to data and code); Appendix A.3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No standalone runtime bug owns its own finding. The missing-loader crash is owned
by `missing-dataset-loaders` (routing step 1: the artefact is absent, not present-
but-broken). The `import torch_dct as dct` that is never used and the unused
`from timm.models import create_model` are noted under `no-dependency-spec`.

## difference

```yaml finding
id: lmutual-distill-temperature
category: difference
topic: "loss function / regularization"
title: "L_mutual implemented as DistillKL(T=4.0) — extra softening temperature and ×T² scaling not in Eq. 14"
severity: low
confidence: high
status: finding
file: models/mel/fscil_trainer.py
line_start: 8
line_end: 19
quote: |
  class DistillKL(nn.Module):
      """KL divergence for distillation"""

      def __init__(self, T):
          super(DistillKL, self).__init__()
          self.T = T

      def forward(self, y_s, y_t):
          p_s = F.log_softmax(y_s / self.T, dim=1)
          p_t = F.softmax(y_t / self.T, dim=1)
          loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
          return loss
claim: "L_mutual (Eq. 14) is computed by criterion_Cross = DistillKL(T=4.0) (fscil_trainer.py:59) applied to logits that already carry the tau=16 cosine scaling. DistillKL re-divides the logits by T=4 before softmax (effective sharpness exp(4*S) instead of the paper's exp(16*S)) and multiplies the class-summed KL by T**2 = 16; paper Eq. 14 specifies a plain bidirectional softmax-KL at the full tau=16 sharpness with no T**2 factor."
concern: "The implemented regularizer softens the two distributions by a factor of 4 and scales the term by 16 relative to the literal Eq. 11/14 (effective weight gamma*T**2 = 8x vs gamma = 0.5), so the paper's stated loss does not match the code; both are valid KL formulations so this is a description-vs-implementation difference, not a bug."
resolution: "Authors: please update Eq. 11/14 (or the code) so the distillation temperature T=4.0 and the T**2 scaling are reflected, and clarify the intended weight of the mutual term."
cross_refs: []
check_script: _audit_code/check_lmutual_temperature.py
paper_ref: "Eq. 11 and Eq. 14 (Section 3.3.2)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding. The evaluation uses the established FG-FSCIL protocol
with fixed, published class splits (`data/index_list/*`, from PFR), inference on
all seen classes, prototype classifiers, and a fixed seed — there is no train/test
leakage, no test-set tuning (the "max_acc" best-epoch selection at
`fscil_trainer.py:74` is on the base test set but the comparison is symmetric
across all methods under the same harness, and is the field-standard PFR
protocol), and the metric (per-session top-1 accuracy) fits the task. The
single-seed / no-error-bar choice is explicitly disclosed in checklist Q7 and
follows CEC/PFR convention, so it is recorded as an acknowledged limitation, not
a finding.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                            |
|-------------|------------|--------------|------------------------------------------------------------|
| missing     | 4          | high         | 3/4 main-dataset loaders absent; ablation + aux exps + deps |
| bug         | 0          | -            | crash from missing loaders is owned by `missing-dataset-loaders` |
| difference  | 1          | low          | L_mutual uses DistillKL(T=4.0), not the plain Eq. 14 KL     |
| methodology | 0          | -            | standard FG-FSCIL protocol, fixed published splits, no leakage |

## 5. Closing lists

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing]** `missing-dataset-loaders` — only Stanford Dogs runs; CUB200,
   Stanford Cars, FGVCAircraft loader modules are absent, so 3/4 of Table 1 /
   Figure 3 cannot be reproduced from the repo (high / high).
2. **[missing]** `missing-ablation-harness` — the M0–M5 ablation (Table 2) has no
   toggles; only the full method is runnable (medium / high).
3. **[missing]** `missing-aux-experiments` — Figs 4, 8, 9 (opt-strategy variants,
   conventional benchmarks, alternate backbones) have no code (medium / high).
4. **[missing]** `no-dependency-spec` — no requirements/env file, unlisted/unused
   deps (torch_dct, timm), no pretrained checkpoint, single example command
   (medium / high).
5. **[difference]** `lmutual-distill-temperature` — implemented L_mutual uses an
   extra distillation temperature T=4 and ×T² scaling absent from Eq. 11/14
   (low / high).

### Items that genuinely look fine
- The two method components match the equations: meta-category vector
  `F_high ⊙ leaky_relu(m)` (Network.py:95-97 ↔ Eq. 4) and its incremental
  attention update with mixing weight `way/test_class` (fscil_trainer.py:137-145
  ↔ Eq. 7-9), and the dual-branch CE + bidirectional KL base loss
  (helper.py:29-33 ↔ Eq. 11-14).
- Hyperparameters match the paper: `part_weight`=β=0.5, `complex_weight`=γ=0.5,
  `part_num`=N=4, `temperature`=τ=16, `seed`=1 (train.py:12,49,52-54).
  (The README's `-gamma 0.1` is the LR-scheduler decay, unused under the
  `Cosine` schedule — not the loss γ; no mismatch.)
- Inference uses only high-level features (`test()` reports `acc1` from `logits1`
  only, helper.py:121-135 ↔ Eq. 15), consistent with the headline M5/`S_high`
  ablation row.
- Splits are fixed published index lists (`data/index_list/*`), seeding is
  comprehensive when seed≠0 (utils.py:88-99); no leakage or test-set tuning.
- Dataset configuration (base/incremental class counts, ways, sessions) in
  `set_up_datasets` matches Appendix A.3 (e.g. Stanford Dogs 80 base + 8×5-way).

### Open questions for the authors
- Were the released numbers produced from this exact repo state on all four
  datasets, given that 3/4 loaders are not present? (Confirms whether the omission
  is an upload oversight vs the numbers coming from a fuller internal tree.)
- What is the intended weight/temperature of `L_mutual` — the literal Eq. 14
  (γ=0.5, τ=16) or the implemented DistillKL(T=4.0)×T²? (`lmutual-distill-temperature`.)
