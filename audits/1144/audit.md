# Audit — 1144 "Graph Your Own Prompt" (GCR / Graph Consistency Layer)

## 1. Summary

The repo (`code/Darcyddx__graph-prompt/`, single commit `afb4577`, no release
tag) is a fork of the `pytorch-cifar100` training harness extended with the
paper's Graph Consistency Layer (`gcr.py`), a class-mask indicator
(`indicator.py`), per-layer loss weighting (`computeweight.py`), and a
multi-dataset `train.py`/`eval.py`. It supports CIFAR-10, CIFAR-100 and Tiny
ImageNet for a fixed set of CNN/transformer backbones under `models/`.

What I did: read every top-level script and the GCR-touching model
(`models/mobilenet.py`); mapped the loss code in `train.py` against Eqs. 1–7 of
the paper (the masked prediction graph `P = M ⊙ S` and the Frobenius alignment
loss are faithfully implemented); checked the dependency/setup story; and ran
two read-only checks under `_audit_code/` (AST import scan + direct invocation of
every weighting scheme). The most consequential findings: three of the seven
weighting schemes the paper evaluates crash on import (`math` never imported),
the ImageNet-1K / transformer experiments (Table 4) have no code path in the
repo, no script aggregates the reported per-config / mean±std-over-10-runs
tables, and "best accuracy" is selected by monitoring the test set directly
(no validation split).

## 2. Traceability table

Reported numbers come from training-run logs; no script in the repo aggregates
table cells, computes mean/std over the 10 (or 3) runs, or renders the figures.
"Repo location" below points to the code that *would compute* a single run's
accuracy, where one exists.

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — CIFAR-10 acc per (model × E/M/L/…/Full × weighting) | `train.py` (per-run) + `eval.py` | not run | — | PARTIAL: single-run acc reproducible for CNNs; no aggregation / mean±std harness |
| Table 1 mean acc 94.07% (Late) vs 93.32% baseline | (no aggregation script) | — | — | MISSING (mean over 10 runs not computed by any script) |
| Table 2 — CIFAR-100 acc; Late 74.74% vs 72.95% | `train.py` + `eval.py` | not run | — | PARTIAL (single-run only; see note) |
| Table 3 — Tiny ImageNet acc | `train.py` (tiny_imagenet) + `eval.py` | not run | — | PARTIAL (single-run only) |
| Table 4 — ImageNet-1K, iFormer / ViT / ViG | (none) | — | — | MISSING (no ImageNet-1K loader, no iFormer/ViG/MAE model, no AdamW/AMP/cosine path) |
| Fig. 6 ∆gain heatmaps over 7 weighting schemes | `computeweight.py` + `train.py` | — | — | MISSING for `sqrt`/`arccos`/`cosine` schemes (crash, see `computeweight-missing-math-import`); no heatmap script |
| Fig. 1/3/4/5 graph & feature-map visualizations | `gcr.py` (graph); `tsne.py` (t-SNE only) | — | — | MISSING (no script renders Fig. 3/4/5 feature/graph visualizations; `tsne.py` is unrelated CIFAR-10 t-SNE) |
| 98.1% → 99.8% (Kaggle cats vs dogs, Fig. 3/4) | (none) | — | — | MISSING (no cats-vs-dogs dataset path or training config in repo) |

## 3. Findings

## missing

```yaml finding
id: imagenet1k-transformer-experiments-absent
category: missing
topic: "result traceability"
title: "ImageNet-1K / iFormer / ViG experiments (Table 4) have no code path"
severity: high
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/train.py
line_start: 335
line_end: 363
quote: |
      parser.add_argument('-dataset', type=str, default='cifar100', 
                        choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                        help='dataset to use for training')
      parser.add_argument('-net', type=str, default="mobilenet", help='net type')
claim: "train.py only accepts cifar10/cifar100/tiny_imagenet and uses SGD+MultiStepLR; the repo has no ImageNet-1K data loader, no iFormer/ViG/MAE model files, and no AdamW/AMP/cosine-annealing/grad-clip training path that the paper specifies for transformers."
concern: "Table 4 (ImageNet-1K with iFormer-S/B, ViT, ViG; the headline +1.4% iFormer gain) and the transformer training regime described in §4.1 cannot be reproduced from the released code."
resolution: "Authors: please release the ImageNet-1K loader, the iFormer/ViG/MAE model definitions, and the AdamW/AMP/cosine-annealing training script used for Table 4."
cross_refs: ["transformer-training-config-absent"]
check_script: null
paper_ref: "Table 4; §4.1 lines 511-517"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: transformer-training-config-absent
category: missing
topic: "evaluation consistency"
title: "AdamW/AMP/cosine-annealing transformer training regime not in repo"
severity: medium
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/train.py
line_start: 417
line_end: 422
quote: |
      loss_function = nn.CrossEntropyLoss()
      optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
      train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  
      iter_per_epoch = len(training_loader)
      warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
claim: "train.py hardcodes SGD + MultiStepLR (no AMP, no gradient clipping, no AdamW, no cosine annealing) for all backbones, including the ViT/Swin/MobileViT transformers it can instantiate."
concern: "The paper states transformers use 'AdamW, lr 1e-4, cosine annealing, weight decay 5e-2, batch size 256, AMP, 10-epoch warm-up, gradient clipping' (§4.1); none of that is in the released training loop, so the transformer rows of Tables 1-3 are not reproducible as described."
resolution: "Authors: provide the transformer-specific optimizer/schedule/AMP configuration, or confirm transformers were also trained with the SGD/MultiStepLR loop in the repo."
cross_refs: ["imagenet1k-transformer-experiments-absent"]
paper_ref: "§4.1 lines 511-513"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-multirun-aggregation-harness
category: missing
topic: "result traceability"
title: "No script aggregates table cells or computes mean±std over 10 runs"
severity: medium
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/train.py
line_start: 507
line_end: 534
quote: |
      for epoch in range(1, settings.EPOCH + 1):
          if epoch > args.warm:
              train_scheduler.step(epoch)

          if args.resume:
              if epoch <= resume_epoch:
                  continue

          train(epoch)
          acc = eval_training(epoch)

          # Save best performance model immediately when improvement is detected
          if best_acc < acc:
              prev_best = best_acc
              best_acc = acc
              best_epoch = epoch

              logger.info(f'Saving best weights file to {best_model_path}')
              print(f'Saving best weights file to {best_model_path}')
              torch.save(net.state_dict(), best_model_path)

              logger.info(f'Best accuracy updated: {prev_best:.6f} -> {best_acc:.6f} at epoch {best_epoch}')
claim: "train.py runs a single training run and logs one best accuracy; the repo contains no harness that sweeps the 7 weighting schemes × stage configs × backbones, nor any code that computes the mean/std over the '10 runs (CIFAR) / 3 runs (ImageNet-1K)' reported in Tables 1-4 and Fig. 6."
concern: "Every headline number is a mean±std over repeated runs across a config grid, but no script in the repo produces those aggregates or the per-config grids, so the reported tables cannot be regenerated end-to-end."
resolution: "Authors: provide the sweep driver and the aggregation script (mean/std over seeds) that produced Tables 1-4 and Fig. 6, including the seeds used."
cross_refs: []
paper_ref: "Tables 1-4; Fig. 6; §4.1 lines 516-517"
tags: [reforms:2, forensics:hidden-iteration]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: catsvsdogs-and-visualization-scripts-absent
category: missing
topic: "result traceability"
title: "Cats-vs-dogs results and Fig. 3/4/5 visualization scripts not in repo"
severity: low
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/settings.py
line_start: 24
line_end: 41
quote: |
  DATASET_CONFIGS = {
      'cifar10': {
          'num_classes': CIFAR10_NUM_CLASSES,
          'mean': CIFAR10_TRAIN_MEAN,
          'std': CIFAR10_TRAIN_STD
      },
      'cifar100': {
          'num_classes': CIFAR100_NUM_CLASSES,
          'mean': CIFAR100_TRAIN_MEAN,
          'std': CIFAR100_TRAIN_STD
      },
      'tiny_imagenet': {
          'num_classes': TINY_IMAGENET_NUM_CLASSES,
          'mean': TINY_IMAGENET_TRAIN_MEAN,
          'std': TINY_IMAGENET_TRAIN_STD,
          'data_path': TINY_IMAGENET_PATH
      }
  }
claim: "Only cifar10/cifar100/tiny_imagenet are configured; there is no Kaggle cats-vs-dogs config and no script that renders the feature-map (Fig. 3), relational-graph (Figs. 1/4/5) visualizations. Only an unrelated CIFAR-10 t-SNE script (tsne.py) exists."
concern: "The repeatedly cited '98.1% -> 99.8%' cats-vs-dogs result and the qualitative figures cannot be reproduced from the repo."
resolution: "Authors: add the cats-vs-dogs data/config and the figure-generation scripts, or note that these qualitative results were produced off-repo."
cross_refs: []
paper_ref: "Figs. 1, 3, 4, 5; lines 235, 334"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: unpinned-and-missing-deps
category: missing
topic: "dependencies / environment"
title: "Dependencies unpinned; matplotlib & scikit-learn used but not installed"
severity: low
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/setup.sh
line_start: 14
line_end: 15
quote: |
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
  pip install numpy einops pillow
claim: "setup.sh installs unversioned numpy/einops/pillow and unversioned pytorch; tsne.py imports matplotlib and sklearn (scikit-learn) which setup.sh never installs."
concern: "No version pins means the environment is not exactly rebuildable, and tsne.py will fail at import with ModuleNotFoundError for matplotlib/sklearn under the documented environment."
resolution: "Authors: pin versions and add matplotlib + scikit-learn (and matplotlib for tsne) to the install instructions, ideally via a requirements.txt/environment.yml."
cross_refs: []
paper_ref: null
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: computeweight-missing-math-import
category: bug
topic: "loss weighting"
title: "sqrt/arccos/cosine weighting schemes crash (math never imported)"
severity: high
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/computeweight.py
line_start: 28
line_end: 44
quote: |
      if method == 'linear':
          return x
      elif method == 'sqrt':
          return math.sqrt(x)
      elif method == 'squared':
          return x ** 2
      elif method == 'equal':
          if num_active_graphs and num_active_graphs > 0:
              return 1.0 / num_active_graphs
          return 1.0
      elif method == 'arccos':
          return math.acos(1 - 2 * x) / math.pi
      elif method == 'cosine':
          return (1 + math.cos(math.pi * x)) / 2
      else:
          logger.warning(f"Invalid weight method '{method}'. Using linear weighting.")
          return x
claim: "calculate_weight() calls math.sqrt / math.acos / math.cos / math.pi and logger.warning, but computeweight.py imports only `torch` (no `import math`, no `logger`). Verified by invocation in _audit_code/check_weight_methods.py: linear/squared/equal/adaptive return values, while sqrt/arccos/cosine raise `NameError: name 'math' is not defined`."
concern: "Three of the seven weighting schemes the paper evaluates ('square root, ..., cosine, arccosine'; Fig. 6 / Tables 1-3) cannot be run in the released code, and any unrecognized method hits the `logger`-NameError instead of falling back to linear."
resolution: "Add `import math` (and define/import `logger`) in computeweight.py; confirm whether the sqrt/arccos/cosine rows in the paper were produced by this code."
cross_refs: ["no-multirun-aggregation-harness"]
check_script: _audit_code/check_weight_methods.py
paper_ref: "§3.2 lines 228-244; §4.1 line 515; Fig. 6"
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: classifier-head-num-classes-hardcoded
category: bug
topic: "model wiring"
title: "Backbone num_classes fixed at default; CIFAR-100 run uses a 10-way head"
severity: medium
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/utils.py
line_start: 182
line_end: 184
quote: |
      elif args.net == 'mobilenet':
          from models.mobilenet import mobilenet
          net = mobilenet()
claim: "get_network() instantiates every backbone with no class-count argument, so the classifier head keeps the file default (e.g. mobilenet/googlenet -> class_num=10, mobilevit -> num_classes=200) regardless of -dataset; nothing in train.py overrides it."
concern: "Running the README's CIFAR-100 example (`-net mobilenet`) builds a 10-output head on a 100-class problem; the user must hand-edit model source to match each dataset, so the documented commands do not train correctly out of the box."
resolution: "Thread dataset_config['num_classes'] into get_network()/the model constructors, or document that the head must be edited per dataset (the README only notes this informally)."
cross_refs: []
paper_ref: null
tags: [lones:stage-3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: graph-loss-normalized-by-numel
category: difference
topic: "loss definition"
title: "Implemented GCR loss divides Frobenius norm by element count; Eq. 5 does not"
severity: low
confidence: medium
status: finding
file: code/Darcyddx__graph-prompt/train.py
line_start: 145
line_end: 157
quote: |
                  for i, relationship in enumerate(graph_list[:-1]):
                      if args.use_detach:
                          dist = torch.norm((ground_truth_embedding * similarity_indicator).detach() - relationship,
                                            p=2) ** 2 / ground_truth_embedding.numel()
                      else:
                          dist = torch.norm((ground_truth_embedding * similarity_indicator) - relationship,
                                            p=2) ** 2 / ground_truth_embedding.numel()

                      original_index = active_indices[i]
                      weight = calculate_weight(original_index, args.num_elements, args.weight_method, num_active_graphs)
                      dist = dist * weight

                      total_graph_loss_batch += dist
claim: "The per-layer alignment term is computed as ||P - F||^2 / numel(P), i.e. a mean squared difference, whereas Eq. 5 in the paper writes the unnormalized squared Frobenius norm ||triu(F)-triu(P)||_F^2."
concern: "With λ=1 the effective graph-loss scale differs from the equation by a 1/numel factor; the implemented (mean) version is itself valid and matches the 1/n^2 form used in Theorem 1, so this is a description-vs-code mismatch rather than an error."
resolution: "Authors: confirm the loss used in experiments includes the 1/numel normalization and align Eq. 5 (or note the normalization)."
cross_refs: []
paper_ref: "Eq. 5 (lines 220-226); Theorem 1 normalization (lines 311-313)"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: model-selection-on-test-set
category: methodology
topic: "model selection / data splitting"
title: "Best accuracy chosen by monitoring the test set; no validation split"
severity: medium
confidence: high
status: finding
file: code/Darcyddx__graph-prompt/train.py
line_start: 281
line_end: 329
quote: |
  @torch.no_grad()
  def eval_training(epoch=0, tb=True):
      start = time.time()
      net.eval()

      test_loss = 0.0  
      correct = 0.0

      for (images, labels) in test_loader:

          if args.gpu:
              images = images.cuda()
              labels = labels.cuda()

          outputs, _ = net(images, [0] * (args.num_elements + 1))  
          loss = loss_function(outputs, labels)

          test_loss += loss.item()
          _, preds = outputs.max(1)
          correct += preds.eq(labels).sum()
claim: "Each epoch is evaluated on the held-out test set (test_loader), and train.py saves the checkpoint and reports best_acc = max test accuracy over all 200 epochs (lines 519-528); there is no separate validation split, so the reported number is the best-epoch test accuracy."
concern: "Selecting the best epoch by test-set accuracy uses the test set for model selection, which biases the reported accuracy upward (best-of-200 peeking); the absolute numbers in Tables 1-3 are optimistic relative to a fixed last-epoch or validation-selected protocol."
resolution: "Authors: confirm whether reported accuracies are best-epoch-on-test or last-epoch; if best-epoch, report fixed-epoch or validation-selected accuracy. Note: this follows the upstream pytorch-cifar100 convention and is applied equally to baseline and GCR, so relative gains are less affected than absolute values."
cross_refs: []
paper_ref: "Tables 1-3"
tags: [leakage:L1.1, reforms:5, whalen:pitfall-2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 5          | high         | ImageNet-1K/transformer experiments, multi-run aggregation, cats-vs-dogs & figure scripts, deps |
| bug         | 2          | high         | sqrt/arccos/cosine weighting crash (no `import math`); classifier head class-count hardcoded |
| difference  | 1          | low          | Graph loss divides by numel; Eq. 5 unnormalized (both valid) |
| methodology | 1          | medium       | Best accuracy selected on the test set; no validation split |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **[bug]** `sqrt`/`arccos`/`cosine` weighting schemes crash with `NameError` (`import math` missing) — 3 of the 7 evaluated schemes can't run in the released code (`computeweight-missing-math-import`).
2. **[missing]** ImageNet-1K + iFormer/ViG/MAE (Table 4) and the AdamW/AMP transformer regime have no code path in the repo (`imagenet1k-transformer-experiments-absent`).
3. **[missing]** No harness aggregates the per-config tables or the mean±std over 10/3 runs that constitute every headline number (`no-multirun-aggregation-harness`).
4. **[methodology]** Reported accuracy is best-epoch-on-test with no validation split — optimistic absolute numbers (`model-selection-on-test-set`).
5. **[bug]** Backbone `num_classes` is fixed at file defaults; the README's CIFAR-100 command builds a 10-way head (`classifier-head-num-classes-hardcoded`).
6. **[missing]** Cats-vs-dogs result and the Fig. 3/4/5 visualizations have no scripts; `tsne.py` is unrelated (`catsvsdogs-and-visualization-scripts-absent`).

### Items that genuinely look fine
- The masked prediction graph `P = M ⊙ S` (Eq. 3) is correctly implemented: `ground_truth_embedding` (softmax-output relationship graph, `graph_list[-1]`) × `similarity_indicator` (the class mask M from `indicator.py`) (`train.py:120-121,132-136`).
- The alignment loss `||F - P||^2` (Eq. 4) and the upper-triangular-only construction (`gcr.py:33-34`, `indicator.py:19-20`) match the paper's strictly-upper-triangular design.
- The adaptive weighting `softmax(-L_graph)` (`computeweight.py:17`) matches Eq. 6, and runs without error.
- CNN training config (200 epochs, lr 0.1, milestones 60/120/160, batch 128, wd 5e-4, momentum 0.9) matches §4.1 for CNNs exactly (`train.py:418-419`, `settings.py:44-45`).
- The GCL is genuinely parameter-free as claimed (`RelationshipLayer` has no learnable parameters; `gcr.py:6-36`).
- Train/test are the standard torchvision CIFAR splits and the Tiny ImageNet train/val split; no resampling or pre-split fitting (`utils.py:286-309,336-360`).

### Open questions for the authors
- Were the `sqrt`/`arccos`/`cosine` weighting rows (Fig. 6, Tables) produced by this exact `computeweight.py`, given it cannot import those branches? (high-severity / the result depends on the answer)
- Were transformer/ImageNet-1K results produced by a separate codebase not included here? If so, can it be released?
- Are the reported accuracies best-epoch-on-test or last-epoch? (determines whether absolute numbers are optimistically biased)
