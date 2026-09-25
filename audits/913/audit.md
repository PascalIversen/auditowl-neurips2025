# Audit — Equivariance Everywhere All At Once: A Recipe for Graph Foundation Models (paper 913)

## Summary

The repository (`benfinkelshtein__EquivarianceEverywhere`, commit `1dfe387`) is the
official code base for the paper. It implements the proposed graph foundation model
("TS-GNN" / GFM, exposed as `gnn_type` `MEAN_GNN` (TS-Mean) and `GAT` (TS-GAT)),
the least-squares / pooling equivariant mixing transforms, a 29-dataset PyG loader,
a split helper, and a single train/eval driver (`main.py` → `experiment.py`). Training
logs go to Neptune. I read every Python file under `code/`, traced the paper's
result tables/figures to producing code, and wrote three read-only checks under
`_audit_code/` (`check_baselines_present.py`, `check_str_print_zip.py`,
plus their `out/` outputs). I did **not** run the model (it requires a GPU/Triton,
a Neptune account, and dataset downloads). PDF page rendering was unavailable in this
environment, so paper quotes are taken from the provided plain-text extraction and
attributed to `paper.pdf` via `paper_ref`; I kept confidence calibrated accordingly.

Headline findings: (1) the code implements only the proposed GFM — none of the
baselines reported alongside it (GraphAny, end-to-end MeanGNN/GAT, and the symmetry
ablations DSS-Mean / TS-SGC / TS-GCN / TS-GCNII in Tables 5–6) are in the repo, so most
table rows have no producing code; (2) the optimizer is `AdamW` in code but the paper
twice states "Adam"; (3) a split-index bug (`% ndim` instead of `% shape[1]`) collapses
the per-seed official-split selection to only two of the ~10 shipped splits.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — TS-Mean / TS-GAT (GFM) test acc, trained on cora | `main.py` + `experiment.py` (`run_multiple_data`/`test`) with README "best configs" | not run (GPU/Triton/Neptune needed) | n/a | Code present, value not reproduced here |
| Table 1 — MeanGNN, GAT (end-to-end baselines) | (none) | — | — | MISSING (no baseline model code) |
| Table 1 — GraphAny | (none; only a split URL comment) | — | — | MISSING (no GraphAny code) |
| Fig. 3 / Table 7 — TS-Mean zero-shot acc, trainset 1/3/5/7/9 | `main.py --train_test_setup inc_trainset --train_size {1,3,5,7,9}` (`train_test_setup.py:29-43`) | not run | n/a | Code present, value not reproduced |
| Fig. 3 / Table 7 — GraphAny zero-shot acc | (none) | — | — | MISSING (no GraphAny code) |
| Table 3 — non-parametric least-squares variants (±bias, ±mean) | `transforms/least_squares.py` computes LS weights, but no standalone evaluation driver for the "non-parametric" rows | — | — | MISSING (no driver producing these accuracies) |
| Table 5 — DSS-Mean, TS-Mean with varying symmetries | (none for DSS-Mean) | — | — | MISSING (only full TS-GNN implemented) |
| Table 6 — TS-SGC, TS-GCN, TS-GCNII | (none; `GNNType` only has GAT, MEAN_GNN) | — | — | MISSING (no SGC/GCN/GCNII cores) |
| Table 8 — Train/Val/Test ratios | `helpers/split_data.py:14-40` (`graphany_mask_splits`, 20·#classes train) + PyG official masks | ratios match 20·#classes/N | ✓ (spot-checked deezer 0.1%, co-cs 1.6%) | Verified consistent |
| "5 random seeds 0,1,2,3,4" | `helpers/constants.py:6` `SEEDS = list(range(5))` | [0,1,2,3,4] | ✓ | Verified |

## missing

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "baselines"
title: "No code for GraphAny, end-to-end MeanGNN/GAT, or symmetry-ablation baselines"
severity: high
confidence: high
status: finding
file: helpers/gnn_type.py
line_start: 29
line_end: 59
quote: |
  class GNNType(Enum):
      """
          an object for the different core
      """
      GAT = auto()
      MEAN_GNN = auto()
claim: "The repo only implements the proposed GFM with two core GNNs (GAT, MEAN_GNN); a repo-wide search (_audit_code/check_baselines_present.py) finds no implementation of GraphAny, of the end-to-end MeanGNN/GAT baselines, or of the symmetry-ablation models DSS-Mean / TS-SGC / TS-GCN / TS-GCNII."
concern: "The headline comparisons in Table 1 and Fig. 3 (vs GraphAny and end-to-end GNNs) and the symmetry ablations in Tables 5-6 cannot be reproduced from this repo because the competing/ablated models have no producing code."
resolution: "Authors: please add (or point to) the code that produces the GraphAny, end-to-end MeanGNN/GAT, DSS-Mean, TS-SGC, TS-GCN and TS-GCNII numbers, run under the same split/seed/metric harness as the GFM."
cross_refs: ["§5", "Table 1", "Table 5", "Table 6", "Figure 3"]
check_script: _audit_code/check_baselines_present.py
paper_ref: "Table 1; Table 5; Table 6; Figure 3"
tags: [reforms:5, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: nonparam-ls-table3-no-driver
category: missing
topic: "result traceability"
title: "Table 3 non-parametric least-squares accuracies have no evaluation driver"
severity: medium
confidence: medium
status: finding
file: transforms/least_squares.py
line_start: 16
line_end: 52
quote: |
  def solve_ls_with_bias_cv(x: Tensor, y: Tensor, lambdas: List[float] = None, k_folds: int = 10) -> Tuple[Tensor, Tensor]:
      if lambdas is None:
          lambdas = [1e-15, 1e-8, 1e-6, 1e-4, 1e-2]  # log scale
claim: "The least-squares closed-form solver is present and used as a pre-transform inside the GFM, but no script evaluates the standalone 'non-parametric least-squares variants with and without bias and mean aggregation' as a classifier to produce Table 3's accuracies."
concern: "Table 3's reported accuracies cannot be traced to a script that computes them; only the GFM end-to-end path exercises this code."
resolution: "Authors: provide the script that evaluates the non-parametric LS variants reported in Table 3."
cross_refs: ["Table 3"]
paper_ref: "Table 3"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec-and-hard-neptune-dep
category: missing
topic: "expected code completeness / dependencies"
title: "No requirements file; Neptune (with placeholder token) is a hard run-time dependency"
severity: medium
confidence: high
status: finding
file: experiment.py
line_start: 45
line_end: 45
quote: |
        self.neptune_logger = neptune.init_run(project=args.project, api_token=API_TOKEN)  # your credentials
claim: "There is no requirements.txt / environment.yml / pyproject.toml in the repo (only README pip lines); `Experiment.__init__` unconditionally calls `neptune.init_run` with the required `--project` arg and `API_TOKEN='...'` placeholder (helpers/constants.py:9), so every train/eval run requires a Neptune account and a manually inserted token."
concern: "The environment is not fully pinned and the only entrypoint cannot run without an external logging service + credentials, blocking out-of-the-box reproduction."
resolution: "Authors: add a pinned dependency file and make Neptune optional (guard init_run / allow an offline or no-op logger) so the code runs without external credentials."
cross_refs: []
paper_ref: "README Installation"
tags: [reforms:1, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: split-index-modulo-ndim
category: bug
topic: "data splitting"
title: "Per-seed split selection uses `% ndim` (==2) instead of `% shape[1]`, using only 2 of ~10 splits"
severity: medium
confidence: high
status: finding
file: helpers/split_data.py
line_start: 59
line_end: 68
quote: |
      if train_mask.ndim == 2:
          # ! Multiple splits
          # Modified: Use the ${seed} split if not specified!
          split_index = seed
          # Avoid invalid split index
          split_index = (split_index % train_mask.ndim)
          train_mask = train_mask[:, split_index].squeeze()
          val_mask = val_mask[:, split_index].squeeze()
          if test_mask.ndim == 2:
              test_mask = test_mask[:, split_index].squeeze()
claim: "For datasets whose masks ship as [n_nodes, n_splits] (Heterophilous, WebKB, Actor, WikipediaNetwork, texas — typically 10 splits), the intended `split_index = seed` is reduced by `% train_mask.ndim`, but `ndim` is always 2 for a 2-D mask, so only columns 0 and 1 are ever selected (seeds 0,2,4 -> split 0; seeds 1,3 -> split 1)."
concern: "The code's own intent (one official split per seed, comment 'Use the ${seed} split') is contradicted: across the 5 seeds only two of the ~10 official splits are evaluated, so the reported mean/std over 'five random seeds' does not span five distinct official splits for these datasets."
resolution: "Replace `train_mask.ndim` with `train_mask.shape[1]` (number of splits). Authors: confirm whether reported numbers used distinct per-seed splits or only splits 0/1."
cross_refs: ["splits-claim-vs-official"]
paper_ref: "Section 'Datasets'; reproducibility checklist (seeds 0-4)"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: str-print-mislabel
category: bug
topic: "logging / reporting"
title: "Console str_print zips 3 split names against a length-2 metric tensor, mislabeling printed metrics"
severity: low
confidence: high
status: finding
file: helpers/utils.py
line_start: 50
line_end: 54
quote: |
      elif metric_mean is not None and metric_std is not None:
          path += ' '
          for split_name, mean, std in zip(['train', 'val', 'test'], metric_mean, metric_std):
              path += f'{split_name}={round(mean.item() * 100, DECIMAL)}+-{round(std.item() * 100, DECIMAL)},'
          path = path[:-1]
claim: "`metric_mean`/`metric_std` are the length-2 tensors [val, test] (helpers/metrics.py:14-15 `get_fold_metrics`), but str_print zips them against the 3 names ['train','val','test']; zip truncates, so the printed 'train=' shows the val value, 'val=' shows the test value, and the test value is dropped (verified in _audit_code/check_str_print_zip.py)."
concern: "The human-readable console summary mislabels which split each accuracy belongs to; the value reported as 'val' is actually test accuracy."
resolution: "Use the names ['val','test'] (matching get_fold_metrics) in str_print's zip. Note: Neptune logging at experiment.py:103-113 uses correct enumerate(['val','test']) indices, so logged values are not affected — only the printed string."
cross_refs: []
paper_ref: "n/a (logging only)"
tags: []
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: adamw-vs-adam
category: difference
topic: "optimizer"
title: "Code uses AdamW; paper states Adam"
severity: low
confidence: high
status: finding
file: experiment.py
line_start: 158
line_end: 158
quote: |
              optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.lr)
claim: "The training loop optimizes with `torch.optim.AdamW`, whereas the paper states (twice) that all models are optimized with Adam: 'We optimize all models using the Adam optimizer' (Section 'Datasets'/setup) and 'optimizer choice (Adam)' (reproducibility checklist)."
concern: "AdamW applies decoupled weight decay (default 0.01) absent from plain Adam, so the implemented optimizer differs from the described one; both are valid but the description is inaccurate."
resolution: "Authors: confirm whether AdamW (with what weight decay) or Adam was used for the reported numbers, and reconcile the paper text."
cross_refs: []
paper_ref: "Section 5 setup ('Adam optimizer'); reproducibility checklist Q6 ('optimizer choice (Adam)')"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: splits-claim-vs-official
category: difference
topic: "data splitting"
title: "Paper says 'official / standard splits'; for several datasets the code synthesizes 20-per-class splits"
severity: low
confidence: medium
status: finding
file: helpers/split_data.py
line_start: 43
line_end: 53
quote: |
  def get_masks(data: Data, seed: int) -> Tuple[Tensor, Tensor, Tensor]:
      if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask"):
          train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
      else:
          n_nodes = data.x.shape[0]
          label = data.y
          num_class = label.max().item() + 1
          train_mask, val_mask, test_mask = graphany_mask_splits(
              n_nodes, label, num_train_nodes=20 * num_class, seed=seed
          )
claim: "The paper says it uses datasets' 'respective official splits' / 'standard, commonly used splits', but for PyG datasets that ship no masks (CitationFull, Amazon, Coauthor, Airports, AttributedGraphDataset, LastFMAsia, DeezerEurope) the code generates a fresh GraphAny-style stratified split with 20*#classes training nodes."
concern: "For those datasets the 'official split' is not used; instead a 20-per-class split is synthesized (this is itself a standard and valid protocol, and the resulting ratios match Table 8, so this is a description mismatch rather than an invalid procedure)."
resolution: "Authors: clarify in the paper which datasets use shipped official splits vs the synthesized 20-per-class GraphAny split."
cross_refs: ["split-index-modulo-ndim"]
paper_ref: "Section 'Datasets' ('respective official splits'); Table 8"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: pca-fit-on-full-graph
category: methodology
topic: "preprocessing / leakage"
title: "PCA fit on the full graph's node features before train/test split (full-cora, co-cs, co-physics)"
severity: low
confidence: medium
status: question
file: transforms/pca.py
line_start: 15
line_end: 19
quote: |
      def __call__(self, data: Data) -> Data:
          if data.x.shape[1] > self.n_components:
              data.x = standardize(x=data.x)
              x = data.x.numpy()
              data.x = torch.from_numpy(self.pca.fit_transform(x))
claim: "PCA components are fit on all nodes of the graph (data.x includes val/test nodes) as a load-time transform, before split_data_per_fold assigns train/val/test masks; this affects full-cora, co-cs, co-physics (get_pca_components)."
concern: "Fitting the dimensionality reduction on the full node set uses test-node feature statistics; in the transductive node-classification setting test features are legitimately available, and PCA is label-free, so the leakage is weak — but the component basis is not train-only as a strict leakage-aware protocol would require."
resolution: "Authors: confirm the transductive framing makes full-graph PCA acceptable, or refit PCA on training nodes only and report whether the affected datasets' numbers change."
cross_refs: []
paper_ref: "Section 'Datasets' (PCA to 2048 components)"
tags: [leakage:L2.1, reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

```yaml finding
id: hyperparam-selection-criterion-undocumented
category: methodology
topic: "hyperparameter tuning"
title: "No model-selection script; criterion for choosing README 'best configs' / epoch not in repo"
severity: medium
confidence: low
status: question
file: experiment.py
line_start: 159
line_end: 166
quote: |
              with tqdm.tqdm(total=self.max_epochs, file=sys.stdout) as pbar:
                  losses_n_metric, state_dict =\
                      self.trainer(data_list=data_list, model=model, seed=seed, optimizer=optimizer, pbar=pbar)

              # save
              os.makedirs(self.save_load_path, exist_ok=True)
              with open(model_path, "wb") as f:
                  torch.save(state_dict, f)
claim: "Training saves the final-epoch state_dict (and fixed-epoch checkpoints); there is no validation-based early-stopping or automated sweep/selection script. The README lists single 'best configs' per setup, but the criterion used to pick them (and the epoch among the saved checkpoints) is not encoded anywhere in the repo."
concern: "Tables 10-11 list swept ranges (lr, lp_ratio, epochs, hidden dim) but the selection criterion is undocumented in code; if the reported test accuracy were used to choose configs/epochs that would be test-set tuning — but there is no evidence in the code that it was (no test-based selection logic exists)."
resolution: "Authors: state how the README best-configs and the reported epoch were selected (on validation accuracy?), and ideally include the selection/sweep script."
cross_refs: []
paper_ref: "Tables 10-11; reproducibility checklist Q6"
tags: [reforms:6, forensics:hidden-iteration]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|-----------------------------------------------------------------------|
| missing     | 3          | high         | Baselines/ablation models absent; Table 3 driver absent; no deps pin. |
| bug         | 2          | medium       | Split-index `% ndim` collapses to 2 splits; console metric mislabel.  |
| difference  | 2          | low          | AdamW vs Adam; "official splits" vs synthesized 20-per-class.         |
| methodology | 2          | medium       | Full-graph PCA (question); undocumented HP/epoch selection (question).|

## Top take-aways

1. **[missing] No baseline / ablation code** (`baselines-not-in-repo`, high/high): GraphAny, end-to-end MeanGNN/GAT, and Tables 5-6 symmetry ablations (DSS-Mean, TS-SGC, TS-GCN, TS-GCNII) have no implementation — most comparison rows are not reproducible from this repo.
2. **[bug] Split-index `% ndim`** (`split-index-modulo-ndim`, medium/high): for multi-split datasets only splits 0/1 are ever used across the 5 seeds, contradicting the code's own "use the ${seed} split" intent.
3. **[missing] No dependency pin + hard Neptune dependency** (`no-dependency-spec-and-hard-neptune-dep`, medium/high): the only entrypoint cannot run without external credentials and there is no requirements file.
4. **[missing] Table 3 non-parametric LS has no evaluation driver** (`nonparam-ls-table3-no-driver`, medium/medium).
5. **[difference] AdamW vs Adam** (`adamw-vs-adam`, low/high): paper states Adam twice; code uses AdamW.
6. **[methodology, question] Undocumented hyperparameter/epoch selection** (`hyperparam-selection-criterion-undocumented`, medium/low): no selection script; criterion not encoded.

## Items that genuinely look fine

- **Least-squares mixing fit train-only**: `solve_ls_with_bias_cv` is called on `data.x[data.train_mask]` / `data.y_mat[data.train_mask]` (`transforms/least_squares.py:108-132`), so the LS pre-transform does not leak val/test labels or features.
- **Per-node feature normalization**: `normalize`/`standardize` operate over the feature axis within each node (`dim=1`), so no cross-node (train↔test) statistic is shared.
- **Seeds match the paper**: `SEEDS = list(range(5))` = [0,1,2,3,4] (`helpers/constants.py:6`).
- **Test-time label propagation matches README**: at test (`is_batch=False`) the full train mask is used for propagation (`models/gfm.py:79-81`).
- **Table 8 split ratios are consistent** with the synthesized 20·#classes training count (spot-checked deezer ≈0.1%, co-cs ≈1.6%).
- **No early-stopping-on-test leakage**: the final-epoch model is saved/loaded; test loss is logged but never used to pick a checkpoint/epoch.

## Open questions for the authors

- Were the reported numbers for multi-split datasets computed over five distinct official splits, or (per the `% ndim` bug) only splits 0 and 1? (`split-index-modulo-ndim`)
- How were the README "best configs" and the reported epoch selected — on validation accuracy? Is there a sweep/selection script? (`hyperparam-selection-criterion-undocumented`)
- Is full-graph PCA acceptable under the transductive framing, or should it be train-only? (`pca-fit-on-full-graph`)
- Where is the code that produces the GraphAny / end-to-end / symmetry-ablation baseline numbers? (`baselines-not-in-repo`)
