# Audit — Unifying Reconstruction and Density Estimation via Invertible Contraction Mapping in One-Class Classification (NeurIPS 2025)

## Summary

The repo `code/wxl1122__URD/` is the official URD code (README: "Unifying Reconstruction
and Density Estimation … Accepted by NeurIPS 2025"). It contains a single tabular
anomaly-detection pipeline under `Unifying_INN/`: `main.py` (train + eval entrypoint),
`utils/loadData.py` (data loading / split), `utils/utils.py` (losses + metrics),
`model/INN.py` (encoder + flow wrapper), and `flow_net/` (an i-ResNet/monotone-block
invertible network). The paper reports results across three domains — tabular
(Table 1, 20 datasets; ablations Tables 3, 5), natural-image CIFAR-10 and industrial
MVTec AD (Table 2; ablations Tables 4, 5). I read the paper (paper.pdf), traced every
numbered table to code, and ran deterministic checks under `_audit_code/`
(`check_repo_completeness.py`, output `_audit_code/out/repo_completeness.json`) to
enumerate which datasets and code paths are present and to confirm the data-load path,
the absence of any image-domain code, and that test-set evaluation is interleaved with
training. The repo is READ-ONLY; no repo file was modified.

The repository covers only a small fraction of the paper's claimed experiments: it has
no image-domain code at all (Table 2, the entire visual-AD half of the paper), only
4 of the 20 tabular datasets, no ablation/sweep harness (Tables 3–5), and the data
loader's hardcoded absolute path prevents loading even the four bundled datasets out of
the box.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1, tabular AP/AUROC, 20 datasets, headline 78.64/91.19 | `Unifying_INN/main.py` + `loadData.py` + `utils.py:aucPerformance` | not runnable as-is (hardcoded path; 16/20 datasets absent) | — | PARTIAL / NOT RUNNABLE |
| Table 1 individual rows (e.g. Wine 99.99/100.0, Wbc 94.72/99.24) | only glass/thyroid/wbc/wine `.mat` present | — | — | PARTIAL (4/20 datasets) |
| Table 2, CIFAR-10 AUROC + MVTec image/pixel AUROC | (none) | — | — | MISSING (no image code) |
| Table 3, tabular ablation RE/DE components | (none — no component-toggle harness) | — | — | MISSING |
| Table 4, CIFAR-10/MVTec ablation | (none — no image code) | — | — | MISSING |
| Table 5, hyperparameter sweep K×R (tabular + MVTec) | (none — no sweep harness; K/R hardcoded) | — | — | MISSING |
| Headline AP gain "+11.2/4.67 vs baseline" (RE/DE ablation) | (none) | — | — | MISSING (ablation) |
| AD score = −α·log p + ‖x1−x2‖₂ (eval) | `Unifying_INN/main.py:45`, `utils.py:24-27` | direction correct, α=0.01 hardcoded | ✓ (logic) | Verified (logic only) |
| Train/test split = 50% normal train, rest+all anomalies test (MCM convention) | `Unifying_INN/loadData.py:11-26` | matches MCM partitioning | ✓ | Verified |

## missing

```yaml finding
id: image-domain-code-missing
category: missing
topic: "result traceability / image experiments"
title: "No CIFAR-10 / MVTec image-AD code; Table 2 and Table 4 unreproducible"
severity: high
confidence: high
status: finding
file: Unifying_INN/main.py
line_start: 52
line_end: 64
quote: |
  def main(args):
      setup_seed(42)
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      #data_loader
      train_x, train_y, test_x, test_y=loadData(args.datasetname)
      train_dataset=torch.utils.data.TensorDataset(torch.tensor(train_x))
      train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
      test_dataset=torch.utils.data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
      test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
      input_dim=torch.tensor(train_x).shape[1]

      #model
      model=UniINN(input_dim=input_dim).to(device)
claim: "The only entrypoint loads tabular .mat features and a plain MLP encoder; nothing in the repo loads CIFAR-10, uses the DO2HSC CNN, loads MVTec AD, uses a frozen WideResNet-50 backbone, or computes pixel-level AUROC."
concern: "Table 2 (CIFAR-10 + MVTec image/pixel AUROC) and Table 4 (image ablation) — the entire visual-AD half of the paper, including the headline MVTec 99.5/98.3 — have no code that produces them."
resolution: "Authors: please add the image-domain training/evaluation code (CIFAR-10 + MVTec, backbones, pixel-AUROC computation) used for Tables 2 and 4."
cross_refs: ["datasets-missing"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 2; Table 4; §4.1 'Implementation Details'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: datasets-missing
category: missing
topic: "data availability"
title: "Only 4 of 20 tabular datasets bundled; fetch script covers 13 with mismatched paths"
severity: high
confidence: high
status: finding
file: Unifying_INN/get_data.sh
line_start: 4
line_end: 19
quote: |
  #breastw
  mkdir -p data/breastw
  curl -o data/breastw/breastw.mat https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=1

  #cardio
  mkdir -p data/cardio
  curl -o data/cardio/cardio.mat https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1


  #glass
  mkdir -p data/glass
  curl -o data/glass/glass.mat https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=1

  #ionosphere
  mkdir -p data/ionosphere
  curl -o data/ionosphere/ionosphere.mat https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=1
claim: "Table 1 reports 20 tabular datasets, but only glass/thyroid/wbc/wine .mat files are present (see _audit_code/out/repo_completeness.json); get_data.sh fetches only 13, and it downloads into per-dataset subdirs (data/<name>/<name>.mat) whereas loadData.py reads a flat path data/<name>.mat, so even fetched files would not be found."
concern: "16 of the 20 Table-1 rows cannot be reproduced from the repo; the fetch script is both incomplete (13<20) and points to a directory layout the loader does not use."
resolution: "Authors: provide all 20 datasets (or working download links) and align get_data.sh output paths with the path loadData.py reads."
cross_refs: ["hardcoded-data-path"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 1 (20 tabular datasets)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-harness-missing
category: missing
topic: "ablations"
title: "No ablation/hyperparameter-sweep code for Tables 3 and 5"
severity: medium
confidence: high
status: finding
file: Unifying_INN/main.py
line_start: 70
line_end: 80
quote: |
  for epoch in range(args.num_epoch):
      model.train()
      loss_list = []
      for i, x_train in enumerate(train_loader):
          x_raw=x_train[0].to(device).float()   
          x1,x2,prior,post=model(x_raw)
          loss=reconstruction_loss(x1, x2)+0.5*(prior_loss(prior)+post_loss(post))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          loss_list.append(loss.item())
claim: "The single training loop always uses the full RE+DE loss with K/R fixed in model construction (denoise_net(latent_dim, 3)); there is no flag or harness to drop the RE or DE component (Table 3) or to sweep K and R (Table 5)."
concern: "Table 3 (RE/DE component ablation, incl. the headline '+11.2/4.67 vs baseline' gain) and Table 5 (K×R sweep) cannot be reproduced — the code that produces those rows is absent."
resolution: "Authors: provide the ablation driver(s) that toggle RE/DE and sweep K, R."
cross_refs: []
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 3; Table 5; §4.2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: hardcoded-data-path
category: bug
topic: "data loading"
title: "loadData reads a hardcoded absolute path on the authors' machine"
severity: high
confidence: high
status: finding
file: Unifying_INN/utils/loadData.py
line_start: 5
line_end: 5
quote: |
      dataset=loadmat('/home/qxl/ddpm/UniINN/data/'+datasetname+'.mat')
claim: "The data loader always reads from the absolute path /home/qxl/ddpm/UniINN/data/<name>.mat regardless of where the repo is checked out, ignoring the bundled Unifying_INN/data/ directory and the --save_folder/cwd."
concern: "Out of the box the pipeline crashes with FileNotFoundError for every user, so none of the tabular results (Table 1) are reproducible without editing the source."
resolution: "Authors: replace the hardcoded path with a relative path (e.g. Unifying_INN/data/) or a configurable --data_dir argument."
cross_refs: ["datasets-missing"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: cudnn-determinism-typo
category: bug
topic: "reproducibility / seeding"
title: "Typo 'deterministirc' makes cuDNN-determinism flag a silent no-op"
severity: low
confidence: high
status: finding
file: Unifying_INN/main.py
line_start: 21
line_end: 21
quote: |
      torch.backends.cudnn.deterministirc = True
claim: "The misspelled attribute 'deterministirc' creates a new unused attribute on torch.backends.cudnn; the real flag torch.backends.cudnn.deterministic is never set, so cuDNN remains nondeterministic."
concern: "Combined with dropout (Encoder, p=0.2) and GPU nondeterminism, reported single-number AUROC/AP values may not be exactly reproducible; impact is small for the conclusions."
resolution: "Authors: fix the typo to torch.backends.cudnn.deterministic = True."
cross_refs: []
check_script: _audit_code/check_repo_completeness.py
paper_ref: "§4.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

(No `difference` findings: where the code and paper diverge, the divergence is either an
absent artefact — routed to `missing` — or a break — routed to `bug`. The
implemented procedure for the four bundled tabular datasets is otherwise consistent with
the paper's description: MCM 50% partitioning, AD score = −α·log p + reconstruction
distance with α=0.01, λ=0.5 loss weighting, AUROC/AP metrics.)

## methodology

```yaml finding
id: test-set-model-selection
category: methodology
topic: "hyperparameter tuning / model selection"
title: "No validation set; test AUROC evaluated and printed every 5 epochs during training"
severity: medium
confidence: medium
status: finding
file: Unifying_INN/main.py
line_start: 82
line_end: 85
quote: |
          if (epoch+1) % 5 == 0:
              print('epoch: {} loss: {}'.format(epoch+1, avg_loss))
              torch.save(model.state_dict(), os.path.join(args.save_folder, args.datasetname+'.pth'))
              evaluation_batch(model, test_loader,device)
claim: "There is no validation split (loadData returns only train and test); every 5 epochs the checkpoint is overwritten and evaluation_batch runs on the test set and prints AUROC/AP, so the test metric is monitored throughout training and a user reads the best/desired printed value."
concern: "With the test set as the only signal observed during training and no held-out validation set, epoch/checkpoint selection (and the reported single number) can be biased toward the test set; the paper reports one AUROC/AP per dataset without specifying which epoch is selected."
resolution: "Authors: confirm whether the reported numbers are at a fixed final epoch (independent of the printed test AUROC) or selected using the test metric; if the latter, add a held-out validation set for model/epoch selection."
cross_refs: []
check_script: _audit_code/check_repo_completeness.py
paper_ref: "§4.1; Table 1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 3          | high         | No image-domain code (Table 2/4); 4/20 datasets; no ablation harness.  |
| bug         | 2          | high         | Hardcoded absolute data path crashes loading; cuDNN-determinism typo.  |
| difference  | 0          | -            | Implemented tabular procedure matches the paper where present.         |
| methodology | 1          | medium       | No validation set; test AUROC monitored every 5 epochs during training.|

### Top take-aways (≤6, by severity × confidence)
1. `missing` (high/high) — The entire visual-AD half (CIFAR-10 Table 2, MVTec, image ablation Table 4, incl. headline MVTec 99.5/98.3) has no code at all. [`image-domain-code-missing`]
2. `bug` (high/high) — `loadData.py` hardcodes `/home/qxl/ddpm/UniINN/data/...`, so the pipeline crashes for any user out of the box; no tabular result reproduces without editing source. [`hardcoded-data-path`]
3. `missing` (high/high) — Only 4 of 20 Table-1 datasets are bundled; `get_data.sh` fetches 13 into directories the loader doesn't read. [`datasets-missing`]
4. `missing` (medium/high) — No ablation/sweep harness for Tables 3 and 5 (RE/DE components, K×R). [`ablation-harness-missing`]
5. `methodology` (medium/medium) — No validation set; test AUROC is evaluated and printed every 5 epochs, enabling test-based epoch selection. [`test-set-model-selection`]
6. `bug` (low/high) — `cudnn.deterministirc` typo silently disables the determinism flag. [`cudnn-determinism-typo`]

### Items that genuinely look fine
- Train/test partitioning (`loadData.py:11-26`): 50% of normal samples for train, the rest plus all anomalies for test — matches the MCM convention the paper cites; standardization (thyroid/satimage-2) is fit on train only and applied to test, no leakage.
- AD scoring direction (`main.py:45`, `utils.py:24-27`): higher score = more anomalous, label 1 = anomaly, `roc_auc_score(labels, score)` consistent; α=0.01 weighting matches the paper's tabular setting.
- Loss composition (`main.py:76`): `reconstruction_loss + 0.5*(prior_loss+post_loss)` matches the paper's λ=0.5 weighting.
- Dependencies are pinned (`requirements.txt`).

### Open questions for the authors
- Are the reported Table-1 numbers taken at a fixed final epoch, or selected using the test AUROC printed during training (see `test-set-model-selection`)?
- Where is the image-domain code (CIFAR-10 / MVTec) and the ablation/sweep drivers used for Tables 2–5?
