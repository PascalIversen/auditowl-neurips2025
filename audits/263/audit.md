# Audit — Paper 263: "On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity"

## 1. Summary

The paper is an empirical study arguing that generalization in flow matching is
**not** driven by the stochasticity of the conditional-flow-matching (CFM) target.
It (a) shows the closed-form optimal velocity `û*` becomes near-deterministic
early in high-dimensional data (Fig. 1), (b) correlates failure to learn `û*`
with generalization (Fig. 2), (c) localizes generalization to small `t` via a
hybrid `û*`-then-`uθ` sampler (Fig. 3), and (d) introduces **Empirical Flow
Matching (EFM)**, regressing against a Monte-Carlo estimate `û*_M` of the
closed-form, reporting it does not hurt and slightly improves FID on CIFAR-10 /
CelebA (Fig. 4) and FMNIST (App. C tables).

The audited primary artefact is `code/generativemodels__closedformfm/`
(stated in the paper as `https://github.com/generativemodels/closedformfm`).
The other two cloned repos (`annegnx__PnP-Flow`, `atong01__conditional-flow-matching`)
are a different paper and the standard CFM dependency library, respectively, and
are referenced only as dependencies/related context.

What the repo actually contains: a toy-2D trainer (`src/train_toy2d.py`), a
CIFAR-10 trainer (`src/train_cifar10.py`) supporting CFM / OTCFM / EFM via Hydra
config, the closed-form velocity computation (`src/utils/mean_cfm.py`), an
offline FID-logging script (`src/utils/compute_fid.py`), and CFM utilities. The
top-level `main.py` is a `print("Hello from prigml!")` stub. `NOTES.md` is an
authors' internal scratchpad whose TODO checklist still lists "import a figure
code", "import the rest", "add license", "add tests" as unchecked.

What I did:
- Read the paper (PDF + text extraction), all `src/` files, all configs, and
  `pyproject.toml` / `uv.lock`.
- Wrote and ran `_audit_code/check_completeness.py` (file-existence + grep over
  the repo) to confirm which paper artefacts have producing code. Output:
  `_audit_code/out/completeness.csv`.
- Verified reachability of suspected defects by tracing call sites with grep.

Headline: the repo is **substantially incomplete**. Only the EFM/CFM/OTCFM
CIFAR-10 training loop (one input to Fig. 4) and a toy-2D trainer are present.
No code produces Figs. 1, 2, 3, the CelebA experiments, or the App. C MNIST/FMNIST
tables. The closed-form EFM target itself (the paper's core technical
contribution) **is** implemented and is methodologically faithful to Eq. (8).

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Closed-form target `û*_M` (Eq. 6/8, Alg. 2) | `src/utils/mean_cfm.py:136-188` (`get_full_velocity_field_batch`) | softmax over M batch samples, weighted velocities | ✓ (faithful to Eq. 8) | Verified |
| EFM training loop (Alg. 2) | `src/train_cifar10.py:190-207` (`expected_ucond and icfm`) | — (training) | ✓ structurally | Verified (logic) |
| CFM / OTCFM baselines (Fig. 4) | `src/train_cifar10.py:158-201` | — (training) | ✓ via `model_name` | Verified (logic) |
| Fig. 1a cosine-sim histograms (2-moons, CIFAR-10) | (none) | — | — | MISSING |
| Fig. 1c Imagenette dimension-vs-alignment | (none) | — | — | MISSING |
| Fig. 2 left: `E‖uθ−û*‖²` vs t, vs #samples | (none — no driver / no per-t error logging over subsamples) | — | — | MISSING |
| Fig. 2 middle: DINOv2 test FID-10k vs #samples | (none — training logs Inception train FID-5k only) | — | — | MISSING |
| Fig. 2 right: nearest-neighbor distance vs #samples | (none; helper `closest_image` exists but no driver) | — | — | MISSING |
| Fig. 3: hybrid `û*`→`uθ` LPIPS-vs-τ (CIFAR-10, CelebA) | (none; no τ-sweep driver, no LPIPS) | — | — | MISSING |
| Fig. 4 CelebA-64 FID curves | (none; paper uses pnpflow training script, App. D.5) | — | — | MISSING (off-repo) |
| App. C Tables 1–2 FMNIST FID | (none) | — | — | MISSING |
| Fig. 4 test FID (offline, both embeddings) | `src/utils/compute_fid.py:38-128` (`log_fid`) | FID train+test, Inception/DINO | plausibly ✓ | Partial (driver present; depends on saved models) |
| Table 3 hyperparameters | `conf/train_cifar10.yaml:1-43` | bs128, lr2e-4, ema0.9999, gradclip1, ch128 | ✓ | Verified |

No numerical value in the repo could be matched against a specific reported
number because (a) no figure/table-producing scripts ship and (b) running the
training is infeasible in this read-only audit (GPU + 500k-iteration training).
Verification above is structural (does the code implement the described
procedure) rather than value-level.

## 3. Findings

## missing

```yaml finding
id: missing-figure-experiment-code
category: missing
topic: "result traceability / repository provenance"
title: "No code produces Figures 1, 2, 3 or the App. C tables"
severity: high
confidence: high
status: finding
file: NOTES.md
line_start: 127
line_end: 129
quote: |
  - [ ] import a figure code, maybe the histogram one, for 2d toy data
  - [ ] import the rest
  - [ ] add license
claim: "The repo ships only a toy-2D trainer, a CIFAR-10 CFM/OTCFM/EFM trainer, and an offline FID-logging script; it contains no driver that computes the cosine-similarity histograms (Fig. 1a), the Imagenette dimension sweep (Fig. 1c), the velocity-error / test-FID / nearest-neighbour curves over dataset sizes (Fig. 2), the hybrid û*→uθ LPIPS-vs-τ experiment (Fig. 3), or the MNIST/FMNIST FID tables (App. C)."
concern: "Most of the paper's empirical evidence (3 of 4 main figures plus the appendix tables) has no producing code in the repo, so those results cannot be reproduced or independently checked; the authors' own NOTES.md confirms the figure code was never imported."
resolution: "Authors: please add the scripts that generate Figures 1–3 and the App. C tables (or point to where they live), including the per-time velocity-error logging and the LPIPS-based hybrid-sampler driver."
cross_refs: ["missing-lpips-dependency", "missing-celeba-training", "missing-readme"]
check_script: _audit_code/check_completeness.py
paper_ref: "Figures 1, 2, 3; Appendix C Tables 1-2"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-celeba-training
category: missing
topic: "result traceability"
title: "CelebA training (Figs. 3-4) is not in the repo"
severity: medium
confidence: high
status: finding
file: src/utils/compute_fid.py
line_start: 60
line_end: 69
quote: |
      elif cfg.dataset.lower() == "celeba":
          train_feat = ft_extractor.get_features(
              CelebA(split='train', root="../data", download=True,     transform=transforms.Compose(
                  [transforms.CenterCrop(178),
                   transforms.Resize([64, 64]),])), name="celeba64_train")
          test_feat = ft_extractor.get_features(
              CelebA(split='test', root="../data", download=True, transform=transforms.Compose(
                  [transforms.CenterCrop(178),
                   transforms.Resize([64, 64]),])), name="celeba64_test")
claim: "CelebA appears only inside the FID feature-extraction branch of compute_fid.py; no training script for CelebA exists in the repo (the only trainer, src/train_cifar10.py, is CIFAR-10 specific)."
concern: "Figures 3 (right) and 4 (bottom) report CelebA-64 results, but the model that produces those samples is trained off-repo via the pnpflow library (Appendix D.5), so the CelebA half of the headline experiments is not reproducible from this code."
resolution: "Authors: please add (or link) the exact CelebA training entrypoint and config used for Figs. 3-4, since the paper states it relies on an external library's training script."
cross_refs: ["missing-figure-experiment-code"]
paper_ref: "Appendix D.5; Figures 3 and 4 (CelebA)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-lpips-dependency
category: missing
topic: "dependencies / result traceability"
title: "LPIPS metric (Fig. 3) is neither implemented nor declared as a dependency"
severity: medium
confidence: high
status: finding
file: pyproject.toml
line_start: 7
line_end: 19
quote: |
  dependencies = [
      "clean-fid>=0.1.35",
      "fld",
      "hydra-core>=1.3.2",
      "hydra-submitit-launcher>=1.2.0",
      "matplotlib>=3.10.7",
      "mlflow>=3.5.1",
      "pandas>=2.3.3",
      "pot>=0.9.6.post1",
      "setuptools>=80.9.0",
      "torchcfm>=1.0.7",
      "torchdyn>=1.0.6",
  ]
claim: "Figure 3 measures generalization with the LPIPS metric (dist(x,D)=min_i LPIPS(x,x_i)), but no source file references lpips and the dependency list contains no lpips package."
concern: "The metric used for the entire Section 3.3 / Figure 3 conclusion ('generalization occurs early') has no producing code or declared dependency, so that experiment cannot be reproduced as described."
resolution: "Authors: add the LPIPS computation and its package to the dependency spec, or point to the script implementing dist(x,D)."
cross_refs: ["missing-figure-experiment-code"]
check_script: _audit_code/check_completeness.py
paper_ref: "Section 3.3; Figure 3"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-readme
category: missing
topic: "expected code completeness"
title: "README.md referenced by pyproject.toml does not exist"
severity: low
confidence: high
status: finding
file: pyproject.toml
line_start: 5
line_end: 5
quote: |
  readme = "README.md"
claim: "pyproject.toml declares readme = \"README.md\", but no README.md exists at the repo root (confirmed by _audit_code/check_completeness.py)."
concern: "There is no results table or 'exact commands to reproduce' documentation; the only run instructions live in NOTES.md, an internal scratchpad with cluster-specific SSH/SLURM steps and unchecked TODOs, so a complete-submission README is absent."
resolution: "Authors: add a README with the results table and the exact commands (per figure) to reproduce the paper's numbers."
cross_refs: ["missing-figure-experiment-code"]
check_script: _audit_code/check_completeness.py
paper_ref: "NeurIPS checklist Q5 (open access to code)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: rescaled-loss-overwritten
category: bug
topic: "loss computation"
title: "Rescaled-loss branch is immediately overwritten by the unrescaled loss"
severity: low
confidence: high
status: finding
file: src/train_cifar10.py
line_start: 204
line_end: 207
quote: |
                    if rescaled:
                        t_ = pad_t_like_x(t, x0)
                        loss = torch.mean(((vt - ut) * (1 - t_)) ** 2)
                    loss = torch.mean((vt - ut) ** 2)
claim: "When cfg.loss.rescaled is True, the rescaled loss assigned at line 206 is unconditionally overwritten by the plain MSE at line 207 (line 207 is not in an else branch), so the rescaling never takes effect."
concern: "Any CIFAR-10 run configured with rescaled=True would silently train on the unrescaled loss instead, contradicting the code's evident intent; the shipped configs use rescaled=False so the default reported runs are unaffected, hence low severity."
resolution: "Authors: guard line 207 with `else:` (or remove it) so the rescaled branch is honoured; confirm no reported CIFAR-10 result used rescaled=True."
cross_refs: []
paper_ref: "train_cifar10.yaml loss.rescaled"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: live-ipdb-breakpoint
category: bug
topic: "runtime / dead code"
title: "Live ipdb.set_trace() left in non-batched velocity-field helper"
severity: low
confidence: high
status: finding
file: src/utils/mean_cfm.py
line_start: 105
line_end: 106
quote: |
      import ipdb; ipdb.set_trace()
      return utot
claim: "get_full_velocity_field_ (the non-batched code path, reached when get_full_velocity_field is called with batch=False) contains an uncommented import ipdb; ipdb.set_trace() that would drop into a debugger / crash if ipdb is absent."
concern: "This path is not on the default training route (callers use batch=True, and solve_ode_t — the only caller passing batch defaults — is itself never invoked in the repo), so it is latent dead/non-default code; it would break any future use of the non-batched estimator."
resolution: "Authors: remove the stray ipdb breakpoint (and the similar commented ones) before release."
cross_refs: []
check_script: _audit_code/check_completeness.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: online-fid-train-only-inception
category: difference
topic: "evaluation consistency"
title: "In-training FID logs Inception train-FID-5k only; paper headlines DINOv2 test FID"
severity: low
confidence: medium
status: finding
file: src/train_cifar10.py
line_start: 98
line_end: 100
quote: |
      ft_extractor = InceptionFeatureExtractor(save_path="features")
      train_feat = ft_extractor.get_features(
          CIFAR10(train=True, root=root, download=True), name="cifar10_train")
claim: "The training loop computes FID online only against CIFAR-10 train features with the Inception extractor (FID-5k, train_cifar10.py:238-250); the paper's main metric is DINOv2 test FID-10k (Fig. 2 middle, Fig. 4)."
concern: "The reported test-set / DINOv2 numbers are not produced by the training loop but by the separate offline log_fid script (compute_fid.py), which both train and test features support; this is a benign division of labour rather than a contradiction, so it is flagged low/medium as a faithfulness note, not an error."
resolution: "Authors: confirm Fig. 2/4 test-FID values come from compute_fid.py::log_fid on saved checkpoints, and document which embedding/sample-count produced each panel."
cross_refs: ["missing-figure-experiment-code"]
paper_ref: "Metrics paragraph (Sec. 4.2); Fig. 2 middle, Fig. 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The core technical contribution — the Monte-Carlo
closed-form target `û*_M` — is implemented faithfully to Eq. (8): the softmax in
`get_full_velocity_field_batch` (`mean_cfm.py:175`) is taken over the M batch
samples (`dim=0`), the velocities are `(b - x)/σ_t`, and the EFM loop reuses the
current `x1` as `b(1)` (`train_cifar10.py:192-196`, the crucial trick from
Prop. 2). Generative-modeling evaluation here uses held-out test FID, which is
the appropriate generalization measure for this task; there is no train/test
split leakage concern of the supervised kind, and no statistical tests are
claimed. N/A for: pretraining contamination (DINOv2/Inception are standard
frozen FID feature extractors, not the model under study), temporal integrity
(no time-series data), target/shortcut leakage (unsupervised generative task).

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 4          | high         | Figs. 1-3, CelebA training, App. C tables, LPIPS dep, README all absent |
| bug         | 2          | low          | rescaled loss overwritten; live ipdb breakpoint (both off default path) |
| difference  | 1          | low          | online FID is Inception/train-only vs paper's DINOv2/test headline      |
| methodology | 0          | -            | EFM target faithful to Eq. 8; test-FID evaluation appropriate           |

### Top take-aways (≤6, ranked by severity × confidence)
1. **[missing]** No code produces Figures 1, 2, 3 or the App. C tables — most of
   the paper's empirical evidence is unreproducible from this repo
   (`missing-figure-experiment-code`, high/high).
2. **[missing]** CelebA experiments (Figs. 3-4) have no training code in-repo;
   trained off-repo via pnpflow (`missing-celeba-training`, medium/high).
3. **[missing]** The LPIPS metric underpinning the entire Fig. 3 "generalization
   is early" conclusion is neither implemented nor a declared dependency
   (`missing-lpips-dependency`, medium/high).
4. **[missing]** No README / reproduction commands / results table; only an
   internal scratchpad with unchecked TODOs (`missing-readme`, low/high).
5. **[bug]** `rescaled=True` silently trains the unrescaled loss
   (`rescaled-loss-overwritten`, low/high; default configs use False).
6. **[bug]** Live `ipdb.set_trace()` in the non-batched velocity helper
   (`live-ipdb-breakpoint`, low/high; off the default code path).

### Items that genuinely look fine
- The closed-form EFM target (`get_full_velocity_field_batch`, `mean_cfm.py:136-188`)
  faithfully implements Eq. (8): softmax over the M batch samples, correct
  velocity directions, correct σ_t scaling.
- The EFM training loop reuses the current sample `x1` as `b(1)`
  (`train_cifar10.py:192-196`), implementing the unbiasedness trick of Prop. 2.
- CFM / OTCFM / EFM are all selectable from one configurable trainer
  (`train_cifar10.py:158-201`), matching the three methods compared in Fig. 4.
- Hyperparameters in `conf/train_cifar10.yaml` match Table 3 (bs 128, lr 2e-4,
  EMA 0.9999, grad-clip 1, 128 channels).
- Evaluation uses held-out **test**-set FID (compute_fid.py loads both train and
  test), which is the right generalization metric for this generative task.

### Open questions for the authors
- Which script and config produced each panel of Figures 2 and 4, and what
  embedding / sample count was used for each (the in-training logger only emits
  Inception train FID-5k)?
- Were any reported CIFAR-10 runs trained with `rescaled=True`? If so, they were
  silently unrescaled (see `rescaled-loss-overwritten`).
- Can the Fig. 1 (cosine-similarity / Imagenette dimension) and Fig. 3 (hybrid
  τ-sweep / LPIPS) drivers be released?
