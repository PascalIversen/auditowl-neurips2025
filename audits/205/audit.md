# Audit — Paper 205: *Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration* (ReCalX, NeurIPS 2025)

## 1. Summary

**What the repo is.** `github.com/thomdeck/recalx` (cloned at commit
`222807c`, a single squashed commit dated 2026-02-23, after the paper
submission) is a **minimal demonstration package**, not the experiment pipeline
that produced the paper's numbers. It contains a small Python package
(`recalx/`: `calibration.py`, `perturbations.py`, `model.py`,
`image_wrapper.py`) and two illustrative notebooks (`demo_tabular.ipynb` on the
Electricity dataset, `demo_images.ipynb` on a single cat/dog image with
DenseNet121). The README explicitly frames the notebooks as "demos." The
NeurIPS checklist (paper p.17) said "Code will be openly released after
potential acceptance"; what was released is this demo, not the full pipeline.

**What I did.** I read the full paper text (`paper.pdf` / `paper_text.txt`; note
the PDF is 22 pages — main text, references, and checklist only; the
Appendices A–D that the paper repeatedly cites for proofs and *all*
experimental details are **not in the provided PDF**). I read every source file
and both notebooks. I wrote three read-only checks under `_audit_code/`:
- `check_repo_inventory.py` — greps the repo for code producing each paper
  artefact (Tables 1–3, Figs 2–5, regression experiments).
- `check_ece_kl.py` — tests whether `recalx.calibration.ece_kl` matches the
  estimator the paper claims to use.
- `check_perturb_level_inference.py` — tests the perturbation-level inference
  used to select temperatures during attribution.

**Headline.** The released code does **not contain the experimental protocol
for any headline result.** None of Tables 1–3 or Figures 2–5 (the empirical core
of the paper) is produced by any script: there is no multi-dataset / multi-model
calibration-error harness, no sensitivity (robustness) code, no image
temperature-learning code (the image temperatures are a hand-supplied CSV), and
no regression experiments. The theoretical results (Theorems 3.2/3.4,
Props 4.1/4.2) are math and are out of code scope. Separately, two of the few
pieces of logic that *are* present are defective: the binned `ece_kl` is not the
estimator the paper names, and the perturbation-level inference reads almost
every attribution query as "fully perturbed."

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — CE_KL (AVG/MAX) for MLP & tabular-ResNet over Electricity/Covertype/Credit/Pol, mean-replacement (24 cells × 3 methods) | (none) — only Electricity MLP demo; no Covertype/Credit/Pol, no tabular-ResNet harness; demo uses a different CE metric | — | — | MISSING (no pipeline) |
| Table 2 — CE_KL for ResNet50/DenseNet121/ViT/SigLIP, zero & blur perturbations | (none) — no ViT/SigLIP, no blur, no CE harness, no temperature-learning for images | — | — | MISSING |
| Table 3 — Avg/Max Sensitivity, 4 models × 3 explainers × 2 perturbations | (none) — no sensitivity/robustness code at all | — | — | MISSING |
| Fig. 2 — normalized CE across 10 tabular datasets (MLP/ResNet) + 95% CIs | (none) — only 1 dataset, no CI code | — | — | MISSING |
| Fig. 3 — CE vs perturbation level for image classifiers | (none) | — | — | MISSING |
| Fig. 4 — remove-and-retrain on 4 tabular datasets, 3 seeds, 1000 samples, Shapley | `demo_tabular.ipynb` cells 16–20 (1 dataset, **1 seed**, **500 samples**) | qualitative only | partial | PARTIAL (demo only; differs from paper protocol) |
| Fig. 5 — qualitative Shapley maps (DenseNet121, SigLIP) | `demo_images.ipynb` cells 13–16 (DenseNet121, single image; no SigLIP) | qualitative | partial | PARTIAL (demo only) |
| Appendix C — regression experiments, quantile CE on 2 tabular datasets | (none) | — | — | MISSING |
| Appendix C — LIME global importances; sensitivity ablations; bin-count / sample-count sensitivity | (none) | — | — | MISSING |
| Image ReCalX temperatures (input to Table 2 / Fig 5) | `data/densenet_temperatures.csv` (10 hand-rounded monotone values) — **not produced by any code in the repo** | — | unverifiable | MISSING (no producing code) |
| KL CE estimator of Popordanoska et al. [51] (Sec. 5) | `recalx/calibration.py:55-81` `ece_kl` is a 15-bin top-1 binning, not the cited kernel estimator | binned approx | ✗ (different estimator) | MISMATCH |
| ReCalX adaptive-temperature selection during attribution | `recalx/calibration.py:203-228`; `recalx/image_wrapper.py:50-77` | inferred level ≈1.0 for non-reference samples | ✗ | MISMATCH (wiring bug) |

## 3. Findings

## missing

```yaml finding
id: no-table-figure-pipeline
category: missing
topic: "result traceability / repository provenance"
title: "No code produces any headline table or figure (Tables 1-3, Figs 2-5)"
severity: high
confidence: high
status: finding
file: README.md
line_start: 71
line_end: 74
quote: |
  ## Demos

  - **demo_tabular.ipynb** -- Train an MLP, learn ReCalX temperatures, compare calibration errors across perturbation levels, and validate explanation quality via remove-and-retrain.
  - **demo_images.ipynb** -- Load a pretrained DenseNet121 with ReCalX, visualize confidence under perturbation, and compare Shapley explanations.
claim: "The repo ships only a minimal package plus two single-dataset/single-image demo notebooks; the inventory grep (out/inventory.txt) finds no sensitivity (Table 3) code, no KernelSHAP/FeatureAblation, no ViT/SigLIP, no blur perturbation, no Covertype/Credit/Pol datasets, no regression/quantile-CE code, and no multi-dataset/multi-model calibration-error harness."
concern: "Every quantitative result in the paper (all of Tables 1-3 and Figs 2-5) is untraceable to code, so none of the empirical claims can be reproduced or checked from this repository."
resolution: "Authors: release the full experiment pipeline (data loaders for all datasets/models, the calibration-error harness, the sensitivity experiments, and the table/figure generation scripts) used to produce Tables 1-3 and Figures 2-5."
cross_refs: ["densenet-temps-placeholder", "regression-experiments-missing"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Tables 1-3; Figures 2-5; Section 5"
tags: [reforms:1, reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: densenet-temps-placeholder
category: missing
topic: "result traceability / trained artefacts"
title: "Image ReCalX temperatures supplied as a CSV with no code that learns them"
severity: high
confidence: high
status: finding
file: data/densenet_temperatures.csv
line_start: 1
line_end: 11
quote: |
  perturbation_level,temperature
  0.0,1.023
  0.1,1.156
  0.2,1.287
  0.3,1.412
  0.4,1.534
  0.5,1.651
  0.6,1.763
  0.7,1.871
  0.8,1.975
  0.9,2.084
claim: "The image demo and any image ReCalX result depend on per-bin temperatures loaded from data/densenet_temperatures.csv (image_wrapper.py:40-44), but no script in the repo fits these temperatures on ImageNet validation data; the values are smooth, monotone, evenly stepped and read like a placeholder."
concern: "The temperatures driving the image experiments (Table 2, Fig 5) are an unverifiable hand-supplied artefact with no producing code, so the image-side calibration claims cannot be reproduced."
resolution: "Authors: provide the script that learns image temperatures (the ImageNet perturbation sweep + per-bin cross-entropy minimization) and confirm whether the released CSV is the learned output or an illustrative placeholder."
cross_refs: ["no-table-figure-pipeline"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Section 4 (ReCalX); Table 2"
tags: [reforms:2, forensics:hidden-iteration]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: regression-experiments-missing
category: missing
topic: "evaluation coverage"
title: "Regression / quantile-calibration experiments (Appendix C) absent from repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  In Appendix C, we conducted corresponding experiments on two tabular regression datasets, which demonstrate that ReCalX effectively reduces the quantile-based calibration error [35] under explainability-specific perturbations for regression tasks as well.
claim: "The paper claims regression experiments using quantile-based calibration error on two tabular regression datasets; the repo contains no regression model, no quantile-CE code, and no isotonic/affine recalibrator (inventory grep: ABSENT)."
concern: "A claimed extension of the method (regression) has no supporting code, so that contribution is unreproducible."
resolution: "Authors: release the regression experiment code (probabilistic regression model, affine/isotonic calibrator, quantile-CE metric) referenced in Appendix C."
cross_refs: ["no-table-figure-pipeline"]
check_script: _audit_code/check_repo_inventory.py
paper_ref: "Section 4 'Applying ReCalX beyond Classification'; Appendix C"
tags: [reforms:1]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: perturb-level-vs-single-reference
category: bug
topic: "perturbation-level inference"
title: "Perturbation level mis-inferred: non-reference samples read as ~100% perturbed"
severity: high
confidence: high
status: finding
file: recalx/calibration.py
line_start: 220
line_end: 234
quote: |
      def _infer_perturbation_level(self, x: torch.Tensor) -> float:
          """Infer perturbation level by comparing to unperturbed reference."""
          if self.unperturbed_input is None:
              return 0.0
          
          # Fraction of features that differ from original
          diff = (x != self.unperturbed_input).float()
          level = float(diff.mean().item())
          return round(level, 1)
      
      def set_unperturbed_input(self, x: torch.Tensor):
          """Set reference input for perturbation inference."""
          if x.dim() == 1:
              x = x.unsqueeze(0)
          self.unperturbed_input = x[0].detach()
claim: "The temperature bin is chosen from the fraction of features that differ from a single stored reference vector (set to one sample via set_unperturbed_input). When Captum's Shapley/LIME attribution feeds many different samples through forward(), any sample other than the reference differs from it in nearly all features, so the inferred perturbation level is ~1.0 regardless of how many features were actually masked."
concern: "check_perturb_level_inference.py shows an UNPERTURBED non-reference sample is read as level 1.0 (should be 0.0), so during attribution almost every query gets the maximum-perturbation temperature, contradicting the code's stated intent of selecting the temperature for the actual perturbation level."
resolution: "Authors: confirm how the perturbation level is meant to be inferred per query during attribution; the level should be measured relative to the per-sample unperturbed input (or computed from the coalition mask Captum supplies), not relative to one global reference vector."
cross_refs: ["no-table-figure-pipeline"]
check_script: _audit_code/check_perturb_level_inference.py
paper_ref: "Section 4, f^pi_ReCalX(x,S) adaptive temperature T(S)"
tags: [lones:stage-4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: ece-kl-not-paper-estimator
category: difference
topic: "calibration metric"
title: "ece_kl is a 15-bin top-1 estimator, not the cited Popordanoska et al. [51] estimator"
severity: medium
confidence: high
status: finding
file: recalx/calibration.py
line_start: 55
line_end: 81
quote: |
  def ece_kl(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
      """KL-divergence based Expected Calibration Error.
      
      Measures calibration error as KL divergence between predicted
      and empirical distributions within bins.
      """
      predictions = np.argmax(probs, axis=1)
      confidences = np.max(probs, axis=1)
      accuracies = (predictions == labels).astype(float)
      
      bins = np.linspace(0, 1, n_bins + 1)
      ece_kl_value = 0.0
      
      for i in range(n_bins):
          in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
          
          if np.sum(in_bin) > 0:
              empirical_acc = np.mean(accuracies[in_bin])
              empirical_dist = np.array([1 - empirical_acc, empirical_acc])
              empirical_dist = np.clip(empirical_dist, 1e-45, None)
              
              for conf in confidences[in_bin]:
                  pred_dist = np.array([1 - conf, conf])
                  pred_dist = np.clip(pred_dist, 1e-45, None)
                  ece_kl_value += np.sum(empirical_dist * np.log(empirical_dist / pred_dist))
      
      return float(ece_kl_value / len(probs))
claim: "The paper (Sec. 5) states CE_KL was computed with the 'consistent and asymptotically unbiased estimator proposed by [51]' (Popordanoska et al., a kernel/KDE estimator), but the repo's ece_kl is a 15-bin top-1-confidence binning that collapses the K-class prediction to a binary [1-conf, conf] distribution."
concern: "check_ece_kl.py confirms this metric depends on n_bins and is invariant to the non-top-1 probability mass, so it is a different (biased, bin-dependent) estimator than the one the paper names; the demo's reported CE_KL values are therefore not the paper's quantity. (Routed as difference because the paper's tables are not produced here, so this is a paper-vs-code metric mismatch in the demo, not the source of a reported number.)"
resolution: "Authors: confirm which estimator produced Tables 1-2; if the kernel estimator of [51] was used, include its implementation in the repo rather than this binned proxy."
cross_refs: ["no-table-figure-pipeline"]
check_script: _audit_code/check_ece_kl.py
paper_ref: "Section 5: 'consistent and asymptotically unbiased estimator proposed by [51]'"
tags: [reforms:7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: retrain-demo-protocol-differs
category: difference
topic: "remove-and-retrain protocol"
title: "Demo remove-and-retrain uses 1 seed / 500 samples vs paper's 3 seeds / 1000 samples"
severity: low
confidence: high
status: finding
file: demo_tabular.ipynb
line_start: 1
line_end: 1
quote: |
  SEEDS = [0]
claim: "The paper states global importances are averaged over 1000 samples and retraining is performed over 3 random seeds [29] (paper p.8). The demo notebook's remove-and-retrain cell uses SEEDS = [0] (one seed) and computes importance over X_val[:500] (500 samples)."
concern: "The released remove-and-retrain demonstration does not follow the paper's stated protocol, so it neither reproduces Fig. 4 nor provides the variability the paper reports; this is a demo-vs-paper difference, not the source of a paper number."
resolution: "Authors: provide the Fig. 4 script run with the paper's protocol (3 seeds, 1000 samples, all four tabular datasets, MLP and tabular ResNet)."
cross_refs: ["no-table-figure-pipeline"]
paper_ref: "Section 5 'Global Remove and Retrain Fidelity'; Figure 4"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

N/A for a standalone leakage/baseline assessment: the experimental pipeline that
would carry such defects (splits, baselines, metric choices on the real
datasets) is not in the repo (see `no-table-figure-pipeline`), so there is no
implemented procedure to evaluate for methodological validity. The one
metric-validity concern that *is* implemented (`ece_kl`) is filed as a
`difference` because it does not feed any reported number. No methodology
finding is asserted on prose alone (No-Extrapolation Rule).

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 3 | high | No code produces any table/figure; image temperatures and regression experiments absent. |
| bug | 1 | high | Perturbation level mis-inferred against a single global reference → wrong temperature bin during attribution. |
| difference | 2 | medium | `ece_kl` is not the cited estimator; remove-and-retrain demo deviates from paper protocol. |
| methodology | 0 | - | Real experiment pipeline not present to assess; no defect asserted on prose alone. |

### Top take-aways (≤6, ranked)
1. **[missing] No pipeline for any headline result** — Tables 1-3 and Figs 2-5 are untraceable to code; the repo is a demo, not the experiment tree. (`no-table-figure-pipeline`, high/high)
2. **[bug] Perturbation-level inference is broken** — non-reference attribution queries are read as ~100% perturbed, selecting the wrong temperature; verified empirically. (`perturb-level-vs-single-reference`, high/high)
3. **[missing] Image temperatures are an unproduced placeholder CSV** — no code learns `densenet_temperatures.csv`. (`densenet-temps-placeholder`, high/high)
4. **[difference] `ece_kl` ≠ the estimator the paper cites [51]** — binned top-1 proxy, n_bins-dependent, ignores non-top-1 mass. (`ece-kl-not-paper-estimator`, medium/high)
5. **[missing] Regression / quantile-CE experiments absent** — claimed Appendix C contribution has no code. (`regression-experiments-missing`, medium/high)
6. **[difference] Remove-and-retrain demo deviates from paper protocol** — 1 seed / 500 samples vs 3 seeds / 1000 samples. (`retrain-demo-protocol-differs`, low/high)

### Items that genuinely look fine
- `temperature_scaling` (calibration.py:103-130) correctly minimizes cross-entropy over a bounded scalar temperature via L-BFGS-B — a standard, valid temperature-scaling implementation.
- Tabular demo data handling: `StandardScaler` is fit on train only and applied to val/test, and `feature_means` (the perturbation baseline) are computed from the training split only (demo_tabular cell 3) — no obvious train/test scaling leakage in the demo.
- Temperature scaling preserves the argmax/ranking (monotone, T>0), consistent with the paper's "accuracy-preserving / information-preserving" claim (Prop. 4.2).
- The `perturb_features` mean-replacement perturbation (perturbations.py:10-42) matches the paper's described "mean value replacement" strategy for tabular data.

### Open questions for the authors
- Was the full experiment pipeline (all datasets/models, sensitivity experiments, table/figure generators) run off-repo, and will it be released? Without it none of Tables 1-3 / Figs 2-5 is reproducible.
- Is `data/densenet_temperatures.csv` the learned output or an illustrative placeholder, and what script produced it?
- Which CE_KL estimator produced Tables 1-2 — the kernel estimator of [51] (as stated) or a binned one? The repo only ships a binned proxy.
- The provided PDF is 22 pages and contains no Appendices A-D, although they are cited for all proofs and experimental details — please confirm the appendix/supplement location (this audit could not verify against it).
