# Code-repository audit — GSRF (NeurIPS 2025, paper 1908)

## 1. Summary

GSRF extends 3D Gaussian Splatting to the RF domain (complex-valued Gaussians with a
Fourier–Legendre radiance basis, orthographic splatting, CUDA complex-valued ray
tracing). It is evaluated on three tasks/datasets: RFID spatial-spectrum synthesis
(PSNR/MSE), BLE RSSI synthesis (MAE in dBm) plus a KNN localization application, and
5G CSI prediction (SNR in dB). Headline claims are large efficiency gains and
quality improvements over baselines (NeRF2, WRF-GS, R2F2, FIRE).

The cloned repo (`code/nesl__GSRF/`, ~3000 files, mostly the bundled GLM third-party
library) contains the GSRF training and inference pipeline for all three datasets:
`train_rfid.py`, `main_ble.py`, `main_csi.py`, the three `inference_*.py` scripts,
the Gaussian model (`scene/gaussian_model.py`), FLE evaluation (`utils/fle_utils.py`),
losses (`utils/loss_utils.py`), dataset readers, and three CUDA submodules. The three
`inference_*.py` scripts compute GSRF's *own* metrics on the test set (PSNR/MSE/SSIM/LPIPS
for RFID, MAE for BLE, SNR for CSI) and write per-sample CSVs and summary JSONs.

What I did: read the paper's evaluation (§5) and FLE appendix; read all non-third-party
Python and the three YAML configs; and ran two read-only check scripts under
`_audit_code/` — `check_repo_completeness.py` and `check_traceability.py` — to confirm,
by grep/AST, the absence of baseline/ablation/localization/density code and the
FLE-degree config value. Outputs are in `_audit_code/out/`.

The repo reproduces GSRF's own per-task metrics, but **none of the baseline numbers,
ablation rows, the localization application, or the data-efficiency (measurement-density)
experiment are computed by any script in the repo** — these are the comparisons that
carry the paper's headline claims. Separately, every shipped config sets the FLE degree
to 9 (100 coefficients), contradicting the paper's repeated claim of L = 3 (16
coefficients).

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| GSRF RFID PSNR / MSE per-sample + CDF (Fig 3) | `inference_rfid.py:102-120` | computed (needs weights+data) | — | Verified (GSRF side only) |
| RFID "median improvements 21.2% PSNR, 56.4% MSE over NeRF2; 5.7% / 19.3% over WRF-GS" | (none — no NeRF2/WRF-GS code) | — | — | MISSING (no baseline code) |
| RFID training-time 0.27 h vs 5.01 / 1.61 h (Fig 4); inference 4.18 ms vs 352.73 / 7.58 ms (Fig 5) | (none — only GSRF timing) | — | — | MISSING (no baseline code) |
| RFID ablation Table 1 (PSNR 20.51 / 20.89 / 21.30 / 22.64) | (none — no SH/phase-off/no-Fourier toggles) | — | — | MISSING (no ablation code) |
| RFID "phase removal −8.37%", "Fourier-loss removal −6.28%" | (none) | — | — | MISSING (no ablation code) |
| RFID measurement-density / "9.8× less training data" (Fig 6) | (none — no density subsampling) | — | — | MISSING (no density experiment) |
| BLE GSRF RSSI MAE 4.09 dBm + CDF (Fig 9) | `inference_ble.py:50-84` | computed (needs weights+data) | — | Verified (GSRF side only) |
| BLE "32.79% improvement over NeRF2 (6.09 dBm)" | (none — no NeRF2 code) | — | — | MISSING (no baseline code) |
| BLE-based localization "31.40%" (Fig 10) | (none — no KNN fingerprinting code) | — | — | MISSING (no localization code) |
| CSI GSRF SNR mean 20.99 dB + dist (Fig 8) | `inference_csi.py:71-89` | computed (needs weights+data) | — | Verified (GSRF side only) |
| CSI baselines R2F2 / FIRE / NeRF2 SNR (Fig 8) | (none) | — | — | MISSING (no baseline code) |
| FLE degree L = 3 → 16 coefficients (§5.4, App. Table 4) | `arguments/configs/*/exp1.yaml` (`fle_degree: 9` → 100) | degree 9 | ✗ | MISMATCH (see fle-degree finding) |
| App. Table 4: FLE-degree sweep (L=1..≥5 PSNR/time) | (none — single `fle_degree` value, no sweep harness) | — | — | MISSING (no degree-sweep code) |

## 3. Findings

## missing

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "baselines / result traceability"
title: "No baseline (NeRF2, WRF-GS, R2F2, FIRE) code; all comparison numbers untraceable"
severity: high
confidence: high
status: finding
file: inference_rfid.py
line_start: 80
line_end: 99
quote: |
      psnr_list = []
      mse_list = []
      ssim_list = []
      lpips_list = []
      names_list = []

      viewpoint_stack = scene.getTestSpectrums().copy()

      # run inference on test set
      for step_idx, viewpoint_cam in enumerate(tqdm(viewpoint_stack, desc="Inference")):

          render_pkg = render(viewpoint_cam, gaussians, pipeline_para_args)

          spectrum = render_pkg["render"]

          gt_spectrum = viewpoint_cam.spectrum.cuda()
          spec_name = viewpoint_cam.spectrum_name

          pred_np = spectrum.detach().cpu().numpy()
          gt_np = gt_spectrum.cpu().numpy()
claim: "The inference scripts compute only GSRF's own metrics on the test set; a repo-wide grep finds no implementation, wrapper, or eval harness for NeRF2, WRF-GS, R2F2, or FIRE (check_traceability.py: baseline_impl_in_py == [])."
concern: "Every headline comparison — RFID 21.2%/56.4% over NeRF2 and 5.7%/19.3% over WRF-GS, the 18.56x/5.96x training and 84.39x/1.81x inference speedups, BLE 32.79% over NeRF2, and the CSI SNR ranking over R2F2/FIRE/NeRF2 — depends on baseline numbers that no script in the repo produces, so the central efficiency-and-quality claims are not reproducible from this code."
resolution: "Authors: provide the baseline training/inference/timing code (or the exact upstream commits and commands used) so each reported comparison value can be regenerated under the same data, split, and hardware."
cross_refs: ["ablation-code-missing", "localization-code-missing", "density-experiment-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "§5.1 (Fig 3-5), §5.2 (Fig 8), §5.3 (Fig 9)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-code-missing
category: missing
topic: "ablations"
title: "Table 1 ablation (SH-vs-FLE, phase-off, no-Fourier-loss) has no code path"
severity: high
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  Removing the phase channel while retaining only the amplitude results in a 8.37%
  reduction in PSNR compared to the full model with phase inclusion (second vs. last column).
claim: "The paper's Table 1 reports four configurations (Radiance/SH, Phase-off, no-Fourier-loss, full GSRF) with PSNR 20.51/20.89/21.30/22.64; a repo-wide grep finds no spherical-harmonics radiance branch, no amplitude-only/phase-disable switch, and no toggle to drop the Fourier loss as an ablation (check_traceability.py: ablation_toggles == []). The Fourier loss can only be zeroed via lambda_dfourier but there is no SH or phase-off variant."
concern: "The ablation that justifies the paper's three core design choices (FLE over SH, complex phase modeling, Fourier loss) cannot be reproduced because the alternative model variants are not implemented in the repo."
resolution: "Authors: add the SH-radiance, amplitude-only (phase-disabled), and no-Fourier-loss model variants and the script that produced Table 1, or point to where these toggles live."
cross_refs: ["baselines-not-in-repo", "fle-degree-config-vs-paper"]
check_script: _audit_code/check_traceability.py
paper_ref: "Table 1, §5.4 Ablation Study"
tags: [reforms:5]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: localization-code-missing
category: missing
topic: "downstream application"
title: "BLE KNN-fingerprinting localization (31.40% gain, Fig 10) absent from repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  The K Nearest Neighbors (KNN) identifies the K nearest matches and estimates the
  unknown transmitter position as the average of these K positions [29].
claim: "The paper reports a fingerprint-based localization experiment in which synthetic RSSI databases from GSRF and NeRF2 feed a KNN locator, with GSRF 'outperform[ing] NeRF2 by 31.40% on average' (Fig 10). No KNN/fingerprinting/localization code exists in the repo (check_traceability.py: localization_code == []; the only KNN is simple-knn used for Gaussian initialization distances)."
concern: "The localization result — the paper's claimed real-world sensing payoff — has no producing script, so the 31.40% figure is untraceable and unreproducible from the repo."
resolution: "Authors: provide the fingerprint-database construction and KNN localization script (with K and error metric) used for Fig 10."
cross_refs: ["baselines-not-in-repo"]
check_script: _audit_code/check_traceability.py
paper_ref: "§5.3 BLE-Based Localization, Fig 10"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: density-experiment-missing
category: missing
topic: "data efficiency experiment"
title: "Measurement-density / '9.8x less training data' study (Fig 6) not in repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
line_start: null
line_end: null
quote: |
  9.8× less training data to achieve comparable
claim: "The paper reports a measurement-density experiment (Fig 6) where GSRF trained at 0.8 measurements/ft^3 matches NeRF2 trained at varying densities 'obtained by random sampling from the original 70% training set', concluding '9.8x less training data'. No density-sweep or train-subsampling-by-density code exists in the repo (check_traceability.py: density_experiment == []); the RFID config trains on a fixed 0.8-ratio random split only."
concern: "The data-efficiency claim (a headline contribution) cannot be reproduced: there is no code that subsamples the training set to varying densities or runs the corresponding NeRF2 sweep."
resolution: "Authors: provide the density-subsampling harness and the per-density training/eval driver used for Fig 6."
cross_refs: ["baselines-not-in-repo"]
check_script: _audit_code/check_traceability.py
paper_ref: "§5.1 Measurement Density, Fig 6"
tags: [reforms:2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dependency-spec-file
category: missing
topic: "reproducibility / dependencies"
title: "No requirements.txt / environment.yml; deps only as unpinned README pip commands"
severity: low
confidence: high
status: finding
file: README.md
line_start: 11
line_end: 16
quote: |
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

  pip install -e submodules/simple-knn -e submodules/complex-gaussian-tracer -e submodules/complex-gaussian-tracer-csi

  pip install tqdm plyfile matplotlib scikit-image lpips seaborn pyyaml
  pip install "numpy<2"
claim: "There is no requirements.txt, environment.yml, pyproject.toml, or root setup.py (check_traceability.py: root_dependency_files == []); the dependency set is given only as README pip commands, with torch pinned but tqdm/plyfile/matplotlib/scikit-image/lpips/seaborn/pyyaml unpinned."
concern: "Unpinned scientific-stack versions (notably lpips and scikit-image, which directly affect the reported LPIPS/SSIM/PSNR values) make the exact environment non-rebuildable, weakening reproducibility."
resolution: "Authors: add a pinned dependency manifest (requirements.txt or environment.yml) capturing exact versions of all packages."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "README §1 Environment Setup"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No technical bugs found that would prevent the GSRF pipeline from running or that
contradict the code's own intent. The three inference scripts, the Gaussian model,
FLE evaluation, and loss functions are internally consistent on read; I could not
execute them (CUDA submodules + external datasets/weights required), so this is a
finding-free pass on the parts I could statically verify, not a guarantee of runtime
correctness.

## difference

```yaml finding
id: fle-degree-config-vs-paper
category: difference
topic: "model configuration"
title: "All configs use FLE degree 9 (100 coeffs); paper repeatedly claims L=3 (16 coeffs)"
severity: medium
confidence: high
status: finding
file: arguments/configs/rfid/exp1.yaml
line_start: 9
line_end: 9
quote: |
    fle_degree: 9                     # max Fourier-Legendre expansion degree
claim: "Every shipped config (rfid/ble/csi exp1.yaml) sets fle_degree: 9, and gaussian_model.py allocates (max_fle_degree+1)^2 = 100 coefficients per channel (load_from_pcd line 113), with active degree ramped up to 9 during training. The paper states 'Both FLE and SH are implemented with a degree of L = 3, resulting in 16 coefficients each' (§5.4) and the appendix concludes 'We find that L = 3 provides the best balance' (App. Table 4)."
concern: "The released model is run at degree 9 (100 coefficients), not the L=3 (16-coefficient) configuration the paper claims was used and recommends, so the shipped artefact does not match the described/recommended model and the appendix degree-ablation conclusion is inconsistent with the defaults."
resolution: "Authors: confirm which degree produced the reported numbers; if L=3, update the configs to fle_degree: 3, otherwise correct the paper's stated degree and the App. Table 4 recommendation."
cross_refs: ["ablation-code-missing"]
check_script: _audit_code/check_traceability.py
paper_ref: "§5.4 (L=3, 16 coeffs); Appendix FLE Degree, Table 4"
tags: [forensics:git-archaeology, reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: csi-normalization-pre-split
category: methodology
topic: "data splitting / preprocessing leakage"
title: "CSI normalization statistics fitted on the full dataset before train/test split"
severity: low
confidence: medium
status: question
file: scene/csi_dataset.py
line_start: 48
line_end: 57
quote: |
      up_re_mean, up_im_mean = up_re.mean(), up_im.mean()
      up_std = np.sqrt(up_re.var() + up_im.var())
      down_re_mean, down_im_mean = down_re.mean(), down_im.mean()
      down_std = np.sqrt(down_re.var() + down_im.var())

      # normalize complex CSI values
      up_re = (up_re - up_re_mean) / (up_std + 1e-8)
      up_im = (up_im - up_im_mean) / (up_std + 1e-8)
      down_re = (down_re - down_re_mean) / (down_std + 1e-8)
      down_im = (down_im - down_im_mean) / (down_std + 1e-8)
claim: "The uplink/downlink mean and std used to normalize all CSI samples (and to denormalize predictions before the SNR computation in inference_csi.py:60-64) are computed over the entire dataset (all samples) at lines 48-51, before the train/test split at lines 79-88, so test-set statistics enter the normalization of both the encoder inputs and the prediction target."
concern: "Fitting normalization on all data (including the held-out test set) is a mild train-on-test leakage; here the impact is limited because the additive means cancel in the SNR numerator (S - S_hat) and only the global down_std scaling affects the power normalization, but the encoder inputs are still normalized with test-inclusive statistics."
resolution: "Authors: refit normalization statistics on the training split only and re-report CSI SNR, or confirm the effect on the 20.99 dB result is negligible."
cross_refs: []
check_script: _audit_code/check_traceability.py
paper_ref: "§5.2 CSI dataset"
tags: [leakage:L1.2, reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

Note on data splitting (not a finding): RFID/BLE/CSI all use a random split over
transmitter positions for a single-scene reconstruction/interpolation task, which is
the standard NeRF2 evaluation protocol for this problem; there are no repeated-unit or
augmentation issues, so a random per-position split is appropriate here. The
during-training "test" PSNR in `utils/train_utils.py:41-90` is logging only (5 random
test samples) and is not used for checkpoint or hyperparameter selection, so it is not
leakage.

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 5 | high | Baselines, ablations, localization, density experiment, and dependency spec all absent. |
| bug | 0 | - | No intent-contradicting defects found in statically reviewed code. |
| difference | 1 | medium | All configs use FLE degree 9 (100 coeffs) vs paper's claimed L=3 (16 coeffs). |
| methodology | 1 | low | CSI normalization fitted on full dataset before split (question, limited impact). |

### Top take-aways (≤6, by severity × confidence)
1. **[missing] baselines-not-in-repo** — No NeRF2/WRF-GS/R2F2/FIRE code; every headline comparison and speedup number is untraceable. (high / high)
2. **[missing] ablation-code-missing** — Table 1's SH-vs-FLE, phase-off, and no-Fourier-loss variants are not implemented. (high / high)
3. **[difference] fle-degree-config-vs-paper** — Shipped configs run degree 9 (100 coeffs), contradicting the paper's repeated L=3 (16 coeffs) claim and its own recommendation. (medium / high)
4. **[missing] localization-code-missing** — BLE KNN localization (31.40% gain) has no producing script. (medium / high)
5. **[missing] density-experiment-missing** — The "9.8× less training data" data-efficiency study (Fig 6) is not in the repo. (medium / high)
6. **[methodology] csi-normalization-pre-split** — CSI normalization stats computed over all data before the split (limited-impact leakage, filed as question). (low / medium)

### Items that genuinely look fine
- GSRF's own per-task metrics ARE computed by the repo: PSNR/MSE/SSIM/LPIPS (`inference_rfid.py:102-105`), MAE in dBm (`inference_ble.py:50`), and SNR in dB (`inference_csi.py:71-73`), each with per-sample CSVs and summary stats.
- Train/test splitting is a standard random per-transmitter-position split (`scene/dataset_readers.py:68-88`), appropriate for single-scene reconstruction; no augmentation-before-split or repeated-unit leakage.
- The CSI autoencoder/encoder is trained only on the training split (`main_csi.py:52,56`), so no encoder-side train/test leakage.
- During-training test PSNR is logging only and not used for checkpoint/hyperparameter selection (`utils/train_utils.py:41-90`) — not leakage.
- FLE basis math in `utils/fle_utils.py` (associated-Legendre recurrences + complex Fourier azimuthal terms) is internally consistent with the paper's Eq. 9.

### Open questions for the authors
- Which FLE degree produced the reported numbers — the configs' 9 or the paper's L=3? (fle-degree-config-vs-paper)
- Were baseline numbers (NeRF2/WRF-GS/R2F2/FIRE), the ablation rows, the localization result, and the density sweep produced off-repo? If so, can that code or the exact commands be released? (the four `missing` findings)
- Does refitting CSI normalization on the training split only change the 20.99 dB SNR? (csi-normalization-pre-split)
