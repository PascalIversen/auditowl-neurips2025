# Code-repository audit — Paper 3759, *Lfp2vec*

## 1. Summary

The repository `tianxiao18__Lfp2vec` is the author code for a self-supervised
LFP region-decoding model (wav2vec 2.0 adapted to local field potential). It
contains training/eval scripts (`script/*.py`), baseline decoders, preprocessing
code for four datasets (Allen, IBL, Neuronexus, Macaque), two analysis notebooks
(`script/experiments.ipynb`, `script/post_processing.ipynb` — which produce the
reported confusion matrices, per-class accuracies, and post-processing numbers),
a conda env spec (`blind_localization.yml`), a `requirements.txt`, and a README.

What I did:
- Read the paper (PDF + text extraction) and mapped every quantitative claim
  (Figs 2–5, abstract numbers, ablation, SWR test) to repo code.
- Traced the headline pipeline: training scripts (`wav2vec_random_init.py`,
  `w2v2_across_session*.py`, `wav2vec_disease.py`, `monkey_w2v2.py`) write
  `*_results.pickle`; the two notebooks consume those pickles to make the figures.
- Ran three deterministic checks under `_audit_code/`:
  - `check_setup_completeness.py` → `out/setup_completeness.json` (README setup
    path, data dirs, weights, dependency files).
  - `check_torch_import.py` → `out/torch_import.json` (scripts that use `torch.`
    without importing torch; cross-checked star-imports to remove false positives).
- Inspected both notebooks cell-by-cell to find the exact metric/post-processing
  computations behind Fig 2/6/7.

Overall: the code is a research-cluster dump. No input data, no preprocessed
pickles, no result pickles, and no trained weights are in the repo; every
training/analysis entry point reads from hardcoded absolute paths
(`/scratch/...`, `/vast/...`). The headline numbers therefore cannot be
recomputed from the repo as shipped. There is also a metric mismatch (paper
reports balanced accuracy / macro-F1; the checked-in figure code computes plain
accuracy) and an undocumented hand-tuned class weighting inside the
post-processing step.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Fig 2a balanced accuracy & macro-F1 (Lfp2vec vs baselines, 3 datasets) | training scripts compute plain `accuracy` (`evaluate.load("accuracy")`); `post_processing.ipynb` cell computes `np.mean(pred==labels)` + per-class accuracy | — | ✗ (metric not the reported balanced-acc/F1) | MISMATCH / MISSING metric (see `metric-balacc-f1-not-computed`) |
| Fig 2a (right) Silhouette 0.576±0.026 (and session/probe scores) | (none found in any `.py` or notebook code cell) | — | — | MISSING (`silhouette-linprobe-missing`) |
| Fig 2a (right) linear-probing accuracy 0.921±0.004 | `wav2vec_random_init.py:332-383` defines a `LinearProber` used *during SSL training*; no script reports the headline 0.921 value | — | — | MISSING (no driver producing the reported value) |
| Fig 2b / Fig 6 confusion matrices | `post_processing.ipynb` cells (confusion_matrix on `*_results.pickle`) | depends on absent pickles | — | code present, inputs MISSING |
| Fig 2c / 3d PCA projections | `blind_localization/data/PCAviz.py`, called in training scripts | — | plotting only | not required by Rule G |
| Fig 2d channel-wise map + post-processing | `post_processing.ipynb` (temporal+spatial smoothing) | uses hand-tuned `class_weights=[1,1,5,2,1]` | ✗ vs paper's πc | DIFFERENCE (`postproc-class-weights`) |
| Fig 2e across-lab zero/one-shot matrix | `w2v2_across_session_pretrained.py`, `monkey_w2v2.py` | depends on absent pickles + hardcoded paths | — | inputs MISSING |
| Fig 3b macaque accuracy & macro-F1 | `monkey_w2v2.py` | plain accuracy stored; balanced-acc/F1 not computed | ✗ | MISMATCH (see `metric-balacc-f1-not-computed`) |
| Fig 4a disease accuracy & F1 | `wav2vec_disease.py` (`compute_metrics`=accuracy) | plain accuracy | ✗ | MISMATCH + see `disease-session-level-label` |
| Fig 5 ablation (#pretraining trials × init) | `--ssl`/`--rand_init` flags toggle init; no script sweeps the number of pretraining trials (6k…400k) | — | — | MISSING driver (`ablation-trialcount-missing`) |
| Suppl. SWR t-test (p=0.158) | `experiments.ipynb` (`ttest_ind`, n=12 trials) | p=0.158 (not significant; reported as exploratory) | n/a | present, not a headline claim |
| Pretrained Lfp2vec weights (abstract + checklist Q5/Q13) | (none) | — | — | MISSING (`missing-pretrained-weights`) |
| All training inputs (raw/lfp/spectrogram `.pkl`) | (none; hardcoded `/scratch//vast` paths) | — | — | MISSING (`missing-data-pickles`) |

Deterministic backups: `_audit_code/out/setup_completeness.json`,
`_audit_code/out/torch_import.json`.

## 3. Findings

### missing

```yaml finding
id: missing-data-pickles
category: missing
topic: "result traceability / data availability"
title: "No input data in repo; all entry points read hardcoded /scratch & /vast paths"
severity: high
confidence: high
status: finding
file: script/wav2vec_random_init.py
line_start: 874
line_end: 876
quote: |
    elif data == "Neuronexus":
        sessions_list = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
        pickle_path = f'/scratch/th3129/region_decoding/data/Neuronexus/lfp'
claim: "The headline training/eval scripts load preprocessed data from absolute cluster paths (e.g. /scratch/th3129/..., /vast/th3129/..., /scratch/cl7201/...); _audit_code/check_setup_completeness.py confirms 0 .pkl/.pickle data files and none of the README-promised data/<DS>/{lfp,raw,spectrogram} directories exist in the repo."
concern: "None of the figures' underlying values can be recomputed from the repo as shipped: the preprocessed inputs the scripts read are absent and the paths are non-portable."
resolution: "Authors: ship the preprocessed pickles (or a fetch+preprocess command that writes to repo-relative paths) so the *_results.pickle files consumed by the notebooks can be regenerated."
cross_refs: ["hardcoded-cluster-paths", "missing-pretrained-weights"]
check_script: _audit_code/check_setup_completeness.py
paper_ref: "Checklist Q5 'open access to data and code'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-pretrained-weights
category: missing
topic: "expected code completeness"
title: "Pretrained Lfp2vec weights promised in abstract/checklist but absent"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  "By releasing our code and pretrained weights, we aim to spur further work on domain adaptation, spatiotemporal regularization, and ethical deployment of ML methods for neural data."
claim: "The abstract, Broader Impact, and checklist answers Q5/Q13 promise pretrained Lfp2vec checkpoints; _audit_code/check_setup_completeness.py finds 0 weight files (*.bin/*.safetensors/*.pt/*.pth/*.ckpt) anywhere in the repo, and scripts re-train from facebook/wav2vec2-base each run."
concern: "Pretraining (50 epochs SSL on large LFP corpora) is expensive and the run is nondeterministic, so absence of released weights blocks reproduction of the zero-shot/transfer results without large compute."
resolution: "Authors: release the pretrained Lfp2vec checkpoints referenced in the abstract and checklist."
cross_refs: ["missing-data-pickles"]
check_script: _audit_code/check_setup_completeness.py
paper_ref: "Broader Impact; Checklist Q5, Q13"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: silhouette-linprobe-missing
category: missing
topic: "result traceability"
title: "No code computes the Fig 2a Silhouette (0.576) or linear-probing (0.921) values"
severity: medium
confidence: medium
status: finding
file: paper.pdf
quote: |
  "Lfp2vec achieved a markedly higher Silhouette score for region clustering (0.576±0.026) than the spectrogram (–0.054±0.015), SimCLR (–0.062±0.050), and BrainBERT (0.146±0.026) baselines, while showing no meaningful clustering by session (–0.053±0.009) or probe identity (–0.266±0.006)."
claim: "The reported Silhouette scores and the 0.921±0.004 linear-probing accuracy in Fig 2a (right) have no producing script: a repo-wide search finds `silhouette` only inside embedded plotly/d3 output JSON in experiments.ipynb (no code cell calls silhouette_score), and the only LinearProber (wav2vec_random_init.py) is a per-epoch SSL diagnostic that does not report the headline value."
concern: "A central quantitative claim distinguishing Lfp2vec from baselines (anatomy clusters but session/probe do not) is not traceable to any computation in the repo."
resolution: "Authors: provide the script that computes the per-model Silhouette scores and linear-probing accuracies with their error bars."
cross_refs: []
paper_ref: "Section 6.1; Figure 2a right"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablation-trialcount-missing
category: missing
topic: "ablations"
title: "Fig 5 ablation over number of pretraining trials has no driver script"
severity: medium
confidence: medium
status: question
file: paper.pdf
quote: |
  "on the Neuronexus dataset, the audio-initialized model with only a small amount of LFP pre-training (6k trials) already matches the performance of a randomly-initialized model trained on over 400k trials."
claim: "The repo exposes --ssl and --rand_init flags (init choice) but no script varies the amount of self-supervised pretraining data (6k…400k trials); the curve in Fig 5 has no producing driver I could locate."
concern: "The ablation supporting the paper's central 'audio init + in-domain SSL' claim cannot be reproduced from the repo without the (absent) sweep harness."
resolution: "Authors: point to or add the script that sweeps the number of unlabeled pretraining trials and records decoding accuracy."
cross_refs: ["missing-data-pickles"]
paper_ref: "Section 6.5; Figure 5"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

```yaml finding
id: readme-setup-path-broken
category: missing
topic: "expected code completeness / reproducibility"
title: "README setup commands reference a nonexistent env file and preprocessing dir; no pip step"
severity: medium
confidence: high
status: finding
file: README.md
line_start: 12
line_end: 18
quote: |
  conda env create -f environment.yml
  conda activate lfp2vec
  ```

  ## Data

  The repository expects LFP data and corresponding labels in the following structure, stored as `.pkl` files. And the preprocessing script is stored in script/dataset_preprocessing/
claim: "README instructs `conda env create -f environment.yml`, but no environment.yml exists (only blind_localization.yml); the README never instructs `pip install -r requirements.txt`, yet transformers/evaluate/datasets/accelerate are absent from blind_localization.yml (verified by _audit_code/check_setup_completeness.py); and the referenced script/dataset_preprocessing/ directory does not exist."
concern: "The documented installation/preprocessing path fails as written, so a reader cannot rebuild the environment or locate preprocessing code from the README."
resolution: "Authors: fix the env filename, add the pip-install step (or merge HF deps into the conda yml), and correct the preprocessing-directory path."
cross_refs: ["missing-data-pickles"]
check_script: _audit_code/check_setup_completeness.py
paper_ref: "README"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: missing-torch-import
category: bug
topic: "runtime correctness"
title: "w2v2_across_session.py calls torch.* without importing torch (NameError)"
severity: low
confidence: high
status: finding
file: script/w2v2_across_session.py
line_start: 233
line_end: 235
quote: |
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
claim: "torch is used at lines 233-234 but is never imported in this module and there is no `from ... import *` that would inject it (verified by _audit_code/check_torch_import.py, which excludes the spectrogram/simclr scripts that receive torch via star-imports of modules that `import torch`)."
concern: "Running this across-session script raises NameError: name 'torch' is not defined the first time it reaches the device-setup line, so it cannot run as shipped."
resolution: "Add `import torch` to script/w2v2_across_session.py."
cross_refs: []
check_script: _audit_code/check_torch_import.py
paper_ref: "n/a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-cluster-paths
category: bug
topic: "portability"
title: "Result-producing notebooks hardcode absolute /scratch result/data paths"
severity: medium
confidence: high
status: finding
file: script/post_processing.ipynb
line_start: 185
line_end: 185
quote: |
    "    walk_dir = '/scratch/mkp6112/LFP/region_decoding/results/Neuronexus/spectrogram/wave2vec2/across_session/'\n",
claim: "The notebook that produces the confusion matrices, per-class accuracies, and post-processing accuracy reads result pickles exclusively from absolute cluster paths (/scratch/mkp6112/..., /scratch/th3129/...) for all three data modes; ripple .mat files and raw pickles are likewise absolute."
concern: "Even if a user regenerates result pickles, the figure code points at another user's filesystem and will not find them without editing every path."
resolution: "Parameterise the result/data directories (CLI arg or repo-relative default)."
cross_refs: ["missing-data-pickles"]
paper_ref: "Figures 2b, 2d, 6, 7"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: metric-balacc-f1-not-computed
category: difference
topic: "evaluation consistency"
title: "Paper reports balanced accuracy & macro-F1; checked-in figure code reports plain accuracy"
severity: medium
confidence: medium
status: finding
file: script/post_processing.ipynb
line_start: 215
line_end: 217
quote: |
    "    logits, labels, trials, channels, acc = load_results(walk_dir, session)\n",
    "    pred = np.argmax(logits, axis=1)\n",
    "    results[session] = np.mean(pred == labels)\n",
claim: "Section 5.2 says 'We report both balanced accuracy and macro F1'; the wav2vec2 training scripts compute plain accuracy via evaluate.load(\"accuracy\") and store test_acc, and post_processing.ipynb computes np.mean(pred==labels) plus per-class accuracy — not balanced accuracy or macro-F1. Balanced-acc/F1 helpers exist in the repo (utils.py, decoder.py, visualizer.py) but are wired only to the non-wav2vec baseline decoders, not to the Lfp2vec result pickles."
concern: "The headline Fig 2a/3b/4a metric the figures plot for Lfp2vec is not produced by any checked-in script consuming the model's result pickles, so the reported balanced-acc/F1 values are not reproducible from the repo and may differ from plain accuracy on these imbalanced classes."
resolution: "Authors: provide the exact script that converts the stored logits/labels into the reported balanced accuracy and macro-F1 (with error bars) for the wav2vec2 models."
cross_refs: ["missing-data-pickles"]
paper_ref: "Section 5.2; Figures 2a, 3b, 4a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: postproc-class-weights
category: difference
topic: "post-processing"
title: "Post-processing applies an undocumented hand-tuned class weighting [1,1,5,2,1]"
severity: medium
confidence: high
status: finding
file: script/post_processing.ipynb
line_start: 402
line_end: 410
quote: |
    "class_weights = [1,1,5,2,1]\n",
    "\n",
    "for ch in unique_channels:\n",
    "    logits_of_interest = logits[channels == ch]\n",
    "\n",
    "    # average logits from all trials of same channel with weights, to get the max logits\n",
    "    average_logits = np.mean(logits_of_interest, axis=0)\n",
    "    weighted_average_logits = average_logits * class_weights\n",
    "    majority_vote = np.argmax(weighted_average_logits)\n",
claim: "The post-processing (Fig 2d / Fig 7) multiplies per-channel averaged logits by a fixed hand-set vector [1,1,5,2,1] (up-weighting CA2 ×5, CA3 ×2) before argmax."
concern: "The paper (§4.4 / Appendix D) describes the prior πc as 'uniform or estimated empirically from the training distribution', not an ad-hoc per-class weight; this hand-tuned weighting boosts exactly the underrepresented CA2/CA3 classes the paper highlights as improved, and is set on the same data being scored, so the post-processing improvement may be partly hand-fitting rather than the described principled smoothing."
resolution: "Authors: clarify how [1,1,5,2,1] was chosen (and on which split), or replace it with the uniform/empirical prior described in the paper and re-report post-processing accuracy."
cross_refs: []
paper_ref: "Section 4.4; Appendix D"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

```yaml finding
id: disease-session-level-label
category: methodology
topic: "sample independence / evaluation validity"
title: "Disease classification scored per-trial over n=7 session-level labels (1 healthy mouse in test)"
severity: medium
confidence: medium
status: finding
file: script/wav2vec_disease.py
line_start: 403
line_end: 404
quote: |
  def session_to_disease(session):
      return int("AD" in session)
claim: "The AD/healthy label is a constant function of the session name; with 7 sessions (5 AD, 2 healthy) the test set is one AD + one healthy session (test_sess = random.choice(AD) + random.choice(healthy)), and accuracy/F1 are computed over every channel×trial in those sessions as if independent."
concern: "All channels/trials within a session share one label and strong within-session correlations, so per-trial scoring inflates the effective n and the disease claim rests on a single held-out healthy mouse — the metric measures session discriminability, not generalizable AD vs control classification."
resolution: "Authors: report subject-level (per-session) performance with leave-one-subject-out across all 7 mice, and state the number of independent animals behind the Fig 4a error bars."
cross_refs: []
paper_ref: "Section 6.4; Figure 4a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                          |
|-------------|------------|--------------|----------------------------------------------------------|
| missing     | 5          | high         | No data/weights in repo; headline metric & ablation drivers absent |
| bug         | 2          | medium       | Missing torch import; figure notebooks hardcode /scratch paths     |
| difference  | 2          | medium       | Plain accuracy vs reported balanced-acc/F1; undocumented class weights |
| methodology | 1          | medium       | Disease task scored per-trial over session-level labels (n=7 mice) |

### Top take-aways (≤6, ranked)
1. **[missing] missing-data-pickles** (high/high): no input data in repo; every entry point reads absolute `/scratch//vast` paths — headline numbers not recomputable as shipped.
2. **[missing] missing-pretrained-weights** (medium/high): pretrained Lfp2vec weights promised in abstract & checklist are absent; retraining is expensive/nondeterministic.
3. **[difference] metric-balacc-f1-not-computed** (medium/medium): paper reports balanced accuracy & macro-F1 but the checked-in figure code computes plain accuracy on the model result pickles.
4. **[difference] postproc-class-weights** (medium/high): post-processing uses a hand-tuned `[1,1,5,2,1]` class weighting that up-weights the very CA2/CA3 classes claimed as improved, not the πc described in the paper.
5. **[missing] silhouette-linprobe-missing** (medium/medium): the Silhouette 0.576 and linear-probing 0.921 values in Fig 2a have no producing code.
6. **[methodology] disease-session-level-label** (medium/medium): disease accuracy/F1 scored per-trial over 7 session-level labels with one healthy mouse in test.

### Items that genuinely look fine
- **Across-session split is not leaky.** In `wav2vec_random_init.py` the test set comes from a *held-out session* and test trials use a disjoint trial-index set (`test_tr_idx`) from train/val trials; train/val sessions are disjoint from the test session. No train/test channel or session overlap.
- **Model architecture matches the paper.** `results/config.json` and the inline `w2v2_config` match Tables 2–3 (7-layer CNN, kernels (10,3,3,3,3,2,2), strides (5,2,2,2,2,2,2), 12-layer/12-head transformer, 2×320 codebooks, lr 1e-5 pretrain / 3e-5 finetune, batch 32, grad-accum 4, warmup 0.1).
- **Dependencies are pinned in `blind_localization.yml`** (python 3.11.9, torch 2.4.0, scikit-learn 1.5.1, etc.) even though the README points at the wrong filename.
- **Public-data fetch code exists** (`data/Allen/data_download.py` via AllenSDK; `script/ibl_preprocessing/downloader.py` via the IBL ONE API), so the public datasets are in principle obtainable.

### Open questions for the authors
- Where is the exact script that turns stored logits/labels into the reported balanced accuracy and macro-F1 with error bars? (`metric-balacc-f1-not-computed`)
- How was the post-processing weight vector `[1,1,5,2,1]` chosen, and on which split? (`postproc-class-weights`)
- Which script produces the Fig 5 pretraining-trial-count curve? (`ablation-trialcount-missing`, filed as a question.)
