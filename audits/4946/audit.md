# Code-repository audit — Gaze-VLM (NeurIPS 2025, paper 4946)

## 1. Summary

The repository (`anupampani/Gaze-VLM`, single commit "Add files via upload") contains a
fork of the OpenFlamingo training codebase plus a `data_obtain/` folder for building the
gaze-augmented Ego4D dataset. The paper's central claim is that a KL-divergence gaze
attention regularizer (Eq. 6–7) improves SBERT "semantic similarity" scores for
egocentric future-event prediction (+~9–11%) and current-activity understanding (+~5–7%)
across **five** VLM architectures (Table 1), with further tables on regularization scale
(Table 2), anticipation horizon (Table 3), OOD EGTEA+ generalization (Table 4), runtime
(Table 5), hallucination reduction (Table 6), and attention-gaze top-10 overlap (§4.8).

I read the paper (`paper.pdf` / `paper_text.txt`) and the repo, focusing on: the
gaze-regularization loss (`open_flamingo/open_flamingo/train/train_utils_attention.py`),
the training driver (`gaze_wAttention.py`), the evaluation driver
(`gaze_evaluation.py` → `gaze_score2`), the dataset/split construction
(`data_obtain/loader_new.py`, `data.py`). I wrote two read-only checks under
`_audit_code/`: `check_kl_divergence.py` (reproduces the repo's KL function and shows it
does not compute the paper's KL) and `check_teacher_forced_eval.py` (documents the
teacher-forced eval data flow). I did **not** run the model end-to-end: there is no
dependency specification, no dataset, no weights, and ~124 hardcoded `insert path` /
`/home/pani3` placeholders, so the code is not executable as shipped.

Headline concerns: (a) the metric driving every headline number is computed via a
**teacher-forced forward pass over a sequence that already contains the ground-truth
answer**, not autoregressive generation; (b) the **train/test split is a random split over
heavily-overlapping sliding-window sequences from the same videos**, putting near-duplicate
windows in both train and test; (c) the **KL regularizer is implemented incorrectly**
(double softmax + `log_target=True` on plain probabilities) so it does not compute Eq. 6;
(d) **no code exists** for 4 of the 5 architectures or for most of the paper's tables.

## 2. Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — OpenFlamingo future/activity scores (BASE/OURS) | `gaze_evaluation.py` → `gaze_score2` (`train_utils_attention.py:719-756`) | code present; metric is teacher-forced (see methodology) | — | PRESENT but metric questionable |
| Table 1 — Modified OpenFlamingo, LaViLa Narrator, InternVL, OpenLLaVA rows | (none) | — | — | MISSING (no train/eval driver for these 4) |
| Table 2 — effect of λ (0/100/1000) | training `reg_value=100` hardcoded (`train_utils_attention.py:297`); no λ=0/1000 variants or sweep | partial | — | MISSING (no sweep / λ as configurable arg) |
| Table 3 — anticipation horizon τa=2s/5s | (none; `sequence_length/prediction_length` are constructor args in `loader_new.py` but no horizon experiment driver) | — | — | MISSING |
| Table 4 — EGTEA+ OOD generalization | (none) | — | — | MISSING (no EGTEA loader/eval) |
| Table 5 — runtime (1.7s vs 2.3s) | (none) | — | — | MISSING (no timing harness) |
| Table 6 — hallucination CI (0.205→0.140) | (none; human evaluation) | — | — | MISSING (no code / human-study artefacts) |
| §4.8 — attention-gaze top-10 overlap (42%→68%) | (none) | — | — | MISSING (no overlap computation) |
| Tables 8–21 (appendix ablations) | mostly (none) | — | — | MISSING (no drivers for singular-vs-aggregated, gaze-as-text, self-attention, gaze-embedded-image, STA mAP, coarse narration, etc.) |
| Eq. 6 KL regularizer DKL(At‖H̃t) | `train_utils_attention.py:225-272` (`KL_divergence`) | not a valid KL (see bug) | ✗ | MISMATCH |
| Dataset construction (GPT-4V captions, occlusion filtering) | `data_obtain/*.py` | scripts present (need API keys, RAFT, raw Ego4D) | — | PRESENT (not runnable as-is) |

## 3. Findings

## missing

```yaml finding
id: missing-multi-arch-and-tables
category: missing
topic: "result traceability / repository completeness"
title: "Code only covers OpenFlamingo; 4 of 5 architectures and most paper tables have no code"
severity: high
confidence: high
status: finding
file: open_flamingo/open_flamingo/train/data.py
line_start: 377
line_end: 377
quote: |
  def preprocess_gaze_attention_internvl(sample, clip_processor, tokenizer, max_images=5, max_tokens=128, input_size=448, max_num=1):
claim: "The repo contains an OpenFlamingo training driver (gaze_wAttention.py) and evaluator (gaze_evaluation.py), plus one InternVL preprocessing helper, but no training/evaluation driver for the Modified OpenFlamingo, LaViLa Narrator, InternVL, or OpenLLaVA rows of Table 1, and no code computing Tables 3-6, §4.8 overlap, or the appendix ablation tables (8-21)."
concern: "The headline claim is generalization across five architectures (Table 1) plus OOD (Table 4), runtime (Table 5), hallucination (Table 6) and alignment (§4.8) results; none of these numbers can be reproduced because the producing code is absent."
resolution: "Authors: please provide the training and evaluation drivers for LaViLa, InternVL, OpenLLaVA, the EGTEA+ OOD evaluation, the runtime measurement, the top-10 attention-gaze overlap computation, and the human-evaluation hallucination protocol."
cross_refs: ["missing-deps-and-data", "teacher-forced-eval"]
paper_ref: "Table 1, Table 4, Table 5, Table 6, §4.8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-deps-and-data
category: missing
topic: "dependencies / data / reproducibility"
title: "No dependency spec, no dataset/weights, ~124 'insert path' placeholders — repo not runnable"
severity: high
confidence: high
status: finding
file: open_flamingo/open_flamingo/train/gaze_evaluation.py
line_start: 121
line_end: 124
quote: |
    checkpoint_path = os.path.join("insert path/open_flamingo/open_flamingo/models/review2/gaze_model.pth")
    if os.path.isfile(checkpoint_path):
        dist.barrier()
        checkpoint = load_checkpoint_eval("insert path/open_flamingo/open_flamingo/models/review2/gaze_model.pth")
claim: "There is no requirements.txt / environment.yml / setup.py anywhere in the repo (the train/README references an environment.yml that does not exist), no shared dataset or download script for the gaze-annotated Ego4D subset, no trained weights, and 124 hardcoded 'insert path' or '/home/pani3' / '/home/anupam' paths across the train/eval scripts."
concern: "The code cannot be installed or executed as shipped, and the dataset and weights underlying every reported number are absent, so no result is reproducible; the README states 'Installation: Instructions and code coming soon'."
resolution: "Authors: please add a pinned dependency file, a dataset access/build path with resolvable Ego4D gaze-subset references, the trained checkpoints, and replace placeholder paths with CLI args."
cross_refs: ["missing-multi-arch-and-tables"]
check_script: _audit_code/check_teacher_forced_eval.py
paper_ref: "README.md; train/README.md line 2 (environment.yml); Appendix 'Training and Evaluation details'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: kl-regularizer-double-softmax
category: bug
topic: "gaze regularization loss"
title: "KL_divergence double-softmaxes inputs and mishandles log_target; does not compute Eq. 6"
severity: high
confidence: high
status: finding
file: open_flamingo/open_flamingo/train/train_utils_attention.py
line_start: 261
line_end: 269
quote: |
        # Apply softmax to the logits
        target_log_probs = F.softmax(target, dim=-1)
        source_log_probs = F.softmax(source, dim=-1)
        
        # Calculate KL divergence for each pair of target and source distributions
        hold = torch.nn.functional.kl_div(source_log_probs, target_log_probs, reduction='sum', log_target=True)
        if hold <0:
            hold =0
        kl_loss += hold
claim: "The gaze target (`target_dist` from calculate_gaze_proportions_batch) is already a normalized probability distribution summing to 1, yet it is passed through F.softmax again; the model attention is likewise softmaxed; then F.kl_div is called with `log_target=True` while both arguments are plain probabilities (not log-probabilities), and `torch.nn.functional.kl_div`'s first argument is expected to be log-probabilities."
concern: "The computed quantity is not the KL divergence DKL(At||H̃t) of Eq. 6 — the double softmax compresses the distributions and `log_target=True` treats the probability target as if it were a log-probability (its effective target exp(softmax(gaze)) sums to ~257, not 1, per _audit_code/out/kl_divergence.txt), so the regularizer that is the paper's core contribution is mathematically wrong."
resolution: "Authors: confirm whether the released code matches the experiments; the correct form is F.kl_div(attn.log(), gaze, reduction='sum') with already-normalized distributions and no extra softmax. Re-run with the corrected loss."
cross_refs: []
check_script: _audit_code/check_kl_divergence.py
paper_ref: "Eq. (6), Eq. (7)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: checkpoint-selected-on-train-loss
category: bug
topic: "model selection"
title: "Best checkpoint chosen by lowest TRAINING loss; validation loader is commented out"
severity: medium
confidence: high
status: finding
file: open_flamingo/open_flamingo/train/gaze_wAttention.py
line_start: 293
line_end: 295
quote: |
        if( trainloss < best_loss):
            best_loss=trainloss
            save_checkpoint_gaze(model, optimizer, scheduler, epoch, "/home/pani3/gaze_model2_10.pth")
claim: "The training driver saves the 'best' checkpoint based on the lowest training loss, and the validation dataloader (lines 205-206) is commented out, so no validation set is used for model selection."
concern: "Selecting the checkpoint by training loss rather than a held-out validation set risks reporting an over-fit model and contradicts the 64/16/20 train/val/test split the data pipeline constructs; the validation split is built but never consumed."
resolution: "Authors: confirm which checkpoint produced the paper numbers and whether model selection used the held-out validation split rather than training loss."
cross_refs: ["sliding-window-random-split-leakage"]
paper_ref: "Appendix 'Training and Evaluation details'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: reg-value-hardcoded
category: difference
topic: "hyperparameters / ablation"
title: "Regularization strength hardcoded to 100; λ=0 and λ=1000 ablation (Table 2) not in code"
severity: low
confidence: high
status: finding
file: open_flamingo/open_flamingo/train/train_utils_attention.py
line_start: 297
line_end: 297
quote: |
    reg_value = 100
claim: "The KL weight is hardcoded to 100 inside train_gaze_attention; it is not exposed as a CLI/config argument, and there is no code path that sets it to 0 or 1000."
concern: "Table 2 reports a λ ablation over {0, 100, 1000}, but only λ=100 is realizable from the shipped code, so the ablation cannot be reproduced without editing the source; the value 100 matches the paper's reported best, which is consistent."
resolution: "Authors: expose λ as an argument and provide the configs used for the λ=0 and λ=1000 rows of Table 2."
cross_refs: ["missing-multi-arch-and-tables"]
paper_ref: "Table 2 (λ=0/100/1000)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

```yaml finding
id: teacher-forced-eval
category: methodology
topic: "evaluation validity"
title: "Headline SBERT scores computed by teacher-forced reconstruction of the ground-truth answer"
severity: high
confidence: high
status: finding
file: open_flamingo/open_flamingo/train/train_utils_attention.py
line_start: 734
line_end: 739
quote: |
                images,overlays,gaze,input_ids, attention_mask, labels = prepare_batch_gaze_attention(batch, tokenizer, device_id,image_token_id,endofchunk_token_id)
                outputs,attn_weights,cosine_sim= model(images,gaze,input_ids, attention_mask, labels)
                logits = outputs.logits 
                token_ids = logits_to_token_ids(logits)
                predicted_text = decode_token_ids_new(tokenizer,token_ids)
                ground_truth = decode_token_ids_new(tokenizer,input_ids)
claim: "At evaluation, `input_ids` already contains the ground-truth annotation text (data.py:289 builds combined_text = '<image>...<image>{annotations_text}<|endofchunk|>{eos}' with no separate question/prompt). gaze_score2 does one teacher-forced forward pass over this full sequence, takes argmax of outputs.logits as the 'prediction' (logits_to_token_ids), and decodes BOTH the prediction and the ground truth from the same sequence; the SBERT cosine is then computed between them. No .generate() is used in this path."
concern: "Each position's logits are conditioned on the true preceding answer tokens, so the model is scored on copying an answer it was shown rather than generating one — this inflates the absolute 'semantic similarity' scores that constitute every headline number and is not a valid measure of generation quality."
resolution: "Authors: confirm whether reported Table 1-4 scores used teacher-forced argmax decoding or autoregressive generation; if teacher-forced, re-evaluate with model.generate() over an answer-free prompt and report the gap."
cross_refs: ["missing-multi-arch-and-tables"]
check_script: _audit_code/check_teacher_forced_eval.py
paper_ref: "§4 'evaluation methodology is based on semantic similarity scores'; Tables 1-4"
tags: [reforms:6, lones:stage-7]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: sliding-window-random-split-leakage
category: methodology
topic: "data splitting / sample independence"
title: "Random split over overlapping sliding-window sequences puts near-duplicate windows in train and test"
severity: high
confidence: high
status: finding
file: data_obtain/loader_new.py
line_start: 70
line_end: 78
quote: |
    def split_data(self, sequences):
        # Split the dataset into train+val (80%) and test (20%)
        train_val, test = train_test_split(sequences, test_size=0.2, random_state=42)
        
        # Now split the 80% of train+val into train (80% of 80%) and val (20% of 80%)
        # This results in a 64% train, 16% val, and 20% test split of the original dataset
        train, val = train_test_split(train_val, test_size=0.2, random_state=42)
        
        return train, val, test
claim: "Sequences are generated with a stride-1 sliding window over each video's 1-fps frames (loader_new.py:56, window i and i+1 share sequence_length-1 frames), and split_data then applies a plain random train_test_split over the pooled sequences with no grouping by video or by time."
concern: "Consecutive windows from the same video are near-duplicates (5 of 5 future frames overlap with neighbors; the same future annotation recurs), so a random split places highly-overlapping train and test windows from the same clip — the test set is not independent of training, inflating the reported scores and the BASE-vs-OURS gain."
resolution: "Authors: use a video-disjoint (group) split or temporally blocked split, quantify the per-video overlap between train and test, and report scores under a leakage-aware split."
cross_refs: ["teacher-forced-eval", "checkpoint-selected-on-train-loss"]
paper_ref: "§4.1 Dataset Construction"
tags: [leakage:L1.1, reforms:6, whalen:pitfall-2]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## 4. Scoreboard

| Category | # findings | Max severity | Note (one line) |
|---|---|---|---|
| missing | 2 | high | No deps/data/weights; only OpenFlamingo covered; Tables 3-6/§4.8/appendix have no code. |
| bug | 2 | high | KL regularizer (Eq. 6) computed wrong (double softmax + log_target); checkpoint picked on train loss. |
| difference | 1 | low | λ hardcoded to 100; Table 2 λ-sweep not reproducible from code. |
| methodology | 2 | high | Teacher-forced answer-reconstruction metric; random split over overlapping sliding windows. |

## 5. Closing lists

**Top take-aways** (≤6, severity × confidence):
1. [methodology] `teacher-forced-eval` — every headline SBERT score is a teacher-forced reconstruction of the ground-truth answer the model is fed, not generation (high/high).
2. [methodology] `sliding-window-random-split-leakage` — random split over stride-1 overlapping windows leaks near-duplicate sequences from the same video into the test set (high/high).
3. [bug] `kl-regularizer-double-softmax` — the gaze KL loss (paper's core contribution, Eq. 6) is implemented incorrectly (double softmax + `log_target=True` on plain probabilities) (high/high).
4. [missing] `missing-multi-arch-and-tables` — only OpenFlamingo has training/eval code; LaViLa/InternVL/OpenLLaVA, EGTEA+ OOD, runtime, hallucination, and top-10-overlap results have no producing code (high/high).
5. [missing] `missing-deps-and-data` — no dependency spec, no dataset, no weights, ~124 `insert path`/`/home/pani3` placeholders; repo is not runnable (high/high).
6. [bug] `checkpoint-selected-on-train-loss` — best checkpoint chosen by training loss; the validation split is built but the val loader is commented out (medium/high).

**Items that genuinely look fine**:
- Gaze patch granularity: `calculate_gaze_proportions_batch` (16×16=256 patches over 224×224) is a valid pixel→patch aggregation consistent with Eq. 5 and a ViT-L/14 grid.
- Seeding: `random_seed` sets torch/numpy/python seeds (gaze_wAttention.py:73-76).
- The reported best λ=100 matches the hardcoded `reg_value=100`, and the SBERT model (`all-MiniLM-L6-v2`, cosine) matches the paper's stated semantic-similarity metric.
- Dataset-construction scripts (`data_obtain/`) implement the described GPT-4V captioning and RAFT optical-flow occlusion filtering (Algorithm 2; ε=20, η=0.60 are present in `gaze_generator3.py`).

**Open questions for the authors**:
- Were the Table 1-4 numbers produced by teacher-forced argmax decoding (`gaze_score2`) or by autoregressive `model.generate()`? (Decides the severity of `teacher-forced-eval`.)
- Was model selection done on the held-out validation split, or on training loss as the shipped driver does?
- Does the released `KL_divergence` match the loss used for the reported experiments, or was a corrected KL used off-repo?
- Will the gaze-annotated Ego4D subset, trained checkpoints, and the per-architecture drivers be released?
