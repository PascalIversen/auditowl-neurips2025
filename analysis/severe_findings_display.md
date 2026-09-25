## MISSING

### #570 · Exploring Neural Granger Causality with xLSTMs Unveiling Tem

**Table 3 ablations have no runnable path: the shipped driver always builds the xLSTM forecaster and trains it with the α-loss proximal optimiser — no flag or code path selects the plain LSTM (ablation I) or the standard Group-Lasso optimiser (ablation II)**

_confidence: high · topic: missing ablation code_

**Claim:** The driver unconditionally instantiates componentXLSTM and trains it with train_model_ista, which only uses the alpha-loss / proximal optimisation; no config flag or code path selects the plain LSTM/cLSTM forecaster (ablation I) or the standard Group-Lasso optimiser (ablation II). _audit_code/out/artefacts.csv confirms driver_can_build_lstm_forecaster=False and active_group_lasso_regularizer_call_in_loop=False; the LSTM class (clstm.py:35) and the group-lasso regularize() (clstm.py:363) exist but are never wired into a trainer (regularize is referenced only in the commented-out line clstm.py:640).

**Concern:** Table 3 attributes GC-xLSTM's gains to the xLSTM architecture (row I) and the joint optimisation (row II) via two ablations, but neither ablated configuration can be produced by the shipped code, so the central design-justification claims are not reproducible.

**Ask:** Authors: please provide the configs/scripts that run the LSTM-forecaster and Group-Lasso variants used for Table 3, or confirm they were run from off-repo code.

**Evidence:** GC-xLSTM/xlstm_neural_gc.py:81-83 ; clstm.py:35 (LSTM class) and clstm.py:363 (Group-Lasso regularize) exist but are never wired into a trainer · paper: Table 3 rows (I) LSTM/Joint and (II) xLSTM/Group-Lasso · check: _audit_code/check_artefacts.py

### #2818 · Scaling Up Parameter Generation: A Recurrent Diffusion Approach

**Table 2 (ADE20K/COCO) evaluation is a stub that prints 'Code for testing is coming soon!' — neither the evaluation nor the checkpoint-collection code for the two downstream tasks is present and runnable**

_confidence: high · topic: placeholder eval code_

**Claim:** The active test commands for the COCO-detection and ADE20K-segmentation tasks simply echo 'Code for testing is coming soon!' (register.py 55-67); the real commands sit commented-out and call test.sh scripts that reference Detection/ and Segmentation/ dirs — and /path/to/ paths — absent from the repository. [...] neither downtask folder ships a checkpoint/ directory or any train.py, so the source checkpoints cannot be built either.

**Concern:** Every value in Table 2 (mIoU 47.1, mAcc 57.5, mAP Bbox 44.5, mAP Seg 39.6) is unreproducible from the repository: neither the checkpoint-collection nor the evaluation for the detection and segmentation downstream tasks is present and runnable.

**Ask:** Authors: please commit the Detection/ and Segmentation/ evaluation codebases (or pin the exact external repos and commits), the checkpoint-collection/training scripts for both downstream tasks, and replace the placeholder test commands, so the Table 2 numbers can be reproduced.

**Evidence:** dataset/register.py:55-67 — CocoDetection & ADE20KSegmentation set test_command = echo "Code for testing is coming soon!" ; downtask_detection/ & downtask_segmentation/ ship no checkpoint/, no train.py, and a test.sh referencing Detection/, Segmentation/, and /path/to dirs absent from the repo · paper: Table 2 (mIoU 47.1, mAcc 57.5, mAP Bbox 44.5, mAP Seg 39.6) · check: _audit_code/check_repo_facts.py

## MISMATCH

### #4991 · Data Efficient Adaptation in Large Language Models via Conti

**Code implements a single-level (J=1) Haar DWT of the LoRA-A matrix with a learnable gate on the approximation coefficients, then an inverse DWT + MLP — none of the SVD-truncation core-feature estimate (Eq.6-7) or the multi-scale heat-kernel network (Eq.8-10) of §3.2 appears anywhere in src**

_confidence: high · topic: method mismatch_

**Claim:** The 'Wavelet Kernel-based Knowledge Retention' module is implemented as a single-level (J=1) Haar DWT of the lora_A weight matrix with a learnable gate theta on the approximation coefficients cA, followed (forward, 184-196) by an inverse DWT and an MLP. [...] no truncated-SVD core-feature estimate (Eq.6-7) and no multi-scale heat-kernel network (Eq.8-10) appear anywhere in the repo (grep for svd/singular/heat/eckart returns nothing in src).

**Concern:** The paper's central retention mechanism, derived in §3.2 as SVD-truncation denoising filtered by a series of learnable heat kernels at multiple scales, is a different operation from the implemented single-level Haar DWT gate, so the method evaluated is not the method described.

**Ask:** Authors: clarify whether the reported results use the Haar-DWT implementation or the SVD/heat-kernel derivation in §3.2, and reconcile the paper's Eq.7-10 with the released code.

**Evidence:** src/T5_run_wavelet.py:159-164 (inverse DWT + MLP at forward 184-196) ; grep for svd/singular/heat/eckart returns nothing in src · paper: §3.2, Eq. (6)-(10)

### #768 · A High Dimensional Statistical Method for Optimizing Transfe

**Fisher matrix A built from per-batch min-max-normalized gradients, not the empirical Fisher described**

_confidence: high · topic: altered Fisher estimate_

**Claim:** Each layer's per-batch Fisher gradient is min-max rescaled to [-1, 1] before the outer product that accumulates A = Θ^T J Θ, so A is built from normalized gradients rather than the raw empirical-Fisher gradients.

**Concern:** The paper states J(θ0) is the empirical Fisher = mean of raw gradient outer products (Algorithm 1 line 10; Appendix E uses Θ^T J(θ0) Θ); the code instead linearly rescales each layer's gradient to [-1,1] before forming the outer product, an undocumented transformation that changes A=Θ^TJΘ and hence the chosen α*/s*.

**Ask:** Authors: document the min-max normalization and justify why the selection still corresponds to minimizing the theoretical measure (14), or report results using the un-normalized empirical Fisher.

**Evidence:** misc.py:294-306 · paper: Algorithm 1 line 10; Appendix E Eq. (72) · check: _audit_code/check_objective_and_split.py

## TECHNICAL BUG

### #4946 · Gaze VLM Bridging Gaze and VLMs through Attention Regulariza

**KL regularizer does not compute Eq. 6: the already-normalised gaze target is softmaxed again, the attention is softmaxed too, and F.kl_div is called with log_target=True on plain (non-log) probabilities — so its effective target sums to ~257, not 1**

_confidence: high · topic: incorrect KL loss_

**Claim:** The gaze target (target_dist) is already a normalized probability distribution summing to 1, yet it is passed through F.softmax again; the attention is likewise softmaxed; then F.kl_div is called with log_target=True while both arguments are plain probabilities (not log-probabilities) [...] kl_div's first argument is expected to be log-probabilities.

**Concern:** The computed quantity is not the KL divergence DKL(At||H̃t) of Eq. 6 — the double softmax compresses the distributions and log_target=True treats the probability target as if it were a log-probability (its effective target exp(softmax(gaze)) sums to ~257, not 1, per _audit_code/out/kl_divergence.txt), so the regularizer that is the paper's core contribution is mathematically wrong.

**Ask:** Authors: confirm whether the released code matches the experiments; the correct form is F.kl_div(attn.log(), gaze, reduction='sum') with already-normalized distributions and no extra softmax. Re-run with the corrected loss.

**Evidence:** open_flamingo/open_flamingo/train/train_utils_attention.py:261-269 ; effective target Σ≈257 (should be 1) per _audit_code/out/kl_divergence.txt · paper: Eq. (6), Eq. (7) · check: _audit_code/check_kl_divergence.py

### #4991 · Data Efficient Adaptation in Large Language Models via Conti

**Eq. 12 regularization silently never applied: the reg loop guards on isinstance(module, LinearWaveletFilter) against the adapter class, but the modules actually inserted are a separate same-named class from the run script — isinstance is always False, so reg_loss stays 0**

_confidence: high · topic: inert regularizer_

**Claim:** The trainer guards the regularization loop with isinstance(module, LinearWaveletFilter) imported from waveletLoRAAdapter (line 12). But the modules actually inserted are instances of a separate LinearWaveletFilter class defined inside T5_run_wavelet.py:137 / Llama3_run_wavelet.py:139. [...] The two classes are distinct objects, so isinstance is always False, the loop body never runs, and reg_loss stays 0.

**Concern:** The paper's 'Controlled Knowledge Updating' contribution — the λ1‖θ‖^a + λ2‖MLP‖^b term of Eq.12 and the asymmetric a≥b regularization that Table 4 grid-searches — is never present in the loss, so a core advertised component is inert and Table 4 (AA varying 74.8→85.5 with (a,b)) cannot be produced by this code.

**Ask:** Authors: confirm whether reg was active in the reported runs; if so, import the same LinearWaveletFilter class in both trainer and run script (or unify on one definition) and re-verify that reg_loss is non-zero during training.

**Evidence:** src/uie_trainer_lora.py:95-108 (adapter LinearWaveletFilter) vs T5_run_wavelet.py:137 / Llama3_run_wavelet.py:139 (distinct same-named class) · paper: Eq. (12); Table 4 · check: _audit_code/check_reg_isinstance_mismatch.py

## METHODOLOGICAL VAILIDTY

### #4946 · Gaze VLM Bridging Gaze and VLMs through Attention Regulariza

**Headline SBERT scores come from teacher-forced reconstruction: one forward pass over a sequence that already contains the ground-truth answer, argmax of the logits taken as the 'prediction', and SBERT cosine measured between that prediction and the truth decoded from the same sequence — .generate() is never called**

_confidence: high · topic: teacher-forced eval_

**Claim:** At evaluation, input_ids already contains the ground-truth annotation text (data.py:289 builds combined_text with no separate question/prompt). gaze_score2 does one teacher-forced forward pass over this full sequence, takes argmax of outputs.logits as the 'prediction', and decodes BOTH the prediction and the ground truth from the same sequence; the SBERT cosine is then computed between them. [...] No .generate() is used in this path.

**Concern:** Each position's logits are conditioned on the true preceding answer tokens, so the model is scored on copying an answer it was shown rather than generating one — this inflates the absolute 'semantic similarity' scores that constitute every headline number and is not a valid measure of generation quality.

**Ask:** Authors: confirm whether reported Table 1-4 scores used teacher-forced argmax decoding or autoregressive generation; if teacher-forced, re-evaluate with model.generate() over an answer-free prompt and report the gap.

**Evidence:** open_flamingo/open_flamingo/train/train_utils_attention.py:734-739 ; data.py:289 input_ids already contain the answer text · paper: §4 'evaluation methodology is based on semantic similarity scores'; Tables 1-4 · check: _audit_code/check_teacher_forced_eval.py

### #5202 · Graph Neural Network Based Action Ranking for Planning

**Reported 'Best Model' is chosen by highest test-set coverage, not validation loss: main.py scores up to 6 checkpoints per run on the test set and reports the one with the best success_rate_with_monitor (TEST coverage), contradicting the paper's stated validation-loss criterion**

_confidence: medium · topic: test-set leakage_

**Claim:** main.py evaluates up to 6 checkpoints per run on the test set (num_models_to_test=2 for each of all_model_types=['validation','training','combined']) and log_model_metrics then reports, as 'Best Model', the checkpoint with the highest success_rate_with_monitor, which is the test-set coverage (check_selection_on_test.py confirms the active definition selects by 'success_rate_with_monitor (TEST coverage)').

**Concern:** Selecting which checkpoint/run to report by its score on the test set is test-set leakage into model selection and contradicts the paper's stated criterion 'we select the model checkpoint that achieves the lowest loss on the validation set', biasing the reported Coverage/PQR upward.

**Ask:** Authors: confirm whether the Table 2 numbers come from the validation-loss checkpoint or from this best-by-test-coverage selection; if the latter, re-report using only the validation-selected checkpoint (and validation-selected hyperparameters).

**Evidence:** ploi/test_utils.py:391-395 — selects by success_rate_with_monitor (TEST coverage) · paper: Section 3.4: 'we select the model checkpoint that achieves the lowest loss on the validation set for evaluation' · check: _audit_code/check_selection_on_test.py
