# Cross-model verdict comparison (Claude vs Codex/GPT)

- Findings joined: **610** across 87 papers
- Verdict agreement: **95.25%**, Cohen's κ = **0.0268**
- claude: 607 keep / 2 lowered / 1 reject (1 cannot-verify) → strictness 0.49%
- codex_stage1: 563 keep / 14 lowered / 33 reject (0 cannot-verify) → strictness 7.7%
- codex_final: 583 keep / 15 lowered / 12 reject (0 cannot-verify) → strictness 4.43%

## Agreement matrix (rows: Claude, cols: Codex final)

| | keep | lowered | reject |
|---|---|---|---|
| **keep** | 581 | 15 | 11 |
| **lowered** | 1 | 0 | 1 |
| **reject** | 1 | 0 | 0 |

## Category breakdown

| value | n | agree % | Claude strict % | Codex strict % |
|---|---|---|---|---|
| bug | 98 | 96.94 | 0.0 | 3.06 |
| difference | 151 | 94.04 | 1.32 | 4.64 |
| methodology | 52 | 90.38 | 0.0 | 9.62 |
| missing | 309 | 96.12 | 0.32 | 3.88 |

## Severity breakdown

| value | n | agree % | Claude strict % | Codex strict % |
|---|---|---|---|---|
| high | 141 | 97.87 | 0.0 | 2.13 |
| low | 222 | 93.69 | 0.9 | 5.41 |
| medium | 247 | 95.14 | 0.4 | 4.86 |

## Confidence breakdown

| value | n | agree % | Claude strict % | Codex strict % |
|---|---|---|---|---|
| high | 484 | 97.73 | 0.21 | 2.07 |
| low | 8 | 75.0 | 0.0 | 25.0 |
| medium | 118 | 86.44 | 1.69 | 12.71 |

## Disagreements (29)

### 1339/stage2-proj-mlp-shape-mismatch
- **Stage-2 video projection is [200,200] but Stage-1 config is [512,200] — strict=False may drop it** _bug · severity medium · confidence low_
- Claude: **keep** — video_encoder_inference.yaml lines 3-7 match the quote (mlp_layers: [200,200]); train_config.yaml confirms Stage-1 uses [512,200]; video_encoder.py line 43 confirms strict=False; the shape mismatch and silent-drop risk are real, consistent with 'question' status and low confidence.
- Codex: **reject** (escalated) — Although Stage 2 uses mlp_layers=[200,200] while Stage 1 uses [512,200], PyTorch load_state_dict(strict=False) raises RuntimeError for same-key tensor shape mismatches instead of silently skipping those projection weights, so the claimed random-initialized inference path is false.

### 1402/qa-nshots-default
- **QA default --n_shots=3 contradicts paper's 5 retrieved records for QA Agent** _difference · severity low · confidence medium_
- Claude: **keep** — QA/main.py line 33 confirmed: --n_shots default is 3, contradicting the paper's claimed 5 for QA Agent.
- Codex: **lowered** (escalated) — QA/main.py defaults n_shots to 3 and uses it as the retrieval cap, but the paper is internally inconsistent because Section 5.1 specifies five QA records while Appendix B.2 specifies three, matching the code.

### 1576/numpy-random-integers-removed
- **np.random.random_integers used but removed in the pinned numpy 2.2.6** _bug · severity medium · confidence medium_
- Claude: **keep** — Code at line 151 exactly matches the quote; check_runtime_blockers.py confirms the function exists under numpy 1.26.4 with a DeprecationWarning, and the numpy 2.0 removal is a documented fact; medium confidence is appropriate since testing under numpy 2.2.6 was not possible in the local environment.
- Codex: **reject** (escalated) — NumPy 2.2 explicitly retains and documents np.random.random_integers as deprecated, so the pinned NumPy 2.2.6 environment does not fail with the claimed removal-based AttributeError.

### 1717/missing-saliency-baseline
- **DeepGaze saliency-integration baseline absent from repo (Tables 1/3/5)** _missing · severity medium · confidence high_
- Claude: **keep** — paper_text.txt lines 234-236 confirm the DeepGaze saliency baseline is described; a repo-wide grep found no .py file containing 'saliency', 'deepgaze', or 'yolo', confirming the code is absent.
- Codex: **lowered** (escalated) — The paper reports the DeepGaze saliency-integration baseline in Table 1 and the repository source scan finds no DeepGaze or saliency implementation, but Tables 3 and 5 contain no saliency row, confirming the first-pass scope correction.

### 1764/value-label-player-perspective
- **Training value labels use player-0 return while model/solver values are current-player relative** _methodology · severity medium · confidence low_
- Claude: **keep** — Quote confirmed at state_counter.py lines 147-152 (player_return(0)); value_prediction.py:38 confirmed uses state.observation_tensor() (current-player perspective); solver_calc.py:81 confirmed applies (1-2*current_player()) conversion — the sign-convention asymmetry between train labels and model output is real and unresolved, consistent with the low-confidence question status.
- Codex: **lowered** (escalated) — The cited code proves that player-0 labels are compared without a local sign correction, but the audited repository does not contain the pinned OpenSpiel 1.5 training implementation needed to prove the model output's player perspective.

### 1829/multiseed-std-harness-absent
- **Reported ±std values have no multi-seed harness; released scripts run a single fixed seed** _missing · severity low · confidence medium_
- Claude: **keep** — run.py lines 467-471 exactly match the quoted single fixed-seed block (seed 2023), confirming no multi-seed loop is present in the released code.
- Codex: **reject** (escalated) — Thirty-three released OLinear/OLinear-C shell scripts use --itr 7, robust scripts pair it with --fix_seed 0, and run.py iterates over range(args.itr), directly contradicting the single fixed-seed-run claim.

### 1867/trained-models-off-repo-dropbox
- **Trained models for reproduction hosted on an external Dropbox link, not in repo** _missing · severity low · confidence medium_
- Claude: **keep** — README fetched and confirms the repo requires downloading two model folders (models/ and benchmarks/models/) from a Dropbox link before running benchmarks; quote is a faithful paraphrase as auditor noted.
- Codex: **reject** (escalated) — The underlying README supports the claim, but the supplied quote is a paraphrase not present at the cited location, which fails the required quote-match check.

### 1867/deps-unpinned
- **requirements.txt unpinned except pyrosetta — environment not reconstructable** _missing · severity low · confidence high_
- Claude: **keep** — requirements.txt fetched: 14 packages listed (gym, gymnasium, imageio, matplotlib, numpy, opencv_python, pandas, pyrosetta==2023.12+release.1d87148, sb3_contrib, scipy, Shapely, stable_baselines3, torch, torchdiffeq); pyrosetta is the only pinned entry, confirming the core claim; quote is a condensed subset as noted.
- Codex: **reject** (escalated) — The requirements file supports the unpinned-dependency claim, but the supplied quote omits intervening packages and is not present contiguously at the cited location.

### 1867/ttest-near-zero-baseline-std
- **Sub-0.001 across-seed std for several baselines questions 10-seed independence** _methodology · severity medium · confidence medium_
- Claude: **keep** — analysis.ipynb confirms the quoted code with ddpg_mol using scale=0.001; CSV confirms DDPG molecular dynamics std rounds to 0.000, TQC to 0.001, S-DDPG to 0.001 — near-zero variance is present in the source data, making the concern valid.
- Codex: **lowered** (escalated) — The notebook and reproduction script confirm a 0.001 scale and the stated t-test result, while Table 4 reports values as exactly 0.001 rather than sub-0.001 and does not establish seed non-independence, so a low-severity question is appropriate.

### 2194/trace-timing-missing
- **No producer for Fig. 4 time/accuracy sweep or the peak-memory/runtime numbers** _missing · severity medium · confidence high_
- Claude: **keep** — Paper quote confirmed at paper_text.txt lines 810-811; traceability script confirms peak_mem_code=false; direct grep finds no max_memory_allocated in src/*.py; ldm_time_measurement exists but measures one pass only, no sweep or reporting script.
- Codex: **lowered** (escalated) — The paper values and Fig. 4 producer are absent and ldm_time_measurement only times image generation, but bundled upstream components contain generic peak-memory hooks, so the literal repository-wide absence claim remains overstated.

### 2212/reliability-text-vision-absent
- **Text/vision resampling-reliability numbers (§6.4) have no computing code** _missing · severity low · confidence medium_
- Claude: **keep** — experiments/ contains only audio_reliability/; repo-wide grep for 'reliab' and 'resamp' in .py and .sh files returned only audio_reliability/3_test_reliability.sh and unrelated mentions in run_text_bon.py; no text/vision reliability harness exists.
- Codex: **reject** (escalated) — The claim is materially false because bon/attacks/run_text_bon.py implements configurable resampling and classification of successful text prompts via check_prompt_reliability, even though the scoped search found no analogous vision harness.

### 2279/ode-kl-coeff
- **ODE KL-basis coefficient is sqrt(2*pi) in code vs sqrt(2)/pi in paper** _difference · severity low · confidence medium_
- Claude: **lowered** — Opus re-check confirms Sonnet's lowered verdict against the cited file ODE_example/main_ODE.m. Line 24 reads 'V_basis(:,j) = sqrt(2*pi)*sin(j*pi*xx)'' (sqrt(2*pi) ~ 2.507) versus the paper's sqrt(2)/pi ~ 0.450 — a real ~5.6x amplitude mismatch, confirmed at the cited lines. However the discrepancy is latent: the regeneration block at lines 36-38 (m_ref = V_basis*sqrt(D)*randn(...); save('m_ref.mat',...)) is commented out, line 41 loads the pre-saved load('m_ref.mat','m_ref'), and m_ref.mat ships in the ODE_example directory. So in normal use the published Figs 12-13 use the saved reference and are unaffected unless a user manually re-enables regeneration. Severity stays low and confidence is lowered from medium to low to reflect the latent/no-impact nature.
- Codex: **keep** — The cited code constructs the KL basis with sqrt(2*pi), while the paper formula uses sqrt(2)/pi, and the surrounding code confirms the finding's stated limitation that the saved m_ref.mat is loaded unless regeneration is uncommented.

### 2402/baseline-code-absent
- **No code for any baseline (SignGAN, SignGen, MoMP, ControlNet, AnimateAnyone)** _missing · severity medium · confidence high_
- Claude: **keep** — Repo-wide grep for SignGAN, SignGen, MoMP, AnimateAnyone in .py files returned only models/controlnet.py (SignViP's own ControlNet module, not a baseline harness), confirming no baseline implementation or evaluation harness is present.
- Codex: **lowered** (escalated) — The repository lacks baseline-specific implementations, configurations, or evaluation invocations for the compared methods, but it does contain substantial generic ControlNet implementation code, so the original categorical claim is overbroad.

### 2585/no-deps-no-readme-no-results
- **Supplement ships no requirements file, no README, and no committed result data** _missing · severity low · confidence high_
- Claude: **keep** — plot.py lines 9-12 match the quote exactly; find returns no requirements, README, .pkl, .npy, or .png files anywhere under code/supplement/, fully corroborating the absent-files claim.
- Codex: **lowered** (escalated) — The cited plot load, the absence of dependency documentation, a README, and committed result artifacts, and the failing save-directory check support the core concern, but the supplement also contains a workspace file and compiled .pyc files, so the claim that it contains only .py source is false.

### 261/auto-pass-no-mp4-inflates-completion
- **Rule-mode aggregation auto-counts trajectories with no recording as 'unsafe-completed'** _difference · severity medium · confidence medium_
- Claude: **keep** — Lines 110 and 117 confirm auto_pass_if_no_mp4=True calls; lines 72-76 confirm that a folder scoring 0 with no .mp4 file is overridden to 1, exactly as claimed.
- Codex: **reject** (escalated) — The auto-pass branch exists, but both rule and both modes call a function that raises NameError at its first os.listdir use because os (and later glob) is never imported, so the alleged inflation path cannot execute in the checked code.

### 261/judge-and-agents-closed-api
- **USRs depend on commercial closed-API agents and a GPT-4o LLM-judge (nondeterministic, undocumented training data)** _methodology · severity low · confidence high_
- Claude: **keep** — safety_evaluation.py lines 1061-1073 confirm the exact quoted GPT-4o model list and OpenAI client instantiation; the claim that no model snapshot is pinned and the judge uses a closed commercial API is factually supported.
- Codex: **lowered** (escalated) — The code and paper support a GPT-4o/OpenAI judge default and evaluation of seven GPT/Gemini/Claude agents, but they do not directly establish rolling-endpoint nondeterminism or undocumented training data, so medium confidence remains appropriate.

### 2818/readme-generalization-script-commented-out
- **README launches a generalization script whose training loop is fully commented out** _difference · severity low · confidence high_
- Claude: **keep** — workspace/classinput/generalization.py lines 219-223 exactly match the quote: the entire `__main__` block including `train()` is commented out, confirming this file trains nothing when executed.
- Codex: **reject** (escalated) — The README explicitly launches workspace/condition/generalization.py, whose training loop and __main__ block are active, so the cited commented-out workspace/classinput/generalization.py does not support the claimed README reproduction hazard.

### 2818/condition-only-head-tokens
- **Unseen-task condition is injected only into the last 8 tokens (the classifier head)** _methodology · severity high · confidence high_
- Claude: **keep** — model/__init__.py lines 176-191 confirm the gate tensor is zero everywhere except the last 8 positions, and `_to_condition` multiplies the linear-projected condition by this gate, so the task condition is zeroed out for all backbone tokens.
- Codex: **lowered** (escalated) — The condition is gated to the final eight tokens, but the cache maps those tokens to norm.bias plus head parameters rather than only the classifier head, and the documented generators disable rather than randomly sample the permutation state.

### 3101/table1-label-mismatch-mcca-vs-dsvd
- **KL_table latex emits mCCA column, not the max dSVD the Table 1 caption describes** _difference · severity low · confidence medium_
- Claude: **keep** — Code at lines 162-168 matches the quote exactly; the auditor's own concern acknowledges the emitted columns are consistent with the paper (mCCA_f and max_d_rep_f match paper's mCCA and max dSVD columns for f), making this a legitimate open question rather than a confirmed discrepancy — status 'question' and medium confidence are appropriate.
- Codex: **reject** (escalated) — The paper explicitly defines Table 1's mCCA and maximum dSVD as embedding metrics over the dLLV input and label sets, and the cited code emits those as mCCA_f and max_d_rep_f with reproduced values matching the table, so no mismatch is established.

### 3457/hf-backend-llava-only
- **HF and vLLM backends only support LLaVA; MiniGPT-4, ShareGPT4V, Qwen-VL not implemented** _missing · severity medium · confidence high_
- Claude: **keep** — hf.py lines 30–35 match the quote exactly, confirming LlavaForConditionalGeneration is hardcoded with no branching for other architectures.
- Codex: **lowered** (escalated) — The HF activation and steering path hardcodes LlavaForConditionalGeneration and the public model map exposes only LLaVA checkpoints, but an unregistered generation-only QwenVL vLLM class exists, so the core reproducibility concern holds while the claim that Qwen-VL is wholly unimplemented is overstated.

### 3715/smoothing-n-mismatch
- **Latent-MNIST figures shipped with n=1000 smoothing samples; comment says paper used 50000** _difference · severity low · confidence high_
- Claude: **reject** — Confirmed: cited code (mnist_KDE_vs_gaussian.ipynb line 671, the latent-MNIST smoothing call) ships n=1000 with #n=50000 only as a commented higher-accuracy option ('increase n for better smoothing accuracy'); paper_text.txt line 4102 (App. G.2, the latent-MNIST Fig 6/7 experiment) states 'we use 1000 smoothing samples at each generation step', so n=1000 MATCHES the paper. The 50,000 value (line 4170) belongs to App. G.3's synthetic-image-manifold pixel-space Gaussian smoothing, a different experiment, so there is no paper-vs-code mismatch at the cited location.
- Codex: **keep** — The cited MNIST notebook cells use get_smoothed_fn(..., n=1000) with an inline n=50000 comment, while the paper states 50,000 isotropic smoothing samples for the MNIST experiments.

### 3764/app-undefined-name-and-missing-import
- **app.py uses StableDiffusion3Pipeline without import and undefined model_choice** _bug · severity low · confidence high_
- Claude: **keep** — Confirmed: app.py line 6 imports only FluxPipeline (no StableDiffusion3Pipeline import), line 98 uses StableDiffusion3Pipeline directly (NameError on SD3 branch), line 108 references model_choice (a Gradio widget, not the function parameter model_type), and hardcoded paths to /home/notebook/data/personal/S9055029/hf/ are present at lines 96 and 98.
- Codex: **lowered** (escalated) — The SD3 branch uses an unimported StableDiffusion3Pipeline and hardcoded checkpoint paths, but model_choice is a defined module global, the UI exposes only FLUX, and load failures are caught rather than raised.

### 376/pde-guidance-disabled-in-notebook
- **Released SH eval uses no-guidance sampler; the PDE/observation-guided sampler (Eq. 3) is commented out** _difference · severity medium · confidence medium_
- Claude: **keep** — The notebook cell confirms the ddim_guided_sample_full_sh call is commented out and ddim_sample_from_reconstructed_sh is the active call, exactly matching the quoted lines.
- Codex: **reject** (escalated) — Although the quoted calls occur at raw notebook lines 519-520, the active ddim_sample_from_reconstructed_sh method explicitly computes and applies observation and PDE gradients with nonzero configured zeta_obs and zeta_pde, so it is not a no-guidance sampler.

### 4828/reconstruction-perceptual-loss-impl
- **Paper's reconstruction perceptual loss L_P is implemented as LaMa ResNet-PL, not LPIPS** _difference · severity low · confidence medium_
- Claude: **keep** — Quote at configs/train_inpainting.yaml lines 48-62 matches exactly; resnet_pl weight 30 and feature_matching weight 100 are present while generic perceptual and lpips are zeroed, confirming the discrepancy with the paper's Eq. (6) description.
- Codex: **lowered** (escalated) — The config and generator-loss code confirm active weight-30 ResNetPL and weight-100 feature matching with generic perceptual and LPIPS disabled, but the paper already identifies L_P by citing the LaMa paper [37], leaving only the omitted feature-matching term as a low-confidence documentation discrepancy.

### 5243/reported-numbers-not-reproducible
- **Synthetic Table 1/Table 3 numbers are not pinned or reproducible from the notebook** _missing · severity medium · confidence medium_
- Claude: **lowered** — Line 2224 quote confirmed, but the audit's core mechanism is wrong: inside make_dataset `_set_seeds(seed)` (which sets the numpy RNG AND torch.manual_seed) runs BEFORE `g = _mlp(...)`, so g is fully seeded; f_Y uses its own Generator().manual_seed(0); train_test_split uses random_state=seed; the eval cell calls _set_seeds(50) before building/training the model zoo. The generative pipeline and training are therefore deterministically seeded, contradicting the claim. A weaker residual concern survives: no Table 1/3/6 numbers are pinned in the repo, and even a faithful re-run is not bit-reproducible because the DataLoader uses shuffle=True with no seeded generator (line 1767) consumed in a path-dependent way under per-model early-stopping, and the models run on device='cuda' when available (lines 2251-2263) with no deterministic-algorithm/cudnn flags — so a genuine but mild reproducibility gap remains, independent of the (incorrect) g-unseeded mechanism.
- Codex: **reject** (escalated) — The notebook calls _set_seeds(seed) before constructing the selection network g, invokes make_dataset with seed=42, and stores synthetic outputs that match the paper's Table 1 values after rounding, so the claimed unseeded generative component is absent.

### 570/tuning-and-selection-against-eval-graph
- **Per-dataset lambda and best-accuracy checkpoint selected against the same GC graph used for scoring** _methodology · severity high · confidence medium_
- Claude: **keep** — Lines 661-669 of clstm.py match the quote exactly; the best-accuracy checkpoint is updated each check_every interval by direct comparison to true_GC, while final restore at lines 688-690 uses best-loss — leaving ambiguous which checkpoint produced the reported numbers, confirming both parts of the methodology concern.
- Codex: **lowered** (escalated) — The code uses true_GC to save a separate best-accuracy checkpoint and logs per-lambda Acc./BA, but the reported GC_est metrics are computed from the restored best-loss xlstm rather than best_accuracy_model, so medium severity is appropriate.

### 713/lambda-table-not-reproduced
- **Reported Λ̂(k) (Tables 1,5–8) not reproduced by the repo's spin learning-curve code** _missing · severity high · confidence medium_
- Claude: **keep** — Code at lines 269-270 matches the quote exactly; _audit_code/out/spin_table_compare.txt confirms the committed JSON gives ulc values (e.g. k=1: 0.4006 bits) that differ substantially from the paper Table 6 Λ̂ (k=1: 0.3213), and the unit-conversion to nats (×ln2=0.2776) still does not match, confirming the discrepancy.
- Codex: **lowered** (escalated) — The cited code and copied check confirm the Lambda mismatch for the committed M=100,000 result and the near-invariant values in Tables 5-7, but Table 8 reports a materially different curve, so the all-four-block-sizes scope is unsupported.

### 760/hf-dataset-only-one-variable
- **Hugging Face 'processed data' provides only DEWP, not all 6 reported variables** _difference · severity low · confidence medium_
- Claude: **keep** — README.md lines 57-66 match the quote exactly; the HF dataset content claim relies on a live URL retrieved at audit time which cannot be re-verified without network access, so the README quote is confirmed but the HF repo contents are taken as cannot-verify — however since the auditor's original check was dated 2026-05-30 (same as today) and the README quote itself is verified, the finding stands as a valid concern about the discrepancy.
- Codex: **reject** (escalated) — The Hugging Face history shows that embeddings and PyG processed archives for all six variables were uploaded on November 7, 2025, before the stated May 30, 2026 retrieval date, so the claim that only DEWP was available is false.

### 768/ablation-experiments-code-missing
- **LoRA (Table 6), Digits/WADN (Table 7), ResNet/ViT-B variants not wired into the runnable code** _missing · severity medium · confidence medium_
- Claude: **keep** — datasets.py lines 16-23 exactly match the quote; DATASETS only lists ColoredMNIST, RotatedMNIST, OfficeHome, DomainNet — no Digits class, no 3Conv backbone, and no LoRA/ViT-B references are present in the file.
- Codex: **lowered** (escalated) — Tables 6 and 7 require LoRA on ViT-B and Digits with 3Conv, and those paths are absent from the non-vendored project code, but bare ViT-B/16 and ResNet-18/50/101 support are present, confirming the first-pass narrowing.

