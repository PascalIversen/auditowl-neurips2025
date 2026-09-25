# Code audit — VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction (NeurIPS 2025, #2278)

## Summary

The repository `VITA-MLLM/VITA` (audited commit `35d064a`, no submission tag) is the
authors' training + inference codebase for VITA-1.5. It contains: model definitions
(`vita/model/`), a single training entrypoint (`vita/train/train.py`) driven by Stage-1/2
shell scripts under `script/train/`, an inference demo (`video_audio_demo.py`), a web/vLLM
real-time demo (`web_demo/`), a vendored fork of VLMEvalKit (`VLMEvalKit/`) used to produce
the image-benchmark numbers (Table 2), and custom Video-MME inference/scoring scripts
(`videomme/`) used for Table 3. No model weights are shipped; the README points to
HuggingFace downloads. The paper reports three headline result tables: image understanding
(Table 2), video understanding (Table 3), and ASR (Table 4), plus a three-stage training
methodology (vision-language; audio input; audio output / end-to-end TTS).

What I did: I mapped each reported table/figure to producing code; grepped the whole tree
(excluding the VLMEvalKit fork) for ASR metric/dataset code, CTC speech-encoder training,
and Stage-3 codec/decoder training; read the training entrypoint and all training shell
scripts; read the Video-MME inference (`yt_video_inference_qa_imgs.py`) and scoring
(`parse_answer.py`) scripts; and inspected the VLMEvalKit VITA wrapper. Deterministic
checks live in `_audit_code/check_artifacts.py` (output `_audit_code/out/artifacts.json`).

Headline findings: (1) **Table 4 (ASR) has no producing code at all** — nothing in the repo
computes WER/CER or references aishell/LibriSpeech/test-net/test-meeting. (2) The paper's
**Stage 3 audio-output training (codec + NAR/AR speech decoder)** and **Stage 2.1(a) CTC
speech-encoder training** are described in the paper but **absent from the code** (training
entrypoint and shell scripts never touch the TTS/codec modules; the audio encoder is always
frozen and loaded pretrained). (3) The Video-MME scorer uses a **hardcoded per-category
denominator** (`num_total += 30`) rather than counting the actual questions present.
The image-benchmark path (Table 2) is the best-supported result and looks fine.

## Traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 2 (image: MMB/MMS/MMMU/MathV/Hal/AI2D/OCR/MMVet/MME, VITA-1.5 rows) | `VLMEvalKit/vlmeval/vlm/vita/vita_qwen2.py` (model wrapper) + `VLMEvalKit/run.py`; README cmd | not run (needs weights+benchmarks+GPU) | — | Code present (inference path); values not re-derivable offline |
| Table 3 (Video-MME w/o & w/ sub) | `videomme/yt_video_inference_qa_imgs.py` + `videomme/parse_answer.py` | not run (needs Video-MME data) | — | Code present; scorer uses hardcoded denominator (see `videomme-hardcoded-denominator`) |
| Table 3 (MVBench, TempCompass) | (none found) | — | — | MISSING (no MVBench/TempCompass eval driver in repo) |
| Table 4 (ASR: CER on aishell-1/test-net/test-meeting; WER on dev/test clean/other) | (none) | — | — | MISSING (no WER/CER code, no dataset refs) — `asr-eval-missing` |
| Stage 1.1/1.2/1.3 vision training | `script/train/pretrain_mlp_qwen.sh`, `finetune_qwen.sh`, `finetuneTask*` + `vita/train/train.py` | — | — | Present (training scripts map to vision stages) |
| Stage 2.1(b) audio-adapter alignment | `script/train/pretrain_audio_mlp_qwen.sh` (`--tune_audio_mlp_adapter True`) | — | — | Present |
| Stage 2.1(a) CTC speech-encoder training | (none) | — | — | MISSING (no CTC training loop; encoder always frozen) — `ctc-encoder-training-missing` |
| Stage 2.2 audio SFT | `script/train/finetuneTaskNeg_qwen*.sh` (audio data configs) | — | — | Present |
| Stage 3.1 Codec training / Stage 3.2 NAR+AR decoder training | (none) | — | — | MISSING (TTS/codec modules used only by web_demo inference, never trained) — `stage3-audio-output-training-missing` |
| README headline: avg image perf 59.8 → 70.8; WER 18.4 → 7.5 | README prose only | — | — | Derived from Tables 2/4; ASR side not reproducible (see `asr-eval-missing`) |
| Model weights / pretrained audio encoder | README HF links; none in repo | — | — | External (HF); not in repo (expected for a 7B model) |

Notes: the image (Table 2) and video (Table 3) inference paths require author weights,
the benchmark datasets, and GPUs, and Table 2 additionally requires an LLM judge; values
cannot be re-derived in this read-only offline audit, so those rows are marked "code present,
not re-derived" rather than verified or mismatched.

## missing

```yaml finding
id: asr-eval-missing
category: missing
topic: "result traceability / ASR evaluation"
title: "Table 4 ASR results have no producing code in the repo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 289
line_end: 291
quote: |
  Evaluating on these benchmarks:
  ```
  CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_TEST_EN_V11 MMBench_TEST_CN_V11 MMStar MMMU_DEV_VAL MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet MME --model vita_qwen2 --verbose
claim: "The repo's only documented evaluation harness is VLMEvalKit for image benchmarks (Table 2) and the videomme/ scripts for Video-MME (Table 3); a tree-wide search (excluding the VLMEvalKit fork) finds no code computing WER/CER and no reference to the ASR datasets aishell-1, test-net, test-meeting, dev/test-clean, or dev/test-other used in Table 4."
concern: "The paper's Table 4 ASR numbers (CER 2.2/8.4/10.0; WER 3.3/7.2/3.4/7.5) — a headline contribution ('outperforms specialized speech models') — cannot be reproduced or traced to any script in the repository."
resolution: "Authors: please add the ASR inference + WER/CER scoring code and dataset preparation used to produce Table 4, or point to where it lives."
cross_refs: ["ctc-encoder-training-missing", "stage3-audio-output-training-missing"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Table 4 (Evaluation on ASR Benchmarks)"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stage3-audio-output-training-missing
category: missing
topic: "training pipeline / Stage 3 audio output"
title: "Stage 3 codec + NAR/AR speech-decoder training is described but not in the code"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  Stage 3.1 Codec Training. The goal of this step is to train a codec model with a single codebook
claim: "The paper describes Stage 3.1 (train a single-codebook codec) and Stage 3.2 (train NAR+AR speech decoders on text-speech pairs); in the repo the TTS/codec modules in vita/model/vita_tts/ are imported only by the inference demos (web_demo/server.py, web_demo/web_ability_demo.py) — the training entrypoint vita/train/train.py contains no reference to tts/codec/nar/decoder, and no script under script/train/ trains them (_audit_code/out/artifacts.json: stage3_training_shell_scripts=[], train_py_mentions_tts_codec=false)."
concern: "The end-to-end speech-output capability is a core paper claim ('without separate ASR and TTS modules'), but the code to train the codec and the NAR/AR decoders — i.e. to reproduce the speech-generation model — is absent; only inference-time use of pretrained decoders is shipped."
resolution: "Authors: please add the Stage 3.1 codec training and Stage 3.2 NAR/AR decoder training code, or confirm these were trained off-repo and only weights are released."
cross_refs: ["asr-eval-missing"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Section 3.3.3 Stage 3 (Audio Output Tuning)"
tags: [reforms:3, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ctc-encoder-training-missing
category: missing
topic: "training pipeline / Stage 2.1 audio encoder"
title: "Stage 2.1(a) CTC speech-encoder training is described but absent from the code"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  recognition systems, using a Connectionist Temporal Classification (CTC) loss function [54] to train
claim: "The paper's Stage 2.1 step (a) trains the speech encoder with a CTC loss on 11,000 hours of speech-transcription pairs; in the repo no training loop uses CTCLoss/ctc_loss (_audit_code/out/artifacts.json: ctc_training_loop_files=[]), the only CTC references (vita/model/vita_tts/audioLLM.py) load a frozen CTC module for inference (self.ctc = ctc.eval()), and every audio training script sets --freeze_audio_encoder True while loading a pretrained 'audio-encoder ... 11wh-tunning' checkpoint."
concern: "The audio encoder that underlies the ASR results is consumed as a pretrained download; the code that trains it (the CTC stage) is not in the repo, so that part of the pipeline is not reproducible."
resolution: "Authors: please release the CTC speech-encoder training code, or state explicitly that the audio encoder is provided only as pretrained weights."
cross_refs: ["asr-eval-missing", "stage3-audio-output-training-missing"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Section 3.3.2 Stage 2.1 (a) Speech Encoder Training"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: mvbench-tempcompass-eval-missing
category: missing
topic: "result traceability / video benchmarks"
title: "No driver for the MVBench and TempCompass columns of Table 3"
severity: medium
confidence: medium
status: question
file: README.md
line_start: 294
line_end: 296
quote: |
  ### Video-MME
  #### Data Preparation
  Download the [Video-MME dataset](https://github.com/BradyFU/Video-MME) and extract the frames, saving them as images to improve IO efficiency.
claim: "The README's video-evaluation section and the videomme/ scripts cover only Video-MME; Table 3 also reports MVBench and TempCompass for VITA-1.5, but I found no repo script that runs or scores those two benchmarks (they may be intended to run through the vendored VLMEvalKit, which is not documented for VITA video eval)."
concern: "Two of the four Table-3 columns lack a clearly identifiable producing path in the repo, so those numbers are not straightforwardly reproducible."
resolution: "Authors: please point to the MVBench / TempCompass evaluation entrypoint (VLMEvalKit config or a script), or add it."
cross_refs: []
paper_ref: "Table 3 (MVBench, TempCompass columns)"
tags: [reforms:3]
```

## bug

```yaml finding
id: videomme-hardcoded-denominator
category: bug
topic: "evaluation scoring / Video-MME"
title: "Video-MME scorer hardcodes 30 questions per category instead of counting them"
severity: medium
confidence: medium
status: finding
file: videomme/parse_answer.py
line_start: 91
line_end: 103
quote: |
        num_total += 30
        cate_df = pd.read_csv(f"{result_dir}/{video_type}/{category}.csv")
        correct = 0
        for (cate_id, cate_row) in cate_df.iterrows():

            gt = cate_row[["答案一", "答案二", "答案三"]]
            pred = cate_row[["模型回答一", "模型回答二", "模型回答三"]]

            pred = pred.apply(extract_characters_regex)

            correct += (gt.to_numpy() == pred.to_numpy()).sum()
            
        num_correct += correct
claim: "For each (video_type, category) the script adds a fixed constant 30 to the accuracy denominator (num_total += 30) while the numerator sums correct answers over the actual rows of the CSV (each row contributing up to 3 questions); the denominator never reads len(cate_df) (_audit_code/out/artifacts.json: parse_answer_hardcoded_num_total=true, parse_answer_counts_rows=false)."
concern: "If any (video_type, category) CSV does not contain exactly 10 videos / 30 questions (e.g. dropped/`continue`-skipped videos in yt_video_inference_qa_imgs.py, or NaN answers), the reported Video-MME accuracy in Table 3 is computed against a wrong, fixed denominator rather than the number actually evaluated."
resolution: "Authors: confirm every category CSV always has exactly 30 questions, or change num_total to count the actual number of (row x 3) questions scored; report whether skipped videos were handled."
cross_refs: ["mvbench-tempcompass-eval-missing"]
check_script: _audit_code/check_artifacts.py
paper_ref: "Table 3 (Video-MME w/o sub, w/ sub)"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: videomme-sampling-decoding
category: difference
topic: "evaluation / decoding settings"
title: "Video-MME inference uses sampling (do_sample=True) with max 10 new tokens"
severity: low
confidence: medium
status: finding
file: videomme/yt_video_inference_qa_imgs.py
line_start: 291
line_end: 307
quote: |
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=video_frames.half().cuda(),
                            audios=audios,
                            sf_masks=sf_masks,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            max_new_tokens=10,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            shared_v_pid_stride=None#2#16#8#4#1#None,
                        )
claim: "The Video-MME multiple-choice inference uses stochastic decoding (do_sample=True) at temperature 0.01, whereas the image-benchmark wrapper (VLMEvalKit/vlmeval/vlm/vita/vita_qwen2.py:205) uses deterministic decoding (do_sample=False); the paper does not specify the video decoding settings."
concern: "Sampling (even at low temperature) makes the Video-MME numbers non-deterministic and not exactly reproducible; the two evaluation paths use inconsistent decoding, undocumented in the paper."
resolution: "Authors: confirm the decoding configuration used for the reported Video-MME results and whether sampling was intentional; consider greedy decoding for a fixed-answer MCQ task."
cross_refs: ["videomme-hardcoded-denominator"]
paper_ref: "Table 3"
tags: [reforms:4]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology findings. The procedures that *are* implemented (image-benchmark inference
via VLMEvalKit with deterministic decoding; the three-stage progressive-training entrypoint;
the Video-MME MCQ scoring as a letter-match) are individually valid for their tasks. The
material problems are absence of producing code (the `missing` findings) and one scoring
bug, not an invalid implemented procedure. N/A topics: data-splitting / sample-independence
/ target-leakage / temporal integrity — VITA-1.5 is evaluated on fixed external public
benchmarks with no author-constructed train/test split, so split-leakage checks are
structurally inapplicable. Pretraining-contamination between the (undisclosed, internal)
speech training data and the public ASR test sets is a legitimate concern in principle, but
neither the data nor the eval code is in the repo, so it cannot be assessed here — it is
folded into `asr-eval-missing` rather than asserted as a separate finding.

## Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                          |
|-------------|------------|--------------|--------------------------------------------------------------------------|
| missing     | 4          | high         | ASR eval (Table 4) and Stage-3/CTC audio training have no producing code |
| bug         | 1          | medium       | Video-MME scorer hardcodes 30-question denominator                       |
| difference  | 1          | low          | Video-MME uses sampling decoding; image path uses greedy; undocumented   |
| methodology | 0          | -            | Implemented procedures are valid; issues are absence + one scoring bug   |

## Top take-aways (<=6, ranked by severity x confidence)

1. **[missing] `asr-eval-missing`** — Table 4 (ASR, a headline contribution) has no
   WER/CER computation or dataset reference anywhere in the repo; the numbers are not
   reproducible. (high / high)
2. **[missing] `stage3-audio-output-training-missing`** — the Stage 3 codec + NAR/AR
   speech-decoder *training* (the end-to-end TTS contribution) is absent; the TTS modules
   are only used for inference. (high / high)
3. **[missing] `ctc-encoder-training-missing`** — the Stage 2.1 CTC speech-encoder training
   is absent; the audio encoder is always loaded frozen/pretrained. (medium / high)
4. **[bug] `videomme-hardcoded-denominator`** — the Video-MME scorer divides by a fixed
   30 per category instead of the number of questions actually scored, which biases the
   Table 3 accuracy if any video was skipped. (medium / medium)
5. **[missing] `mvbench-tempcompass-eval-missing`** — no clear producing path for the
   MVBench / TempCompass columns of Table 3. (medium / medium)
6. **[difference] `videomme-sampling-decoding`** — Video-MME inference samples
   (do_sample=True), making those numbers non-deterministic and inconsistent with the
   greedy image-eval path. (low / medium)

## Items that genuinely look fine

- **Image-benchmark path (Table 2).** The VLMEvalKit VITA-Qwen2 wrapper exists
  (`VLMEvalKit/vlmeval/vlm/vita/vita_qwen2.py`), is registered in `config.py:140-142`, and
  uses deterministic decoding (`do_sample=False`, `temperature=0.01`, lines 205-207); the
  README gives the exact `run.py` command. This is the best-supported result.
- **Three-stage vision/audio-input training entrypoint.** `vita/train/train.py` plus the
  `script/train/*.sh` family expose the freeze/tune flags that map onto Stages 1.1–1.3 and
  2.1(b)/2.2 (`tune_mm_mlp_adapter`, `tune_audio_mlp_adapter`, `unfreeze_vision_tower`).
- **Dependencies are reasonably pinned** (`requirements.txt`: `torch==2.3.1`,
  `transformers==4.41.1`, etc.), so the core environment is rebuildable.
- **Inference for speech output is present** (`vita/model/vita_tts/`, `web_demo/`), even
  though the corresponding training code is not.

## Open questions for the authors

- Where is the ASR inference + WER/CER scoring code for Table 4, and what exact dataset
  splits were used? (`asr-eval-missing`)
- Were the codec and NAR/AR decoders (Stage 3) and the CTC speech encoder (Stage 2.1a)
  trained with code that can be released, or only shipped as weights?
  (`stage3-audio-output-training-missing`, `ctc-encoder-training-missing`)
- Do all Video-MME category CSVs contain exactly 30 questions, and how were skipped /
  missing videos handled in the reported accuracy? (`videomme-hardcoded-denominator`)
- Which decoding configuration produced the reported Video-MME numbers?
  (`videomme-sampling-decoding`)
