# Code-repository audit — LeVo: High-Quality Song Generation with Multi-Preference Alignment (NeurIPS 2025, #3109)

## 1. Summary

The cited repository (`tencent-ailab/songgeneration`, audited commit `b1b03ec`) is an
**inference-only** release of a song-generation model. It contains the LeLM model
definition (`codeclm/models/`), the Music Codec / Flow1dVAE tokenizer code
(`codeclm/tokenizer/`), a single inference entrypoint (`generate.py` + `generate.sh`), a
Gradio demo (`tools/gradio/`), and demo input files (`sample/*.jsonl`). Model weights are
hosted on HuggingFace and downloaded at setup; the only in-repo weight artefact
(`tools/new_auto_prompt.pt`) is an unmaterialised Git-LFS pointer stub (a fetch artefact,
treated as present per the audit instructions).

What I did: read the paper (`paper.pdf`) and all tables/appendices; enumerated the full
repo file tree; grepped the entire Python tree for the result-producing components the
paper claims (LeLM training loop, DPO loss, preference-data construction, parameter
interpolation, and the objective metrics FAD / PER / MuQ-T / MuQ-A / Audiobox-Aesthetic).
Deterministic checks live in `_audit_code/check_repo_completeness.py`
(output `_audit_code/out/repo_completeness.json`).

**Two structural facts dominate this audit.**

1. **Repository provenance.** The audited `main` is **SongGeneration 2 / LeVo 2** — a
   *later, different* system (released 2026-03-01, 4B params, "hybrid LLM-Diffusion"
   architecture, multilingual, PER 8.55%, evaluated by 20 professionals on six *different*
   dimensions). `README.md:9-10` itself labels the NeurIPS paper's system "SongGeneration
   (old version)" and points its technical report at arXiv 2506.07520. The repo's
   "Evaluation Performance" section reports SongGeneration-2 numbers, not the paper's
   Table 1/2. The paper cites `main`, which is a moving branch, not a submission-tagged
   commit, so the artefact that produced the paper's tables is not what is checked out.

2. **No result-producing code is present.** The repo computes none of the paper's
   reported numbers. There is no LeLM training code, no DPO / multi-preference alignment
   code (Stage 3), no preference-data construction (Strategies 1–3), no parameter
   interpolation/merging, and no objective-metric computation (FAD, PER, MuQ-T/A,
   Audiobox-Aesthetic). Every quantitative claim in the paper is therefore untraceable to
   the repo. These route to `missing` (Rule G).

Because the repo is a sound, runnable inference release whose code does not *contradict* a
valid method (it simply does not contain the experimental harness), the bulk of findings
are `missing`. I found no `bug`, no `methodology`, and one minor `difference` /
statistical-integrity item internal to the paper.

## 2. Traceability table

Every quantitative artefact in the paper is matched against repo code that *computes* it
(not merely runs the model). "(none)" means no such computation exists in the repo.

| Paper artefact | Repo location (computes value) | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 — FAD / MuQ-T / MuQ-A / PER / CE/CU/PC/PQ (LeVo + 7 baselines + 4 ablations) | (none — no metric code) | — | — | MISSING |
| Table 2 / 7 / 8 — subjective MOS (OVL/MEL/HAM/SSC/AQ/LYC) + CI95 | (none — no MOS harness) | — | — | MISSING |
| Table 3 / 10 — DPO strategy & interpolation-coefficient sweeps | (none — no DPO/interpolation code) | — | — | MISSING |
| Table 5 — Codec reconstruction (VISQOL / SPK_SIM / WER) | (none — no VISQOL/SPK_SIM/WER eval) | — | — | MISSING |
| Table 6 — Codec RTF (Wav2Code/Code2Latent/Latent2Wav) | (none — no RTF benchmark script) | — | — | MISSING |
| Table 9 (App. F) — module-wise AQ; LeLM row = 0.21 | (none) | — | internally inconsistent (see `table9-lelm-aq-inconsistent`) | MISSING + flagged |
| Table 11 — Pearson corr (subjective vs Audiobox) | (none) | — | — | MISSING |
| App. H — memorization: 5-gram=0.0001, Levenshtein=0.0012 | (none — no overlap/Levenshtein script) | — | — | MISSING |
| §3.4 — DPO preference data (20k lyrics, 60k win-lose pairs) | (none) | — | — | MISSING |
| §3.4 / §3.5 — DPO fine-tuning + linear param interpolation | (none) | — | — | MISSING |
| §3.5 — Stage 1/2/3 training (265K steps, 32×H20) | (none — no training loop for LeLM) | — | — | MISSING |
| §4.1 — eval inputs: 20 lyrics + 20 audio prompts | (none — only demo `lyrics.jsonl` (5) & `test100` (100)) | — | — | MISSING |
| Model inference (LeLM + Codec, audio out) | `generate.py`, `codeclm/` | runnable (weights ext.) | n/a (produces audio, not metrics) | PRESENT |

Deterministic support: `_audit_code/out/repo_completeness.json` shows DPO_loss=0,
preference_data=0, FAD=0, MuQ=0, Audiobox=0, reward_model=0 signature hits; the only
`.backward()` hits are inside MERT/codec sub-modules, and the 40 "PER" hits are the
substring "per" in `model_*rvq.py`, not the phoneme-error-rate metric.

## 3. Findings

## missing

```yaml finding
id: no-eval-metric-code
category: missing
topic: "result traceability"
title: "No code computes any objective metric in Tables 1/3/5/6/10 (FAD, PER, MuQ, Audiobox)"
severity: high
confidence: high
status: finding
file: README.md
line_start: 148
line_end: 152
quote: |
  Once everything is set up, you can run the inference script using the following command:

  ```bash
  sh generate.sh ckpt_path lyrics.jsonl output_path
  ```
claim: "The repo's only entrypoint is `generate.sh`/`generate.py`, which generates audio; a full-tree grep finds zero implementations of FAD, PER, MuQ-T/MuQ-A, or Audiobox-Aesthetic (the metrics underlying Tables 1, 3, 5, 6, 10)."
concern: "None of the paper's headline objective numbers (e.g. LeVo PER 7.2, MuQ-T 0.34, FAD 2.68) can be traced to or reproduced from any script in the repository."
resolution: "Authors: please release the evaluation harness that computes FAD/PER/MuQ/Audiobox over the generated songs, or point to the exact external scripts and versions used."
cross_refs: ["no-dpo-training-code", "no-lelm-training-code", "missing-eval-input-set", "repo-is-songgeneration2-not-levo"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Tables 1, 3, 5, 6, 10; §4.1 Evaluations"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-dpo-training-code
category: missing
topic: "multi-preference alignment (DPO)"
title: "DPO multi-preference alignment, preference-data construction, and interpolation merge are absent"
severity: high
confidence: high
status: finding
file: paper.pdf
quote: |
  The above paired data are then used for DPO fine-tuning to address the challenges posed by scarce
  high-quality data in the dimensions of lyrics alignment, prompt consistency, and musicality.
claim: "The paper's central contribution (Stage-3 DPO multi-preference alignment via three preference strategies plus DNI-style linear parameter interpolation, Tables 3 & 10) has no implementation: a full-tree grep finds no DPO loss, no win-lose/preference-pair construction, and no parameter-interpolation/merge code."
concern: "The headline novelty 'first multi-preference DPO for song generation' and all DPO ablations (Table 3/10) are entirely unverifiable from the repo."
resolution: "Authors: please release the DPO fine-tuning code, the preference-data construction scripts (Strategies 1-3), and the interpolation/merging procedure."
cross_refs: ["no-eval-metric-code", "no-lelm-training-code"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "§3.4 DPO-based Multi-Preference Alignment; Tables 3, 10"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: no-lelm-training-code
category: missing
topic: "training protocol"
title: "No LeLM training code for the three-stage paradigm (pre-train / modular extension / alignment)"
severity: high
confidence: high
status: finding
file: codeclm/trainer/codec_song_pl.py
line_start: 1
line_end: 4
quote: |
  """
  Main model for using CodecLM. This will combine all the required components
  and provide easy access to the generation API.
  """
claim: "The file named `trainer/codec_song_pl.py` is a Lightning module containing only an `__init__`, an inference mask helper, and an LR-scheduler class; it has no `training_step`, `forward`, loss, or `.backward()`. No script implements the paper's three-stage training (265K steps; Stage 1/2/3) or the modular-extension freezing strategy."
concern: "The training procedure that produces every model in the paper (and the ablations w/o stage 2, w/o AR decoder, w/o dual-track) cannot be reproduced or inspected."
resolution: "Authors: please release the training entrypoints and configs for the three stages, including the freezing schedule for modular extension training."
cross_refs: ["no-dpo-training-code", "no-eval-metric-code"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "§3.5 Three-Stage Training Paradigm; Appendix B"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: repo-is-songgeneration2-not-levo
category: missing
topic: "repository provenance"
title: "Audited main is SongGeneration 2 (LeVo 2), a later/different system than the paper's LeVo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 1
line_end: 12
quote: |
  # SongGeneration 2

  <p align="center"><img src="img/logo.jpg" width="40%"></p>

  #### SongGeneration 2

  [![Project Page](https://img.shields.io/badge/Project%20Page-GitHub-blue)](https://github.com/tencent-ailab/songgeneration) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/tencent/SongGeneration) [![Live Playground](https://img.shields.io/badge/Live%20PlayGround-Demo-orange)](https://huggingface.co/spaces/waytan22/SongGeneration-LeVo) [![Samples](https://img.shields.io/badge/Audio%20Samples-Page-green)](https://levo-demo.github.io/levo_v2_demo/)

  #### SongGeneration (old version)
  [![Technical Report](https://img.shields.io/badge/Technical%20Report-Arxiv-red)](https://arxiv.org/abs/2506.07520) [![Samples](https://img.shields.io/badge/Audio%20Samples-Page-green)](https://levo-demo.github.io/)

  🚀 We introduce LeVo 2 (SongGeneration 2), an open-source music foundation model designed to shatter the ceiling of open-source AI music by achieving true commercial-grade generation. 
claim: "The audited commit (b1b03ec, a single shallow commit on a moving `main`, no tags) is 'SongGeneration 2 / LeVo 2': a 4B-param hybrid LLM-Diffusion model evaluated on six different dimensions (Overall/Melody/Arrangement/Sound Quality-Instrument/Sound Quality-Vocal/Structure) with PER 8.55%; the README explicitly relabels the paper's LeVo as 'SongGeneration (old version)' pointing at arXiv 2506.07520."
concern: "The paper cites this repo as its source code, but the checked-out artefact is a later, architecturally different system; the README's 'Evaluation Performance' reports SongGeneration-2 numbers, none of which are the paper's Table 1/2 values, so the repo cannot substantiate the paper's results."
resolution: "Authors: please provide a submission-tagged commit (or branch) corresponding to the NeurIPS 'LeVo' system whose code matches the paper's 2B LeLM architecture and evaluation."
cross_refs: ["no-eval-metric-code", "no-dpo-training-code", "no-lelm-training-code"]
paper_ref: "Abstract footnote URL https://github.com/tencent-ailab/songgeneration; README.md"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: missing-eval-input-set
category: missing
topic: "evaluation inputs"
title: "Paper's evaluation set (20 lyrics + 20 audio prompts) is not in the repo"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  For evaluation, we generate 20 distinct lyrics and accompanying text descriptions with a large
  language model, and select 20 unseen music clips as audio prompts.
claim: "Appendix D states the evaluation used 20 lyrics + 20 audio prompts; the repo ships only demo inputs (`sample/lyrics.jsonl`, 5 lines; `sample/test100_v2_sg_des.jsonl`, 100 lines, which README ties to the SongGeneration-2 100-song benchmark), neither of which is the paper's 20-item set."
concern: "Without the exact evaluation inputs, the paper's comparison numbers cannot be regenerated even if a metric harness existed."
resolution: "Authors: please release the 20 lyrics, 20 text descriptions, and 20 audio-prompt clips used for the paper's Tables 1-3."
cross_refs: ["no-eval-metric-code"]
check_script: _audit_code/check_repo_completeness.py
paper_ref: "Appendix D Detailed Experimental Settings"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No findings. The inference code (`generate.py`, `codeclm/`) is internally consistent;
the single in-repo `.pt` is an unmaterialised LFS pointer stub (a fetch artefact, not a
code defect). I did not execute the model (no GPU / external weights), so runtime
behaviour beyond static inspection is out of scope.

## difference

No standalone `difference` findings: the repo contains no metric/training code whose
*valid* logic could disagree with the paper, so the paper-vs-code discrepancies all route
to `missing` (Rule routing step 1). The provenance mismatch is filed under `missing`
(`repo-is-songgeneration2-not-levo`) because the paper-described LeVo artefact is absent
from the audited commit rather than present-but-different.

## methodology

No findings. The audit cannot reach methodology questions (leakage, baseline fairness,
metric fit, statistical tests) because the procedures that would embody them — splitting,
metric computation, DPO, baselines — are not in the repo. The one statistical-integrity
observation that is checkable from the paper itself is filed below as a `question` (it is
an internal table inconsistency, not a code defect).

```yaml finding
id: table9-lelm-aq-inconsistent
category: methodology
topic: "statistical integrity"
title: "Table 9 'LeLM' audio-quality value 0.21 is inconsistent with the surrounding values and prose"
severity: low
confidence: medium
status: question
file: paper.pdf
quote: |
  As shown in Table 9, the overall degradation in audio
  quality mainly arises from three sources: VAE reconstruction loss (0.02), Codec reconstruction loss
  (0.04), and LeLM modeling loss (0.09).
claim: "Table 9 lists module AQ MOS of GT 3.81, VAE Recon 3.79, Codec Recon 3.75, and LeLM 0.21; the text attributes a LeLM modeling loss of 0.09, which would imply a LeLM AQ near 3.66, not 0.21 (the differences 3.81-3.79=0.02 and 3.79-3.75=0.04 match the stated VAE/Codec losses)."
concern: "The reported LeLM AQ of 0.21 appears to be a typographical/transcription error (likely ~3.66 or a 0.21 drop), leaving the absolute audio-quality of LeVo-generated audio in this analysis ambiguous."
resolution: "Authors: please confirm whether the Table 9 'LeLM' AQ entry should read ~3.66 (consistent with a 0.09 modeling loss) and correct the table."
cross_refs: []
paper_ref: "Table 9 and surrounding text, Appendix F"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line)                                                        |
|-------------|------------|--------------|------------------------------------------------------------------------|
| missing     | 5          | high         | No metric/DPO/training/eval-input code; audited repo is SongGeneration 2, not the paper's LeVo. |
| bug         | 0          | -            | Inference code internally consistent; not executed (external weights/GPU). |
| difference  | 0          | -            | All paper-vs-code gaps route to `missing` (the relevant code is absent). |
| methodology | 1          | low          | One paper-internal table inconsistency, filed as a `question`.         |

### Top take-aways (≤6, ranked by severity × confidence)

1. **[missing] `repo-is-songgeneration2-not-levo`** — the audited `main` is the later
   *SongGeneration 2 / LeVo 2* system (4B, hybrid LLM-Diffusion, different eval), which the
   README itself distinguishes from the paper's "old version" LeVo; the cited repo does not
   correspond to the artefact that produced the paper. (high/high)
2. **[missing] `no-eval-metric-code`** — no code computes FAD, PER, MuQ-T/A, or
   Audiobox-Aesthetic; none of Tables 1/3/5/6/10 are traceable. (high/high)
3. **[missing] `no-dpo-training-code`** — the paper's central contribution (multi-preference
   DPO + interpolation, Tables 3 & 10) has no implementation in the repo. (high/high)
4. **[missing] `no-lelm-training-code`** — no three-stage training code; all models and
   ablations are non-reproducible from the repo. (high/high)
5. **[missing] `missing-eval-input-set`** — the paper's 20-lyric/20-prompt evaluation set
   is not shipped. (medium/high)
6. **[methodology/question] `table9-lelm-aq-inconsistent`** — Table 9 LeLM AQ = 0.21
   contradicts the stated 0.09 modeling loss; likely a typo. (low/medium)

### Items that genuinely look fine

- The inference pipeline (`generate.py` + `generate.sh` + `codeclm/`) is coherent: CLI flags
  in `generate.sh` map to `generate.py` arguments, `config.yaml`+`model.pt` loading is
  consistent, and generation modes (mixed/vocal/bgm/separate) are wired correctly.
- Dependencies are specified and pinned (`requirements.txt`, `requirements_nodeps.txt`),
  including exact `torch`/`transformers` versions — the environment is rebuildable.
- Hosting large model weights on HuggingFace (rather than in-repo) is a legitimate,
  documented choice; the README gives exact `huggingface-cli` download commands.
- The LeLM and Music-Codec *model definitions* (architecture) are present and match the
  paper's described component structure (decoder-only LM + AR decoder; MuEncoder→RVQ→DiT→VAE).

### Open questions for the authors

- Is there a submission-time tag/branch of this repo that corresponds to the NeurIPS LeVo
  (2B) system rather than the current SongGeneration-2 `main`?
- Will the evaluation harness (FAD/PER/MuQ/Audiobox), the DPO/preference-data pipeline, the
  three-stage training code, and the exact 20-item evaluation set be released, so that the
  paper's tables become reproducible?
- Is the Table 9 LeLM AQ value (0.21) a transcription error?
