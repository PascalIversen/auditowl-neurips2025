"""Checks which paper artefacts have producing code in the repo: ASR eval (Table 4),
Stage-3 audio-output training, CTC speech-encoder training (Stage 2.1a), parse_answer
denominator assumption, and pinned reproduction commands.
Supports findings: asr-eval-missing, stage3-audio-output-training-missing,
ctc-encoder-training-missing, videomme-hardcoded-denominator."""
import os, re, subprocess, json

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "VITA-MLLM__VITA")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def grep_count(pattern, exclude_substr=None):
    """Count files (non-binary) containing regex pattern, excluding paths matching exclude_substr."""
    hits = []
    for root, _, files in os.walk(REPO):
        if "/.git" in root:
            continue
        for fn in files:
            if not fn.endswith((".py", ".sh", ".md", ".json")):
                continue
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, REPO)
            if exclude_substr and exclude_substr in rel:
                continue
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue
            if re.search(pattern, txt, re.IGNORECASE):
                hits.append(rel)
    return hits

results = {}

# 1. ASR evaluation: any code computing WER/CER or referencing aishell/librispeech/test-net/test-meeting?
asr_metric = grep_count(r"\b(wer|cer)\b|word.?error.?rate|character.?error.?rate|jiwer|levenshtein|edit.?distance", exclude_substr="VLMEvalKit")
asr_data = grep_count(r"aishell|librispeech|wenetspeech|test.?net|test.?meeting|dev.?clean|dev.?other|test.?clean|test.?other", exclude_substr="VLMEvalKit")
results["asr_metric_files"] = asr_metric
results["asr_dataset_ref_files"] = asr_data

# 2. Stage-3 audio output training: any training driver for NAR/AR speech decoder or codec?
stage3_train = grep_count(r"train.*(tts|codec|nar|ar.?decoder|llm2tts|vqvae)|(tts|codec|nar|llm2tts|vqvae).*train", exclude_substr="VLMEvalKit")
# narrow: training shell scripts mentioning tts/codec
train_sh = []
sd = os.path.join(REPO, "script", "train")
for fn in sorted(os.listdir(sd)):
    with open(os.path.join(sd, fn), errors="ignore") as f:
        t = f.read()
    if re.search(r"tts|codec|nar|llm2tts|vqvae|decoder", t, re.IGNORECASE):
        train_sh.append(fn)
results["stage3_training_shell_scripts"] = train_sh
results["train_shell_scripts_all"] = sorted(os.listdir(sd))

# 3. train.py: does it reference TTS/codec/speech-decoder modules for training?
with open(os.path.join(REPO, "vita", "train", "train.py"), errors="ignore") as f:
    train_py = f.read()
results["train_py_mentions_tts_codec"] = bool(re.search(r"tts|codec|llm2tts|nar|ar_decoder|speech_decoder", train_py, re.IGNORECASE))

# 4. README results table / exact paper numbers present as text? check for any of the Table-2/4 numbers
readme = open(os.path.join(REPO, "README.md"), errors="ignore").read()
paper_numbers = ["2.2", "8.4", "7.5", "70.8", "59.8"]  # ASR + headline avgs cited in README prose
results["readme_has_numbers"] = {n: (n in readme) for n in paper_numbers}

# 5. CTC speech-encoder training (Stage 2.1a): is there a training loop with CTCLoss,
#    or is every reference inference-only (ctc.eval(), frozen)?
ctc_train = grep_count(r"CTCLoss|nn\.CTCLoss|ctc_loss|loss.*ctc.*backward", exclude_substr="VLMEvalKit")
results["ctc_training_loop_files"] = ctc_train  # expect [] -> no CTC *training* code

# 6. parse_answer.py hardcoded denominator: does it add a flat constant per category
#    rather than counting actual rows/questions in the CSV?
pa = open(os.path.join(REPO, "videomme", "parse_answer.py"), errors="ignore").read()
results["parse_answer_hardcoded_num_total"] = bool(re.search(r"num_total\s*\+=\s*30", pa))
results["parse_answer_counts_rows"] = bool(re.search(r"len\(\s*cate_df\s*\)", pa))

with open(os.path.join(OUT, "artifacts.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(json.dumps(results, indent=2, ensure_ascii=False))
