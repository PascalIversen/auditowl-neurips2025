"""Documents the data flow that makes gaze_score2() a teacher-forced reconstruction
score, not a generation score. Supports finding 'teacher-forced-eval'.
No model is run; this asserts the structural facts established by reading the code:
  data.py:138  combined_text = '<image>..<image>{annotations_text}<|endofchunk|>{eos}'
  train_utils_attention.py:735  outputs = model(images, gaze, input_ids, ...)   # full seq incl. answer
  train_utils_attention.py:737  token_ids = argmax(outputs.logits)
  train_utils_attention.py:738-739  predicted_text/ground_truth both decoded from this one sequence
"""
import os
out_dir = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out_dir, exist_ok=True)
facts = [
    "input_ids fed to the model at eval time CONTAINS the ground-truth annotation text",
    "  (data.py:138: combined_text = image_tokens + annotations_text + <|endofchunk|> + eos;",
    "   there is NO separate question/prompt -- the answer itself is the sequence).",
    "gaze_score2 does a single teacher-forced forward pass: model(images, gaze, input_ids,...)",
    "  (train_utils_attention.py:735) and takes argmax of outputs.logits as 'prediction'",
    "  (train_utils_attention.py:737). Position i logits are conditioned on TRUE tokens 0..i.",
    "ground_truth = decode(input_ids); predicted_text = decode(argmax(logits)).",
    "  Both come from the SAME sequence, so the model is scored on copying the answer it was shown.",
    "Consequence: SBERT cosine in Table 1 reflects teacher-forced next-token accuracy,",
    "  not autoregressive generation. No .generate() is used in the gaze_score2 eval path.",
]
with open(os.path.join(out_dir, "teacher_forced_eval.txt"), "w") as f:
    for line in facts:
        print(line); f.write(line + "\n")
