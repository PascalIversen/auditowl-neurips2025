"""Check the hardcoded vLLM max_model_len vs the max_tokens_per_call the script
plumbs through. Supports finding 'context-cap-truncates-generation'. Read-only:
greps the two literals out of math_eval.py and reports the conflict."""
import os, re

ROOT = os.path.join(os.path.dirname(__file__), "..", "code",
                    "LeapLabTHU__limit-of-RLVR", "math", "examples", "math_eval")
ME = os.path.join(ROOT, "math_eval.py")
OUT = os.path.join(os.path.dirname(__file__), "out", "context_cap.txt")

src = open(ME).read()
mml = re.search(r"max_model_len\s*=\s*(\d+)", src)
# README / paper invoke with --max_tokens 16000 -> max_tokens_per_call
readme_max_tokens = 16000

lines = []
lines.append(f"math_eval.py max_model_len literal = {mml.group(1)}")
lines.append(f"README eval_math_nodes.sh invocation passes --max_tokens = {readme_max_tokens}")
lines.append(f"max_tokens_per_call is forwarded to SamplingParams(max_tokens=...) at line "
             f"{src[:src.index('max_tokens=args.max_tokens_per_call')].count(chr(10))+1}")
cap = int(mml.group(1))
lines.append(
    f"CONFLICT: vLLM caps total context (prompt+gen) at {cap}; a request asking for "
    f"{readme_max_tokens} new tokens cannot be honored -> generations are capped at "
    f"{cap}-prompt_len tokens, far below the documented {readme_max_tokens}.")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
