"""Checks that QA/main.py hardcodes model='gpt-4o' in llm() and ignores --core_model. Supports finding qa-model-hardcoded."""
import re, os, json

ROOT = os.path.join(os.path.dirname(__file__), "..", "code", "dsh3n77__MINJA")
qa_main = os.path.join(ROOT, "QA", "main.py")
src = open(qa_main).read()

out = {}
# the active llm() helper
m = re.search(r'def llm\(prompt\):.*?model="([^"]+)"', src, re.DOTALL)
out["llm_hardcoded_model"] = m.group(1) if m else None
# CLI arg default
m2 = re.search(r'--core_model".*?default="([^"]+)"', src)
out["core_model_default"] = m2.group(1) if m2 else None
# is args.core_model used anywhere except its definition?
uses = [i+1 for i, line in enumerate(src.splitlines()) if "core_model" in line]
out["core_model_line_numbers"] = uses
out["core_model_used_in_llm_call"] = "model=" in src and "core_model" in src.split("def llm")[1].split("def ")[0]

with open(os.path.join(os.path.dirname(__file__), "out", "qa_model_hardcode.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
