"""Counts victim-target pairs configured for the QA agent vs the 9 the paper reports. Supports finding qa-missing-pairs-config."""
import json, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "code", "dsh3n77__MINJA")
vj = json.load(open(os.path.join(ROOT, "QA", "victim.json")))
rap = json.load(open(os.path.join(ROOT, "rap", "victim_target_pair", "victim_target.json")))

out = {
    "qa_num_victim_entries": len(vj),
    "qa_victims": [e.get("victim") for e in vj],
    "qa_has_target_field": all("target" in e for e in vj),
    "rap_num_pairs": len(rap),
    "paper_qa_pairs_claimed": 9,
}
with open(os.path.join(os.path.dirname(__file__), "out", "qa_victim_pairs.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
