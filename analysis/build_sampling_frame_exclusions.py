#!/usr/bin/env python3
"""Record why the 6 papers dropped from the 106-draw sample were excluded.

Section 2.1's inclusion rule is the gate that defines the denominator of
every headline rate ("87% release code", "8% have code for every result").
The release ships audits/ for the 100 empirical papers only; the 6 excluded
draws (audits/theory/, audits/excluded_justified_no_code/) are not
redistributed, so a reader could not previously check which draws were
dropped or why. This reads the checklist answer + justification straight out
of each excluded paper's text and commits the result, so the exclusion is
checkable without the raw PDFs.

The two categories are distinct exclusion mechanisms, not one: `theory`
papers answer NA to the checklist's code-availability question with
no-experiments language (the is_theory() signal in fetch_paper_inputs.py);
`justified_no_code` papers are empirical (No/Yes answers) but the checklist
justification for withholding code was accepted rather than triggering a
"missing code" finding -- see commit 07e7ab4.

Run: python analysis/build_sampling_frame_exclusions.py   (needs audits/theory/
and audits/excluded_justified_no_code/ present -- not part of this release)
"""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "sampling_frame_exclusions.json"

PAPERS = [
    ("theory", "1906_Simultaneous_Swap_Regret_Minimization_via_KL_Calibration"),
    ("theory", "218_Tighter_CMI_Based_Generalization_Bounds_via_Stochastic_Proje"),
    ("theory", "2788_Smoothed_Agnostic_Learning_of_Halfspaces_over_the_Hypercube"),
    ("theory", "459_Sampling_from_multi_modal_distributions_with_polynomial_quer"),
    ("justified_no_code", "1308_Neural_Collapse_is_Globally_Optimal_in_Deep_Regularized_ResN"),
    ("justified_no_code", "4393_Self_Supervised_Discovery_of_Neural_Circuits_in_Spatially_Pa"),
]

QA_RE = re.compile(
    r"open access to (?:the )?data and code.*?answer:\s*\[([^\]]*)\]\s*"
    r"justification:\s*(.*?)\s*guidelines:",
    re.I | re.S,
)


def extract_qa(text: str):
    text = unicodedata.normalize("NFKC", text)  # PDF ligatures: "Justiﬁcation" -> "Justification"
    m = QA_RE.search(text)
    if not m:
        return None, None
    answer = m.group(1).strip()
    justification = re.sub(r"\s+", " ", m.group(2)).strip()
    return answer, justification


def main():
    if not any((ROOT / "audits" / dirname).exists()
               for dirname in ("theory", "excluded_justified_no_code")):
        print(f"{ROOT / 'audits' / 'theory'} / excluded_justified_no_code not present "
              f"(not part of this release) -- refusing to overwrite {OUT.name}")
        return

    out = {}
    for category, folder in PAPERS:
        dirname = "theory" if category == "theory" else "excluded_justified_no_code"
        d = ROOT / "audits" / dirname / folder
        meta = (d / "metadata.txt").read_text()
        paper_number = re.search(r"paper_number:\s*(\S+)", meta).group(1)
        title = re.search(r"title:\s*(.+)", meta).group(1).strip()
        text = (d / "paper_text.txt").read_text(encoding="utf-8", errors="ignore")
        answer, justification = extract_qa(text)
        if answer is None:
            raise SystemExit(f"could not find checklist Q&A for {folder}")
        out[paper_number] = {
            "title": title,
            "category": category,
            "checklist_question": "Open access to data and code",
            "checklist_answer": answer,
            "checklist_justification": justification,
        }

    if len(out) != 6:
        raise SystemExit(f"expected 6 excluded papers, found {len(out)}")

    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    n_theory = sum(1 for v in out.values() if v["category"] == "theory")
    n_no_code = sum(1 for v in out.values() if v["category"] == "justified_no_code")
    print(f"wrote {OUT}: {n_theory} theory, {n_no_code} justified_no_code")


if __name__ == "__main__":
    main()
