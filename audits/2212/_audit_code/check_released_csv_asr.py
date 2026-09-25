"""Cross-checks the released successful-jailbreak CSVs (docs/assets/data/*) against
the paper's reported final text ASRs. These CSVs contain only label==1 rows (one per
successful request), so unique idx count / 159 is an implied ASR. Supports the
traceability table and finding csv-circuitbreaking-asr-mismatch. Read-only."""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "code" / "jplhughes__bon-jailbreaking"
OUT = Path(__file__).resolve().parent / "out" / "released_csv_asr.csv"

paper_text_asr = {
    "Claude 3.5 Sonnet": 78,
    "GPT-4o": 87,            # "87% ASR for text inputs ... at N=7,200 on GPT-4o"
    "Gemini Pro": 50,
    "Llama3 8B": 94,
    "Circuit Breaking": 52,
}

df = pd.read_csv(REPO / "docs" / "assets" / "data" / "text_jailbreaks.csv")
rows = []
for model, paper in paper_text_asr.items():
    sub = df[df.model == model]
    uniq = sub["idx"].nunique()
    implied = round(uniq / 159 * 100, 1)
    verdict = "MATCH" if abs(implied - paper) <= 3 else "MISMATCH"
    rows.append({
        "model": model,
        "unique_success_idx": uniq,
        "implied_asr_pct": implied,
        "paper_asr_pct": paper,
        "verdict": verdict,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print("\nNote: all rows have label==1:",
      set(df.label.unique()) == {1.0})
