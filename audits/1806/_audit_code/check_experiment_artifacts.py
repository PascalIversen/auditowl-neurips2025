#!/usr/bin/env python3
"""Read-only scan: does the repo contain any experiment-reproduction artefacts
(baseline implementations, benchmark functions, seed loops, result tables)?
Supports finding `experiment-harness-missing`. Greps the whole repo (py+ipynb)
for the names of the baselines/benchmarks the paper reports."""
import pathlib, re, json

REPO = pathlib.Path(__file__).resolve().parent.parent / "code" / "brunzema__bayesqp"
OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Terms that would have to appear if the paper's experiments were reproducible here.
terms = {
    "TuRBO_baseline": r"turbo",
    "SAASBO_baseline": r"saasbo",
    "SCBO_baseline": r"scbo",
    "logEI_baseline": r"log\s*ei",
    "MPD_baseline": r"\bmpd\b|probability of descent",
    "SpeedReducer_benchmark": r"speed.?reducer",
    "Gramacy_benchmark": r"gramacy",
    "Ackley_benchmark": r"ackley",
    "RFF_within_model": r"random fourier|within.?model|rff",
    "seed_loop_32": r"32",
}

files = [p for p in REPO.rglob("*") if p.suffix in (".py", ".ipynb") and ".git" not in p.parts]
results = {}
for name, pat in terms.items():
    hits = []
    rx = re.compile(pat, re.I)
    for f in files:
        txt = f.read_text(errors="ignore")
        for i, line in enumerate(txt.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(REPO)}:{i}")
    results[name] = hits

report = {"scanned_files": [str(f.relative_to(REPO)) for f in files], "term_hits": results}
print(json.dumps(results, indent=2))
(OUT / "experiment_artifacts.json").write_text(json.dumps(report, indent=2))
