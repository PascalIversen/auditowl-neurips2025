"""Check which paper-reported eval artefacts (metric code, prompts, data, baselines) exist in the HiFlow repo. Supports the `missing` findings."""
import os, re, json, glob

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "Bujiazi__HiFlow")
REPO = os.path.abspath(REPO)

py_files = glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True)
all_text = ""
for f in py_files + glob.glob(os.path.join(REPO, "*.sh")) + glob.glob(os.path.join(REPO, "*.md")) + glob.glob(os.path.join(REPO, "*.txt")):
    try:
        all_text += "\n" + open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        pass

# Tokens that would indicate evaluation / data / baseline machinery
probes = {
    "FID computation":      r"\b(FrechetInceptionDistance|fid|frechet|pytorch_fid|cleanfid|InceptionV3|fid_score)\b",
    "Inception Score":      r"\b(inception_score|InceptionScore|\bIS\b)\b",
    "CLIP score":           r"\b(clip_score|CLIPScore|clip\.load|open_clip|ViT-B/32)\b",
    "patch metric":         r"\b(patch.?fid|patch.?is|FIDpatch|ISpatch|patchify.*metric)\b",
    "prompt/caption set":   r"\b(captions?\.txt|prompts?\.txt|caption_list|prompt_list|\.json.*prompt)\b",
    "LAION-High-Res data":  r"\b(LAION|laion|laion-high)\b",
    "baseline DemoFusion":  r"\b(DemoFusion|demofusion)\b",
    "baseline DiffuseHigh": r"\b(DiffuseHigh|diffusehigh)\b",
    "baseline I-Max":       r"\b(I-?Max|imax)\b",
    "baseline BSRGAN":      r"\b(BSRGAN|bsrgan)\b",
    "latency timing":       r"\b(time\.time|perf_counter|latency|timeit)\b",
}
results = {}
for name, pat in probes.items():
    hits = []
    for f in py_files:
        txt = open(f, encoding="utf-8", errors="ignore").read()
        if re.search(pat, txt):
            hits.append(os.path.relpath(f, REPO))
    results[name] = hits

out = {
    "repo_python_files": sorted(os.path.relpath(f, REPO) for f in py_files),
    "n_python_files": len(py_files),
    "eval_artifact_presence": {k: (v if v else "ABSENT") for k, v in results.items()},
}
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
outpath = os.path.join(os.path.dirname(__file__), "out", "eval_artifacts.json")
json.dump(out, open(outpath, "w"), indent=2)
print(json.dumps(out, indent=2))
