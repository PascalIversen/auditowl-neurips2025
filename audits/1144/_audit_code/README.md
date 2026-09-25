# Audit checks for 1144 Graph Your Own Prompt

Read-only checks supporting findings in `../audit.md`.

- `out/check_math_import.txt` — AST scan: `computeweight.py` uses `math.*` and
  `logger.*` but imports neither (supports `computeweight-missing-math-import`).
- `out/check_weight_methods.txt` — imports `calculate_weight` and calls every
  weighting scheme; `sqrt`, `arccos`, `cosine` raise `NameError`
  (supports `computeweight-missing-math-import`).

Re-run:
```
cd _audit_code
python ../_audit_code/run_checks.py   # (or re-run the inline snippets from audit history)
```
The two `out/*.txt` files were produced by the inline snippets recorded in the
audit transcript; both are read-only and touch only files under `code/`.
