# Rater Codebook — AuditOwl human validation

You will see a set of **findings**: each claims a discrepancy between an ML paper and its
released code. For each, open the frozen repo at the cited `file:line` and the paper, then
score **two independent axes**. You are **blind to the tool's own severity/confidence** —
judge from the code and paper only. Do not discuss with other raters until adjudication.

---

## AXIS 1 — Correctness: *is the claimed discrepancy real?*

| code | label | when to use |
|---|---|---|
| **C1** | **True** | The discrepancy exists exactly as described. |
| **C2** | **Partially true** | A real discrepancy exists, but the finding overstates it, mislabels the category, or mis-attributes the cause/scope. |
| **C3** | **False** | The claim does not hold — it misreads the code, flags **intended / justified** behavior as an error ("works as intended"), or the claimed-missing code exists somewhere in the released materials (tag as *retrieval miss*, see Rules). |
| **C4** | **Unverifiable** | You cannot confirm or refute from the released materials (e.g. needs data or compute you don't have, or specialist judgment outside your field). |

Judge Axis 1 **technically** — no domain expertise needed for most (missing files, broken
imports, hardcoded paths, metric definitions). Write a one-line reason for **C3** and **C4**.

## AXIS 2 — Relevance: *does it matter?*  (leave blank / N/A if Axis 1 was C3 False)

Ask: *would it benefit reproducibility if this were fixed? Is there still a reasonable way for a
reader to reproduce or verify the affected result without too much nuisance and work? Is the code
quality acceptable for a NeurIPS paper (high-impact research code)?* Anchor to the **NeurIPS
Paper Checklist** bar (main results incl. baselines reproducible; exact commands/environment).

| code | label | when to use |
|---|---|---|
| **R1** | **Relevant** | Influences reproducibility or correctness of a main result. |
| **R2** | **Minor** | Worth a minor point in a review, but results don't depend on it or a reader can work around it with low effort. |
| **R3** | **Trivial** | Correct but easily correctable (the "effective false positive" / nitpick). |
| **RU** | **Unverifiable** | You can't tell. |
| **R0** | **N/A** | Axis 1 was C3 (False). |

Write a one-line reason for **R3** and **RU**.

---

## Worked examples (illustrative — not from any paper in the rated sample)

**Example A — C1 True + R3 Trivial (effective false positive).**
Finding: *"No script generates the Figure 4 ablation chart."* You check the repo: there is indeed
no plotting code, but the numbers behind the figure are all shipped in `results/ablation.csv`.
→ **Axis 1 = C1 True** (the script is genuinely absent).
→ **Axis 2 = R3 Trivial** — the figure can be re-plotted from the shipped numbers in a few lines;
a "reasonable avenue" to verify the result exists.
*Reason: "true, but figure trivially re-plottable from released results."*

**Example B — C1 True + R1 Relevant.**
Finding: *"None of the compared baselines is implemented in the repo."* A repo-wide scan confirms
zero baseline code; the paper's headline claim is that the method outperforms those baselines.
→ **Axis 1 = C1 True.**
→ **Axis 2 = R1 Relevant** — the comparative claim can't be reproduced at all, and NeurIPS explicitly
requires baselines to be reproducible or their omission stated.
*This is what a genuinely high-value finding looks like — contrast it with Example A.*

**Example C — C4 Unverifiable (illustrative).**
Finding: *"Theorem 2's stated rate doesn't match the constant used in the code."* The claim needs
domain judgment you don't have and the derivation isn't in the frozen materials.
→ **Axis 1 = C4 Unverifiable** → route to the matched domain expert.
*Reason: "needs specialist review of the derivation."*

---

## Rules
- **Evidence base:** you judge against the **full author release as of the audit snapshot date** —
  the repo **including all branches** plus the **NeurIPS supplemental ZIP** (both provided to you).
  If code the finding claims is missing exists anywhere in that set, score **C3** and note
  *"retrieval miss"* in your reason (vs *"reasoning error"* for a misread of code the tool had).
  Code released **after** the snapshot date does not count — note it, but don't let it flip your verdict.
- **Blind to severity/confidence.** Don't seek out the tool's labels.
- **Independent.** No discussion until adjudication.
- **When torn between C1 and C2**, use C2 if the *core* discrepancy is real but a detail is off.
- **When torn between R1 and R2/R3**, ask the NeurIPS "reasonable avenue to reproduce/verify" test;
  if a competent reader is left unable to reproduce or verify a claim the paper *makes*, it's R1.
- **Default to C4/RU over guessing.** An honest "unverifiable" is more useful than a coin-flip.
