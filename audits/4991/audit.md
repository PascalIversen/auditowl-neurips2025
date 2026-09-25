# Audit — DEAL: Data Efficient Adaptation in LLMs via Continuous Low-Rank Fine-Tuning (NeurIPS 2025, #4991)

## 1. Summary

The repo `Applied-Machine-Learning-Lab/DEAL` (single commit "Add files via upload",
no submission tag) implements DEAL, a continual-learning LoRA method. Training is a
chain of `src/T5_run_wavelet.py` / `src/Llama3_run_wavelet.py` invocations
(`scripts/*.sh`, `scripts_llama/*.sh`), one per task, each loading the previous
task's saved adapter. The CL benchmarks (15 datasets) are shipped as fixed
`train/dev/test.json` under `CL_Benchmark/`; metrics (`exact_match`→Accuracy,
`rouge1`) are computed in `src/compute_metrics.py` and aggregated per-Dataset by
the trainer. The two contributed modules are a `LinearWaveletFilter` (a Haar DWT
gate + MLP replacing each `lora_A`) and a regularization term added in
`UIETrainer.training_step`.

I read the paper (PDF), the two run scripts, the trainer, the wavelet adapter, the
dataset builder, the metrics, all task configs, and the shell scripts. I ran two
read-only checks under `_audit_code/`: an AST analysis proving the regularization
`isinstance` guard can never match (`check_reg_isinstance_mismatch.py`), and sample
counts of the shipped data (inline). No GPU code was executed; no repo file was
modified.

Three substantive findings emerged: (a) the `Eq.12` regularization is silently
never applied because the trainer's `isinstance` test references a different
`LinearWaveletFilter` class than the one inserted into the model; (b) `scripts/long.sh`
(the T5 15-task headline run) invokes the Llama script with a FLAN-T5 path and
crashes on the first task; (c) the implemented retention mechanism is a single-level
Haar DWT gate, not the SVD-truncation + heat-kernel wavelet network derived in §3.2.

## 2. Result-traceability table

The repo ships NO precomputed outputs/logs (`logs_and_outputs*` absent), so no
reported number can be matched against a stored computed value; the column "Computed
value" is "—(not run)" wherever a script exists but produces nothing checked-in.
"Repo location" gives the script that *would* compute the number.

| Paper artefact | Repo location | Computed value | Matches | Status |
|---|---|---|---|---|
| Table 1, T5+DEAL 3-Task AA 87.7 / R-1 89.3 | `scripts/TC.sh` → `T5_run_wavelet.py` + `compute_metrics.py` | —(not run) | — | script present; no output shipped |
| Table 1, T5+DEAL 4-Task AA 78.5 / R-1 82.5 | `scripts/standard.sh` → `T5_run_wavelet.py` | —(not run) | — | script present; no output shipped |
| Table 1, T5+DEAL 15-Task AA 73.9 / R-1 79.1 | `scripts/long.sh` (FLAN-T5 → `Llama3_run_wavelet.py`) | crash | ✗ | BUG: script crashes on task 1 (see `long-t5-script-crashes`) |
| Table 1, LLaMA+DEAL (all three) | `scripts_llama/*.sh` → `Llama3_run_wavelet.py` | —(not run) | — | scripts present; no output shipped |
| Table 1, SeqLoRA / O-LoRA / PerTaskFT rows | (none) | — | — | MISSING: no baseline scripts/configs in repo |
| Table 1 & Table 4 "bold = two-sided t-test p<0.05" | (none) | — | — | MISSING: no statistical-test code, no per-run logs |
| Fig 2a adapter-update (Only A / Only B / Both 75.6) | (none) | — | — | MISSING: code always trains A only, B frozen; no toggle |
| Fig 2b task-order (Order1/2/3 73.1–75.6) | `configs/order{1,2,3}_configs` + `scripts/standard.sh` | —(not run) | — | order configs present; no output shipped |
| Fig 2c LoRA-rank sweep (r=4/8/16/32) | `--lora_dim` arg in run scripts | —(not run) | — | arg supports it; no sweep script/output |
| Table 3 kernel comparison (xe^-x / splines / heat) | (none) | — | — | MISSING: only `wave='haar'` DWT implemented; no SVD/heat-kernel/alt-kernel code |
| Table 4 regularization grid (a,b) → AA 74.8–85.5 | `--theta_norm_p`,`--mlp_norm_p`,`--lambda1/2` | inert | ✗ | reg term never applied (see `reg-loss-never-applied`); grid cannot move AA |
| Table 8/9 efficiency (throughput/latency/mem) | (none) | — | — | MISSING: no timing/profiling code |
| Table 10 extra-baseline comparison (EWC/LwF/…) | (none) | — | — | MISSING: no code for these baselines |

## 3. Findings

### missing

```yaml finding
id: baselines-not-in-repo
category: missing
topic: "baselines / result traceability"
title: "No baseline (SeqLoRA, O-LoRA, PerTaskFT, EWC, LwF, …) code in repo"
severity: high
confidence: high
status: finding
file: README.md
line_start: 66
line_end: 70
quote: |
  ## Models and Usage

  ### T5 Model

  Training and evaluation scripts are in `scripts/`, outputs and logs are in `logs_and_outputs/`.
claim: "The repo ships only the DEAL training/eval pipeline (scripts/, scripts_llama/, src/*_run_wavelet.py). There is no script, config, or module that produces the SeqLoRA, O-LoRA, PerTaskFT (Table 1) or EWC/LwF/L2P/Replay/ProgPrompt/LFPT5/LB-CL (Table 10) numbers."
concern: "Every comparison in Table 1 and Table 10 — the basis for the 'consistently outperforms baselines' headline claim — relies on baseline numbers that no code in the repo computes, so the central comparison is not reproducible from this artefact."
resolution: "Authors: please add the baseline training/eval scripts and configs (or point to the exact upstream commit/command used), so the reported baseline AA/R-1 can be reproduced under the identical split/metric/budget."
cross_refs: ["reg-loss-never-applied"]
paper_ref: "Table 1; Table 10; Section 4 'Baselines'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: stat-test-not-implemented
category: missing
topic: "statistical integrity"
title: "Claimed two-sided t-test (p<0.05) significance has no code and no per-run logs"
severity: medium
confidence: high
status: finding
file: paper.pdf
quote: |
  Note. Bold indicates the statistically significant improvements
  (i.e., two-sided t-test with p < 0.05) over the best baseline.
claim: "Table 1 and Table 4 mark results as significant via a two-sided t-test; the repo contains no statistical-test code (no scipy.stats/ttest anywhere) and ships no per-seed/per-run scores. The run scripts never pass --seed, so a single default-seed run is produced per task, providing no sample for a t-test."
concern: "The significance annotation (which underwrites the 'statistically significant improvement over the best baseline' claim) cannot be reproduced or verified from the artefact, and a single deterministic run cannot support a t-test."
resolution: "Authors: provide the per-run scores, the number of runs/seeds, and the t-test script (sample definition, paired vs unpaired, one/two-sided)."
cross_refs: []
paper_ref: "Table 1 note; Table 4 note"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: ablations-not-runnable-from-repo
category: missing
topic: "ablations"
title: "Adapter-update (Fig 2a), kernel (Table 3) and efficiency (Tab 8/9) ablations absent"
severity: medium
confidence: medium
status: finding
file: src/T5_run_wavelet.py
line_start: 328
line_end: 334
quote: |
        logger.info("** wavelet-based wrapping: freeze lora_B, unfreeze lora_A. **")
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = True
            elif 'lora_B' in name:
                param.requires_grad = False
claim: "The code unconditionally trains lora_A and freezes lora_B; there is no flag to select 'Only B' or 'Both A&B' (Fig 2a), no alternative-kernel implementation (Table 3 only ships wave='haar'), and no timing/memory profiling (Tables 8/9)."
concern: "Fig 2a's 'Both A&B = 75.6' (the chosen/best configuration), Table 3's kernel comparison, and Tables 8/9 efficiency numbers cannot be reproduced because the code path they describe is not present."
resolution: "Authors: add the configuration switches and kernel variants used for Fig 2a / Table 3, and the profiling script for Tables 8/9."
cross_refs: ["reg-loss-never-applied", "wavelet-method-differs-from-paper"]
paper_ref: "Figure 2a; Table 3; Tables 8-9"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### bug

```yaml finding
id: reg-loss-never-applied
category: bug
topic: "controlled knowledge updating / regularization"
title: "Eq.12 regularization silently never applied (isinstance against wrong class)"
severity: high
confidence: high
status: finding
file: src/uie_trainer_lora.py
line_start: 95
line_end: 108
quote: |
        if "adapter" in self.path.lower():
            for name, module in model.named_modules():
                if not isinstance(module, LinearWaveletFilter):
                    continue
                
                if module.theta.requires_grad:
                    theta_norm = torch.norm(module.theta.view(-1), p=self.theta_norm_p)
                    reg_loss += self.lambda1 * theta_norm

                mlp_params = [p.view(-1) for p in module.mlp.parameters() if p.requires_grad]
                if mlp_params:
                    mlp_cat  = torch.cat(mlp_params)
                    mlp_norm = torch.norm(mlp_cat, p=self.mlp_norm_p)
                    reg_loss += self.lambda2 * mlp_norm
claim: "The trainer guards the regularization loop with `isinstance(module, LinearWaveletFilter)` where `LinearWaveletFilter` is imported from `waveletLoRAAdapter` (line 12). But the modules actually inserted into the model are instances of the *separate* `LinearWaveletFilter` class defined inside `T5_run_wavelet.py` (line 137) / `Llama3_run_wavelet.py` (line 139); those run scripts never import the adapter-module class. The two classes are distinct objects, so `isinstance` is always False, the loop body never runs, and `reg_loss` stays 0."
concern: "The paper's 'Controlled Knowledge Updating' contribution — the λ1‖θ‖^a + λ2‖MLP‖^b term of Eq.12 and the asymmetric a≥b regularization that Table 4 grid-searches — is never present in the loss, so a core advertised component is inert and Table 4 (AA varying 74.8→85.5 with (a,b)) cannot be produced by this code."
resolution: "Authors: confirm whether reg was active in the reported runs; if so, import the same LinearWaveletFilter class in both trainer and run script (or unify on one definition) and re-verify that reg_loss is non-zero during training."
cross_refs: ["ablations-not-runnable-from-repo"]
check_script: _audit_code/check_reg_isinstance_mismatch.py
paper_ref: "Eq. (12); Table 4"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: long-t5-script-crashes
category: bug
topic: "repository provenance / runnability"
title: "scripts/long.sh (T5 15-task) crashes: FLAN-T5 path matches no branch in Llama script"
severity: medium
confidence: high
status: finding
file: scripts/long.sh
line_start: 10
line_end: 17
quote: |
  model_path="google/flan-t5-large"

  for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    round=$((i + 1))
    output_dir="logs_and_outputs/long/outputs/${round}-${dataset}"

    python src/Llama3_run_wavelet.py \
claim: "scripts/long.sh runs the T5 15-task benchmark but invokes src/Llama3_run_wavelet.py with model_name_or_path=google/flan-t5-large. In that script the model/tokenizer are only assigned under `if 'adapter' in path` or `elif 'llama' in path` (lines 283-341); 'google/flan-t5-large' matches neither on the first task, so config/tokenizer/model are never bound and execution raises UnboundLocalError/NameError before training."
concern: "The T5 15-task headline result (Table 1: T5+DEAL AA 73.9 / R-1 79.1) cannot be reproduced with the provided script, since it fails on the very first task."
resolution: "Authors: provide the script actually used for the T5 15-task run (presumably T5_run_wavelet.py with the long_configs), or add a base-model branch in Llama3_run_wavelet.py."
cross_refs: []
paper_ref: "Table 1, T5 15-Task column"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### difference

```yaml finding
id: wavelet-method-differs-from-paper
category: difference
topic: "method faithfulness (knowledge retention)"
title: "Code uses a single-level Haar DWT gate, not the SVD-truncation + heat-kernel network of §3.2"
severity: high
confidence: high
status: finding
file: src/T5_run_wavelet.py
line_start: 159
line_end: 164
quote: |
        self.dwt = DWTForward(J=1, wave=self.wavelet, mode='zero').to(device)
        self.idwt = DWTInverse(wave=self.wavelet, mode='zero').to(device)
        weight_4d = self.original_weight.unsqueeze(0).unsqueeze(0).to(device)
        cA, _ = self.dwt(weight_4d)
        cA_shape = cA.shape
        self.theta = nn.Parameter(torch.ones(cA_shape, device=device))
claim: "The 'Wavelet Kernel-based Knowledge Retention' module is implemented as a single-level (J=1) Haar discrete wavelet transform of the lora_A weight matrix with an element-wise learnable gate `theta` on the approximation coefficients cA, followed (in forward, lines 184-196) by an inverse DWT and an MLP. No singular value decomposition, no truncated-SVD core-feature estimate (Eq.6-7), and no multi-scale heat-kernel wavelet network (Eq.8-10) appear anywhere in the repo (grep for svd/singular/heat/eckart returns nothing in src)."
concern: "The paper's central retention mechanism, derived in §3.2 as SVD-truncation denoising filtered by a series of learnable heat kernels at multiple scales, is a different operation from the implemented single-level Haar DWT gate, so the method evaluated is not the method described."
resolution: "Authors: clarify whether the reported results use the Haar-DWT implementation or the SVD/heat-kernel derivation in §3.2, and reconcile the paper's Eq.7-10 with the released code."
cross_refs: ["ablations-not-runnable-from-repo"]
paper_ref: "Section 3.2, Eq. (6)-(10)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: adapter-update-both-vs-A-only
category: difference
topic: "adapter update strategy"
title: "Code always trains A and freezes B, but paper's default/best is 'Both A & B'"
severity: medium
confidence: high
status: finding
file: src/T5_run_wavelet.py
line_start: 328
line_end: 333
quote: |
        logger.info("** wavelet-based wrapping: freeze lora_B, unfreeze lora_A. **")
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = True
            elif 'lora_B' in name:
                param.requires_grad = False
claim: "On every continual-learning step the code unfreezes lora_A and freezes lora_B (the wavelet/MLP module replaces lora_A only; lora_B is passed through unchanged in replace_lora_params). This is the 'Only Adapter A' configuration."
concern: "Figure 2a reports 'Both A & B' (75.6 AA) as the highest and presents it as the adopted strategy, while the released code implements 'Only A' (reported as 72.8); the main-table runs therefore use a strategy the ablation says is sub-optimal, or the released code differs from what produced Table 1."
resolution: "Authors: confirm which adapter-update strategy produced the Table 1 numbers and, if 'Both A&B', release that code path."
cross_refs: ["reg-loss-never-applied", "ablations-not-runnable-from-repo"]
paper_ref: "Figure 2a"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: train-sample-count-vs-readme
category: difference
topic: "data / experimental setup"
title: "README says 1,000 train / 500 val per task; shipped data + scripts use far more"
severity: low
confidence: high
status: finding
file: README.md
line_start: 62
line_end: 62
quote: |
  *All datasets are reformulated into an instruction-following format, with each task limited to 1,000 training and 500 validation examples. For full details on dataset preprocessing, task ordering, and prompt construction, see the Appendix in our paper.*
claim: "The README claims 1,000 train / 500 val per task, but the shipped CL_Benchmark/*/train.json hold 250-14,000 examples (e.g. dbpedia 14,000, yahoo 10,000, amazon 5,000) and the scripts set neither --max_train_samples nor --max_num_instances_per_task below its 10,000 default, so up to 10,000 train examples per task are actually used."
concern: "The 'data-efficient / small-scale dataset' framing central to the paper's motivation is not enforced by the released configuration, so the data budget actually used is much larger than stated."
resolution: "Authors: clarify the train/val sizes used for the reported numbers and set the corresponding caps in the scripts (or trim the data files)."
cross_refs: []
paper_ref: "README; Appendix B.1"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

### methodology

No standalone methodology finding beyond those routed above. The actual evaluation
protocol is sound where present: the final-task `test_tasks.json` aggregates the
held-out `test.json` of *all* prior tasks, so Average Accuracy is computed over all
tasks after the final task (matching the AA definition), with no train/test overlap
in the shipped split files. The reproducibility-threatening defects are owned by
`bug` (`reg-loss-never-applied`, `long-t5-script-crashes`), `difference`
(`wavelet-method-differs-from-paper`), and `missing` (baselines, stat test).

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 3          | high         | No baseline code (Tables 1/10); no stat-test; several ablations not runnable. |
| bug         | 2          | high         | Eq.12 regularization never applied (class-identity isinstance); T5 15-task script crashes. |
| difference  | 3          | high         | Retention is Haar DWT not SVD/heat-kernel; trains A-only vs paper's "Both"; data-size mismatch. |
| methodology | 0          | -            | Eval/AA protocol itself is sound; issues owned by other categories. |

## 5. Closing lists

### Top take-aways (≤6, by severity × confidence)
1. **[bug] `reg-loss-never-applied`** — the controlled-knowledge-updating regularization (Eq.12) is silently inert because the trainer's `isinstance` guard tests the wrong `LinearWaveletFilter` class; Table 4's regularization grid cannot move the metric. (high/high)
2. **[difference] `wavelet-method-differs-from-paper`** — the released retention module is a single-level Haar DWT gate, not the SVD-truncation + multi-scale heat-kernel network derived in §3.2. (high/high)
3. **[missing] `baselines-not-in-repo`** — no code computes any SeqLoRA/O-LoRA/PerTaskFT/EWC/LwF… baseline, so the headline "outperforms baselines" comparison is not reproducible from the artefact. (high/high)
4. **[bug] `long-t5-script-crashes`** — `scripts/long.sh` feeds a FLAN-T5 path to the Llama script, which has no matching branch and crashes on task 1; the T5 15-task headline cell is not reproducible. (medium/high)
5. **[difference] `adapter-update-both-vs-A-only`** — code always trains A only / freezes B, but Fig 2a presents "Both A & B" as best/adopted. (medium/high)
6. **[missing] `stat-test-not-implemented`** — the two-sided-t-test significance markings have no code and no per-run logs, and scripts run a single fixed-seed run. (medium/high)

### Items that genuinely look fine
- Average-Accuracy protocol: the final task's `test_tasks.json` includes the held-out `test.json` of every earlier task, so AA is over all tasks after the final task (definition in Appendix D), with no train/test leakage in the split files.
- Metric code (`compute_metrics.py`): exact-match → accuracy and ROUGE-1 F1 are computed correctly per the paper's definitions; per-dataset grouping is correct.
- `requirements.txt` is fully pinned (torch/transformers/peft/datasets etc.); LoRA rank defaults (16 for T5, 32 for Llama) and λ1=0.01/λ2=0.001 match Appendix B.2/Table 6.
- Task orders in `configs/order{1,2,3}_configs` match Table 2.

### Open questions for the authors
- Were the reported Table 1 numbers produced with the regularization active (i.e., before the `isinstance` class-identity issue), and with "Both A & B" rather than "Only A"? If so, the released code does not match the runs.
- Which implementation (Haar DWT vs §3.2 SVD/heat-kernel) generated the reported results?
- How were the t-test significance markings computed, and over how many runs/seeds?
