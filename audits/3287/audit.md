# Audit — CARMANIA (NeurIPS 2025, paper #3287)

## 1. Summary

CARMANIA augments next-token (NT) pretraining of a LLaMA-style genomic transformer
with a **transition-matrix (TM) loss** that aligns the model's predicted bigram
distribution with the empirical per-sequence 4×4 nucleotide bigram matrix. The paper
reports an extensive empirical program: pretraining on three datasets; ablations on
TM loss, β, attention window, Markov order, and architecture; 5-fold CV on the
Genomic Benchmark (Table 2); 10-fold CV on 18 Nucleotide-Transformer tasks (Table 3);
FAISS best-hit retrieval on Scorpio-Gene-Taxa and AMR (Tables 4, 5); BGC classification
(Table 6); protein/HyenaDNA/Caduceus cross-architecture studies (Tables 7, 8); and
figures for training curves, inference time, window size, and long-range Hamming
retention (Figs 2–8).

The cloned repo (`code/EESI__carmania/`, commit `2c6bddb`, no tags) contains **only**
the model definition (`carmania/{configuration,model,modeling_carmania,tokenization,loss}.py`),
a single hardcoded pretraining loop (`train.py`), two sequence-*generation* demo
notebooks, a README, and a logo. There is **no** fine-tuning code, cross-validation
harness, FAISS retrieval, metric computation, baseline code, data download, figure/table
generation, second-order-TM variant, or β sweep, and **no dependency specification**.
The repo therefore reproduces essentially none of the paper's reported numbers.

What I ran (read-only, under `_audit_code/`):
- `check_repo_coverage.py` — lists git-tracked files and greps every tracked `.py`
  for tokens that would indicate result-producing code (CV, FAISS, metrics, plotting,
  2nd-order TM, data download, baselines, Hamming retention, β sweep, dependency files).
  Output: `out/repo_coverage.txt` — all such categories ABSENT; no dependency spec.
- `check_tmloss_shapes.py` — reproduces the `TMLoss` einsum + `[:, :-1, :-1]` slice
  against the (B,4,4) ground-truth bigram built by `train.py`, to confirm the slice is
  a benign PAD-drop (not a shape bug). Output: `out/tmloss_shapes.txt` — shapes align,
  KL finite.

The TM-loss core (`carmania/loss.py`) and the architecture config both look correct and
match the paper's description (see "Items that genuinely look fine").

## 2. Traceability table

Repo paths are relative to `code/EESI__carmania/`. "(none)" = no code in the repo
computes the value.

| Paper artefact                                              | Repo location | Computed value | Matches paper | Status |
|------------------------------------------------------------|---------------|----------------|---------------|--------|
| Eq. 5 / §3.2 TM loss (KL of pred vs empirical bigram)      | carmania/loss.py:9-27 | code present (produces loss tensor at train time) | n/a (no scalar reported) | Verified (implements described loss) |
| §3.2 / Eq. 6 full loss `L = L_NT + βL_TM`                  | train.py:84-90 | code present | ✓ | Verified |
| Architecture (5 layers, hidden 1024, MLP 4608, 16/4 heads, win 128, SiLU; Table 12 wide) | carmania/configuration_carmania.py:7-22 | matches | ✓ | Verified |
| Fig. 2 NT-loss & TM-loss curves (β=0 vs β=1, 31k steps)    | (none) | — | — | MISSING (no logging/plot script; β not exposed) |
| Table 1 ablation (F1 0.873/0.882/0.883; BLEU; rel-FLOPs)   | (none) | — | — | MISSING (no eval/FLOPs code) |
| Table 2 Genomic Benchmark 5-fold CV (8 tasks)             | (none) | — | — | MISSING (no fine-tune/CV/metric code) |
| Table 3 NT-tasks 10-fold CV (18 tasks, MCC/F1/acc)        | (none) | — | — | MISSING (no fine-tune/CV/metric code) |
| Table 4 Scorpio-Gene-Taxa (FAISS retrieval, 3 splits)     | (none) | — | — | MISSING (no FAISS/eval code) |
| Table 5 AMR F1 macro (FAISS best-hit)                     | (none) | — | — | MISSING (no FAISS/eval code) |
| Table 6 BGC 5-fold CV accuracy (0.484 etc.)              | (none) | — | — | MISSING (no eval code) |
| Tables 7/8 protein & Caduceus TM studies                  | (none) | — | — | MISSING |
| Table 9 β sensitivity (0/0.5/1.0/5.0)                     | (none) | — | — | MISSING (β not a sweepable arg; default β=1) |
| Tables 10/11 second-order TM                              | (none) | — | — | MISSING (loss is first-order only) |
| Table 14/15 model-size & seq-len scaling, rel-FLOPs       | (none) | — | — | MISSING |
| Table 16 ConvNova comparison (5 seeds)                    | (none) | — | — | MISSING |
| Fig. 3 inference-time / window-size                       | (none) | — | — | MISSING |
| Fig. 4 long-range Hamming-similarity retention            | (none) | — | — | MISSING |
| Figs 5/8 t-SNE embeddings                                 | (none) | — | — | MISSING |
| Table 13 training hyperparams (wd 0.2, eps 1e-6, betas, warmup 400, grad-clip 0.85/2, 2 epochs) | train.py:46-71 | wd=default(0.01), no eps/betas override, warmup default 100, **no grad clipping**, epochs default 4 | ✗ | MISMATCH (see `train-hparams-mismatch`) |

Every numbered table/figure value in the paper routes to a MISSING row. They are owned
collectively by the `missing` findings below rather than re-filed per row.

## 3. Findings

## missing

```yaml finding
id: eval-pipeline-missing
category: missing
topic: "result traceability / evaluation code"
title: "No fine-tuning, CV, retrieval, or metric code — none of Tables 1-16 / Figs 2-8 reproducible"
severity: high
confidence: high
status: finding
file: train.py
line_start: 109
line_end: 112
quote: |
  # === Run ===
  if __name__ == "__main__":
      fasta_path = "./train.fasta" 
      train(fasta_path)
claim: "The only executable entrypoint in the repo is a single self-supervised pretraining loop; the repo contains no classification fine-tuning head, no 5-/10-fold cross-validation harness, no FAISS best-hit retrieval, no metric computation (F1/MCC/accuracy/BLEU/perplexity), and no baseline code."
concern: "Every quantitative result in the paper (Tables 1-16, Figs 2-8) is produced by downstream evaluation that is entirely absent from the repo, so no headline number can be reproduced or checked from the released code."
resolution: "Authors: please release the fine-tuning + cross-validation + FAISS retrieval + metric scripts (with exact commands) that produced Tables 2-6 and the ablation tables, or state where they live."
cross_refs: ["dependency-spec-missing", "figure-table-scripts-missing", "second-order-tm-missing"]
check_script: _audit_code/check_repo_coverage.py
paper_ref: "Tables 1-16, Figures 2-8; Checklist Q5 'we will release the code ... upon acceptance'"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: dependency-spec-missing
category: missing
topic: "expected code completeness / dependencies"
title: "No requirements.txt / setup.py / environment.yml — environment cannot be rebuilt"
severity: medium
confidence: high
status: finding
file: train.py
line_start: 1
line_end: 16
quote: |
  import os
  import torch
  import wandb
  from torch.utils.data import Dataset, DataLoader
  from torch.cuda.amp import GradScaler, autocast
  from transformers import get_scheduler
  from Bio import SeqIO
  from typing import List, Tuple, Optional
  import numpy as np
  from torch import nn
  from carmania.tokenization_carmania import CarmaniaTokenizer
  from carmania.modeling_carmania import CarmaniaModel
  from carmania.configuration_carmania import CarmaniaConfig
  from carmania.loss import TMLoss
  from tqdm import tqdm
  from transformers import AutoModel
claim: "The code imports torch, transformers, wandb, Bio (Biopython), numpy, tqdm, and optionally flash_attn (modeling_carmania.py:24-36), but the repo ships no requirements.txt, setup.py, pyproject.toml, or environment.yml pinning any version."
concern: "Without a dependency specification the training/inference environment cannot be reconstructed, and unpinned transformers/torch versions can change tokenizer and attention behaviour."
resolution: "Authors: add a pinned dependency file (versions of torch, transformers, flash-attn, biopython, wandb)."
cross_refs: ["eval-pipeline-missing"]
check_script: _audit_code/check_repo_coverage.py
paper_ref: "Checklist Q5 (open access to code)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: second-order-tm-missing
category: missing
topic: "ablations / higher-order TM"
title: "Second-order TM ablation (Tables 10/11) absent — loss is first-order only"
severity: medium
confidence: high
status: finding
file: carmania/loss.py
line_start: 14
line_end: 24
quote: |
  def forward(self, logits, true_probs):
      probs = F.softmax(logits, dim=2)
      p1, p2 = probs[:, :-1, :], probs[:, 1:, :]
      pred_bigram = torch.einsum('bti,btj->bij', p1, p2)

      pred_bigram = pred_bigram[:, :-1, :-1]
      row_sums = pred_bigram.sum(dim=-1, keepdim=True).clamp_min(1)
      pred_bigram = pred_bigram / row_sums

      pred_bigram += self.epsilon
      true_probs += self.epsilon
claim: "TMLoss only ever forms a first-order (bigram) transition matrix; there is no code path, flag, or tensor formulation for the second-order (trigram) TM loss the paper ablates."
concern: "The paper's claim that first-order TM beats second-order TM (Tables 10, 11) cannot be reproduced because the second-order variant is not in the repo."
resolution: "Authors: release the higher-order TM implementation used for Tables 10/11."
cross_refs: ["eval-pipeline-missing"]
check_script: _audit_code/check_repo_coverage.py
paper_ref: "Section A.5, Tables 10 and 11"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: figure-table-scripts-missing
category: missing
topic: "result traceability / figures"
title: "No data-loading or plotting code for any pretraining dataset or figure"
severity: medium
confidence: high
status: finding
file: train.py
line_start: 18
line_end: 34
quote: |
  class DNASequenceDataset(Dataset):
      def __init__(self, fasta_path: str, tokenizer: CarmaniaTokenizer):
          self.input_ids = []
          self.bigrams = []
          print("Loading and tokenizing sequences...")
          for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
              seq = str(record.seq)
              token_ids, bigram_matrix = tokenizer.encode_with_bigram(seq)
              self.input_ids.append(token_ids)
              self.bigrams.append(bigram_matrix)

          self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
          self.bigrams = torch.tensor(self.bigrams, dtype=torch.float32)
          row_sums = self.bigrams.sum(axis=2, keepdims=True) 
          self.bigrams = self.bigrams / row_sums
          
          print(f"Loaded {len(self.input_ids)} sequences.")
claim: "Training reads a single local FASTA (`./train.fasta`); there is no script to download or build the GRCh38 / Basic-Genome / Scorpio-Gene-Taxa / MiBiG / AMR datasets (Tables 17-18), and no code generating any figure (training curves, inference-time, window-size, Hamming retention, t-SNE)."
concern: "Reviewers cannot obtain the pretraining or downstream datasets in the expected format, nor regenerate any figure, from the repo alone."
resolution: "Authors: provide dataset fetch/preprocessing scripts (or resolvable accessions in the train.fasta format) and the figure-generation code."
cross_refs: ["eval-pipeline-missing"]
check_script: _audit_code/check_repo_coverage.py
paper_ref: "Tables 17-18; Figures 2-8"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

No technical bugs found. The one slice that looked suspicious — `pred_bigram[:, :-1, :-1]`
in `carmania/loss.py:19` — was verified to be a deliberate drop of the PAD row/column so the
predicted matrix matches the (4×4) nucleotide-only ground truth (see `_audit_code/out/tmloss_shapes.txt`).
The `train.py` hardcoded path `./train.fasta` is a usability nuisance, not a defect contradicting
the code's own intent.

## difference

```yaml finding
id: train-hparams-mismatch
category: difference
topic: "training hyperparameters"
title: "train.py omits weight decay, grad clipping, eps/betas, and uses different warmup/epoch defaults than Table 13"
severity: low
confidence: high
status: finding
file: train.py
line_start: 46
line_end: 71
quote: |
  def train(
      fasta_path,
      batch_size=32,
      epochs=4,
      learning_rate=5e-4,
      model_name="carmania",
      beta= 1 ,
      seq_length = 2000,
      num_warmup_steps = 100,
      fp16= True,
      device="cuda:0"
  ):
      wandb.init(project="carmania")

      # Setup
      config = CarmaniaConfig(seq_length=seq_length)
      tokenizer = CarmaniaTokenizer(model_max_length=config.seq_length, calculate_bigram=True)
      dataset = DNASequenceDataset(fasta_path, tokenizer)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
      model = CarmaniaModel(config).to(device)
      TM1_loss = TMLoss()
      NT_loss = nn.CrossEntropyLoss(ignore_index=4) # [PAd]==4

      optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
      scaler = torch.amp.GradScaler(enabled=fp16)
      scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(dataloader)*epochs)
claim: "The AdamW optimizer is created with only `lr` (so weight_decay=0.01 default, eps=1e-8, betas=(0.9,0.999)); there is NO gradient clipping in the training loop; and the defaults are epochs=4, num_warmup_steps=100 — whereas paper Table 13 specifies weight_decay=0.2, Adam epsilon=1e-6, warmup 400, max-grad-norm 0.85/2, and 2 epochs. There is no argparse/CLI, so these are run as-is."
concern: "The released training script does not reproduce the paper's documented training recipe (notably grad clipping and weight decay are entirely absent), so a re-run from the repo uses a different optimizer configuration than reported."
resolution: "Authors: align train.py defaults with Table 13 (add weight_decay, eps, gradient clipping, warmup=400, epochs=2) or expose them as arguments."
cross_refs: []
paper_ref: "Table 13 (Parameter Ranges for Model Training)"
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## methodology

No methodology finding is supportable from the released code. The paper's evaluation
protocol (5-/10-fold CV, FAISS best-hit retrieval, train/test splits, baseline tuning)
is **not present in the repo** (owned by `eval-pipeline-missing`), so per Rule B I cannot
assess whether splits leak, baselines are fairly tuned, or metrics fit the class balance —
those would be speculation from prose alone. The TM-loss objective and architecture that
*are* present are methodologically coherent. See "Open questions for the authors".

N/A scopes: Temporal integrity — N/A (no time-series data; genomic position is spatial,
splits are by sequence). Pretraining contamination — partially N/A: the model is the paper's
own contribution (pretrained from scratch); no external pretrained encoder is used, though
train/test overlap between pretraining and downstream genomes is unassessable without the
eval code.

## 4. Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 4          | high         | Entire downstream eval/CV/retrieval/figure/2nd-order-TM/dependency stack absent; no paper number reproducible. |
| bug         | 0          | -            | Suspicious PAD slice verified benign; no defects found. |
| difference  | 1          | low          | train.py drops weight decay + grad clipping and uses different warmup/epoch defaults vs Table 13. |
| methodology | 0          | -            | Eval protocol not in repo; cannot assess leakage/baselines/metrics without speculation. |

### Top take-aways (≤6, by severity × confidence)
1. **[missing] `eval-pipeline-missing`** (high/high): the repo is model + a single pretraining
   loop only; no fine-tuning, CV, FAISS retrieval, metric, or baseline code exists, so none of
   Tables 1-16 or Figs 2-8 can be reproduced.
2. **[missing] `dependency-spec-missing`** (medium/high): no requirements/setup/environment file;
   environment is not rebuildable and key deps are unpinned.
3. **[missing] `second-order-tm-missing`** (medium/high): only first-order TM exists; the
   second-order ablation (Tables 10/11) has no code.
4. **[missing] `figure-table-scripts-missing`** (medium/high): no dataset fetch/preprocessing
   or figure-generation code for any pretraining/downstream dataset or figure.
5. **[difference] `train-hparams-mismatch`** (low/high): released train.py omits weight decay
   and gradient clipping and uses different warmup/epoch defaults than Table 13.

### Items that genuinely look fine
- **TM loss** (`carmania/loss.py:9-27`): the einsum bigram, the `[:, :-1, :-1]` PAD-drop, and
  KL formulation match Appendix B and Eq. 5; shapes align with the (4×4) ground truth
  (`_audit_code/out/tmloss_shapes.txt`).
- **Architecture config** (`carmania/configuration_carmania.py:7-22`): 5 layers, hidden 1024,
  intermediate 4608, 16/4 heads, window 128, SiLU — matches the "wide" model in Table 12.
- **Ground-truth bigram construction** (`carmania/tokenization_carmania.py:67-86`,
  `train.py:31-32`): per-sequence 4×4 bigram counts, row-normalized, excluding PAD — matches §4.1.
- **Pretrained weights**: three CARMANIA checkpoints are publicly hosted on Hugging Face
  (README.md:19-21), partially satisfying the trained-weights requirement.

### Open questions for the authors
- Where are the fine-tuning / cross-validation / FAISS-retrieval scripts that produced
  Tables 2-6? Do the downstream splits keep related genomic fragments (same genome/species)
  disjoint across train/test, or can fragments of one genome appear in both? (Unverifiable
  without the eval code — `eval-pipeline-missing`.)
- For the FAISS best-hit retrieval (Tables 4/5), is the query item excluded from the index
  it is matched against (no self-match leakage)?
- How were the AMR/Genomic-Benchmark baselines (HyenaDNA, Caduceus, etc.) tuned relative to
  CARMANIA — same budget, same splits?
