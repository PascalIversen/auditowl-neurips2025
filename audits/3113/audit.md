# Audit — L-MTP: Leap Multi-Token Prediction (NeurIPS 2025, paper 3113)

## Summary

The repo (`github.com/Xiaohao-Liu/L-MTP`, single commit `ffe8cf5`, "update") contains a
custom L-MTP model implementation (`models/lmtp/`), a vanilla baseline (`models/vanilla/`),
KV-cache-patched HuggingFace backbones (`models/base/`), head-accuracy / tree-construction
eval scripts (`eval/`), per-model training configs (`configs/`), and a vendored copy of
LLaMA-Factory used as the training driver. L-MTP is trained with HuggingFace/LLaMA-Factory's
standard `compute_loss`, which calls `LMTPModel.forward(labels=...)` →
`LMTPModel.loss_function` to add the multi-head leap loss.

I read the paper (PDF + `paper_text.txt`) including the appendix pseudo-code (Appendix B.1) and
the methodology equations (Eq. 4–5), then read the core code: `models/lmtp/model.py`,
`models/lmtp/head.py`, `models/lmtp/config.py`, `models/vanilla/model.py`, `models/__init__.py`,
`LLaMA-Factory/src/llamafactory/model/loader.py`, the `configs/`, and the `eval/` scripts. I
ran two read-only checks under `_audit_code/`: `check_medusa_import.py` (AST scan confirming the
imported `models/medusa` module is absent) and `check_leap_positions.py` (compares the code's
per-head label offsets to the paper's leap positions). Task accuracies in Table 1 are computed by
external harnesses (lm_eval / evalplus / MathRuler) per the README and are not in the repo; this is
acceptable, but the speedup measurement harness (Fig. 6, Table 2) is also absent, and the head-
accuracy plotting path (`eval/get_all.py`, `eval/plot_tree.py`) cannot import.

The most consequential issues: (1) the package entrypoint `models/__init__.py` and two eval
scripts import a `models/medusa` module that does not exist in the repo, so importing `models`
(required by both training via `loader.py` and head-accuracy eval) raises `ModuleNotFoundError`;
(2) the leap stride is hardcoded `skip_token = 3` while the paper's default is `k = 2`; (3) all
training configs use `num_heads: 3` while the paper's default is `n = 4`; (4) a hardcoded absolute
path `/home/storage/LMTP` is required for the LLaMA-Factory training integration to find the
`models` package.

## Result-traceability table

| Paper artefact | Repo location | Computed value | Matches paper | Status |
|---|---|---|---|---|
| Table 1 task accuracies / pass@1 (all models, k=2,n=4) | external harnesses (lm_eval/evalplus/MathRuler), README; no repo script | — | — | MISSING-BY-DESIGN (external eval; acceptable but not reproducible from repo, and k/n defaults differ — see findings) |
| Table 2 / Fig. 6 speedup ratios (tokens/s) | `models/lmtp/model.py` `stream_generate` yields `accepted_tokens`; no driver times generation or aggregates tokens/s | — | — | MISSING (no speedup-measurement script) |
| Fig. 7 / Fig. 8 / Fig. 9 per-head prediction accuracy | `eval/head_accuracy.py` (computes), `eval/get_all.py` (driver) | (not run; needs GPU + checkpoints) | — | PARTIAL — compute code present but `eval/get_all.py:5` import fails (see missing-medusa-module) |
| Fig. 11 token tree / `accept_nodes` | `eval/gen_results.py:explore_graph`, `eval/get_all.py` | (not run) | — | PARTIAL — present but same broken import |
| L-MTP loss (Eq. 4/5), leap stride k=2 | `models/lmtp/model.py:198-226` | implements skip_token=3 stride | ✗ (k=2 vs 3) | MISMATCH (see skip-token-mismatch) |
| Theorem 3 / Fig. 5 simulated curves | (none — no simulation script) | — | — | MISSING (theory illustration not in repo; low severity) |

## missing

```yaml finding
id: speedup-measurement-missing
category: missing
topic: "result traceability / inference speedup"
title: "No script measures the tokens/s speedup ratios reported in Fig. 6 and Table 2"
severity: medium
confidence: medium
status: finding
file: models/lmtp/model.py
line_start: 432
line_end: 440
quote: |
            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ) if not no_decode else "",
                "accepted_tokens": accept_length + 1,
            }
claim: "stream_generate / ea_generate yield per-step accepted_tokens, but no script in the repo times generation or aggregates tokens-per-second into the speedup ratios reported in Fig. 6 (per-model speedups) or Table 2 (Medusa-extension 1.83x..2.43x)."
concern: "The headline efficiency claims (up to 4x faster, +22% over MTP, Table 2 ratios) cannot be reproduced from the repo because the timing/aggregation harness that produces them is absent."
resolution: "Authors: please add the driver that runs generation under each paradigm, measures wall-clock tokens/s, and computes the reported speedup ratios."
cross_refs: []
check_script: _audit_code/out/medusa_import_check.txt
paper_ref: "Figure 6; Table 2"
tags: [reforms:2, heil:silver]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: model-package-no-deps
category: missing
topic: "dependencies / completeness"
title: "Top-level L-MTP package (models/, eval/) ships no dependency specification"
severity: low
confidence: high
status: finding
file: models/lmtp/model.py
line_start: 1
line_end: 22
quote: |
  import transformers
  from .head import Head
  from safetensors.torch import save_file, load_file
  import os
  from typing import Callable, Optional, Tuple, Union

  from ..base import AutoModel
  import torch
  import torch.nn as nn

  from ..base.kv_cache import initialize_past_key_values
  from transformers.generation.utils import GenerateDecoderOnlyOutput, CausalLMOutputWithPast
  from ..utils import prepare_logits_processor, reset_tree_mode, reset_past_key_values,  evaluate_posterior, generate_candidates
  from ..utils import initialize_tree, tree_decoding, generate_tree_buffers, update_inference_inputs
  from ..utils_eg import initialize_tree as initialize_tree_eagle, tree_decoding as tree_decoding_eagle, update_inference_inputs as update_inference_inputs_eagle

  from peft import PeftModel, PeftConfig

  from transformers.modeling_utils import PreTrainedModel
  from .config import LMTPConfig

  from transformers.loss.loss_utils import ForCausalLMLoss
claim: "The repo root has no requirements.txt / environment.yml for the custom L-MTP code, which depends on safetensors, peft, fastchat, fire, networkx, and an unspecified transformers version exposing transformers.loss.loss_utils.ForCausalLMLoss; only LLaMA-Factory has its own requirements.txt."
concern: "Without a pinned environment for the custom code (notably the exact transformers version that exposes transformers.loss.loss_utils), the L-MTP model is not guaranteed to import/run on a fresh machine."
resolution: "Authors: provide a requirements/lock file (or pin transformers) for the top-level models/ and eval/ code."
cross_refs: []
paper_ref: "Appendix F (Reproducibility)"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## bug

```yaml finding
id: missing-medusa-module
category: bug
topic: "broken imports / package entrypoint"
title: "models/__init__.py and eval scripts import a models/medusa module that does not exist"
severity: high
confidence: high
status: finding
file: models/__init__.py
line_start: 3
line_end: 6
quote: |
  from .lmtp.model import LMTPModel
  from .medusa.model import MedusaModel
  from .medusa.model_official import MedusaModel as MedusaModelOfficial
  from .vanilla.model import Vanilla
claim: "The package entrypoint imports from .medusa.model and .medusa.model_official, but models/medusa/ is absent from the repo (git ls-files lists no medusa file; only lmtp, vanilla, base under models/). eval/get_all.py:5 and eval/plot_tree.py:5 import the same missing module."
concern: "Any `import models` / `from models import get_mtp_model` — required by LLaMA-Factory training (loader.py:171) and by the head-accuracy eval (get_all.py) — raises ModuleNotFoundError, so neither training nor head-accuracy evaluation runs as shipped."
resolution: "Authors: add the missing models/medusa/ package or remove the dead medusa imports from models/__init__.py, eval/get_all.py, and eval/plot_tree.py."
cross_refs: ["model-package-no-deps"]
check_script: _audit_code/check_medusa_import.py
paper_ref: "Appendix F (Reproducibility); README Inference section"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: hardcoded-storage-path
category: bug
topic: "hardcoded absolute path"
title: "LLaMA-Factory training integration hardcodes /home/storage/LMTP on sys.path"
severity: high
confidence: high
status: finding
file: LLaMA-Factory/src/llamafactory/model/loader.py
line_start: 168
line_end: 183
quote: |
    if model_args.mtp_type is not None and model_args.mtp_type != "vanilla":
        import sys
        sys.path.append(os.path.join("/home/storage/LMTP"))
        from models import get_mtp_model

        model = get_mtp_model(
            model_args.mtp_type,
            model_args.model_name_or_path,
            tokenizer,
            model,
            train_mode=is_trainable,
            num_head=model_args.num_heads,
            head_weight=model_args.head_weight,
            stage1_pretrained_path=model_args.stage1_pretrained_path,
            train_lm_head=model_args.train_lm_head
        )
claim: "The only place that wires L-MTP into LLaMA-Factory training appends a hardcoded absolute path '/home/storage/LMTP' to sys.path to locate the `models` package, and every stage-1/stage-2 config sets stage1_pretrained_path under '/home/storage/LMTP/saves/...'."
concern: "On any machine where the repo is not checked out at /home/storage/LMTP, `from models import get_mtp_model` fails (or imports nothing), so the documented training commands (README) do not run as shipped."
resolution: "Authors: make the path configurable/relative (e.g. derive from repo root or an env var) and parameterise the stage1_pretrained_path defaults in the configs."
cross_refs: ["missing-medusa-module"]
paper_ref: "README Training section"
tags: [reforms:2, heil:bronze]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

## difference

```yaml finding
id: skip-token-mismatch
category: difference
topic: "leap stride hyperparameter"
title: "Code hardcodes leap stride skip_token=3; paper default is k=2"
severity: medium
confidence: high
status: finding
file: models/lmtp/model.py
line_start: 219
line_end: 224
quote: |
        skip = self.skip_token
        for i, logits_ in enumerate(logits):
            h_logits = logits_[:, : -(skip*(i+1))].contiguous()
            h_labels = labels[..., skip*(i+1) :].contiguous()
            loss_i = self.loss_fct(logits=h_logits, labels=h_labels, vocab_size=self.config.vocab_size, **kwargs)
            loss += loss_i * alpha
claim: "loss_function uses self.skip_token as the leap stride; self.skip_token is hardcoded to 3 at model.py:128 (`self.skip_token = self.config.skip_token = 3`) and is never overridden by any config (configs set only mtp_type/num_heads/head_weight) or by get_mtp_model. The paper states 'We set k = 2 and n = 4 by default' (Section 5.1)."
concern: "The shipped default trains additional heads at offsets {3,6,9} (skip=3) rather than the paper's k=2 leaping pattern, so re-running the configs as-is does not reproduce the paper's default L-MTP setting; the head's leap positions differ from Eq. (4)/(5) with k=2."
resolution: "Authors: confirm which k was used for the reported Table 1 numbers, and expose skip_token (k) as a config field instead of hardcoding 3 in LMTPModel.__init__."
cross_refs: ["num-heads-mismatch"]
check_script: _audit_code/check_leap_positions.py
paper_ref: "Section 5.1 (Implementation details): 'We set k = 2 and n = 4 by default.'"
tags: [reforms:3, forensics:post-hoc-selection]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: num-heads-mismatch
category: difference
topic: "number of prediction heads"
title: "All training configs set num_heads: 3; paper default is n = 4"
severity: low
confidence: high
status: finding
file: configs/s2/qwen25_7B_base_lmtp_stage2.yaml
line_start: 4
line_end: 6
quote: |
  mtp_type: lmtp2
  num_heads: 3
  head_weight: 0.1
claim: "Every stage-1 and stage-2 config in configs/ sets num_heads: 3 (verified across all 12 yaml files), whereas the paper's default is n = 4 (Section 5.1). The eval drivers (eval/get_all.py, eval/head_accuracy.py) also default num_head=3."
concern: "The shipped configs do not match the paper's default n=4 setting used for the headline Table 1 numbers, so the provided configs reproduce a different (n=3) configuration than reported."
resolution: "Authors: clarify whether Table 1 used n=4 (paper) or n=3 (configs), and ship configs matching the reported default."
cross_refs: ["skip-token-mismatch"]
paper_ref: "Section 5.1 (Implementation details): 'We set k = 2 and n = 4 by default.'"
tags: [reforms:3, forensics:post-hoc-selection]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: true
```

```yaml finding
id: leap-start-offset
category: difference
topic: "leap position indexing"
title: "Additional heads trained at offsets k, 2k, 3k rather than paper's k+1, 2k+1, 3k+1"
severity: low
confidence: medium
status: question
file: models/lmtp/model.py
line_start: 220
line_end: 222
quote: |
        for i, logits_ in enumerate(logits):
            h_logits = logits_[:, : -(skip*(i+1))].contiguous()
            h_labels = labels[..., skip*(i+1) :].contiguous()
claim: "The additional heads (0-indexed i) are supervised on tokens at offset skip*(i+1) ahead, i.e. {k, 2k, 3k}. The paper's Eq.(4)/(5) place leaping heads at t+k(i-1)+1, i.e. additional-head offsets {k+1, 2k+1, 3k+1}; for k=2 these are {3,5,7} but the code (even with skip=2) supervises {2,4,6}."
concern: "The implemented leap positions are uniformly shifted by one token relative to the paper's stated formula; both are valid 'leap' patterns, so this is a faithfulness mismatch rather than an invalid procedure, but it means the trained heads do not target the exact positions described."
resolution: "Authors: confirm the intended leap offsets and reconcile the code's skip*(i+1) indexing with Eq.(4)/(5)'s t+k(i-1)+1."
cross_refs: ["skip-token-mismatch"]
check_script: _audit_code/check_leap_positions.py
paper_ref: "Eq. (4) and Eq. (5); Section 3.1"
tags: [reforms:3]
validator_pass:
  quote_match: true
  control_flow: true
  condition_satisfiable: false
```

## methodology

N/A — within the scope auditable from the code, no leakage / split / metric-validity problems
were found. This is a generative-LLM efficiency paper evaluated on standard public benchmarks
(MATH500, GSM8K, MBPP(+), HumanEval(+), MMLU, IFEval) via external harnesses; there is no
train/test split constructed in-repo, no constructed-negative or pair task, and no pretraining-
contamination claim to bound (the base LLMs are off-the-shelf and used as the NTP baseline too).
The provided NTP and MTP baselines share the model/training code (`models/vanilla/`,
`models/lmtp/`), and the paper states the same training setting is used for MTP "to ensure
fairness", consistent with MTP being L-MTP with stride 1.

## Scoreboard

| Category    | # findings | Max severity | Note (one line) |
|-------------|------------|--------------|-----------------|
| missing     | 2          | medium       | No speedup-measurement harness (Fig. 6 / Table 2); no deps spec for custom package. |
| bug         | 2          | high         | `models/medusa` import is dead (breaks training + head-eval); hardcoded `/home/storage/LMTP` path. |
| difference  | 3          | medium       | skip_token=3 vs paper k=2; configs num_heads=3 vs paper n=4; leap offset shifted by 1. |
| methodology | 0          | -            | No leakage/split/metric issues found within code scope. |

## Top take-aways

- **[bug, high]** `missing-medusa-module`: `models/__init__.py` (and `eval/get_all.py`,
  `eval/plot_tree.py`) import a non-existent `models/medusa` package, so `import models` raises
  `ModuleNotFoundError` — breaks both the training path (via `loader.py`) and head-accuracy eval.
- **[bug, high]** `hardcoded-storage-path`: training only finds the `models` package via a
  hardcoded absolute `/home/storage/LMTP` on `sys.path`; the documented train commands fail
  elsewhere.
- **[difference, medium]** `skip-token-mismatch`: leap stride is hardcoded `skip_token=3`, but the
  paper's default is `k=2`; no config exposes it, so the shipped default does not reproduce the
  paper's stated setting.
- **[missing, medium]** `speedup-measurement-missing`: no repo script times generation / computes
  the tokens/s speedup ratios reported in Fig. 6 and Table 2.
- **[difference, low]** `num-heads-mismatch`: every config uses `num_heads: 3`, but the paper's
  default is `n=4`.
- **[difference, low/question]** `leap-start-offset`: additional heads are supervised at offsets
  {k,2k,3k} rather than the paper's {k+1,2k+1,3k+1}.

## Items that genuinely look fine

- Head architecture (`models/lmtp/head.py`) matches Appendix B.5: residual SiLU block + linear
  head, each head's `lm_head` initialized from the base model's head weight (head.py:71-77).
- The MTP baseline reuses the same model/training code as L-MTP (configs `mtp_type: lmtp1/lmtp2`,
  shared `LMTPModel`), consistent with the paper's "same training setting … to ensure fairness".
- Task accuracies (Table 1) are computed by external standard harnesses (lm_eval / evalplus /
  MathRuler), which is a legitimate reason the repo does not itself contain those eval scripts.
- The stage-2 loss correctly separates the NTP head loss (`ori_loss`) from the weighted additional-
  head loss with `alpha = head_weight` (model.py:212-224), matching Eq. (5)'s β weighting.

## Open questions for the authors

- Which `(k, n)` actually produced the Table 1 numbers — the paper's default (k=2, n=4) or the
  shipped configs/code default (skip_token=3, num_heads=3)? (`skip-token-mismatch`,
  `num-heads-mismatch`.)
- Where is the missing `models/medusa/` package, and was it present when the reported numbers were
  produced? (`missing-medusa-module`.)
- What harness produced the speedup ratios in Fig. 6 and Table 2? (`speedup-measurement-missing`.)
