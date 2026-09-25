# Human-eval worksheet — #1333 · 1333

**17 distinct defects** (the 10 PDF+text audit runs' findings, merged by defect). Detection = how many of the 10 runs surfaced the defect (high = robust; 1 = one run only). Severity & confidence are the auditor's own labels (spread shown where runs disagreed); the wording/quote is taken from the highest-confidence run that cited code.

Tick **one** box per defect (put an `x`):

- **correct & relevant** — true *and* a substantive reproducibility issue worth raising
- **correct but wrong severity** — true and worth raising, but the severity label is miscalibrated (e.g. an out-of-the-box crash with a trivial fix tagged high that's really low/medium)
- **correct but not relevant** — technically true but trivial / nitpick / already acknowledged
- **unsure** — can't decide without resources beyond the frozen repo + paper
- **false** — the claim misreads the code/paper and does not hold

Frozen code: `1333/code_frozen/`  ·  paper: `audits/1333/paper.pdf`

---

### F01 · Latent restoration network R_theta (Stage-2 core, Eq.7/9) is absent — Stage-2 only auto-encodes via the VAE; L_Res training not implemented

_category: Missing code / data · topic: training protocol_

**severity: high  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** The VAEadapter trainer only alternates FHF-LoRA (encoder) and PHF-LoRA (decoder) HF-alignment steps; there is no phase that trains a restoration network with the standard restoration loss LRes = ||Dψ*(zres) - Iclean||1 (Eq. 7) before LoRA fine-tuning.
- **Concern:** Eq. 7 (training Rθ with frozen VAE) is described as the first sub-step of Stage 2 and is the source of the restored latent; its absence means the described training protocol is not reproducible from this repo.
- **Ask:** Authors: provide the LRes pre-training script/config (which model, which loss, frozen-VAE setting) and how its checkpoint feeds the LoRA stage.
- **Evidence:** `basicsr/models/VAEadapter_model.py:148-167` · paper: Section 4.2, Eq. 7
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r04#1)
- **Quoted at `basicsr/models/VAEadapter_model.py:148-167`:**
```
def optimize_parameters(self, current_iter):
    loss_dict = OrderedDict()
    hf_gt = self._hf(self.gt)

    if self._is_fhf_step(current_iter):
        # FHF-LoRA: update encoder LoRA only.
        self._set_training_stage(train_enc=True, train_dec=False, train_disc=False)
        self.optimizer_enc.zero_grad()

        recon, posterior, z = self._forward_g()
        hf_pred = self._hf(recon)
        l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
        l_hf_fid.backward()
        self.optimizer_enc.step()

        self.output = recon
        self.posterior = posterior
        self.latent = z
... (+2 more lines)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F02 · No evaluation / inference / metric entrypoint — none of Tables 1–5 or Fig. 2 can be computed from the repo

_category: Missing code / data · topic: result traceability / evaluation_

**severity: high  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** The repo ships only basicsr/train.py; there is no test.py / inference / benchmark script, the two provided configs set `metrics: ~` (no metrics), no config evaluates a trained model on UHD-LL/UHD-blur/etc., and no FID metric exists at all (paper Table 3 reports FID).
- **Concern:** Every PSNR/SSIM/LPIPS/FID/NIQE/user-study/FLOPs/runtime number in Tables 1–5 and Figure 2 is therefore untraceable to code that computes it, so none of the reported results can be reproduced or verified.
- **Ask:** Authors: add the evaluation entrypoint(s) and metric configuration used to produce each table (including FID), with the exact commands and the trained checkpoints.
- **Evidence:** `basicsr/metrics/__init__.py:1-8` · paper: Tables 1–5; Figure 2
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r04#2)
- **Quoted at `basicsr/metrics/__init__.py:1-8`:**
```
from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F03 · Stage-2 config requests VAE arch type 'RAVAE' which is not registered → KeyError at network build

_category: Technical bug · topic: config / arch registry_

**severity: high  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** RAVAEHFLora reads vae_config['type'] and calls ARCH_REGISTRY.get(arch_type). configs/stage2_hflora.yml sets network_g.vae_config.type: RAVAE (line 42), but no class named 'RAVAE' is registered — only 'RAVAE_EQ' and 'RAVAEHFLora' (verified by AST scan in _audit_code/out/registry_vs_configs.csv: stage2_hflora.yml,RAVAE,UNREGISTERED). ARCH_REGISTRY.get raises KeyError when the name is absent (basicsr/utils/registry.py:62-66).
- **Concern:** Building the Stage-2 model from the shipped example config crashes immediately with KeyError("No object named 'RAVAE' found in 'arch' registry!"), so Stage 2 cannot be trained as released without editing the config to 'RAVAE_EQ'.
- **Ask:** Authors: change stage2 vae_config.type to 'RAVAE_EQ' (or register a 'RAVAE' alias) and confirm the intended backbone.
- **Evidence:** `basicsr/archs/LHVAE_hflora_arch.py:36-38` · paper: configs/stage2_hflora.yml line 42
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r08#4)
- **Quoted at `basicsr/archs/LHVAE_hflora_arch.py:36-38`:**
```
        cfg = deepcopy(vae_config)
        arch_type = cfg.pop('type')
        self.vae = ARCH_REGISTRY.get(arch_type)(**cfg)
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[x]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

can be fixed easily. severity low

---

### F04 · No dependency specification (no requirements.txt / environment.yml / setup.py) for a heavy multi-dependency repo

_category: Missing code / data · topic: expected code completeness_

**severity: medium  (varied: high, medium)  ·  confidence: high  ·  detection: 10/10 runs**

- **Claim:** The repo imports third-party packages (pyiqa, torch, torchvision, tqdm) and downloads DINOv2 via torch.hub.load('facebookresearch/dinov2', ...) (EQvae_model.py:115), yet check_completeness.py finds dependency_spec_files => NONE (no requirements.txt, setup.py, environment.yml, pyproject.toml).
- **Concern:** Without pinned dependencies the environment cannot be reliably rebuilt; unpinned pyiqa/torch versions can change metric values and break the (frozen) APIs used here.
- **Ask:** Add a requirements.txt / environment.yml pinning torch, torchvision, pyiqa, tqdm, numpy, pyyaml and the DINOv2 hub revision.
- **Evidence:** `basicsr/models/VAEadapter_model.py:9-13` · paper: NeurIPS checklist Q5
- **Found in runs:** r01, r02, r03, r04, r05, r06, r07, r08, r09, r10  (representative: r10#2)
- **Quoted at `basicsr/models/VAEadapter_model.py:9-13`:**
```
import pyiqa

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F05 · experiments_root / training output dir hardcoded to the authors' cluster path — breaks any other machine

_category: Technical bug · topic: hardcoded absolute paths_

**severity: low  ·  confidence: high  ·  detection: 8/10 runs**

- **Claim:** For every training run (is_train=True), the experiments root (where checkpoints, logs, training states, and visualizations are written) is hardcoded to the absolute author-cluster path /fs-computility/ai4sData/liuyidi/model/experiments/<name>, instead of the stock BasicSR osp.join(root_path, 'experiments', opt['name']).
- **Concern:** On any machine without that exact directory tree, training fails to write checkpoints/logs (FileNotFoundError / PermissionError) or silently writes to an unexpected absolute location, blocking out-of-the-box training reproduction.
- **Ask:** Authors: derive experiments_root from root_path (or a config field) rather than a hardcoded absolute path.
- **Evidence:** `basicsr/utils/options.py:158-164` · paper: basicsr/utils/options.py:159
- **Found in runs:** r01, r02, r03, r05, r06, r08, r09, r10  (representative: r08#5)
- **Quoted at `basicsr/utils/options.py:158-164`:**
```
    if is_train:
        experiments_root = osp.join('/fs-computility/ai4sData/liuyidi/model', 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F06 · Hardcoded author-machine absolute paths injected via sys.path.append in core modules (non-existent elsewhere)

_category: Technical bug · topic: hardcoded path_

**severity: high  (varied: medium, low)  ·  confidence: high  ·  detection: 7/10 runs**

- **Claim:** train.py and LHVAE_arch.py prepend the author's absolute path /fs-computility/ai4sData/liuyidi/code/LatentGen to sys.path; LHVAE_arch.py's __main__ also opens .../LatentGen/options/debug.yml.
- **Concern:** These dead absolute paths are remnants of the author environment; they do not break import on their own (append silently ignores missing dirs) but signal the repo was not cleaned for release and the __main__ profiling block (the only place FLOPs/params could be computed) cannot run as shipped.
- **Ask:** Remove the author-specific sys.path.append lines and the hardcoded debug.yml path; ship a runnable FLOPs/params script.
- **Evidence:** `basicsr/train.py:7-8` · paper: n/a
- **Found in runs:** r01, r02, r04, r05, r07, r08, r10  (representative: r02#7)
- **Quoted at `basicsr/train.py:7-8`:**
```
import sys
sys.path.append("/fs-computility/ai4sData/liuyidi/code/LatentGen")
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F07 · Datasets and pretrained VAE/DINOv2 weights are referenced by config but not shipped; no fetch script

_category: Missing code / data · topic: expected code completeness_

**severity: medium  (varied: high, medium)  ·  confidence: high  ·  detection: 5/10 runs**

- **Claim:** Stage-2 config loads ./weights/stage1_eqvae.pth and Stage-1 config loads ./weights/dinov2_vits14.pth and reads data from ./datasets/train/{gt,lq}; none of these files/dirs exist in the repo and there is no download/fetch script or accession.
- **Concern:** Stage 2 cannot start without the Stage-1 VAE checkpoint, Stage 1 cannot start without the DINOv2 weights (LInv hard-errors if the path is empty), and neither stage has training data, so the pipeline cannot be run as shipped.
- **Ask:** Release the LH-VAE checkpoint and DINOv2 weights (or a documented download), and provide data-preparation scripts / accessions for the UHD benchmarks (UHD-LL, UHD-blur, UHD-haze, UHD-rain, UHD-snow, UHD denoising).
- **Evidence:** `configs/stage2_hflora.yml:33-36` · paper: §5 Experiments; NeurIPS checklist Q5 ('released upon acceptance')
- **Found in runs:** r01, r02, r05, r07, r09  (representative: r02#2)
- **Quoted at `configs/stage2_hflora.yml:33-36`:**
```
network_g:
  type: RAVAEHFLora
  pretrain_vae_path: ./weights/stage1_eqvae.pth
  pretrain_param_key: params
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[x]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

severity med or low, pretraiend weights often not shipped. But no info present on where to get them

---

### F08 · LEqv is computed on the latent of the PERTURBED input, not the clean-image latent z_clean as in Eq. 5

_category: Paper–code mismatch · topic: loss definition (paper vs code)_

**severity: low  ·  confidence: medium  (varied: high, medium)  ·  detection: 4/10 runs**

- **Claim:** Paper Eq. 5 defines LEqv = ||D(Downs(z_clean)) - Downs(I_clean)||_1 with z_clean from the clean image. In optimize_parameters (lines 298-330) z is the latent of the PDPS-perturbed input (self.net_g(self.pdps_input)), and that perturbed-image latent is passed to _compute_eqv_loss, while the target gt_down is the downsampled clean image.
- **Concern:** The equivariance constraint is applied to the perturbed-image latent rather than the clean-image latent the equation specifies; the implemented variant is a defensible (degradation-robust) alternative but does not match Eq. 5.
- **Ask:** Authors: clarify whether LEqv should use the clean-image latent (Eq. 5) or the perturbed-input latent as implemented; align code or paper accordingly.
- **Evidence:** `basicsr/models/EQvae_model.py:283-293` · paper: Section 4.1, Eq. 5
- **Found in runs:** r02, r04, r07, r08  (representative: r07#8)
- **Quoted at `basicsr/models/EQvae_model.py:283-293`:**
```
def _compute_eqv_loss(self, z):
    if self.lambda_eqv <= 0:
        return z.new_zeros(())
    if self.eqv_scale_factor <= 0 or self.eqv_scale_factor >= 1:
        raise ValueError(f'eqv.scale_factor must be in (0, 1), got {self.eqv_scale_factor}')

    z_down = F.interpolate(z, scale_factor=self.eqv_scale_factor, mode='bilinear', align_corners=False)
    net_g = self.get_bare_model(self.net_g)
    pred_down = net_g.decode(z_down)
    gt_down = F.interpolate(self.gt, size=pred_down.shape[2:], mode='bilinear', align_corners=False)
    return F.l1_loss(pred_down, gt_down)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F09 · README is an empty one-line stub (no commands, no results table); code-availability deferred

_category: Missing code / data · topic: data / weights / availability_

**severity: low  (varied: medium, low)  ·  confidence: high  ·  detection: 3/10 runs**

- **Claim:** The NeurIPS checklist (Q5) states code will be released only upon acceptance; the README is empty (0 lines beyond the title), and the configs point to absent artefacts: DINOv2 weights ./weights/dinov2_vits14.pth (stage1) and the Stage-1 VAE checkpoint ./weights/stage1_eqvae.pth (stage2), plus datasets ./datasets/train|val that are not provided and have no fetch script.
- **Concern:** No reproduction instructions, no trained weights, and no resolvable dataset path means a reader cannot run either stage end-to-end as released.
- **Ask:** Provide a README with exact commands, the DINOv2 and Stage-1 VAE checkpoints (or a download script), and dataset preparation instructions/links.
- **Evidence:** `paper.pdf` · paper: NeurIPS checklist Q5; configs/stage1_eqvae.yml:94-95, configs/stage2_hflora.yml:35
- **Found in runs:** r05, r08, r10  (representative: r10#3)
- **Quoted at `paper.pdf`:**
```
Answer:[No]
Justification:We will release the source code upon acceptance of the paper.
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F10 · Decoder PHF-LoRA step also minimizes the HF-fidelity L1 loss, not the perception-only (GAN) loss of Eq. 9

_category: Paper–code mismatch · topic: evaluation consistency (paper vs code)_

**severity: medium  (varied: medium, low)  ·  confidence: high  (varied: high, medium)  ·  detection: 3/10 runs**

- **Claim:** In the PHF-LoRA (decoder) step the code optimizes l_hf_fid (high-frequency L1 fidelity loss) plus the adversarial GAN loss; the fidelity term dominates (lambda_hf_fid=1.0 vs lambda_gan=0.1).
- **Concern:** Paper Eq. 9 specifies the decoder LoRA is driven solely by the perception-oriented GAN loss LHFGAN; adding the fidelity L1 term contradicts the paper's clean perception/fidelity decoupling and changes what PHF-LoRA optimizes.
- **Ask:** Confirm whether the decoder LoRA should be optimized by the GAN loss alone (Eq. 9); if the L1 HF-fidelity term is intended, document it in the method.
- **Evidence:** `basicsr/models/VAEadapter_model.py:188-202` · paper: Section 4.2, Eq. 9
- **Found in runs:** r05, r06, r10  (representative: r10#8)
- **Quoted at `basicsr/models/VAEadapter_model.py:188-202`:**
```
# 2) Decoder LoRA (generator) step
self.optimizer_dec.zero_grad()
recon, posterior, z = self._forward_g()
hf_pred = self._hf(recon)

l_total = recon.new_tensor(0.0)
l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
l_total = l_total + l_hf_fid
loss_dict['l_hf_fid'] = l_hf_fid

if self.net_d_hf is not None and self.cri_gan is not None and self.lambda_gan > 0:
    pred_fake_g = self.net_d_hf(hf_pred)
    l_hf_gan = self.cri_gan(pred_fake_g, True, is_disc=False) * self.lambda_gan
    l_total = l_total + l_hf_gan
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

Would be a non issue if the authors woudnt describe they use GAN-only and don't mention the F1 fidelity term. Leaning towards low severity though

---

### F11 · LoRA alpha blending mismatch — opposite LoRA scaled at runtime by alpha=0.5 instead of frozen at base / weight-delta blending

_category: Paper–code mismatch · topic: Stage-2 alternating optimization_

**severity: low  ·  confidence: medium  ·  detection: 3/10 runs**

- **Claim:** During the FHF (encoder-fidelity) step the decoder LoRA is frozen (train_dec=False) but still contributes to the forward pass, because _forward_g calls net_g(lq, alpha=0.5) and set_alpha sets dec_scale=1-alpha=0.5 (LHVAE_hflora_arch.py:83-91), so the decoder is psi*+0.5*delta_psi, not the frozen base psi* of Eq. 8.
- **Concern:** Paper Eq. 8 evaluates the fidelity loss with the decoder at its frozen base parameters D_{psi*}; the code instead applies a half-strength decoder LoRA, a mismatch between the described and implemented FHF objective (the implemented version is still valid, hence difference).
- **Ask:** Authors: confirm whether FHF-LoRA training should set alpha=1 (decoder LoRA off) during encoder steps to match Eq. 8, or clarify the intended decoder state.
- **Evidence:** `basicsr/models/VAEadapter_model.py:152-161` · paper: §4.2 Eq. 8
- **Found in runs:** r01, r09, r10  (representative: r01#7)
- **Quoted at `basicsr/models/VAEadapter_model.py:152-161`:**
```
if self._is_fhf_step(current_iter):
    # FHF-LoRA: update encoder LoRA only.
    self._set_training_stage(train_enc=True, train_dec=False, train_disc=False)
    self.optimizer_enc.zero_grad()

    recon, posterior, z = self._forward_g()
    hf_pred = self._hf(recon)
    l_hf_fid = self.cri_pix(hf_pred, hf_gt) * self.lambda_hf_fid
    l_hf_fid.backward()
    self.optimizer_enc.step()
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

unclear if and how this was applied during training for the paper results. _forward_g applies the same fixed α=0.5 in both phases. The code does not implement the per-phase freezing

---

### F12 · Backbone-integration experiments absent — Table 5(c) backbones (Restormer/NAFNet/SFHformer) and Table 3/4 integration/generalization have no code

_category: Missing code / data · topic: result traceability_

**severity: low  (varied: high, medium)  ·  confidence: high  ·  detection: 2/10 runs**

- **Claim:** The repo contains only the two stage trainers (EQVAEModel, VAEadapter). There is no code integrating LH-VAE into PromptIR / Diff-Plugin / CosAE (Table 3) nor any unseen/composite-degradation generalization harness (Table 4).
- **Concern:** Two full experimental sections of the paper (standard-resolution versatility and generalization) cannot be reproduced because their drivers are absent.
- **Ask:** Release the PromptIR/Diff-Plugin/CosAE integration code and the unseen/composite-degradation evaluation scripts.
- **Evidence:** `basicsr/models/VAEadapter_model.py:18-20` · paper: Tables 3 and 4 (Sections 5.2)
- **Found in runs:** r07, r10  (representative: r10#4)
- **Quoted at `basicsr/models/VAEadapter_model.py:18-20`:**
```
@MODEL_REGISTRY.register()
class VAEadapter(BaseModel):
    """Stage-2 trainer for HF-LoRA alternating optimization."""
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

---

### F13 · Shipped configs are '*_example' placeholders with toy batch sizes and generic (non-paper) hyperparameters

_category: Paper–code mismatch · topic: experimental setup / configs_

**severity: medium  (varied: medium, low)  ·  confidence: high  (varied: high, medium)  ·  detection: 2/10 runs**

- **Claim:** Both configs are named *_example (stage1_eqvae_example, stage2_hflora_example), use generic single-GPU settings, batch_size_per_gpu: 2, total_iter: 100000, and placeholder data roots ./datasets/train/gt etc.; they do not encode the six-degradation / four-degradation training recipes, per-degradation datasets, or the α and LoRA-rank settings that back Tables 1-5.
- **Concern:** The paper reports specific UHD all-in-one results (Tables 1-2) and an α sweep (Table 5d), but the released configs are illustrative templates, so the exact experimental settings used for the paper cannot be recovered from the repo.
- **Ask:** Authors: release the actual training/eval configs (data manifests, degradation mix, iters, LoRA rank/alpha, λ_Inv/λ_Eqv/λ_hf_fid/λ_gan, α values) used for each reported table.
- **Evidence:** `configs/stage2_hflora.yml:1-5` · paper: Tables 1-5; Sec. 5 Experiments
- **Found in runs:** r03, r04  (representative: r03#5)
- **Quoted at `configs/stage2_hflora.yml:1-5`:**
```
name: stage2_hflora_example
model_type: VAEadapter
num_gpu: 1
manual_seed: 123
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

could maybe be changed to missing category. But also correct like this.

---

### F14 · Key architecture modules are placeholder/substitute stubs (e.g. vgg_arch), not the paper's components

_category: Missing code / data · topic: repository provenance_

**severity: high  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** Several components are explicitly self-described as stand-ins: VGGFeatureExtractor 'returns the input itself' (vgg_arch.py:4-8); restormer.TransformerBlock is a 'Lightweight spatial transformer-style block' (restormer.py:5); fourmer.ProcessBlock is a 'Lightweight substitute for Fourmer process block' (fourmer.py:5). The paper states the latent restoration network is SFHformer, which does not appear here.
- **Concern:** If the perceptual/feature and restoration components are placeholders rather than the real modules, the repo is not the artefact that produced the paper's numbers, and the reported results (which depend on these exact modules) cannot be reproduced from it.
- **Ask:** Authors: confirm whether these are placeholders, and release the actual SFHformer restoration network and the real VGG/perceptual modules used for the reported metrics.
- **Evidence:** `basicsr/archs/vgg_arch.py:4-8` · paper: Section 5.3 (SFHformer); Table 1/2 (LPIPS)
- **Found in runs:** r06  (representative: r06#3)
- **Quoted at `basicsr/archs/vgg_arch.py:4-8`:**
```
class VGGFeatureExtractor(nn.Module):
    """Fallback VGG feature extractor placeholder.

    Returns a dict with requested layer names, each mapped to the input itself.
    """
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[ ]`   correct but not relevant `[x]`   unsure `[ ]`   false `[ ]`

**Notes:**

The modules are stubs but also not important to generate the main results. The most important architectures are there VAE, LoRA, DINO, losses

---

### F15 · basicsr/__init__.py imports `.test` but basicsr/test.py does not exist → import crashes on load

_category: Technical bug · topic: import / packaging_

**severity: high  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** `basicsr/__init__.py` executes `from .test import *`, but there is no `basicsr/test.py` in the repo (only `__init__.py` and `train.py` at the package root). Any `import basicsr` — including `basicsr/train.py`'s `from basicsr.data import ...`, which triggers the package `__init__` — raises ModuleNotFoundError: No module named 'basicsr.test'.
- **Concern:** The training entrypoint cannot be imported/run at all; the framework is broken on first import before any data or config is involved.
- **Ask:** Either add the missing `basicsr/test.py` (the standard BasicSR inference entrypoint, which would also supply the absent evaluation pipeline) or remove the `from .test import *` line.
- **Evidence:** `basicsr/__init__.py:7` · paper: n/a
- **Found in runs:** r09  (representative: r09#4)
- **Quoted at `basicsr/__init__.py:7`:**
```
from .test import *
```

**Verdict:**   correct & relevant `[ ]`   correct but wrong severity `[x]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

correct but severity is low, because it can be fixed very easily by removing a line. It is a reproducibility nuisance, but not a severe one.

---

### F16 · Stage-1 reconstruction encodes the perturbed image, not I_clean as defined in Eq. 2

_category: Paper–code mismatch · topic: stage 1 VAE loss_

**severity: high  ·  confidence: high  ·  detection: 1/10 runs**

- **Claim:** Stage-1 feeds the PDPS-perturbed/degraded image (pdps_input) through the VAE and computes the reconstruction loss l_recon = cri_pix(recon, gt) against the clean image (line 318).
- **Concern:** Eq. 2 writes L_VAE = ||D_psi(E_phi(I_clean)) - I_clean||_1 (encode the CLEAN image); the code reconstructs the clean target from the degraded input. This is a valid restoration-style objective but differs from the equation as written, so the paper's formula does not match the implemented loss.
- **Ask:** Authors: update Eq. 2 to reflect that the encoder input is the PDPS-perturbed image, or confirm which input the reported Stage-1 used.
- **Evidence:** `basicsr/models/EQvae_model.py:298-318` · paper: Section 4.1, Eq. (2)–(3)
- **Found in runs:** r06  (representative: r06#7)
- **Quoted at `basicsr/models/EQvae_model.py:298-318`:**
```
self.pdps_input, t_value, pdps_branch = self._build_pdps_input(current_iter)
raw_out = self.net_g(self.pdps_input)
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

minor notation slip in the paper, but both paper and code version would be valid. They just differ, so a reproduction from the paper only would likely get different results.

---

### F17 · LInv adds an undocumented trainable 1x1 conv projector before the DINOv2 alignment

_category: Paper–code mismatch · topic: loss definition (paper vs code)_

**severity: low  ·  confidence: medium  ·  detection: 1/10 runs**

- **Claim:** LInv aligns the encoder latent to DINOv2 features through a learned 1x1 conv projector added on the fly and optimized with optimizer_g (EQvae_model.py:270-281, 144-153); the paper states a direct feature-space distance d(z'_deg, f_VFM) with no projector.
- **Concern:** A learned projector can absorb the distribution gap, weakening the intended degradation-invariance constraint and adding an undescribed result-affecting component (paper omission).
- **Ask:** Authors: confirm whether a learned projection layer was part of LInv in the reported runs; if so, describe it; if not, remove it.
- **Evidence:** `basicsr/models/EQvae_model.py:144-153` · paper: Section 4.1, Eq. 4
- **Found in runs:** r07  (representative: r07#9)
- **Quoted at `basicsr/models/EQvae_model.py:144-153`:**
```
def _ensure_inv_projector(self, in_ch, out_ch):
    if in_ch == out_ch:
        return None
    if self.inv_projector is not None:
        if self.inv_projector.in_channels == in_ch and self.inv_projector.out_channels == out_ch:
            return self.inv_projector
    self.inv_projector = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0).to(self.device)
    if hasattr(self, 'optimizer_g'):
        self.optimizer_g.add_param_group({'params': self.inv_projector.parameters()})
    return self.inv_projector
```

**Verdict:**   correct & relevant `[x]`   correct but wrong severity `[ ]`   correct but not relevant `[ ]`   unsure `[ ]`   false `[ ]`

**Notes:**

This projector is needed just from the shapes of the tensors. Should have been mentioned in the paper.

---

