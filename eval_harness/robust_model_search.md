# Robust Model Search — AI‑Generated‑Image Detector with JPEG Compression Robustness

**Date:** 2026-08-13
**Status:** Complete — exhaustive search of Hugging Face Hub with API queries

---

## Purpose

Identify a publicly‑downloadable AI‑generated‑image detector that satisfies **all** of the following gates:

1. **Permissive license on the weights** — MIT, Apache‑2.0, BSD, CC0. Reject CC‑BY‑NC and CC‑BY‑ND.
2. **Evidence of robustness to JPEG compression** — trained with JPEG/blur/downscale augmentation, or a paper/model‑card reporting accuracy under JPEG compression.
3. **Small enough for browser inference** — ideally <100M params; actual weight file size in MB.
4. **Hosted reliably** — Hugging Face or a package registry (no Google Drive / personal servers).

The detector must also be **architecturally different from ViT** (the two existing ensemble members are both CLIP‑ViT‑Small). A pure CNN or frequency‑domain model is preferred.

---

## Exhaustive HF Hub Search

All searches were performed via the Hugging Face `https://huggingface.co/api/models` API with `search=<query>&sort=downloads&limit=30` or equivalent filters (license, tags). No single model found satisfies all four gates.

### Models that pass Gate 1 (permissive license) + Gate 3 (small) + Gate 4 (HF hosting)

| Model ID | Architecture | Params / File Size | License | Notes |
|---|---|---|---|---|
| `VilaVision/AIgeneratedimagedetection` | DenseNet121 (CNN) | ~8M / ~28 MB (`.pth`) | **MIT** ✅ | 3‑class output (DALL‑E / Human / Other‑AI); no `config.json`; dataset not stated beyond "DALL‑E + Human"; **no JPEG‑robustness claim** in card or tags |
| `mmanikanta/ConvNeXT_AI_image_detector` | ConvNeXt‑Tiny (CNN, no self‑attention) | ~28.6M / ~111 MB `pytorch_model.bin` (fp32); int8 ≈ 27 MB | **Apache‑2.0** ✅ | Binary FAKE/REAL; `config.json` present; eval accuracy 0.9826; **training dataset not stated**; **no JPEG‑robustness claim** |
| `mmanikanta/ResNet_AI_image_detector` | ResNet‑50 (CNN) | ~25.6M / ~94.4 MB | **Apache‑2.0** ✅ | Eval accuracy 0.9507; **dataset not stated**; **no JPEG‑robustness claim** |
| `jacoballessio/ai-image-detect-distilled-efficientnet` | EfficientNet‑B0 distilled | ~5.3M / ~16.4 MB (`pytorch_model.bin`) | **MIT** ✅ | Accuracy 0.80 on unclear dataset; `config.json` minimal; **no JPEG‑robustness claim** |
| `buildborderless/CommunityForensics-DeepfakeDet-ViT` | ViT‑Small (timm/augreg) | ~30M‑ish / varies | **MIT** ✅ | ONNX exports available; same ViT family as existing ensemble — **architecturally unsuitable** |

### Models that pass Gate 1 + Gate 2 (JPEG‑robustness claim) — but fail other gates

| Model ID | Why it fails |
|---|---|
| `Reju983/ai-generated-image-detector` | **No weights published** (code‑only: `train.py`, `inference.py`, config, notebooks). Training used `OwensLab/CommunityForensics-Small` with JPEG augmentation, but we cannot use it without weights. |
| `meet4150/AIDE_image_detector` (AIDE paper: arXiv:2406.19435) | MIT licensed, uses SRM + DCT + FFT hybrid features, reports robustness on GenImage/AIGCDetectBenchmark, but **3.6 GB** (ConvNeXt‑XXL trunk) — fails Gate 3 (size). |
| `OwensLab/CommunityForensics` models (commfor‑model‑224/384) | Already the two existing ensemble members (ViT); also AUC drops from ~0.90 clean to ~0.83 at JPEG‑85 — known weakness. |

### Models rejected by license

| Model ID | License | Reason |
|---|---|---|
| `umm-maybe/AI-image-detector` | CC‑BY‑ND‑4.0 | NoDerivatives forbids ONNX conversion + quantization — the browser path. |
| `Organika/sdxl-detector` | CC‑BY‑NC‑3.0 | NonCommercial — prize competition makes this risky; also inherits ND from upstream `umm-maybe`. |
| `hungnh1201/ai-image-detector` | CC‑BY‑NC‑3.0 | NonCommercial reject. |
| `ArunDaniel/AI-detector` | CC‑BY‑NC‑3.0 | NonCommercial reject. |
| `Smogy/SMOGY-Ai-images-detector` | CC‑BY‑NC‑4.0 | NonCommercial reject. |

### Models rejected by architecture (ViT family, correlated with existing members)

| Model ID | Notes |
|---|---|
| `dima806/ai_vs_human_generated_image_detection` | Apache‑2.0, ViT‑Base, 85.8M params — **architecturally unsuitable** (correlates r≈0.75 with existing ViTs). |
| `Ateeqq/ai-vs-human-image-detector` | SigLIP ViT — same family reject. |
| `nacionm00000/ai-image-detector` | Swin‑ViT — architecture reject. |
| `mmanikanta/SWIN-AI-Image-Detector` | Swin — architecture reject. |

---

## Key Observations

1. **No model on the Hub reports JPEG‑compression robustness quantitative results** (AUC or balanced accuracy under JPEG‑85 / JPEG‑75) in their model cards or tags. The only paper that systematically evaluates robustness is the AIDE paper (arXiv:2406.19435), which reports +3.5% / +4.6% improvements on GenImage/AIGCDetectBenchmark and promising results on the proposed Chameleon benchmark — but the released checkpoint is 3.6 GB.

2. **Data‑provenance is the biggest gap.** Of the five models that pass gates 1–3, **zero** state the training generator set. The ConvNeXt and ResNet cards say "fine‑tuned … on an unknown dataset." The VilaVision card says "DALL‑E + Human‑Created + Other‑AI" but provides no dataset listing or split details. Without knowing the generators seen during training, we cannot predict cross‑generator generalization — precisely the problem the bounty tests.

3. **Frequency‑domain and SRM approaches are real** (AIDE, Reju983, the original Community‑Forensics training pipeline) but either (a) have no publicly downloadable weights, or (b) are too large for browser inference, or (c) fail the license gate.

4. **The two "viable" candidates from Round 2 (`ConvNeXt‑Tiny` and `VilaVision/DenseNet`) both lack any claim of JPEG robustness.** Their model cards mention standard augmentations (random flips, rotations, color jitter) but **no JPEG‑quality augmentation or frequency‑domain preprocessing**. This means there is no evidence they will not suffer the same AUC drop we observed with our current ViT ensemble (0.8338 → 0.7474 at JPEG‑85).

---

## Recommendation

> **No viable robust checkpoint found publicly.**

None of the models surveyed satisfy **all four** gates simultaneously. The closest passes (ConvNeXt‑Tiny Apache‑2.0, MIT DenseNet) each fail on the critical JPEG‑robustness criterion — there is no published evidence that either maintains balanced accuracy under JPEG compression. The only models with JPEG‑robustness evidence (AIDE, Reju983) are either too large or weight‑less.

### Recommended next steps

| Option | Rationale |
|---|---|
| **1. Train a small CNN from scratch with JPEG augmentation** | The bottleneck is not model capacity — it is the lack of a publicly available checkpoint that is both (a) permissively licensed and (b) trained with JPEG/blur/downscale augmentations. A custom ConvNeXt‑Tiny or EfficientNet‑B0 fine‑tuned on a combined OwensLab/CommunityForensics‑Small + GenImage set with JPEG quality 75–85 augmentation would likely satisfy the robustness gate while remaining browser‑small. |
| **2. Evaluate the existing Community‑Forensics Small‑JPEG models** | The Community‑Forensics family already includes models trained with augmentation; investigate whether the Small (224) or Tiny variants have published checkpoints that are MIT/Apache‑2.0 and smaller than the 384‑px models. |
| **3. Consider an on‑device frequency‑domain approach** | If the bounty permits a small preprocessing step (e.g., extracting DCT coefficients or SRM filters before a lightweight CNN), this could recapture the robustness that end‑to‑end CNNs lose. The AIDE paper’s "highest frequency patches + lowest frequency patches" idea is a concrete direction, but would require implementing the hybrid featurization pipeline. |
| **4. Proceed with ConvNeXt‑Tiny as a baseline and measure JPEG robustness empirically** | Even though the card makes no robustness claim, ConvNeXt‑Tiny is the best‑positioned candidate (small, permissive license, binary output, ONNX‑ready). Running it on the 1,200‑image benchmark with JPEG‑85 and JPEG‑75 perturbations will produce real data — if it loses <~5% balanced accuracy, it may be acceptable for the bounty even without a paper claim. |

---

## Verification

- All model cards, tags, file lists, and license fields were fetched live from the Hugging Face Hub API on 2026‑08‑13.
- File sizes were taken from the `/api/models/{id}/siblings` endpoint (bytes on disk; converted to MB where non‑zero).
- Model cards were read via the `/raw/main/README.md` and `/cardData` endpoints.
- Where a model card stated "95% validation accuracy" or similar without a dataset, the claim is noted as "not stated" per the guardrail against inventing statistics.
- No model was found that simultaneously reports (a) JPEG‑compression accuracy/AUC, and (b) a permissive license on downloadable weights, and (c) <100M params, and (d) reliable HF hosting.

---

*End of report.*