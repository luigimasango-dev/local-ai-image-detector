# Round 2 — Third Ensemble Member: Candidate Model Research

Status: COMPLETE (all findings verified 2026-08-13 against live sources)

Date: 2026-08-13

## Context

We already ship a working ensemble of **two Community-Forensics models** (both
CLIP-ViT-Small, MIT, ~0.8338 balanced accuracy on our holdout). We want a
**third member** that earns its place only if it satisfies ALL four gates:

1. **Permissive license on the WEIGHTS** — MIT / Apache-2.0 / BSD / CC0.
   Reject CC-BY-NC and CC-BY-ND (we must convert to ONNX and quantize; ND
   forbids that, and a prize competition makes NC risky). License must be
   stated with a source URL. "not found" if not explicitly stated.
2. **Architecturally DIFFERENT from ViT** — CNN / frequency-domain (DCT) /
   patch-texture / reconstruction-error. Another ViT correlates with our two
   existing members (r≈0.75) and buys nothing.
3. **Small enough for a browser** — ideally < ~100M params; note actual weight
   file size in MB.
4. **Hosted reliably** — Hugging Face or a package registry. Google Drive /
   personal servers are a hosting risk (bounty evaluators build in a clean
   environment).

---

## Headline result

Two genuinely viable candidates pass all four gates, both small CNNs on the
Hugging Face Hub:

- **mmanikanta/ConvNeXT_AI_image_detector** (Apache-2.0, ConvNeXt-Tiny, ~28.6M
  params / ~111 MB fp32, binary FAKE/REAL, standard HF transformers CNN).
- **VilaVision/AIgeneratedimagedetection** (MIT, DenseNet121, ~8M params /
  ~28 MB, but 3-class output and a raw `.pth` with no config).

Recommended first try: **ConvNeXt-Tiny** (see Recommendation). The most
interesting frequency-domain candidate (Reju983, SwinV2+DCT+FFT+SRM, trained on
the same 4,803-generator Community-Forensics dataset) fails on hosting: it
publishes **training code but no weights**.

Note on the lead "`AP6621/AI_generated_image_detection`": this id does not
resolve as a standalone repo — it **HTTP-307 redirects** to
`VilaVision/AIgeneratedimagedetection` (verified
https://huggingface.co/AP6621/AI_generated_image_detection). Everything below
therefore documents the VilaVision model.

---

## Candidate table

| # | Model id (owner/repo) | Architecture (& ≠ ViT?) | Params / file size | License (+ source URL) | Trained on (generators) | Reported accuracy (+ source) | Hosting | Verdict |
|---|----------------------|------------------------|--------------------|------------------------|--------------------------|------------------------------|---------|----------|
| 1 | `VilaVision/AIgeneratedimagedetection` | DenseNet121 (**CNN**) | ~8M / **28.4 MB** (`.pth`) | **MIT** — card `license: mit` [card](https://huggingface.co/VilaVision/AIgeneratedimagedetection), API `license:mit` | DALL-E + "Other AI" + Human; dataset/generators **not stated** beyond that | "95% validation accuracy" [card](https://huggingface.co/VilaVision/AIgeneratedimagedetection) — no dataset detail | HF | **VIABLE** (MIT, CNN, tiny — but 3-class, raw `.pth` w/o config, thin card) |
| 2 | `mmanikanta/ConvNeXT_AI_image_detector` | **ConvNeXt-Tiny** (pure CNN, no self-attention) | ~28.6M / **111.3 MB** `pytorch_model.bin` ≈106 MB | **Apache-2.0** — card frontmatter [card](https://huggingface.co/mmanikanta/ConvNeXT_AI_image_detector) | **not stated** ("fine-tuned … on an unknown dataset", card) | 0.9826 eval acc [eval_results.json](https://huggingface.co/mmanikanta/ConvNeXT_AI_image_detector/raw/main/eval_results.json), [card](https://huggingface.co/mmanikanta/ConvNeXT_AI_image_detector) | HF, standard `ConvNextForImageClassification` | **VIABLE** (all gates pass; training-data provenance unknown) |
| 3 | `mmanikanta/ResNet_AI_image_detector` | **ResNet-50** (CNN) | ~25.6M / **94.4 MB** ≈90 MB | **Apache-2.0** — card frontmatter [card](https://huggingface.co/mmanikanta/ResNet_AI_image_detector) | **not stated** ("unknown dataset", card) | 0.9507 eval acc [eval_results.json](https://huggingface.co/mmanikanta/ResNet_AI_image_detector/raw/main/eval_results.json) | HF, standard `ResNetForImageClassification` | **VIABLE** (weaker than #2; same caveats) |
| 4 | `onnx-community/Deep-Fake-Detector-v2-Model-ONNX` | **ViT-Base** (`google/vit-base-patch16-224-in21k`) | 86M / fp32 343.4 MB · fp16 171.8 MB · int8 87.3 MB · q4 49.7 MB | **Apache-2.0** [card](https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX) | "curated dataset of real and deepfake images" (face-focused; generators **not stated**) | 0.9212 test acc [card](https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX) | HF, ONNX-ready | **REJECTED-architecture** (ViT = same family as both existing members) |
| 5 | `capcheck/ai-image-detection` | **ViT-Base** | 85.8M / safetensors F32 ≈343 MB | **Apache-2.0** (inherited, per card) [card](https://huggingface.co/capcheck/ai-image-detection) | CIFAKE (32×32 CIFAR-derived; pre-2024 generators) | not stated (delegates to dima806 card) | HF | **REJECTED-architecture** (ViT) + **CIFAKE 32×32** — card concedes poor generalization to modern generators |
| 6 | `meet4150/AIDE_image_detector` | Hybrid: **SRM** 30-filter bank + 2× **ResNet-50** on **DCT** views + frozen **ConvNeXt-XXL** trunk + MLP | 54.4M trainable + ConvNeXt-XXL ≈819M frozen; **`model.safetensors` = 3.59 GB** | **MIT** (frontmatter + LICENSE file) [card](https://huggingface.co/meet4150/AIDE_image_detector), [arXiv 2406.19435](https://arxiv.org/abs/2406.19435) | AIDE "multi-source" real-vs-fake; card does not enumerate sources | 77.9% val @ep19 (best 78.58% @ep17) [card](https://huggingface.co/meet4150/AIDE_image_detector) | HF | **REJECTED-size** (3.6 GB; ConvNeXt-XXL trunk) |
| 7 | `Reju983/ai-generated-image-detector` | Hybrid **SwinV2-Tiny + SRM + DCT + FFT** 4-branch | claims ~28.6M — **NO weights published** (only `train.py`, `inference.py`, config, notebooks) | **Apache-2.0** frontmatter [card](https://huggingface.co/Reju983/ai-generated-image-detector) | `OwensLab/CommunityForensics-Small` (556K imgs, **4,803 generators**) | not stated | HF — but **weights absent** (tree lists no `.safetensors`/`.bin`/`.pth`) | **REJECTED-hosting** (code-only; would require self-training) |
| 8 | `Ricehunter/efficientnet-ai-human-train-genimage-test` | **EfficientNet-B0** (CNN) | ~5.3M / **16.4 MB** (`.pth`) | **not found** — no README, no LICENSE, no license tag [repo](https://huggingface.co/Ricehunter/efficientnet-ai-human-train-genimage-test) | config: train `ai-vs-human-generated-dataset`, test `genimage-subset-detection` [config.json](https://huggingface.co/Ricehunter/efficientnet-ai-human-train-genimage-test/raw/main/config.json) | 0.9925 val acc [history.json](https://huggingface.co/Ricehunter/efficientnet-ai-human-train-genimage-test/raw/main/history.json) | HF | **REJECTED-license** (unstated ⇒ cannot assume permissive) |
| 9 | `bharathh04/ai-image-detector-vit` | **MobileNetV2** (CNN — repo name misleads; `model_type: mobilenet_v2`) | not stated (`model.safetensors`) | **not found** — LICENSE fetch = 404; card "License: [More Information Needed]" [card](https://huggingface.co/bharathh04/ai-image-detector-vit) | not stated | not stated | HF | **REJECTED-license** (unstated) |

---

## What the specific leads turned out to be

### Lead 1 — AP6621/AI_generated_image_detection → VilaVision/AIgeneratedimagedetection
- **Architecture**: DenseNet121 (CNN) fine-tuned on ImageNet-pretrained
  weights with a single FC classifier → **3 classes**: DALL-E / Human-Created /
  Other-AI [card](https://huggingface.co/VilaVision/AIgeneratedimagedetection).
  This is the architectural diversity we want (pure CNN, no attention).
- **License**: MIT — confirmed twice (model-card frontmatter `license: mit` and
  Hub API `cardData.license = "mit"`). Source:
  https://huggingface.co/VilaVision/AIgeneratedimagedetection
- **Size**: `densenet_finetuned_dense.pth` = 28,434,639 bytes, ~27.1 MB;
  DenseNet121 ≈ 8M params [file tree](https://huggingface.co/api/models/VilaVision/AIgeneratedimagedetection/tree/main)
- **Class labels**: 3-class (DALL-E, Human, Other AI) — **not binary**; an
  ensemble member needs a real/fake score, so this requires folding classes.
- **Caveats**: no `config.json`/preprocessor config; raw PyTorch state dict
  (manual loader + own ONNX export); the card's "Model Download" link points at
  parent `alokpandey/DenseNet-DH3Classifier`, which **no longer resolves** via
  the Hub API. Training dataset/generators beyond "DALL-E + other AI" are not
  stated. Accuracy "95%" with no dataset detail.
- **Verdict: VIABLE but second-choice** — MIT + CNN + tiny file qualifies, but
  multi-class output and an unconfig'd `.pth` add conversion risk vs. #2.

### Lead 2 — onnx-community/Deep-Fake-Detector-v2-Model-ONNX
- **Architecture**: **ViT-Base** (`google/vit-base-patch16-224-in21k`) with a
  classification head — [card](https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX).
- **License**: Apache-2.0 [card](https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX).
- **Size**: 86M params; ONNX files — fp32 343.4 MB, fp16 171.8 MB, int8 87.3 MB,
  q4 49.7 MB [file tree](https://huggingface.co/api/models/onnx-community/Deep-Fake-Detector-v2-Model-ONNX/tree/main).
- **Verdict**: **REJECTED-architecture.** It is a ViT and thus the same model
  family that already correlates at r=0.75 in our ensemble. The ONNX-ready,
  quantized, Apache-2.0 packaging is otherwise ideal — but it would add download
  size for a ViT whose correlated mistakes we already absorb.

### Lead 3 — capcheck/ai-image-detection
- **Architecture**: **ViT-Base**, 85.8M params [card](https://huggingface.co/capcheck/ai-image-detection).
- **License**: Apache-2.0.
- **Training data**: **CIFAKE** — AI vs real images. CIFAKE is built on 32×32
  CIFAR-derived images; the model card itself warns "Performance on modern AI
  generators (Flux, Midjourney v6, DALL-E 3, Stable Diffusion 3) may vary" and
  its Limitations list pre-2024 generators as the training bulk. Agreed: this is
  a low-resolution, older-generator detector → weak for full-resolution modern
  generators.
- **Verdict**: **REJECTED-architecture** (ViT), and independently weak on the
  generalization axis the bounty tests.

---

## What the broader HF sweep found

Using the Hub API (`/api/models?search=…&sort=downloads`), the top-downloaded
"ai image detection" models are **almost all ViT/Swin** and were screened out on
architecture or license:

- ViT/SigLIP/CLIP family (arch reject): `dima806/ai_vs_real_image_detection`,
  `dima806/ai_vs_human_generated_image_detection`, `umm-maybe/AI-image-detector`
  (now CC-BY-4.0 tag), `Ateeqq/ai-vs-human-image-detector`,
  `wkaandemir/ai-image-detector` (MIT, ViT), `Hemg/AI-VS-REAL-IMAGE-DETECTION`,
  `DhanushHN10/AI-Generated-Image-Detector-Vision-Transformer` (MIT, ViT),
  `NYUAD-ComNets/NYUAD_AI-generated_images_detector` (ViT, apache-2.0),
  `timkond/diffusion-detection` (BEiT = ViT variant, apache-2.0),
  `slxhere/UnivFD` (CLIP ViT-L weights; UnivFD is ViT-based — arch reject; no
  license stated either).
- Swin family (arch reject): `nacionm00000/ai-image-detector` (apache-2.0),
  `mmanikanta/SWIN-AI-Image-Detector` (apache-2.0),
  `SoraExplora/AIimageDetector` (cc0-1.0 but Swin),
  `ArunDaniel/AI-detector` (**CC-BY-NC-3.0** — license reject too),
  `Smogy/SMOGY-Ai-images-detector` (**CC-BY-NC-4.0** — license reject),
  `NehaBardeDUKE/…` (Swin, no license).
- License rejects: `hungnh1201/ai-image-detector` (Swin, CC-BY-NC-3.0).
- Text-model false positives: `SuperAnnotate/ai-detector(-low-fpr)` are
  RoBERTa **text** detectors, not image.

So the search confirms: permissive + non-ViT + loadable + small is a rare
combination on the Hub. The only frequency-channeled contenders —
AIDE (`meet4150`, MIT, 3.6 GB) and the SwinV2+DCT+FFT+SRM detector
(`Reju983`, apache-2.0, no weights) — each fail one other gate, and the
only clean CNNs are the three above (DenseNet121-MIT, ConvNeXt-Tiny-Apache,
ResNet-50-Apache).

---

## Gates check on the two viable candidates

| Gate | ConvNeXt-Tiny (`mmanikanta`) | DenseNet121 (`VilaVision`) |
|------|------------------------------|---------------------------|
| 1. Permissive weight license | ✅ Apache-2.0 (card) | ✅ MIT (card + API) |
| 2. ≠ ViT | ✅ Pure CNN (depthwise conv + LayerScale, no attention) | ✅ Pure CNN |
| 3. Browser-small | ✅ ~28.6M params / ~106 MB fp32 (int8 → ~27 MB) | ✅ ~8M / ~27 MB |
| 4. Reliable hosting | ✅ HF, standard transformers model | ✅ HF, but raw `.pth`, no config |
| Extra risk | Training dataset **not stated** (generalization unverified). | 3-class output; parent repo 404s; dataset **not stated**. |

---

## Recommendation

**Try `mmanikanta/ConvNeXT_AI_image_detector` (ConvNeXt-Tiny) next.**

Why this one:
1. Passes all four gates on paper — **Apache-2.0** (card at
   https://huggingface.co/mmanikanta/ConvNeXT_AI_image_detector), a **pure CNN**
   (maximally different from CLIP-ViT-S), **~28.6M params / ~106 MB** fp32
   (int8 ≈ 27 MB in the browser), hosted on the Hugging Face Hub.
2. It is a **standard HF transformers CNN** (`ConvNextForImageClassification`,
   binary FAKE/REAL, 224×224) with a `pytorch_model.bin` + `config.json` — the
   cleanest possible ONNX export and quantization path of everything surveyed
   (torch.onnx/optimum handle ConvNeXt with no custom ops).
3. Its own eval-set accuracy (0.9826) is the best of the CNN candidates, and the
   only real weakness — **"trained on an unknown dataset"** — is precisely what
   our `eval_harness` measures on our own holdout before we wire it in. A poorly
   generalizing CNN costs us a benchmark run, not a submission.
4. It is genuinely additive: a CNN makes a different class of mistakes than two
   ViTs that already correlate at r=0.75, so it can lift the ensemble even at
   similar per-model accuracy.

**Runner-up:** `VilaVision/AIgeneratedimagedetection` (MIT, DenseNet121, ~27 MB)
if we want the permissive-MIT + 4× smaller footprint — but only after we accept
the 3-class → binary fold and write a custom loader for the config-less `.pth`.

**Honest caveat:** there is no on-Hub CNN detector with complete provenance
(dataset + license + accuracy + ONNX-ready) meeting all criteria. The two viable
candidates both have "training data not stated" cards. If clean provenance or
modern-generator coverage is a hard requirement, then strictly the answer is
"no fully-documented viable third member found" — and the best we can do is
evaluate the ConvNeXt-Tiny on our holdout and let the number decide.

---

## Verification log

- 2026-08-13 — `AP6621/AI_generated_image_detection` returns 307 → VilaVision;
  documented under its real id.
- 2026-08-13 — Verified all file sizes, card frontmatter licenses, and
  eval_results via the Hub API (`/api/models/…` and `/tree/main`).
- 2026-08-13 — AIDE paper identity confirmed as "A Sanity Check for
  AI-generated Image Detection" (Shilin Yan et al.) arXiv:2406.19435.
- 2026-08-13 — Reju983 tree confirmed to contain no weight files (recursive).
- 2026-08-13 — Ricehunter & bharathh04 confirmed to have no LICENSE file
  (HTTP 404 on `raw/main/LICENSE`).

---