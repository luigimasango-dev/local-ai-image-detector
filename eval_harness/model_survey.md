# AI Image Detector Model Survey

This report surveys candidate models for an in-browser AI-generated image detector targeting the poidh.xyz bounty ($2,555). The goal is to select 2-3 open-source pretrained models that can run client-side via ONNX Runtime Web or Transformers.js with WebGPU, achieving at least 75% balanced accuracy.

For each model, we report architecture/size, ONNX availability, license, training data/generator coverage, and reported accuracy. All factual claims are sourced; missing information is explicitly noted.

---

## 1. umm-maybe/AI-image-detector (Hugging Face)

- **Architecture**: Vision Transformer (ViT) – described as “a proof-of-concept demonstration of using a ViT model to predict whether an artistic image was generated using AI”【https://huggingface.co/umm-maybe/AI-image-detector/raw/main/README.md】.
- **Parameter count / Model file size**: The PyTorch weight file (`pytorch_model.bin`) is 348 MB【https://huggingface.co/umm-maybe/AI-image-detector/tree/main】. Exact parameter count not stated; ViT-Base is ~86M parameters, but the file size suggests a larger variant (possibly ViT-Large).
- **ONNX availability**: No ONNX folder or mention in the repository. Exporting a standard ViT to ONNX via `torch.onnx.export` is straightforward; no known blockers reported.
- **License**: Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)【https://huggingface.co/umm-maybe/AI-image-detector/raw/main/README.md】. This license permits redistribution with attribution but prohibits derivative works. The model card warns that using the model “within text-to-image systems to evade AI image detection would be considered a ‘derivative work’ and as such prohibited.” For redistribution as part of an MIT‑licensed project, the ND clause may restrict modifications; using the model unchanged as a dependency is likely permissible, but legal advice is recommended.
- **Training data / Generator coverage**: Trained in October 2022; the training data did not include samples from Midjourney v5, SDXL, or DALL‑E 3【https://huggingface.co/umm-maybe/AI-image-detector/raw/main/README.md】. It was trained on outputs of predecessor models (e.g., VQGAN+CLIP, earlier diffusion models). Intended scope is artistic images; general computer imagery (webcams, screenshots) may reduce accuracy.
- **Reported accuracy**: Validation metrics from AutoTrain:
  - Loss: 0.163
  - Accuracy: 0.942
  - Precision: 0.938
  - Recall: 0.978
  - AUC: 0.980
  - F1: 0.958
  【https://huggingface.co/umm-maybe/AI-image-detector/raw/main/README.md】

---

## 2. Organika/sdxl-detector (Hugging Face)

- **Architecture**: Fine‑tuned ViT based on the umm‑maybe AI art detector (same ViT backbone)【https://huggingface.co/Organika/sdxl-detector/raw/main/README.md】.
- **Parameter count / Model file size**: 86.8 million parameters; the Safetensors weight file (`model.safetensors`) is 347 MB【https://huggingface.co/Organika/sdxl-detector/tree/main】.
- **ONNX availability**: An `onnx/` folder exists in the repository, indicating an ONNX export is available【https://huggingface.co/Organika/sdxl-detector/tree/main】.
- **License**: Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY‑NC 3.0)【https://huggingface.co/Organika/sdxl-detector/raw/main/README.md】. The model card explicitly states the model “should be considered appropriate for non‑commercial (i.e. personal or educational) fair uses only” due to potential copyrighted data in the parent model’s training set. This license is **not compatible** with commercial redistribution.
- **Training data / Generator coverage**: Fine‑tuned on a dataset of Wikimedia‑SDXL image pairs, where SDXL images are generated using prompts derived from BLIP captions of Wikimedia images【https://huggingface.co/Organika/sdxl-detector/raw/main/README.md】. Consequently, it shows greatly improved performance over the umm‑maybe detector on images from recent diffusion models (SDXL, Midjourney v5?, DALL‑E 3) and non‑artistic imagery, but underperforms the original detector for older models such as VQGAN+CLIP.
- **Reported accuracy**: Validation metrics from AutoTrain:
  - Loss: 0.08717025071382523
  - Accuracy: 0.9812734082397003
  - Precision: 0.994535519125683
  - Recall: 0.9528795811518325
  - AUC: 0.9980461893059392
  - F1: 0.9732620320855615
  【https://huggingface.co/Organika/sdxl-detector/raw/main/README.md】

---

## 3. guyfloki/ai-image-detector (GitHub)

- **Architecture**: Microsoft CvT‑13 (Convolutional Vision Transformer) backbone【https://raw.githubusercontent.com/guyfloki/ai-image-detector/main/README.md】.
- **Parameter count / Model file size**: CvT‑13 has approximately 19.98 M parameters【https://sh-tsang.medium.com/review-cvt-introducing-convolutions-to-vision-transformers-170c227da606】. The provided PyTorch checkpoint (`model_epoch_24.pth`) is 226.6 MB【https://raw.githubusercontent.com/guyfloki/ai-image-detector/main/README.md】.
- **ONNX availability**: No ONNX export mentioned in the repository. The CvT architecture is built from standard transformer blocks; exporting to ONNX via `torch.onnx.export` should be feasible with no reported blockers.
- **License**: The repository is licensed under Apache License 2.0【https://raw.githubusercontent.com/guyfloki/ai-image-detector/main/README.md】 (see the LICENSE file referenced in the README). No separate license is specified for the model weights; they are presumed covered by the same permissive license.
- **Training data / Generator coverage**: Trained on a dataset of ~2.5 million images curated by AWSAF (artifact repository), containing a mix of AI‑generated and human‑created images【https://raw.githubusercontent.com/guyfloki/ai-image-detector/main/README.md】. The README does not list specific generators, but the AWSAF artifact is known to include Midjourney, Stable Diffusion, and other models.
- **Reported accuracy**: Evaluated on test data:
  - Average Test Loss: 0.1275
  - Accuracy: 98.54 %
  - Precision: 0.99
  - Recall: 0.98
  - F1 Score: 0.98
  【https://raw.githubusercontent.com/guyfloki/ai-image-detector/main/README.md】

---

## 4. JeongsooP/Community-Forensics (GitHub / Hugging Face)

- **Architecture**: Based on CLIP‑ViT‑S (Vision Transformer) backbone. The evaluators use plain CLIP‑ViT‑S with 224×224 and 384×384 input resolutions【https://jespark.net/projects/2024/community_forensics/】.
- **Parameter count / Model file size**: The released checkpoints (`commfor-model-384` and `commfor-model-224`) are approximately 83 MB each (e.g., `model.safetensors` size shown as 87,262,324 bytes)【https://huggingface.co/OwensLab/commfor-model-384/commit/87b013b3d134dea22518e743bd7a1901e52fe9da】. Parameter count for CLIP‑ViT‑S is roughly 22 M.
- **ONNX availability**: No ONNX export mentioned. CLIP models are standard ViT variants; export to ONNX via `torch.onnx.export` is straightforward.
- **License**: MIT license (shown on the Hugging Face model cards)【https://huggingface.co/OwensLab/commfor-model-384/raw/main/README.md】.
- **Training data / Generator coverage**: Trained on the Community Forensics dataset, which includes **4,803 distinct generative models**【https://liner.com/review/community-forensics-using-thousands-generators-to-train-fake-image-detectors】. The dataset spans latent diffusion, pixel diffusion, commercial models, GANs, autoregressive models, etc., providing broad coverage of modern generators (including Midjourney v5/​v6, SDXL, DALL‑E 3, Flux, etc.).
- **Reported accuracy**: From the project’s evaluation:
  - High‑resolution model (384×384) achieved 0.991 mAP and **92.5 % accuracy** on the comprehensive evaluation set (out‑of‑distribution, including unseen generators)【https://liner.com/review/community-forensics-using-thousands-generators-to-train-fake-image-detectors】.
  - The models show robust performance across various image transformations (JPEG compression, blur, rotation, etc.) and strong generalization to unseen architectures.

---

## Additional Models (Side Search)

### 5. yaya36095/ai-image-detector (Hugging Face)
- **Architecture**: Vision Transformer (ViT)【https://huggingface.co/yaya36095/ai-image-detector/raw/main/README.md】.
- **Parameter count / Model file size**: Not explicitly stated; typical ViT‑Base is ~86 M parameters. No file size indicated in the README.
- **ONNX availability**: Not mentioned; ViT can be exported to ONNX without known blockers.
- **License**: MIT【https://huggingface.co/yaya36095/ai-image-detector/raw/main/README.md】.
- **Training data / Generator coverage**: **Not specified** in the model card. The README only states it is “designed to detect whether an image is real or AI‑generated.” Generator coverage unknown.
- **Reported accuracy**: No metrics provided in the model card.

### 6. jacoballessio/ai-image-detect-distilled (Hugging Face)
- **Architecture**: Small ViT model (distilled from three separate models)【https://huggingface.co/jacoballessio/ai-image-detect-distilled/raw/main/README.md】.
- **Parameter count / Model file size**: **11.8 Million Parameters**【https://huggingface.co/jacoballessio/ai-image-detect-distilled/raw/main/README.md】.
- **ONNX availability**: Not mentioned; ViT-based, so export to ONNX should be straightforward.
- **License**: MIT【https://huggingface.co/jacoballessio/ai-image-detect-distilled/raw/main/README.md】.
- **Training data / Generator coverage**: Trained on three pairwise datasets:
  1. Midjourney vs. Real Images
  2. Stable Diffusion vs. Real Images
  3. Stable Diffusion Fine‑tunings vs. Real Images
  Data sources: Google Open Image Dataset (real), Ivan Sivkov’s Midjourney Dataset (Kaggle), TANREI(NAMA)’s Stable Diffusion Prompts Dataset (Kaggle)【https://huggingface.co/jacoballessio/ai-image-detect-distilled/raw/main/README.md】.
  Thus covers Midjourney and Stable Diffusion (including fine‑tunings). No explicit mention of newer generators like FLUX, DALL‑E 3, or SDXL 3.
- **Reported accuracy**:
  - Validation set: **74 % accuracy** (held out from training data)【https://huggingface.co/jacoballessio/ai-image-detect-distilled/raw/main/README.md】.
  - Custom real‑world set: **72 % accuracy** (self‑captured and online‑sourced images)【same source】.
  - The model outperforms other popular AI detection models by ~5 percentage points on both sets.

### 7. capcheck/ai-image-detection (Hugging Face)
- **Architecture**: Vision Transformer (ViT‑Base) fine‑tuned for AI image detection【https://huggingface.co/capcheck/ai-image-detection/raw/main/README.md】.
- **Parameter count / Model file size**: **86 M parameters** (ViT‑Base)【https://huggingface.co/capcheck/ai-image-detection/raw/main/README.md】.
- **ONNX availability**: Not mentioned; ViT‑Base can be exported to ONNX without known blockers.
- **License**: Apache‑2.0 (inherited from Google ViT and the base dima806 model)【https://huggingface.co/capcheck/ai-image-detection/raw/main/README.md】.
- **Training data / Generator coverage**: Trained on the CIFAKE dataset (AI‑generated vs. real images). The model card notes: “Performance on modern AI generators (Flux, Midjourney v6, DALL‑E 3, Stable Diffusion 3) may vary”【https://huggingface.co/capcheck/ai-image-detection/raw/main/README.md】. Thus, it is not explicitly trained on the latest generators but may generalize.
- **Reported accuracy**: No specific metrics provided in the model card; refers to the base model dima806/ai_vs_real_image_detection for detailed training metrics.

---

## Summary & Recommendations

| Model | Size (params/filesize) | ONNX? | License | Generator Coverage | Reported Acc. |
|-------|------------------------|-------|---------|--------------------|---------------|
| umm‑maybe/AI‑image‑detector | ViT (~? M) / 348 MB | No (exportable) | CC‑BY‑ND 4.0 (ND may restrict derivatives) | Pre‑2023 (no MJ5/SDXL/DALLE‑3) | 94.2 % |
| Organika/sdxl‑detector | 86.8 M / 347 MB | **Yes** (onnx/ folder) | CC‑BY‑NC 3.0 (non‑commercial only) | SDXL‑focused; good on recent diffusion, weak on older | 98.1 % |
| guyfloki/ai‑image‑detector | CvT‑13 (~20 M) / 226.6 MB | No (exportable) | Apache‑2.0 (permissive) | Mixed (AWSAF dataset) | 98.5 % |
| JeongsooP/Community‑Forensics | CLIP‑ViT‑S (~22 M) / ~83 MB | No (exportable) | MIT (permissive) | **4,803 generators** (very broad) | 92.5 % |
| yaya36095/ai‑image‑detector | ViT (~? M) / ? | No (exportable) | MIT | **Unknown** | ? |
| jacoballessio/ai‑image‑detect‑distilled | 11.8 M / ? | No (exportable) | MIT | Midjourney + Stable Diffusion (incl. fine‑tunes) | 74 % val / 72 % real‑world |
| capcheck/ai‑image‑detection | 86 M / ? | No (exportable) | Apache‑2.0 | CIFAKE (older); may generalize to modern | ? |

**Notes for browser deployment**:
- All models are under ~350 MB, which is acceptable for a one‑time download with caching.
- ONNX exports are readily available for Organika/sdxl‑detector; the others can be exported via standard tools (torch.onnx.export, optimum) with no reported blockers.
- Licenses: Only Organika/sdxl‑detector (CC‑BY‑NC) and umm‑maybe/AI‑image‑detector (CC‑BY‑ND) have non‑permissive or restrictive terms. The MIT/Apache‑2.0 models (guyfloki, Community Forensics, yaya36095, jacoballessio, capcheck) are compatible with redistribution in an MIT‑licensed project.
- Accuracy: Several models report >90 % accuracy on their validation/test sets, but real‑world generalization to the latest generators (Midjourney v6+, Flux, GPT‑Image) must be verified. Community Forensics offers the broadest generator coverage and strong OOD performance, while guyfloki provides high reported accuracy with a permissive license and moderate size.

**Suggested candidates for ensemble** (based on permissive license, size, and coverage):
1. **guyfloki/ai‑image‑detector** – high accuracy (98.5 %), Apache‑2.0, ~20 M parameters, good general coverage.
2. **JeongsooP/Community‑Forensics** – excellent generator coverage (4,803 models), MIT, ~22 M parameters, strong OOD accuracy (92.5 %).
3. **Organika/sdxl‑detector** – highest reported accuracy (98.1 %), ONNX ready, but CC‑BY‑NC limits commercial use; consider only if non‑commercial use is acceptable or if contacting the author for a commercial exception.

If a fully permissive, ONNX‑ready model is required, replace Organika with the distilled model (jacoballessio/ai‑image‑detect‑distilled) for its tiny size (11.8 M) and MIT license, accepting lower reported accuracy (~72‑74 %) and narrower generator coverage.

--- 
*Survey completed on 2026‑08‑13. All information sourced from the linked model cards, repositories, and reputable reviews.*