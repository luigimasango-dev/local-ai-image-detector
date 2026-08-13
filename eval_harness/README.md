# AI Image Detector — Eval Harness

Internal proxy benchmark for evaluating AI-generated-image detection models
(real vs AI classification) for a Chrome-extension bounty that requires
clearing **75% balanced accuracy** on a private benchmark. This harness builds
our own proxy eval set because the real bounty benchmark is private.

> **PROXY BENCHMARK — READ THIS FIRST**
>
> The eval set built here is a **proxy**, not the real private bounty
> benchmark. Results on this set are **directional** (useful for relative
> comparison and quick iteration) and **not a guarantee** of the score on the
> real benchmark. Treat any number below 0.75 balanced accuracy with the
> appropriate salt — the real dataset may be harder, easier, or differently
> distributed.

---

## What the eval set is

### Primary source: AI Detector Arena Benchmark v0.1

**Important finding (verified 2026-08-13):** the Hugging Face repository
`aidetectarena/ai-image-detector-benchmark` is **empty** ("The dataset is
currently empty"). Its dataset card links to the real release:

- Homepage: <https://aidetectarena.com/datasets/v0.1>
- CDN mirror zip (the archive we fetch): <https://aidetectarena-benchmark.nyc3.cdn.digitaloceanspaces.com/datasets-archive/benchmark-v0.1.zip>
- GitHub (metadata, evaluation scripts): <https://github.com/AI-Detect-Arena/benchmark-dataset>
- DOI: [10.5281/zenodo.18620634](https://doi.org/10.5281/zenodo.18620634)

`fetch_dataset.py` attempts to load the HF dataset via the `datasets` library
first; when it turns out to be empty/fails (which it does), it logs that
plainly and falls back to the CDN mirror, which is the actual v0.1 release of
the same dataset.

**Dataset facts (v0.1):**

| Metric | Value |
|--------|-------|
| Total images | 2,038 |
| AI-generated | 1,018 |
| Real photos | 1,020 |
| AI generators | 17 |
| Categories | 6 (portrait, landscape, art, food, animal, product) |
| Real source | Unsplash Lite (Unsplash License) |
| Distortions | none applied (original quality) |
| Full archive size | ~1.39 GiB |
| Release date | Feb 2026 |

**Generator coverage (17):** Flux Pro v1.1, Flux 2 Flex, Flux Schnell, GPT
Image 1.5, Gemini 3 Pro, Grok Aurora, Stable Diffusion 3.5 Large, Ideogram v3,
Leonardo Phoenix, Recraft v3, Hunyuan v3, Seedream v3, Seedream v4, Qwen 2512,
GLM Image, Wan v2.6, Z Image.

### How we sample

The full archive is ~1.4 GiB. To keep the download reasonable we do **not**
pull the whole zip. `fetch_dataset.py` reads the zip's central directory via a
small tail range request, then downloads **only the sampled image files** with
HTTP `Range` requests. Default sample: **600 AI + 600 real**, stratified
across the 6 categories, deterministically seeded (`--seed`, default 42).
Raise with `--ai-count 1000 --real-count 1000` (up to the available
1018/1020) if you want nearly the full set.

### Supplementary slice (optional): CIFAKE

The AIDetectArena set is all **modern** generators (late 2025 / early 2026
models). It has no older-generation coverage (e.g. Stable Diffusion v1.x-era
outputs, low-res crops). To add a different distribution, pass
`--include-cifake`. It pulls a small balanced slice (default 200 fake + 200
real) from `yanbax/CIFAKE_autotrain_compatible` via the `datasets` library:

- 32×32 crops from Stable Diffusion v1.4 (AI side) + real CIFAR-10 photos.
- Label semantics handled explicitly (we read the class-names / fall back to
  the CIFAKE convention 0=fake/AI, 1=real) — not assumed.
- These are tiny images; they will be easier for most detectors and are meant
  as a distribution-probe slice, not a primary benchmark.

### Known gaps / biases (say it out loud)

1. **All modern generators.** The primary set is dominated by 2025–2026
   models (Flux, GPT Image, Gemini, Grok, SD3.5, …). If the real bounty
   benchmark skews older (SD 1.x/2.x, Midjourney, DALL-E 2), our numbers will
   be optimistic for modern-aware detectors.
2. **No Midjourney/DALL-E 2 in the generator list.** The FAQ mentions
   Midjourney/DALL-E 3 in prose, but the v0.1 generator table lists 17 models
   and *does not* include a distinct `midjourney` entry (GPT Image 1.5 covers
   the OpenAI line). Coverage of Midjourney in particular is **not verified**.
3. **Real images are single-source.** Real photos come from Unsplash Lite
   only — a photography aesthetic. The private benchmark's real class may
   contain phone snapshots, screenshots, or heavily compressed web images.
4. **No distortions / pristine quality.** v0.1 applies no JPEG/resize
   augmentation. Real-world Chrome-extension input will be re-encoded by
   social platforms; our proxy does not model that degradation.
5. **No overlap guarantee.** This is a *different* dataset from the private
   one. Good here ≠ good there.

---

## Installation

```bash
pip install -r requirements.txt
# torch: if you want a CPU-only wheel to keep install small:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Scripts

### 1. Fetch the eval set

```bash
python eval_harness/fetch_dataset.py
# options:
#   --ai-count 600 --real-count 600   # sample sizes (default 600/600)
#   --seed 42                          # deterministic stratification
#   --include-cifake --cifake-per-class 200
```

Outputs:
- `eval_harness/data/images/...` — the image files.
- `eval_harness/data/labels.csv` — `image_id,filepath,label,generator` where
  **label = 1 for AI-generated, 0 for real**, and `filepath` is relative to
  the `eval_harness/data/` directory.
- `eval_harness/data/manifest.json` — provenance summary (sources, counts).

### 2. Run inference

```bash
python eval_harness/run_inference.py --model umm-maybe/AI-image-detector \
    --backend transformers --device cpu --batch-size 16
```

- `--model` is any HF model id or local path loadable as a transformers
  image-classification model.
- Output: `eval_harness/predictions/<model_name>.json`, a dict mapping
  `image_id -> probability_ai_generated` (float in [0,1]).
- **Label-order safety:** the script inspects the model config
  (`id2label`/`label2id`) and the pipeline's returned labels to work out which
  output class is "AI-generated" vs "real" by keyword — it does **not** assume
  index 0/1. If the mapping is ambiguous it fails loudly instead of guessing,
  and accepts `--ai-label=<exact class string>` (e.g. `--ai-label=FAKE`) as an
  explicit override.

### 2b. Split into TUNE / HOLDOUT (do this before fitting anything)

```bash
python eval_harness/split_eval_set.py            # 50/50, stratified, seed 42
```

Fusion weights and the confidence threshold are **fitted parameters**. Fitting
them on the same images we then quote a score from inflates that score and it
will not transfer to the private bounty benchmark. So:

- Fit weights/threshold on `data/split_tune.txt`
- Quote the **go/no-go number** from `data/split_holdout.txt`
- Report both, so the size of the gap between them is visible

The split is stratified by `(label, generator)` so both halves carry the same
label balance and generator mix. Outputs `split_tune.txt`, `split_holdout.txt`,
`split_summary.json`.

### 3. Score predictions

```bash
python eval_harness/score.py \
    --predictions eval_harness/predictions/model_a.json \
                   eval_harness/predictions/model_b.json \
    --weights 0.5 0.5 \
    --threshold 0.65 \
    --split eval_harness/data/split_holdout.txt
```

- Combines models with a weighted average per image (default: equal weights).
- Reports **balanced accuracy = (sensitivity + specificity) / 2** at the given
  threshold (default **0.65**, matching the bounty), with AI as the positive
  class, plus the full confusion matrix (TP/FP/TN/FN).
- Also prints accuracy and balanced accuracy at reference thresholds
  0.50 / 0.60 / 0.65 / 0.70 / 0.75.
- Images with no prediction from *any* model are excluded and reported — we
  never silently assume 0.5. Images missing from some models are combined over
  the models that did score them (weights re-normalized).
- **Specificity is reported as a first-class metric** (true-negative rate on
  real photos). The competing claim on this bounty was publicly criticised for
  flagging real photos as AI; balanced accuracy alone can hide that when
  sensitivity is high.
- **Per-generator AI recall breakdown** — a headline number can hide a
  generator we are completely blind to. Generators scoring below 0.5 recall are
  flagged `<-- weak`.
- **95% bootstrap CI** on balanced accuracy (positive and negative classes
  resampled independently). Disable with `--no-ci`. With ~1,200 images a
  difference of a point or two between two ensembles is usually noise — check
  whether the intervals overlap before preferring one.
- `--split <file>` restricts scoring to listed image_ids; `--json-out <path>`
  also writes the metrics as JSON.

---

## Interface contract (stable — used by downstream jobs)

`labels.csv` and the `score.py` CLI are stable and are reused by a separate
downstream job that evaluates other candidate models. Do not change without a
coordinated update:

- `labels.csv` columns: `image_id,filepath,label,generator`; label ∈ {0,1}
  with **1 = AI-generated**; filepath relative to `eval_harness/data/`.
- Prediction JSON: `{"<image_id>": <float 0..1>}` = probability that the image
  is AI-generated.
- `score.py` flags: `--predictions` (nargs+), `--weights` (nargs+ float, equal
  by default), `--threshold` (default 0.65), `--labels` (default
  `data/labels.csv`).

---

## Tests

```bash
python tests/test_label_mapping.py     # class-label resolution
python tests/test_scoring.py           # balanced-accuracy math
# or: python -m pytest tests/ -q
```

Both suites exist because of bugs found in the first generated version of
`run_inference.py`, and both fail against that old code:

1. **Class-index lookup bug (silent, severe).** The transformers
   image-classification pipeline returns the label *string*
   (`{"label": "artificial", "score": 0.17}`), never the integer index. The
   original code keyed predictions by index, so the AI-class lookup missed on
   every image and fell through to summing all class probabilities — assigning
   **1.0 to every image**. That does not crash; it produces a tidy, entirely
   meaningless **0.5000 balanced accuracy** for every model tested. Fixed by
   resolving through `config.label2id` (`resolve_label_index`).
2. **Substring keyword matching.** Class names were matched by substring, so
   `"portrait"` contains `"ai"` and `"organic"` contains `"gan"` — either would
   silently invert the AI/real mapping. Now matched on whole tokens, with
   `not_ai` / `non-ai` collapsed to the real side.

If you ever see exactly `0.5000` balanced accuracy from a model, suspect the
class mapping before concluding the model is useless.

## Status / honesty notes

- The HF repo `aidetectarena/ai-image-detector-benchmark` is empty; the CDN
  mirror (which the card links to) is the real v0.1 release and is what this
  harness downloads. No accuracy numbers are fabricated anywhere; nothing is
  hardcoded — every number in the output is computed from real downloaded
  data.
- If a download fails mid-run, the script reports the failing images and
  continues; a later re-run skips already-downloaded files (size check).
- `run_inference.py` and `score.py` produce no metrics on their own — they only
  emit probabilities / compute scores from real model outputs.

## License notes

- AIDetectArena data: CC-BY-4.0 (data), MIT (code/eval scripts).
- CIFAKE: MIT; images derived from CIFAR-10 (MIT) and Stable Diffusion outputs.
- Unsplash photos: Unsplash License.
