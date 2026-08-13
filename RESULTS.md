# Results log

Every number here comes from a real run against the proxy eval set. Nothing is
estimated or carried over from a paper. Where a number is from a small sample,
the sample size is stated.

**Eval set:** AIDetectArena Benchmark v0.1 sample — 1,200 images (600 AI / 600
real), 17 generators, real photos from Unsplash Lite. Split stratified by
(label, generator): TUNE 599, HOLDOUT 601. See `eval_harness/README.md` for
known gaps.

**Bar to clear:** ≥0.75 balanced accuracy at a 0.65 confidence threshold.
Internal target ≥0.80 on HOLDOUT, for margin against the private benchmark.

---

## Baselines

### umm-maybe/AI-image-detector — 0.5218 balanced accuracy (TUNE, n=599)

Run 2026-08-13. **Barely above chance. Rejected.**

| Metric | Value |
|---|---|
| Balanced accuracy @0.65 | **0.5218** (95% CI 0.5018–0.5419) |
| Sensitivity (AI recall) | 0.0870 |
| Specificity (real recall) | 0.9567 |
| Confusion | TP 26 / FP 13 / TN 287 / FN 273 |

The failure mode is entirely on the AI side: it calls almost everything "real."
Per-generator recall confirms it is blind to modern generators —
`flux_2_flex` **0.0000**, `hunyuan_v3` **0.0000**, nothing above `sd_3.5_large`
at 0.20.

This is the expected result, not a surprise: the model was trained in October
2022, before SDXL, Midjourney v5 and DALL·E 3, and our eval set is entirely
2025–26 generators. It confirms the generation gap the published benchmark
study warns about is real and severe, and it independently validates that the
harness measures something — a broken pipeline would have produced a flat
0.5000, not this sensitivity/specificity asymmetry and per-generator spread.

Rejected on licence anyway (CC-BY-ND, see `MODEL_LICENSING.md`); the accuracy
result makes the licence question moot.

**Useful takeaway for ensemble design:** the scarce quantity is *sensitivity to
modern generators*, not specificity. A second ensemble member that merely adds
more "this looks real" confidence adds nothing.

---

## Primary candidate

### Community-Forensics (`OwensLab/commfor-model-224`)

MIT licensed, 21.67M params, ViT-Small @ 224px, patch 16. Sigmoid output
confirmed to be P(AI) by reading their source, then verified empirically
(polarity check, not assumed). Full run: 1,200 images in 131.9s on CPU, 0
skipped, 0 failed batches.

**Headline: AUC 0.9017 — the ranking is strong. The 0.65 threshold is the
problem, not the model.**

| Threshold | Split | Balanced accuracy |
|---|---|---|
| 0.65 (raw, as the bounty grades) | TUNE | 0.7091 |
| **0.0553 (optimal, fitted on TUNE)** | **TUNE** | **0.8513** |
| **0.0553 (same threshold, unseen data)** | **HOLDOUT** | **0.7805** (95% CI 0.7473–0.8121) |

HOLDOUT detail at t=0.0553: sensitivity 0.7010, specificity 0.8600,
confusion TP 211 / FP 42 / TN 258 / FN 90.

**Read this carefully — the two numbers that matter are 0.8513 and 0.7805.**
The first is what we'd get by fitting and reporting on the same data; the
second is what we get on images the threshold never saw. The **−0.071 gap** is
the overfitting the tune/holdout split exists to expose. Only 0.7805 is real.

**Verdict: clears 0.75, but not with enough margin.** The CI lower bound
(0.7473) sits *below* the bar. Shipping on this alone is a coin-flip against a
private benchmark that is likely differently distributed from ours.

#### Two concrete implications

1. **Calibration is now a hard engineering requirement, not a nicety.** The
   optimal cut is ~0.055, but the bounty grades at 0.65. The extension must
   emit a *calibrated* score so its decision boundary lands at 0.65. This is
   ordinary probability calibration — a monotonic transform that changes
   neither the ranking nor which images are flagged at the optimum — not
   benchmark gaming. Worth ~+0.14 balanced accuracy versus emitting the raw
   sigmoid, which would score 0.71 and fail.
2. **Where the remaining error lives.** Per-generator recall on TUNE @0.65 is
   uneven: strong on `flux_2_flex` (0.75), `glm_image` (0.72), `qwen_2512`
   (0.70), `ideogram_v3` (0.68); weak on `flux_pro_v1.1` (0.14),
   `grok_aurora` (0.14), `seedream_v4` (0.14), `z_image` (0.14),
   `gemini_3_pro` (0.22). A second ensemble member is worth adding only if it
   is strong on *those* generators — another model with the same blind spots
   would add cost and no accuracy.

---

### Community-Forensics 384 (`OwensLab/commfor-model-384`)

AUC **0.8822** — slightly *worse* than the 224 model despite 3× the pixels.
Higher input resolution does not help this task. Alone it is the weaker member,
but it earns its place in the ensemble by being *wrong differently* (see below).

---

## ✅ GATE CLEARED — fused ensemble, calibrated

**`commfor-224` + `commfor-384`, equal weights, Platt-calibrated on TUNE,
evaluated on HOLDOUT at the bounty's own 0.65 threshold:**

| Metric | Value |
|---|---|
| **Balanced accuracy** | **0.8338** (95% CI **0.8055 – 0.8636**) |
| Sensitivity (AI recall) | 0.7475 |
| Specificity (real recall) | 0.9200 |
| Confusion | TP 225 / FP 24 / TN 276 / FN 76 |
| Bar to clear | 0.75 |
| Internal target | 0.80 |

**The CI lower bound (0.8055) clears the bar**, which the single model's did not
(its lower bound was 0.7473, *below* 0.75). That is the difference between
"probably passes" and "passes".

### Why this result is trustworthy

- **TUNE → HOLDOUT drop: +0.0059.** Essentially zero, and slightly *positive*.
  The calibration did not memorize the tune split. Compare the hard-threshold
  approach on the single model, which dropped −0.071.
- **0.65 is now the peak of the threshold sweep** (0.8203 at 0.50, 0.8338 at
  0.65, 0.8089 at 0.75). Calibration put the operating point exactly where the
  graders read it, rather than us hoping their threshold suited our model.
- **Log loss more than halved** (0.79 → 0.38), so the scores became genuinely
  better probabilities — not merely shifted.

### Why the ensemble works

The two members correlate at only **r = 0.7508** ("moderately correlated") and
disagree by >0.5 on 10.2% of images. That residual diversity is the whole
gain: 224 alone calibrated to 0.7839, the pair to 0.8338 (**+0.050**). Two
highly-correlated models would have cost double the download and inference for
nothing.

### False positives — the failure mode that sank claim #1014

Specificity **0.9200**: 24 false positives out of 300 real photos (8%). The
competing claim was publicly criticised for exactly this. Ours is not immune,
but it is measured and reported rather than discovered by a reviewer.

### Where it is still weak (per-generator recall @0.65, HOLDOUT)

`grok_aurora` **0.3077**, `flux_pro_v1.1` 0.4667, `gpt_image_1.5` 0.4706. Strong
on `wan_v2.6` 0.9444, `qwen_2512` 0.8947, `z_image` 0.8824. If the private
benchmark over-weights Grok or GPT-Image relative to ours, our real score lands
below this one. This is the single largest known risk to the number above.

---

---

## ⚠️ ROBUSTNESS FAILURE — the result does not survive realistic web images

**This is the most consequential finding in this document. Read it before
acting on the 0.8338 above.**

Rebuilt the HOLDOUT set under the `combo` condition — downscale longest edge to
512px, re-encode at JPEG quality 75 — which is roughly what any social platform
or CMS does to an uploaded image. Same models, same calibration.

| Condition | Balanced accuracy @0.65 | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Pristine (as benchmarked) | **0.8338** | ~0.90 | 0.7475 | 0.9200 |
| 512px + JPEG 75 | **0.7290** (CI 0.6942–0.7639) | **0.7778** | 0.5781 | 0.8800 |
| **Change** | **−0.105** | **−0.12** | **−0.169** | −0.040 |

**0.7290 is below the 0.75 bar.** The CI upper bound (0.7639) barely reaches it.

### This is capability loss, not a calibration problem

The obvious hope is that the clean-fitted calibration simply mismatches the
degraded score distribution, and refitting would recover it. It would not:

- **AUC drops from ~0.90 to 0.7778.** AUC is threshold-free, so this is the
  ranking itself degrading — the models genuinely cannot separate degraded AI
  images from degraded real ones as well.
- **Even at the optimal threshold**, degraded balanced accuracy tops out at
  **0.7324** — still below the bar. Recalibration buys back some of the loss
  (0.6013 → 0.7324 at best threshold) but cannot cross 0.75.

Sensitivity takes almost all the damage (−0.169 vs −0.040 for specificity):
JPEG quantization destroys exactly the high-frequency generator fingerprints
these detectors rely on, so degraded AI images start looking like real ones.

### Why this probably matters for the bounty, not just in theory

The bounty describes its benchmark as *"a held-out set of real and AI-generated
images assembled from publicly available datasets **and additional
web-realistic samples**."* Web-realistic almost certainly means re-encoded. If
a meaningful share of their set is degraded, our true score sits somewhere
between 0.73 and 0.83 — and the low end fails.

### Third member tested: ConvNeXt-Tiny — robust, but too weak to save it

`mmanikanta/ConvNeXT_AI_image_detector` (Apache-2.0, ConvNeXt-Tiny, 28.6M
params) was the best CNN candidate found. It is architecturally unrelated to our
two ViTs and correlates with them at only **r = 0.24–0.30** (vs r = 0.64 between
the two CF models) — genuinely diverse.

And it is essentially **immune to compression**:

| Model | AUC clean | AUC degraded | Loss |
|---|---|---|---|
| CF ensemble | ~0.90 | 0.7778 | **−0.12** |
| ConvNeXt-Tiny | 0.6706 | 0.6672 | **−0.003** |

But its standalone discrimination is poor. Its model card reports 0.9826 eval
accuracy; on our modern-generator holdout it scores **AUC 0.6706** — barely
better than chance-plus. That gap is a textbook case of a model that has learned
its own test set rather than the task, and it is exactly why every candidate
gets re-measured here rather than trusted from its card.

Fusing all three on degraded images:

| Ensemble | Degraded AUC | Best achievable balanced accuracy |
|---|---|---|
| CF-224 + CF-384 | 0.7778 | 0.7324 |
| **+ ConvNeXt (0.4/0.4/0.2)** | **0.7993** | **0.7390** |

The diversity is real — AUC improves by +0.021 — but **0.7390 is still below the
0.75 bar**. A robust-but-weak member cannot rescue a strong-but-fragile pair.
ConvNeXt is not the answer.

### The fix is a more robust model, not more calibration

During the third-model search we found detectors trained on
`OwensLab/CommunityForensics-Small` **with explicit social-media robustness
augmentation** — random JPEG QF 30–95, Gaussian blur, downscale-upscale — which
is precisely this failure mode addressed at training time. That is now the
highest-value lead, ahead of squeezing more from the current pair.

### How much degradation can we actually survive?

We cannot observe what fraction of the private benchmark is re-compressed, so
rather than guess, `mixed_benchmark_estimate.py` sweeps the ratio (5 seeds per
point, holdout only, calibration fitted on TUNE):

| % of benchmark web-realistic | Balanced accuracy | Verdict |
|---|---|---|
| 0% | 0.8338 | PASS |
| 25% | 0.8072 | PASS |
| 50% | 0.7842 | PASS |
| 60% | 0.7686 | PASS |
| **~70%** | **~0.75** | **break-even** |
| 75% | 0.7526 | FAIL |
| 100% | 0.7290 | FAIL |

**We survive a benchmark that is up to roughly two-thirds heavily-degraded.**
That is a more comfortable position than the 0.7290 worst case alone suggests —
a benchmark composed *entirely* of aggressively re-compressed images is an
unlikely construction.

**A hypothesis that turned out wrong:** fitting the calibration on a 50/50
clean+degraded mix should, intuitively, do better on a mixed benchmark. It does
not — it is *slightly worse at every ratio* (e.g. 0.7759 vs 0.7842 at 50%).
Platt scaling is monotonic, so it cannot repair ranking; fitting on two
distributions just compromises between them and lands slightly worse on both.
**Keep the clean-fitted calibration.** Recorded because it is the kind of
plausible idea that would otherwise get re-tried later.

### The milder condition is the one that should worry us

`combo` was chosen as a harsh case. So we also measured **JPEG 85 with no
downscaling at all** — visually near-lossless, and roughly the *floor* of what
any image published on the web has been through:

| Condition | Balanced accuracy @0.65 | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Pristine (benchmark as shipped) | 0.8338 | ~0.90 | 0.7475 | 0.9200 |
| **JPEG 85 only** | **0.7474** (CI 0.7158–0.7806) | 0.8284 | 0.5748 | 0.9200 |
| 512px + JPEG 75 | 0.7290 | 0.7778 | 0.5781 | 0.8800 |

**Most of the loss comes from JPEG compression itself, not from downscaling.**
Going from pristine to merely-JPEG-85 costs **−0.086**; adding an aggressive
downscale and dropping to quality 75 costs only a further −0.018.

This reframes the whole risk assessment, and not favourably:

- Our eval set's "pristine" images are original-quality archive files. **That
  condition barely exists on the open web** — essentially every image a browser
  encounters has been JPEG-encoded at least once.
- So the realistic expectation is the **JPEG 85 row (0.7474)**, not the pristine
  row — and that sits *just below* the 0.75 bar, with the confidence interval
  straddling it.
- The earlier "we survive up to ~70% degraded" conclusion assumed the
  alternative was pristine images. If the true mix is "mildly compressed" vs
  "heavily compressed" rather than "pristine" vs "heavily compressed", our
  expected score is around 0.73–0.75 — a coin flip against the bar.

Note the failure is entirely on the AI side again: specificity is unchanged at
0.9200 while sensitivity falls 0.7475 → 0.5748. Compression is not making real
photos look artificial; it is scrubbing the generator fingerprints out of AI
images.

**Verdict: do NOT submit on this ensemble.** Not because it is far off — it is
close — but because the most probable real-world condition lands within noise of
the pass/fail line, and a bounty submission is pass/fail with no partial credit.
Solving robustness (#17) is worth more than any further calibration or fusion
tuning, both of which are now exhausted.

---

## Standing caveats (do not let the 0.8338 obscure these)

1. **This is a proxy benchmark.** The private one may be harder, easier, or
   simply different. 0.8338 here does not guarantee ≥0.75 there.
2. **Our real images are all Unsplash Lite** — a professional-photography
   aesthetic. The private set's "web-realistic samples" likely include phone
   snaps, screenshots and heavily compressed images. Untested (#9).
3. **Our images are pristine.** No JPEG recompression or resizing, which is how
   images actually arrive in a browser, and a known weak point for detectors.
   Untested (#10) — and a plausible way this result fails to transfer.
4. **Calibration is fitted on our distribution.** If the private set's score
   distribution differs, the Platt parameters transfer imperfectly. The AUC
   (0.90) is the distribution-independent part; the calibration is not.

## Open experiments

- Second permissively-licensed detector with a *different* architecture (#13),
  targeted specifically at `grok_aurora` / `flux_pro_v1.1` / `gpt_image_1.5`.
  Candidate spotted: an MIT-licensed DenseNet detector (CNN, genuinely
  different from these two ViTs) — unverified.
- Robustness under JPEG recompression / downscaling (#10).
- ONNX export + in-browser latency and download-size measurement (#8).

---

## Shipping configuration (final)

**Ensemble:** `commfor-model-224` + `commfor-model-384`, fp16 ONNX, equal
weights, Platt-calibrated (A=0.6758, B=2.0932) fitted on TUNE only.

| Condition | Balanced accuracy @0.65 | Note |
|---|---|---|
| Pristine (HOLDOUT) | **0.8321** | fp16, i.e. what actually ships |
| JPEG 85 | 0.7474 | realistic web floor |
| 512px + JPEG 75 | 0.7290 | deliberate worst case |

Bar to clear: 0.75.

### Precision format: fp16, decided by measurement

| Format | cf224 AUC | cf384 AUC | Size (both) |
|---|---|---|---|
| PyTorch FP32 | 0.8854 | 0.9039 | — |
| **fp16 (shipping)** | **0.8847** | **0.9042** | **~87 MB** |
| int8 | 0.8421 | 0.8740 | ~43 MB |

int8 halves the download again but costs 0.043 / 0.030 AUC. With the pass bar
only a few points away, that trade is not worth taking. fp16 is lossless within
noise.

### Rejected: a third ensemble member

`jacoballessio/ai-image-detect-distilled-efficientnet` (MIT, 16 MB, CNN) is
genuinely compression-robust — AUC 0.7482 clean → 0.7389 at JPEG 85, a drop of
only 0.009 versus the ViT pair's 0.055 — and usefully decorrelated (r = 0.43).
It also improves raw ranking on degraded images (AUC 0.8284 → 0.8329) and
best-achievable accuracy (0.7556 → 0.7738).

**But at the graded 0.65 threshold, with calibration fitted the same way, the
trio is worse on both conditions:**

| Ensemble | Clean @0.65 | JPEG 85 @0.65 |
|---|---|---|
| **CF pair** | **0.8321** | **0.7474** |
| + EfficientNet | 0.8172 | 0.7342 |

A useful reminder that AUC is not the objective — the objective is accuracy at
a *fixed* threshold someone else chose. A model can rank better and still cost
you accuracy at the operating point that gets graded. Keeping the pair.

### False positives on non-photographic real content

100 images drawn programmatically (charts, UI mockups, text documents, logos),
guaranteed non-AI by construction, at t=0.65:

| Class | n | False positives | Rate |
|---|---|---|---|
| logo_graphic | 25 | 0 | 0.00 |
| screenshot_ui | 25 | 0 | 0.00 |
| text_document | 25 | 0 | 0.00 |
| chart_diagram | 25 | 5 | 0.20 |
| **overall** | **100** | **5** | **0.05** |

The feared failure mode — "synthetic-looking content reads as AI-generated" —
largely did not materialise, and the 5% rate is *better* than the 8% on real
Unsplash photographs. Charts are the exception and worth watching.

### Browser parity (extension/test_page.html)

6/6 decisions match the Python harness at t=0.65. Worst raw score drift 0.136,
which is expected: canvas resampling and PIL resampling are different
algorithms. Decision agreement, not score identity, is the meaningful test.
~240 ms/image on WebGPU after a 3.5 s model load; the WASM fallback is ~20×
slower, which is why the WebGPU-capable ORT bundle is the one vendored.

---

## Retrained classifier head — tested, and REJECTED

An attempt to fix the compression weakness without a GPU: freeze the ViT
backbone, extract 384-d features on CPU, and retrain only the final
`Linear(384->1)` head on clean **and** compressed images. Roughly 385 trainable
parameters, minutes of CPU time, no GPU rental required.

It looked like a decisive win:

| | Original head | Retrained v1 | Retrained v2 |
|---|---|---|---|
| HOLDOUT clean | 0.8338 (pair) | 0.8802 | 0.8786 |
| HOLDOUT compressed | 0.7290 | **0.8620** | **0.8553** |
| False positives, charts/logos/UI | 0.05 | 0.12 | **0.00** |

It also passed a genuinely hard test. **Leave-generators-out** (four folds,
entire generators withheld from training) gave mean AUC **0.9593** on
generators never seen, a gap of only 0.0407 — and the worst fold withheld
`gpt_image_1.5` and `flux_pro_v1.1`, our weakest, and still scored 0.9213.

### What killed it

Luigi's own phone videos. 99 frames extracted from handheld footage —
noisy, badly lit, motion-blurred, already codec-compressed — the closest thing
available to the bounty's "web-realistic" real class:

| On 99 phone-video frames | False positives |
|---|---|
| **Original head** | **1 / 99 (1%)** |
| Retrained head | **15 / 99 (15%)** |

Fifteen times worse on the category that matters most. **Ship the original
head.**

### Why it failed, and why the validation missed it

Linear probing on 384-d features with ~1,200 samples fits whatever notion of
"real" it is shown. Trained on Unsplash, it learned "professional photograph =
real" and flagged charts. Trained on Unsplash + charts, it learned "stock photo
or chart = real" and flagged phone footage. Each fix moved the blind spot
rather than removing it.

The original head does not have this problem because its authors trained it on
556K images across ~4,803 generators — breadth we cannot reproduce from a
1,200-image benchmark, at any amount of cleverness.

**The methodological lesson is the valuable part:** leave-generators-out
validated the *AI* side and passed emphatically — twice — while being
structurally blind to this failure, because Unsplash sat on the "real" side of
every fold. A held-out test only protects the axis it actually varies. The real
side needed its own out-of-distribution test, and only a genuinely different
source of real images could provide it.

Had we trusted the headline numbers and the one passing validation, this would
have shipped and very likely lost the bounty.
