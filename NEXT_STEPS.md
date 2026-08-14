# Where this stands, and what to do next

Written 2026-08-14, after checking the live bounty page and running three
independent reviews of the project.

---

## The short version

The extension works, is honest, and is well engineered. **It is also no longer
competitive on raw accuracy**, because the field went from 1 claim to 16 in
about seventeen hours.

Do not submit today. There is a specific, identified path to a competitive
submission, and it reuses everything already built.

---

## 1. Competitive position (checked on the live page, not remembered)

| Claim | Reported balanced accuracy @0.65 |
|---|---|
| PxShield (#1030) | 97.1% |
| Blur (#1025) | 95.6% |
| OriginLens (#1033) | 91.3% clean / **87.3% web-degraded** / 84.6% hard |
| sieve (#1027) | 91.3% / **87.3%** / 84.6% |
| local-ai-image-detector (#1028) | 89.2% |
| Local Lens (#1023) | 81.3–94.6% |
| PixelWitness (#1026) | 82.5% |
| **This project** | **83.2% clean / 74.7% JPEG-85 / 72.9% hard** |

Winner-take-all; the first *valid* claim wins. Eight or more claims sit ahead
of us reporting better numbers, so submitting now wins only if essentially all
of them fail.

**Three things cut the other way, and they are not nothing:**

1. **Every one of those numbers is self-graded**, on a benchmark each claimant
   chose. The bounty page itself records that a previously-tested open-source
   detector failed to reach 60% on the real benchmark. Our numbers come with
   published train/test image ID lists (`eval_harness/splits/`) that anyone can
   check; most of theirs cannot be checked at all.
2. **The maintainer has explicitly committed to testing false positives.**
   After a complaint that claim #1014 "flags a lot of real photos falsely as ai
   generated", kenny replied: *"yes we will review thoroughly for both ai
   detection and false positives"*. Our false-positive numbers are the
   strongest part of this project — **1%** on 363 frames of consumer phone
   video, 5% on charts/logos/UI, 8% on studio photography. A detector claiming
   97% that flags ordinary photos will not survive that review.
3. **Claim #1028 uses this project's exact name** (`local-ai-image-detector`),
   the same stack (ONNX Runtime Web, WebGPU→WASM), and an "explicit graphic
   gate" for charts — the same failure mode we found and fixed. The approach is
   sound; we are simply not first.

---

## 2. The opening the competition handed us

OriginLens (#1033) and sieve (#1027) are **independent** claimants reporting
**byte-identical** numbers: 91.3 / 87.3 / 84.6. Identical numbers from
different people means a shared upstream artifact with published results.

OriginLens's repository contains `training/vendor/cf_models.py` and
`cf_transforms.py` — **`cf` is Community Forensics, the same base model this
project already uses.** Their "FT1 artifact" is a *fine-tune* of it.

This matters more than anything else in this document:

- We concluded, after an extensive search, that no public detector was both
  accurate and compression-robust. **That conclusion was wrong**, and two
  strangers demonstrated it.
- The gap is not the base model — we already have it. The gap is a fine-tune
  with compression augmentation.
- Their reported **87.3% after web resize + JPEG q60** is precisely the
  weakness we spent hours failing to close (we sit at 74.7%).
- Community Forensics is MIT, so a derivative is legally clean *provided the
  fine-tuned weights are themselves published under a usable licence* — which
  must be verified, not assumed.

---

## 3. What to do, in order

### A. The artifact — IDENTIFIED, and it changes the picture

| Field | Value |
|---|---|
| File | `ft1_best_fp16.onnx` |
| Size | 43,779,538 bytes |
| SHA-256 | `87277637277f7d4f82222f0aaf1b4b114132ca13e5e4f02f71a65693e9606aed` |
| Licence | **MIT** |
| Source | [sieve v0.1.0 release](https://github.com/Phineas1500/sieve-ai-image-detector/releases/tag/v0.1.0), published 2026-08-13 20:26 UTC |
| What it is | a **fine-tune of Community Forensics** — the same base model this project already uses (`training/vendor/cf_models.py`, `cf_transforms.py` in the OriginLens repo) |

So the compression-robust model does exist, it is MIT, and it is a fine-tune of
our own base — not some exotic architecture we missed. Our earlier conclusion
that no such model existed was wrong, and the right lesson is that **we searched
model hubs but never looked at what the competition was shipping.**

**But adopting it does not win the bounty**, and this is the decisive point:

- sieve *created* that model and filed claim **#1027** with it.
- OriginLens took it and filed **#1033**.
- The rules give priority to the **earliest valid claim**. A third submission
  using the same artifact lands behind both, at the same accuracy.

Taking it would move us from 74.7% to roughly 87% on compressed images and
still leave us third in line among the users of that one file.

### B. Re-verify honestly before claiming anything
Whatever model we adopt gets measured the same way as everything else here:
fitted on TUNE only, reported from HOLDOUT, plus the compression conditions,
plus the false-positive sets (phone frames, charts, Unsplash). No self-graded
homework.

### C. Install and test on real websites — **needs Luigi**
The last functional unknown. The cross-origin *mechanism* is verified
(`extension/test_crossorigin.html`: canvas taints as expected on a real CDN
image, HTTP-cache fallback recovers it). What is untested is the full
production path — content script → service worker → offscreen → badge — on
actual websites. See `INSTALL.md`. Check: real percentages rather than grey "—"
markers, no console errors, zero extension-initiated network requests, badges
appearing on lazy-loaded images after scrolling, and images inside embedded
tweets. Then repeat with DevTools set to **Offline**, which is the graders'
actual condition.

### D. Push to GitHub — **needs Luigi**
The repo is committed locally (61 files, ~590 KB; weights are regenerated by
`build.py` rather than committed). Publishing is an account-level action and a
decision, so it is not automated here.

### E. Then submit, leading with the honest evidence
Lead the claim with the false-positive numbers and the published splits, not
just the headline accuracy. Given that a competitor was publicly criticised for
undisclosed false positives and the maintainer committed to checking them, our
candour is a competitive asset, not a liability. **Do not polish it out.**

---

## 4. What is already done and does not need revisiting

- Real in-browser inference, verified against the Python harness (6/6 decisions
  agree at the graded threshold), ~200 ms/image on WebGPU.
- Reproducible build from a clean clone, tested — which found and fixed two
  bugs that would only ever have appeared on someone else's machine.
- Offline operation verified by capturing every network request.
- Cross-origin images recovered from Chrome's HTTP cache rather than re-fetched.
- Eight independently-audited defects fixed (timeouts, WASM main-thread
  blocking, lazy-load placeholders, shared badges, iframe/shadow-DOM/CSS
  background coverage, popup status, stale docs, dependency pinning).
- Licence position clean: MIT throughout, upstream attribution shipped, two
  otherwise-strong models rejected on licence grounds.
- Full disclosure of what was fitted and on what data, with the exact image ID
  lists published so overlap can be measured rather than trusted.

---

## 5. The decision

**Submit, but do not invest further.**

- Submitting costs only gas. It is a free option: if the claims ahead fail the
  real benchmark — and the bounty page itself records a prior detector scoring
  under 60% — we are in the queue with numbers that are independently checkable
  and the strongest false-positive evidence in the field.
- Do not spend further days optimising. With sixteen claims ahead and
  earliest-valid-wins, realistic odds are **under 5%**, and no amount of model
  swapping changes the queue position.
- If you want the model anyway: it is MIT and legitimately reusable, and it
  would make this a genuinely good tool regardless of the bounty outcome.

## 6. Honest risk statement

Even with a better model, this is now a race we entered late against sixteen
competitors, several of whom have done serious work. The realistic case for
winning rests on other claims failing the real benchmark or failing the
false-positive review — which the bounty's own text suggests is plausible, but
which is not something we control.

What *is* controllable: submitting something that is genuinely reproducible,
genuinely offline, genuinely honest about its weaknesses, and genuinely good at
not calling real photographs fake. That is the strongest hand this project can
play.
