# Model licensing — what we can actually ship

Verified 2026-08-13 by reading each model's own page. This drives model
selection as hard as accuracy does: a detector we cannot legally ship is worth
zero regardless of how well it scores.

## What the bounty actually requires

Two separate things, and conflating them leads to the wrong conclusion:

1. **Our submission** must be MIT-licensed. That is *our code*.
2. **The weights** need not be redistributed by us. The bounty explicitly
   allows: *"The extension may perform a one-time download of publicly
   available model weights during initial setup."*

So the question is not only "can we redistribute these weights" but also "does
this licence restrict the *use* and *modification* we need" — because ONNX
conversion and quantization are modifications.

## Verdicts

| Model | Licence | Verdict |
|---|---|---|
| **JeongsooP/Community-Forensics** | **MIT** | ✅ **Clean.** Use freely, modify, quantize, redistribute. |
| **guyfloki/ai-image-detector** (CvT-13) | **Apache-2.0** | ✅ Clean licence, ⚠️ hosting risk (see below). |
| **Organika/sdxl-detector** | **CC-BY-NC-3.0** | ❌ **NonCommercial.** |
| **umm-maybe/AI-image-detector** | **CC-BY-ND-4.0** | ❌ **NoDerivatives.** |

### Why the two rejected ones are genuinely rejected, not just awkward

- **umm-maybe — CC-BY-ND (NoDerivatives).** ND forbids distributing modified
  versions. Converting to ONNX and quantizing to run in a browser is exactly a
  modified version. This is the one technique the whole project depends on, so
  ND is disqualifying rather than inconvenient.
- **Organika — CC-BY-NC (NonCommercial).** NC restricts commercial use, and
  this is a competition for a ~$2,555 prize. Whether that is "commercial" is
  genuinely arguable — but the bounty rules let maintainers *"disqualify any
  submission that technically satisfies the letter of these rules while clearly
  violating their spirit."* Betting the whole build on a licence argument we
  might lose at judging time is a bad trade when a cleanly-MIT alternative
  exists.
- Note also that **Organika is fine-tuned from umm-maybe**, so it inherits the
  ND problem upstream regardless of its own stated licence. Two of our four
  candidates were never really independent.

### guyfloki — clean licence, brittle hosting

Apache-2.0 is fine. The problem is the weights (~226 MB) are hosted on **Google
Drive**, not a package registry. Google Drive is hostile to programmatic
download (interstitial confirm pages, quota limits, no stable direct URL), and
the bounty evaluators build from source in a clean environment. A one-time
download that fails on their machine fails the submission. Usable only if we
can mirror the weights somewhere stable — which is permitted by Apache-2.0
(attribution required), unlike the other two.

## Consequence for model selection

**Community-Forensics is the primary candidate**, on both licence and merit:
MIT, weights on Hugging Face (`OwensLab/commfor-model-384`,
`OwensLab/commfor-model-224`), ViT with configurable Small/Tiny sizes, and it
leads the published zero-shot cross-generator benchmark (it aggregates training
across hundreds of generators, which is precisely the generalization property
this bounty tests).

The 224-input Small/Tiny variants matter for us specifically: they are the ones
plausibly small and fast enough to run per-image in a browser. Confirm actual
file size before committing.

The ensemble plan does **not** change, but the pool it draws from does: build
around Community-Forensics, and look for a second architecturally-different
permissively-licensed detector rather than assuming the two HF models were
available.

## Still to verify

- Exact checkpoint file sizes for `commfor-model-224` (Small and Tiny).
- Whether Community-Forensics exports cleanly to ONNX (plain ViT should, but
  confirm — custom ops or dynamic control flow would block the browser path).
- A second permissively-licensed detector with a *different* architecture, so
  the ensemble has genuine diversity rather than two ViTs making correlated
  mistakes.
