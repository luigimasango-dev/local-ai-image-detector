# Architecture — local AI-image-detector Chrome extension

Target: poidh.xyz bounty #323. MV3 extension, ≥75% balanced accuracy at a 0.65 confidence
threshold, all inference in-browser, no cloud, no local server, offline after first-run
model download.

## Where inference runs: offscreen document, not the service worker

MV3 service workers have **no DOM** and are **terminated after ~30s idle**. Both are
disqualifying here:

- `onnxruntime-web`'s WebGL backend and the capability probe both need `document.createElement('canvas')`
  (see `C:\Dev\worldmonitor\src\services\ml-capabilities.ts:69`). No `document` in a SW.
- Model load is the expensive step (hundreds of MB decompressed into GPU/WASM memory). A worker
  that dies every 30s would reload per image — unusable.

**Decision: run inference in an offscreen document** (`chrome.offscreen`, reason `DOM_PARSER` /
`WORKERS`). It has a full DOM, WebGPU/WebGL access, and persists while the extension is active.
One instance is shared across all tabs, so the model loads once regardless of tab count — this
matters on a 16GB machine with many tabs open.

Rejected alternative: inference inside each content script. Every tab would hold its own copy of
the model weights. Memory blows up linearly with tab count.

## Message flow

```
content script (per tab)          service worker            offscreen document
─────────────────────────         ───────────────           ──────────────────
scan <img> elements
 ├ filter: skip < 128px,          route message      ──►    ensemble inference
 │   skip data: URIs already                                 ├ model A (ONNX)
 │   seen, dedupe by src                                     ├ model B (ONNX)
 │                                                           └ fuse → p(AI)
 └ post {imgId, src/bitmap}  ──►                     ◄──    {imgId, score}
                             ◄──  relay result
 render badge w/ confidence
```

The content script sends image data (as an `ImageBitmap` via transferable, or the resolved URL
for the offscreen doc to fetch from cache) — never off-device.

## Inference layer

Adapt the pipeline-management pattern from `C:\Dev\worldmonitor\src\workers\ml.worker.ts:132-181`:

- `loadedPipelines: Map<modelId, pipeline>` — models stay resident after first load
- `loadingPromises: Map` guard so concurrent requests don't trigger duplicate loads
- `progress_callback` → post load progress to the popup for the one-time setup UI
- Execution-provider fallback chain `webgpu → webgl → wasm`, reusing the probe logic in
  `ml-capabilities.ts:23-64` (drop the `isMobileDevice`/desktop gate — irrelevant for a Chrome
  desktop extension, and the memory-budget floor should be re-tuned for our model sizes)

## Offline-after-setup (a hard bounty requirement)

The evaluators disable internet after the initial model download and block localhost APIs. So:

- Weights are fetched **once** on install, then persisted (Cache Storage / IndexedDB via
  transformers.js `env.useBrowserCache`, with `env.allowRemoteModels = false` after setup).
- Set `env.localModelPath` to the extension's bundled path, or ship weights in the package
  outright if size permits — simplest way to prove the offline property.
- No `fetch()` to any remote host in the inference path. QA (phase 5) verifies this empirically
  with the network tab, not by reading the code.

## Scoring contract

The fusion layer must produce a single `p(AI-generated) ∈ [0,1]` per image so the extension's
threshold behaviour matches `eval_harness/score.py` exactly — same weights, same combination
rule. If the extension and the harness disagree on how scores fuse, the internal benchmark
number is meaningless. This is the single most important correctness constraint in the build.

## Open decisions (resolved in phase 3)

- Which 2–3 models make the ensemble — pending `eval_harness/model_survey.md` (Hermes) and the
  per-model accuracy sweep.
- Fusion rule: weighted average vs. small logistic-regression combiner.
- Final confidence threshold (bounty mandates evaluation at 0.65; our fusion should be
  calibrated so 0.65 is the right operating point, not just tested there).
