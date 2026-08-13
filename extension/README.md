# Local AI-Image Detector — Chrome Extension (MV3)

Detects AI-generated images using **only in-browser inference**. No cloud, no
external APIs, no local server. After a one-time model download the extension
works fully offline.

Target: poidh.xyz bounty #323. See `../ARCHITECTURE.md` for the full design.

> **Status: functional.** Real inference runs here: two Community-Forensics
> ViT models (MIT), fp16 ONNX, fused and Platt-calibrated so the emitted score
> is a probability and 0.65 is a meaningful decision boundary. Verified in
> Chrome against the offline Python harness — 6/6 decisions agree, ~190 ms per
> image on WebGPU after a ~3.5 s model load.
>
> **Run `python build.py` in the repo root before loading.** Model weights are
> regenerated from the published MIT checkpoints rather than committed, so a
> fresh clone has none and the extension will not run until you build.

---

## Loading the extension (unpacked)

1. Open `chrome://extensions`.
2. Enable **Developer mode** (top-right toggle).
3. Click **Load unpacked** and select the `extension/` directory in this repo.
4. The extension icon appears in the toolbar. The popup shows which execution
   provider resolved (`webgpu` or `wasm`) and whether the models are loaded.
5. Browse any page with photographs. The first image is slower while the models
   load; after that it is roughly 190 ms per image on WebGPU.

## Message-flow architecture

```
content script (per tab)        service worker (background.js)     offscreen document (offscreen.html+js)
──────────────────────          ────────────────────────────       ────────────────────────────────────
scan <img> elements
 ├ filter: skip <128px,
 │   dedupe by src, only
 │   images in/near viewport
 └ send {src, dataUrl} ─────►   ensure offscreen doc exists   ───► scoreImage()
                                route message (request-id map)    ├ commfor-224 (fp16 ONNX)
                                relay result back                  ├ commfor-384 (fp16 ONNX)
                             ◄─ {requestId, score}                 └ fuse → calibrate → p(AI)
 ◄── score relayed ─────────
render badge w/ confidence
```

Why inference lives in an **offscreen document** and not the service worker:

- MV3 service workers have **no DOM**. `onnxruntime-web`'s WebGL backend and our
  capability probe both need `document.createElement('canvas')`.
- Service workers are **killed after ~30s idle**. Model load is the expensive
  step (hundreds of MB); a dying worker would reload per image.

An offscreen document persists while the extension is active, has full
WebGPU/WebGL access, and is **shared across all tabs** — weights load once
regardless of tab count. See `background.js` header comments for the
lifecycle-race handling (`chrome.offscreen.createDocument` errors if two tabs
race to create it; the second call is caught and treated as success).

## Permissions (audited, minimal)

See `manifest.json`. Settled on:

- **`offscreen`** — required to create the offscreen inference document
  (`chrome.offscreen.createDocument`).
- **`storage`** — popup enable/disable toggle + (later) load-progress state via
  `chrome.storage.local`.
- **`<all_urls>`** host permissions — required for the content script to run on
  any page (`content_scripts.matches`) and to render badges on arbitrary sites.

Deliberately **not** requested: `tabs`, `scripting`, `downloads`, `webRequest`,
`activeTab`, `alarms`. Nothing uses them. No background `service_worker` keeps a
network socket; the worker only routes messages and re-creates the offscreen
document on demand.

## Files

| File | Role |
|---|---|
| `manifest.json` | MV3 manifest. `offscreen` + `storage` permissions, `<all_urls>` host perms, `content_scripts` at `document_idle` with `styles.css`. |
| `background.js` | Service worker. **Routing only, no inference.** Ensures the offscreen document exists (guarding the create race), relays content↔offscreen messages, keeps a request-id map so concurrent per-tab requests get their replies. |
| `offscreen.html` | The inference host page. Loads `capabilities.js` then `offscreen.js`. |
| `offscreen.js` | Inference host. Creates and caches the ONNX sessions, serializes inference so one slow image cannot block the message loop, and handles messages. |
| `inference.js` | Preprocessing (matching the Python harness exactly), ensemble fusion, and the Platt calibration constants. |
| `capabilities.js` | WebGPU → WASM execution-provider detection, adapted from `reference/ml-capabilities.ts` (TS→JS, mobile/desktop gate dropped, `ML_THRESHOLDS` inlined). Must run in the offscreen document (needs a DOM). |
| `content.js` | Finds images in every frame and open shadow root. Skips <128px, dedupes per element, waits for lazy-loaded images to actually load, caps in-flight requests, and renders one badge per image in a wrapper it owns. |
| `popup.html` / `popup.js` | Status UI: detected execution provider, model load status/progress, enable/disable toggle persisted via `chrome.storage.local`. |
| `styles.css` | Badge styling, dark pill + light border, readable over light and dark images, `pointer-events: none`. |

## Known limitations

- **Cross-origin images cannot be scored offline.** Drawing an image from
  another domain onto a canvas taints it, so its pixels are unreadable without
  re-fetching over the network. Since the extension must work with the network
  disabled, those images are marked skipped (a grey "—" badge) rather than
  silently ignored or quietly re-downloaded.
- **CSS `background-image`, `<canvas>` and `<video>` posters are not analyzed.**
  Only real `<img>` elements are, plus images inside iframes and open shadow
  roots.
- **Images below 128x128 are skipped** as icons/spacers. This also skips some
  small avatars and thumbnails.
- **Heavily re-compressed images are the accuracy weak point** — roughly 0.73-0.75
  balanced accuracy versus 0.83 on pristine images. See `../RESULTS.md`.
- **Without WebGPU** the WASM fallback is roughly 20x slower. Inference runs in
  a worker and is serialized so the page stays responsive, but a gallery page
  will take noticeably longer to finish.
