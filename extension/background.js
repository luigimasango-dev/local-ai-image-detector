// MV3 service worker — routing only. NO inference here.
//
// MV3 constraints driving this design:
//   1. Service workers have NO DOM. onnxruntime-web's WebGL backend and our
//      own capability probe both need `document.createElement('canvas')`.
//   2. Service workers are terminated after ~30s idle. Model load is the
//      expensive step (hundreds of MB decompressed into GPU/WASM memory); a
//      worker that dies every 30s would reload the model per image.
//
// So inference runs in a persistent chrome.offscreen document that has a DOM,
// WebGPU/WebGL access, and is shared across ALL tabs — weights load once
// regardless of tab count. This worker's only job is to keep that document
// alive and relay messages between content scripts and the offscreen doc.

const OFFSCREEN_DOCUMENT_PATH = "offscreen.html";
const OFFSCREEN_DOCUMENT_REASON = "WORKERS"; // "DOM_PARSER" also valid; WORKERS fits inference

// in-flight requests keyed by request id so concurrent per-tab requests get
// their correct replies. The offscreen document replies include the same id.
const pendingRequests = new Map();

// Ceiling on how long a single image may take. Generous because the WASM
// fallback (no WebGPU) is ~20x slower than the GPU path — but finite, because
// an unanswered request leaves the content script waiting forever.
const REQUEST_TIMEOUT_MS = 45000;

// A closed tab's requests can never be delivered; drop them so the map does
// not grow for the life of the service worker.
chrome.tabs.onRemoved.addListener((tabId) => {
  for (const [id, p] of pendingRequests) {
    if (p.tabId === tabId) {
      clearTimeout(p.timer);
      pendingRequests.delete(id);
    }
  }
});

// ---------------------------------------------------------------------------
// Offscreen document lifecycle
// ---------------------------------------------------------------------------

async function hasOffscreenDocument() {
  // chrome.offscreen.hasDocument is the clean check; fall back to scanning
  // runtime contexts for the path (works in all supported Chrome versions).
  if (typeof chrome.offscreen?.hasDocument === "function") {
    const exists = await chrome.offscreen.hasDocument();
    return exists;
  }
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  return contexts.some((ctx) => ctx.documentUrl?.includes(OFFSCREEN_DOCUMENT_PATH));
}

async function ensureOffscreenDocument() {
  if (await hasOffscreenDocument()) return;

  // Race guard: two tabs can fire their first score request at the same
  // instant, both see "no document", both call createDocument. The second
  // call rejects with "Only a single offscreen document may be created" —
  // catch that and treat it as "the document now exists".
  try {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT_PATH,
      reasons: [OFFSCREEN_DOCUMENT_REASON],
      justification: "Hosts local in-browser image inference (WebGPU/WebGL/WASM) with model weights cached across the whole extension.",
    });
  } catch (err) {
    const msg = String(err && err.message ? err.message : err);
    if (!/Only a single offscreen document|already exists/i.test(msg)) {
      throw err;
    }
  }
}

// ---------------------------------------------------------------------------
// Message relay
// ---------------------------------------------------------------------------

// content script -> service worker -> offscreen document -> back to content.
// popup -> service worker -> offscreen document (status queries).
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message) return false;
  const isContent = message.source === "content";
  const isPopup = message.source === "popup";
  if (!isContent && !isPopup) return false;

  (async () => {
    try {
      await ensureOffscreenDocument();

      if (isContent && message.type === "analyze") {
        const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;

        // Every pending request MUST eventually be answered. Without this the
        // content script awaits forever, the image is already marked seen so it
        // is never retried, and pendingRequests grows without bound — a page of
        // images with no badges and no errors, which is indistinguishable from
        // "still working".
        const timer = setTimeout(() => {
          const p = pendingRequests.get(requestId);
          if (!p) return;
          pendingRequests.delete(requestId);
          p.sendResponse({ error: "timed out waiting for inference" });
        }, REQUEST_TIMEOUT_MS);

        pendingRequests.set(requestId, {
          sendResponse,
          tabId: sender.tab?.id,
          timer,
        });

        // Ask the offscreen document to score this image. The promise rejects
        // if the offscreen document is gone or throws before replying, so it
        // must be caught — otherwise the failure is swallowed and the request
        // hangs until the timeout.
        chrome.runtime
          .sendMessage({
            source: "background",
            type: "score",
            requestId,
            src: message.src,
            dataUrl: message.dataUrl,
            naturalWidth: message.naturalWidth,
            naturalHeight: message.naturalHeight,
          })
          .catch((err) => {
            const p = pendingRequests.get(requestId);
            if (!p) return;
            clearTimeout(p.timer);
            pendingRequests.delete(requestId);
            p.sendResponse({
              error: String(err && err.message ? err.message : err),
            });
          });

        // reply is delivered via the relay listener below.
        return;
      }

      if (message.type === "get-status") {
        const resp = await chrome.runtime.sendMessage({
          source: "background",
          type: "get-status",
        });
        sendResponse(resp);
        return;
      }

      if (message.type === "ping") {
        sendResponse({ ok: true });
        return;
      }

      sendResponse({ error: `Unknown message type: ${message.type}` });
    } catch (err) {
      sendResponse({
        error: String(err && err.message ? err.message : err),
      });
    }
  })();

  return true; // keep the message channel open for the async reply
});

// offscreen document -> service worker -> content script.
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.source !== "offscreen") return false;

  if (message.type === "score-result" && message.requestId) {
    const pending = pendingRequests.get(message.requestId);
    if (pending) {
      clearTimeout(pending.timer);
      pendingRequests.delete(message.requestId);
      pending.sendResponse(
        message.error
          ? { error: message.error }
          : { ok: true, score: message.score, modelIds: message.modelIds }
      );
    }
    sendResponse({ ok: true });
    return false;
  }

  if (message.type === "status") {
    // forwarded model-load status for the popup
    sendResponse({ ok: true });
    return false;
  }

  return false;
});

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Keep the offscreen document alive for the lifetime of the extension session.
// MV3 service workers can die after ~30s idle; if this worker restarts, the
// offscreen document may have been closed by the browser — re-create on demand.
chrome.runtime.onStartup.addListener(() => {
  ensureOffscreenDocument().catch((err) =>
    console.error("[background] startup offscreen create failed:", err)
  );
});

chrome.runtime.onInstalled.addListener(() => {
  ensureOffscreenDocument().catch((err) =>
    console.error("[background] install offscreen create failed:", err)
  );
});

chrome.runtime.onSuspend.addListener(() => {
  console.log("[background] service worker suspending");
});
