// Popup — status UI. Persists an enable/disable toggle in chrome.storage.local
// (default enabled) and reads execution-provider + model-load status from the
// offscreen document via the service worker.

const els = {
  enabled: document.getElementById("enabled"),
  provider: document.getElementById("provider"),
  modelStatus: document.getElementById("model-status"),
  progress: document.getElementById("progress"),
  status: document.getElementById("status"),
};

const DEFAULT_SETTINGS = { enabled: true };

function setStatus(text) {
  els.status.textContent = text;
}

// ---------------------------------------------------------------------------
// Toggle
// ---------------------------------------------------------------------------

async function loadEnabled() {
  const { enabled } = await chrome.storage.local.get(DEFAULT_SETTINGS);
  els.enabled.checked = enabled;
}
loadEnabled().catch((e) => setStatus("storage read failed: " + e));

els.enabled.addEventListener("change", async () => {
  await chrome.storage.local.set({ enabled: els.enabled.checked });
  setStatus(els.enabled.checked ? "Detection on" : "Detection off");
});

// Refresh the flag if it changes in another surface (options page, another
// popup instance).
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.enabled) {
    els.enabled.checked = changes.enabled.newValue;
  }
});

// ---------------------------------------------------------------------------
// Status polling
// ---------------------------------------------------------------------------

async function refreshStatus() {
  try {
    const resp = await chrome.runtime.sendMessage({
      source: "popup",
      type: "get-status",
    });
    if (resp && resp.capabilities) {
      const c = resp.capabilities;
      els.provider.textContent = describeProvider(c);
      els.modelStatus.textContent =
        resp.loadedModels && resp.loadedModels.length
          ? resp.loadedModels.join(", ")
          : "not loaded";
    } else if (resp && resp.error) {
      setStatus("status error: " + resp.error);
    }
  } catch (e) {
    setStatus("status error: " + e);
  }
}

function describeProvider(c) {
  if (c.hasWebGPU) return "WebGPU";
  if (c.hasWebGL) return "WebGL";
  return "WASM";
}

refreshStatus();

// Runtime messages from the offscreen document (sent via chrome.runtime),
// which broadcast to every extension context including this popup.
chrome.runtime.onMessage.addListener((msg) => {
  if (!msg) return false;
  if (msg.type === "model-progress" && typeof msg.progress === "number") {
    els.modelStatus.textContent = msg.modelId;
    els.progress.textContent = Math.round(msg.progress * 100) + "%";
  } else if (msg.type === "model-loaded") {
    els.modelStatus.textContent = msg.modelId + " (loaded)";
  }
  return false;
});