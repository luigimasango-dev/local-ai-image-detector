// Capability detection for ONNX Runtime Web (WebGPU -> WebGL -> WASM).
// Adapted from reference/ml-capabilities.ts, converted TypeScript -> plain JS.
// Differences from the source:
//   - isMobileDevice()/desktop gate removed (irrelevant for a desktop Chrome
//     extension, and only shrinks the test matrix).
//   - ML_THRESHOLDS import inlined below as a constant (no TS config here).
//   - Probe order and thresholds kept identical to the source so the fitted
//     model expectations stay valid.
//
// IMPORTANT: this file must only ever run inside the OFFSCREEN document.
// checkWebGLSupport() needs document.createElement('canvas') — the MV3 service
// worker has no DOM, so probing from there silently returns false for WebGL.
// capabilities.js is loaded by offscreen.html before offscreen.js.

const ML_THRESHOLDS = {
  // Memory floor at which we consider inference viable (MB). Inlined from
  // ml-config.ts; re-tuned for the chosen ensemble sizes in phase 3.
  memoryBudgetMB: 1024,
  minSupportedMemoryMB: 256,
};

const memoryBudgetMB = ML_THRESHOLDS.memoryBudgetMB;

function checkWebGLSupport() {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    return !!gl;
  } catch {
    return false;
  }
}

async function checkWebGPUSupport() {
  try {
    if (!("gpu" in navigator)) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null && adapter !== undefined;
  } catch {
    return false;
  }
}

// Validates a tiny SIMD wasm module; confirms the runtime has WASM SIMD
// instructions available (fast path for onnxruntime-web's wasm backend).
function checkSIMDSupport() {
  try {
    return (
      typeof WebAssembly.validate === "function" &&
      WebAssembly.validate(
        new Uint8Array([
          0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
          3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
        ])
      )
    );
  } catch {
    return false;
  }
}

function checkThreadsSupport() {
  return typeof SharedArrayBuffer !== "undefined";
}

// Rough head-room estimate so we can warn before loading a large model.
function estimateAvailableMemory() {
  const deviceMemory = navigator.deviceMemory; // GiB on Chromium
  if (deviceMemory) {
    return Math.min(deviceMemory * 256, memoryBudgetMB);
  }
  return 256; // conservative default when deviceMemory is unavailable
}

let cachedCapabilities = null;

function detectMLCapabilities() {
  if (cachedCapabilities) return cachedCapabilities;

  const hasWebGL = checkWebGLSupport();
  const hasWebGPU = false; // resolved below to keep detect() async-free for offscreen startup

  let recommendedExecutionProvider = "wasm";
  let hasWebGpu = false;

  // Chromium exposes navigator.gpu synchronously-only via requestAdapter, which
  // is async. For the synchronous offscreen.startup path we do a synchronous
  // feature-detect of the API surface and treat the async adapter probe as a
  // refinement in refreshWebGPU (below).
  if ("gpu" in navigator) hasWebGpu = true;
  if (hasWebGL) recommendedExecutionProvider = "webgl";
  if (hasWebGpu) recommendedExecutionProvider = "webgpu";

  const hasSIMD = checkSIMDSupport();
  const hasThreads = checkThreadsSupport();
  const estimatedMemoryMB = estimateAvailableMemory();

  cachedCapabilities = {
    isSupported: (hasWebGL || hasWebGpu) && estimatedMemoryMB >= ML_THRESHOLDS.minSupportedMemoryMB,
    hasWebGL,
    hasWebGPU: hasWebGpu,
    hasSIMD,
    hasThreads,
    estimatedMemoryMB,
    recommendedExecutionProvider,
    recommendedThreads: hasThreads ? Math.min(navigator.hardwareConcurrency || 4, 4) : 1,
  };

  return cachedCapabilities;
}

// Async refinement of the WebGPU flag: sync probe above only checks the API
// surface; this actually asks for an adapter. Call once at offscreen startup
// and re-roll the recommended provider if WebGPU becomes viable.
async function refreshWebGPU() {
  const hasWebGPU = await checkWebGPUSupport();
  if (hasWebGPU !== cachedCapabilities.hasWebGPU) {
    cachedCapabilities = {
      ...cachedCapabilities,
      hasWebGPU,
      recommendedExecutionProvider: hasWebGPU
        ? "webgpu"
        : cachedCapabilities.hasWebGL
          ? "webgl"
          : "wasm",
    };
  }
  return cachedCapabilities;
}

// Fire-and-forget: the popup reads the (possibly stale) sync result first,
// then this async refinement lands a WebGPU-capable update shortly after.
refreshWebGPU();