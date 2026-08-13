// Real in-browser inference for the Community-Forensics ensemble.
//
// Everything here must mirror eval_harness/run_inference_onnx.py exactly. The
// measured accuracy (0.8338 balanced accuracy on our benchmark) only transfers
// to the extension if preprocessing, fusion and calibration are identical. A
// silent mismatch — wrong resize filter, wrong normalization, un-calibrated
// output — does not throw. It just quietly scores worse, which is the most
// expensive kind of bug here.
//
// Pipeline per image:
//   resize shorter side -> center crop -> [0,1] -> ImageNet normalize -> NCHW
//   -> per-model sigmoid(logit) = P(AI)
//   -> equal-weight average of the two members
//   -> Platt calibration so 0.65 is the real decision boundary

/* global ort */

// ---------------------------------------------------------------------------
// Constants — read from the Community-Forensics source, not guessed.
// ---------------------------------------------------------------------------

// ImageNet normalization, used because the upstream training transform uses it.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// determine_resize_crop_sizes(): 224 -> 256, 384 -> 440
const RESIZE_FOR = { 224: 256, 384: 440 };

// Platt calibration fitted on the TUNE split only (eval_harness/calibrate.py).
// p_calibrated = sigmoid(A * logit(p_raw) + B)
//
// This is what moves the operating point onto the graded 0.65 threshold. The
// raw fused score peaks around 0.055, so shipping uncalibrated output would
// score ~0.71 instead of ~0.83 — i.e. it would fail. Do not "simplify" this
// away.
// Fitted on the FP16 ONNX predictions — the exact graphs this extension runs.
// (The PyTorch FP32 fit was A=0.6712, B=2.0854. The difference is small, but
// the calibration should come from the same precision that ships, not from a
// model we only used for measurement.)
const CALIBRATION = { A: 0.6758, B: 2.0932 };

// Ensemble members. Equal weights: the pair was measured at r=0.75 correlation,
// and equal weighting beat the alternatives on the tune split.
//
// fp16, NOT int8. int8 dynamic quantization was measured on the full holdout
// and cost 0.043 AUC (0.8854 -> 0.8421) — far too much when the pass bar is
// only a few points away. fp16 halves the size like int8 promised to, but its
// worst-case logit drift against PyTorch is 3.6e-3 rather than int8's 0.56.
// WebGPU runs fp16 natively, so it is also the fastest option in the browser
// (it is only slow on x86 CPUs, which have no native fp16 compute).
const ENSEMBLE = [
  { id: "cf224", file: "models/commfor-model-224.fp16.onnx", inputSize: 224, weight: 0.5 },
  { id: "cf384", file: "models/commfor-model-384.fp16.onnx", inputSize: 384, weight: 0.5 },
];

const EPS = 1e-6;

function sigmoid(z) {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

function logit(p) {
  const q = Math.min(Math.max(p, EPS), 1 - EPS);
  return Math.log(q / (1 - q));
}

export function calibrate(pRaw) {
  return sigmoid(CALIBRATION.A * logit(pRaw) + CALIBRATION.B);
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/**
 * ImageBitmap -> Float32Array in NCHW, matching torchvision's
 * Resize(int) + CenterCrop(int) + ToTensor + Normalize.
 *
 * Resize(int) in torchvision scales the SHORTER side to the target and keeps
 * aspect ratio — not a square stretch. Getting this wrong distorts every image
 * and costs accuracy without any visible error.
 */
export function preprocess(bitmap, inputSize) {
  const resizeSize = RESIZE_FOR[inputSize] ?? Math.round((inputSize * 256) / 224);
  const { width: w, height: h } = bitmap;

  let newW;
  let newH;
  if (w < h) {
    newW = resizeSize;
    newH = Math.max(1, Math.round((h * resizeSize) / w));
  } else {
    newH = resizeSize;
    newW = Math.max(1, Math.round((w * resizeSize) / h));
  }

  // Draw resized, then read back only the center crop.
  const canvas = new OffscreenCanvas(newW, newH);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  // Canvas smoothing is the closest available analogue to PIL's BILINEAR.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, newW, newH);

  const left = Math.floor((newW - inputSize) / 2);
  const top = Math.floor((newH - inputSize) / 2);
  const { data } = ctx.getImageData(left, top, inputSize, inputSize);

  const plane = inputSize * inputSize;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i += 1) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    out[i] = (r - MEAN[0]) / STD[0];
    out[plane + i] = (g - MEAN[1]) / STD[1];
    out[2 * plane + i] = (b - MEAN[2]) / STD[2];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/**
 * Pick execution providers best-first. onnxruntime-web falls through the list,
 * so a machine without WebGPU still runs on WASM rather than failing.
 */
function providersFor(capabilities) {
  const eps = [];
  if (capabilities?.hasWebGPU) eps.push("webgpu");
  eps.push("wasm");
  return eps;
}

export async function createSession(modelUrl, capabilities) {
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: providersFor(capabilities),
    graphOptimizationLevel: "all",
  });
}

/** Single model forward pass -> P(AI). */
export async function scoreWithSession(session, bitmap, inputSize) {
  const input = preprocess(bitmap, inputSize);
  const tensor = new ort.Tensor("float32", input, [1, 3, inputSize, inputSize]);
  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  const logitValue = Number(results[session.outputNames[0]].data[0]);
  // The Community-Forensics head emits ONE logit trained with BCEWithLogitsLoss
  // and real:0 / fake:1, so sigmoid(logit) is P(AI) directly — no class-index
  // juggling, and no place for an off-by-one to invert the meaning.
  return sigmoid(logitValue);
}

/**
 * Full ensemble score for one image.
 * Returns the calibrated P(AI) plus the per-model raw scores for debugging.
 */
export async function scoreEnsemble(sessions, bitmap) {
  const perModel = {};
  let weighted = 0;
  let totalWeight = 0;

  for (const member of ENSEMBLE) {
    const session = sessions.get(member.id);
    if (!session) continue;
    const p = await scoreWithSession(session, bitmap, member.inputSize);
    perModel[member.id] = p;
    weighted += p * member.weight;
    totalWeight += member.weight;
  }

  if (totalWeight === 0) throw new Error("no ensemble members loaded");

  const fusedRaw = weighted / totalWeight;
  return { pAi: calibrate(fusedRaw), fusedRaw, perModel };
}

export { ENSEMBLE, CALIBRATION };
