#!/usr/bin/env python3
"""Score the eval set using the exported ONNX graphs, not PyTorch.

Why this exists
---------------
Every accuracy number we have was measured on PyTorch FP32 weights. The
extension ships quantized ONNX. Those are not the same model: int8 shifted
logits by up to 0.78 on a probe batch and flipped a decision for the 384
variant. Since we sit within noise of the 0.75 bar, an unmeasured quantization
loss could decide pass/fail on its own.

This script is also the reference implementation for the browser: it does the
preprocessing in the exact order the extension must, so any drift between the
two shows up here first.

Preprocessing (must match run_inference_commfor.py, which read it from the
Community-Forensics source):
    Resize shorter side to resize_size (256 for 224-input, 440 for 384-input)
    -> CenterCrop(input_size)
    -> scale to [0,1]
    -> Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    -> NCHW float32

Usage:
    python run_inference_onnx.py --model ../models/commfor-model-224.int8.onnx \
        --input-size 224 [--labels ...] [--data-dir ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("onnx_infer")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PRED_DIR = Path(__file__).resolve().parent / "predictions"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Community-Forensics determine_resize_crop_sizes(): 224 -> 256, 384 -> 440
RESIZE_FOR = {224: 256, 384: 440}


def preprocess(img, input_size: int):
    """PIL image -> (1,3,H,W) float32 numpy, matching the training transform."""
    import numpy as np
    from PIL import Image

    resize_size = RESIZE_FOR.get(input_size, int(input_size * 256 / 224))
    img = img.convert("RGB")

    # Resize shorter side to resize_size, preserving aspect (torchvision Resize(int))
    w, h = img.size
    if w < h:
        new_w, new_h = resize_size, max(1, round(h * resize_size / w))
    else:
        new_h, new_w = resize_size, max(1, round(w * resize_size / h))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop(input_size)
    left = (new_w - input_size) // 2
    top = (new_h - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC in [0,1]
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)[None, ...]                   # NCHW
    return np.ascontiguousarray(arr, dtype=np.float32)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Path to .onnx")
    p.add_argument("--input-size", type=int, required=True, choices=[224, 384])
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_PRED_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default=None, help="Override output filename stem")
    args = p.parse_args()

    try:
        import numpy as np  # noqa: F401
        import onnxruntime as ort
        from PIL import Image
    except ImportError as exc:
        print(f"ERROR: {exc}. pip install onnxruntime pillow numpy", file=sys.stderr)
        return 1

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    log.info("Loaded %s (input '%s')", args.model.name, in_name)

    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    out: dict[str, float] = {}
    skipped = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        path = args.data_dir / r["filepath"]
        try:
            with Image.open(path) as im:
                x = preprocess(im, args.input_size)
        except Exception as exc:  # noqa: BLE001
            skipped.append((r["image_id"], str(exc)))
            continue
        logit = float(sess.run(None, {in_name: x})[0].reshape(-1)[0])
        out[r["image_id"]] = sigmoid(logit)
        if i % 200 == 0:
            log.info("  %d/%d", i, len(rows))

    elapsed = time.time() - t0
    tag = args.tag or args.model.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / f"{tag}.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")

    log.info("Done in %.1fs (%.3fs/image): %d predictions, %d skipped",
             elapsed, elapsed / max(len(out), 1), len(out), len(skipped))

    # Same polarity guard the PyTorch path uses -- a silently inverted graph
    # would otherwise look like a merely-mediocre model.
    truth = {r["image_id"]: int(r["label"]) for r in rows}
    pos = [v for k, v in out.items() if truth.get(k) == 1]
    neg = [v for k, v in out.items() if truth.get(k) == 0]
    if pos and neg:
        mp, mn = sum(pos) / len(pos), sum(neg) / len(neg)
        log.info("POLARITY: mean P(AI) label=1 %.3f | label=0 %.3f", mp, mn)
        if mp <= mn:
            log.error("POLARITY INVERTED — do not trust these predictions.")
    log.info("WROTE %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
