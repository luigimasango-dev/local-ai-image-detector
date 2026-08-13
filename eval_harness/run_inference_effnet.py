#!/usr/bin/env python3
"""Score the eval set with jacoballessio/ai-image-detect-distilled-efficientnet.

This model needs its own loader: its config declares model_type
"custom_efficientnet", which `transformers` does not recognise, and its
published from_pretrained is broken (it string-concatenates a local path, so it
cannot load from the Hub at all). The architecture is plain
torchvision efficientnet_b0 with num_classes=2, so we rebuild it directly and
load the state dict.

POLARITY WARNING: the config carries no id2label, so nothing states which of
the two output classes means "AI-generated". Rather than guess, this script
tries both mappings and reports which one separates the known labels, refusing
to emit predictions if neither does. Guessing here would silently invert every
score and look like a mediocre model rather than a bug.

Usage:
    python run_inference_effnet.py [--labels ...] [--data-dir ...] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("effnet")

REPO = "jacoballessio/ai-image-detect-distilled-efficientnet"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PRED_DIR = Path(__file__).resolve().parent / "predictions"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224  # preprocessor_config.json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_PRED_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    try:
        import torch
        import numpy as np
        from torchvision import models, transforms
        from huggingface_hub import hf_hub_download
        from PIL import Image
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    weights_path = hf_hub_download(REPO, "pytorch_model.bin")
    model = models.efficientnet_b0(num_classes=2)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Published checkpoint nests the torchvision module under "efficientnet.".
    state = { (k[len("efficientnet."):] if k.startswith("efficientnet.") else k): v
              for k, v in state.items() }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning("state_dict mismatch — missing=%d unexpected=%d",
                    len(missing), len(unexpected))
        if len(missing) > 10:
            log.error("Too many missing keys; the architecture does not match.")
            return 1
    model.eval()
    log.info("Loaded EfficientNet-B0 (%.1fM params)",
             sum(q.numel() for q in model.parameters()) / 1e6)

    tf = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    probs_class1: dict[str, float] = {}
    skipped = 0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            tensors, ids = [], []
            for r in chunk:
                try:
                    with Image.open(args.data_dir / r["filepath"]) as im:
                        tensors.append(tf(im.convert("RGB")))
                    ids.append(r["image_id"])
                except Exception:  # noqa: BLE001
                    skipped += 1
            if not tensors:
                continue
            logits = model(torch.stack(tensors))
            sm = torch.softmax(logits, dim=1)[:, 1]  # P(class index 1)
            for image_id, v in zip(ids, sm.tolist()):
                probs_class1[image_id] = float(v)
            if (start // args.batch_size) % 10 == 0:
                log.info("  %d/%d", min(start + args.batch_size, len(rows)), len(rows))

    elapsed = time.time() - t0
    log.info("Inference done in %.1fs (%d preds, %d skipped)",
             elapsed, len(probs_class1), skipped)

    # --- resolve polarity empirically -------------------------------------
    truth = {r["image_id"]: int(r["label"]) for r in rows}
    pos = [v for k, v in probs_class1.items() if truth.get(k) == 1]
    neg = [v for k, v in probs_class1.items() if truth.get(k) == 0]
    if not pos or not neg:
        log.error("Need both classes present to resolve polarity.")
        return 1
    mp, mn = sum(pos) / len(pos), sum(neg) / len(neg)
    log.info("mean P(class1): AI-labelled %.3f | real-labelled %.3f", mp, mn)

    margin = abs(mp - mn)
    if margin < 0.02:
        log.error("Classes are not separated either way (margin %.3f). "
                  "This model tells us nothing on this data; refusing to guess.",
                  margin)
        return 1

    if mp > mn:
        log.info("Polarity: class index 1 = AI-generated")
        out = probs_class1
    else:
        log.info("Polarity: class index 0 = AI-generated (INVERTING)")
        out = {k: 1.0 - v for k, v in probs_class1.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / "effnet_b0.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("WROTE %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
