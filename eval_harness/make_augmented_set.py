#!/usr/bin/env python3
"""Build recompressed / resized variants of the eval set.

Why this matters more than it sounds
------------------------------------
Our eval set is pristine: original-quality images straight from the benchmark
archive. That is not how images reach a browser. Every social platform, CMS and
CDN re-encodes and downscales, and AI-detectors are known to degrade sharply
under recompression because much of the signal they use lives in high-frequency
detail that JPEG quantization destroys first.

So a detector can score well on a clean benchmark and be much weaker in the
actual product. This script builds the degraded variants so we can measure that
gap instead of hoping it is small.

Conditions:
  jpeg95 / jpeg85 / jpeg75  -- recompression at descending quality
  scale512 / scale256       -- downscale longest edge, re-encode at q=90
  combo                     -- scale 512 then JPEG 75 (the realistic web case)

Usage:
    python make_augmented_set.py --condition jpeg75 --split data/split_holdout.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

CONDITIONS = {
    "jpeg95": {"quality": 95, "max_edge": None},
    "jpeg85": {"quality": 85, "max_edge": None},
    "jpeg75": {"quality": 75, "max_edge": None},
    "scale512": {"quality": 90, "max_edge": 512},
    "scale256": {"quality": 90, "max_edge": 256},
    "combo": {"quality": 75, "max_edge": 512},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--split", type=Path, default=None,
                   help="Only augment ids in this split file (saves time).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = p.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow required (pip install pillow)", file=sys.stderr)
        return 1

    cfg = CONDITIONS[args.condition]
    out_root = args.data_dir / f"aug_{args.condition}"
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.split:
        keep = {l.strip() for l in args.split.read_text(encoding="utf-8").splitlines() if l.strip()}
        rows = [r for r in rows if r["image_id"] in keep]

    print(f"Condition {args.condition}: quality={cfg['quality']} "
          f"max_edge={cfg['max_edge']} | {len(rows)} images")

    out_rows, failed = [], 0
    for i, r in enumerate(rows, 1):
        src = args.data_dir / r["filepath"]
        # Always .jpg -- the point is that it has been through JPEG.
        rel = Path(r["filepath"]).with_suffix(".jpg")
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                if cfg["max_edge"]:
                    w, h = im.size
                    if max(w, h) > cfg["max_edge"]:
                        s = cfg["max_edge"] / max(w, h)
                        im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                                       Image.LANCZOS)
                im.save(dst, "JPEG", quality=cfg["quality"])
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {r['image_id']}: {exc}")
            failed += 1
            continue
        out_rows.append({
            "image_id": r["image_id"],          # same id -> comparable to clean run
            "filepath": str(rel).replace("\\", "/"),
            "label": r["label"],
            "generator": r["generator"],
        })
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}")

    labels_out = out_root / "labels.csv"
    with open(labels_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "filepath", "label", "generator"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} images ({failed} failed) -> {out_root}")
    print(f"Labels: {labels_out}")
    print(f"\nScore this condition with:")
    print(f"  python run_inference_commfor.py --model OwensLab/commfor-model-224 \\")
    print(f"      --labels {labels_out} --data-dir {out_root} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
