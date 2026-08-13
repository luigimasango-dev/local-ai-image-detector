#!/usr/bin/env python3
"""Where do we call a real photo AI-generated?

Balanced accuracy averages away the asymmetry that actually annoys users. A
browsing extension that slaps "AI-generated" on someone's holiday photo is worse
than one that misses a Midjourney render -- and the competing claim on this
bounty was publicly criticised for exactly that. So false positives get their
own report, broken down by subject category, plus the specific images we get
most confidently wrong.

Usage:
    python false_positive_report.py --predictions predictions/fused_calibrated.json \
        [--split data/split_holdout.txt] [--threshold 0.65]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def category_of(filepath: str) -> str:
    """images/real/animal/animal_036_x.jpg -> animal"""
    parts = Path(filepath).parts
    return parts[2] if len(parts) >= 3 else "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.split:
        keep = {l.strip() for l in args.split.read_text(encoding="utf-8").splitlines() if l.strip()}
        rows = [r for r in rows if r["image_id"] in keep]

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    real = [r for r in rows if r["label"] == "0" and r["image_id"] in preds]
    if not real:
        print("No real images with predictions.", file=sys.stderr)
        return 1

    t = args.threshold
    print("=" * 66)
    print(f"FALSE-POSITIVE REPORT  (real photos wrongly called AI, t={t})")
    print("=" * 66)

    fp = [r for r in real if float(preds[r["image_id"]]) >= t]
    print(f"Real images        : {len(real)}")
    print(f"False positives    : {len(fp)}  ({100*len(fp)/len(real):.1f}%)")
    print(f"Specificity        : {1 - len(fp)/len(real):.4f}")

    by_cat: dict[str, list] = defaultdict(list)
    for r in real:
        by_cat[category_of(r["filepath"])].append(r)

    print("\n" + "-" * 66)
    print("By subject category (a category far above the mean is a real bug,")
    print("not noise -- it means a whole class of photo triggers the detector):")
    print(f"  {'category':<14} {'n':>5} {'FPs':>5} {'FP rate':>9}")
    overall = len(fp) / len(real)
    for cat in sorted(by_cat, key=lambda c: -sum(
            1 for r in by_cat[c] if float(preds[r["image_id"]]) >= t) / max(len(by_cat[c]), 1)):
        items = by_cat[cat]
        n_fp = sum(1 for r in items if float(preds[r["image_id"]]) >= t)
        rate = n_fp / len(items) if items else 0.0
        flag = "  <-- well above average" if rate > overall * 1.75 and n_fp >= 3 else ""
        print(f"  {cat:<14} {len(items):>5} {n_fp:>5} {rate:>9.4f}{flag}")

    print("\n" + "-" * 66)
    print(f"Most confidently wrong real photos (top {args.top}):")
    worst = sorted(real, key=lambda r: -float(preds[r["image_id"]]))[: args.top]
    for r in worst:
        score = float(preds[r["image_id"]])
        mark = "FP" if score >= t else "ok"
        print(f"  [{mark}] {score:.4f}  {r['filepath']}")

    print("\n" + "-" * 66)
    print("Score distribution on real photos:")
    buckets = [(0, .1), (.1, .3), (.3, .5), (.5, .65), (.65, .8), (.8, .9), (.9, 1.01)]
    for lo, hi in buckets:
        n = sum(1 for r in real if lo <= float(preds[r["image_id"]]) < hi)
        bar = "#" * int(60 * n / len(real))
        side = "  (flagged)" if lo >= t else ""
        print(f"  {lo:.2f}-{hi:.2f} {n:>4} |{bar}{side}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
