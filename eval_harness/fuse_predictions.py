#!/usr/bin/env python3
"""Fuse several models' prediction files into one, so the fused score can be
analyzed and calibrated exactly like a single model.

Ensembling only helps when members make *different* mistakes. This script also
reports the correlation between members, because two highly-correlated models
cost twice the download and inference time for almost no accuracy gain -- which
matters a lot when the thing has to run in a browser.

Usage
-----
    python fuse_predictions.py --predictions a.json b.json --weights 0.5 0.5 \
        --out predictions/fused.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import load_labels  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", nargs="+", type=Path, required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sets = [json.loads(f.read_text(encoding="utf-8")) for f in args.predictions]
    names = [f.stem for f in args.predictions]
    weights = args.weights or [1.0 / len(sets)] * len(sets)
    if len(weights) != len(sets):
        p.error("--weights count must match --predictions count")

    common = set(sets[0])
    for s in sets[1:]:
        common &= set(s)
    print(f"Models: {len(sets)} | common images: {len(common)}")
    for n, s in zip(names, sets):
        print(f"  {n:<40} {len(s)} predictions")
    if len(common) < max(len(s) for s in sets):
        print(f"  NOTE: fusing over the {len(common)} images every model scored.")

    ordered = sorted(common)
    if len(sets) > 1:
        print("\nPairwise correlation between members:")
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                r = pearson([float(sets[i][k]) for k in ordered],
                            [float(sets[j][k]) for k in ordered])
                verdict = ("nearly redundant" if r > 0.95 else
                           "highly correlated" if r > 0.85 else
                           "usefully diverse" if r < 0.7 else "moderately correlated")
                print(f"  {names[i]} vs {names[j]}: r={r:.4f}  ({verdict})")

    total_w = sum(weights)
    fused = {
        k: sum(w * float(s[k]) for w, s in zip(weights, sets)) / total_w
        for k in ordered
    }

    # Disagreement diagnostic: where members disagree most, an ensemble either
    # earns its keep or actively hurts.
    truth = load_labels(args.labels)
    if len(sets) > 1:
        spreads = [(max(float(s[k]) for s in sets) - min(float(s[k]) for s in sets), k)
                   for k in ordered]
        spreads.sort(reverse=True)
        big = [k for sp, k in spreads if sp > 0.5]
        print(f"\nImages where members disagree by >0.5: {len(big)} "
              f"({100*len(big)/len(ordered):.1f}%)")
        if big:
            correct = sum(1 for k in big
                          if (fused[k] >= 0.5) == (truth.get(k) == 1))
            print(f"  Fused call is right on {correct}/{len(big)} of them "
                  f"({100*correct/len(big):.1f}%) at a 0.5 cut.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fused, indent=1), encoding="utf-8")
    print(f"\nWrote {len(fused)} fused predictions: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
