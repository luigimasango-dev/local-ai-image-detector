#!/usr/bin/env python3
"""Apply a fitted calibration to a prediction file.

Emits calibrated probabilities so they can be scored with the full report
(confidence interval, per-generator breakdown) by score.py, and so the exact
same transform can be ported into the extension.

Usage:
    python apply_calibration.py --predictions predictions/fused.json \
        --calibration calibration/fused.json --out predictions/fused_cal.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    if cal.get("method") != "platt":
        p.error(f"unsupported calibration method: {cal.get('method')}")
    a, b = float(cal["a"]), float(cal["b"])

    raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    out = {k: sigmoid(a * logit(float(v)) + b) for k, v in raw.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Applied platt(a={a:.4f}, b={b:.4f}) to {len(out)} predictions")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
