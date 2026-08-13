#!/usr/bin/env python3
"""Fit a calibration so the emitted confidence score means what it says.

Why
---
Community-Forensics ranks AI vs real well (AUC ~0.90) but its raw sigmoid is
badly calibrated on this data: the optimal decision boundary sits near 0.055,
while the bounty grades at 0.65. Emitting raw scores would score ~0.71 and
fail; the same model, calibrated, clears the bar.

This is Platt scaling: fit  p = sigmoid(a * logit(s) + b)  on the TUNE split.
It is a *monotonic* transform, so it changes neither the ranking nor which
images are flagged at the optimum -- it only maps scores onto the probability
scale where 0.65 genuinely means "65% confident this is AI-generated". That is
the honest reading of the bounty's requirement to "display a confidence score",
and it is ordinary ML practice, not benchmark gaming.

Fit on TUNE only. Verify on HOLDOUT. Never fit on the data you report from.

Usage
-----
    python calibrate.py --predictions predictions/model.json \
        --tune data/split_tune.txt --holdout data/split_holdout.txt \
        --out calibration/model.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import load_labels, load_split  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(scores: list[float], labels: list[int],
              iters: int = 2000, lr: float = 0.05) -> tuple[float, float]:
    """Fit a, b in p = sigmoid(a * logit(s) + b) by gradient descent on log loss.

    Plain gradient descent keeps this dependency-free and the problem is
    two-dimensional and convex, so it converges reliably.
    """
    x = [logit(s) for s in scores]
    y = labels
    n = len(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for xi, yi in zip(x, y):
            p = sigmoid(a * xi + b)
            err = p - yi
            ga += err * xi
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def balanced_acc(scores: list[float], labels: list[int], t: float) -> float:
    tp = sum(1 for s, y in zip(scores, labels) if y == 1 and s >= t)
    fn = sum(1 for s, y in zip(scores, labels) if y == 1 and s < t)
    tn = sum(1 for s, y in zip(scores, labels) if y == 0 and s < t)
    fp = sum(1 for s, y in zip(scores, labels) if y == 0 and s >= t)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (sens + spec) / 2.0


def log_loss(scores: list[float], labels: list[int]) -> float:
    total = 0.0
    for s, y in zip(scores, labels):
        p = min(max(s, EPS), 1.0 - EPS)
        total -= math.log(p) if y == 1 else math.log(1.0 - p)
    return total / len(scores)


def subset(preds: dict, truth: dict, ids: set[str]):
    keys = [k for k in preds if k in truth and k in ids]
    return [float(preds[k]) for k in keys], [truth[k] for k in keys]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--tune", type=Path, default=DEFAULT_DATA_DIR / "split_tune.txt")
    p.add_argument("--holdout", type=Path, default=DEFAULT_DATA_DIR / "split_holdout.txt")
    p.add_argument("--target-threshold", type=float, default=0.65)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    truth = load_labels(args.labels)
    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    tune_ids, hold_ids = load_split(args.tune), load_split(args.holdout)

    s_tune, y_tune = subset(preds, truth, tune_ids)
    s_hold, y_hold = subset(preds, truth, hold_ids)
    print(f"TUNE {len(s_tune)} images | HOLDOUT {len(s_hold)} images")

    t = args.target_threshold
    print(f"\n--- BEFORE calibration (raw scores) ---")
    print(f"  TUNE    bal_acc @{t}: {balanced_acc(s_tune, y_tune, t):.4f}   "
          f"log loss {log_loss(s_tune, y_tune):.4f}")
    print(f"  HOLDOUT bal_acc @{t}: {balanced_acc(s_hold, y_hold, t):.4f}   "
          f"log loss {log_loss(s_hold, y_hold):.4f}")

    a, b = fit_platt(s_tune, y_tune)
    print(f"\nFitted Platt scaling on TUNE only: a={a:.4f}, b={b:.4f}")

    c_tune = [sigmoid(a * logit(s) + b) for s in s_tune]
    c_hold = [sigmoid(a * logit(s) + b) for s in s_hold]

    print(f"\n--- AFTER calibration ---")
    ba_tune = balanced_acc(c_tune, y_tune, t)
    ba_hold = balanced_acc(c_hold, y_hold, t)
    print(f"  TUNE    bal_acc @{t}: {ba_tune:.4f}   "
          f"log loss {log_loss(c_tune, y_tune):.4f}")
    print(f"  HOLDOUT bal_acc @{t}: {ba_hold:.4f}   "
          f"log loss {log_loss(c_hold, y_hold):.4f}   <-- THE NUMBER THAT COUNTS")

    print(f"\n  HOLDOUT gain from calibration: "
          f"{ba_hold - balanced_acc(s_hold, y_hold, t):+.4f}")
    print(f"  TUNE -> HOLDOUT drop: {ba_hold - ba_tune:+.4f}  "
          f"(large negative = calibration overfitted the tune split)")

    verdict = "CLEARS" if ba_hold >= 0.75 else "MISSES"
    print(f"\n  >>> {verdict} the 0.75 bar on unseen data at t={t}: {ba_hold:.4f}")
    if 0.75 <= ba_hold < 0.80:
        print("      Margin is thin. The private benchmark is differently")
        print("      distributed; treat this as 'not yet', not 'ship it'.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "method": "platt",
            "a": a, "b": b,
            "formula": "p_calibrated = sigmoid(a * logit(p_raw) + b)",
            "fitted_on": args.tune.name,
            "target_threshold": t,
            "tune_balanced_accuracy": ba_tune,
            "holdout_balanced_accuracy": ba_hold,
            "source_predictions": args.predictions.name,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote calibration: {args.out}")
        print("The extension MUST apply this same transform before thresholding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
