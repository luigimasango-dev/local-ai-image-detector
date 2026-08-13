#!/usr/bin/env python3
"""Estimate the score on a benchmark that mixes pristine and web-realistic images.

The bounty describes its held-out set as "assembled from publicly available
datasets AND additional web-realistic samples". We score 0.8338 on pristine
images and 0.7290 on re-compressed ones, so the honest expected score is
somewhere between -- and where exactly depends on a mix ratio we cannot observe.

Rather than pick one number and hope, this sweeps the mix ratio and reports the
whole curve, so the decision is made against a range instead of a guess. It also
fits the calibration on a MIXED tune split, which matters: a calibration fitted
only on pristine images puts the decision boundary in the wrong place for
degraded ones.

Usage:
    python mixed_benchmark_estimate.py \
        --clean-tune predictions/fused.json \
        --degraded-tune predictions/aug_combo_tune/fused.json \
        --clean-holdout predictions/fused.json \
        --degraded-holdout predictions/aug_combo/fused.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import fit_platt, logit, sigmoid  # noqa: E402
from score import load_labels, load_split  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def balanced_acc(scores: list[float], labels: list[int], t: float) -> float:
    tp = sum(1 for s, y in zip(scores, labels) if y == 1 and s >= t)
    fn = sum(1 for s, y in zip(scores, labels) if y == 1 and s < t)
    tn = sum(1 for s, y in zip(scores, labels) if y == 0 and s < t)
    fp = sum(1 for s, y in zip(scores, labels) if y == 0 and s >= t)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (sens + spec) / 2.0


def build_mix(clean: dict, degraded: dict, ids: list[str],
              frac_degraded: float, seed: int) -> list[float]:
    """Per image, take the degraded score with probability frac_degraded."""
    rng = random.Random(seed)
    out = []
    for i in ids:
        use_deg = rng.random() < frac_degraded
        src = degraded if (use_deg and i in degraded) else clean
        out.append(float(src[i]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-tune", type=Path, required=True)
    p.add_argument("--degraded-tune", type=Path, required=True)
    p.add_argument("--clean-holdout", type=Path, required=True)
    p.add_argument("--degraded-holdout", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--tune-split", type=Path, default=DEFAULT_DATA_DIR / "split_tune.txt")
    p.add_argument("--holdout-split", type=Path, default=DEFAULT_DATA_DIR / "split_holdout.txt")
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    truth = load_labels(args.labels)
    tune_ids = sorted(load_split(args.tune_split) & set(truth))
    hold_ids = sorted(load_split(args.holdout_split) & set(truth))

    ct = json.loads(args.clean_tune.read_text(encoding="utf-8"))
    dt = json.loads(args.degraded_tune.read_text(encoding="utf-8"))
    ch = json.loads(args.clean_holdout.read_text(encoding="utf-8"))
    dh = json.loads(args.degraded_holdout.read_text(encoding="utf-8"))

    tune_ids = [i for i in tune_ids if i in ct and i in dt]
    hold_ids = [i for i in hold_ids if i in ch and i in dh]
    y_tune = [truth[i] for i in tune_ids]
    y_hold = [truth[i] for i in hold_ids]
    print(f"TUNE {len(tune_ids)} | HOLDOUT {len(hold_ids)} images "
          f"(each available clean AND degraded)")

    # Calibrate on a 50/50 mixed tune set -- a calibration fitted only on
    # pristine data mis-places the boundary for degraded input.
    mixed_tune = build_mix(ct, dt, tune_ids, 0.5, args.seed)
    a, b = fit_platt(mixed_tune, y_tune)
    print(f"Platt fitted on 50/50 MIXED tune: a={a:.4f}, b={b:.4f}")

    # Compare against a pristine-only calibration to show why the mix matters.
    a0, b0 = fit_platt([float(ct[i]) for i in tune_ids], y_tune)
    print(f"Platt fitted on CLEAN-only tune  : a={a0:.4f}, b={b0:.4f}")

    t = args.threshold
    print(f"\nExpected balanced accuracy at t={t} vs. share of the benchmark")
    print("that is web-realistic (re-compressed):\n")
    print(f"  {'% degraded':>11} {'mixed-cal':>10} {'clean-cal':>10}   verdict")
    for frac in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
        # Average over several seeds so a single unlucky draw is not the answer.
        accs_mixed, accs_clean = [], []
        for s in range(5):
            mix = build_mix(ch, dh, hold_ids, frac, args.seed + s)
            accs_mixed.append(balanced_acc(
                [sigmoid(a * logit(v) + b) for v in mix], y_hold, t))
            accs_clean.append(balanced_acc(
                [sigmoid(a0 * logit(v) + b0) for v in mix], y_hold, t))
        m = sum(accs_mixed) / len(accs_mixed)
        c = sum(accs_clean) / len(accs_clean)
        verdict = "PASS" if m >= 0.75 else "FAIL"
        print(f"  {frac*100:>10.0f}% {m:>10.4f} {c:>10.4f}   {verdict}")

    print("\nReading this table:")
    print("  'mixed-cal' is what we would score having calibrated on both clean")
    print("  and degraded data. 'clean-cal' is calibrating on pristine only --")
    print("  the difference is the cost of assuming the benchmark is clean.")
    print("  The break-even row is the most degraded benchmark we can survive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
