#!/usr/bin/env python3
"""Separate a model's discriminative power from its calibration.

Balanced accuracy at a fixed threshold conflates two very different things:

  1. Can the model *rank* AI above real at all?   -> AUC, threshold-free
  2. Is 0.65 the right place to cut?              -> calibration

A model can rank almost perfectly and still score badly at t=0.65 simply
because its probabilities are squashed toward zero. That is a calibration
problem, and calibration is fixable; poor ranking is not.

The bounty evaluates at a fixed 0.65 threshold, but we control what score the
extension emits. Fitting a monotonic calibration so our operating point lands
at 0.65 is ordinary probability calibration -- it does not change the ranking
or which images are called AI at the optimum, it just puts the decision
boundary where the graders read it.

Usage:
    python analyze_threshold.py --predictions predictions/model.json \
        [--split data/split_tune.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import load_labels, load_split  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def auc_roc(scores: dict[str, float], truth: dict[str, int]) -> float:
    """AUC via the rank-sum (Mann-Whitney U) identity, with tie handling."""
    pairs = sorted(((scores[i], truth[i]) for i in scores), key=lambda p: p[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan")

    # Average ranks across ties so tied scores don't bias the statistic.
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def balanced_acc_at(scores, truth, t) -> tuple[float, float, float]:
    tp = sum(1 for i in scores if truth[i] == 1 and scores[i] >= t)
    fn = sum(1 for i in scores if truth[i] == 1 and scores[i] < t)
    tn = sum(1 for i in scores if truth[i] == 0 and scores[i] < t)
    fp = sum(1 for i in scores if truth[i] == 0 and scores[i] >= t)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (sens + spec) / 2.0, sens, spec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--split", type=Path, default=None)
    args = p.parse_args()

    truth = load_labels(args.labels)
    if args.split:
        ids = load_split(args.split)
        truth = {k: v for k, v in truth.items() if k in ids}

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    scores = {k: float(v) for k, v in preds.items() if k in truth}
    truth = {k: v for k, v in truth.items() if k in scores}

    n_pos = sum(1 for v in truth.values() if v == 1)
    n_neg = len(truth) - n_pos
    print(f"Scored {len(scores)} images ({n_pos} AI / {n_neg} real)")
    print(f"Score range: {min(scores.values()):.6f} .. {max(scores.values()):.6f}")

    auc = auc_roc(scores, truth)
    print(f"\nAUC (threshold-free ranking quality) : {auc:.4f}")
    print("  ^ this is the model's real discriminative ceiling.")
    print("    1.00 = perfect ranking, 0.50 = coin flip.")

    # Sweep every distinct score as a candidate cut point.
    candidates = sorted(set(scores.values()))
    best = max(
        ((balanced_acc_at(scores, truth, t)[0], t) for t in candidates),
        key=lambda x: x[0],
    )
    best_ba, best_t = best
    _, bsens, bspec = balanced_acc_at(scores, truth, best_t)

    ba65, s65, p65 = balanced_acc_at(scores, truth, 0.65)

    print(f"\nAt the bounty threshold t=0.65:")
    print(f"  balanced accuracy {ba65:.4f}   (sens {s65:.4f} / spec {p65:.4f})")
    print(f"\nAt the OPTIMAL threshold t={best_t:.6f}:")
    print(f"  balanced accuracy {best_ba:.4f}   (sens {bsens:.4f} / spec {bspec:.4f})")
    print(f"\n  Head-room from calibration alone: "
          f"{best_ba - ba65:+.4f} balanced accuracy")

    print("\nThreshold sweep:")
    print(f"    {'t':>10} {'bal_acc':>9} {'sens':>7} {'spec':>7}")
    for t in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9]:
        ba, se, sp = balanced_acc_at(scores, truth, t)
        mark = "  <-- bounty" if t == 0.65 else ""
        print(f"    {t:>10.4f} {ba:>9.4f} {se:>7.4f} {sp:>7.4f}{mark}")

    if best_ba >= 0.75 > ba65:
        print(f"\n*** The model CLEARS the 0.75 bar when correctly calibrated "
              f"({best_ba:.4f}) but MISSES it at the raw 0.65 cut ({ba65:.4f}). "
              f"Calibration is the fix, not a different model. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
