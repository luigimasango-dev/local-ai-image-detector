#!/usr/bin/env python3
"""Score one or more prediction JSON files against data/labels.csv.

Combines per-model probabilities with a weighted average, then reports
balanced accuracy (positive class = AI-generated) at a confidence threshold,
a confusion matrix, and accuracy at several reference thresholds.

Usage
-----
    python score.py \
        --predictions predictions/model_a.json predictions/model_b.json \
        --weights 0.5 0.5 \
        --threshold 0.65

Notes
-----
- ``label`` in labels.csv = 1 for AI-generated (positive class), 0 for real.
- ``probability_ai_generated`` >= threshold  =>  predicted AI (positive).
- Balanced accuracy = (sensitivity + specificity) / 2:
      sensitivity (true positive rate)  = TP / (TP + FN)
      specificity (true negative rate)  = TN / (TN + FP)
- If any label has no prediction from ANY model it is omitted from scoring
  and reported (we don't silently assume a 0.5 score).
- Predictions missing from some models are combined over the models that
  DID score that image (weights are re-normalized over the available models).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

REFERENCE_THRESHOLDS = (0.5, 0.6, 0.65, 0.7, 0.75)


def load_labels(labels_csv: Path) -> dict[str, int]:
    """Return image_id -> label (0/1)."""
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["image_id"]: int(r["label"]) for r in rows}


def load_generators(labels_csv: Path) -> dict[str, str]:
    """Return image_id -> generator name (or 'unknown')."""
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        r["image_id"]: (r.get("generator") or "unknown").strip() or "unknown"
        for r in rows
    }


def load_split(path: Path) -> set[str]:
    """Read a newline-delimited list of image_ids."""
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not ids:
        raise ValueError(f"{path}: split file is empty")
    return ids


def bootstrap_ci(
    combined: dict[str, float],
    truth: dict[str, int],
    threshold: float,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for balanced accuracy.

    Resamples the positive and negative classes independently, since balanced
    accuracy is an average of two class-conditional rates -- resampling the
    pooled set would let the class balance drift and widen the interval for the
    wrong reason.
    """
    pos = [i for i in combined if truth[i] == 1]
    neg = [i for i in combined if truth[i] == 0]
    if not pos or not neg:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        p_s = [combined[rng.choice(pos)] for _ in range(len(pos))]
        n_s = [combined[rng.choice(neg)] for _ in range(len(neg))]
        sens = sum(1 for v in p_s if v >= threshold) / len(p_s)
        spec = sum(1 for v in n_s if v < threshold) / len(n_s)
        stats.append((sens + spec) / 2.0)
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (lo, hi)


def load_predictions(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object (image_id -> prob)")
    cleaned = {}
    for k, v in data.items():
        try:
            cleaned[str(k)] = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{path}: non-numeric probability for key {k!r}")
    return cleaned


def combine_scores(
    labels: dict[str, int],
    pred_sets: list[dict[str, float]],
    weights: list[float],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Weighted mean of per-model probabilities per image.

    Returns (combined, warnings) where warnings[image_id] lists the models
    that had no prediction for that image.
    """
    total_weight = sum(weights)
    combined: dict[str, float] = {}
    warnings: dict[str, list[str]] = {}

    for image_id in labels:
        wsum = 0.0
        psum = 0.0
        available: list[str] = []
        for ws, ps in zip(weights, pred_sets):
            if image_id in ps:
                wsum += ws
                psum += ps[image_id] * ws
            else:
                available.append("missing")
        if wsum > 0:
            combined[image_id] = psum / wsum
        else:
            warnings[image_id] = [m for m in available if m] or ["all"]
    return combined, warnings


def confusion_at(combined: dict[str, float], truth: dict[str, int], threshold: float):
    tp = fp = tn = fn = 0
    for image_id, prob in combined.items():
        y_true = truth[image_id]
        y_pred = 1 if prob >= threshold else 0
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def metrics(combined: dict[str, float], truth: dict[str, int], threshold: float) -> dict:
    tp, fp, tn, fn = confusion_at(combined, truth, threshold)
    n = tp + fp + tn + fn
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "accuracy": accuracy,
    }


def print_generator_breakdown(combined, truth, generators, threshold):
    """Per-generator recall on the AI class, plus specificity on real images.

    A headline balanced accuracy can hide a generator we are blind to. This is
    the view that says which one.
    """
    if not generators:
        return
    by_gen: dict[str, list[str]] = defaultdict(list)
    for image_id in combined:
        if truth[image_id] == 1:
            by_gen[generators.get(image_id, "unknown")].append(image_id)
    if not by_gen:
        return

    print("-" * 64)
    print(f"  Per-generator AI recall (t={threshold}):")
    print(f"    {'generator':<26} {'n':>5} {'recall':>8}")
    rows = []
    for gen, ids in by_gen.items():
        hits = sum(1 for i in ids if combined[i] >= threshold)
        rows.append((hits / len(ids), gen, len(ids), hits))
    for recall, gen, n, hits in sorted(rows):
        flag = "  <-- weak" if recall < 0.5 else ""
        print(f"    {gen[:26]:<26} {n:>5} {recall:>8.4f}{flag}")


def print_report(combined, truth, threshold, generators=None, ci=None, title=None):
    m = metrics(combined, truth, threshold)
    print("=" * 64)
    print(f"EVAL REPORT{' — ' + title if title else ''} (positive class = AI-generated)")
    print("=" * 64)
    # Counted over the SCORED subset, not the whole labels file, so these
    # always reconcile with n above.
    print(f"Images scored : {m['n']}")
    print(f"AI (positive) : {sum(1 for i in combined if truth[i] == 1)}")
    print(f"Real (negative): {sum(1 for i in combined if truth[i] == 0)}")
    print("-" * 64)
    print(f"Threshold t={m['threshold']}")
    ci_str = f"   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
    print(f"  Balanced accuracy : {m['balanced_accuracy']:.4f}{ci_str}")
    print(f"  Accuracy          : {m['accuracy']:.4f}")
    print(f"  Sensitivity (TPR) : {m['sensitivity']:.4f}   (AI recall)")
    print(f"  Specificity (TNR) : {m['specificity']:.4f}   (real recall — "
          f"the false-positive-on-real-photos metric)")
    print("-" * 64)
    print("  Confusion matrix:")
    print(f"    TP (AI->AI)   : {m['tp']}")
    print(f"    FP (Real->AI) : {m['fp']}")
    print(f"    TN (Real->Real): {m['tn']}")
    print(f"    FN (AI->Real) : {m['fn']}")
    print("-" * 64)
    print("  Reference thresholds:")
    print(f"    {'t':>5} {'acc':>8} {'bal_acc':>9}")
    for t in REFERENCE_THRESHOLDS:
        r = metrics(combined, truth, t)
        mark = "  <-- bounty threshold" if abs(t - 0.65) < 1e-9 else ""
        print(f"    {t:>5.2f} {r['accuracy']:>8.4f} {r['balanced_accuracy']:>9.4f}{mark}")
    print_generator_breakdown(combined, truth, generators, threshold)
    print("=" * 64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="One or more prediction JSON files.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Model weights (defaults to equal weights).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65, help="Default confidence threshold."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_DATA_DIR / "labels.csv",
        help="Ground-truth labels.csv.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="Restrict scoring to image_ids listed in this file "
             "(see split_eval_set.py). Fit on split_tune.txt, report the "
             "go/no-go number from split_holdout.txt.",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Skip the bootstrap confidence interval (faster).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the metrics to this path as JSON.",
    )
    args = parser.parse_args()

    if args.threshold < 0 or args.threshold > 1:
        parser.error("--threshold must be in [0, 1]")

    if len(args.predictions) == 1 and args.weights and len(args.weights) != 1:
        parser.error("--weights must match number of --predictions")

    labels = load_labels(args.labels)
    if not labels:
        print("ERROR: labels.csv is empty or unreadable.", file=sys.stderr)
        return 1
    generators = load_generators(args.labels)

    split_name = None
    if args.split:
        split_ids = load_split(args.split)
        missing = split_ids - set(labels)
        if missing:
            print(f"WARNING: {len(missing)} ids in {args.split.name} are not in "
                  f"labels.csv and were ignored.", file=sys.stderr)
        labels = {k: v for k, v in labels.items() if k in split_ids}
        if not labels:
            print(f"ERROR: no overlap between {args.split} and labels.csv.",
                  file=sys.stderr)
            return 1
        split_name = args.split.stem
        print(f"Split filter: {args.split.name} -> {len(labels)} images")

    pred_sets = [load_predictions(Path(p)) for p in args.predictions]

    if args.weights:
        if len(args.weights) != len(pred_sets):
            parser.error(
                f"--weights has {len(args.weights)} values but "
                f"{len(pred_sets)} prediction files were given"
            )
        weights = args.weights
    else:
        weights = [1.0 / len(pred_sets)] * len(pred_sets)

    # Model names for reporting.
    model_labels = [Path(p).stem for p in args.predictions]

    print("Combining models with weights:")
    for ml, w, ps in zip(model_labels, weights, pred_sets):
        n_scored = sum(1 for img in labels if img in ps)
        print(f"  {ml:>40s}  w={w:.3f}  covers={n_scored}/{len(labels)} images")

    combined, warnings = combine_scores(labels, pred_sets, weights)

    n_unscored = len(warnings)
    if n_unscored:
        print(f"WARNING: {n_unscored} images have no prediction from any model "
              f"and were EXCLUDED from scoring:")
        for img in list(warnings)[:10]:
            print(f"  - {img}")
        if n_unscored > 10:
            print(f"  - ... and {n_unscored - 10} more")

    ci = None
    if not args.no_ci:
        ci = bootstrap_ci(combined, labels, args.threshold)

    print_report(
        combined, labels, args.threshold,
        generators=generators, ci=ci, title=split_name,
    )

    if args.json_out:
        payload = metrics(combined, labels, args.threshold)
        payload.update({
            "split": split_name,
            "models": model_labels,
            "weights": weights,
            "ci95_balanced_accuracy": list(ci) if ci else None,
            "excluded_unscored": n_unscored,
        })
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote metrics JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())