"""Tests for score.py metrics against hand-computed expected values.

Balanced accuracy is the number the entire go/no-go decision rests on, so it
gets checked against cases whose answer is known by construction rather than
against whatever the code happens to output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_harness"))

from score import bootstrap_ci, combine_scores, confusion_at, metrics  # noqa: E402


def test_perfect_classifier_scores_one():
    truth = {"a": 1, "b": 1, "c": 0, "d": 0}
    combined = {"a": 0.99, "b": 0.90, "c": 0.01, "d": 0.10}
    m = metrics(combined, truth, 0.65)
    assert m["balanced_accuracy"] == 1.0
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 2, 0)


def test_everything_predicted_ai_gives_exactly_half():
    """The signature of the label-mapping bug: constant 1.0 for every image.

    It does not crash and it does not look absurd -- it looks like a tidy
    0.5000. This test pins that signature so it is recognizable.
    """
    truth = {"a": 1, "b": 1, "c": 0, "d": 0}
    combined = {k: 1.0 for k in truth}
    m = metrics(combined, truth, 0.65)
    assert m["balanced_accuracy"] == 0.5
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 0.0


def test_balanced_accuracy_ignores_class_imbalance():
    """9 AI + 1 real, detector nails AI and misses the single real image."""
    truth = {f"ai{i}": 1 for i in range(9)}
    truth["real0"] = 0
    combined = {k: 0.99 for k in truth}  # everything called AI
    m = metrics(combined, truth, 0.65)
    # Plain accuracy flatters this badly; balanced accuracy does not.
    assert m["accuracy"] == 0.9
    assert m["balanced_accuracy"] == 0.5


def test_threshold_boundary_is_inclusive():
    truth = {"a": 1, "b": 0}
    combined = {"a": 0.65, "b": 0.64999}
    m = metrics(combined, truth, 0.65)
    assert m["tp"] == 1, "score exactly at threshold must count as positive"
    assert m["tn"] == 1


def test_weighted_combination_respects_weights():
    labels = {"x": 1}
    preds = [{"x": 1.0}, {"x": 0.0}]
    combined, _ = combine_scores(labels, preds, [0.75, 0.25])
    assert abs(combined["x"] - 0.75) < 1e-9


def test_images_missing_from_one_model_renormalize():
    """An image only model A scored must use A's value, not a diluted average."""
    labels = {"x": 1}
    preds = [{"x": 0.8}, {}]  # model B has no prediction for x
    combined, warnings = combine_scores(labels, preds, [0.5, 0.5])
    assert abs(combined["x"] - 0.8) < 1e-9, "weights must renormalize over available models"
    assert not warnings


def test_image_scored_by_nobody_is_excluded_not_assumed():
    labels = {"x": 1, "y": 0}
    preds = [{"x": 0.9}]
    combined, warnings = combine_scores(labels, preds, [1.0])
    assert "y" not in combined, "unscored image must be excluded, never defaulted to 0.5"
    assert "y" in warnings


def test_confusion_counts_sum_to_scored_total():
    truth = {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0}
    combined = {"a": 0.9, "b": 0.2, "c": 0.8, "d": 0.1, "e": 0.66}
    tp, fp, tn, fn = confusion_at(combined, truth, 0.65)
    assert tp + fp + tn + fn == len(combined)
    assert (tp, fp, tn, fn) == (1, 2, 1, 1)


def test_bootstrap_ci_brackets_the_point_estimate():
    truth = {f"ai{i}": 1 for i in range(50)}
    truth.update({f"re{i}": 0 for i in range(50)})
    combined = {k: (0.9 if v == 1 else 0.1) for k, v in truth.items()}
    # Flip a few to make it imperfect so the interval has width.
    for k in ["ai0", "ai1", "ai2", "re0", "re1"]:
        combined[k] = 1.0 - combined[k]
    point = metrics(combined, truth, 0.65)["balanced_accuracy"]
    lo, hi = bootstrap_ci(combined, truth, 0.65, n_boot=500)
    assert lo <= point <= hi, f"CI [{lo}, {hi}] must bracket point estimate {point}"
    assert 0.0 <= lo <= hi <= 1.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
