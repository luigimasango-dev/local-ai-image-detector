"""Regression tests for detector class-label resolution.

Both functions under test guard against silent-wrong-answer bugs: a bad class
mapping does not crash, it just produces a confident, meaningless score for
every image (typically a tidy-looking 0.5000 balanced accuracy). These tests
exist because both bugs were present in the first generated version of
run_inference.py.

Run:  python -m pytest tests/ -q      (or)  python tests/test_label_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_harness"))

from run_inference import _ai_score, build_label_index, resolve_label_index  # noqa: E402


# --------------------------------------------------------------------------
# _ai_score: which class name means "AI-generated"
# --------------------------------------------------------------------------

# (class_names, expected_ai_index) -- vocabularies real HF detectors export
VOCAB_CASES = [
    (["artificial", "human"], 0),        # umm-maybe/AI-image-detector
    (["Real", "Fake"], 1),               # Organika/sdxl-detector style
    (["REAL", "FAKE"], 1),
    (["ai", "real"], 0),
    (["AI-Generated", "Real"], 0),
    (["not_ai", "ai"], 1),               # negation trap
    (["non-ai", "ai_generated"], 1),     # negation trap
    (["natural", "synthetic"], 1),
    (["photograph", "diffusion"], 1),
    (["human_made", "machine_made"], 1),
]

# Words that must NOT register as AI. Substring matching on "ai"/"gan" would
# score every one of these as the AI class and silently invert the mapping.
SUBSTRING_BLEED = ["portrait", "captain", "chain", "organic", "certain", "detail"]


def test_vocabularies_resolve_to_the_right_class():
    for names, expected_ai in VOCAB_CASES:
        scores = [_ai_score(n) for n in names]
        ai_idxs = [i for i, s in enumerate(scores) if s > 0]
        real_idxs = [i for i, s in enumerate(scores) if s < 0]
        assert len(ai_idxs) == 1, f"{names}: ambiguous AI side (scores={scores})"
        assert len(real_idxs) == 1, f"{names}: ambiguous real side (scores={scores})"
        assert ai_idxs[0] == expected_ai, f"{names}: got ai_idx={ai_idxs[0]}, want {expected_ai}"


def test_substrings_do_not_bleed_into_ai_score():
    for word in SUBSTRING_BLEED:
        assert _ai_score(word) == 0, f"{word!r} scored {_ai_score(word)}, want 0"


# --------------------------------------------------------------------------
# resolve_label_index: pipeline returns label STRINGS, not indices
# --------------------------------------------------------------------------

def _cfg(id2label):
    return SimpleNamespace(
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        num_labels=len(id2label),
    )


def test_label_strings_resolve_to_indices():
    idx = build_label_index(_cfg({0: "artificial", 1: "human"}))
    assert resolve_label_index("artificial", idx) == 0
    assert resolve_label_index("human", idx) == 1
    # case and whitespace insensitive
    assert resolve_label_index("  ARTIFICIAL ", idx) == 0


def test_label_n_fallback_when_config_has_no_vocabulary():
    idx = build_label_index(SimpleNamespace())
    assert resolve_label_index("LABEL_0", idx) == 0
    assert resolve_label_index("LABEL_1", idx) == 1
    assert resolve_label_index("1", idx) == 1


def test_unknown_label_returns_none_rather_than_guessing():
    idx = build_label_index(_cfg({0: "artificial", 1: "human"}))
    assert resolve_label_index("something_else", idx) is None


def test_full_extraction_path_yields_a_real_probability():
    """End-to-end shape check on the exact bug that was fixed.

    A pipeline result is a list of {"label": <string>, "score": float}. The
    original code keyed these by integer index, so the AI-class lookup always
    missed and every image collapsed to the same score.
    """
    config = _cfg({0: "artificial", 1: "human"})
    idx = build_label_index(config)
    pipeline_result = [
        {"label": "human", "score": 0.83},
        {"label": "artificial", "score": 0.17},
    ]
    probs = {}
    for item in pipeline_result:
        i = resolve_label_index(item["label"], idx)
        assert i is not None, f"failed to resolve {item['label']!r}"
        probs[i] = item["score"]

    ai_idx = 0  # "artificial"
    assert probs[ai_idx] == 0.17, "AI probability must survive the round trip"
    assert len(probs) == 2, "both classes must resolve"


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
