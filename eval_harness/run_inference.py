#!/usr/bin/env python3
"""Run a model over every image in data/labels.csv and write predictions.

Supported backends:
  - ``transformers`` (default): a Hugging Face ``transformers``
    image-classification pipeline, given a model id or local path
    (e.g. ``umm-maybe/AI-image-detector``).

Output
------
``predictions/<model_name>.json`` — a flat dict mapping
``image_id -> probability_ai_generated`` (float in [0, 1]).

Label-order handling
--------------------
We do NOT assume index 0 == real / index 1 == AI. Different models export
different label vocabularies and orderings (e.g. ``["REAL", "FAKE"]`` vs
``["AI-Generated", "Real"]`` vs integer ids ``["0", "1"]``). The script
inspects the model config's ``id2label``/``label2id`` (and the pipeline's
returned label strings) and maps classes to the AI/real semantics by keyword.

If the mapping cannot be determined automatically (ambiguous labels such as
``"0"/"1"``), the script errors out rather than guessing, unless you pass an
explicit ``--ai-label`` override.

Usage
-----
    python run_inference.py --model umm-maybe/AI-image-detector \
        --backend transformers [--device cpu] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_inference")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PRED_DIR = Path(__file__).resolve().parent / "predictions"

# Keywords that mark a class label as the "AI-generated" side.
AI_KEYWORDS = (
    "ai", "fake", "generated", "generative", "synthetic", "diffusion", "gan",
    "artificial", "midjourney", "dalle", "dall", "sdxl", "sd", "stablediffusion",
    "deepfake", "unreal", "cgi", "machine",
)
# Keywords that mark a class label as the "real / photograph" side.
REAL_KEYWORDS = (
    "real", "photo", "photograph", "natural", "authentic", "genuine",
    "original", "camera", "actual", "human", "nonai", "notai",
)
# Negation markers: "not_ai" / "non-ai" must not be read as the AI class.
NEGATIONS = ("not", "non", "no")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _ai_score(label: str) -> int:
    """Score a class label: >0 means the AI side, <0 means the real side.

    Matches whole tokens, not substrings. Substring matching is unsafe here:
    "ai" appears inside "portrait", "captain" and "chain", and "gan" inside
    "organic" -- any of which would silently invert the class mapping.
    """
    low = label.lower()
    tokens = _TOKEN_RE.findall(low)
    # Treat "not_ai" / "non-ai" as a single real-side token.
    collapsed: list[str] = []
    skip_next = False
    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if tok in NEGATIONS and i + 1 < len(tokens):
            collapsed.append(tok + tokens[i + 1])  # e.g. "not"+"ai" -> "notai"
            skip_next = True
        else:
            collapsed.append(tok)

    score = 0
    for tok in collapsed:
        if tok in REAL_KEYWORDS:
            score -= 1
        elif tok in AI_KEYWORDS:
            score += 1
    return score


def build_label_index(config) -> dict[str, int]:
    """Map lowercased class-label strings -> class index, from a model config."""
    label2idx: dict[str, int] = {}
    for name, idx in (getattr(config, "label2id", None) or {}).items():
        label2idx[str(name).strip().lower()] = int(idx)
    for idx, name in (getattr(config, "id2label", None) or {}).items():
        label2idx.setdefault(str(name).strip().lower(), int(idx))
    return label2idx


def resolve_label_index(raw_label, label2idx: dict[str, int]) -> int | None:
    """Map a pipeline-returned label to its class index.

    The transformers image-classification pipeline returns the human-readable
    label STRING (e.g. "artificial", "FAKE"), never the integer index. Keying
    predictions by index and looking them up directly misses every single time
    and degenerates to a constant score for every image -- which surfaces as a
    plausible-looking 0.5000 balanced accuracy rather than as a crash. Always
    resolve through the config vocabulary.
    """
    key = str(raw_label).strip().lower()
    if key in label2idx:
        return label2idx[key]
    if key.startswith("label_"):  # HF default when id2label is absent
        try:
            return int(key[len("label_") :])
        except ValueError:
            return None
    try:  # some models really do return bare "0"/"1"
        return int(key)
    except ValueError:
        return None


def sanitize_model_name(model: str) -> str:
    """Turn a model id/path into a safe filename."""
    name = model.strip().rstrip("/").replace("/", "__").replace("\\", "__")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "model"


def load_labels(labels_csv: Path) -> list[dict]:
    with open(labels_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_image_path(data_dir: Path, filepath: str) -> Path:
    """Resolve a filepath relative to the data directory."""
    return (data_dir / filepath).resolve()


# ---------------------------------------------------------------------------
# transformers backend
# ---------------------------------------------------------------------------


def resolve_class_semantics(model, ai_label: str | None):
    """Decide which class index is 'AI-generated' for a transformers model.

    Returns (ai_idx, real_idx, mapping_description). Raises ValueError if the
    mapping cannot be determined and no override was given.
    """
    id2label = getattr(model.config, "id2label", None)
    label2id = getattr(model.config, "label2id", None)

    # Build ordered list of class names by output index (0..n-1).
    class_names: list[str] = []
    n_classes = getattr(model.config, "num_labels", None)

    if id2label:
        id2label = {int(k): str(v) for k, v in id2label.items()}
        if n_classes is None:
            n_classes = max(id2label) + 1 if id2label else None
        class_names = [id2label.get(i, "") for i in range(n_classes or 1)]
    elif label2id:
        # label2id maps name -> index; invert.
        ids = {}
        for name, idx in label2id.items():
            idx = int(idx)
            ids.setdefault(idx, []).append(str(name))
        n_classes = max([*ids] + [0]) + 1
        class_names = ["".join(ids.get(i, [])) for i in range(n_classes)]
    else:
        log.warning(
            "Model config has no id2label/label2id; falling back to "
            "the pipeline's returned label strings."
        )

    log.info("Resolved class names for %s: %s", type(model).__name__, class_names)

    if ai_label is not None:
        ai_label = str(ai_label)
        for i, name in enumerate(class_names):
            if str(name).strip().lower() == ai_label.strip().lower():
                log.info("Using explicit --ai-label = %r -> index %d", ai_label, i)
                return i, None, f"explicit ai_label={ai_label}"
        raise ValueError(
            f"--ai-label {ai_label!r} not found in model classes {class_names}"
        )

    if not class_names or all(not c for c in class_names):
        raise ValueError(
            "Model exposes no class-label names (no id2label/label2id). "
            "Cannot determine which output is 'AI-generated'. Pass "
            "--ai-label=<exact class string> if you know it."
        )

    scores = {i: _ai_score(c) for i, c in enumerate(class_names)}
    log.info("Class keyword scores: %s", scores)

    ai_idxs = [i for i, s in scores.items() if s > 0]
    real_idxs = [i for i, s in scores.items() if s < 0]

    if len(ai_idxs) == 1 and len(real_idxs) == 1 and ai_idxs[0] != real_idxs[0]:
        return ai_idxs[0], real_idxs[0], f"keyword: {class_names}"

    # Ambiguous / no strong keywords -> hard fail rather than guess.
    raise ValueError(
        "Could not uniquely identify the AI-generated output class from labels "
        f"{class_names!r} (scored {scores}). Candidate mapping is ambiguous.\n"
        "Inspect the model config (id2label) and rerun with "
        "--ai-label=<exact class string> (e.g. --ai-label='FAKE')."
    )


def run_transformers(
    model_id: str,
    paths: list[Path],
    image_ids: list[str],
    batch_size: int,
    device: str | None,
    ai_label: str | None,
) -> dict[str, float]:
    try:
        import torch  # noqa: F401  (ensures torch importable & device cuda works)
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "transformers/torch are not installed. Run: "
            "pip install -r requirements.txt"
        ) from exc

    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    pipe = pipeline("image-classification", model=model_id, **kwargs)
    model = pipe.model

    ai_idx, real_idx, mapping = resolve_class_semantics(model, ai_label)
    log.info("Class mapping decided: %s (ai_idx=%s real_idx=%s)", mapping, ai_idx, real_idx)

    label2idx = build_label_index(model.config)

    def resolve_idx(raw_label) -> int | None:
        return resolve_label_index(raw_label, label2idx)

    # Run in batches.
    out: dict[str, float] = {}
    unresolved_labels: set[str] = set()
    for start in range(0, len(paths), batch_size):
        chunk_paths = paths[start : start + batch_size]
        chunk_ids = image_ids[start : start + batch_size]
        results = pipe([str(p) for p in chunk_paths], top_k=None)
        for image_id, per_image in zip(chunk_ids, results):
            # per_image is a list of {"label": str, "score": float} dicts.
            probs: dict[int, float] = {}
            for item in per_image:
                idx = resolve_idx(item["label"])
                if idx is None:
                    unresolved_labels.add(str(item["label"]))
                    continue
                probs[idx] = float(item["score"])

            if ai_idx in probs:
                prob_ai = probs[ai_idx]
            elif real_idx is not None and real_idx in probs:
                # Binary model that only surfaced the real-side score.
                prob_ai = 1.0 - probs[real_idx]
            else:
                raise RuntimeError(
                    f"Neither the AI class (idx={ai_idx}) nor the real class "
                    f"(idx={real_idx}) appeared in the pipeline output for "
                    f"{image_id!r}. Resolved classes: {sorted(probs)}; "
                    f"unresolved labels: {sorted(unresolved_labels)}. "
                    "Rerun with --ai-label=<exact class string>."
                )
            out[image_id] = float(prob_ai)
        log.info("Processed %d/%d", min(start + batch_size, len(paths)), len(paths))

    if unresolved_labels:
        log.warning(
            "Some returned labels could not be mapped to a class index and "
            "were ignored: %s",
            sorted(unresolved_labels),
        )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


BACKENDS = {"transformers": run_transformers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--backend", choices=sorted(BACKENDS), default="transformers"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_DATA_DIR / "labels.csv",
        help="Ground-truth labels file (image_id,filepath,...).",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_PRED_DIR
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default=None, help="e.g. cpu, 0, cuda (transformers pipeline)"
    )
    parser.add_argument(
        "--ai-label",
        default=None,
        help="Explicit exact class label string for the AI-generated class.",
    )
    args = parser.parse_args()

    if not args.labels.is_file():
        log.error("Could not find %s. Run fetch_dataset.py first.", args.labels)
        return 1

    rows = load_labels(args.labels)
    if not rows:
        log.error("labels.csv is empty.")
        return 1
    log.info("Loaded %d labeled images from %s", len(rows), args.labels)

    paths = [resolve_image_path(args.data_dir, r["filepath"]) for r in rows]
    image_ids = [r["image_id"] for r in rows]
    missing = [str(p) for p, r in zip(paths, rows) if not p.is_file()]
    if missing:
        log.error(
            "%d image files are missing from data/ (e.g. %s). Re-run fetch_dataset.py.",
            len(missing),
            missing[0],
        )
        return 1

    runner = BACKENDS[args.backend]
    preds = runner(
        args.model, paths, image_ids, args.batch_size, args.device, args.ai_label
    )

    model_name = sanitize_model_name(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    log.info(
        "WROTE %s (%d predictions, mean prob=%.3f)",
        out_file,
        len(preds),
        sum(preds.values()) / len(preds) if preds else 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())