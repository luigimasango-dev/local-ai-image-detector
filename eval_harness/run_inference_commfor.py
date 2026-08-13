#!/usr/bin/env python3
"""Run the Community-Forensics detector over the eval set.

Community-Forensics (CVPR 2025, arXiv 2411.04125,
https://github.com/JeongsooP/Community-Forensics) is our primary detector
candidate. Its published weights (``OwensLab/commfor-model-224`` / ``-384``)
are NOT loadable as a plain ``transformers`` image-classification pipeline --
the model card was pushed with PyTorchModelHubMixin and points at the repo for
loading. The canonical load path is the repo's own model class:

    _commfor_src/models.py::class ViTClassifier (timm ViT + 1-out linear head)
    _commfor_src/eval_using_huggingface.ipynb::
        model = models.ViTClassifier.from_pretrained('OwensLab/commfor-model-224')

Facts read from the cloned source (do not "fix" without re-reading):

* Model : ``ViTClassifier`` (models.py:6) -- timm ViT-Small-patch16 + a
  ``nn.Linear(384 -> 1)`` head. Constructor kwargs come from the HF
  ``config.json`` (model_size/input_size/patch_size/freeze_backbone/device).
  The published ``config.json`` hardcodes ``"device": "cuda"``; on a CPU-only
  box this must be patched to ``cpu`` before ``from_pretrained``.
* Polarity : the head outputs a SINGLE logit and training uses
  ``nn.BCEWithLogitsLoss`` with ``real:0, fake:1`` (utils.py:evaluate_one_epoch
  applies ``torch.sigmoid`` then scores against the labels). Therefore
  ``P(AI) = sigmoid(logit)`` -- no class-index juggling, index 0 is the AI
  probability.
* Preprocessing (mode='test', dataloader.py:get_transform)::
      Resize(resize_size) -> CenterCrop(crop_size) -> ToTensor([0,1])
      -> Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) -> float32
  where ``determine_resize_crop_sizes`` maps input_size 224 -> (256, 224)
  and 384 -> (440, 384). Mean/std are ImageNet constants BECAUSE THE SOURCE
  USES THEM, not because we assumed so.

Output: ``predictions/<sanitized_model_name>.json`` -- flat dict
``image_id -> probability_ai_generated`` (float 0..1), the fixed contract
consumed by score.py / the downstream job. Predictions are written to disk
after every batch, so an interruption leaves partial output behind.

CLI
---
    python eval_harness/run_inference_commfor.py \
        --model OwensLab/commfor-model-224 \
        --device cpu --batch-size 16 [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_inference_commfor")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_PRED_DIR = SCRIPT_DIR / "predictions"
# Cloned repo (see task steps) + downloaded weights live under the project root.
CLONE_DIR = (SCRIPT_DIR.parent / "_commfor_src").resolve()
CACHE_ROOT = (SCRIPT_DIR.parent / "_commfor_cache").resolve()

# Standard ImageNet normalization -- verbatim from dataloader.py:282-283.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def sanitize_model_name(model: str) -> str:
    """Turn a model id/path into a safe filename (matches run_inference.py)."""
    name = model.strip().rstrip("/").replace("/", "__").replace("\\", "__")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "model"


def load_labels(labels_csv: Path) -> list[dict]:
    with open(labels_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_image_path(data_dir: Path, filepath: str) -> Path:
    return (data_dir / filepath).resolve()


def select_rows(rows: list[dict], limit: int | None) -> list[dict]:
    """Pick which rows to run.

    labels.csv is grouped: all 600 real rows first, then all 600 AI rows. A
    strict "first N" slice would therefore never touch an AI image for N <= 600
    and would make the polarity check meaningless. ``--limit N`` instead takes
    the first N/2 rows of each class (row order preserved, deterministic) so a
    fast smoke test always sees both classes. With no limit, every row runs.
    """
    if limit is None:
        return rows
    if limit <= 0:
        raise ValueError("--limit must be a positive integer (or omitted)")
    real = [r for r in rows if r["label"] == "0"]
    fake = [r for r in rows if r["label"] == "1"]
    n_real = (limit + 1) // 2
    n_fake = limit // 2
    picked = real[:n_real] + fake[:n_fake]
    log.info(
        "limit=%d -> %d rows (%d label=0, %d label=1); labels.csv is grouped "
        "real-then-AI so a plain 'first N' slice would not cover the AI class.",
        limit,
        len(picked),
        len([r for r in picked if r["label"] == "0"]),
        len([r for r in picked if r["label"] == "1"]),
    )
    return picked


def download_and_prepare_model(model_id: str, device: str) -> Path:
    """Download weights + config into a local cache and make it CPU-loadable.

    Returns the local dir ready for ``ViTClassifier.from_pretrained(dir)``.
    The published config.json hardcodes ``"device": "cuda"``; from_pretrained
    forwards config keys straight into the model constructor, so on a CPU-only
    machine we patch our local copy of the config before loading. The patch
    only ever touches files under CACHE_ROOT (never the HF hub cache).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Run: pip install huggingface_hub"
        ) from exc

    cache_dir = CACHE_ROOT / sanitize_model_name(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", model_id, cache_dir)
    cfg_path = hf_hub_download(model_id, "config.json", local_dir=str(cache_dir))
    hf_hub_download(model_id, "model.safetensors", local_dir=str(cache_dir))

    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    if cfg.get("device") != device:
        old = cfg.get("device")
        cfg["device"] = device
        Path(cfg_path).write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
        log.info("Patched model config device %r -> %r (%s)", old, device, cfg_path)
    return cache_dir


def build_transform(input_size: int):
    """Replicate dataloader.py:get_transform(mode='test') exactly.

    resize/crop sizes come from determine_resize_crop_sizes (dataloader.py:269):
    224 -> (256, 224), 384 -> (440, 384). ``ctrans.ToTensor_range(0, 1)`` is
    identity on top of torchvision's to_tensor (uint8 -> float32 in [0,1]),
    and ``ConvertImageDtype(float32)`` is a no-op after it, so the equivalent
    torchvision pipeline is Resize -> CenterCrop -> ToTensor -> Normalize.
    """
    from torchvision import transforms

    if input_size == 224:
        resize_size, crop_size = 256, 224
    elif input_size == 384:
        resize_size, crop_size = 440, 384
    else:
        raise ValueError(f"Unsupported input_size {input_size}; expected 224 or 384")

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


def load_model(model_id: str, device: str):
    """Load ViTClassifier.from_pretrained from the (patched) local copy."""
    if not CLONE_DIR.is_dir():
        raise SystemExit(
            f"Cloned source not found at {CLONE_DIR}. Clone the "
            "Community-Forensics repo into ./_commfor_src/ first."
        )
    sys.path.insert(0, str(CLONE_DIR))
    try:
        import torch  # noqa: F401  (timm/torch import side effects)
        import models
    except ImportError as exc:
        raise SystemExit(
            "Could not import the repo's models.py. Required deps: timm, "
            "huggingface_hub, safetensors, torch. Run: pip install timm"
        ) from exc

    model_dir = download_and_prepare_model(model_id, device)
    t0 = time.time()
    model = models.ViTClassifier.from_pretrained(str(model_dir))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded %s in %.1fs (%.1fM params)", model_id, time.time() - t0, n_params / 1e6)
    return model, model_dir


def run_inference(
    model,
    model_dir: Path,
    rows: list[dict],
    data_dir: Path,
    batch_size: int,
    device: str,
    out_file: Path,
) -> tuple[dict[str, float], list[str]]:
    """Run batches; write predictions to disk after each batch.

    Returns (predictions, skipped_image_ids). Corrupt/unreadable images are
    skipped and reported rather than aborting the run.
    """
    import torch

    transform = build_transform(
        json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["input_size"]
    )
    paths = [resolve_image_path(data_dir, r["filepath"]) for r in rows]
    image_ids = [r["image_id"] for r in rows]
    labels = [r["label"] for r in rows]

    preds: dict[str, float] = {}
    skipped: list[str] = []
    n_failed_batches = 0
    t_start = time.time()

    def flush():
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_file.with_suffix(out_file.suffix + ".tmp")
        tmp.write_text(
            json.dumps(preds, indent=2), encoding="utf-8"
        )
        os.replace(tmp, out_file)

    for start in range(0, len(rows), batch_size):
        chunk = list(zip(image_ids, paths, labels))[start : start + batch_size]

        batch_ok: list[tuple[str, int]] = []
        batch_tensors = []
        for image_id, path, label in chunk:
            try:
                import PIL.Image

                img = PIL.Image.open(path).convert("RGB")
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                log.warning("Skipping unreadable image %s: %s", image_id, exc)
                skipped.append(image_id)
                continue
            batch_ok.append((image_id, int(label)))
            batch_tensors.append(transform(img))

        if not batch_tensors:
            flush()  # keep whatever we have on disk
            continue

        x = torch.stack(batch_tensors)
        with torch.no_grad():
            try:
                logits = model(x.to(device))
            except Exception as exc:  # noqa: BLE001
                log.error("Inference failed for batch %d: %s", start // batch_size, exc)
                n_failed_batches += 1
                flush()
                continue
        probs = torch.sigmoid(logits).reshape(-1).tolist()

        for (image_id, _label), prob in zip(batch_ok, probs):
            preds[image_id] = float(prob)

        flush()
        log.info(
            "Batch %d/%d done (%d preds so far, %d skipped)",
            start // batch_size + 1,
            (len(rows) + batch_size - 1) // batch_size,
            len(preds),
            len(skipped),
        )

    log.info(
        "Inference finished in %.1fs: %d predictions, %d skipped, "
        "%d failed batches",
        time.time() - t_start,
        len(preds),
        len(skipped),
        n_failed_batches,
    )
    return preds, skipped


def polarity_check(preds: dict[str, float], rows: list[dict]) -> None:
    """Print mean P(AI) for covered label=1 vs label=0 rows.

    The whole point of this printout is to catch an inverted model output.
    The label=1 mean MUST be above the label=0 mean; if it is not, we say so
    loudly rather than silently flipping.
    """
    covered = [r for r in rows if r["image_id"] in preds]
    if not covered:
        print("POLARITY CHECK: no covered rows to compare.", flush=True)
        return
    fake = [preds[r["image_id"]] for r in covered if r["label"] == "1"]
    real = [preds[r["image_id"]] for r in covered if r["label"] == "0"]
    m_fake = sum(fake) / len(fake) if fake else float("nan")
    m_real = sum(real) / len(real) if real else float("nan")
    print(
        f"POLARITY CHECK ({len(fake)} label=1, {len(real)} label=0): "
        f"mean P(AI) on label=1: {m_fake:.3f} | on label=0: {m_real:.3f}",
        flush=True,
    )
    if fake and real:
        if m_fake < m_real:
            print(
                "!!! POLARITY INVERTED: label=1 mean is BELOW label=0 mean. "
                "The predictions file may still be written, but do NOT trust "
                "it until the mapping is fixed. We are NOT silently flipping it.",
                flush=True,
            )
        else:
            print("Polarity OK: label=1 scores above label=0.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="OwensLab/commfor-model-224",
        help="HF model id (default OwensLab/commfor-model-224).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for the model (cpu, cuda).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only a balanced N-image smoke subset (N/2 per class; "
             "labels.csv is grouped real-then-AI).",
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PRED_DIR)
    args = parser.parse_args()

    if not args.labels.is_file():
        log.error("Could not find %s. Run fetch_dataset.py first.", args.labels)
        return 1

    rows = load_labels(args.labels)
    if not rows:
        log.error("labels.csv is empty.")
        return 1
    rows = select_rows(rows, args.limit)
    log.info(
        "Loaded %d labeled images from %s (total rows in file: %d)",
        len(rows),
        args.labels,
        len(load_labels(args.labels)),
    )

    missing = [
        str(resolve_image_path(args.data_dir, r["filepath"]))
        for r in rows
        if not resolve_image_path(args.data_dir, r["filepath"]).is_file()
    ]
    if missing:
        log.error(
            "%d image files are missing from data/ (e.g. %s). Re-run fetch_dataset.py.",
            len(missing),
            missing[0],
        )
        return 1

    model, model_dir = load_model(args.model, args.device)

    out_file = args.output_dir / f"{sanitize_model_name(args.model)}.json"
    preds, skipped = run_inference(
        model,
        model_dir,
        rows,
        args.data_dir,
        args.batch_size,
        args.device,
        out_file,
    )

    polarity_check(preds, rows)

    print(
        f"WROTE {out_file} ({len(preds)} predictions; skipped "
        f"{len(skipped)} unreadable image(s): {skipped[:10]}"
        f"{' ...' if len(skipped) > 10 else ''})"
    )
    if skipped:
        log.warning("Skipped %d unreadable image(s): %s", len(skipped), skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
