#!/usr/bin/env python3
"""Fetch a labeled real-vs-AI image evaluation set and write data/labels.csv.

Primary source: ``aidetectarena/ai-image-detector-benchmark`` on Hugging Face.

IMPORTANT FINDING (verified 2026-08-13): the Hugging Face repository
``aidetectarena/ai-image-detector-benchmark`` is EMPTY (the dataset card says
"The dataset is currently empty"). Its card links to the canonical release
located at:

    https://aidetectarena.com/datasets/v0.1
    (CDN mirror: https://aidetectarena-benchmark.nyc3.cdn.digitaloceanspaces.com/datasets-archive/benchmark-v0.1.zip)
    (GitHub metadata/eval: https://github.com/AI-Detect-Arena/benchmark-dataset)

This script therefore (1) attempts to load the HF dataset with the ``datasets``
library, (2) detects that it is empty/fails, (3) logs it plainly, and (4) falls
back to the CDN mirror zip, which is the actual v0.1 release of the same
dataset (2,038 images: 1,018 AI + 1,020 real, 17 generators, 6 categories).

Because the full archive is ~1.4 GiB, we do NOT download the whole zip. We
fetch the zip's central directory (a ~4 MB tail range request) and then use
HTTP Range requests to download only the image files we keep, so the bytes
transferred are close to the size of the sampled images themselves.

Optional supplement: ``--include-cifake`` pulls a small balanced slice from
``yanbax/CIFAKE_autotrain_compatible`` (Stable Diffusion v1.4-era synthetic
images + real CIFAR-10 photos, 32x32 px) via the Hugging Face ``datasets``
library. This adds older-generation / lower-resolution generator coverage that
the modern AIDetectArena set lacks. Off by default.

Output
------
- Images are written under ``data/images/...``
- ``data/labels.csv`` with columns: ``image_id,filepath,label,generator``
  where label = 1 (AI-generated) or 0 (real photograph), and filepath is
  relative to the ``data/`` directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_dataset")

HF_PRIMARY = "aidetectarena/ai-image-detector-benchmark"

CDN_ZIP_URL = (
    "https://aidetectarena-benchmark.nyc3.cdn.digitaloceanspaces.com/"
    "datasets-archive/benchmark-v0.1.zip"
)
# ZIP layout: every real file lives under benchmark-v0.1/...
ZIP_ROOT = "benchmark-v0.1"
METADATA_IN_ZIP = f"{ZIP_ROOT}/metadata/images_metadata.csv"

CIFAKE_HF = "yanbax/CIFAKE_autotrain_compatible"
CIFAKE_TARGET_GPU = 1  # HF CIFAKE label value for GPU/Fake (AI). See README.

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "eval-harness/1.0 (internal proxy benchmark)",
        "Accept-Encoding": "identity",
    }
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class DownloadError(RuntimeError):
    pass


def head_length(url: str) -> int:
    resp = SESSION.head(url, allow_redirects=True, timeout=60)
    resp.raise_for_status()
    length = int(resp.headers.get("Content-Length", "0"))
    if not length:
        raise DownloadError(f"HEAD {url} returned no Content-Length")
    return length


def get_bytes(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    """Fetch bytes [start, end] inclusive via HTTP Range."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(3):
        try:
            resp = SESSION.get(url, headers=headers, timeout=timeout)
            if resp.status_code in (416,):
                raise DownloadError(f"Range {start}-{end} unsatisfiable (416)")
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            if attempt == 2:
                raise DownloadError(f"GET range {start}-{end}: {exc}") from exc
            time.sleep(1.0 + attempt)
    raise AssertionError  # unreachable


# ---------------------------------------------------------------------------
# Remote ZIP reading (central directory + per-file range extraction)
# ---------------------------------------------------------------------------


def read_zip_central_directory(url: str) -> list[dict]:
    """Parse EOCD + central directory from the tail of the zip."""
    total = head_length(url)
    tail_len = min(8_000_000, total)
    tail = get_bytes(url, total - tail_len, total - 1, timeout=180)

    eocd_off = tail.rfind(b"\x50\x4b\x05\x06")
    if eocd_off < 0:
        raise DownloadError("End-of-central-directory signature not found")
    cd_size = int.from_bytes(tail[eocd_off + 12 : eocd_off + 16], "little")
    cd_start_in_tail = eocd_off - cd_size
    if cd_start_in_tail < 0:
        raise DownloadError("Central directory extends before fetched tail")
    cd = tail[cd_start_in_tail:eocd_off]

    entries: list[dict] = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"\x50\x4b\x01\x02":
            break
        csize, fsize = struct.unpack("<II", cd[pos + 20 : pos + 28])
        namelen, extralen, commentlen = struct.unpack(
            "<HHH", cd[pos + 28 : pos + 34]
        )
        lhoff = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        name = cd[pos + 46 : pos + 46 + namelen].decode("utf-8", "replace")
        entries.append(
            {
                "name": name,
                "lhoff": lhoff,
                "csize": csize,
                "fsize": fsize,
            }
        )
        pos += 46 + namelen + extralen + commentlen
    return entries


def read_member(url: str, entry: dict) -> bytes:
    """Read a zip member's uncompressed bytes via two range requests."""
    lhoff = entry["lhoff"]
    csize = entry["csize"]
    fsize = entry["fsize"]

    # Local file header (30 fixed bytes) -> name len + extra len
    lh = get_bytes(url, lhoff, lhoff + 29)
    if lh[:4] != b"\x50\x4b\x03\x04":
        raise DownloadError(f"Bad local header for {entry['name']}")
    method = struct.unpack("<H", lh[8:10])[0]
    name_len, extra_len = struct.unpack("<HH", lh[26:30])
    data_start = lhoff + 30 + name_len + extra_len
    data_end = data_start + csize - 1
    comp = get_bytes(url, data_start, data_end)

    if method == 0:  # stored
        raw = comp
    elif method == 8:  # deflate
        raw = zlib.decompress(comp, -15)
    else:
        raise DownloadError(
            f"Unsupported compression method {method} for {entry['name']}"
        )
    if len(raw) != fsize:
        raise DownloadError(
            f"Size mismatch for {entry['name']}: got {len(raw)}, "
            f"expected {fsize}"
        )
    return raw


# ---------------------------------------------------------------------------
# Sampling & download
# ---------------------------------------------------------------------------


def sample_ids(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Deterministically select `count` rows, stratified by category."""
    if count <= 0:
        return []
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    rng = random.Random(seed)
    for group in by_cat.values():
        rng.shuffle(group)
    picked: list[dict] = []
    cats = list(by_cat)
    rng.shuffle(cats)
    idx = {c: 0 for c in cats}
    while len(picked) < count:
        added = False
        for c in cats:
            if len(picked) >= count:
                break
            if idx[c] < len(by_cat[c]):
                picked.append(by_cat[c][idx[c]])
                idx[c] += 1
                added = True
        if not added:
            break  # ran out of images
    return picked


def derive_generator(meta: dict, label: int) -> str:
    """Best available generator name for a metadata row.

    The upstream ``generator`` column is lossy: it truncates multi-word names
    ("Sd" for SD 3.5 Large, "Gpt" for GPT Image 1.5, "Z" for Z Image, "Flux"
    for Flux 2 Flex) and, worse, collapses Seedream v3 and v4 into a single
    "Seedream" bucket -- which is why it shows 16 generators where the dataset
    documents 17. The filename carries the exact name, so prefer it:

        images/ai/art/seedream_v4_art_08.png  ->  seedream_v4

    Falls back to the metadata column if the filename does not parse.
    """
    if label == 0:
        return "unsplash_lite"

    filename = (meta.get("filename") or "").rsplit("/", 1)[-1]
    category = (meta.get("category") or "").strip()
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    marker = f"_{category}_"
    if category and marker in stem:
        prefix = stem.split(marker)[0].strip()
        if prefix:
            return prefix

    return (meta.get("generator") or "").strip() or "unknown"


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "filepath", "label", "generator"]
        )
        writer.writeheader()
        writer.writerows(rows)


def zip_path_for(metadata_filename: str) -> str:
    return f"{ZIP_ROOT}/{metadata_filename}"


def fetch_from_zip(
    data_dir: Path,
    ai_rows: list[dict],
    real_rows: list[dict],
    ai_count: int,
    real_count: int,
    seed: int,
    url: str = CDN_ZIP_URL,
) -> list[dict]:
    """Download sampled images from the remote zip and return labels rows."""
    entries = read_zip_central_directory(url)
    by_path = {e["name"]: e for e in entries if e["fsize"] > 0}

    available_ai = [r for r in ai_rows if zip_path_for(r["filename"]) in by_path]
    available_real = [
        r for r in real_rows if zip_path_for(r["filename"]) in by_path
    ]
    if len(available_ai) != len(ai_rows):
        log.warning(
            "%d of %d AI metadata rows have no file in the zip; dropping",
            len(ai_rows) - len(available_ai),
            len(ai_rows),
        )
    if len(available_real) != len(real_rows):
        log.warning(
            "%d of %d real metadata rows have no file in the zip; dropping",
            len(real_rows) - len(available_real),
            len(real_rows),
        )

    ai_pick = sample_ids(available_ai, ai_count, seed)
    real_pick = sample_ids(available_real, real_count, seed + 1)
    log.info(
        "Sampling %d AI + %d real images (stratified by category, seed=%d)",
        len(ai_pick),
        len(real_pick),
        seed,
    )

    labels: list[dict] = []
    tasks: list[tuple[str, dict]] = []
    for meta in ai_pick + real_pick:
        label = 1 if meta["is_ai"] == "true" else 0
        filepath = meta["filename"]  # e.g. images/ai/art/qwen_2512_art_08.png
        # Reject path traversal just in case (metadata is trusted but cheap to check).
        rel = Path(filepath.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise DownloadError(f"Unsafe filepath in metadata: {filepath}")
        out_path = data_dir / rel
        label_row = {
            "image_id": meta["id"],
            "filepath": str(rel).replace("\\", "/"),
            "label": label,
            "generator": derive_generator(meta, label),
        }
        labels.append(label_row)
        tasks.append((str(out_path), by_path[zip_path_for(meta["filename"])]))

    def download_one(args: tuple[str, dict]) -> tuple[str, bool]:
        out_path, entry = args
        # Skip if already downloaded with the expected size.
        if os.path.isfile(out_path) and os.path.getsize(out_path) == entry["fsize"]:
            return out_path, True
        raw = read_member(url, entry)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, out_path)
        return out_path, True

    downloaded = skipped = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_one, t): t for t in tasks}
        for fut in as_completed(futures):
            out_path, _ = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                log.error("FAILED %s: %s", out_path, exc)
                continue
            if os.path.isfile(out_path):
                downloaded += 1
            else:
                skipped += 1
            if downloaded % 200 == 0 and downloaded:
                log.info("Downloaded %d images so far...", downloaded)

    log.info("Downloaded %d images (%d already cached).", downloaded, skipped)
    return labels


def load_zip_metadata(url: str) -> tuple[list[dict], list[dict]]:
    """Read images_metadata.csv out of the CDN zip and split AI/real."""
    entries = read_zip_central_directory(url)
    meta_entry = next((e for e in entries if e["name"] == METADATA_IN_ZIP), None)
    if meta_entry is None:
        raise DownloadError(f"{METADATA_IN_ZIP} not found in {url}")
    raw = read_member(url, meta_entry)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    rows = list(reader)
    log.info(
        "Metadata loaded: %d rows (AI=%d, real=%d) from %s",
        len(rows),
        sum(1 for r in rows if r["is_ai"] == "true"),
        sum(1 for r in rows if r["is_ai"] == "false"),
        METADATA_IN_ZIP,
    )
    ai = [r for r in rows if r["is_ai"] == "true"]
    real = [r for r in rows if r["is_ai"] == "false"]
    return ai, real


def try_hf_primary():
    """Attempt the `datasets`-library load; returns None if empty/fails."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("`datasets` library not installed; skipping HF attempt.")
        return None
    log.info("Trying Hugging Face dataset: %s", HF_PRIMARY)
    try:
        ds = load_dataset(HF_PRIMARY)
    except Exception as exc:  # noqa: BLE001
        log.warning("HF load_dataset(%s) raised: %s", HF_PRIMARY, exc)
        return None
    sizes = {k: len(v) for k, v in ds.items()}
    log.info("HF dataset splits/sizes: %s", sizes)
    if not any(n for n in sizes.values()):
        log.warning(
            "HF dataset %s is EMPTY (verified). Falling back to the "
            "v0.1 release zip on the CDN mirror that the dataset card links to.",
            HF_PRIMARY,
        )
        return None
    return ds


# ---------------------------------------------------------------------------
# CIFAKE supplement
# ---------------------------------------------------------------------------


def fetch_cifake_supplement(
    data_dir: Path,
    per_class: int,
    seed: int,
) -> list[dict]:
    """Pull a small balanced CIFAKE slice via the `datasets` library."""
    from datasets import load_dataset

    log.info("Loading CIFAKE supplement from %s", CIFAKE_HF)
    ds = load_dataset(CIFAKE_HF, split="train", streaming=False)
    features = ds.features.get("label")
    names = getattr(features, "names", None)
    log.info("CIFAKE label feature names: %s", names)

    # Determine AI (FAKE) vs real class ids heuristically.
    fake_ids: list[int] = []
    real_ids: list[int] = []
    if names:
        for i, n in enumerate(names):
            low = str(n).lower()
            if any(k in low for k in ("fake", "ai", "gpu", "generated", "synthetic", "1real", "0real")):
                fake_ids.append(i)
            else:
                real_ids.append(i)
    if not fake_ids or not real_ids:
        # CIFAKE convention: 0 = FAKE (AI), 1 = REAL
        fake_ids, real_ids = [CIFAKE_TARGET_GPU], [1 - CIFAKE_TARGET_GPU]
    log.info("CIFAKE using fake_ids=%s real_ids=%s", fake_ids, real_ids)

    fake_rows, real_rows = [], []
    for i, ex in enumerate(ds):
        lab = ex["label"]
        if lab in fake_ids and len(fake_rows) < per_class:
            fake_rows.append((i, ex["image"]))
        elif lab in real_ids and len(real_rows) < per_class:
            real_rows.append((i, ex["image"]))
        if len(fake_rows) >= per_class and len(real_rows) >= per_class:
            break
    log.info("CIFAKE selected %d fake + %d real crops", len(fake_rows), len(real_rows))

    out_dir = data_dir / "images" / "cifake"
    labels: list[dict] = []
    for idx, (orig_index, img) in enumerate(fake_rows):
        path = out_dir / "ai" / f"cifake_{orig_index:06d}.jpg"
        img.save(path)
        labels.append(
            {
                "image_id": f"cifake_{orig_index:06d}",
                "filepath": str(path.relative_to(data_dir)).replace("\\", "/"),
                "label": 1,
                "generator": "stable_diffusion_v1.4_cifake",
            }
        )
    for idx, (orig_index, img) in enumerate(real_rows):
        path = out_dir / "real" / f"cifake_{orig_index:06d}.jpg"
        img.save(path)
        labels.append(
            {
                "image_id": f"cifake_{orig_index:06d}",
                "filepath": str(path.relative_to(data_dir)).replace("\\", "/"),
                "label": 0,
                "generator": "cifar10_real",
            }
        )
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument("--ai-count", type=int, default=600, help="AI images to pull")
    parser.add_argument(
        "--real-count", type=int, default=600, help="real images to pull"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-cifake",
        action="store_true",
        help="Add a small CIFAKE (older SD, 32x32) supplement slice.",
    )
    parser.add_argument(
        "--cifake-per-class", type=int, default=200, dest="cifake_per_class"
    )
    parser.add_argument(
        "--url",
        default=CDN_ZIP_URL,
        help="CDN mirror URL for the AIDetectArena v0.1 zip.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    all_labels: list[dict] = []
    seen_ids: set[str] = set()

    def add(labels: list[dict]) -> None:
        for row in labels:
            if row["image_id"] in seen_ids:
                log.warning("Duplicate image_id %s; skipping", row["image_id"])
                continue
            seen_ids.add(row["image_id"])
            all_labels.append(row)

    attempt_hf = try_hf_primary()

    ai_rows, real_rows = load_zip_metadata(args.url)
    # Cap requested counts at what is actually available.
    ai_count = min(args.ai_count, len(ai_rows))
    real_count = min(args.real_count, len(real_rows))
    labels = fetch_from_zip(
        data_dir,
        ai_rows,
        real_rows,
        ai_count,
        real_count,
        args.seed,
        url=args.url,
    )
    add(labels)

    cifake_labels: list[dict] = []
    if args.include_cifake:
        try:
            cifake_labels = fetch_cifake_supplement(
                data_dir, args.cifake_per_class, args.seed
            )
        except Exception as exc:  # noqa: BLE001
            log.error("CIFAKE supplement failed: %s", exc)
            log.error("Proceeding WITHOUT CIFAKE; primary set is intact.")
        add(cifake_labels)

    if len(all_labels) < 2:
        log.error("No labels produced — dataset fetch failed hard.")
        return 1

    all_labels.sort(
        key=lambda r: (r["label"] == 1, r["image_id"])
    )  # real first, then AI
    labels_csv = data_dir / "labels.csv"
    write_csv(labels_csv, all_labels)

    n_ai = sum(1 for r in all_labels if r["label"] == 1)
    n_real = len(all_labels) - n_ai
    log.info("WROTE %s", labels_csv)
    log.info(
        "Total rows: %d (AI=%d, real=%d) | generator values: %s",
        len(all_labels),
        n_ai,
        n_real,
        sorted({r["generator"] for r in all_labels}),
    )

    # Convenience manifest for downstream jobs.
    manifest = {
        "primary": {
            "name": "aidetectarena/ai-image-detector-benchmark (v0.1)",
            "hf_repo_empty": attempt_hf is None,
            "source": "https://aidetectarena.com/datasets/v0.1",
            "zip_url": args.url,
            "total_available": {"ai": len(ai_rows), "real": len(real_rows)},
        },
        "sampled": {"ai": n_ai, "real": n_real, "total": len(all_labels)},
        "cifake_added": bool(cifake_labels),
        "cifake_per_class": args.cifake_per_class,
        "seed": args.seed,
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("WROTE %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())