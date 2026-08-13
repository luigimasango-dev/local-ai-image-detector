#!/usr/bin/env python3
"""Build the "real diverse" false-positive eval set from Wikimedia Commons.

Our existing real eval set is 100% Unsplash Lite professional photography and
the detector false-positives on 8% of it overall, 14.3% on portraits. The
bounty benchmark scores against "web-realistic samples", so we need many kinds
of *non-AI* image that a browsing extension actually encounters -- including
ones that *look* synthetic without being AI-generated (screenshots, charts,
logos, scans). This script pulls free-licensed images from Wikimedia Commons
(https://commons.wikimedia.org/w/api.php) across six classes and writes them to
``eval_harness/data/real_diverse/``.

Classes (generator values in labels.csv):
    amateur_photo    ordinary snapshots, badly lit, noisy, handheld
    screenshot       software / web-page UI screenshots            (HIGH RISK)
    diagram_chart    charts, graphs, infographics, technical diagrams (HIGH RISK)
    logo_graphic     logos / flat vector-ish raster graphics       (HIGH RISK)
    scanned_document scanned pages, historical documents, maps
    low_quality_web  heavily compressed or low-resolution photographs

Guards (the whole point of the set is measuring false positives, so a
contaminated "real" image would silently corrupt the measurement):
    * Free license only: public domain / CC0 / CC-BY / CC-BY-SA. NC/ND, GFDL
      and everything else is SKIPPED (checked per image).
    * >= 128x128 px (matches what the extension processes) and <= 10 MB.
    * Explicitly excludes AI-generated files: any title / description /
      artist / categories containing AI markers ("AI-generated", "Stable
      Diffusion", "Midjourney", "DALL-E", "text-to-image", ...) is skipped.
      If we cannot verify an image is non-AI we skip it.
    * All output is converted to RGB JPEG for consistency.

Output (written incrementally so an interruption leaves partial state):
    real_diverse/labels.csv       image_id,filepath,label,generator
                                 (label is ALWAYS 0; generator is the class name;
                                  filepath is relative to real_diverse/)
    real_diverse/attribution.csv  image_id,source_url,author,license
    real_diverse/manifest.json    per-class counts, license histogram, sizes

Re-running resumes where it left off: completed image_ids are recovered from
labels.csv (files that still exist) and their Commons titles are recovered from
the encoded source URLs in attribution.csv, so already-fetched files are not
re-downloaded and the per-class target is decremented accordingly.

Usage:
    python fetch_real_diverse.py [--per-class 50] [--interval 0.3] [--seed 7]

Politeness: every API request and every file download waits at least
``--interval`` seconds behind the previous one and retries 429 responses with a
back-off (honoring Retry-After when present). A descriptive User-Agent is set.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from PIL import Image, UnidentifiedImageError

API = "https://commons.wikimedia.org/w/api.php"
UA = (
    "local-ai-image-detector-evalharness/1.0 "
    "(repository: local-ai-image-detector; maintainer: repo-local; "
    "contact: repo issues)"
)

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "real_diverse"

MIN_DIM = 128          # extension ignores smaller images
MAX_BYTES = 10_000_000 # ~10 MB originals are plenty for eval
MAX_DIM = 8_000        # guard against unpixellable behemoths (memory)
OUT_MAX_SIDE = 2048    # downscale large originals so output stays web-sized
MAX_CAND = 400         # per-class candidate cap before filtering
CAND_PAGES = 3         # category member pages (<=500 files each) to walk

CLASS_ORDER = [
    "amateur_photo",
    "screenshot",
    "diagram_chart",
    "logo_graphic",
    "scanned_document",
    "low_quality_web",
]

# (type, weekday) sources per class; `type` is "cat" (category) or "search".
SOURCES: dict[str, list[tuple[str, str]]] = {
    "amateur_photo": [
        ("cat", "Category:Amateur photographs"),
        ("cat", "Category:Photographs by amateurs"),
        ("cat", "Category:Snapshots"),
        ("cat", "Category:Mobile phone photographs"),
        ("cat", "Category:Taken with mobile phones"),
        ("search", 'amateur snapshot photo filetype:bitmap'),
        ("search", 'holiday snapshot family filetype:bitmap'),
    ],
    "screenshot": [
        ("cat", "Category:Screenshots"),
        ("cat", "Category:Screenshots of software"),
        ("cat", "Category:Screenshots of web browsers"),
        ("search", 'screenshot world wide web page filetype:bitmap'),
    ],
    "diagram_chart": [
        ("cat", "Category:Charts"),
        ("cat", "Category:Diagrams"),
        ("cat", "Category:Infographics"),
        ("cat", "Category:Flowcharts"),
        ("cat", "Category:Technical diagrams"),
        ("search", 'bar chart data visualization filetype:bitmap'),
    ],
    "logo_graphic": [
        ("cat", "Category:Logos"),
        ("cat", "Category:Icons"),
        ("cat", "Category:Logo designs"),
        ("cat", "Category:Sports logos"),
        ("search", 'flat logo raster png filetype:bitmap'),
    ],
    "scanned_document": [
        ("cat", "Category:Scanned texts"),
        ("cat", "Category:Scanned documents"),
        ("cat", "Category:Historical documents"),
        ("cat", "Category:Manuscripts"),
        ("cat", "Category:Old maps"),
        ("search", 'scanned page document filetype:bitmap'),
    ],
    "low_quality_web": [
        ("search", 'low resolution photo filetype:bitmap'),
        ("search", 'poor quality photograph filetype:bitmap'),
        ("search", 'blurry photo filetype:bitmap'),
        ("search", 'small thumbnail photo filetype:bitmap'),
        ("cat", "Category:Low resolution images"),
    ],
}

# ---------------------------------------------------------------------------
# AI-contamination filter. If any scanned text hits these, we skip the file.
# Being conservative is correct: a contaminated "real" set silently corrupts
# the false-positive measurement, which is the entire point of this exercise.
# ---------------------------------------------------------------------------
AI_RE = re.compile(
    r"""\bai[- ]?generated\b
    |\bgenerated\s+image\b
    |\bgenerated\s+by\s+(an\s+)?(ai|artificial\s+intelligence)\b
    |\bstable[- ]?diffusion\b
    |\bmidjourney\b
    |\bdall[-· ]?e\b
    |\bopenai\b
    |\btext[- ]?to[- ]?image\b
    |\bcraiyon\b
    |\bnightcaf[ée]\b
    |\bleonardo\s+ai\b
    |\bbing\s+image\s+creator\b
    |\bgenerative\s+(ai|image|imagery)\b
    |\bdiffusion\s+model\b
    |\bsynthetic\s+image\b
    |\bai[- ]?(art|image|imagery|photo)\b
    """,
    re.I | re.X,
)
# AI-related Commons categories a file must not be (checked against the file's
# own categories, extracted from extmetadata).
AI_CATEGORY_RE = re.compile(
    r"""ai[- ]?generated|stable[- ]?diffusion|midjourney|dall[-· ]?e|
    ai[- ]?art|ai[- ]?image|generated[- ]?imagery|text[- ]?to[- ]?image|
    generative|synthetic[- ]?image|deep[- ]?fake|openai""",
    re.I | re.X,
)

# ---------------------------------------------------------------------------
# License filter: only PD / CC0 / CC-BY / CC-BY-SA.
# ---------------------------------------------------------------------------
LICENSE_REJECT_RE = re.compile(
    r"\bnc\b|\bnd\b|noderiv|no\s+derivative|noncommercial|non[- ]commercial|gfdl",
    re.I,
)
LICENSE_ACCEPT_RE = re.compile(
    r"public\s+domain|\bcc0\b|\bcc\s+by\b|\bcc-by\b|attribution|"
    r"pd[-_\s](self|us|usgov|in|art|old|canada|uk|eu|text|ancient|1923|1996|scan|flag|china)|"
    r"^\s*pd\b",
    re.I,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_real_diverse")

# ---------------------------------------------------------------------------
# HTTP / rate limiting
# ---------------------------------------------------------------------------


class Throttle:
    """Min-gap throttle shared by every API call and download."""

    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = now - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.monotonic()


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": UA,
        "Accept-Encoding": "identity",
    }
)
RL = Throttle(0.3)


def api_get(params: dict, retries: int = 8) -> dict:
    """GET the Commons API, throttled, with 429/error back-off."""
    last = None
    for attempt in range(retries):
        RL.wait()
        try:
            resp = SESSION.get(API, params=params, timeout=90)
        except requests.RequestException as exc:
            last = exc
            log.warning("request error (%s); retry %d", exc, attempt + 1)
            time.sleep(5 + attempt * 2)
            continue
        if resp.status_code == 429:
            delay = int(resp.headers.get("Retry-After", "30") or 30)
            log.warning(
                "HTTP 429 (rate limit); backing off %ss (try %d/%d)",
                delay, attempt + 1, retries,
            )
            time.sleep(min(delay, 60))
            continue
        if resp.status_code != 200:
            log.warning(
                "HTTP %d from API; retry %d", resp.status_code, attempt + 1
            )
            time.sleep(5 + attempt * 2)
            continue
        try:
            return resp.json()
        except ValueError:
            log.warning("non-JSON API body; retry %d", attempt + 1)
            time.sleep(5 + attempt * 2)
    raise RuntimeError(
        f"API request failed after {retries} attempts: {params}. Last: {last}"
    )


def api_error(d: dict) -> str | None:
    return d.get("error", {}).get("info")


# ---------------------------------------------------------------------------
# Candidate collection (category members + search)
# ---------------------------------------------------------------------------


def category_titles(category: str) -> list[str]:
    titles: list[str] = []
    cont: dict = {}
    for _ in range(CAND_PAGES):
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmnamespace": "6",
            "cmlimit": "500",
            "format": "json",
        }
        params.update(cont)
        d = api_get(params)
        if api_error(d):
            log.info("  skip source %s: %s", category, api_error(d))
            return titles
        for m in d.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
        if "continue" in d:
            cont = {"cmcontinue": d["continue"]["cmcontinue"]}
        else:
            break
    return titles


def search_titles(query: str, max_pages: int = 8) -> list[str]:
    titles: list[str] = []
    cont: dict = {}
    for _ in range(max_pages):
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": "6",
            "srlimit": "50",
            "format": "json",
        }
        params.update(cont)
        d = api_get(params)
        if api_error(d):
            log.info("  skip search %r: %s", query, api_error(d))
            return titles
        for m in d.get("query", {}).get("search", []):
            titles.append(m["title"])
        if "continue" in d:
            cont = {"sroffset": d["continue"]["sroffset"]}
        else:
            break
    return titles


def collect_candidates(
    cls: str, claimed: set[str], cap: int = MAX_CAND
) -> list[str]:
    """Gather candidate file titles for a class, minus already-claimed ones."""
    pool: list[str] = []
    seen: set[str] = set()
    for kind, source in SOURCES[cls]:
        if len(pool) >= cap:
            break
        titles = category_titles(source) if kind == "cat" else search_titles(source)
        for t in titles:
            if t in claimed or t in seen:
                continue
            seen.add(t)
            pool.append(t)
        if pool:
            log.info("  %s from %s (pool=%d)", len(titles), source, len(pool))
    return pool


# ---------------------------------------------------------------------------
# imageinfo (batch, cached)
# ---------------------------------------------------------------------------


def now_value(d) -> str:
    return ((d or {}).get("value") or "").strip()


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def load_metadata_cache(path: Path) -> dict[str, dict]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            log.warning("ignoring corrupt metadata cache %s", path)
    return {}


def save_metadata_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache), encoding="utf-8")


def fetch_imageinfo(titles: list[str], cache: dict) -> dict[str, dict]:
    """Guarantee an info entry for every title, using/extending the cache."""
    missing = [t for t in titles if t not in cache]
    for i in range(0, len(missing), 50):
        batch = missing[i : i + 50]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        }
        d = api_get(params)
        for page in d.get("query", {}).get("pages", {}).values():
            title = page.get("title")
            ii = (page.get("imageinfo") or [None])[0]
            if title and ii:
                cache[title] = ii
    return {t: cache[t] for t in titles if t in cache}


# ---------------------------------------------------------------------------
# Per-image acceptance checks
# ---------------------------------------------------------------------------


def file_categories(raw: str) -> list[str]:
    raw = raw or ""
    names = re.findall(r'"?Category:([^"]+)"?', raw)
    if names:
        return names
    # extmetadata returns category names pipe-delimited, possibly wrapped in a
    # JSON-ish list-of-lists string.
    cats = [c for block in re.split(r"[\[\]\"]", raw) if block
            for c in block.split("|")]
    return [c.strip() for c in cats if c.strip()]


def is_ai_contaminated(title: str, ii: dict) -> bool:
    ext = ii.get("extmetadata") or {}
    text = " ".join(
        [
            title,
            now_value(ext.get("ImageDescription")),
            now_value(ext.get("Artist")),
            now_value(ext.get("Credit")),
            now_value(ext.get("ObjectName")),
            json.dumps(now_value(ext.get("Categories"))),
        ]
    ).lower()
    if AI_RE.search(text):
        return True
    for cat in file_categories(now_value(ext.get("Categories"))):
        if AI_CATEGORY_RE.search(cat):
            return True
    return False


def is_free_license(name: str) -> bool:
    if not name:
        return False
    n = name.lower()
    if LICENSE_REJECT_RE.search(n):
        return False
    return bool(LICENSE_ACCEPT_RE.search(n))


def accept(meta: dict):
    """Yield ('ok', reason) or ('reject', reason) for a candidate's imageinfo."""
    title = meta["title"]
    ii = meta["info"]
    width = ii.get("width", 0)
    height = ii.get("height", 0)
    size = ii.get("size", 0) or 0
    mime = ii.get("mime") or ""
    ext = ii.get("extmetadata") or {}

    if is_ai_contaminated(title, ii):
        return "reject", "ai_marker"
    if width < MIN_DIM or height < MIN_DIM:
        return "reject", "too_small"
    if size > MAX_BYTES:
        return "reject", "too_big"
    if width > MAX_DIM or height > MAX_DIM:
        return "reject", "too_large_dims"
    if mime not in {
        "image/png", "image/jpeg", "image/jpg", "image/tiff",
        "image/webp", "image/bmp", "image/x-ms-bmp",
    }:
        return "reject", f"mime_{mime or 'unknown'}"
    license_name = now_value(ext.get("LicenseShortName")) or now_value(
        ext.get("License")
    )
    if not is_free_license(license_name):
        return "reject", f"license_{license_name or 'unknown'}"
    author = strip_html(now_value(ext.get("Artist"))) or strip_html(
        now_value(ext.get("Credit"))
    )
    return "ok", {"license": license_name, "author": author}


# ---------------------------------------------------------------------------
# Directories / CSV contract
# ---------------------------------------------------------------------------


def ensure_out(out: Path) -> None:
    for cls in CLASS_ORDER:
        (out / cls).mkdir(parents=True, exist_ok=True)


def load_state(out: Path):
    """Recover completed rows + title->image_id from prior partial output."""
    rows: dict[str, dict] = {}      # image_id -> labels row
    attr: dict[str, dict] = {}      # image_id -> attribution row
    done: dict[str, str] = {}       # commons title -> image_id
    labels_csv = out / "labels.csv"
    if labels_csv.is_file():
        for r in csv.DictReader(labels_csv.open(newline="", encoding="utf-8")):
            p = out / r["filepath"]
            if p.is_file():
                rows[r["image_id"]] = r
            else:
                log.info("dropping stale row %s (file missing)", r["image_id"])
    attr_csv = out / "attribution.csv"
    if attr_csv.is_file():
        for r in csv.DictReader(attr_csv.open(newline="", encoding="utf-8")):
            attr[r["image_id"]] = r
            title = title_from_url(r["source_url"])
            if title:
                done[title] = r["image_id"]
    return rows, attr, done


def title_from_url(url: str) -> str | None:
    if "wiki/" not in url:
        return None
    tail = url.split("wiki/", 1)[1]
    return unquote(tail)


def write_labels(out: Path, rows: dict[str, dict]) -> None:
    fieldnames = ["image_id", "filepath", "label", "generator"]
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rid in sorted(rows):
            w.writerow(rows[rid])


def write_attribution(out: Path, attr: dict[str, dict]) -> None:
    fieldnames = ["image_id", "source_url", "author", "license"]
    with open(out / "attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rid in sorted(attr):
            w.writerow(attr[rid])


def download_convert(url: str, out_path: Path) -> bool:
    """Download an original and save a normalized RGB JPEG. Return success.

    upload.wikimedia.org rate-limits aggressively: retry 429 with a real
    back-off (honoring Retry-After) instead of giving up on the first one.
    """
    attempts = 10
    for attempt in range(attempts):
        RL.wait()
        try:
            resp = SESSION.get(url, timeout=240)
        except requests.RequestException as exc:
            log.warning("  download request error %s (try %d)", exc, attempt + 1)
            time.sleep(10 + attempt * 5)
            continue
        if resp.status_code == 429:
            delay = int(resp.headers.get("Retry-After", "60") or 60)
            log.warning(
                "  download HTTP 429; back off %ss (try %d/%d)",
                delay, attempt + 1, attempts,
            )
            time.sleep(min(delay, 90))
            continue
        if resp.status_code != 200:
            log.warning(
                "  download HTTP %d for %s (try %d/%d)",
                resp.status_code, url, attempt + 1, attempts,
            )
            time.sleep(10 + attempt * 5)
            continue
        break
    else:
        log.warning("  download gave up after %d attempts: %s", attempts, url)
        return False

    try:
        raw = resp.content
        if len(raw) > MAX_BYTES + 500_000:
            log.warning("  download unexpectedly large (%dB)", len(raw))
            return False
        img = Image.open(io.BytesIO(raw))
        img.load()
        rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        log.warning("  decode failed for %s: %s", url, exc)
        return False
    if max(rgb.size) > OUT_MAX_SIDE:
        scale = OUT_MAX_SIDE / max(rgb.size)
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.LANCZOS,
        )
    tmp = out_path.with_suffix(".jpg.part")
    try:
        rgb.save(tmp, "JPEG", quality=88, optimize=True)
        os.replace(tmp, out_path)
    except (OSError, ValueError):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--interval", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    RL.interval = args.interval
    out: Path = args.out
    ensure_out(out)
    cache_path = out / "_imageinfo_cache.json"

    rows, attr, done = load_state(out)
    claimed = set(done)  # titles already owned by some completed image

    log.info(
        "Resuming: %d completed rows, %d titles claimed",
        len(rows), len(claimed),
    )
    cache = load_metadata_cache(cache_path)

    per_class_done: dict[str, int] = {}
    for r in rows.values():
        cls = r["generator"]
        per_class_done[cls] = per_class_done.get(cls, 0) + 1

    stats: dict[str, dict] = {}
    rejected_counts: dict[str, dict[str, int]] = {}
    for cls in CLASS_ORDER:
        rejected_counts[cls] = {}

    for cls in CLASS_ORDER:
        target = args.per_class
        completed = per_class_done.get(cls, 0)
        need = max(0, target - completed)
        log.info("== %s  (have %d, need %d) ==", cls, completed, need)
        if need <= 0:
            continue

        cands = collect_candidates(cls, claimed)
        rng = random.Random(args.seed + len(claimed))
        rng.shuffle(cands)
        cands = cands[:MAX_CAND]
        log.info("  candidates after shuffle: %d", len(cands))
        if not cands:
            log.warning("  no candidates for %s", cls)
            continue

        info = fetch_imageinfo(cands, cache)
        accepted: list[tuple[str, dict, str, str]] = []
        missing = 0
        for t in cands:
            ii = info.get(t)
            if not ii:
                missing += 1
                continue
            verdict, detail = accept({"title": t, "info": ii})
            if verdict == "ok":
                accepted.append((t, ii, detail["license"], detail["author"]))
            else:
                reason = detail if isinstance(detail, str) else "unknown"
                rejected_counts[cls][reason] = (
                    rejected_counts[cls].get(reason, 0) + 1
                )
        save_metadata_cache(cache_path, cache)
        log.info(
            "  %d accepted of %d checked (%d no-info); rejects: %s",
            len(accepted), len(cands), missing,
            dict(sorted(rejected_counts[cls].items(), key=lambda kv: -kv[1])),
        )
        if not accepted:
            log.warning("  nothing accepted for %s", cls)
            continue

        idx_avail = {int(f.stem.split("_")[-1]) for f in (out / cls).glob(f"{cls}_*.jpg")}
        idx = 0
        while idx + 1 in idx_avail:
            idx += 1
        got = 0
        for title, ii, lic, author in accepted:
            if got >= need:
                break
            idx += 1
            image_id = f"{cls}_{idx:04d}"
            src_url = f"https://commons.wikimedia.org/wiki/{quote(title)}"
            fname = f"{cls}/{image_id}.jpg"
            dest = out / cls / f"{image_id}.jpg"
            log.info("  [%s/%s] %s -> %s", got + 1, need, title, image_id)
            if not dest.is_file():
                if not download_convert(ii["url"], dest):
                    idx -= 1
                    rejected_counts[cls]["download_fail"] = (
                        rejected_counts[cls].get("download_fail", 0) + 1
                    )
                    continue
            rows[image_id] = {
                "image_id": image_id,
                "filepath": fname,
                "label": "0",
                "generator": cls,
            }
            attr[image_id] = {
                "image_id": image_id,
                "source_url": src_url,
                "author": author,
                "license": lic,
            }
            claimed.add(title)
            done[title] = image_id
            write_labels(out, rows)
            write_attribution(out, attr)
            got += 1
        per_class_done[cls] = completed + got
        log.info("%s: +%d (total %d)", cls, got, per_class_done[cls])

    save_metadata_cache(cache_path, cache)

    total_bytes = sum(
        f.stat().st_size for f in out.rglob("*.jpg")
    )
    manifest = {
        "per_class": dict(per_class_done),
        "total_images": len(rows),
        "total_bytes": total_bytes,
        "license_histogram": {},
        "seed": args.seed,
        "interval": args.interval,
    }
    lic_hist: dict[str, int] = {}
    for a in attr.values():
        l = ".".join((a["license"] or "unknown").strip().split())
        lic_hist[l] = lic_hist.get(l, 0) + 1
    manifest["license_histogram"] = dict(
        sorted(lic_hist.items(), key=lambda kv: -kv[1])
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info("WROTE %s", out / "manifest.json")
    log.info("DONE: %d images, %.1f MiB on disk", len(rows), total_bytes / 2**20)
    return 0


if __name__ == "__main__":
    sys.exit(main())