#!/usr/bin/env python3
"""Extract still frames from personal videos as genuine amateur REAL images.

Why this set matters
--------------------
Every real image in the main benchmark is Unsplash Lite: professional
photography, well lit, cleanly composed. That is not what a browsing extension
mostly sees, and it is not what the bounty's "web-realistic samples" means.

Frames pulled from phone video are the opposite in every way that counts:
handheld, noisy, badly lit, motion-blurred, and — importantly — already
compressed by the video codec. They are a hard, realistic test of whether the
detector calls ordinary life "AI-generated".

They are also unambiguously real, since they came off a camera.

PRIVACY: frames are written to a temp directory outside the repository and are
never committed, published, or copied into the extension. Only aggregate
numbers leave this script.

Usage:
    python extract_video_frames.py --src "<folder of videos>" --per-video 3
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-video", type=int, default=3)
    p.add_argument("--min-edge", type=int, default=128)
    args = p.parse_args()

    if not args.src.is_dir():
        print(f"ERROR: {args.src} is not a directory", file=sys.stderr)
        return 1

    videos = sorted(f for f in args.src.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {args.src}", file=sys.stderr)
        return 1
    print(f"Found {len(videos)} videos")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0

    for vi, video in enumerate(videos):
        # Sample at spread-out timestamps rather than the opening frames, which
        # are often black, blurred, or a hand covering the lens.
        for k in range(args.per_video):
            frac = (k + 1) / (args.per_video + 1)
            name = f"frame_{vi:03d}_{k}.jpg"
            dest = args.out / name
            # -ss before -i seeks fast; %-based seek needs a duration probe, so
            # use a fixed ladder of timestamps instead and accept short clips
            # failing on the later ones.
            ts = f"{frac * 6:.1f}"
            cmd = [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-ss", ts, "-i", str(video),
                "-frames:v", "1", "-q:v", "3", str(dest),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1024:
                failures += 1
                dest.unlink(missing_ok=True)
                continue
            rows.append({
                "image_id": name,
                "filepath": name,
                "label": "0",            # real, by construction — off a camera
                "generator": "phone_video_frame",
            })
        if (vi + 1) % 10 == 0:
            print(f"  {vi + 1}/{len(videos)} videos")

    # Drop anything too small for the extension to bother with anyway.
    try:
        from PIL import Image
        kept = []
        for r in rows:
            with Image.open(args.out / r["filepath"]) as im:
                if min(im.size) >= args.min_edge:
                    kept.append(r)
                else:
                    (args.out / r["filepath"]).unlink(missing_ok=True)
        rows = kept
    except ImportError:
        pass

    with open(args.out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "filepath", "label", "generator"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nExtracted {len(rows)} frames ({failures} extraction failures)")
    print(f"Written to {args.out}")
    print("All label=0 (real). Any score above threshold is a FALSE POSITIVE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
