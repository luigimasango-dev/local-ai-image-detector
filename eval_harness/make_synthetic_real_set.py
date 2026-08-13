#!/usr/bin/env python3
"""Generate non-photographic REAL images: charts, diagrams, UI, text, logos.

Why these matter
----------------
Our eval set's real class is entirely Unsplash photography. A browsing
extension meets a lot of images that are real (nobody's AI made them) but are
not photographs at all: charts, screenshots, logos, memes, scanned text. These
are the highest-risk false positives, because "not photographic" is exactly the
cue a detector might have learned to associate with "generated".

Everything here is drawn deterministically by matplotlib/PIL, so it is
guaranteed non-AI by construction — no dataset licensing, no rate limits, and
no risk of accidentally including an AI image mislabelled as real (a real
hazard when scraping image hosts).

This complements, rather than replaces, a diverse photographic set.

Usage:
    python make_synthetic_real_set.py [--per-class 25]
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "synthetic_real"


def make_charts(out_dir: Path, n: int, rng: random.Random) -> list[tuple[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    made = []
    kinds = ["line", "bar", "scatter", "pie", "heatmap"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=110)
        x = np.arange(12)
        if kind == "line":
            for s in range(rng.randint(1, 3)):
                ax.plot(x, np.cumsum(np.random.randn(12)), marker="o", label=f"series {s+1}")
            ax.legend()
        elif kind == "bar":
            ax.bar(x, np.random.randint(5, 90, 12))
        elif kind == "scatter":
            ax.scatter(np.random.randn(120), np.random.randn(120), alpha=.6)
        elif kind == "pie":
            vals = np.random.randint(5, 40, rng.randint(3, 6))
            ax.pie(vals, labels=[f"part {j}" for j in range(len(vals))], autopct="%1.0f%%")
        else:
            ax.imshow(np.random.rand(12, 12), cmap="viridis")
            fig.colorbar(ax.images[0])
        ax.set_title(f"Quarterly figures {2019 + (i % 7)}")
        fig.tight_layout()
        name = f"chart_{i:03d}.png"
        fig.savefig(out_dir / name)
        plt.close(fig)
        made.append((name, "chart_diagram"))
    return made


def make_ui_mockups(out_dir: Path, n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Flat UI panels — the visual signature of a screenshot."""
    from PIL import Image, ImageDraw

    made = []
    for i in range(n):
        w, h = 900, 620
        bg = (250, 250, 252) if i % 2 == 0 else (24, 26, 31)
        fg = (30, 32, 38) if i % 2 == 0 else (232, 234, 238)
        accent = [(52, 120, 246), (34, 160, 90), (200, 70, 60)][i % 3]
        img = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w, 56], fill=accent)                     # title bar
        d.rectangle([0, 56, 220, h], fill=(bg[0] ^ 12, bg[1] ^ 12, bg[2] ^ 12))
        for r in range(8):                                          # sidebar rows
            y = 80 + r * 40
            d.rectangle([18, y, 200, y + 22], fill=fg if r == i % 8 else (150, 150, 158))
        for r in range(9):                                          # text lines
            y = 90 + r * 52
            d.rectangle([250, y, 250 + rng.randint(240, 600), y + 16], fill=(150, 150, 158))
        d.rectangle([250, h - 90, 430, h - 44], fill=accent)        # button
        name = f"ui_{i:03d}.png"
        img.save(out_dir / name)
        made.append((name, "screenshot_ui"))
    return made


def make_text_documents(out_dir: Path, n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Dense text pages — scanned-document / meme-caption territory."""
    from PIL import Image, ImageDraw

    words = ("the quick brown fox jumps over lazy dogs while machinery hums "
             "across factory floors and copper wire spools turn steadily").split()
    made = []
    for i in range(n):
        w, h = 780, 1000
        paper = (252, 250, 244) if i % 3 else (255, 255, 255)
        img = Image.new("RGB", (w, h), paper)
        d = ImageDraw.Draw(img)
        y = 60
        while y < h - 60:
            line = " ".join(rng.choice(words) for _ in range(rng.randint(7, 12)))
            d.text((60, y), line, fill=(35, 33, 30))
            y += 26
        if i % 4 == 0:
            d.rectangle([60, 40, 300, 46], fill=(35, 33, 30))
        name = f"doc_{i:03d}.png"
        img.save(out_dir / name)
        made.append((name, "text_document"))
    return made


def make_logos(out_dir: Path, n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Flat vector-style marks: large areas of pure colour, hard edges."""
    from PIL import Image, ImageDraw

    made = []
    for i in range(n):
        size = 512
        bg = (255, 255, 255) if i % 2 else (18, 18, 22)
        img = Image.new("RGB", (size, size), bg)
        d = ImageDraw.Draw(img)
        c1 = (rng.randint(30, 230), rng.randint(30, 230), rng.randint(30, 230))
        c2 = (rng.randint(30, 230), rng.randint(30, 230), rng.randint(30, 230))
        shape = i % 4
        if shape == 0:
            d.ellipse([100, 100, 412, 412], fill=c1)
            d.ellipse([170, 170, 342, 342], fill=bg)
        elif shape == 1:
            d.polygon([(256, 90), (430, 400), (82, 400)], fill=c1)
        elif shape == 2:
            d.rectangle([110, 110, 402, 402], fill=c1)
            d.rectangle([180, 180, 332, 332], fill=c2)
        else:
            for k in range(6):
                a = k * math.pi / 3
                d.ellipse([256 + 120 * math.cos(a) - 60, 256 + 120 * math.sin(a) - 60,
                           256 + 120 * math.cos(a) + 60, 256 + 120 * math.sin(a) + 60],
                          fill=c1 if k % 2 else c2)
        name = f"logo_{i:03d}.png"
        img.save(out_dir / name)
        made.append((name, "logo_graphic"))
    return made


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--per-class", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 1

    made: list[tuple[str, str]] = []
    made += make_charts(args.out, args.per_class, rng)
    made += make_ui_mockups(args.out, args.per_class, rng)
    made += make_text_documents(args.out, args.per_class, rng)
    made += make_logos(args.out, args.per_class, rng)

    with open(args.out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "filepath", "label", "generator"])
        w.writeheader()
        for name, cls in made:
            # label 0 = real / non-AI, true by construction: we drew these.
            w.writerow({"image_id": name, "filepath": name, "label": "0", "generator": cls})

    print(f"Wrote {len(made)} non-AI images to {args.out}")
    counts: dict[str, int] = {}
    for _, cls in made:
        counts[cls] = counts.get(cls, 0) + 1
    for cls, n in sorted(counts.items()):
        print(f"  {cls:<18} {n}")
    print("\nAll are real (non-AI) by construction. Any score above the")
    print("threshold here is a FALSE POSITIVE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
