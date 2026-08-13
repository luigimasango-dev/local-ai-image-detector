#!/usr/bin/env python3
"""Split the eval set into a TUNE half and a HOLDOUT half.

Why this exists
---------------
Fusion weights and the confidence threshold are fitted parameters. Fitting them
on the same images we then quote a balanced-accuracy number from produces an
optimistically biased score that will not transfer to the bounty's private
benchmark. Fit on TUNE, report the go/no-go number from HOLDOUT, and quote both
so the size of the gap is visible.

The split is stratified by (label, generator) so both halves carry the same
label balance and the same generator mix -- an unstratified split can put most
of one generator on one side and make the halves incomparable.

Usage
-----
    python split_eval_set.py                 # 50/50, seed 42
    python split_eval_set.py --tune-frac 0.5 --seed 42
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


def load_rows(labels_csv: Path) -> list[dict]:
    with open(labels_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_split(
    rows: list[dict], tune_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split image_ids into (tune, holdout), stratified by label AND generator."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        key = (r["label"], (r.get("generator") or "unknown").strip() or "unknown")
        strata[key].append(r["image_id"])

    rng = random.Random(seed)
    tune: list[str] = []
    holdout: list[str] = []
    for key in sorted(strata):
        ids = sorted(strata[key])  # sort first so the shuffle is reproducible
        rng.shuffle(ids)
        n_tune = round(len(ids) * tune_frac)
        tune.extend(ids[:n_tune])
        holdout.extend(ids[n_tune:])
    return sorted(tune), sorted(holdout)


def summarize(rows: list[dict], ids: set[str]) -> dict:
    subset = [r for r in rows if r["image_id"] in ids]
    by_label: dict[str, int] = defaultdict(int)
    by_gen: dict[str, int] = defaultdict(int)
    for r in subset:
        by_label[r["label"]] += 1
        by_gen[(r.get("generator") or "unknown").strip() or "unknown"] += 1
    return {
        "n": len(subset),
        "ai": by_label.get("1", 0),
        "real": by_label.get("0", 0),
        "generators": dict(sorted(by_gen.items())),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--tune-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not 0.0 < args.tune_frac < 1.0:
        p.error("--tune-frac must be strictly between 0 and 1")

    if not args.labels.exists():
        print(f"ERROR: {args.labels} not found. Run fetch_dataset.py first.", file=sys.stderr)
        return 1

    rows = load_rows(args.labels)
    if not rows:
        print("ERROR: labels.csv has no rows.", file=sys.stderr)
        return 1

    tune, holdout = stratified_split(rows, args.tune_frac, args.seed)

    overlap = set(tune) & set(holdout)
    assert not overlap, f"BUG: {len(overlap)} ids in both halves"
    assert len(tune) + len(holdout) == len(rows), "BUG: split lost or duplicated rows"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "split_tune.txt").write_text("\n".join(tune) + "\n", encoding="utf-8")
    (args.out_dir / "split_holdout.txt").write_text("\n".join(holdout) + "\n", encoding="utf-8")

    summary = {
        "seed": args.seed,
        "tune_frac": args.tune_frac,
        "tune": summarize(rows, set(tune)),
        "holdout": summarize(rows, set(holdout)),
    }
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"TUNE    : {summary['tune']['n']:>5} images "
          f"({summary['tune']['ai']} AI / {summary['tune']['real']} real)")
    print(f"HOLDOUT : {summary['holdout']['n']:>5} images "
          f"({summary['holdout']['ai']} AI / {summary['holdout']['real']} real)")
    print(f"\nWrote split_tune.txt, split_holdout.txt, split_summary.json to {args.out_dir}")
    print("\nFit fusion weights and the threshold on TUNE only:")
    print("  python score.py --predictions ... --split data/split_tune.txt")
    print("Then quote the go/no-go number from HOLDOUT:")
    print("  python score.py --predictions ... --split data/split_holdout.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
