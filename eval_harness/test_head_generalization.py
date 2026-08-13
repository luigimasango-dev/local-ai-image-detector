#!/usr/bin/env python3
"""Does the retrained head generalize to generators it never saw?

The danger
----------
Retraining the classifier head on our benchmark lifted compressed-image
balanced accuracy from 0.7290 to 0.8620. That is a suspiciously large jump, and
there is an obvious way for it to be an illusion:

The original Community-Forensics head was trained across ~4,803 generators. Our
replacement has seen 17. It may simply have memorized what *these* generators
look like, rather than what generation looks like in general. The bounty grades
against a private set built from generators we have never touched, so a head
that learned our 17 would collapse there — while looking excellent here.

The test
--------
Leave-generators-out. Hold out entire generators from head training, then score
only images from those unseen generators. Real images stay in both splits (they
are not generator-specific), so this isolates generator generalization.

If held-out-generator performance stays close to seen-generator performance,
the head learned something general. If it drops sharply, it memorized, and we
must ship the original head instead regardless of how good the headline looks.

Usage:
    python test_head_generalization.py --folds 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def balanced_acc(scores, labels, t: float) -> float:
    tp = sum(1 for s, y in zip(scores, labels) if y == 1 and s >= t)
    fn = sum(1 for s, y in zip(scores, labels) if y == 1 and s < t)
    tn = sum(1 for s, y in zip(scores, labels) if y == 0 and s < t)
    fp = sum(1 for s, y in zip(scores, labels) if y == 0 and s >= t)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (sens + spec) / 2.0


def auc_roc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("generalize")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="OwensLab/commfor-model-224")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--aug-dir", type=Path, default=DEFAULT_DATA_DIR / "aug_combo")
    p.add_argument("--aug-tune-dir", type=Path, default=DEFAULT_DATA_DIR / "aug_combo_tune")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--cache", type=Path,
                   default=Path(__file__).resolve().parent / "_feat_cache.pt")
    args = p.parse_args()

    import torch
    from run_inference_commfor import load_model
    from train_robust_head import extract_features

    # ------------------------------------------------------------------
    # Features over EVERY image (clean + compressed), cached — extraction
    # dominates runtime and the folds all reuse the same vectors.
    # ------------------------------------------------------------------
    if args.cache.exists():
        log.info("Loading cached features from %s", args.cache.name)
        blob = torch.load(args.cache, weights_only=False)
        X, y, gens = blob["X"], blob["y"], blob["gens"]
    else:
        model, _ = load_model(args.model, device="cpu")
        model.eval()

        rows_all = []
        for labels_csv, data_dir in [
            (args.labels, args.data_dir),
            (args.aug_tune_dir / "labels.csv", args.aug_tune_dir),
            (args.aug_dir / "labels.csv", args.aug_dir),
        ]:
            if not Path(labels_csv).exists():
                continue
            for r in csv.DictReader(open(labels_csv, newline="", encoding="utf-8")):
                rows_all.append((r, data_dir))

        log.info("Extracting features for %d image instances", len(rows_all))
        feats, ys, gs = [], [], []
        # extract_features takes a homogeneous (rows, dir) pair, so group.
        by_dir: dict[Path, list] = {}
        for r, d in rows_all:
            by_dir.setdefault(d, []).append(r)
        for d, rows in by_dir.items():
            log.info("  %s (%d)", Path(d).name, len(rows))
            Xd, yd, ids = extract_features(model, rows, Path(d), args.input_size, 16, log)
            gmap = {r["image_id"]: r["generator"] for r in rows}
            feats.append(Xd)
            ys.append(yd)
            gs.extend(gmap[i] for i in ids)
        X = torch.cat(feats)
        y = torch.cat(ys)
        gens = gs
        torch.save({"X": X, "y": y, "gens": gens}, args.cache)
        log.info("Cached features to %s", args.cache.name)

    ai_generators = sorted({g for g, lab in zip(gens, y.tolist()) if lab == 1})
    log.info("AI generators present: %d", len(ai_generators))

    # Deterministic folds over generators.
    folds = [ai_generators[i::args.folds] for i in range(args.folds)]

    print("\n" + "=" * 72)
    print("LEAVE-GENERATORS-OUT: can the retrained head handle UNSEEN generators?")
    print("=" * 72)

    seen_scores, unseen_scores = [], []
    for fi, held in enumerate(folds):
        held_set = set(held)
        is_held_ai = torch.tensor([1 if g in held_set else 0 for g in gens])
        is_real = (y == 0)

        # BUG FIXED: the original masks put EVERY real image in both train and
        # test, and merged TUNE with HOLDOUT, so holdout AI images were trained
        # on too. The "unseen generator" AUC was therefore measured against
        # in-sample negatives and was inflated. Reals must be split as well.
        #
        # Real images alternate deterministically between train and test so both
        # sides keep negatives, but no negative appears in both.
        real_idx = torch.nonzero(is_real).squeeze(1)
        real_train = torch.zeros_like(is_real)
        real_test = torch.zeros_like(is_real)
        real_train[real_idx[0::2]] = True
        real_test[real_idx[1::2]] = True

        # Train: half the reals + AI images from generators NOT held out.
        train_mask = real_train | ((y == 1) & (is_held_ai == 0))
        # Test: the other half of the reals + AI from held-out generators only.
        test_mask = real_test | ((y == 1) & (is_held_ai == 1))
        assert not bool((train_mask & test_mask).any()), "train/test overlap"

        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]

        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        head = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xtr_n = (Xtr - mu) / sd
        for _ in range(args.epochs):
            opt.zero_grad()
            lossf(head(Xtr_n).squeeze(1), ytr).backward()
            opt.step()

        with torch.no_grad():
            s_unseen = torch.sigmoid(head(((Xte - mu) / sd)).squeeze(1)).tolist()
            s_seen = torch.sigmoid(head(Xtr_n).squeeze(1)).tolist()

        a_unseen = auc_roc(s_unseen, yte.tolist())
        a_seen = auc_roc(s_seen, ytr.tolist())
        b_unseen = balanced_acc(s_unseen, yte.tolist(), 0.5)
        unseen_scores.append(a_unseen)
        seen_scores.append(a_seen)

        print(f"\nFold {fi + 1}: held out {len(held)} generators")
        print(f"  {', '.join(held)}")
        print(f"  AUC on generators it TRAINED on : {a_seen:.4f}")
        print(f"  AUC on generators NEVER SEEN    : {a_unseen:.4f}   "
              f"(balanced acc {b_unseen:.4f})")

    mean_seen = sum(seen_scores) / len(seen_scores)
    mean_unseen = sum(unseen_scores) / len(unseen_scores)
    gap = mean_seen - mean_unseen

    print("\n" + "=" * 72)
    print(f"mean AUC, generators seen in training : {mean_seen:.4f}")
    print(f"mean AUC, generators never seen       : {mean_unseen:.4f}")
    print(f"generalization gap                    : {gap:.4f}")
    print("\nOriginal (unmodified) head, for reference: AUC 0.8854 clean / "
          "0.7778 compressed")
    print("=" * 72)

    if mean_unseen >= 0.88 and gap < 0.06:
        print("\nVERDICT: the head GENERALIZES. It performs nearly as well on")
        print("generators it never saw, so the gain is not memorization.")
    elif mean_unseen >= 0.85:
        print("\nVERDICT: mostly generalizes, with a modest gap. Still likely")
        print("better than the original head on compressed images, but expect")
        print("the private benchmark to land below our headline number.")
    else:
        print("\nVERDICT: DOES NOT GENERALIZE WELL. The head leans on")
        print("generator-specific cues. Prefer the original head — our")
        print("benchmark number would not transfer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
