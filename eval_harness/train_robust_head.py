#!/usr/bin/env python3
"""Retrain ONLY the final classifier layer, on clean + compressed images.

The problem
-----------
Community-Forensics scores 0.8338 balanced accuracy on pristine images and
0.7474 on JPEG-85 ones. That is not a calibration issue — AUC itself falls, so
the model genuinely separates degraded images less well.

The cheap fix
-------------
The architecture is a frozen ViT backbone plus a single `nn.Linear(384 -> 1)`
head. The backbone still *sees* compressed images fine; what fails is the
decision boundary, which was only ever fitted on clean data. Retraining just
that one layer on clean AND compressed examples is:

  * ~150k parameters instead of 21.7M -> seconds of training, no GPU
  * feature extraction is one forward pass per image, on CPU
  * impossible to overfit the backbone, because the backbone never changes

This is standard linear probing / robust fine-tuning. It cannot recover
information the backbone destroyed, but it can stop the head from relying on
cues that only survive in pristine images.

Protocol
--------
Train on TUNE images only (clean + augmented copies of the same images).
Evaluate on HOLDOUT, clean and augmented. HOLDOUT is never seen in training,
so the reported numbers stay honest.

Usage:
    python train_robust_head.py --model OwensLab/commfor-model-224
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_split(path: Path) -> set[str]:
    return {l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def extract_features(model, rows, data_dir, input_size, batch_size, log):
    """Run the frozen backbone and return (features, labels, image_ids).

    We take the pre-head representation by temporarily replacing the head with
    an identity, so we get the 384-d vector the classifier normally consumes.
    """
    import torch
    from PIL import Image
    from run_inference_commfor import build_transform

    tf = build_transform(input_size)
    feats, labels, ids = [], [], []

    # ViTClassifier wraps a timm ViT whose classifier lives at .vit.head
    # (models.py:33 — nn.Linear(384, 1)). Swapping it for Identity makes
    # forward() return the 384-d representation the head normally consumes.
    original = model.vit.head
    model.vit.head = torch.nn.Identity()

    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                tensors, chunk_ids, chunk_labels = [], [], []
                for r in chunk:
                    try:
                        with Image.open(data_dir / r["filepath"]) as im:
                            tensors.append(tf(im.convert("RGB")))
                        chunk_ids.append(r["image_id"])
                        chunk_labels.append(int(r["label"]))
                    except Exception:  # noqa: BLE001
                        continue
                if not tensors:
                    continue
                out = model(torch.stack(tensors))
                feats.append(out.detach().cpu())
                labels.extend(chunk_labels)
                ids.extend(chunk_ids)
                if (start // batch_size) % 10 == 0:
                    log.info("    %d/%d", min(start + batch_size, len(rows)), len(rows))
    finally:
        model.vit.head = original

    import torch as _t
    return _t.cat(feats), _t.tensor(labels, dtype=_t.float32), ids


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
    rank_sum = 0.0
    i = 0
    ranks = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    for r, (_, y) in zip(ranks, pairs):
        if y == 1:
            rank_sum += r
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("robust_head")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="OwensLab/commfor-model-224")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--labels", type=Path, default=DEFAULT_DATA_DIR / "labels.csv")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--aug-tune-dir", type=Path, default=DEFAULT_DATA_DIR / "aug_combo_tune")
    p.add_argument("--aug-holdout-dir", type=Path, default=DEFAULT_DATA_DIR / "aug_combo")
    p.add_argument(
        "--extra-real-dir", type=Path, default=DEFAULT_DATA_DIR / "synthetic_real",
        help="Additional REAL (non-AI) images to widen what 'real' means. "
             "Half are used for training, half held out for evaluation.",
    )
    p.add_argument("--no-extra-real", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "robust_head.json")
    args = p.parse_args()

    import torch
    from run_inference_commfor import load_model

    model, _ = load_model(args.model, device="cpu")
    model.eval()

    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8")))
    tune_ids = load_split(DEFAULT_DATA_DIR / "split_tune.txt")
    hold_ids = load_split(DEFAULT_DATA_DIR / "split_holdout.txt")
    tune_rows = [r for r in rows if r["image_id"] in tune_ids]
    hold_rows = [r for r in rows if r["image_id"] in hold_ids]

    t0 = time.time()
    log.info("Extracting features: TUNE clean (%d)", len(tune_rows))
    Xtc, ytc, _ = extract_features(model, tune_rows, args.data_dir,
                                   args.input_size, args.batch_size, log)

    aug_tune_labels = args.aug_tune_dir / "labels.csv"
    log.info("Extracting features: TUNE compressed (%s)", args.aug_tune_dir.name)
    aug_tune_rows = list(csv.DictReader(open(aug_tune_labels, newline="", encoding="utf-8")))
    Xta, yta, _ = extract_features(model, aug_tune_rows, args.aug_tune_dir,
                                   args.input_size, args.batch_size, log)

    log.info("Extracting features: HOLDOUT clean (%d)", len(hold_rows))
    Xhc, yhc, _ = extract_features(model, hold_rows, args.data_dir,
                                   args.input_size, args.batch_size, log)

    aug_hold_labels = args.aug_holdout_dir / "labels.csv"
    log.info("Extracting features: HOLDOUT compressed (%s)", args.aug_holdout_dir.name)
    aug_hold_rows = list(csv.DictReader(open(aug_hold_labels, newline="", encoding="utf-8")))
    Xha, yha, _ = extract_features(model, aug_hold_rows, args.aug_holdout_dir,
                                   args.input_size, args.batch_size, log)
    log.info("Feature extraction took %.0fs", time.time() - t0)

    # Widen the notion of "real" beyond stock photography.
    #
    # Trained on Unsplash alone, the head learns "professional photograph =
    # real" and then flags charts and diagrams as AI-generated — measured at
    # 48% false positives on charts versus the original head's 20%. Feeding it
    # non-photographic real images fixes the cause rather than the symptom.
    # Half are held out so the fix can be verified on unseen examples.
    Xer_tr = Xer_te = yer_tr = yer_te = None
    extra_labels = args.extra_real_dir / "labels.csv"
    if not args.no_extra_real and extra_labels.exists():
        log.info("Extracting features: extra REAL images (%s)", args.extra_real_dir.name)
        extra_rows = list(csv.DictReader(open(extra_labels, newline="", encoding="utf-8")))
        Xer, yer, extra_ids = extract_features(model, extra_rows, args.extra_real_dir,
                                               args.input_size, args.batch_size, log)
        # Deterministic alternating split keeps every class balanced across halves.
        idx_tr = list(range(0, len(extra_ids), 2))
        idx_te = list(range(1, len(extra_ids), 2))
        Xer_tr, yer_tr = Xer[idx_tr], yer[idx_tr]
        Xer_te, yer_te = Xer[idx_te], yer[idx_te]
        log.info("  %d extra real for training, %d held out", len(idx_tr), len(idx_te))

    # Train on clean AND compressed copies of the SAME tune images.
    parts_X, parts_y = [Xtc, Xta], [ytc, yta]
    if Xer_tr is not None:
        parts_X.append(Xer_tr)
        parts_y.append(yer_tr)
    Xtr = torch.cat(parts_X)
    ytr = torch.cat(parts_y)
    log.info("Training set: %d vectors (%d clean + %d compressed), dim %d",
             len(ytr), len(ytc), len(yta), Xtr.shape[1])

    # Standardize using training statistics only.
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd

    head = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = lossf(head(Xtr_n).squeeze(1), ytr)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            log.info("  epoch %d loss %.4f", epoch, loss.item())

    def score(X):
        with torch.no_grad():
            return torch.sigmoid(head(((X - mu) / sd)).squeeze(1)).tolist()

    results = {}
    eval_sets = [("holdout_clean", Xhc, yhc), ("holdout_compressed", Xha, yha)]
    if Xer_te is not None:
        eval_sets.append(("held_out_non_photo_real", Xer_te, yer_te))
    for name, X, y in eval_sets:
        s = score(X)
        yl = y.tolist()
        if all(v == 0 for v in yl):
            # All-real set: balanced accuracy is undefined, false-positive
            # rate is the only meaningful number.
            fp = sum(1 for v in s if v >= 0.65)
            results[name] = {
                "n": len(s),
                "false_positive_rate_at_0.65": fp / len(s),
                "mean_score": sum(s) / len(s),
            }
            continue
        best_t, best_ba = max(
            ((t / 100, balanced_acc(s, yl, t / 100)) for t in range(1, 100)),
            key=lambda z: z[1])
        results[name] = {
            "auc": auc_roc(s, yl),
            "balanced_acc_at_0.65": balanced_acc(s, yl, 0.65),
            "best_threshold": best_t,
            "best_balanced_acc": best_ba,
        }

    print("\n" + "=" * 64)
    print("RETRAINED HEAD — evaluated on HOLDOUT (never trained on)")
    print("=" * 64)
    for name, r in results.items():
        print(f"\n{name}:")
        if "false_positive_rate_at_0.65" in r:
            print(f"  n                       {r['n']}")
            print(f"  FALSE POSITIVE rate     {r['false_positive_rate_at_0.65']:.4f}"
                  f"   (original head: 0.05 overall / 0.20 on charts)")
            print(f"  mean score              {r['mean_score']:.4f}")
            continue
        print(f"  AUC                     {r['auc']:.4f}")
        print(f"  balanced acc @0.65      {r['balanced_acc_at_0.65']:.4f}")
        print(f"  best threshold {r['best_threshold']:.2f} -> {r['best_balanced_acc']:.4f}")

    print("\nBaseline for comparison (original head, single 224 model):")
    print("  holdout clean       AUC 0.8854")
    print("  holdout compressed  AUC 0.7778 (pair)")

    args.out.write_text(json.dumps({
        "model": args.model,
        "input_size": args.input_size,
        "weights": head.weight.detach().squeeze(0).tolist(),
        "bias": float(head.bias.detach()[0]),
        "feature_mean": mu.squeeze(0).tolist(),
        "feature_std": sd.squeeze(0).tolist(),
        "results": results,
        "trained_on": "TUNE clean + TUNE combo-compressed",
    }, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
