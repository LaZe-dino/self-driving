"""Train the row-anchor lane network on Atlas sessions (pseudo-labels) and/or TuSimple.

Example (repo root, on the Mac):
    python training/train.py --sessions "data/sessions/*" --tusimple ~/datasets/tusimple/train_set \
        --epochs 30 --batch-size 16 --out training/runs/first

Outputs in --out:
    config.json    every argument + model config (geometry, backbone...)
    split.json     which recordings went to train / val (reproducibility)
    metrics.csv    one row per epoch: losses, learning rate, validation metrics
    last.pt        latest checkpoint (use --resume to continue)
    best.pt        checkpoint with the best validation TuSimple-style accuracy

The loop in one paragraph: for each batch we run the network, compare its
per-row class scores to the targets with cross-entropy (+ smoothness and
exists terms), backpropagate to get gradients, and let AdamW nudge every
weight downhill. The learning rate warms up for one epoch (large steps on
random-ish heads can wreck pretrained features) and then follows a cosine
curve down to 1% of its peak, which usually beats a constant rate. After
every epoch we measure on validation recordings the model never trains on.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from atlas_data.augment import Augmenter, AugmentConfig
from atlas_data.row_anchor import RowAnchorGeometry, decode_logits
from atlas_data.sample import LaneSample
from atlas_data.sessions import LabelPolicy, format_stats, load_sessions
from atlas_data.split import describe, split_by_group
from atlas_data.torch_dataset import LaneDataset
from atlas_data.tusimple import load_tusimple
from atlas_train.losses import LaneLoss
from atlas_train.metrics import LaneMetrics
from atlas_train.model import BACKBONES, LaneNet, ModelConfig, count_parameters
from atlas_train.utils import select_device

try:
    from tqdm import tqdm
except ImportError:  # progress bars are a nicety, not a requirement
    def tqdm(it, **_):  # type: ignore[no-redef]
        return it

METRIC_NAMES = ("row_acc", "presence_recall", "false_presence", "px_err", "tusimple_acc", "fp", "fn")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--sessions", nargs="*", default=[], help="session folders, their parent, or globs")
    d.add_argument("--tusimple", default=None, help="TuSimple root (holds clips/ and label_data_*.json)")
    d.add_argument("--tusimple-json", nargs="*", default=None, help="specific TuSimple label files")
    d.add_argument("--min-confidence", type=float, default=0.8, help="pseudo-label confidence threshold")
    d.add_argument("--statuses", nargs="+", default=["LOCKED"], help="tracker statuses accepted as pseudo-labels")
    d.add_argument("--human-only", action="store_true", help="use only frames a human marked good")
    d.add_argument("--keep-unmeasured-outer", action="store_true",
                   help="keep outer-lane labels even when no line was measured (not recommended)")
    d.add_argument("--max-sigma", type=float, default=None, help="drop lanes whose sigma at 15 m exceeds this (m)")
    d.add_argument("--val-frac", type=float, default=0.15, help="fraction of frames (by recording) for validation")
    d.add_argument("--val-sessions", nargs="*", default=[], help="name substrings of recordings forced into val")
    d.add_argument("--split-file", default=None, help="reuse a split.json (from a previous run or atlas_data.split)")
    d.add_argument("--max-samples", type=int, default=None, help="cap the number of samples (smoke tests)")
    d.add_argument("--no-augment", action="store_true")

    m = ap.add_argument_group("model")
    m.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    m.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    m.add_argument("--input-w", type=int, default=800)
    m.add_argument("--input-h", type=int, default=288)
    m.add_argument("--rows", type=int, default=40, help="row anchors R")
    m.add_argument("--bins", type=int, default=100, help="column bins C")
    m.add_argument("--crop-top-frac", type=float, default=None,
                   help="fraction of image height cropped from the top (default: keep 16:9 aspect)")
    m.add_argument("--hidden-dim", type=int, default=1024)

    t = ap.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--warmup-epochs", type=float, default=1.0)
    t.add_argument("--smooth-weight", type=float, default=0.5, help="structural smoothness loss weight (0 = off)")
    t.add_argument("--exists-weight", type=float, default=0.1, help="lane-exists loss weight (0 = off)")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", default=None, help="continue from a last.pt")
    t.add_argument("--out", required=True, help="run folder, e.g. training/runs/first")
    return ap.parse_args(argv)


def gather_samples(args: argparse.Namespace) -> list[LaneSample]:
    samples: list[LaneSample] = []
    if args.sessions:
        policy = LabelPolicy(
            min_confidence=args.min_confidence,
            auto_statuses=tuple(args.statuses),
            use_unlabeled=not args.human_only,
            outer_requires_measurement=not args.keep_unmeasured_outer,
            max_sigma_m_at_15m=args.max_sigma,
        )
        s, stats = load_sessions(args.sessions, policy)
        print(f"sessions: {len(s)} samples kept\n{format_stats(stats)}")
        samples += s
    if args.tusimple:
        t = load_tusimple(args.tusimple, args.tusimple_json)
        print(f"tusimple: {len(t)} samples")
        samples += t
    if args.max_samples is not None and len(samples) > args.max_samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.max_samples)
    return samples


def make_split(args: argparse.Namespace, samples: list[LaneSample]) -> tuple[dict[str, list[LaneSample]], dict[str, list[str]]]:
    if args.split_file:
        with open(args.split_file, "r", encoding="utf-8") as f:
            groups = json.load(f)
        lookup = {g: s for s, gs in groups.items() for g in gs}
        new = sorted({s.group for s in samples if s.group not in lookup})
        if new:
            print(f"note: {len(new)} recordings not in {args.split_file} go to train")
        out: dict[str, list[LaneSample]] = {"train": [], "val": [], "test": []}
        for s in samples:
            out[lookup.get(s.group, "train")].append(s)
        groups.setdefault("train", []).extend(new)
        return out, groups
    return split_by_group(samples, args.val_frac, 0.0, args.seed, forced_val=args.val_sessions)


def cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, floor: float = 0.01):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, epoch: int) -> dict[str, float]:
    model.train()
    sums: Counter = Counter()
    n = 0
    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        cls = batch["cls"].to(device)
        exists = batch["exists"].to(device)
        row_logits, exists_logits = model(images)
        loss, parts = loss_fn(row_logits, exists_logits, cls, exists)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        bs = images.shape[0]
        n += bs
        for k, v in parts.items():
            sums[k] += v * bs
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, geom: RowAnchorGeometry, sources: list[str]) -> dict[str, float]:
    model.eval()
    metrics = {s: LaneMetrics(geom.absent_class) for s in ["all", *sources]}
    loss_sum, n = 0.0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        cls = batch["cls"].to(device)
        exists = batch["exists"].to(device)
        row_logits, exists_logits = model(images)
        _, parts = loss_fn(row_logits, exists_logits, cls, exists)
        bs = images.shape[0]
        loss_sum += parts["total"] * bs
        n += bs

        logits = row_logits.float().cpu().numpy()
        widths = np.asarray(batch["width"])
        pred_cls = logits.argmax(axis=-1)
        pred_x = np.stack([decode_logits(logits[b], int(widths[b]), geom) for b in range(bs)])
        gt_cls = batch["cls"].numpy()
        gt_x = batch["x"].numpy()
        src = np.asarray(batch["source"])
        metrics["all"].update(pred_cls, pred_x, gt_cls, gt_x, widths)
        for s in sources:
            sel = src == s
            if sel.any():
                metrics[s].update(pred_cls[sel], pred_x[sel], gt_cls[sel], gt_x[sel], widths[sel])
    out = {"val_loss": loss_sum / max(1, n)}
    for s, m in metrics.items():
        prefix = "val_" if s == "all" else f"val_{s}_"
        out.update({prefix + k: float(v) for k, v in m.summary().items()})
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    print(f"device: {device}")

    crop = args.crop_top_frac
    if crop is None:
        crop = RowAnchorGeometry.keep_aspect_crop_frac(args.input_w, args.input_h)
    geom = RowAnchorGeometry(args.input_w, args.input_h, args.rows, args.bins, crop)
    cfg = ModelConfig(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        hidden_dim=args.hidden_dim,
        geometry=geom,
    )

    samples = gather_samples(args)
    if not samples:
        sys.exit("no training samples: check --sessions / --tusimple and the label policy statistics above")
    split, groups = make_split(args, samples)
    print("split (by recording):\n" + describe(split, groups))
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model_config": cfg.to_dict()}, f, indent=2)

    augment = None if args.no_augment else Augmenter(AugmentConfig())
    train_ds = LaneDataset(split["train"], geom, augment)
    val_ds = LaneDataset(split["val"], geom, None)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=len(train_ds) > args.batch_size, pin_memory=pin,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=pin) if len(val_ds) else None
    if val_loader is None:
        print("WARNING: empty validation set; best.pt will simply track the lowest training loss")
    sources = sorted({s.source for s in split["val"]})

    model = LaneNet(cfg).to(device)
    print(f"model: {args.backbone}, {count_parameters(model) / 1e6:.1f} M parameters, "
          f"input {geom.input_w}x{geom.input_h}, {geom.num_rows} rows x {geom.num_bins}+1 classes")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = cosine_with_warmup(optimizer, int(args.warmup_epochs * steps_per_epoch), args.epochs * steps_per_epoch)
    loss_fn = LaneLoss(geom.absent_class, args.smooth_weight, args.exists_weight)

    start_epoch, best_score = 0, -float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", best_score)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    fields = ["epoch", "lr", "seconds", "train_total", "train_cls", "train_smooth", "train_exists", "val_loss"]
    for s in ["all", *sources]:
        prefix = "val_" if s == "all" else f"val_{s}_"
        fields += [prefix + k for k in METRIC_NAMES]
    csv_path = out / "metrics.csv"
    new_csv = not csv_path.exists() or not args.resume
    csv_file = open(csv_path, "w" if new_csv else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
    if new_csv:
        writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        row: dict[str, Any] = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_one_epoch(model, train_loader, loss_fn, optimizer,
                                                                 scheduler, device, epoch).items()})
        if val_loader is not None:
            row.update(evaluate(model, val_loader, loss_fn, device, geom, sources))
            score = row["val_tusimple_acc"]
            score = -float("inf") if math.isnan(score) else score
        else:
            score = -row["train_total"]
        row["seconds"] = round(time.time() - t0, 1)
        writer.writerow(row)
        csv_file.flush()

        is_best = score > best_score
        best_score = max(best_score, score)
        ckpt = {
            "model": model.state_dict(),
            "model_config": cfg.to_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "metrics": {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
            "train_args": {"smooth_weight": args.smooth_weight, "exists_weight": args.exists_weight},
        }
        torch.save(ckpt, out / "last.pt")
        if is_best:
            torch.save(ckpt, out / "best.pt")
        summary = f"epoch {epoch:3d}  loss {row['train_total']:.4f}"
        if val_loader is not None:
            summary += (f"  val_loss {row['val_loss']:.4f}  row_acc {row['val_row_acc']:.3f}"
                        f"  px_err {row['val_px_err']:.1f}  tusimple_acc {row['val_tusimple_acc']:.3f}"
                        f"  fp {row['val_fp']:.3f}  fn {row['val_fn']:.3f}")
        print(summary + ("  *best*" if is_best else "") + f"  ({row['seconds']} s)")
    csv_file.close()
    print(f"done. checkpoints in {out}")


if __name__ == "__main__":
    main()
