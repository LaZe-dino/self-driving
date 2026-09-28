"""Run a trained lane network and write TuSimple-format predictions + visualisations.

Examples (repo root):
    # a recorded session: rows = each frame's logged h_samples
    python training/predict.py --checkpoint training/runs/first/best.pt \
        --session data/sessions/20260928_101500_drive --out training/runs/first/pred

    # the TuSimple test set: rows and raw_file taken from its label file
    python training/predict.py --checkpoint training/runs/first/best.pt \
        --tusimple-json ~/datasets/tusimple/test_set/test_label.json \
        --tusimple-root ~/datasets/tusimple/test_set --out training/runs/first/pred

    # any folder of images: rows = TuSimple's 160..710 step 10, scaled to the image height
    python training/predict.py --checkpoint ... --images some/folder --out ...

For each input, --out gets <name>.predictions.json with one JSON object per
line (the TuSimple submission format, readable by the C++ atlas_eval tool):

    {"raw_file": "images/00000123.jpg",   # same path as the ground truth refers to
     "h_samples": [...],                   # rows the lanes are sampled at
     "lanes": [[x or -2 per h_sample], ...],  # present lanes only, left to right
     "slots": ["left", "right"],           # slot name for each entry of "lanes"
     "run_time": 7.1}                      # ms for the forward pass

Visualisations go to <out>/vis/<name>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from atlas_data.preprocess import load_rgb, preprocess
from atlas_data.row_anchor import anchors_to_h_samples, decode_logits
from atlas_data.sample import SLOT_ORDER
from atlas_data.sessions import read_jsonl
from atlas_train.utils import load_checkpoint, select_device

SLOT_COLORS_BGR = {
    "left_outer": (255, 160, 0),
    "left": (0, 220, 0),
    "right": (0, 220, 255),
    "right_outer": (255, 0, 200),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def default_h_samples(height: int) -> np.ndarray:
    return np.round(np.arange(160, 720, 10) * height / 720.0)


def iter_session(session_dir: Path) -> Iterator[tuple[Path, str, np.ndarray | None, dict]]:
    """(image path, raw_file, h_samples, extras) for every logged frame."""
    for rec in read_jsonl(session_dir / "frames.jsonl"):
        if not rec.get("image"):
            continue
        hs = np.asarray(rec["h_samples"], dtype=np.float64) if rec.get("h_samples") else None
        yield session_dir / rec["image"], rec["image"], hs, {"frame_id": rec.get("frame_id")}


def iter_images(folder: Path) -> Iterator[tuple[Path, str, np.ndarray | None, dict]]:
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p, p.relative_to(folder).as_posix(), None, {}


def iter_tusimple(label_json: Path, root: Path) -> Iterator[tuple[Path, str, np.ndarray | None, dict]]:
    with open(label_json, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                yield root / e["raw_file"], e["raw_file"], np.asarray(e["h_samples"], dtype=np.float64), {}


def draw(image_rgb: np.ndarray, x_anchor: np.ndarray, rows: np.ndarray, present: np.ndarray, crop_top: int) -> np.ndarray:
    """Overlay predicted lane points/polylines on the image (returns BGR)."""
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.line(vis, (0, crop_top), (vis.shape[1] - 1, crop_top), (90, 90, 90), 1)
    for k, slot in enumerate(SLOT_ORDER):
        if not present[k]:
            continue
        pts = [(int(round(x)), int(round(y))) for x, y in zip(x_anchor[k], rows) if x >= 0]
        color = SLOT_COLORS_BGR[slot]
        for a, b in zip(pts, pts[1:]):
            cv2.line(vis, a, b, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(vis, p, 3, color, -1, cv2.LINE_AA)
        if pts:
            cv2.putText(vis, slot, (pts[-1][0] - 30, pts[-1][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--session", nargs="*", default=[], help="session folder(s)")
    ap.add_argument("--images", nargs="*", default=[], help="image folder(s)")
    ap.add_argument("--tusimple-json", default=None, help="TuSimple label file giving raw_file + h_samples")
    ap.add_argument("--tusimple-root", default=None, help="folder raw_file is relative to (default: json's folder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--window", type=int, default=2, help="sub-bin decoding window (bins)")
    ap.add_argument("--exists-threshold", type=float, default=0.5)
    ap.add_argument("--max-vis", type=int, default=100, help="visualisations per input (0 = none)")
    args = ap.parse_args()

    device = select_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    geom = model.cfg.geometry
    use_exists = float(ckpt.get("train_args", {}).get("exists_weight", 0.0)) > 0

    inputs: list[tuple[str, Iterator]] = []
    for s in args.session:
        inputs.append((Path(s).name, iter_session(Path(s))))
    for d in args.images:
        inputs.append((Path(d).name, iter_images(Path(d))))
    if args.tusimple_json:
        j = Path(args.tusimple_json)
        inputs.append((j.stem, iter_tusimple(j, Path(args.tusimple_root) if args.tusimple_root else j.parent)))
    if not inputs:
        sys.exit("nothing to predict: give --session, --images or --tusimple-json")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warmed = False
    for name, items in inputs:
        vis_dir = out / "vis" / name
        pred_path = out / f"{name}.predictions.json"
        n = 0
        with open(pred_path, "w", encoding="utf-8") as f:
            for img_path, raw_file, hs, extras in items:
                if not img_path.exists():
                    print(f"missing image {img_path}, skipped")
                    continue
                img = load_rgb(str(img_path))
                h, w = img.shape[:2]
                if hs is None:
                    hs = default_h_samples(h)
                x = torch.from_numpy(preprocess(img, geom))[None].to(device)
                if not warmed:
                    # The official metric rejects images slower than 200 ms;
                    # keep one-off device initialisation out of run_time.
                    with torch.no_grad():
                        model(x)
                    warmed = True
                t0 = time.perf_counter()
                with torch.no_grad():
                    row_logits, exists_logits = model(x)
                    row_logits = row_logits[0].float().cpu().numpy()
                    exists_p = torch.sigmoid(exists_logits[0]).float().cpu().numpy()
                run_ms = (time.perf_counter() - t0) * 1000.0

                x_anchor = decode_logits(row_logits, w, geom, args.window)
                present = (x_anchor >= 0).sum(axis=1) >= 2
                if use_exists:
                    present &= exists_p > args.exists_threshold
                x_anchor[~present] = -2.0
                lanes = anchors_to_h_samples(x_anchor, h, geom, hs)
                keep = [k for k in range(len(SLOT_ORDER)) if present[k] and (lanes[k] >= 0).any()]
                rec = {
                    "raw_file": raw_file,
                    "h_samples": [int(v) for v in hs],
                    "lanes": [lanes[k].tolist() for k in keep],
                    "slots": [SLOT_ORDER[k] for k in keep],
                    "run_time": round(run_ms, 2),
                    **extras,
                }
                f.write(json.dumps(rec) + "\n")
                if n < args.max_vis:
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis = draw(img, x_anchor, geom.anchor_rows_src(h), present, geom.crop_top_px(h))
                    cv2.imwrite(str(vis_dir / f"{n:06d}.jpg"), vis)
                n += 1
        print(f"{name}: {n} predictions -> {pred_path}")


if __name__ == "__main__":
    main()
