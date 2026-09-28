"""Load the public TuSimple lane benchmark as ground truth.

TuSimple (2017) has 1280x720 highway images with *human* lane annotations.
Each line of a label file (label_data_0313.json, ..., test_label.json) is:

    {"raw_file": "clips/0313-1/6040/20.jpg",
     "h_samples": [160, 170, ..., 710],
     "lanes": [[-2, -2, 632, 625, ...], [...], ...]}

Unlike our sessions, TuSimple lanes come as an unordered list. We assign each
one to a slot by where it meets the bottom of the image relative to the image
centre (the camera is mounted near the car's centre line): nearest lane on
the left is "left", the next one "left_outer", and likewise on the right.

Why mix TuSimple in at all? It is real human ground truth. Our pseudo-labels
can only teach the network to be as good as our classical detector; TuSimple
labels pull it towards the truth and give an honest external benchmark.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from atlas_data.sample import ABSENT, LaneSample, empty_lanes

TUSIMPLE_SIZE = (1280, 720)


def bottom_x(xs: np.ndarray, ys: np.ndarray, height: int, fit_points: int = 5) -> float | None:
    """x where a lane would cross the bottom row, from a line fit to its lowest points."""
    valid = xs >= 0
    if valid.sum() < 2:
        return None
    yv, xv = ys[valid], xs[valid]
    order = np.argsort(yv)[-fit_points:]
    a, b = np.polyfit(yv[order], xv[order], 1)
    return float(a * (height - 1) + b)


def assign_slots(lanes: Iterable[Iterable[float]], h_samples: Iterable[float], width: int, height: int) -> np.ndarray:
    """Order an unordered list of lanes into (NUM_SLOTS, N) slot rows.

    Lanes are split into "left of centre" and "right of centre" by their
    bottom crossing; within each side the one nearest the centre is the ego
    boundary. More than two lanes per side are dropped (we only model four).
    """
    h = np.asarray(list(h_samples), dtype=np.float64)
    out = empty_lanes(len(h))
    centre = (width - 1) / 2.0
    left: list[tuple[float, np.ndarray]] = []
    right: list[tuple[float, np.ndarray]] = []
    for lane in lanes:
        xs = np.asarray(list(lane), dtype=np.float64)
        if xs.shape != h.shape:
            continue
        xb = bottom_x(xs, h, height)
        if xb is None:
            continue
        (left if xb < centre else right).append((xb, np.where(xs >= 0, xs, ABSENT)))
    left.sort(key=lambda t: -t[0])  # nearest centre first
    right.sort(key=lambda t: t[0])
    # SLOT_ORDER = left_outer(0), left(1), right(2), right_outer(3)
    for rank, (_, xs) in enumerate(left[:2]):
        out[1 - rank] = xs
    for rank, (_, xs) in enumerate(right[:2]):
        out[2 + rank] = xs
    return out


def tusimple_label_files(root: str | os.PathLike) -> list[Path]:
    """All label json files directly under a TuSimple root (train and/or test)."""
    r = Path(root)
    files = sorted(Path(p) for p in glob.glob(str(r / "label_data_*.json")))
    if (r / "test_label.json").exists():
        files.append(r / "test_label.json")
    return files


def load_tusimple(
    root: str | os.PathLike,
    label_files: Iterable[str | os.PathLike] | None = None,
    max_samples: int | None = None,
) -> list[LaneSample]:
    """Load TuSimple annotations as LaneSamples.

    Args:
        root: folder that ``raw_file`` paths are relative to.
        label_files: specific json files; default all label_data_*.json
            (and test_label.json) under ``root``.
        max_samples: optional cap, handy for a quick smoke test.
    """
    root = Path(root)
    files = [Path(f) for f in label_files] if label_files else tusimple_label_files(root)
    if not files:
        raise FileNotFoundError(f"no TuSimple label json found under {root}")
    samples: list[LaneSample] = []
    w, h = TUSIMPLE_SIZE
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                raw = entry["raw_file"]
                hs = np.asarray(entry["h_samples"], dtype=np.float64)
                samples.append(
                    LaneSample(
                        image_path=str(root / raw),
                        width=w,
                        height=h,
                        h_samples=hs,
                        lanes=assign_slots(entry["lanes"], hs, w, h),
                        # A clip is one ~1 s recording (20 frames, the last labelled):
                        # the natural unit for splitting.
                        group=f"tusimple:{Path(raw).parent.as_posix()}",
                        source="tusimple",
                        meta={"raw_file": raw, "label_file": path.name},
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
    return samples
