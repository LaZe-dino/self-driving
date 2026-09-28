"""PyTorch Dataset: LaneSample -> (network input tensor, row-anchor targets).

This is the only file in atlas_data that imports torch; everything it calls
is numpy/OpenCV so the logic stays unit-testable without PyTorch.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from atlas_data.augment import Augmenter
from atlas_data.preprocess import load_rgb, preprocess
from atlas_data.row_anchor import RowAnchorGeometry, encode_lanes
from atlas_data.sample import LaneSample


class LaneDataset(Dataset):
    """Loads an image, augments it, and encodes its lanes as row-anchor classes.

    Each item is a dict:
        image:  (3, input_h, input_w) float32 normalised input
        cls:    (4, R) int64 target class per slot/row (C = absent)
        x:      (4, R) float32 exact target x in source pixels (-2 = absent)
        exists: (4,) float32 1 if the slot has >= 2 present rows
        width:  source image width (to convert bins to pixels)
        source: "session" / "tusimple"
    """

    def __init__(self, samples: Sequence[LaneSample], geom: RowAnchorGeometry, augment: Augmenter | None = None):
        self.samples = list(samples)
        self.geom = geom
        self.augment = augment
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def _generator(self) -> np.random.Generator:
        # torch gives every DataLoader worker a different initial seed, so
        # workers do not produce identical "random" augmentations.
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.samples[i]
        img = load_rgb(s.image_path)
        h, w = img.shape[:2]
        lanes = s.lanes.copy()
        hs = s.h_samples.astype(np.float64)
        if (w, h) != (s.width, s.height):
            # Labels refer to (s.width, s.height); rescale them to the actual file.
            lanes = np.where(lanes >= 0, lanes * w / s.width, lanes)
            hs = hs * h / s.height
        if self.augment is not None:
            img, lanes = self.augment(img, lanes, self._generator())
        cls, x = encode_lanes(lanes, hs, w, h, self.geom)
        exists = ((cls != self.geom.absent_class).sum(axis=1) >= 2).astype(np.float32)
        return {
            "image": torch.from_numpy(preprocess(img, self.geom)),
            "cls": torch.from_numpy(cls),
            "x": torch.from_numpy(x.astype(np.float32)),
            "exists": torch.from_numpy(exists),
            "width": w,
            "source": s.source,
        }
