"""Label-consistent data augmentation.

Augmentation shows the network plausible variations of each training image
so it learns "what a lane line looks like" rather than memorising specific
frames. The golden rule: whenever the image moves, the labels must move the
same way. Photometric changes (brightness, contrast, shadows) do not move
anything, so labels stay put; geometric changes (shift, flip) transform the
lane x coordinates too.

All functions work on full-resolution RGB uint8 images and (NUM_SLOTS, N)
lane arrays in pixels, before cropping/resizing for the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from atlas_data.sample import ABSENT


@dataclass(frozen=True)
class AugmentConfig:
    flip_prob: float = 0.5
    max_shift_frac: float = 0.05  # horizontal shift up to +-5% of the width
    shift_prob: float = 0.5
    brightness: float = 0.25  # additive offset up to +-25% of full scale
    contrast: float = 0.25  # multiplicative gain in [1-c, 1+c]
    photometric_prob: float = 0.8
    shadow_prob: float = 0.3
    shadow_darkness: tuple[float, float] = (0.4, 0.8)  # multiply shadowed pixels by this


def hflip(image: np.ndarray, lanes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the image left-right.

    Two things happen to the labels: every x becomes (W-1-x), and the lane
    *slots* swap sides - what was the ego-left line is now on the right. With
    SLOT_ORDER = (left_outer, left, right, right_outer) that swap is just a
    reversal of the slot axis.

    Caveat worth knowing: a flipped right-hand-traffic road looks like
    left-hand traffic (e.g. the yellow centre line moves to the right). For
    predicting geometry that is harmless and doubles the variety of curves.
    """
    w = image.shape[1]
    flipped = np.ascontiguousarray(image[:, ::-1])
    x = np.where(lanes >= 0, (w - 1) - lanes, ABSENT)
    return flipped, x[::-1].copy()


def shift_horizontal(image: np.ndarray, lanes: np.ndarray, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Translate the image by an integer ``dx`` pixels (positive = right).

    Simulates the car sitting slightly off-centre in its lane. Vacated columns
    are filled with black rather than replicated/reflected edges, because
    replicating or mirroring could paint fake, unlabelled lane lines.
    Points pushed outside the image become absent.
    """
    h, w = image.shape[:2]
    out = np.zeros_like(image)
    if dx >= 0:
        out[:, dx:] = image[:, : w - dx]
    else:
        out[:, :dx] = image[:, -dx:]
    x = np.where(lanes >= 0, lanes + dx, ABSENT)
    x = np.where((x >= 0) & (x <= w - 1), x, ABSENT)
    return out, x


def adjust_brightness_contrast(image: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """pixel' = gain * (pixel - 128) + 128 + offset, clipped to [0, 255]."""
    f = image.astype(np.float32)
    f = gain * (f - 128.0) + 128.0 + offset
    return np.clip(f, 0, 255).astype(np.uint8)


def random_shadow(image: np.ndarray, rng: np.random.Generator, darkness: float) -> np.ndarray:
    """Darken a random quadrilateral spanning top to bottom (tree/bridge shadow).

    Shadows are a classic failure case for colour-threshold lane detectors,
    so they are exactly what we want the network to become robust to.
    """
    h, w = image.shape[:2]
    x_top = rng.uniform(0, w, size=2)
    x_bot = rng.uniform(0, w, size=2)
    poly = np.array(
        [[min(x_top), 0], [max(x_top), 0], [max(x_bot), h - 1], [min(x_bot), h - 1]], dtype=np.int32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    out = image.astype(np.float32)
    out[mask.astype(bool)] *= darkness
    return np.clip(out, 0, 255).astype(np.uint8)


class Augmenter:
    """Applies a random combination of the augmentations above."""

    def __init__(self, cfg: AugmentConfig = AugmentConfig()):
        self.cfg = cfg

    def __call__(
        self, image: np.ndarray, lanes: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        if rng.random() < c.photometric_prob:
            gain = rng.uniform(1 - c.contrast, 1 + c.contrast)
            offset = rng.uniform(-c.brightness, c.brightness) * 255.0
            image = adjust_brightness_contrast(image, gain, offset)
        if rng.random() < c.shadow_prob:
            image = random_shadow(image, rng, rng.uniform(*c.shadow_darkness))
        if rng.random() < c.shift_prob and c.max_shift_frac > 0:
            max_dx = int(round(c.max_shift_frac * image.shape[1]))
            dx = int(rng.integers(-max_dx, max_dx + 1))
            if dx != 0:
                image, lanes = shift_horizontal(image, lanes, dx)
        if rng.random() < c.flip_prob:
            image, lanes = hflip(image, lanes)
        return image, lanes
