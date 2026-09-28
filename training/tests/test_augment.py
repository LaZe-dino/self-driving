"""Augmentations must keep image and labels consistent."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from atlas_data.augment import Augmenter, AugmentConfig, adjust_brightness_contrast, hflip, shift_horizontal
from atlas_data.sample import SLOT_ORDER, empty_lanes

W, H = 64, 32
ROWS = np.array([10.0, 20.0, 30.0])


def image_with_marks(lanes):
    """Black image with a white pixel at every labelled lane point."""
    img = np.zeros((H, W, 3), np.uint8)
    for k in range(lanes.shape[0]):
        for x, y in zip(lanes[k], ROWS):
            if x >= 0:
                img[int(y), int(x)] = 255
    return img


def sample_lanes():
    lanes = empty_lanes(len(ROWS))
    lanes[SLOT_ORDER.index("left")] = [20, 18, 16]
    lanes[SLOT_ORDER.index("right")] = [40, 44, 48]
    lanes[SLOT_ORDER.index("right_outer")] = [-2, 60, 63]
    return lanes


def marks_match(img, lanes):
    for k in range(lanes.shape[0]):
        for x, y in zip(lanes[k], ROWS):
            if x >= 0 and img[int(y), int(x)].max() != 255:
                return False
    return True


def test_flip_mirrors_x_and_swaps_slots():
    lanes = sample_lanes()
    img = image_with_marks(lanes)
    fimg, flanes = hflip(img, lanes)
    # old "right" (x=40) becomes new "left" at W-1-40
    assert list(flanes[SLOT_ORDER.index("left")]) == [W - 1 - 40, W - 1 - 44, W - 1 - 48]
    assert list(flanes[SLOT_ORDER.index("right")]) == [W - 1 - 20, W - 1 - 18, W - 1 - 16]
    assert list(flanes[SLOT_ORDER.index("left_outer")]) == [-2, W - 1 - 60, W - 1 - 63]
    assert (flanes[SLOT_ORDER.index("right_outer")] < 0).all()
    assert marks_match(fimg, flanes)
    # flipping twice is the identity
    img2, lanes2 = hflip(fimg, flanes)
    assert np.array_equal(img2, img) and np.array_equal(lanes2, lanes)


@pytest.mark.parametrize("dx", [5, -7])
def test_shift_moves_labels_and_drops_outside(dx):
    lanes = sample_lanes()
    img = image_with_marks(lanes)
    simg, slanes = shift_horizontal(img, lanes, dx)
    assert marks_match(simg, slanes)
    moved = lanes + dx
    expected = np.where((lanes >= 0) & (moved >= 0) & (moved <= W - 1), moved, -2)
    assert np.array_equal(slanes, expected)
    if dx > 0:
        assert (slanes[SLOT_ORDER.index("right_outer")] < 0).all()  # 60+5, 63+5 left the image


def test_photometric_keeps_labels():
    lanes = sample_lanes()
    img = np.full((H, W, 3), 100, np.uint8)
    out = adjust_brightness_contrast(img, 1.2, 10)
    assert out.dtype == np.uint8 and out.shape == img.shape
    aug = Augmenter(AugmentConfig(flip_prob=0, shift_prob=0, photometric_prob=1, shadow_prob=1))
    _, l2 = aug(img, lanes, np.random.default_rng(0))
    assert np.array_equal(l2, lanes)


def test_random_augmenter_stays_consistent():
    rng = np.random.default_rng(1)
    aug = Augmenter(AugmentConfig(photometric_prob=0, shadow_prob=0, shift_prob=1, flip_prob=0.5, max_shift_frac=0.1))
    for _ in range(20):
        lanes = sample_lanes()
        img, out = aug(image_with_marks(lanes), lanes, rng)
        assert marks_match(img, out)
