"""Image preprocessing shared by training, prediction and (documented for) C++.

The C++ integration must reproduce these steps *exactly*, otherwise the
network sees differently-prepared pixels than it was trained on:

  1. Start from the undistorted BGR frame (W x H) - what the logger saved.
  2. Keep rows [round(crop_top_frac * H), H) at full width.
  3. Resize to input_w x input_h with bilinear interpolation.
  4. BGR -> RGB, scale to [0, 1], subtract MEAN and divide by STD per channel.
  5. Lay out as NCHW float32.

MEAN/STD are the ImageNet statistics the pretrained backbone was trained with.
"""

from __future__ import annotations

import cv2
import numpy as np

from atlas_data.row_anchor import RowAnchorGeometry

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb(path: str) -> np.ndarray:
    """Read an image file as an RGB uint8 array (H, W, 3)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_and_resize(image: np.ndarray, geom: RowAnchorGeometry) -> np.ndarray:
    """Steps 2-3 above: crop the lower part, resize to the network input size."""
    top = geom.crop_top_px(image.shape[0])
    return cv2.resize(image[top:], (geom.input_w, geom.input_h), interpolation=cv2.INTER_LINEAR)


def normalize_to_chw(image_rgb: np.ndarray) -> np.ndarray:
    """Step 4-5: uint8 RGB (h, w, 3) -> float32 (3, h, w), ImageNet-normalised."""
    f = image_rgb.astype(np.float32) / 255.0
    f = (f - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(f.transpose(2, 0, 1))


def preprocess(image_rgb: np.ndarray, geom: RowAnchorGeometry) -> np.ndarray:
    """Full pipeline for one RGB image -> (3, input_h, input_w) float32."""
    return normalize_to_chw(crop_and_resize(image_rgb, geom))
