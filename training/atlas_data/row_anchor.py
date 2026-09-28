"""Row-anchor lane representation: lane detection as per-row classification.

This is the formulation of "Ultra Fast Structure-aware Deep Lane Detection"
(Qin et al., ECCV 2020, a.k.a. UFLD). Instead of segmenting every pixel and
then fitting curves, we ask a much smaller question:

    For each lane slot and for each of R fixed image rows ("row anchors"),
    which of C horizontal cells ("column bins") does the lane cross -
    or is the lane absent on that row?

So the network outputs, per slot and row, C + 1 scores: one per column bin
plus a final "absent" class. That is a normal softmax classification, trained
with cross-entropy. Why this works well:

* It is cheap: the output is 4 x R x (C+1) numbers instead of a full-resolution
  mask, so a small fully-connected head can look at the *whole* image at once
  (global context helps when lines are occluded or worn).
* A lane is by construction one x per row, so no clustering or post-processing
  is needed to separate lanes: slot k *is* lane k.
* Sub-bin precision is recovered by taking the softmax-weighted mean of the bin
  centres around the winning bin, instead of just the argmax.

Geometry (all shared with the C++ side through the export sidecar JSON):

* The network sees a crop of the lower part of the image (the sky carries no
  lane information): rows [crop_top_frac * H, H), full width, resized to
  input_w x input_h. With the default 800 x 288 input and a 16:9 source,
  crop_top_frac = 0.36 keeps the aspect ratio (no stretching).
* Row anchor i sits at the centre of the i-th of R equal horizontal bands of
  that crop.
* Column bin j covers source columns [j*W/C, (j+1)*W/C).

Everything in this module is plain numpy so it can be unit-tested without
PyTorch, and mirrored exactly in C++.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from atlas_data.sample import ABSENT, NUM_SLOTS


@dataclass(frozen=True)
class RowAnchorGeometry:
    """Network input size and the row-anchor / column-bin layout."""

    input_w: int = 800
    input_h: int = 288
    num_rows: int = 40  # R: row anchors per lane
    num_bins: int = 100  # C: column bins; class index C means "absent"
    crop_top_frac: float = 0.36  # fraction of the image height cut off at the top

    @staticmethod
    def keep_aspect_crop_frac(input_w: int, input_h: int, src_w: int = 1280, src_h: int = 720) -> float:
        """Top-crop fraction so that a full-width crop has the input's aspect ratio."""
        visible_h = src_w * input_h / input_w
        return float(max(0.0, 1.0 - visible_h / src_h))

    @property
    def absent_class(self) -> int:
        return self.num_bins

    @property
    def num_classes(self) -> int:
        return self.num_bins + 1

    def crop_top_px(self, src_h: int) -> int:
        """First source row inside the network's crop."""
        return int(round(self.crop_top_frac * src_h))

    def anchor_rows_src(self, src_h: int) -> np.ndarray:
        """(R,) source-image rows (float) of the row anchors, top to bottom.

        v_i = top + (i + 0.5) * (H - top) / R - 0.5, with top = round(crop_top_frac * H).
        """
        top = self.crop_top_px(src_h)
        crop_h = src_h - top
        return top + (np.arange(self.num_rows) + 0.5) * crop_h / self.num_rows - 0.5

    def anchor_rows_input(self) -> np.ndarray:
        """(R,) row anchors in network-input pixel rows."""
        return (np.arange(self.num_rows) + 0.5) * self.input_h / self.num_rows - 0.5

    def bin_centres_px(self, src_w: int) -> np.ndarray:
        """(C,) x pixel of each column bin centre in a source image of width src_w."""
        return (np.arange(self.num_bins) + 0.5) * src_w / self.num_bins - 0.5

    def anchor_spacing_px(self, src_h: int) -> float:
        return (src_h - self.crop_top_px(src_h)) / self.num_rows

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RowAnchorGeometry":
        return cls(**{k: d[k] for k in ("input_w", "input_h", "num_rows", "num_bins", "crop_top_frac")})


def resample_lane(xs: np.ndarray, ys: np.ndarray, rows: np.ndarray, max_gap: float | None = None) -> np.ndarray:
    """Linearly interpolate a lane given as x-per-row onto new rows.

    Args:
        xs: (N,) x pixel per sample row; negative means absent on that row.
        ys: (N,) the sample rows.
        rows: (M,) rows to evaluate the lane at.
        max_gap: do not bridge two visible samples that are more than this
            many rows apart (e.g. a gap where the tracker was unsure).

    Returns:
        (M,) x per requested row; ABSENT outside the visible extent. We never
        extrapolate: a label must only claim what was actually seen.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    out = np.full(rows.shape, ABSENT, dtype=np.float64)
    valid = xs >= 0
    if valid.sum() < 2:
        return out
    order = np.argsort(ys[valid])
    yv, xv = ys[valid][order], xs[valid][order]
    inside = (rows >= yv[0]) & (rows <= yv[-1])
    if max_gap is not None:
        idx = np.clip(np.searchsorted(yv, rows, side="left"), 1, len(yv) - 1)
        inside &= (yv[idx] - yv[idx - 1]) <= max_gap
    out[inside] = np.interp(rows[inside], yv, xv)
    return out


def encode_lanes(
    lanes: np.ndarray,
    h_samples: np.ndarray,
    src_w: int,
    src_h: int,
    geom: RowAnchorGeometry,
    max_gap: float | None = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn TuSimple-style lanes into row-anchor classification targets.

    Args:
        lanes: (NUM_SLOTS, N) x pixel per slot and h_sample (negative = absent).
        h_samples: (N,) rows the lanes are sampled at.
        src_w, src_h: size of the image the pixels refer to.

    Returns:
        cls: (NUM_SLOTS, R) int64 target class: column bin in [0, C) or C (absent).
        x: (NUM_SLOTS, R) float64 exact x pixel at each anchor (ABSENT if absent),
            used to measure pixel error, not for the loss.
    """
    rows = geom.anchor_rows_src(src_h)
    x = np.stack([resample_lane(lanes[k], h_samples, rows, max_gap) for k in range(lanes.shape[0])])
    bins = np.floor((x + 0.5) * geom.num_bins / src_w).astype(np.int64)
    present = (x >= 0) & (bins >= 0) & (bins < geom.num_bins)
    cls = np.where(present, bins, geom.absent_class).astype(np.int64)
    x = np.where(present, x, ABSENT)
    return cls, x


def decode_classes(cls: np.ndarray, src_w: int, geom: RowAnchorGeometry) -> np.ndarray:
    """Class indices (..., R) -> x pixel of the bin centre, or ABSENT."""
    cls = np.asarray(cls)
    centres = geom.bin_centres_px(src_w)
    present = cls < geom.num_bins
    x = centres[np.clip(cls, 0, geom.num_bins - 1)]
    return np.where(present, x, ABSENT)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_logits(logits: np.ndarray, src_w: int, geom: RowAnchorGeometry, window: int = 2) -> np.ndarray:
    """Network output (..., R, C+1) -> x pixel per anchor (..., R), ABSENT if absent.

    1. The lane is absent on a row if the winning class (argmax over all C+1)
       is the "absent" class.
    2. Otherwise we find the best column bin and take the softmax-weighted mean
       of bin centres within +-``window`` bins of it. The window matters: if
       the network is torn between two far-apart columns (say, a real line and
       a crack in the road), a global mean would land in between, on neither.
    """
    logits = np.asarray(logits, dtype=np.float64)
    present = logits.argmax(axis=-1) != geom.absent_class
    loc = logits[..., : geom.num_bins]
    best = loc.argmax(axis=-1)
    j = np.arange(geom.num_bins)
    near = np.abs(j - best[..., None]) <= window
    p = _softmax(np.where(near, loc, -np.inf))
    x = (p * geom.bin_centres_px(src_w)).sum(axis=-1)
    return np.where(present, x, ABSENT)


def anchors_to_h_samples(
    x_anchor: np.ndarray,
    src_h: int,
    geom: RowAnchorGeometry,
    h_samples: np.ndarray,
    max_gap: float | None = None,
) -> np.ndarray:
    """Resample per-anchor predictions (NUM_SLOTS, R) onto TuSimple rows.

    Returns (NUM_SLOTS, N) integer pixels with -2 where the lane is absent,
    ready to be written as TuSimple ``lanes``. Gaps longer than 2.5 anchor
    spacings are not bridged.
    """
    rows = geom.anchor_rows_src(src_h)
    if max_gap is None:
        max_gap = 2.5 * geom.anchor_spacing_px(src_h)
    out = np.stack([resample_lane(x_anchor[k], rows, h_samples, max_gap) for k in range(x_anchor.shape[0])])
    return np.where(out >= 0, np.round(out), ABSENT).astype(np.int64)


__all__ = [
    "NUM_SLOTS",
    "RowAnchorGeometry",
    "anchors_to_h_samples",
    "decode_classes",
    "decode_logits",
    "encode_lanes",
    "resample_lane",
]
