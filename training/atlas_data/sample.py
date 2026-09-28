"""The common in-memory sample format shared by every data source.

Both our own logged sessions and the public TuSimple dataset describe lanes
the same way: for a list of image rows (``h_samples``) they give the lane's
x pixel on that row, or -2 when the lane is not visible on that row. We keep
that format and add one thing TuSimple lacks: a fixed *slot* per lane, so the
network always predicts "the ego-left line" in the same output channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Spatial order, left to right across the image. The network's output channel
# k always means SLOT_ORDER[k]. Note this is NOT the C++ LaneSlot enum order
# (left, right, left_outer, right_outer); a horizontal flip is simply a
# reversal of this axis, which is why we chose it.
SLOT_ORDER: tuple[str, ...] = ("left_outer", "left", "right", "right_outer")
NUM_SLOTS = len(SLOT_ORDER)

# TuSimple's marker for "no lane on this row". Anything negative is treated as absent.
ABSENT = -2.0


@dataclass
class LaneSample:
    """One labelled image.

    Attributes:
        image_path: absolute path of the (undistorted) image file.
        width, height: size of the image the labels refer to, in pixels.
        h_samples: (N,) image rows, top to bottom, at which lanes are sampled.
        lanes: (NUM_SLOTS, N) float x pixel per slot and row; ABSENT if missing.
        group: the recording the frame came from ("session:<name>" or
            "tusimple:<clip>"). Train/val splits are made per group, never per
            frame (see atlas_data.split).
        source: "session" or "tusimple"; used to report metrics per source.
        meta: free-form extras (frame id, tracker status, why it was kept...).
    """

    image_path: str
    width: int
    height: int
    h_samples: np.ndarray
    lanes: np.ndarray
    group: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def present_slots(self, min_points: int = 2) -> list[str]:
        """Names of slots that have at least ``min_points`` visible rows."""
        counts = (self.lanes >= 0).sum(axis=1)
        return [SLOT_ORDER[k] for k in range(NUM_SLOTS) if counts[k] >= min_points]


def empty_lanes(num_rows: int) -> np.ndarray:
    """A (NUM_SLOTS, num_rows) array with every entry ABSENT."""
    return np.full((NUM_SLOTS, num_rows), ABSENT, dtype=np.float64)
