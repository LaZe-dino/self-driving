"""Row-anchor encoding/decoding round trips."""

import numpy as np

from atlas_data.row_anchor import (
    RowAnchorGeometry,
    anchors_to_h_samples,
    decode_classes,
    decode_logits,
    encode_lanes,
    resample_lane,
)
from atlas_data.sample import empty_lanes

W, H = 1280, 720
GEOM = RowAnchorGeometry()
H_SAMPLES = np.arange(160, 720, 10, dtype=np.float64)
HALF_BIN = W / GEOM.num_bins / 2


def slanted_lanes():
    lanes = empty_lanes(len(H_SAMPLES))
    lanes[1] = 640 - 0.8 * (H_SAMPLES - 260)   # left, leaning outwards towards the bottom
    lanes[2] = 660 + 0.9 * (H_SAMPLES - 260)   # right
    lanes[1][H_SAMPLES < 300] = -2              # left starts lower down
    lanes[2][lanes[2] > W - 1] = -2             # right leaves the image
    return lanes


def test_default_crop_keeps_aspect():
    assert abs(RowAnchorGeometry.keep_aspect_crop_frac(800, 288) - 0.36) < 1e-9
    rows = GEOM.anchor_rows_src(H)
    assert rows[0] > GEOM.crop_top_px(H) and rows[-1] < H
    assert np.all(np.diff(rows) > 0)
    top = round(0.36 * H)
    assert np.isclose(rows[0], top + 0.5 * (H - top) / GEOM.num_rows - 0.5)


def test_resample_does_not_extrapolate_or_bridge_gaps():
    ys = np.array([100, 110, 120, 200, 210], dtype=float)
    xs = np.array([10, 20, 30, 110, 120], dtype=float)
    out = resample_lane(xs, ys, np.array([90, 105, 150, 205, 220]), max_gap=40)
    assert out[0] < 0 and out[4] < 0  # outside visible extent
    assert np.isclose(out[1], 15) and np.isclose(out[3], 115)
    assert out[2] < 0  # 80-row gap not bridged


def test_encode_decode_round_trip():
    lanes = slanted_lanes()
    cls, x = encode_lanes(lanes, H_SAMPLES, W, H, GEOM)
    assert cls.shape == (4, GEOM.num_rows) and cls.dtype == np.int64
    assert (cls[0] == GEOM.absent_class).all() and (cls[3] == GEOM.absent_class).all()
    present = cls != GEOM.absent_class
    assert present[1].sum() > 10 and present[2].sum() > 10
    assert np.array_equal(present, x >= 0)

    # classes -> bin centres are within half a bin of the exact x
    xc = decode_classes(cls, W, GEOM)
    assert np.all(np.abs(xc[present] - x[present]) <= HALF_BIN + 1e-9)
    assert np.all(xc[~present] < 0)

    # one-hot "logits" decode to the same thing
    logits = np.full(cls.shape + (GEOM.num_classes,), -10.0)
    np.put_along_axis(logits, cls[..., None], 10.0, axis=-1)
    xl = decode_logits(logits, W, GEOM)
    assert np.allclose(xl[present], xc[present], atol=1e-3)
    assert np.all(xl[~present] < 0)

    # back to TuSimple rows: close to the original where both exist
    back = anchors_to_h_samples(xc, H, GEOM, H_SAMPLES)
    rows = GEOM.anchor_rows_src(H)
    for k in (1, 2):
        vis = rows[present[k]]
        inside = (H_SAMPLES >= vis.min()) & (H_SAMPLES <= vis.max()) & (lanes[k] >= 0)
        assert inside.sum() > 5
        assert np.all(back[k][inside] >= 0)
        assert np.all(np.abs(back[k][inside] - lanes[k][inside]) <= HALF_BIN + 1.0)
    assert (back[0] == -2).all()


def test_sub_bin_decoding_uses_neighbours():
    logits = np.full((1, GEOM.num_classes), -20.0)
    logits[0, 40] = 5.0
    logits[0, 41] = 5.0  # equally split between two bins -> halfway
    x = decode_logits(logits, W, GEOM)
    centres = GEOM.bin_centres_px(W)
    assert np.isclose(x[0], (centres[40] + centres[41]) / 2)
    far = logits.copy()
    far[0, 90] = 4.9  # a far-away competitor outside the window is ignored
    assert np.isclose(decode_logits(far, W, GEOM)[0], x[0])


def test_absent_class_wins():
    logits = np.zeros((2, GEOM.num_classes))
    logits[:, GEOM.absent_class] = 3.0
    logits[1, 10] = 5.0
    x = decode_logits(logits, W, GEOM)
    assert x[0] < 0 and x[1] >= 0


def test_geometry_serialisation():
    g = RowAnchorGeometry(640, 256, 30, 80, 0.3)
    assert RowAnchorGeometry.from_dict(g.to_dict()) == g
