"""TuSimple slot assignment and the numpy validation metrics."""

import json

import numpy as np

from atlas_data.row_anchor import RowAnchorGeometry, decode_classes, encode_lanes
from atlas_data.sample import SLOT_ORDER
from atlas_data.tusimple import assign_slots, load_tusimple
from atlas_train.metrics import LaneMetrics

H_SAMPLES = list(range(240, 720, 10))


def line(x_bottom, slope):
    return [x_bottom + slope * (710 - y) for y in H_SAMPLES]


def test_assign_slots_orders_by_bottom_crossing():
    lanes = [line(1100, 0.8), line(300, -0.8), line(760, 0.2), line(100, -0.1), line(560, -0.2)]
    out = assign_slots(lanes, H_SAMPLES, 1280, 720)
    bottoms = {SLOT_ORDER[k]: out[k][-1] for k in range(4)}
    assert bottoms["left"] == 560 and bottoms["right"] == 760
    assert bottoms["left_outer"] == 300 and bottoms["right_outer"] == 1100
    # the third lane on the left (x=100 at the bottom) is dropped: we model two per side


def test_load_tusimple(tmp_path):
    entry = {"raw_file": "clips/0313-1/6040/20.jpg", "h_samples": H_SAMPLES,
             "lanes": [[int(v) if 0 <= v < 1280 else -2 for v in line(500, -0.3)]]}
    (tmp_path / "label_data_0313.json").write_text(json.dumps(entry) + "\n")
    [s] = load_tusimple(tmp_path)
    assert s.group == "tusimple:clips/0313-1/6040" and s.source == "tusimple"
    assert s.present_slots() == ["left"]


def test_metrics_perfect_and_wrong():
    geom = RowAnchorGeometry()
    lanes = np.full((4, len(H_SAMPLES)), -2.0)
    lanes[1] = line(500, -0.3)
    lanes[2] = line(800, 0.3)
    cls, x = encode_lanes(lanes, np.asarray(H_SAMPLES, float), 1280, 720, geom)
    xc = decode_classes(cls, 1280, geom)

    m = LaneMetrics(geom.absent_class)
    m.update(cls[None], xc[None], cls[None], x[None], np.array([1280]))
    r = m.summary()
    assert r["row_acc"] == 1.0 and r["tusimple_acc"] == 1.0
    assert r["fp"] == 0.0 and r["fn"] == 0.0
    assert r["px_err"] <= 1280 / geom.num_bins / 2

    shifted = np.where(xc >= 0, xc + 100, xc)
    m = LaneMetrics(geom.absent_class)
    m.update(cls[None], shifted[None], cls[None], x[None], np.array([1280]))
    r = m.summary()
    assert r["tusimple_acc"] == 0.0 and r["fn"] == 1.0 and r["fp"] == 1.0
