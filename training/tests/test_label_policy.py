"""Label policy: which logged frames become pseudo-labelled training samples."""

import json

import pytest

from atlas_data.sample import SLOT_ORDER
from atlas_data.sessions import LabelPolicy, find_session_dirs, load_session

H_SAMPLES = [400, 410, 420, 430]


def record(fid, status="LOCKED", confidence=0.9, human=None, outer_measured=False, outer_accepted=True):
    lanes = [
        {"slot": "left", "x": [500, 490, 480, 470], "sigma_m_at_15m": 0.1},
        {"slot": "right", "x": [700, 710, 720, 730], "sigma_m_at_15m": 0.1},
        {"slot": "left_outer", "x": [200, 180, 160, 140], "sigma_m_at_15m": 0.3},
        {"slot": "right_outer", "x": [-2, -2, 1000, 1030], "sigma_m_at_15m": 0.3},
    ]
    meas = [{"slot": "left", "accepted": True}, {"slot": "right", "accepted": True}]
    if outer_measured:
        meas.append({"slot": "left_outer", "accepted": outer_accepted})
    return {
        "schema_version": 1,
        "frame_id": fid,
        "image": f"images/{fid:08d}.jpg",
        "labels": {"human": human},
        "status": status,
        "confidence": confidence,
        "h_samples": H_SAMPLES,
        "lanes_image": lanes,
        "measurements": meas,
    }


def make_session(tmp_path, records, labels=None, name="20260928_120000_test", truncated_tail=False):
    d = tmp_path / name
    (d / "images").mkdir(parents=True)
    (d / "session.json").write_text(json.dumps(
        {"schema_version": 1, "image_space": "undistorted", "camera": {"width": 1280, "height": 720}}))
    lines = [json.dumps(r) for r in records]
    text = "\n".join(lines) + "\n"
    if truncated_tail:
        text += '{"frame_id": 99, "status": "LOC'
    (d / "frames.jsonl").write_text(text)
    for r in records:
        (d / r["image"]).write_bytes(b"")  # the loader only checks existence
    if labels is not None:
        (d / "labels.jsonl").write_text("\n".join(json.dumps(l) for l in labels) + "\n")
    return d


def ids(samples):
    return sorted(s.meta["frame_id"] for s in samples)


def test_decide_rules():
    p = LabelPolicy()
    assert p.decide("LOCKED", 0.95, "bad") == (False, "human_bad")
    assert p.decide("SEARCHING", 0.0, "good") == (True, "human_good")
    assert p.decide("LOCKED", 0.85, None) == (True, "pseudo_label")
    assert p.decide("LOCKED", 0.79, None)[0] is False
    assert p.decide("LOCKED", None, None)[0] is False
    assert p.decide("PARTIAL", 0.99, None)[0] is False
    assert p.decide("COASTING", 0.99, None)[0] is False
    assert LabelPolicy(min_confidence=0.5).decide("LOCKED", 0.6, None)[0] is True
    assert LabelPolicy(auto_statuses=("LOCKED", "PARTIAL")).decide("PARTIAL", 0.9, None)[0] is True
    assert LabelPolicy(use_unlabeled=False).decide("LOCKED", 0.99, None)[0] is False


def test_session_policy_and_label_overrides(tmp_path):
    recs = [
        record(1),                                   # confident pseudo-label -> in
        record(2, confidence=0.5),                   # low confidence -> out
        record(3, status="PARTIAL"),                 # wrong status -> out
        record(4, human="bad"),                      # in-record bad -> out
        record(5, status="SEARCHING", confidence=0.1, human="good"),  # human good -> in
        record(6),                                   # overridden bad by labels.jsonl -> out
        record(7, status="COASTING", confidence=0.2),  # overridden good -> in
        record(8),                                   # good then bad (later wins) -> out
    ]
    labels = [
        {"frame_id": 6, "human": "bad"},
        {"frame_id": 7, "human": "good"},
        {"frame_id": 8, "human": "good"},
        {"frame_id": 8, "human": "bad"},
    ]
    d = make_session(tmp_path, recs, labels)
    samples, stats = load_session(d)
    assert ids(samples) == [1, 5, 7]
    assert stats["excluded_human_bad"] == 3
    assert stats["included_pseudo_label"] == 1
    assert stats["included_human_good"] == 2
    s = samples[0]
    assert s.group == "session:" + d.name
    assert s.lanes.shape == (4, len(H_SAMPLES))


def test_outer_lane_requires_accepted_measurement(tmp_path):
    d = make_session(tmp_path, [record(1), record(2, outer_measured=True), record(3, outer_measured=True,
                                                                                  outer_accepted=False)])
    samples = {s.meta["frame_id"]: s for s in load_session(d)[0]}
    lo = SLOT_ORDER.index("left_outer")
    ro = SLOT_ORDER.index("right_outer")
    assert (samples[1].lanes[lo] < 0).all()
    assert list(samples[2].lanes[lo]) == [200, 180, 160, 140]
    assert (samples[3].lanes[lo] < 0).all()
    assert (samples[2].lanes[ro] < 0).all()  # never measured
    keep_all = {s.meta["frame_id"]: s for s in load_session(d, LabelPolicy(outer_requires_measurement=False))[0]}
    assert list(keep_all[1].lanes[ro]) == [-2, -2, 1000, 1030]


def test_sigma_filter(tmp_path):
    d = make_session(tmp_path, [record(1)])
    s = load_session(d, LabelPolicy(max_sigma_m_at_15m=0.2, outer_requires_measurement=False))[0][0]
    assert s.present_slots() == ["left", "right"]


def test_truncated_line_and_discovery(tmp_path):
    d = make_session(tmp_path, [record(1)], truncated_tail=True)
    samples, stats = load_session(d)
    assert ids(samples) == [1]
    assert stats["malformed_line"] == 1
    assert find_session_dirs([str(tmp_path)]) == [d.resolve()]
    assert find_session_dirs([str(tmp_path / "*")]) == [d.resolve()]


def test_missing_image_is_excluded(tmp_path):
    d = make_session(tmp_path, [record(1), record(2)])
    (d / "images" / "00000002.jpg").unlink()
    samples, stats = load_session(d)
    assert ids(samples) == [1]
    assert stats["excluded_missing_image"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
