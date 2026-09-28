"""Splits must be by recording: no session/clip may appear in two splits."""

import numpy as np

from atlas_data.sample import LaneSample, empty_lanes
from atlas_data.split import split_by_group


def fake_samples():
    rng = np.random.default_rng(0)
    out = []
    for g in range(12):
        for i in range(int(rng.integers(5, 40))):
            src = "tusimple" if g % 3 == 0 else "session"
            out.append(LaneSample(f"/img/{g}/{i}.jpg", 1280, 720, np.arange(3.0), empty_lanes(3),
                                  f"{src}:rec{g}", src, {"i": i}))
    return out


def groups_of(samples):
    return {s.group for s in samples}


def test_no_group_in_two_splits_and_nothing_lost():
    samples = fake_samples()
    split, groups = split_by_group(samples, val_frac=0.2, test_frac=0.1, seed=3)
    sets = {k: groups_of(v) for k, v in split.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert sum(len(v) for v in split.values()) == len(samples)
    assert sets["train"] and sets["val"] and sets["test"]
    for k in split:
        assert set(groups[k]) == sets[k]


def test_split_is_deterministic_and_roughly_sized():
    samples = fake_samples()
    a, _ = split_by_group(samples, val_frac=0.25, seed=7)
    b, _ = split_by_group(list(reversed(samples)), val_frac=0.25, seed=7)
    assert groups_of(a["val"]) == groups_of(b["val"])
    frac = len(a["val"]) / len(samples)
    assert 0.1 < frac < 0.5


def test_forced_val_and_single_group():
    samples = fake_samples()
    split, _ = split_by_group(samples, forced_val=["rec4"])
    assert groups_of(split["val"]) == {"session:rec4"}
    one = [s for s in samples if s.group == "session:rec1"]
    split, _ = split_by_group(one, val_frac=0.5)
    assert len(split["train"]) == len(one) and not split["val"]
