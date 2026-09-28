"""Train / validation / test splits by recording (session or clip), never by frame.

Why not just shuffle frames? Consecutive video frames are nearly identical:
the same road, lighting, car in front, even the same smudge on the lens. If
frame 1000 is in training and frame 1001 in validation, the validation score
mostly measures memorisation ("have I seen almost exactly this image?"), and
it will look far better than how the model behaves on a new drive. That is
called *leakage*. Splitting whole sessions (and whole TuSimple clips) means
validation images come from drives the model has never seen - the honest
question we actually care about.

The assignment is deterministic: groups are ordered by a hash of their name
(plus a seed), so the same data always gives the same split, and adding a new
session does not reshuffle the old ones.

CLI (from the repo root):
    PYTHONPATH=training python -m atlas_data.split --sessions data/sessions --out training/runs/split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections import Counter
from typing import Iterable, Sequence

from atlas_data.sample import LaneSample

SPLITS = ("train", "val", "test")


def _stable_key(group: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{group}".encode("utf-8")).hexdigest()


def split_groups(
    group_sizes: dict[str, int],
    val_frac: float = 0.15,
    test_frac: float = 0.0,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Assign whole groups to splits so each split gets roughly its share of frames.

    Groups are visited in hash order; test is filled first, then val, and the
    rest goes to train. Training always keeps at least one group.
    """
    groups = sorted(group_sizes, key=lambda g: _stable_key(g, seed))
    total = sum(group_sizes.values())
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    targets = {"test": test_frac * total, "val": val_frac * total}
    filled = {"test": 0, "val": 0}
    for g in groups:
        target = next((s for s in ("test", "val") if filled[s] < targets[s] and targets[s] > 0), "train")
        out[target].append(g)
        if target != "train":
            filled[target] += group_sizes[g]
    if not out["train"]:
        for s in ("val", "test"):
            if out[s]:
                out["train"].append(out[s].pop())
                break
    if len(groups) < 3 and (val_frac > 0 or test_frac > 0):
        warnings.warn(
            f"only {len(groups)} recording(s): splits by recording will be tiny or empty. "
            "Record more sessions (different days/roads) for a meaningful validation set."
        )
    return out


def split_by_group(
    samples: Sequence[LaneSample],
    val_frac: float = 0.15,
    test_frac: float = 0.0,
    seed: int = 0,
    forced_val: Iterable[str] = (),
) -> tuple[dict[str, list[LaneSample]], dict[str, list[str]]]:
    """Split samples by ``sample.group``.

    Args:
        forced_val: substrings; any group containing one goes to validation
            (e.g. a session name you deliberately hold out). When given, the
            automatic val fraction is not used.

    Returns:
        (samples per split, group names per split)
    """
    sizes = Counter(s.group for s in samples)
    forced = [g for g in sizes if any(f in g for f in forced_val)]
    if forced:
        rest = {g: n for g, n in sizes.items() if g not in forced}
        groups = split_groups(rest, 0.0, test_frac, seed)
        groups["val"] = sorted(forced)
    else:
        groups = split_groups(dict(sizes), val_frac, test_frac, seed)
    lookup = {g: s for s, gs in groups.items() for g in gs}
    out: dict[str, list[LaneSample]] = {s: [] for s in SPLITS}
    for smp in samples:
        out[lookup[smp.group]].append(smp)
    return out, groups


def describe(split_samples: dict[str, list[LaneSample]], split_groups_: dict[str, list[str]]) -> str:
    lines = []
    for s in SPLITS:
        srcs = Counter(x.source for x in split_samples[s])
        lines.append(f"  {s:5s}: {len(split_samples[s]):6d} frames from {len(split_groups_[s]):4d} recordings {dict(srcs)}")
    return "\n".join(lines)


def main() -> None:
    from atlas_data.sessions import LabelPolicy, load_sessions
    from atlas_data.tusimple import load_tusimple

    ap = argparse.ArgumentParser(description="Make a session-level train/val/test split.")
    ap.add_argument("--sessions", nargs="*", default=[], help="session folders, parents or globs")
    ap.add_argument("--tusimple", default=None, help="TuSimple root folder")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--out", required=True, help="output split json")
    args = ap.parse_args()

    samples: list[LaneSample] = []
    if args.sessions:
        s, _ = load_sessions(args.sessions, LabelPolicy(min_confidence=args.min_confidence))
        samples += s
    if args.tusimple:
        samples += load_tusimple(args.tusimple)
    split_samples, groups = split_by_group(samples, args.val_frac, args.test_frac, args.seed)
    print(describe(split_samples, groups))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
