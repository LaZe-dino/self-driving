"""Load Atlas Vision recording sessions as training samples (pseudo-labels).

A session folder written by the C++ app (vision/data_logger.cpp) contains:

    session.json   camera model; the image size lives in camera.width/height
    frames.jsonl   one JSON record per logged frame
    images/*.jpg   the undistorted frame each record points to
    labels.jsonl   (optional) human corrections: {"frame_id": N, "human": "good"|"bad"}

Each record's ``lanes_image`` holds the *classical tracker's* lanes, sampled
TuSimple-style at ``h_samples``. Using those as training targets is called
pseudo-labelling: we did not draw these labels, our own detector did. That
is only useful if we keep the frames where the detector was very likely
right, which is what the LabelPolicy below decides.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from atlas_data.sample import ABSENT, NUM_SLOTS, SLOT_ORDER, LaneSample

# Marker for "no labels.jsonl entry for this frame" (distinct from an explicit null).
_MISSING = object()


@dataclass(frozen=True)
class LabelPolicy:
    """Which logged frames become training samples, and which of their lanes.

    Frame rule (in priority order):
      1. human "bad"  -> always excluded. A person said the output was wrong.
      2. human "good" -> included, whatever the tracker status was.
      3. no human label -> included only if the tracker was confident:
         status in ``auto_statuses`` (default LOCKED) and
         confidence >= ``min_confidence``.

    Lane rule: the tracker's road model always *predicts* all four boundaries
    (outer ones at +-1.5 lane widths), even on a two-lane road where no outer
    line exists. With ``outer_requires_measurement`` an outer slot is kept only
    if a detection was actually accepted for it on that frame, which matches
    what the dashboard draws (and hence what a human reviewer approved).
    """

    min_confidence: float = 0.8
    auto_statuses: tuple[str, ...] = ("LOCKED",)
    use_unlabeled: bool = True
    outer_requires_measurement: bool = True
    max_sigma_m_at_15m: float | None = None
    min_lanes: int = 1

    def decide(self, status: str | None, confidence: float | None, human: str | None) -> tuple[bool, str]:
        """Return (include?, reason). The reason is counted in load statistics."""
        if human == "bad":
            return False, "human_bad"
        if human == "good":
            return True, "human_good"
        if not self.use_unlabeled:
            return False, "unlabeled"
        if status not in self.auto_statuses:
            return False, f"status_{status}"
        if confidence is None or not np.isfinite(confidence) or confidence < self.min_confidence:
            return False, "low_confidence"
        return True, "pseudo_label"


def read_jsonl(path: Path, counter: Counter | None = None) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a .jsonl file, skipping malformed lines.

    The logger appends a line per frame; if the app is killed mid-write, the
    last line may be truncated. We count and skip such lines instead of
    failing the whole session.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if counter is not None:
                    counter["malformed_line"] += 1
                continue
            if isinstance(obj, dict):
                yield obj


def load_human_labels(session_dir: str | os.PathLike) -> dict[int, str | None]:
    """frame_id -> "good" | "bad" | None from labels.jsonl. Later lines override earlier ones."""
    path = Path(session_dir) / "labels.jsonl"
    labels: dict[int, str | None] = {}
    if not path.exists():
        return labels
    for obj in read_jsonl(path):
        if "frame_id" not in obj:
            continue
        human = obj.get("human")
        labels[int(obj["frame_id"])] = human if human in ("good", "bad") else None
    return labels


def _lanes_from_record(rec: dict[str, Any], policy: LabelPolicy, stats: Counter) -> np.ndarray:
    """Build the (NUM_SLOTS, N) lane array from a record's lanes_image."""
    h = rec.get("h_samples") or []
    lanes = np.full((NUM_SLOTS, len(h)), ABSENT, dtype=np.float64)
    measured = {m.get("slot") for m in rec.get("measurements") or [] if m.get("accepted")}
    for lane in rec.get("lanes_image") or []:
        slot = lane.get("slot")
        if slot not in SLOT_ORDER:
            continue
        xs = lane.get("x") or []
        if len(xs) != len(h):
            stats["slot_dropped_bad_length"] += 1
            continue
        if policy.outer_requires_measurement and slot.endswith("_outer") and slot not in measured:
            stats["slot_dropped_unmeasured_outer"] += 1
            continue
        sigma = lane.get("sigma_m_at_15m")
        if policy.max_sigma_m_at_15m is not None and (sigma is None or sigma > policy.max_sigma_m_at_15m):
            stats["slot_dropped_sigma"] += 1
            continue
        lanes[SLOT_ORDER.index(slot)] = [ABSENT if v is None or v < 0 else float(v) for v in xs]
    return lanes


def load_session(session_dir: str | os.PathLike, policy: LabelPolicy = LabelPolicy()) -> tuple[list[LaneSample], Counter]:
    """Load one session folder. Returns (samples, statistics of what was kept/why dropped)."""
    root = Path(session_dir)
    stats: Counter = Counter()
    with open(root / "session.json", "r", encoding="utf-8") as f:
        session = json.load(f)
    if session.get("schema_version") != 1:
        warnings.warn(f"{root}: unexpected schema_version {session.get('schema_version')}")
    if session.get("image_space", "undistorted") != "undistorted":
        warnings.warn(f"{root}: image_space is {session.get('image_space')!r}, expected 'undistorted'")
    width = int(session["camera"]["width"])
    height = int(session["camera"]["height"])
    overrides = load_human_labels(root)
    group = f"session:{root.name}"

    samples: list[LaneSample] = []
    for rec in read_jsonl(root / "frames.jsonl", stats):
        stats["records"] += 1
        fid = int(rec.get("frame_id", -1))
        human = overrides.get(fid, _MISSING)
        if human is _MISSING:
            human = (rec.get("labels") or {}).get("human")
        ok, reason = policy.decide(rec.get("status"), rec.get("confidence"), human)
        if not ok:
            stats[f"excluded_{reason}"] += 1
            continue
        lanes = _lanes_from_record(rec, policy, stats)
        n_lanes = int(((lanes >= 0).sum(axis=1) >= 2).sum())
        if n_lanes < policy.min_lanes:
            stats["excluded_no_lanes"] += 1
            continue
        image_path = root / rec.get("image", "")
        if not rec.get("image") or not image_path.exists():
            stats["excluded_missing_image"] += 1
            continue
        stats[f"included_{reason}"] += 1
        samples.append(
            LaneSample(
                image_path=str(image_path),
                width=width,
                height=height,
                h_samples=np.asarray(rec["h_samples"], dtype=np.float64),
                lanes=lanes,
                group=group,
                source="session",
                meta={
                    "frame_id": fid,
                    "status": rec.get("status"),
                    "confidence": rec.get("confidence"),
                    "human": human,
                    "reason": reason,
                    "image": rec.get("image"),
                },
            )
        )
    return samples, stats


def find_session_dirs(patterns: Iterable[str]) -> list[Path]:
    """Expand paths/globs into session folders (folders holding a session.json).

    Accepts a session folder itself, a parent such as ``data/sessions``, or a
    glob such as ``data/sessions/*`` (quoted or already expanded by the shell).
    """
    found: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern) or [pattern]
        for m in matches:
            p = Path(m)
            if (p / "session.json").exists():
                found.append(p)
            elif p.is_dir():
                found.extend(sorted(c for c in p.iterdir() if (c / "session.json").exists()))
    unique = sorted({p.resolve() for p in found})
    return unique


def load_sessions(patterns: Iterable[str], policy: LabelPolicy = LabelPolicy()) -> tuple[list[LaneSample], Counter]:
    """Load many sessions; statistics are summed over all of them."""
    samples: list[LaneSample] = []
    total: Counter = Counter()
    for d in find_session_dirs(patterns):
        s, stats = load_session(d, policy)
        samples.extend(s)
        total.update(stats)
        total["sessions"] += 1
    return samples, total


def format_stats(stats: Counter) -> str:
    """Human-readable one-key-per-line summary of load statistics."""
    return "\n".join(f"  {k:32s} {v}" for k, v in sorted(stats.items()))
