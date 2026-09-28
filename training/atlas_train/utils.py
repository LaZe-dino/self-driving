"""Device selection and checkpoint helpers shared by train / export / predict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from atlas_train.model import LaneNet, ModelConfig


def select_device(requested: str = "auto") -> torch.device:
    """Pick the fastest available device: Apple GPU (mps) > NVIDIA (cuda) > cpu."""
    if requested != "auto":
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[LaneNet, dict[str, Any]]:
    """Rebuild the model described in a checkpoint and load its weights.

    The backbone is built with pretrained=False: the checkpoint already holds
    every trained weight, so there is nothing to download.
    """
    # weights_only=False: our own checkpoints also store plain config dicts.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    cfg.pretrained = False
    model = LaneNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
