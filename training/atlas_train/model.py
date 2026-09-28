"""A small UFLD-style lane network: CNN backbone -> reduced features -> FC row-anchor head.

    input (B, 3, 288, 800)
      |  backbone: pretrained ResNet18 (or MobileNetV3-small), stride 32
      v
    features (B, 512, 9, 25)          <- "what is where", coarse grid
      |  1x1 conv: 512 -> 8 channels  (cheap channel pooling that KEEPS the grid)
      v
    (B, 8, 9, 25) -> flatten (1800)
      |  dropout -> Linear 1800->1024 -> ReLU -> Linear 1024 -> 4*R*(C+1)
      v
    row_logits (B, 4, R, C+1)         <- per slot, per row anchor: C column bins + "absent"

    features --global average pool--> Linear 512->4 --> exists_logits (B, 4)

Why a fully-connected head instead of the usual convolutional decoder?
Every output (say "left line, row 30, column 42") can then depend on the
*whole* image. That global view is what lets the model follow a lane through
a shadow or a missing dash: evidence from rows above and below counts.

Why reduce channels with a 1x1 conv instead of global pooling? Global pooling
would throw away *where* things are in the image - exactly what we need to
predict column positions. The 1x1 conv pools across channels only, so the
spatial layout survives while the FC layer stays a sensible size.

The auxiliary exists head answers "is there a line in this slot at all?".
It is optional in training (loss weight 0 disables it), but always exported
so the ONNX output contract stays fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from atlas_data.row_anchor import RowAnchorGeometry
from atlas_data.sample import NUM_SLOTS

BACKBONES = ("resnet18", "mobilenet_v3_small")


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    reduce_channels: int = 8
    hidden_dim: int = 1024
    dropout: float = 0.2
    geometry: RowAnchorGeometry = field(default_factory=RowAnchorGeometry)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["geometry"] = self.geometry.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        d = dict(d)
        d["geometry"] = RowAnchorGeometry.from_dict(d["geometry"])
        return cls(**d)


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (feature extractor with output stride 32, its channel count).

    "Pretrained" means ImageNet weights: the network already knows edges,
    textures and shapes, so our few thousand lane images only need to teach
    it the lane-specific part (transfer learning). Weights are downloaded by
    torchvision on first use.
    """
    import torchvision.models as tvm

    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        return body, 512
    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        return m.features, 576
    raise ValueError(f"unknown backbone {name!r}; choose from {BACKBONES}")


class LaneNet(nn.Module):
    """Row-anchor lane detector. forward(x) -> (row_logits, exists_logits)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        g = cfg.geometry
        self.num_rows = g.num_rows
        self.num_classes = g.num_classes
        self.backbone, feat_ch = build_backbone(cfg.backbone, cfg.pretrained)
        self.reduce = nn.Conv2d(feat_ch, cfg.reduce_channels, kernel_size=1)
        # Measure the feature-map size for this input size once, so the FC
        # layer can be sized correctly (it is tied to the input resolution).
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            fh, fw = self.backbone(torch.zeros(1, 3, g.input_h, g.input_w)).shape[-2:]
            self.backbone.train(was_training)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.reduce_channels * fh * fw, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, NUM_SLOTS * g.num_rows * g.num_classes),
        )
        self.exists_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(feat_ch, NUM_SLOTS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        # view(-1, ...) exports to ONNX as a constant-shape Reshape, which
        # OpenCV's cv::dnn importer handles reliably.
        row_logits = self.head(self.reduce(f)).view(-1, NUM_SLOTS, self.num_rows, self.num_classes)
        exists_logits = self.exists_head(f)
        return row_logits, exists_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
