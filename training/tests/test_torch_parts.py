"""Tests that need PyTorch (losses) or torchvision (model). Skipped when unavailable."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed: loss/model tests skipped")

from atlas_data.row_anchor import RowAnchorGeometry
from atlas_train.losses import LaneLoss, smoothness_loss

GEOM = RowAnchorGeometry(input_w=256, input_h=96, num_rows=8, num_bins=20, crop_top_frac=0.36)


def test_losses_are_finite_and_smoothness_zero_for_constant_rows():
    b, s, r, k = 2, 4, GEOM.num_rows, GEOM.num_classes
    logits = torch.randn(b, s, r, k, requires_grad=True)
    target = torch.randint(0, k, (b, s, r))
    total, parts = LaneLoss(GEOM.absent_class)(logits, torch.randn(b, s), target, torch.ones(b, s))
    total.backward()
    assert np.isfinite(parts["total"]) and logits.grad is not None

    same = torch.randn(b, s, 1, k).expand(b, s, r, k)
    present = torch.zeros(b, s, r, dtype=torch.long)
    assert float(smoothness_loss(same, present, GEOM.absent_class)) < 1e-6
    absent = torch.full((b, s, r), GEOM.absent_class)
    assert float(smoothness_loss(torch.randn(b, s, r, k), absent, GEOM.absent_class)) == 0.0


def test_model_output_shapes():
    pytest.importorskip("torchvision", reason="torchvision not installed: model test skipped")
    from atlas_train.model import LaneNet, ModelConfig

    for backbone in ("resnet18", "mobilenet_v3_small"):
        model = LaneNet(ModelConfig(backbone=backbone, pretrained=False, hidden_dim=64, geometry=GEOM)).eval()
        with torch.no_grad():
            rows, exists = model(torch.zeros(2, 3, GEOM.input_h, GEOM.input_w))
        assert rows.shape == (2, 4, GEOM.num_rows, GEOM.num_classes)
        assert exists.shape == (2, 4)
