"""Validation metrics (numpy only, so they are unit-testable without torch).

* row_acc: fraction of (slot, row) cells whose predicted class equals the
  target class - including "absent". Easy to compute, but inflated by the
  many trivially-absent cells, so do not read too much into it alone.
* presence_recall / false_presence: how often a present cell is predicted
  present, and how often an absent cell is wrongly predicted present.
* px_err: mean |x_pred - x_true| in source pixels over cells where both
  agree the lane is present. This is the "how precise" number.
* tusimple_acc / fp / fn: an approximation of the TuSimple benchmark. For
  every ground-truth lane: accuracy = fraction of its points predicted
  within 20 px (scaled to image width / 1280); the image score is the mean
  over its lanes. A predicted lane is a false positive if its accuracy
  against the same-slot ground truth is below 85% (or there is none); a
  ground-truth lane is a false negative under the same test.
  Differences from the official script: we compare slot-to-slot on our row
  anchors instead of best-matching lanes on h_samples, and we do not widen
  the threshold for steep lanes (official uses 20 px / cos(angle)).
  The C++ atlas_eval tool computes the official-style numbers on
  predict.py's TuSimple json.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LaneMetrics:
    def __init__(self, absent_class: int, px_threshold_at_1280: float = 20.0, lane_acc_threshold: float = 0.85):
        self.absent_class = absent_class
        self.px_threshold_at_1280 = px_threshold_at_1280
        self.lane_acc_threshold = lane_acc_threshold
        self.cells = 0
        self.correct = 0
        self.gt_present = 0
        self.pred_present_on_gt = 0
        self.gt_absent = 0
        self.pred_present_on_absent = 0
        self.px_err_sum = 0.0
        self.px_err_n = 0
        self.image_acc_sum = 0.0
        self.images_with_gt = 0
        self.gt_lanes = 0
        self.pred_lanes = 0
        self.fn = 0
        self.fp = 0

    def update(
        self,
        pred_cls: np.ndarray,
        pred_x: np.ndarray,
        gt_cls: np.ndarray,
        gt_x: np.ndarray,
        widths: np.ndarray,
    ) -> None:
        """All arrays (B, S, R) except widths (B,). x in source pixels, negative = absent."""
        pred_cls, gt_cls = np.asarray(pred_cls), np.asarray(gt_cls)
        pred_x, gt_x = np.asarray(pred_x, np.float64), np.asarray(gt_x, np.float64)
        widths = np.asarray(widths, np.float64).reshape(-1)

        gt_p = gt_cls != self.absent_class
        pr_p = pred_cls != self.absent_class
        self.cells += gt_cls.size
        self.correct += int((pred_cls == gt_cls).sum())
        self.gt_present += int(gt_p.sum())
        self.pred_present_on_gt += int((gt_p & pr_p).sum())
        self.gt_absent += int((~gt_p).sum())
        self.pred_present_on_absent += int((~gt_p & pr_p).sum())

        both = gt_p & pr_p & (pred_x >= 0) & (gt_x >= 0)
        err = np.abs(pred_x - gt_x)
        self.px_err_sum += float(err[both].sum())
        self.px_err_n += int(both.sum())

        for b in range(gt_cls.shape[0]):
            thr = self.px_threshold_at_1280 * widths[b] / 1280.0
            lane_accs = []
            for s in range(gt_cls.shape[1]):
                g = gt_p[b, s]
                p = pr_p[b, s]
                acc = 0.0
                if g.sum() >= 2:
                    hit = g & p & (err[b, s] < thr)
                    acc = float(hit.sum()) / float(g.sum())
                    lane_accs.append(acc)
                    self.gt_lanes += 1
                    if acc < self.lane_acc_threshold:
                        self.fn += 1
                if p.sum() >= 2:
                    self.pred_lanes += 1
                    if g.sum() < 2 or acc < self.lane_acc_threshold:
                        self.fp += 1
            if lane_accs:
                self.image_acc_sum += float(np.mean(lane_accs))
                self.images_with_gt += 1

    def summary(self) -> dict[str, Any]:
        def ratio(a: float, b: float) -> float:
            return float(a) / float(b) if b else float("nan")

        return {
            "row_acc": ratio(self.correct, self.cells),
            "presence_recall": ratio(self.pred_present_on_gt, self.gt_present),
            "false_presence": ratio(self.pred_present_on_absent, self.gt_absent),
            "px_err": ratio(self.px_err_sum, self.px_err_n),
            "tusimple_acc": ratio(self.image_acc_sum, self.images_with_gt),
            "fp": ratio(self.fp, self.pred_lanes),
            "fn": ratio(self.fn, self.gt_lanes),
        }
