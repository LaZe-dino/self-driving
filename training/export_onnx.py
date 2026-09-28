"""Export a trained checkpoint to ONNX for the C++ app (cv::dnn), plus a JSON sidecar.

Example:
    python training/export_onnx.py --checkpoint training/runs/first/best.pt
    -> training/runs/first/atlas_lanes.onnx and atlas_lanes.json

ONNX output contract (N = 1, fixed input size; S = 4 slots, R rows, C bins):

    input          float32 [1, 3, input_h, input_w]  RGB, NCHW, (x/255 - mean) / std
    row_logits     float32 [1, S, R, C+1]  raw scores (no softmax). For slot s
                   (order in sidecar "slots") and row anchor r (image row
                   v_r = top + (r+0.5)*(H-top)/R - 0.5, top = round(crop_top_frac*H);
                   listed for 1280x720 in "row_anchors_ref_px"): classes 0..C-1
                   are column bins, bin j centred on x = (j+0.5)*W/C - 0.5
                   in the undistorted image; class C means "no lane on this row".
    exists_logits  float32 [1, S]  raw score that slot s holds a lane
                   (sigmoid > 0.5 = yes). Only meaningful if the sidecar says
                   "exists_trained": true.

Decoding (mirrors atlas_data.row_anchor.decode_logits): per (s, r), if argmax
over all C+1 is C, the lane is absent there; otherwise take the softmax over
the location bins within +-window of the best bin and average their centres.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from atlas_data.preprocess import IMAGENET_MEAN, IMAGENET_STD
from atlas_data.sample import SLOT_ORDER
from atlas_train.utils import load_checkpoint

REFERENCE_SIZE = (1280, 720)


def build_sidecar(geom, ckpt: dict, onnx_name: str, window: int) -> dict:
    ref_w, ref_h = REFERENCE_SIZE
    exists_w = float(ckpt.get("train_args", {}).get("exists_weight", 0.0))
    return {
        "format": "atlas_lanes_row_anchor",
        "version": 1,
        "onnx": onnx_name,
        "input": {
            "name": "input",
            "shape": [1, 3, geom.input_h, geom.input_w],
            "layout": "NCHW",
            "dtype": "float32",
            "color": "RGB",
            "image_space": "undistorted",
            "crop_top_frac": geom.crop_top_frac,
            "crop": "keep rows [round(crop_top_frac*H), H) at full width, then bilinear resize to input_w x input_h",
            "scale": 1.0 / 255.0,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "normalisation": "value = (pixel/255 - mean[c]) / std[c], channel order RGB",
        },
        "outputs": {
            "row_logits": {
                "shape": [1, len(SLOT_ORDER), geom.num_rows, geom.num_classes],
                "meaning": "raw class scores per slot and row anchor; classes 0..num_bins-1 = column bin, "
                           "num_bins = absent",
            },
            "exists_logits": {
                "shape": [1, len(SLOT_ORDER)],
                "meaning": "raw score that the slot holds a lane; sigmoid > 0.5 means present",
            },
        },
        "slots": list(SLOT_ORDER),
        "input_w": geom.input_w,
        "input_h": geom.input_h,
        "num_rows": geom.num_rows,
        "num_bins": geom.num_bins,
        "absent_class": geom.absent_class,
        "row_anchor_px": "v_i = top + (i + 0.5) * (H - top) / num_rows - 0.5, top = round(crop_top_frac * H)",
        "row_anchors_input_px": [round(float(v), 3) for v in geom.anchor_rows_input()],
        "reference_image_size": [ref_w, ref_h],
        "row_anchors_ref_px": [round(float(v), 3) for v in geom.anchor_rows_src(ref_h)],
        "bin_centre_px": "x_j = (j + 0.5) * image_width / num_bins - 0.5",
        "decode": {
            "absent": "argmax over all classes == absent_class",
            "x": "softmax over location bins within +-window of the best location bin; x = sum(p_j * x_j)",
            "window": window,
            "exists_threshold": 0.5,
            "min_points_per_lane": 2,
        },
        "exists_trained": exists_w > 0,
        "backbone": ckpt["model_config"]["backbone"],
        "epoch": ckpt.get("epoch"),
        "val_metrics": {k: v for k, v in ckpt.get("metrics", {}).items() if k.startswith("val_")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a lane checkpoint to ONNX (+ JSON sidecar).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="output .onnx (default: atlas_lanes.onnx next to the checkpoint)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--window", type=int, default=2, help="decode window recorded in the sidecar")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, "cpu")
    geom = model.cfg.geometry
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name("atlas_lanes.onnx")
    dummy = torch.randn(1, 3, geom.input_h, geom.input_w)

    export_kwargs = dict(
        opset_version=args.opset,
        input_names=["input"],
        output_names=["row_logits", "exists_logits"],
        do_constant_folding=True,
    )
    try:
        # The classic TorchScript exporter produces plain graphs that
        # OpenCV's ONNX importer understands best.
        torch.onnx.export(model, (dummy,), str(out), dynamo=False, **export_kwargs)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(model, (dummy,), str(out), **export_kwargs)
    print(f"wrote {out}")

    sidecar = build_sidecar(geom, ckpt, out.name, args.window)
    side_path = out.with_suffix(".json")
    with open(side_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"wrote {side_path}")

    if args.no_verify:
        return
    with torch.no_grad():
        ref_rows, ref_exists = (t.numpy() for t in model(dummy))
    x = dummy.numpy()

    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(out)))
        print("onnx.checker: OK")
    except ImportError:
        print("onnx not installed: skipped graph check")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        rows, exists = sess.run(["row_logits", "exists_logits"], {"input": x})
        print(f"onnxruntime vs torch: max |diff| row_logits {np.abs(rows - ref_rows).max():.2e}, "
              f"exists_logits {np.abs(exists - ref_exists).max():.2e}")
    except ImportError:
        print("onnxruntime not installed (optional): skipped runtime comparison. pip install onnxruntime")

    try:
        import cv2

        net = cv2.dnn.readNetFromONNX(str(out))
        net.setInput(x)
        rows, exists = net.forward(["row_logits", "exists_logits"])
        print(f"cv2.dnn vs torch:     max |diff| row_logits {np.abs(rows - ref_rows).max():.2e}, "
              f"exists_logits {np.abs(exists - ref_exists).max():.2e}  (this is what C++ will run)")
    except Exception as e:  # noqa: BLE001 - report any importer problem, it is the key compatibility test
        print(f"cv2.dnn check failed: {e}")


if __name__ == "__main__":
    main()
