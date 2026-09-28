# Atlas Vision: training a lane network

The C++ app finds lanes with a classical pipeline: colour and edge
thresholds, a bird's-eye view, sliding windows, polynomial fits and a Kalman
tracker. This folder trains a small neural network to do the same job from
the app's own recordings. The C++ side can later run the network through
`cv::dnn`.

Everything here runs on the Mac. On Apple Silicon, PyTorch uses the GPU
through its **MPS** (Metal) backend. The package is unit-tested; it has
not been trained or scored on real sessions or TuSimple yet.

## The loop

```
  collect ──► label ──► train ──► evaluate ──► improve ──► redeploy
     ▲                                                        │
     └────────────────────────────────────────────────────────┘
```

1. **Collect**: drive, or replay a video, with logging on (`--log`). The
   logger (`vision/data_logger.cpp`) saves keyframes plus "interesting"
   frames (status changes, low confidence, rejected measurements) to
   `data/sessions/<timestamp>_<source>/`, together with the tracker's lanes.
2. **Label**: most frames need no manual label, because the tracker's output
   is the label (a *pseudo-label*, explained below). You only mark frames as
   `good` or `bad` in the app. Corrections go to `labels.jsonl`.
3. **Train**: `train.py` learns from confident pseudo-labels, human-approved
   frames and, optionally, the public TuSimple dataset.
4. **Evaluate**: validation runs on *whole recordings* the network never saw
   during training. `predict.py` writes TuSimple-format predictions;
   `atlas_eval --score PRED.json --labels GT.json` scores them with the
   official metric (extra keys such as `slots` and `frame_id` are ignored).
5. **Improve**: look at the failures (`predict.py` visualisations, the
   worst frames), label them, record more of what the network finds hard
   (night, rain, worn paint), then retrain.
6. **Redeploy**: `export_onnx.py` writes `atlas_lanes.onnx` plus a JSON
   sidecar that the C++ app reads.

## Setup (Mac)

```bash
cd ~/Projects/Self-driving
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
python -c "import torch; print(torch.__version__, 'mps:', torch.backends.mps.is_available())"
```

The standard `torch` wheel for macOS arm64 already includes MPS. If a
PyTorch operation is not implemented on MPS yet, run with
`PYTORCH_ENABLE_MPS_FALLBACK=1` so that operation falls back to the CPU.

The optional runtime check needs `pip install onnxruntime`.

Run the tests (numpy-only tests run anywhere; model tests need torchvision):

```bash
python -m pytest training/tests -q
```

## Commands

All commands run from the repo root. Paste one command at a time.

Inspect how the data would be split, by recording (optional):

```bash
PYTHONPATH=training python -m atlas_data.split --sessions "data/sessions/*" \
    --tusimple ~/datasets/tusimple/train_set --out training/runs/split.json
```

Quick smoke test: a few samples, 1 epoch, checks everything is wired up:

```bash
python training/train.py --sessions "data/sessions/*" --max-samples 64 --epochs 1 \
    --batch-size 8 --workers 0 --out training/runs/smoke
```

Real training: your sessions plus TuSimple ground truth:

```bash
python training/train.py --sessions "data/sessions/*" \
    --tusimple ~/datasets/tusimple/train_set \
    --epochs 30 --batch-size 16 --out training/runs/first
```

Predictions and visualisations on a session and on the TuSimple test set:

```bash
python training/predict.py --checkpoint training/runs/first/best.pt \
    --session data/sessions/<one_session> --out training/runs/first/pred
python training/predict.py --checkpoint training/runs/first/best.pt \
    --tusimple-json ~/datasets/tusimple/test_set/test_label.json \
    --tusimple-root ~/datasets/tusimple/test_set --out training/runs/first/pred
```

Score those predictions with the C++ official-metric port (`docs/EVALUATION.md`):

```bash
./build/atlas_eval --score training/runs/first/pred/test_label.predictions.json \
    --labels ~/datasets/tusimple/test_set/test_label.json \
    --out training/runs/first/pred/score.json
```

Export for C++ (writes `training/runs/first/atlas_lanes.onnx` and `atlas_lanes.json`):

```bash
python training/export_onnx.py --checkpoint training/runs/first/best.pt
```

Useful `train.py` options:

| option | meaning |
|---|---|
| `--min-confidence 0.8` | pseudo-label confidence threshold |
| `--statuses LOCKED` | tracker statuses accepted as pseudo-labels |
| `--human-only` | train only on frames you marked `good` |
| `--val-sessions NAME` | force specific recordings into validation |
| `--split-file F` | reuse a saved split so runs can be compared fairly |
| `--backbone mobilenet_v3_small` | a smaller, faster backbone |
| `--smooth-weight`, `--exists-weight` | auxiliary loss weights (0 turns a loss off) |
| `--resume RUN/last.pt` | continue an interrupted run |

Each run folder contains `metrics.csv`. Open it in any spreadsheet, or with
`pandas`, and plot `train_total` against `val_loss`. If training loss keeps
falling while validation loss rises, the model is **overfitting**: it is
memorising the training drives. The fix is more varied recordings, stronger
augmentation or fewer epochs.

TuSimple is available on Kaggle ("TuSimple lane detection"). It is
1280x720 highway footage with human annotations. Point `--tusimple` at the
folder that contains `clips/` and `label_data_*.json`.

## Pseudo-labels: learning from our own detector

Hand-labelling lanes is slow. Our tracker already outputs lanes for every
frame, and the logger stores them TuSimple-style (an x pixel per image row).
Using a model's own output as training labels is called **pseudo-labelling**.

The catch is that a network trained on the detector's output learns to copy
the detector, **mistakes included**. If the classical pipeline mistakes a
tar seam for a lane line, the network learns that too, and it learns it
confidently. We reduce that risk in layers:

1. **Keep only the frames the tracker was sure about.** By default a frame
   is used only if the tracker was `LOCKED` with confidence >= 0.8
   (`atlas_data/sessions.py`, `LabelPolicy`). `PARTIAL`, `COASTING` and
   `SEARCHING` frames are skipped.
2. **Human review wins.** A frame marked `bad` is never used. A frame marked
   `good` is used even when the tracker was unsure. Those are the most
   valuable labels, because they are exactly the hard cases the classical
   pipeline nearly got wrong. Later entries in `labels.jsonl` override
   earlier ones.
3. **Do not label what was not seen.** The road model always *predicts*
   outer lane lines 1.5 lane widths out, even on a two-lane road. An outer
   slot is kept only if a detection was actually accepted for it on that
   frame, which matches what the dashboard draws.
4. **Real ground truth.** Mixing in TuSimple's human annotations pulls the
   network towards the truth rather than towards our detector. It also
   gives an external benchmark that our detector's biases cannot inflate.
5. **Measure against something independent.** Scoring the network against
   pseudo-labels only tells you how well it imitates the tracker. The honest
   numbers come from TuSimple test images and from human-reviewed frames.

`train.py` prints how many frames were kept and why the others were
dropped, for example:

```
  excluded_human_bad                 12
  excluded_low_confidence            340
  excluded_status_PARTIAL            95
  included_human_good                41
  included_pseudo_label              2210
  slot_dropped_unmeasured_outer      1830
```

## How the network sees lanes: row anchors

Following Ultra-Fast-Lane-Detection (UFLD, Qin et al. 2020), we do not
segment pixels. We pick **R fixed image rows** (row anchors) and ask, for
each of the **4 lane slots** (`left_outer, left, right, right_outer`) and
each row, *which of C column bins does the lane cross, or is it absent?*

That turns lane detection into `4 x R` small classification problems with
`C + 1` classes each, trained with ordinary cross-entropy.
`atlas_data/row_anchor.py` explains the formulation in detail.
`atlas_train/model.py` explains why a fully-connected head that sees the
whole image suits the problem.

- Input: the lower 64% of the image (the sky has no lanes), resized to
  800x288. That keeps the 16:9 aspect ratio, so nothing is stretched.
- Defaults: R = 40 rows and C = 100 bins (12.8 px per bin at 1280 wide).
  Sub-bin precision comes from a softmax-weighted average around the best
  bin.
- **Splits are by recording, not by frame.** Consecutive frames are almost
  identical, so a frame-level split lets validation "cheat". See
  `atlas_data/split.py`.
- **Augmentation keeps labels consistent.** Brightness, contrast and
  shadows leave labels unchanged. A horizontal shift moves the x values. A
  mirror flip mirrors the x values and swaps left and right slots. See
  `atlas_data/augment.py`.

## ONNX contract (for the C++ integration)

The sidecar JSON has every number below in machine-readable form.

**Input** `input`: float32 `[1, 3, 288, 800]` (NCHW, RGB).

1. Take the undistorted frame (W x H).
2. Crop rows `[round(crop_top_frac * H), H)` at full width
   (`crop_top_frac = 0.36` by default).
3. Resize to 800x288 with bilinear interpolation.
4. Convert BGR to RGB, apply `x / 255`, subtract
   `mean = (0.485, 0.456, 0.406)` and divide by
   `std = (0.229, 0.224, 0.225)` per channel.

**Output** `row_logits`: float32 `[1, 4, R, C+1]` (default `[1, 4, 40, 101]`),
raw scores with no softmax applied.

- Slot order is `left_outer, left, right, right_outer`. This is **not** the
  C++ `LaneSlot` enum order.
- Row anchor `r` is at image row
  `v_r = top + (r + 0.5) * (H - top) / R - 0.5`, where
  `top = round(crop_top_frac * H)`. The sidecar also lists these rows for
  1280x720 as `row_anchors_ref_px`.
- Class `j < C` means "the lane crosses column bin j", whose centre is at
  `x_j = (j + 0.5) * W / C - 0.5`. Class `C` means "no lane on this row".

**Output** `exists_logits`: float32 `[1, 4]`. Apply a sigmoid; a value
above 0.5 means the slot holds a lane. Use it only if the sidecar's
`exists_trained` is true.

**Decoding**, per slot and row:

1. If the argmax over all `C+1` classes is `C`, the lane is absent on that
   row.
2. Otherwise let `b` be the best *location* bin. Take the softmax over bins
   `[b-2, b+2]` and compute `x = sum(p_j * x_j)`.
3. Keep a lane only if it has at least 2 present rows (and passes the exists
   check when that head was trained).

`atlas_data.row_anchor.decode_logits` is the reference implementation.

## Files

- `atlas_data/`: data code, torch-free except `torch_dataset.py`. It covers
  sessions and the label policy, TuSimple, row-anchor encoding and decoding,
  augmentation, recording-level splits and preprocessing.
- `atlas_train/`: the model, losses, numpy metrics, device selection and
  checkpoint loading.
- `train.py`, `predict.py`, `export_onnx.py`: the command-line tools
  described above.
- `tests/`: pytest tests for the label policy, row-anchor round trips, flip
  and shift consistency, split leakage, TuSimple slot assignment and
  metrics.
