# Measuring lane accuracy: `atlas_eval`

The run summary of `atlas --video ... --headless` says whether perception was
*stable* (status shares, resets, jitter). Stable is not the same as *right*: a
tracker can lock confidently onto a crack in the asphalt. `atlas_eval` compares
Atlas Vision against lanes that **humans** labelled:

| Mode | Ground truth | Answers |
| --- | --- | --- |
| `--tusimple` | TuSimple benchmark: 2782 highway images with lane polylines | How many lane points do we put within ~20 px of the truth? Within how many cm? |
| `--score PRED.json --labels GT.json` | Same TuSimple JSON, scored against an external prediction file (e.g. `training/predict.py`) | Official accuracy / FP / FN without running the C++ pipeline |
| `--session` | Your own `good` / `bad` presses in `atlas --review` | Which tracker states and logging triggers produce bad output? Does our confidence mean anything? |

Build: `cmake --build build` produces `build/atlas_eval` next to `atlas`. The metric is unit-tested; none of these modes has been run on the real TuSimple test set or a real logged drive yet.

---

## 1. TuSimple benchmark

### Download

The official repository is <https://github.com/TuSimple/tusimple-benchmark>.
Its README only points to [issue #3](https://github.com/TuSimple/tusimple-benchmark/issues/3),
which says the original download links are dead and the dataset now lives on Kaggle:
<https://www.kaggle.com/datasets/manideep1108/tusimple> (free Kaggle account needed).
Plan on roughly **20+ GB** of download for train + test (check the Kaggle page
for the exact size); the test set alone is about half.

What you need for evaluation:

- `test_set/clips/...`: 2782 clips, each a folder of 20 frames `1.jpg` .. `20.jpg`
  (1280x720, 20 fps, US highways, day time).
- `test_label.json`: one JSON object per line for the **last** frame of every clip:

```json
{"lanes": [[-2, -2, 632, 625, ...], [...]], "h_samples": [240, 250, ..., 710], "raw_file": "clips/0530/1492626760788443246_0/20.jpg"}
```

`h_samples` are image rows; each lane lists the x pixel on every row, `-2`
where the lane is not visible. The training set (`train_set/`, 3626 clips,
labels in `label_data_0313.json`, `label_data_0531.json`, `label_data_0601.json`)
has the same format and is useful for fitting the camera mount without
touching the test labels (see below).

A comment on issue #3 reports fewer test images in `test_set.zip` than lines in
`test_label.json`. Clips with missing frames are counted as "no prediction"
(accuracy 0) and reported as `missing` in the summary, so the denominator stays
honest.

### Commands

Quick look: first 200 clips, camera mount fitted to the labels.

```bash
./build/atlas_eval --tusimple ~/data/tusimple/test_set --labels ~/data/tusimple/test_label.json --limit 200
```

Full run, results plus an official-format prediction file:

```bash
./build/atlas_eval --tusimple ~/data/tusimple/test_set --labels ~/data/tusimple/test_label.json \
    --fit-labels ~/data/tusimple/train_set/label_data_0313.json --save-camera config/tusimple.yml \
    --out results.json --predictions pred.json
```

Later runs, reusing the fitted camera:

```bash
./build/atlas_eval --tusimple ~/data/tusimple/test_set --labels ~/data/tusimple/test_label.json \
    --camera-model config/tusimple.yml --out results.json
```

`ROOT` (the `--tusimple` argument) is the folder that contains `clips/`;
`raw_file` paths are resolved relative to it. `--labels` can be repeated.
All options: `atlas_eval --help`.

Cost: every clip runs 20 frames through a fresh pipeline, so the full test set
is ~55 000 frames: several minutes to tens of minutes depending on the machine.
Start with `--limit`. Object detection is off by default (TuSimple scoring does
not need it and it is the slowest stage); `--objects` turns it on so that cars
are masked out of the lane search like in a normal run.

### What happens per clip

1. A new `VisionPipeline` (tracker in SEARCHING) is created.
2. Frames `1.jpg` .. `20.jpg` go through it with `media_time = i / 20 s`.
   Frames 1-19 are the tracker's warm-up; only frame 20 is labelled.
3. The tracked road model at frame 20 is rasterised on the label's `h_samples`
   with the same rule the dashboard and the data logger use: a point is drawn
   only where the 1-sigma lateral uncertainty of that boundary is below
   0.4 m. The near limit is 0 m (TuSimple labels reach the bottom of the image,
   ~4 m ahead; the logger stops at 6 m) and the far limit 40 m (`--near-m`,
   `--far-m`). Up to four lanes are predicted: ego left and right always, and
   each outer lane only if the pipeline accepted a measurement of it in the
   last 1 s (the same memory the pipeline uses to decide whether to search for
   it); `--all-slots` always predicts them. Lanes with fewer than two drawn
   points are dropped.
4. With a lens-distortion model, the lanes are drawn in the **original**
   image (distorted), because that is what was labelled.

### The official metric, exactly

`tools/tusimple.cpp` is a line-by-line port of
[`evaluate/lane.py`](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py)
(`LaneEval.bench`, `bench_one_submit`). For one image with gt lanes `G` and
predicted lanes `P`, all sampled on the same `h_samples`:

- **Early outs**: if `run_time > 200 ms` or `|P| > |G| + 2`, the image scores
  accuracy 0, FP 0, FN 1.
- **Threshold per gt lane**: fit `x = k y + b` (ordinary least squares, the
  script uses sklearn `LinearRegression`) to the gt lane's valid points;
  `angle = atan(k)` (0 with fewer than 2 points); `thresh = 20 / cos(angle)` px.
  Slanted lanes get a wider horizontal tolerance, because a horizontal 20 px
  error on a slanted line is a smaller perpendicular error.
- **Point accuracy** of prediction `p` against gt lane `g`: replace every
  missing value (`< 0`) in both by `-100`; count rows where `|p - g| < thresh`;
  divide by the number of **all** `h_samples`, not only the labelled rows. So
  rows where both are missing count as hits, and drawing a lane beyond the
  labelled end (or leaving a labelled row empty) counts as a miss.
- For every gt lane take the best point accuracy over all predictions. It is
  **matched** if that is `>= 0.85`, otherwise it is a false negative.
- `FP = (|P| - matched) / |P|` (0 if there are no predictions),
  `FN = fn / max(min(|G|, 4), 1)`,
  `Accuracy = sum(best accuracies) / max(min(|G|, 4), 1)`.
- **More than 4 gt lanes**: the lowest best-accuracy is subtracted from the
  sum and one false negative is forgiven (if there is one).
- The dataset scores are the plain means over images.

Points where the port could differ from the script, and what was chosen:

- **Matching is not one-to-one.** Several gt lanes may "match" the same
  prediction, so `|P| - matched` can in principle go negative. The script does
  this; the port does too.
- **Denominator.** The script divides by the number of lines in the gt file and
  refuses to run unless every gt image has a prediction. With `--limit N` the
  port divides by N (the images evaluated). Duplicate `raw_file` entries would
  be collapsed by the script's dict; the port counts them separately (the
  official test file has none, as far as I know).
- **`run_time`.** We pass the pipeline's processing time of the labelled frame
  (not including JPEG decoding or the 19 warm-up frames). On any reasonable
  machine this is far below 200 ms, so the rule never fires, but it is
  implemented.
- **Rounding.** Predictions are rounded to integer pixels, as the labels are.
- To check the port against the original, run `atlas_eval ... --predictions
  pred.json` on the **full** test set and then the official script:
  `python2 evaluate/lane.py pred.json test_label.json` (it needs `ujson`,
  `numpy`, `scikit-learn`; it is Python 2 code, `print` statements). The numbers
  should agree to rounding.

Note what the metric rewards: it is a per-point, per-lane test at a fixed pixel
tolerance, with no penalty for predicting nothing (FP = 0) apart from lost
accuracy. Published deep-learning methods score 95-97 % accuracy; a classical
BEV pipeline with an unknown camera should expect much less. The interesting
use is **relative**: did a change move the number, and in which tracker state?

### Our own extra metrics

- **Tracker state at the labelled frame** (share of clips per status, and the
  accuracy within each), plus the share of clips where the tracker reached
  LOCKED on *any* of the 20 frames. A low LOCKED share means 20 frames (1 s)
  is not enough warm-up, or initialisation fails on this camera.
- **Lateral error in metres at 10, 20 and 30 m ahead.** Every gt lane is
  back-projected to the road through the same camera model (flat ground),
  giving its lateral position `Y` at that distance. The ego lines are the
  labelled lanes nearest to `Y = 0` on each side. Error = `|Y_pred - Y_gt|`
  for the tracked left / right boundaries. The table shows how many ego lines
  were labelled at that distance, what share we predicted there (sigma
  < 0.4 m), and the mean / median / 90th percentile error of those. Pixels
  make near errors look huge and far errors tiny; metres say what a controller
  would feel.

`results.json` has the summary, the camera used, and one entry per image
(accuracy, FP, FN, number of predicted / gt lanes, status, confidence, lateral
errors; `null` = not labelled at that distance, `"missed"` = labelled but not
predicted). Sort it by accuracy to find the clips to look at.

### The camera: TuSimple publishes no intrinsics

Atlas Vision measures lanes in metres on the road plane, so it needs the
camera's focal length and its mount (height, pitch, yaw). TuSimple provides
none of them. Options, best first:

1. **Assumed FOV + mount fitted to the labels (the default without
   `--camera-model`).** Intrinsics come from `--fov` (default 60 degrees
   horizontal, principal point at the image centre, no distortion). The
   mount is then solved from human labels (`tools/tusimple.cpp`,
   `fit_mount_from_labels`):
   - On each of the first `--fit-images` (300) label images, a straight line
     `x = a y + b` is fitted to the near two thirds of every lane with >= 6
     points (lanes with > 3 px RMS residual are curved and skipped).
   - The lines' least-squares intersection is that image's vanishing point
     (skipped if the lines are nearly parallel or miss it by > 15 px).
   - The median vanishing point `(u, v)` over images gives
     `pitch = atan((cy - v) / fy)` and `yaw = atan((u - cx) cos(pitch) / fx)`
     (the forward direction projects to `u = cx + fx tan(yaw) / cos(pitch)`,
     `v = cy - fy tan(pitch)`).
   - With that pitch and yaw and a camera height of 1 m, the ego lines are
     back-projected to the ground; the median distance between them is the
     lane width at unit height. All ground distances scale with camera
     height, so `height = 3.7 m / width_at_h1` (`--lane-width`).
   Use `--fit-labels train_set/label_data_*.json` so that the test labels are
   only used for scoring. Fitting on the test labels is an "oracle" calibration:
   it cannot tune lane positions, but it does use test information.
   `--save-camera` writes the result as a normal camera model YAML.
2. **`atlas --estimate-mount`** on a TuSimple-like video (it does the same
   vanishing-point / lane-width estimate from the tracker's own detections),
   then `--camera-model`. TuSimple clips are image folders; turn one into a video
   first, e.g. `ffmpeg -framerate 20 -i clips/.../%d.jpg clip.mp4`, or
   concatenate several straight-road clips.
3. `--no-fit-mount`: level camera 1.4 m high. Only useful to see how much
   the mount matters.

The FOV itself cannot be recovered from lane labels on a flat straight road
(focal length and height trade off). A wrong FOV mostly scales all metric
distances; pixel accuracy is affected less because the lanes are drawn back
into the same image. Try `--fov 50`, `60`, `70` and keep the best, remembering
that this is tuning on the benchmark.

### Caveats

- **Unknown intrinsics** (above). Lateral errors in metres are in the metric
  of the assumed camera, not the real one; they are comparable between runs
  with the same camera, not with other papers.
- **Flat ground.** Both the tracker and the back-projection of the labels
  assume a planar road. Grade changes make far lanes wrong in both, which
  partly cancels in the metre metric and not at all in the pixel metric.
  The pipeline's horizon estimator adapts pitch per frame; the frame-20
  prediction and the back-projection use the pitch that frame was measured
  with.
- **20-frame warm-up.** The tracker needs a few frames to initialise and more
  to reach LOCKED; with 1 s of video some clips end while it is still PARTIAL.
  The labelled frame is scored whatever the state (SEARCHING = no lanes). Our
  temporal filter is an advantage over single-image methods here only if it
  locks within the clip.
- **Outer lanes** assume equal lane widths (`slot_width_multiple`); on roads
  with a narrower shoulder they will be off.
- **Near field.** We extrapolate the road polynomial from 6 m down to the
  bottom of the image. TuSimple labels extend there, and leaving those ~15
  rows empty would cost ~30 % point accuracy per lane.

---

## 2. Scoring an external prediction file

`atlas_eval --score PRED.json --labels GT.json` scores a TuSimple-format
prediction file (one JSON object per line) against labels, using the same
official metric as `--tusimple` (`tools/tusimple.cpp`). Predictions are
matched to labels by `raw_file`. Extra keys such as `slots` and `frame_id`
(written by `training/predict.py`) are ignored.

- Missing predictions score as no lanes.
- If `h_samples` is present and differs from the label, that image is a
  format error and also scores as no lanes. Official submission lines omit
  `h_samples`; that is fine.
- Duplicate `raw_file` lines: the last one wins.
- Predictions with no matching label are ignored.
- `--labels` can be repeated. `--out FILE.json` writes the summary and
  per-image scores.

This path is unit-tested; it has not been run on the real TuSimple test set.

```bash
./build/atlas_eval --score training/runs/first/pred/test_label.predictions.json \
    --labels ~/data/tusimple/test_label.json --out score.json
```

---

## 3. Human labels on logged sessions

Log a drive, then mark frames good or bad in the review tool (`g` / `b`,
`a` / `d` to step). `--log` writes `data/sessions/<time>_<name>/`.

```bash
./build/atlas --video drive.mp4 --log
./build/atlas --review data/sessions/<dir>
./build/atlas_eval --session data/sessions/<dir> --out session_eval.json
```

`frames.jsonl` has one record per logged frame (why it was logged, tracker
status, confidence, ...); `labels.jsonl` has one `{"frame_id":N,"human":"good"}`
line per key press in the review tool (the last press for a frame wins). A
label given live with `g` / `b` during `atlas --video` is stored in the record
itself and is used unless `labels.jsonl` overrides it.

The report shows:

- **labelled good / bad / unlabelled** frames;
- **by tracker status**: e.g. "80 % of LOCKED frames are good, 30 % of
  COASTING frames" tells you where the tracker lies;
- **by logging reason** (`keyframe`, `status_change`, `tracker_event`,
  `low_confidence`, `rejected_measurement`, `human_label`); a frame logged for
  several reasons counts once per reason. `keyframe` frames are the unbiased
  sample; the others are deliberately hard cases;
- **confidence calibration**: good vs bad counts in 5 confidence bins, the
  mean / median confidence of good and bad frames, and
  - **AUC** = P(a random good frame has higher confidence than a random bad
    one). 0.5 = confidence carries no information, 1.0 = it ranks perfectly.
    This is the first number to check: a useful confidence must rank.
  - **Brier score** = mean of `(confidence - [good])^2` and **ECE** (expected
    calibration error) = sum over bins of `share * |mean confidence - share
    good|`. These treat confidence as "probability that a human calls this
    good". The tracker's confidence is defined from the lateral sigma at 15 m,
    not as a probability, so a high ECE with a good AUC just means the scale is
    off (fixable with a monotone mapping), while a low AUC means the signal
    itself is wrong.

Keep in mind that most logged frames are *not* random: triggers select the
interesting moments, so "% good" per reason is not the tracker's overall
accuracy. Label a few hundred `keyframe` frames for that.

---

## Code map

| File | What |
| --- | --- |
| `tools/eval_lanes.cpp` | `atlas_eval` command line (`--tusimple`, `--score`, `--session`) |
| `tools/tusimple.{hpp,cpp}` | label parsing, official metric port, mount fit from labels |
| `tools/tusimple_score.{hpp,cpp}` | `--score`: load prediction files, match by `raw_file`, apply the metric |
| `tools/tusimple_eval.{hpp,cpp}` | per-clip pipeline runs, prediction, lateral error, report |
| `tools/lane_rasterize.{hpp,cpp}` | road model to x-per-row lanes (the logger's `lanes_image` math; distorted-image variant) |
| `tools/session_eval.{hpp,cpp}` | session label report |
| `tests/test_eval.cpp` | metric cases, `--score` matching, rasteriser geometry, mount fit, session parsing |
