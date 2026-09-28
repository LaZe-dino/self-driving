# Atlas Autonomy

A C++17 learning project that builds a camera-only driving perception stack step by step. It has two parts:

- **Waypoint sim** (`atlas` with no arguments): a kinematic bicycle model follows three waypoints with Pure Pursuit and prints `Path complete` after about 8.4 s of simulated time. It needs no OpenCV.
- **Atlas Vision** (`atlas --video FILE` or `atlas --webcam`): a modular perception pipeline in `vision/` that turns a dash-camera stream into metric lane geometry, tracked objects, ego-motion, a predicted path, a live dashboard and an optional training-data log.

Atlas Vision does not drive a car. The predicted path is a rollout of the simulated bicycle model on the perceived road.

## Architecture

Every stage writes into one `PerceptionFrame` per image. The dashboard and the data logger only read that struct; they never compute perception themselves.

```text
FrameSource (capture thread, bounded queue, reconnect)
  -> undistort                       chessboard intrinsics (config/*.yml)
  -> object detection + tracking     YOLOX-tiny ONNX; IoU association; ground-plane Kalman
                                     distance, relative velocity, time-to-collision
  -> bird's-eye view (BEV)           flat-ground homography H = K [r1 r2 t], 6-40 m ahead, +/-7 m, 5 cm/px
  -> ego-motion                      LK optical flow on road 12-40 m inside the lane lines,
                                     RANSAC rigid fit -> speed + yaw rate with uncertainty
  -> lane detection (in BEV)         ridge filter, MAD noise threshold, Lab b-channel for yellow,
                                     weighted least squares  Y = c0 + c1 X + c2 X^2  with covariance
  -> lane tracker                    Kalman road model [y_c, c1, c2, w], odometry motion model,
                                     chi-square gate, LOCKED / PARTIAL / COASTING / SEARCHING
  -> horizon estimator               online camera pitch from the lane vanishing point
  -> road geometry                   offset, heading, curvature, lane width, vanishing point
  -> path prediction                 Vehicle bicycle model + pure_pursuit_steering rollout
  -> Dashboard (window / snapshots)  and  DataLogger (--log)  ->  --review tool
```

## Quick start (macOS)

`data/` is gitignored. Copy or download the Udacity videos into `data/videos/` and (optionally) YOLOX into `data/models/` before running vision — see [Data](#data-not-in-git). Homebrew OpenCV often prints `VIDEOIO(FFMPEG): backend is generally available but can't be used to capture by name`; Atlas tries CAP_ANY, then FFmpeg, then AVFoundation, so that warning is handled.

```bash
brew install cmake opencv
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/atlas_tests
```

Recorded video, lanes only (no object model required):

```bash
./build/atlas --video data/videos/project_video.mp4 --camera-model config/udacity_camera.yml --no-objects
```

With objects, after `data/models/yolox_tiny.onnx` is present:

```bash
./build/atlas --video data/videos/project_video.mp4 --camera-model config/udacity_camera.yml
```

Live camera. Start with `--no-objects` so a missing YOLOX file does not quit the window:

```bash
./build/atlas --webcam --no-objects
```

The configure step must print `OpenCV <version> — Atlas Vision enabled` (4.7+ or 5.x). The first `--webcam` run triggers the macOS camera permission prompt; if it was denied, allow your terminal (or Cursor) in System Settings > Privacy & Security > Camera and restart it. If the default ONNX file is missing, Atlas now warns and keeps running with objects off; `--object-model FILE` still errors if that file is missing.

## Quick start (Windows, MSYS2 UCRT64)

Install the toolchain and libraries from an MSYS2 UCRT64 shell. HighGUI (the OpenCV window) needs Qt6; without `qt6-base` the exe fails to start.

```text
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja \
          mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-qt6-base
```

`C:\msys64\ucrt64\bin` must be on `PATH` so the exe can find the OpenCV and Qt DLLs. Then configure and build from PowerShell in the project folder:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build
```

CMake must print `OpenCV 5.x — Atlas Vision enabled`. The default build type is `RelWithDebInfo`; an unoptimised build makes the per-pixel code several times slower.

Run on a test video:

```powershell
.\build\atlas.exe --video data\videos\project_video.mp4 --camera-model config\udacity_camera.yml
```

Step-by-step Windows instructions and troubleshooting are in [HOW_TO_RUN.md](HOW_TO_RUN.md).

## Data (not in git)

`data/` is gitignored. Put these files there:

| File | Source |
| --- | --- |
| `data/videos/project_video.mp4`, `challenge_video.mp4`, `harder_challenge_video.mp4` | [udacity/CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines) (MIT) |
| `data/camera_cal/*.jpg` | same repo, `camera_cal/` (only needed to redo `--calibrate`) |
| `data/videos/p1_<name>.mp4` | [udacity/CarND-LaneLines-P1](https://github.com/udacity/CarND-LaneLines-P1) `test_videos/` (MIT), e.g. `p1_solidWhiteRight.mp4` |
| `data/models/yolox_tiny.onnx` | [YOLOX 0.1.1rc0 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx) (Apache-2.0); optional with `--no-objects` |

```bash
mkdir -p data/models
curl -L -o data/models/yolox_tiny.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
```

The calibrated camera models for these videos are already in `config/`.

## Command-line modes

Output of `atlas --help`:

```text
atlas --webcam [--camera-index N] [--no-objects]   live camera
atlas --video FILE [--loop]                  recorded video
   --camera-model FILE.yml   calibration (default: --fov 70 deg, level, 1.4 m high)
   --headless                no window; prints telemetry and a run summary
   --max-frames N            stop after N frames (default: run until the source ends)
   --snapshots DIR [--snapshot-every N]   save dashboard images
   --object-model FILE.onnx | --no-objects   (default data/models/yolox_tiny.onnx)
   --flow                    draw optical-flow vectors
   --log [--log-dir DIR] [--log-budget-mb MB]   record training data
   --process-width W         resize frames to width W right after capture (default: native;
                             1280 recommended for 1080p/4K phone video)
atlas --batch-log DIR --camera-model FILE.yml   headless + --log on every .mov/.mp4 in DIR
   [--process-width W] [--log-dir DIR] [--log-budget-mb MB] [--no-objects] [--max-frames N]
atlas --review SESSION_DIR [--snapshots DIR] browse / label logged training data
atlas --calibrate DIR --out FILE.yml         chessboard intrinsics from .jpg/.jpeg/.png photos
atlas --calibrate-video FILE --out FILE.yml  chessboard intrinsics from a video
   [--board 9x6] (inner corners) [--square-mm 25] [--process-width W]
atlas --estimate-mount FILE [--camera-model IN.yml | --fov DEG] --out OUT.yml [--lane-width 3.7]
   [--process-width W]
```

`--board` and `--square-mm` also apply to `--calibrate`. If a camera model's size differs from the processed frames but has the same aspect ratio, it is rescaled automatically (with a notice); a different aspect ratio is an error. Portrait videos are rejected.

Examples (Windows paths; on macOS use `./build/atlas` and `/`). Paste one command at a time; do not paste `#` comment suffixes into zsh.

Waypoint sim (no OpenCV needed):

```powershell
.\build\atlas.exe
```

Recorded video with the calibrated Udacity camera:

```powershell
.\build\atlas.exe --video data\videos\challenge_video.mp4 --camera-model config\udacity_camera.yml
```

Headless benchmark (no window; prints a stability summary at the end):

```powershell
.\build\atlas.exe --video data\videos\project_video.mp4 --camera-model config\udacity_camera.yml --headless
```

Save every 100th dashboard image:

```powershell
.\build\atlas.exe --video data\videos\p1_solidWhiteRight.mp4 --camera-model config\p1_camera.yml --snapshots data\snap\p1 --snapshot-every 100
```

Live webcam (uncalibrated: assumes 70 deg FOV, level, 1.4 m high). `--no-objects` is the first command to try:

```powershell
.\build\atlas.exe --webcam --no-objects --camera-index 0
```

Record training data, then review and label it:

```powershell
.\build\atlas.exe --video data\videos\challenge_video.mp4 --camera-model config\udacity_camera.yml --log
.\build\atlas.exe --review data\sessions\20260928_123259_challenge_video
```

Calibration:

```powershell
.\build\atlas.exe --calibrate data\camera_cal --out config\udacity_intrinsics.yml
.\build\atlas.exe --estimate-mount data\videos\project_video.mp4 --camera-model config\udacity_intrinsics.yml --out config\udacity_camera.yml
```

Phone footage: calibrate from a chessboard video, estimate the mount, auto-label a folder of drives:

```powershell
.\build\atlas.exe --calibrate-video data\videos\iphone_calib\calib.mov --out config\iphone_intrinsics.yml --board 9x6 --square-mm 25 --process-width 1280
.\build\atlas.exe --estimate-mount data\videos\iphone_calib\straight.mov --camera-model config\iphone_intrinsics.yml --out config\iphone_camera.yml --process-width 1280
.\build\atlas.exe --batch-log data\videos\iphone --camera-model config\iphone_camera.yml --process-width 1280 --no-objects
```

Unit tests:

```powershell
.\build\atlas_tests.exe
```

The headless summary reports the share of frames in each tracker status, resets, lane changes, gate rejections, offset/curvature jitter, how often speed was measured, and processing time percentiles. Those numbers say whether perception was *stable*, not just whether it ran.

## Recording your own data (iPhone)

[docs/IPHONE_RECORDING.md](docs/IPHONE_RECORDING.md) is a step-by-step guide: iPhone camera settings (landscape, 1x lens, 1080p/4K 30 fps, stabilization and Cinematic mode off, AE/AF lock), mounting, chessboard calibration with `--calibrate-video`, mount estimation, moving videos to the Mac, auto-labelling drives with `--batch-log`, reviewing sessions, storage budget, and using the iPhone as a live Continuity Camera webcam.

## Measuring lane accuracy

[docs/EVALUATION.md](docs/EVALUATION.md) describes `atlas_eval`: run the pipeline on the TuSimple benchmark (`--tusimple`), score an external prediction file (`--score PRED.json --labels GT.json`), or summarise human good/bad labels on a logged session (`--session`). The metric is unit-tested; it has not been run on the real TuSimple test set.

## Training a lane network

[training/README.md](training/README.md) is the Python training loop: pseudo-labels from `--log` sessions, optional TuSimple ground truth, `predict.py`, and ONNX export. The package is unit-tested; it has not been trained or evaluated on real recordings yet.

## Dashboard legend

| Element | Meaning |
| --- | --- |
| Solid lane line | Measured in this frame |
| Dashed lane line | Tracker memory (no measurement this frame) |
| Shaded band | +/-2 sigma lateral uncertainty; lines are drawn only where sigma < 0.4 m |
| Green / red dots | Lane evidence accepted / rejected by the tracker gate |
| Chevrons | Predicted path; they move at the measured speed |
| Boxes | Tracked objects with distance, relative velocity and TTC |
| BEV panel | Top-down metric road view |
| Paint panel | Lane-paint feature image with the search windows |
| Strip charts | Last 15 s of offset and curvature (with +/-2 sigma), lane width, confidence (coloured by tracker status) and speed |

Keys: `q`/Esc quit, `space` pause, `n` step one frame, `f` toggle optical flow, `r` raw camera only, `g`/`b` label the current frame good/bad (with `--log`).

## Logged dataset format

`--log` writes `data/sessions/<timestamp>_<source>/`:

- `session.json`: camera model (K, distortion, height, pitch, yaw), BEV grid and coordinate conventions.
- `frames.jsonl`: one JSON record per kept frame: why it was kept, labels, tracker state, road geometry, ego-motion, raw lane measurements with points and covariance, tracked objects (box, ground position, relative velocity, sigma), TuSimple-style `h_samples` + `lanes_image`, the predicted path and tracker events.
- `images/*.jpg`: the undistorted frame each record refers to.

A frame is kept on a 1 s keyframe timer, on a tracker status change or event, on low confidence, on a rejected measurement, or when you press `g`/`b`. Each session is capped at 500 MB and the whole `data/sessions` folder at 2000 MB (oldest sessions are deleted first; `--log-budget-mb` changes the total). `--review` browses a session with `a`/`d` and writes `g`/`b` labels to `labels.jsonl`.

## Status

- **Lane pipeline:** tested stable on 6 clips with `--no-objects` on Windows, ~13 ms/frame (~75 fps). Results below. `harder_challenge_video` is a known limitation (hairpin curves leave the BEV, washed-out sun).
- **Object detection:** crashed on Windows with OpenCV 5.0's new DNN engine. The fix (load YOLOX with the classic engine) is implemented but **untested**.
- **macOS and OpenCV 4:** the code has macOS / OpenCV 4 paths (AVFoundation capture, `cv_compat.hpp`), but they have not been built or run on a Mac yet.
- **iPhone / eval / training tooling:** unit-tested (`atlas_tests`, `python -m pytest training/tests`). Not yet run on real iPhone footage, the TuSimple dataset, or a training run.
- **Not done yet:** no long soak test, no live webcam test.

Lane results on the Udacity/P1 videos (lanes only):

| Video | LOCKED | Notes |
| --- | --- | --- |
| project_video | 99.8% | 0 resets, 0 false lane changes, offset jitter 1.5 cm/frame, ~13 ms/frame |
| challenge_video | 86.6% | 5 resets |
| p1_solidWhiteRight / p1_solidYellowLeft | 100% | ~1 cm jitter |
| p1_challenge | 98.8% | |
| harder_challenge_video | 28% | 15 resets: hairpins leave the BEV, sun glare, double yellow, motorcyclist |

## What this is, and what it is not

Atlas Vision borrows ideas that Tesla has described publicly and rebuilds them with classical, explainable tools:

| Public Tesla concept | Atlas Vision version |
| --- | --- |
| HydraNet: one backbone, many task heads | Separate modules (lanes, objects, ego-motion, horizon) writing one `PerceptionFrame` |
| Bird's-eye "vector space" | Flat-ground homography to a metric BEV, lanes as polynomials in metres |
| Temporal memory / video modules | Kalman lane tracker with odometry prediction, coasting and gating |
| Occupancy network | Not implemented; objects are 2D boxes projected to the ground |
| Fleet auto-labelling | `--log` records pseudo-labels from the tracker plus human good/bad labels |

It is **not** Tesla FSD or Autopilot. There is one camera, no neural lane network yet, no fleet data, a flat-ground assumption, and nothing is validated for a real vehicle. It reproduces the concepts, not the implementation.

## Repository layout

| Path | Role |
| --- | --- |
| `position.cpp` | `main`: runs Atlas Vision when given vision arguments, otherwise the waypoint sim |
| `vehicle.*`, `control.*` | Bicycle model, Pure Pursuit (shared by the sim and path prediction) |
| `planning.*`, `world.hpp`, `road.hpp`, `perception.*`, `camera.*`, `viz.*` | Waypoint sim |
| `vision/` | Atlas Vision library and app (see [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the file map) |
| `tests/` | `atlas_tests` unit tests |
| `docs/` | Guides ([iPhone recording](docs/IPHONE_RECORDING.md), [evaluation](docs/EVALUATION.md)) |
| `tools/` | `atlas_eval` (TuSimple, `--score`, session labels) |
| `training/` | Lane-network training ([README](training/README.md)) |
| `config/` | Camera models (`udacity_intrinsics.yml`, `udacity_camera.yml`, `p1_camera.yml`) |
| `data/` (ignored) | Videos, calibration images, model, sessions, snapshots |

Do not commit `build/`, `.exe` files, `data/`, `training/runs/`, `.venv/` or weight files (see `.gitignore`).
