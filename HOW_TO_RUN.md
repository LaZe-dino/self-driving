# How to run Atlas

Windows, MSYS2 UCRT64, PowerShell in `C:\Users\adity\Projects\Self-driving`. For macOS, the full CLI, the dashboard legend and the current status see [README.md](README.md).

## 1. Install (once)

In an MSYS2 UCRT64 shell:

```text
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja \
          mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-qt6-base
```

Make sure `C:\msys64\ucrt64\bin` is on the Windows `PATH`. Check with `g++ --version` in a new terminal (close and reopen Cursor if it is not found).

## 2. Build

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build
```

- The configure output must say `OpenCV 5.x — Atlas Vision enabled`. If it says `sim only`, see troubleshooting.
- Use `-G Ninja`, not `MinGW Makefiles`.
- The programs are `.\build\atlas.exe` and `.\build\atlas_tests.exe`. There should be no `atlas.exe` in the project root.
- `Ctrl+Shift+B` in the editor runs the same build.

## 3. Get data and the model

`data/` is gitignored. Put files here:

| File | Source |
| --- | --- |
| `data/videos/project_video.mp4`, `challenge_video.mp4`, `harder_challenge_video.mp4` | [CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines) |
| `data/camera_cal/calibration*.jpg` | same repo, `camera_cal/` |
| `data/videos/p1_solidWhiteRight.mp4`, `p1_solidYellowLeft.mp4`, `p1_challenge.mp4` | [CarND-LaneLines-P1](https://github.com/udacity/CarND-LaneLines-P1) `test_videos/` (renamed with `p1_`) |
| `data/models/yolox_tiny.onnx` | [YOLOX 0.1.1rc0 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx) (Apache-2.0) |

```powershell
Invoke-WebRequest https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx -OutFile data\models\yolox_tiny.onnx
```

## 4. Run

`data/videos` and `data/models` are gitignored; copy or download those files first (step 3). Paste one command at a time.

Waypoint sim (ends with `Path complete`):

```powershell
.\build\atlas.exe
```

Unit tests:

```powershell
.\build\atlas_tests.exe
```

Recorded video:

```powershell
.\build\atlas.exe --video data\videos\project_video.mp4 --camera-model config\udacity_camera.yml
```

Live camera, lanes only (first command to try; runs until you press `q`):

```powershell
.\build\atlas.exe --webcam --no-objects
```

The vision window stays open until you press `q`, close the window, or the video ends. Drop `--no-objects` once `data/models/yolox_tiny.onnx` is present. If the default model is missing, Atlas warns and continues with objects off instead of quitting.

## 5. Calibrate a camera

Two steps. The Udacity results are already in `config/`.

1. **Intrinsics** (focal length, centre, lens distortion) from 15-20 photos of a 9x6-inner-corner chessboard at different angles:

   ```powershell
   .\build\atlas.exe --calibrate data\camera_cal --out config\udacity_intrinsics.yml
   ```

   Udacity: 15/20 images used, RMS 0.85 px. Below ~1 px is good.

2. **Mount** (height, pitch, yaw) from a video of a straight road with both lane lines visible. The lane vanishing point gives pitch and yaw; the known lane width (default 3.7 m) gives camera height:

   ```powershell
   .\build\atlas.exe --estimate-mount data\videos\project_video.mp4 --camera-model config\udacity_intrinsics.yml --out config\udacity_camera.yml
   ```

   Udacity: h = 1.17 m, pitch -1.87 deg, yaw -2.56 deg. Without chessboard images use `--fov DEG` instead of `--camera-model` (that is how `config/p1_camera.yml` was made); distances and speed then depend on the assumed FOV.

A camera model only fits frames of the same resolution; the app refuses a mismatch.

## 6. Stress tests

Stability summary, no window (LOCKED %, resets, jitter, ms/frame):

```powershell
.\build\atlas.exe --video data\videos\project_video.mp4 --camera-model config\udacity_camera.yml --headless
.\build\atlas.exe --video data\videos\challenge_video.mp4 --camera-model config\udacity_camera.yml --headless
.\build\atlas.exe --video data\videos\harder_challenge_video.mp4 --camera-model config\udacity_camera.yml --headless
.\build\atlas.exe --video data\videos\p1_challenge.mp4 --camera-model config\p1_camera.yml --headless
```

Long run: loop a video, watch memory, drops and reconnects in the 10 s telemetry lines:

```powershell
.\build\atlas.exe --video data\videos\project_video.mp4 --camera-model config\udacity_camera.yml --loop --headless
```

Logging budget: small cap to check oldest-session deletion:

```powershell
.\build\atlas.exe --video data\videos\challenge_video.mp4 --camera-model config\udacity_camera.yml --headless --log --log-budget-mb 50
```

Expected on `project_video`: about 99.8% LOCKED, 0 resets, offset jitter ~1.5 cm, ~13 ms/frame for lanes only. The machine has crashed under load before: run one heavy job at a time and use `--max-frames N` for quick checks.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Exe exits immediately, exit code `-1073741515` / `0xC0000135` | A DLL is missing (usually Qt6 for HighGUI, or OpenCV) | `pacman -S mingw-w64-ucrt-x86_64-qt6-base`; put `C:\msys64\ucrt64\bin` on `PATH` |
| CMake prints `sim only`, or `OpenCV_DIR-NOTFOUND` in `build\CMakeCache.txt` | OpenCV not installed or not found | Install `mingw-w64-ucrt-x86_64-opencv`, delete `build\`, reconfigure |
| `--webcam` prints "This build has no OpenCV" | Same as above | Same as above |
| `--webcam` runs the waypoint map and stops at `Path complete` after a few seconds | A stale `atlas.exe` built before the vision code | Delete `atlas.exe` in the root and in `build\`, rebuild, run `.\build\atlas.exe` |
| Output looks old after a compile error | The previous exe still ran | Fix the `error:` lines and rebuild first |
| Camera does not open | No device, wrong index, or Windows privacy block | Try `--camera-index 1`; Settings > Privacy > Camera > allow desktop apps; close other apps using the camera. The source tries MSMF then DirectShow |
| `objects: ... cannot be loaded` | `data/models/yolox_tiny.onnx` missing | Download it (step 3). `--webcam` without `--no-objects` now warns and continues with objects off; `--object-model FILE` still errors if that file is missing |
| `file not found: data/videos/...` | `data/` is gitignored | Copy or download the clips in step 3 |
| `VIDEOIO(FFMPEG): ... can't be used to capture by name` | Homebrew OpenCV cannot open files via FFmpeg by name | Harmless if a later backend opens the file; Atlas tries CAP_ANY, FFmpeg, then AVFoundation |
| `camera model is WxH but frames are ...` | Calibration from a different resolution | Calibrate for this camera or drop `--camera-model` |
| Very slow (>50 ms/frame) | Debug build | Delete `build\` and reconfigure (default is RelWithDebInfo) |
