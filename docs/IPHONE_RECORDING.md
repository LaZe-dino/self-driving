# Recording your own data with an iPhone

This guide turns an iPhone into Atlas Vision's dash camera: record drives, calibrate the lens once, estimate how the phone is mounted, then auto-label every drive with `--batch-log`. Commands are for the Mac (`./build/atlas`); on Windows use `.\build\atlas.exe` and `\`. The commands are implemented and unit-tested; they have not been run on a real iPhone recording yet.

The one rule behind everything below: **the camera geometry must not change** between calibration and the drives. Same phone, same lens, same video format and resolution, same stabilization settings, same mount position. If any of those change, calibrate again.

## 1. iPhone settings

Set these once in **Settings > Camera** and in the Camera app, and leave them for calibration and every drive.

| Setting | Value | Why |
| --- | --- | --- |
| Orientation | **Landscape**, same way round every time | Atlas rejects portrait video. Rotating the phone changes which way is "up" in the calibration. |
| Lens | **1x main (wide) lens** | 0.5x ultra-wide has strong distortion; zooming or switching lenses changes the focal length. |
| Record Video | **1080p HD at 30 fps** (or **4K at 30 fps**) | Both are 16:9, so one calibration works for either after resizing. 60 fps doubles storage for little gain. |
| Lock Camera (Record Video) | **On** | Stops the phone switching lenses mid-recording. |
| Enhanced Stabilization (Record Video) / **Action Mode** | **Off** | Electronic stabilization crops and warps each frame differently, so the intrinsics change frame to frame. |
| **Cinematic** mode | **Off** (use plain Video) | Synthetic depth-of-field and refocusing change the image. |
| HDR Video (Record Video) | **Off** | 10-bit HDR (HLG/Dolby Vision) decodes with odd colours in OpenCV. |
| Auto FPS (Record Video) | **Off** | Keeps a steady 30 fps. |
| Macro Control | **On** (so auto-macro can be disabled) and macro **off** | Auto-macro switches to the ultra-wide lens. |
| Formats | High Efficiency (HEVC) is fine. Switch to **Most Compatible (H.264)** only if a `.mov` will not open. | Homebrew OpenCV decodes HEVC through FFmpeg; H.264 is the fallback. |
| AE/AF lock | **Long-press** the screen until "AE/AF LOCK" appears | Focus changes shift the focal length slightly ("focus breathing"). Lock with the phone already on its mount, pointed at the road. |

About AE/AF lock: the stock Camera app locks exposure together with focus. That is right for calibration. For drives with big lighting changes (tunnels, low sun) a locked exposure can over- or under-expose; drag the sun slider after locking to bias it, or accept auto exposure and lock only when the light is steady. Apps like Blackmagic Camera can lock focus alone.

## 2. Mounting

- Use a **rigid windshield or dashboard mount**. Phone clamps on air vents vibrate and sag.
- Place it **centred** left-right behind the windshield, **level** (horizon horizontal on screen), looking straight ahead, with the bonnet taking at most the bottom ~10% of the frame. Keep the wipers' swept area in front of the lens.
- **Never move the mount** between calibration, mount estimation and the drives. If you bump it, redo mount estimation (step 4). If you change phone, lens or settings, redo calibration too.
- Start recording before you drive and **do not handle the phone while driving**. Follow local laws on windshield mounts and phone use; stop safely if something needs fixing.
- A phone in direct sun overheats and stops recording; a mount out of the sun and a charger help on long drives.

## 3. Lens calibration (once per phone + settings)

1. Print a chessboard with **10 x 7 squares = 9 x 6 inner corners** (e.g. the OpenCV `pattern.png`, or any generator), at 100% scale. Tape it to something **flat and stiff** (clipboard, foam board). A wavy board ruins calibration.
2. **Measure one square** with a ruler in millimetres (e.g. 25 mm). The size does not change the lens intrinsics, but it is recorded for later use.
3. With the **same settings as the drives** (1x lens, 1080p30 or 4K30, stabilization off, landscape), film the board for **~30-60 s**:
   - move the board (or the phone) slowly, so frames are sharp;
   - cover the **whole frame, especially the corners and edges**, where distortion is strongest;
   - vary the **distance** (board filling 1/4 to 1/2 of the frame) and **tilt** it up to ~45 deg in different directions;
   - keep the **whole board** in view; frames with part of it cut off are skipped.
4. Copy the video to the Mac (step 6) and run:

```bash
./build/atlas --calibrate-video data/videos/iphone/calib.mov --out config/iphone_intrinsics.yml \
    --board 9x6 --square-mm 25 --process-width 1280
```

Atlas samples a frame every 0.5 s, skips blurry frames (variance of the Laplacian inside the board) and near-duplicate board poses, keeps up to 40 well-spread views, calibrates, drops views with more than 2x the median reprojection error and calibrates again. It prints the per-view errors and the final RMS error.

- **Good:** RMS below ~0.5 px at 1280 wide, 20-40 views used.
- **Redo it** if RMS is above 1 px, fewer than 15 views are used, or `fx` and `fy` differ by more than ~1%.

`--process-width` must match the drives (see below). Without it the calibration is at the native video size. A model calibrated at one 16:9 size is rescaled automatically for another 16:9 size (Atlas prints a notice), but calibrating at the processing size is the cleanest.

Photos work too: put `.jpg`/`.jpeg`/`.png` board photos in a folder and use `--calibrate DIR` with the same options. iPhone photos are 4:3 and HEIC by default, so they do **not** match 16:9 video; the video route is the right one for driving data.

## 4. Mount estimation (once per mounting)

Record (or cut) a clip that **starts on a straight, well-marked road** with both lane lines visible for at least 10 s, driving centred in the lane. Mount estimation reads the first 300 frames (10 s at 30 fps; change with `--max-frames`).

```bash
./build/atlas --estimate-mount data/videos/iphone/straight.mov --camera-model config/iphone_intrinsics.yml \
    --out config/iphone_camera.yml --process-width 1280 --lane-width 3.7
```

It finds the lane vanishing point to get pitch and yaw, then the camera height from the known lane width (3.7 m is a typical US highway; many urban lanes are 3.0-3.3 m). Check the result: height should be roughly 1.1-1.5 m for a car windshield, pitch a few degrees. A median vanishing-point spread of more than ~10 px means the road was not straight enough.

## 5. Recording drives

- Check the settings from step 1, lock AE/AF, start recording, then drive.
- One file per drive or per 10-20 minutes is easier to handle than one huge file.
- Daylight, dry roads and clear lane markings give the best automatic labels. Night and rain are useful later, once the basics work.

## 6. Moving videos to the Mac

- **AirDrop** from Photos: in the share sheet tap **Options** and turn on **All Photos Data** so the original `.mov` is sent, not a re-encoded copy. Files land in `~/Downloads`.
- For long drives, a USB cable with **Image Capture** is faster and more reliable.
- Put them in `data/videos/iphone/` (gitignored):

```bash
mkdir -p data/videos/iphone
mv ~/Downloads/IMG_*.MOV data/videos/iphone/
```

Keep the calibration clip separate (e.g. `data/videos/iphone_calib/`) so `--batch-log` does not process it as a drive.

`data/videos` is gitignored and is not cloned with the repo. Copy the `.mov` files here yourself. If OpenCV prints `VIDEOIO(FFMPEG): backend is generally available but can't be used to capture by name`, Atlas retries CAP_ANY, then FFmpeg, then AVFoundation.

## 7. Batch auto-labelling

```bash
./build/atlas --batch-log data/videos/iphone --camera-model config/iphone_camera.yml \
    --process-width 1280 --log-dir data/sessions --no-objects
```

Every `.mov`/`.mp4`/`.m4v` in the folder is processed in order, headless, with logging on. Each file gets a session folder `data/sessions/<timestamp>_<video name>/` and its own run summary; a table at the end lists frames, LOCKED share, confidence, resets and the session folder for every video. Drop `--no-objects` to also log objects (needs `data/models/yolox_tiny.onnx`). Add `--max-frames 300` for a quick trial.

Frames are resized right after decoding, so `--process-width 1280` makes 1080p and 4K footage cost the same. The camera model must be at the processing size or the same aspect ratio (it is rescaled automatically). Atlas prints the native and processed frame size and the rotation metadata; FFmpeg applies the rotation, and a portrait result stops with an error.

## 8. Reviewing

```bash
./build/atlas --review data/sessions/<timestamp>_IMG_1234
```

`a`/`d` step through the logged frames, `g`/`b` label them good or bad (written to `labels.jsonl`). Spot-check each drive before using it for training: bad calibration shows up as lane lines that do not sit on the paint, or a lane width that drifts far from reality.

## Storage budget

| What | Approximate size |
| --- | --- |
| 1080p30 HEVC recording | ~60 MB per minute (~3.5 GB per hour) |
| 4K30 HEVC recording | ~190 MB per minute (~11 GB per hour) |
| Same in H.264 (Most Compatible) | about 2x the HEVC size |
| Logged session at 1280x720 | ~10-20 MB per minute of driving (JPEG keyframes + JSON) |

Session logs are capped at 500 MB each and **2000 MB in total** under `data/sessions`. When a new session would exceed the total, the **oldest sessions are deleted**, including earlier ones from the same batch. Raise the cap for big batches (`--log-budget-mb 20000`) or move finished sessions elsewhere. Delete source videos only after reviewing their sessions.

## Alternative: iPhone as a live Mac webcam (Continuity Camera)

On macOS 13+ with iOS 16+, a nearby iPhone (same Apple ID, Wi-Fi and Bluetooth on) shows up as a camera. Start with `--no-objects` so a missing `data/models/yolox_tiny.onnx` (also gitignored) does not quit the window:

```bash
./build/atlas --webcam --no-objects --camera-index 1 --process-width 1280
```

Drop `--no-objects` after the ONNX file is in `data/models/`. Try indices 0, 1, 2 until the iPhone appears (the MacBook camera is usually 0). Caveats:

- The Mac chooses the capture format and crop. Atlas asks for 1280x720, but the effective field of view may differ from the Camera app's video, so a calibration from a recorded video may not match. Calibrate on the same path if you rely on distances.
- Turn **Center Stage** and **Desk View** off (Control Center > Video Effects): they crop and pan the image.
- Stabilization and lens choice are controlled by the system, not you; geometry can change between sessions.
- It needs the Mac in the car and a wireless link that can drop. For training data, recording on the phone and batch-processing afterwards is more reliable.
