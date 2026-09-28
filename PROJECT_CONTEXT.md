# Atlas Autonomy — handoff for new chat

## My goal

Personal C++ learning project: simplified autonomous driving in simulation, **Tesla-like architecture** (perception → world model → planning → control → vehicle). Tesla Autopilot/FSD is the shape to copy, not the fleet, data, or full neural stack. Classic layered stack first; neural nets later fill the **same** `PerceptionOutput` slot. Learning the boxes matters more than speed.

## Teaching preferences

- Larger slices when asked; line-by-line when stuck.
- On errors: Problem / Why / Fix / Lesson.
- Windows, MSYS2 UCRT64 g++ 16.2.0 at `C:\msys64\ucrt64\bin\g++.exe`.
- Project: `C:\Users\adity\Projects\Self-driving`
- Build: CMake + Ninja — `cmake --build build` then `.\build\atlas.exe`. See `HOW_TO_RUN.md`.
- If compile fails, old `atlas.exe` may still run—don’t trust that output.

## Atlas Vision (current focus)

Camera perception in `vision/`, run with `atlas --video FILE` or `atlas --webcam` (needs OpenCV). Pipeline, CLI, data sources and status are in `README.md`. File map:

- `app.*` — CLI parsing and run modes; `pipeline.*` — runs the stages into one `PerceptionFrame`
- `frame_source.*` — capture thread, bounded queue, reconnect; `camera_model.*`, `calibration.*` — intrinsics, mount, undistort
- `object_detector.*` (YOLOX-tiny via OpenCV DNN), `object_tracker.*` — boxes, ground distance, TTC
- `ground_view.*` — bird's-eye homography; `ego_motion.*` — optical-flow speed and yaw rate
- `lane_detector.*`, `lane_tracker.*`, `lane_types.hpp` — BEV lane fits and Kalman road model
- `horizon_estimator.*` — online pitch; `road_geometry.*` — offset, heading, curvature
- `dashboard.*`, `data_logger.*`, `review.*` — display, training-data log, label browser
- `process_stats.*` — memory/CPU telemetry; `cv_compat.hpp` — OpenCV 4/5 header differences
- `tests/` — `atlas_tests` unit tests (13)

## Waypoint sim (original stack, `atlas` with no arguments)

```text
Road (centerline) → fake_perceive → PerceptionOutput
                              ↓
                         WorldState  (ego + path + lanes + trail)
                              ↓
                    Planning (waypoint index)
                              ↓
            Control (Pure Pursuit + speed P-law)
                              ↓
                    Vehicle (kinematic bicycle)
                              ↓
                    Viz (animated terminal map)
```

Pose is still **ground-truth sim** (no cameras, no localization). `fake_perceive` is the stand-in for cameras + neural nets.

## Repo files

- `vehicle.hpp` / `vehicle.cpp` — bicycle model, `get_wheelbase`
- `control.hpp` / `control.cpp` — Pure Pursuit, longitudinal accel, Config
- `planning.hpp` / `planning.cpp` — distance-to-waypoint, advance index
- `world.hpp` — WorldState
- `road.hpp` — centerline + half_width
- `perception.hpp` / `perception.cpp` — PerceptionOutput, `fake_perceive`
- `viz.hpp` / `viz.cpp` — draw_world, clear_screen, heading glyph
- `position.cpp` — main loop (glue only)
- `CMakeLists.txt`, `HOW_TO_RUN.md`, `.vscode/*`

## Vehicle

- State: x, y, velocity, acceleration, heading, steering, wheelbase=2.5
- `step`: v += a*dt, v≥0; heading += (v/L)*tan(δ)*dt; x,y via cos/sin(heading)

## Control

- Pure Pursuit: δ = atan(2 L sin(α) / Ld), clamp ±0.4 rad
- Lookahead: first path point ≥ 8 m ahead
- Speed: 12 m/s cruise, 4 m/s inside 15 m, accel gain 0.8
- Path: (20,10) → (40,10) → (40,30), reach radius 5 m
- Successful run: three waypoints, Path complete (~8.4 s)

## World / perception / viz

- WorldState: ego, path, target_index, trail, time, PerceptionOutput
- Fake perception: offset centerline by ±half_width → two `#` lane polylines
- Animation on by default (`cfg.animate`, `cfg.frame_ms=50`). Glyph `>^v<` from heading
- Set `cfg.animate = false` in `position.cpp` for numeric logs

## Roadmap

- Done: vehicle, Pure Pursuit, CMake, module split, WorldState, road edges, fake perception, animation; Atlas Vision (lanes, tracker, ego-motion, pitch, objects, dashboard, logger, review)
- **Next:** verify object detection with the classic DNN engine, build/test on macOS, long soak test, live webcam test, better handling of hairpins and glare
- Later: neural lane head trained on logged data, PID/Stanley, ROS 2 optional

## Honest Tesla note

We will not reproduce FSD (data, training, validation, fleet). We **will** reproduce the pipeline: sensors/perception output a scene, world model holds it, planner + control act, vehicle integrates. Swapping `fake_perceive` for a network is the Autopilot-shaped step, not rewriting the stack.
