# Atlas Autonomy

C++ learning sim: bicycle vehicle, Pure Pursuit, a tiny world model, and a synthetic camera. Optional **webcam preview** via OpenCV. The webcam is a sensor window — it does **not** steer a real car, and it does not replace the simulated path yet.

```text
Road → synthetic camera → PerceptionOutput → WorldState → plan → control → Vehicle
Webcam (optional) → OpenCV window (preview only)
```

## Run the sim (no camera)

### Windows (MSYS2 UCRT64)

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build
.\build\atlas.exe
```

Use Ninja, not MinGW Makefiles, unless `mingw32-make` is installed.

### macOS

```bash
brew install cmake
cmake -S . -B build
cmake --build build
./build/atlas
```

A good run prints three `Reached waypoint` lines and `Path complete` (~8 s).

## Webcam preview (Mac is the easy test)

Install OpenCV, **reconfigure** CMake (delete `build` if you configured before OpenCV existed), rebuild, then:

```bash
brew install cmake opencv
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(brew --prefix opencv)"
cmake --build build
./build/atlas --webcam
```

- Allow **Camera** for Terminal (or iTerm/Cursor) in System Settings → Privacy & Security.
- A window titled `Atlas webcam` should show the laptop camera.
- Press `q` in that window to stop.
- CMake should print `OpenCV … — ./atlas --webcam enabled`. If it prints `OpenCV not found`, the prefix path is wrong.

Windows with MSYS2:

```text
pacman -S mingw-w64-ucrt-x86_64-opencv
```

Then delete `build`, reconfigure, and run `.\build\atlas.exe --webcam`.

## GitHub

Do not commit `build/` or `.exe` files (see `.gitignore`).

```bash
git init
git add .
git status
git commit -m "Initial Atlas Autonomy sim with optional OpenCV webcam preview."
git branch -M main
gh repo create atlas-autonomy --public --source=. --remote=origin --push
```

Or create an empty repo on github.com and `git remote add origin …` then `git push -u origin main`.

## Layout

| File | Role |
| --- | --- |
| `vehicle.*` | Bicycle model |
| `control.*` | Pure Pursuit + speed |
| `planning.*` | Waypoint progress |
| `world.hpp` | Ego + path + perception snapshot |
| `camera.*` | Synthetic top-down crop |
| `webcam.*` | Optional OpenCV capture |
| `viz.*` | Terminal map |
| `position.cpp` | Main loop |

## What this is not

Tesla Autopilot on a MacBook camera. No actuators, no trained net, no road from the room image. Next perception work is feeding webcam pixels into the same `PerceptionOutput` type without breaking the sim.
