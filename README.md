# Atlas Autonomy

C++ learning sim: bicycle vehicle, Pure Pursuit, and optional **Atlas Vision** (webcam → lane overlay → steering). `--webcam` does **not** drive a real car. It steers the **simulated** bicycle from lines in the camera image.

```text
./atlas              waypoints → Pure Pursuit → Vehicle
./atlas --webcam     camera → Canny/Hough lanes → steering/speed → Vehicle
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

## Atlas Vision (`--webcam`)

The window stays open until you press **q**. It does **not** quit when a simulated waypoint is reached.

```bash
brew install cmake opencv
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(brew --prefix opencv)"
cmake --build build
./build/atlas --webcam
```

- Allow **Camera** for Terminal (or Cursor) in System Settings.
- Point the camera at a hallway, road, or two parallel edges (tape on the floor works).
- Teal corridor + orange path = Tesla-style overlay. Steering and speed come from those lanes.
- CMake must print `OpenCV … — ./atlas --webcam enabled`.
- This is classical CV (Canny + Hough), not Tesla’s neural occupancy network.

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

Tesla Autopilot / FSD (neural occupancy, real car). `--webcam` is classical lane lines on a laptop camera steering a **simulated** bicycle. Point it at a hallway or tape lines for a fair test.
