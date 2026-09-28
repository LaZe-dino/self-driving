# How to run Atlas

See **[README.md](README.md)** for Windows, macOS, and `--webcam`.

Quick sim on this Windows machine:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build
.\build\atlas.exe
```

Webcam (after OpenCV is installed and CMake is reconfigured):

```powershell
.\build\atlas.exe --webcam
```


- `-G Ninja` uses `ninja.exe` from MSYS2. Do **not** use `-G "MinGW Makefiles"` unless `mingw32-make` is installed.
- The program is `.\build\atlas.exe`, not `.\atlas.exe`.
- If a previous configure failed, delete the `build` folder and run the three commands again.

The run **animates**: the terminal map redraws as the car moves. To print numbers instead, set `cfg.animate = false;` in `position.cpp`.

## What a good run looks like

A moving map with `#` lane edges, `W` waypoints, `*` trail, and `>^v<` for the car. After the motion:

```text
Reached waypoint 1 at t=...
Reached waypoint 2 at t=...
Reached waypoint 3 at t=...
Path complete
```

If the compiler prints `error:`, the new program was not built. An old `atlas.exe` in this folder or in `build\` may still run, so the numbers can look old.

## Files

| File | Job |
| --- | --- |
| `vehicle.hpp` / `vehicle.cpp` | Bicycle model |
| `control.hpp` / `control.cpp` | Pure Pursuit + speed law |
| `planning.hpp` / `planning.cpp` | Advance along the path |
| `world.hpp` | Fused snapshot: ego, path, trail, perception |
| `road.hpp` | Centerline + lane width |
| `perception.hpp` / `perception.cpp` | Fake lanes from the road (NN slot later) |
| `viz.hpp` / `viz.cpp` | Terminal map + animation |
| `position.cpp` | Sim loop (glue) |
| `CMakeLists.txt` | Build recipe |
| `build/atlas.exe` | Program produced by CMake. Do not edit. |

## If `g++` is not recognized

Close Cursor completely, open it again, and open a new terminal. Then run `g++ --version`. You should see `g++.exe (Rev4, Built by MSYS2 project) 16.2.0`.

## Build from the editor

Press `Ctrl+Shift+B`. That configures with Ninja if needed, then builds. Run `.\build\atlas.exe` yourself to see the output.
