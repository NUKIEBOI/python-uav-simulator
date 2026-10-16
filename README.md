# Python UAV Simulator

A realistic fixed-wing surveillance UAV simulator in pure Python — think a
small Predator/Reaper-type drone. It pairs a 6-DOF flight dynamics model
with an **autonomous waypoint autopilot** and a **gimballed scan-and-detect
camera payload**, all visualized in 3D. Built as a portfolio project for
UAV/robotics roles: every subsystem is a separate, readable module.

## Features

- **Real flight physics** — 6-DOF aerodynamics (lift/drag/sideforce,
  control-surface moments with damping, stall behavior), electric pusher-prop
  thrust model, quaternion attitude, 240 Hz substepped integration, and a
  takeoff/landing ground model. Deterministic and NaN-guarded.
- **Autopilot** — waypoint follower with cascaded heading→bank→aileron,
  altitude→pitch→elevator, and airspeed→throttle loops, plus coordinated-turn
  rudder. Fly a full ISR circuit hands-off, or take the sticks anytime.
- **Scan-and-detect payload** — a 2-axis gimballed camera (pan + tilt, or
  auto-scan sweep) that detects ground targets and reports id, type,
  position, range, bearing, and confidence to a live detection log.
- **Usable onboard cameras** — nose FPV and payload feeds rendered like real
  UAV video: letterboxed, center reticle, corner telemetry, blinking REC.
  Detection markers (`[veh-01]`) are drawn on the payload feed itself.
- **3D world** — procedural terrain with a runway, 12 detectable targets
  (landing pads, vehicles, persons), Predator-style airframe built from
  primitives, chase/FPV/payload/tower cameras, HUD, and a battery model
  with low/critical warnings and emergency auto-land.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py            # launch the 3D simulator (needs a display)
python main.py --test     # headless scripted ISR mission, prints a
                          # detection report (no window needed)
python tests/test_uav_model.py   # headless unit tests
```

## Controls

| Key | Action |
|---|---|
| `Up` / `W` | Nose down (push) |
| `Down` / `S` | Nose up (pull) |
| `Left` / `A` | Roll left |
| `Right` / `D` | Roll right |
| `Q` / `E` | Rudder left / right |
| `Shift` | Throttle up (hold) |
| `Ctrl` | Throttle down (hold) |
| `T` / `G` | Elevator trim nose-up / nose-down |
| `B` | Toggle wheel brakes |
| `Z` / `X` | Gimbal tilt down / up |
| `N` / `M` | Gimbal pan left / right |
| `V` | Auto-scan sweep on/off |
| `L` | Detection log panel on/off |
| `U` | Autopilot engage / disengage |
| `C` | Camera: chase → nose FPV → payload → tower |
| `P` | Pause |
| `R` | Reset flight |
| `H` | Toggle help overlay |
| `Esc` | Quit |

**Typical sortie:** full throttle (hold `Shift`), rotate at ~12 m/s (pull `S`
gently), climb out, press `U` for the autopilot circuit, press `V` to start
the scan sweep, then press `C` twice to watch the payload feed pick up
targets — markers appear on the feed itself.

## Architecture

| File | What it does |
|---|---|
| `uav_params.py` | 20-kg-class airframe parameters (mass, inertia, aero coefficients, prop) |
| `atmosphere.py` | ISA atmosphere: density/pressure/temperature vs altitude |
| `flight_model.py` | Headless 6-DOF dynamics core (NumPy only) + ground model |
| `autopilot.py` | Headless waypoint follower: heading/altitude/airspeed holds |
| `battery.py` | 6S LiPo model: draw, voltage sag, low/critical warnings |
| `sensors.py` | Gimballed camera geometry + scan-and-detect + target field |
| `controls.py` | Pilot + payload control state (sticks, gimbal, toggles) |
| `simulator.py` | Panda3D app: world, UAV model, cameras, feed dressing, HUD |
| `main.py` | Entry point; `--test` runs the headless ISR mission |

The physics, autopilot, battery, and sensors are fully headless — they import
only NumPy (or nothing) and are covered by unit tests. Only `simulator.py`
touches Panda3D.

## How the flight model works

Each 1/240 s substep: compute airspeed, angle of attack, and sideslip from
the body velocity; look up lift (with a stall break past the critical
angle), drag (parasite + induced + stall separation), and sideforce;
add pusher-prop thrust and weight; compute pitch/roll/yaw moments from
static stability, damping derivatives, and control deflections; then
integrate angular and translational motion semi-implicitly with a quaternion
attitude. On the ground, tire friction/grip, nosewheel steering, and a pitch
spring model the takeoff roll until lift exceeds weight.

## How the scan-and-detect payload works

The camera is a 2-axis gimbal (pan about body z, tilt about body y) with a
60°×45° field of view and 300 m max range. Every step, each known ground
object is transformed into the camera frame; objects inside the frustum and
range produce a detection with id, type, world position, range, bearing from
the nose, and a confidence that falls with range. **Honesty note:** this is a
*geometric* stand-in for computer vision — it answers "is the target in the
frustum?" Real CV would add false positives, misses, and lighting effects;
the gimbal math, frustum geometry, and detection plumbing are all real.

## Roadmap ideas

- **Waypoint mission planner** — click-to-place waypoints on a map view,
  save/load missions as JSON.
- **MAVLink interface** — speak MAVLink so the sim can be driven by real
  ground-station software (Mission Planner / QGroundControl).
- **Vision upgrade** — replace the geometric detector with an actual CV
  model running on the rendered payload feed.
