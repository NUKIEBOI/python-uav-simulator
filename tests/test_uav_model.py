"""Headless tests for the UAV flight model, autopilot, battery, and camera.

Run:  python tests/test_uav_model.py   (no Panda3D, no window needed)
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from uav_params import SurveillanceUAV
from flight_model import FlightModel, quat_to_matrix
from autopilot import Autopilot
from battery import Battery
from controls import ControlState
from sensors import GimbalCamera, build_target_field


def _finite(fm):
    return all(map(math.isfinite, list(fm.pos) + list(fm.vel_b)
                   + list(fm.quat) + list(fm.omega)))


def test_trimmed_level_flight():
    """Trimmed elevator + matched throttle holds altitude in level flight."""
    fm = FlightModel()
    ctrl = ControlState()
    fm.reset(pos=(0.0, 0.0, 60.0), vel_b=(22.0, 0.0, 0.0), on_ground=False)
    ctrl.elevator = fm.trim_de_for_level(22.0, 60.0)
    ctrl.throttle = 0.31  # ~29 N thrust ~= drag at 22 m/s
    for _ in range(int(15.0 / (1 / 60))):
        fm.step(1 / 60, ctrl)
        assert _finite(fm)
    assert abs(fm.pos[2] - 60.0) < 5.0, f"alt drifted to {fm.pos[2]:.1f} m"
    pitch = fm.attitude_euler()[0]
    assert abs(pitch) < 0.3, f"pitch diverged: {pitch:.2f} rad"
    print("PASS test_trimmed_level_flight "
          f"(alt {fm.pos[2]:.1f} m, pitch {math.degrees(pitch):.1f} deg)")


def test_takeoff():
    """Full throttle, rotate at 12 m/s, establish a climb."""
    fm = FlightModel()
    ctrl = ControlState()
    ctrl.throttle = 1.0
    rotated = False
    liftoff_t = None
    t = 0.0
    while t < 60.0:
        if fm.airspeed() >= 12.0 and not rotated:
            rotated = True
        ctrl.elevator = 0.7 if (rotated and fm.on_ground) else 0.0
        if not fm.on_ground:
            if liftoff_t is None:
                liftoff_t = t
            pitch = fm.attitude_euler()[0]
            ctrl.elevator = max(-1.0, min(1.0, 0.12 + (0.17 - pitch) * 2.0))
        fm.step(1 / 60, ctrl)
        assert _finite(fm)
        t += 1 / 60
    assert liftoff_t is not None, "never lifted off"
    assert not fm.on_ground and fm.pos[2] > 10.0, "no climb established"
    print(f"PASS test_takeoff (liftoff t={liftoff_t:.1f}s, "
          f"alt {fm.pos[2]:.1f} m)")


def test_autopilot_waypoint():
    """Autopilot flies to a waypoint and holds altitude."""
    fm = FlightModel()
    ctrl = ControlState()
    ap = Autopilot(cruise_speed=22.0)
    ap.set_waypoints([(400.0, 0.0, 80.0)])
    fm.reset(pos=(0.0, 0.0, 80.0), vel_b=(22.0, 0.0, 0.0), on_ground=False)
    ctrl.throttle = 0.55
    ap.engage()
    t = 0.0
    while t < 90.0:
        ap.update(fm, ctrl, 1 / 60)
        fm.step(1 / 60, ctrl)
        assert _finite(fm)
        t += 1 / 60
    dist = math.hypot(fm.pos[0] - 400.0, fm.pos[1])
    # Capturing the final waypoint latches a loiter orbit (~90 m radius at
    # the 28-deg bank limit), so "tracking" means a stable nearby orbit.
    assert ap._loiter, "autopilot never captured the waypoint"
    assert dist < 150.0, f"ended {dist:.0f} m from waypoint"
    assert abs(fm.pos[2] - 80.0) < 12.0, f"alt {fm.pos[2]:.1f} m"
    print(f"PASS test_autopilot_waypoint (orbit radius {dist:.0f} m, "
          f"alt {fm.pos[2]:.1f} m)")


def test_camera_detection():
    """Camera pointed straight down at a known pad detects it correctly."""
    fm = FlightModel()
    objects = build_target_field()
    pad = next(o for o in objects if o["id"] == "pad-red")
    fm.reset(pos=(pad["pos"][0], pad["pos"][1], 80.0), on_ground=False)
    cam = GimbalCamera()
    dets = cam.detect(fm, pan=0.0, tilt=math.pi / 2, objects=objects)
    ids = [d["id"] for d in dets]
    assert "pad-red" in ids, f"pad-red not detected (saw {ids})"
    d = next(d for d in dets if d["id"] == "pad-red")
    err = math.hypot(d["pos"][0] - pad["pos"][0], d["pos"][1] - pad["pos"][1])
    assert err < 2.0, f"reported position off by {err:.1f} m"
    assert d["type"] == "landing_pad"
    print(f"PASS test_camera_detection ({len(dets)} detections, "
          f"pad-red conf {d['confidence']:.2f})")


def test_aggressive_maneuvering_stable():
    """60 s of hard maneuvering never produces NaN/inf."""
    fm = FlightModel()
    ctrl = ControlState()
    fm.reset(pos=(0.0, 0.0, 150.0), vel_b=(25.0, 0.0, 0.0), on_ground=False)
    t = 0.0
    while t < 60.0:
        ctrl.elevator = 0.9 * math.sin(2 * math.pi * t / 3.0)
        ctrl.aileron = 0.9 * math.sin(2 * math.pi * t / 2.1 + 1.0)
        ctrl.rudder = 0.6 * math.sin(2 * math.pi * t / 4.7 + 2.0)
        ctrl.throttle = 0.9 if (t % 8.0) < 4.0 else 0.15
        fm.step(1 / 60, ctrl)
        assert _finite(fm), f"non-finite state at t={t:.1f}s"
        t += 1 / 60
    print("PASS test_aggressive_maneuvering_stable")


def test_battery_drain():
    """Sustained cruise drains the pack and raises the low-battery flag."""
    bat = Battery()
    for _ in range(25 * 60):
        bat.step(1.0, throttle=0.7)
    assert bat.soc < 0.75, f"soc {bat.soc:.2f} did not drain"
    assert bat.low_battery, "low_battery flag never raised"
    assert not bat.critical_battery, "should not be critical yet"
    print(f"PASS test_battery_drain (soc {bat.soc:.2f}, "
          f"{bat.voltage:.1f} V, low={bat.low_battery})")


if __name__ == "__main__":
    test_trimmed_level_flight()
    test_takeoff()
    test_autopilot_waypoint()
    test_camera_detection()
    test_aggressive_maneuvering_stable()
    test_battery_drain()
    print("\nALL TESTS PASSED")
