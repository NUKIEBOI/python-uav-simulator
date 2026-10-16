"""Entry point for the Python UAV Simulator.

Usage:
    python main.py            Launch the 3D simulator (needs a display).
    python main.py --test     Headless scripted mission: takeoff, autopilot
                              waypoint circuit with a scan over the target
                              field, then a detection report. No window.
"""

import argparse
import math
import sys


def run_test() -> int:
    """Scripted ISR mission with no graphics. Returns 0 on success."""
    # NOTE: simulator.py (Panda3D) is deliberately NOT imported here.
    from controls import ControlState
    from flight_model import FlightModel, quat_to_matrix
    from autopilot import Autopilot
    from battery import Battery
    from sensors import GimbalCamera, build_target_field, sweep_angles
    import numpy as np

    fm = FlightModel()
    ctrl = ControlState()
    ap = Autopilot(cruise_speed=22.0)
    ap.set_waypoints([(600.0, 100.0, 80.0), (850.0, 500.0, 80.0),
                      (400.0, 500.0, 80.0), (350.0, 100.0, 80.0)])
    ap.loop = True  # fly the circuit repeatedly
    bat = Battery()
    cam = GimbalCamera()
    objects = build_target_field()

    dt = 1.0 / 60.0
    t = 0.0
    phase = "TAKEOFF"
    rotated = False
    liftoff_t = None
    seen = {}
    circuits = 0
    last_wp = 0
    ok = True

    print(f"{'t(s)':>7} {'phase':>9} {'kts':>5} {'alt(ft)':>8} "
          f"{'det':>4} {'note':<24}")
    next_print = 0.0
    while t < 420.0:
        V = fm.airspeed()

        if phase == "TAKEOFF":
            ctrl.throttle = 1.0
            if V >= 12.0 and not rotated:
                rotated = True
            ctrl.elevator = 0.7 if (rotated and fm.on_ground) else 0.0
            if not fm.on_ground:
                if liftoff_t is None:
                    liftoff_t = t
                # Hold ~10 deg nose-up until clear of the ground.
                pitch = fm.attitude_euler()[0]
                ctrl.elevator = max(-1.0, min(1.0, 0.12 + (0.17 - pitch) * 2.0))
                if fm.pos[2] >= 55.0:
                    phase = "MISSION"
                    ap.engage()
        elif phase == "MISSION":
            prev_wp = ap.wp_index
            ap.update(fm, ctrl, dt)
            if ap.wp_index != prev_wp:
                # Wrapped past the last waypoint -> one circuit complete.
                if prev_wp == len(ap.waypoints) - 1 and ap.wp_index == 0:
                    circuits += 1
                last_wp = ap.wp_index
            if circuits >= 2:
                phase = "DONE"

        bat.step(dt, ctrl.throttle)
        fm.step(dt, ctrl)

        # Scan the target field with the auto-sweep while on mission.
        pan, tilt = sweep_angles(t)
        for d in cam.detect(fm, pan, tilt, objects):
            seen[d["id"]] = d

        if not all(map(math.isfinite, list(fm.pos) + list(fm.vel_b)
                       + list(fm.quat) + list(fm.omega))):
            print("FAIL: non-finite state encountered")
            ok = False
            break
        if t >= next_print:
            next_print += 5.0
            note = ap.status if ap.engaged else ""
            print(f"{t:7.0f} {phase:>9} {V * 1.94384:5.0f} "
                  f"{fm.pos[2] * 3.28084:8.0f} {len(seen):4d} {note:<24}")

        if phase == "DONE":
            break
        t += dt

    print(f"\nLiftoff at t={liftoff_t:.1f}s" if liftoff_t else "\nNo liftoff!")
    print(f"Mission time: {t:.0f}s, circuits flown: {circuits}")
    print(f"\nDetection report: {len(seen)}/{len(objects)} targets found")
    for d in sorted(seen.values(), key=lambda d: d["id"]):
        print(f"  {d['id']:10s} {d['type']:11s} pos=({d['pos'][0]:6.0f},"
              f"{d['pos'][1]:6.0f}) range={d['range_m']:5.0f}m "
              f"conf={d['confidence']:.2f}")
    missing = [o["id"] for o in objects if o["id"] not in seen]
    if missing:
        print(f"  missed: {', '.join(missing)}")

    if liftoff_t is None:
        print("FAIL: never got airborne"); ok = False
    if len(seen) < 8:
        print(f"FAIL: only {len(seen)}/12 targets detected"); ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Python UAV Simulator")
    parser.add_argument("--test", action="store_true",
                        help="headless scripted ISR mission, no window")
    args = parser.parse_args()
    if args.test:
        return run_test()
    from simulator import main as sim_main
    sim_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
