"""Simple UAV autopilot: waypoint follower with heading/altitude/airspeed holds.

HEADLESS: only `math` -- no NumPy, no graphics. It reads the flight model's
attitude/velocity and writes pilot-style commands into a ControlState, so the
same code flies the headless tests and the 3D simulator.

How it works (cascaded P/PD loops, the classic UAV recipe):
  * Heading -> bank angle -> ailerons. The outer loop turns a heading error
    into a desired bank (capped ~28 deg); the inner loop drives the ailerons
    to that bank with roll-rate damping.
  * Altitude -> pitch attitude -> elevator. Outer loop turns altitude error
    into a desired pitch (with vertical-speed damping); inner loop drives
    the elevator to that pitch with pitch-rate damping.
  * Airspeed -> throttle (proportional + feedforward base throttle).
  * Sideslip -> rudder (keeps the ball centered in turns).

Waypoints are (x, y, altitude_m) tuples. The active waypoint advances when
the UAV comes within the capture radius; after the last waypoint the UAV
keeps targeting it, which naturally produces a loose orbit (a "loiter").

Sign conventions (body x fwd, y right, z up):
  * roll is left-positive; aileron +1 rolls RIGHT (negative roll rate).
  * pitch is nose-up positive; elevator +1 pulls the nose UP.
  * heading/yaw positive turns the nose RIGHT (toward +y).
"""

import math

from flight_model import quat_to_matrix


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _wrap_pi(a):
    """Wrap an angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class Autopilot:
    def __init__(self, cruise_speed=22.0, capture_radius=25.0):
        self.cruise_speed = float(cruise_speed)
        self.capture_radius = float(capture_radius)
        self.waypoints = []          # list of (x, y, alt_m)
        self.wp_index = 0
        self.engaged = False
        self.emergency_land = False  # set by battery critical (see below)
        self._loiter = False         # latched orbit around the final waypoint
        # Gains -- tuned for the 20 kg airframe; retune if uav_params changes.
        self.kp_h = 1.2              # heading err -> bank
        self.bank_max = 0.50         # rad (~28 deg)
        self.kp_r = 2.5              # bank err -> aileron
        self.kd_r = 0.6              # roll-rate damping
        self.kp_a = 0.06             # alt err (m) -> pitch
        self.kd_a = 0.10             # vertical-speed damping
        self.pitch_max = 0.35        # rad
        self.kp_e = 2.0              # pitch err -> elevator
        self.kd_e = 0.5              # pitch-rate damping
        self.thr_base = 0.55         # feedforward throttle for cruise
        self.kp_v = 0.06             # airspeed err -> throttle
        self.kp_b = 0.8              # sideslip -> rudder
        self.kd_b = 0.2              # yaw-rate damping
        self.orbit_radius = 80.0     # loiter orbit radius (m)
        self.k_orbit = 0.6           # radial correction gain in loiter
        self.loop = False            # if True, wrap to WP0 after the last
        self.status = "OFF"

    # -- mission programming ---------------------------------------------
    def set_waypoints(self, waypoints):
        self.waypoints = [(float(x), float(y), float(z)) for x, y, z in waypoints]
        self.wp_index = 0
        self._loiter = False

    def engage(self):
        self.engaged = True
        self.emergency_land = False
        self._loiter = False
        self.status = "NAV"

    def disengage(self):
        self.engaged = False
        self.emergency_land = False
        self._loiter = False
        self.status = "OFF"

    def trigger_emergency_land(self, home=(0.0, 0.0)):
        """Critical battery: fly home and descend. The simulator/test calls
        this when Battery.critical_battery goes true."""
        self.waypoints = [(float(home[0]), float(home[1]), 0.0)]
        self.wp_index = 0
        self.engaged = True
        self.emergency_land = True
        self.status = "EMERG LAND"

    # -- per-step update ---------------------------------------------------
    def update(self, fm, ctrl, dt):
        """Fill `ctrl` from the current flight state. No-op when disengaged."""
        if not self.engaged or not self.waypoints:
            return
        dt = max(float(dt), 1e-3)

        R = quat_to_matrix(fm.quat)
        pitch, roll, yaw = fm.attitude_euler()
        p, q, r = (float(v) for v in fm.omega)
        u, v, w = (float(x) for x in fm.vel_b)
        V = max(fm.airspeed(), 0.5)
        beta = math.asin(_clamp(v / V, -1.0, 1.0)) if fm.airspeed() > 0.5 else 0.0
        # Vertical speed, positive up.
        vw = R @ fm.vel_b
        vspeed = float(vw[2])

        tx, ty, talt = self.waypoints[self.wp_index]
        dx, dy = tx - fm.pos[0], ty - fm.pos[1]
        dist = math.hypot(dx, dy)
        final = (self.wp_index == len(self.waypoints) - 1)
        if dist < self.capture_radius:
            if final and not self.loop:
                # Captured the last waypoint: latch into a stable orbit
                # instead of chasing the point (which whips the bearing
                # around and makes the UAV weave).
                self._loiter = True
            else:
                self.wp_index = (self.wp_index + 1) % len(self.waypoints)
                tx, ty, talt = self.waypoints[self.wp_index]
                dx, dy = tx - fm.pos[0], ty - fm.pos[1]
                dist = math.hypot(dx, dy)

        # Emergency landing: glide toward home, flare near the ground.
        alt_cmd = talt
        v_cmd = self.cruise_speed
        if self.emergency_land:
            alt_cmd = max(0.0, dist * 0.06)   # ~3.4 deg glidepath to home
            v_cmd = 14.0                       # slow, power-off-ish glide
            if fm.pos[2] < 8.0:
                alt_cmd = 0.0

        # --- Heading -> bank -> aileron ---
        if self._loiter:
            # Orbit: fly near-tangential to the bearing-to-point, with a
            # radial correction that pulls the orbit toward orbit_radius.
            # (des < bearing+90deg turns inward, des > turns outward.)
            radial = _clamp((dist - self.orbit_radius) / self.orbit_radius,
                            -1.0, 1.0)
            des_heading = (math.atan2(dy, dx) + math.pi / 2.0
                           - self.k_orbit * radial)
        else:
            des_heading = math.atan2(dy, dx)
        herr = _wrap_pi(des_heading - yaw)
        bank_cmd = _clamp(-self.kp_h * herr, -self.bank_max, self.bank_max)
        ctrl.aileron = _clamp(-self.kp_r * (bank_cmd - roll) - self.kd_r * p,
                              -1.0, 1.0)

        # --- Altitude -> pitch -> elevator ---
        alt_err = alt_cmd - fm.pos[2]
        pitch_cmd = _clamp(self.kp_a * alt_err - self.kd_a * vspeed,
                           -self.pitch_max, self.pitch_max)
        ctrl.elevator = _clamp(self.kp_e * (pitch_cmd - pitch) - self.kd_e * q,
                               -1.0, 1.0)

        # --- Airspeed -> throttle ---
        thr = self.thr_base + self.kp_v * (v_cmd - V)
        if self.emergency_land and fm.pos[2] < 8.0:
            thr = 0.12  # near-idle for the flare
        ctrl.throttle = _clamp(thr, 0.0, 1.0)

        # --- Sideslip -> rudder (coordinated turns) ---
        ctrl.rudder = _clamp(-self.kp_b * beta - self.kd_b * r, -0.6, 0.6)

        if not self.emergency_land:
            if self._loiter:
                self.status = "LOITER"
            else:
                self.status = f"NAV WP{self.wp_index + 1}/{len(self.waypoints)}"
