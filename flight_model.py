"""6-DOF flight dynamics for a small fixed-wing surveillance UAV.

HEADLESS: this module imports only NumPy (plus uav_params/atmosphere/
controls). No graphics here, so the physics can be unit-tested and run
without a window.

Frames (same conventions as the companion airplane sim)
-------------------------------------------------------
* World: x east-ish, y north-ish, z UP (meters). Flat ground at z = 0.
* Body: x forward, y right, z up (right-handed).
  - Angular rates (p, q, r) about (x, y, z).
  - Positive q (pitch about +y) is nose-DOWN (right-hand rule about the
    right-pointing y axis). Positive r (about +z) is nose-RIGHT.
  - Attitude stored as quaternion [w, x, y, z], body -> world.

Integration: semi-implicit (symplectic) Euler with fixed 240 Hz substeps.
Deterministic (no randomness anywhere) and NaN-guarded.

Ground model: flat runway at z = 0; the CG sits at GEAR_HEIGHT on its gear.
Rolling friction, lateral tire grip, nosewheel steering from rudder, and a
pitch spring that holds the nose down until the pilot/autopilot rotates.
Liftoff when vertical force exceeds weight; touchdown on wheel contact.
"""

import math

import numpy as np

from uav_params import SurveillanceUAV
from atmosphere import Atmosphere
from controls import ControlState

G = 9.80665
GEAR_HEIGHT = 0.35     # m, CG height when sitting on its landing gear
SUBSTEP_HZ = 240.0     # physics substeps per second


# ---------------------------------------------------------------------------
# Quaternion helpers (scalar-first [w, x, y, z])
# ---------------------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12 or not math.isfinite(n):
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Body -> world rotation matrix from a scalar-first quaternion."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    """Quaternion for a rotation of `angle` radians about `axis`."""
    ax = np.array(axis, dtype=float)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = ax / n
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), ax[0] * s, ax[1] * s, ax[2] * s])


def lift_coefficient(alpha: float, uav) -> float:
    """CL(alpha): linear lift curve with a stall break past alpha_crit."""
    cl_max = uav.CL0 + uav.CL_alpha * uav.alpha_crit
    if abs(alpha) <= uav.alpha_crit:
        return uav.CL0 + uav.CL_alpha * alpha
    # Post-stall: decay the peak value as separation grows (positive alpha).
    if alpha > 0:
        excess = min(alpha - uav.alpha_crit, 0.35)
        return cl_max * (1.0 - 0.38 * (excess / 0.35))
    # Deep negative alpha: decay the (negative) linear value toward zero.
    cl_ref = uav.CL0 + uav.CL_alpha * -0.25
    excess = min(-0.25 - alpha, 0.35)
    return cl_ref * (1.0 - 0.38 * (excess / 0.35))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Flight model
# ---------------------------------------------------------------------------

class FlightModel:
    def __init__(self, uav=None):
        self.uav = uav or SurveillanceUAV()
        self.atmo = Atmosphere()
        self.time = 0.0
        self.on_ground = True
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self, pos=None, vel_b=None, quat=None, omega=None,
              on_ground: bool = True):
        self.pos = (np.array([0.0, 0.0, GEAR_HEIGHT]) if pos is None
                    else np.array(pos, float))
        self.vel_b = (np.zeros(3) if vel_b is None
                      else np.array(vel_b, float))
        self.quat = (np.array([1.0, 0.0, 0.0, 0.0]) if quat is None
                     else quat_normalize(np.array(quat, float)))
        self.omega = (np.zeros(3) if omega is None
                      else np.array(omega, float))
        self.on_ground = on_ground
        self.time = 0.0
        self._last = {}

    # -- public helpers ---------------------------------------------------
    def airspeed(self) -> float:
        return float(np.linalg.norm(self.vel_b))

    def attitude_euler(self):
        """Return (pitch_rad nose-up+, roll_rad left-positive, yaw_rad)."""
        R = quat_to_matrix(self.quat)
        pitch = math.asin(_clamp(R[2, 0], -1.0, 1.0))
        roll = math.asin(_clamp(R[2, 1], -1.0, 1.0))
        yaw = math.atan2(R[1, 0], R[0, 0])
        return pitch, roll, yaw

    def info(self) -> dict:
        return dict(self._last)

    def trim_de_for_level(self, V: float, alt_m: float) -> float:
        """Elevator setting (-1..1) that zeroes pitch moment in level flight."""
        uav = self.uav
        rho = self.atmo.density(alt_m)
        q = 0.5 * rho * V * V
        CL_need = uav.mass * G / (q * uav.wing_area)
        alpha = (CL_need - uav.CL0) / uav.CL_alpha

        def cm(de):
            return uav.Cm0 + uav.Cma * alpha + uav.Cmde * de * uav.elevator_limit

        lo, hi = -1.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if cm(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)

    # -- core -------------------------------------------------------------
    def step(self, dt: float, ctrl: ControlState):
        """Advance the simulation by dt seconds (clamped, deterministic)."""
        dt = _clamp(float(dt), 0.0, 0.1)
        n = max(1, int(round(dt * SUBSTEP_HZ)))
        h = dt / n
        for _ in range(n):
            self._substep(h, ctrl)
        self.time += dt

    # -- internals --------------------------------------------------------
    def _forces_moments(self, ctrl: ControlState):
        uav = self.uav
        R = quat_to_matrix(self.quat)

        rho = self.atmo.density(self.pos[2])
        u, v, w = self.vel_b
        V = math.sqrt(u * u + v * v + w * w)
        V_safe = max(V, 0.5)

        # alpha = atan2(-w, u): positive = nose ABOVE the relative wind.
        alpha = _clamp(math.atan2(-w, u), -1.2, 1.2)
        beta = math.asin(_clamp(v / V_safe, -1.0, 1.0)) if V > 0.5 else 0.0
        qbar = 0.5 * rho * V * V
        S, b, cbar = uav.wing_area, uav.wing_span, uav.wing_chord

        # Control surface deflections [rad]
        de = ctrl.elevator_total(uav.trim_authority) * uav.elevator_limit
        da = _clamp(ctrl.aileron, -1, 1) * uav.aileron_limit
        dr = _clamp(ctrl.rudder, -1, 1) * uav.rudder_limit

        # --- Aerodynamic forces (body frame) ---
        stalled = abs(alpha) > uav.alpha_crit and V > 5.0
        CL = lift_coefficient(alpha, uav)
        CD = uav.CD0 + uav.induced_k * CL * CL
        if stalled:
            CD += 0.15 * min(1.0, (abs(alpha) - uav.alpha_crit) / 0.2)
        L = qbar * S * CL
        D = qbar * S * CD
        Y = qbar * S * uav.CY_beta * beta

        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        # e_v: unit airflow vector; e_L: lift direction (tilts slightly
        # forward in body coords at positive alpha -- see airplane sim).
        ev = np.array([ca * cb, sb, -sa * cb])
        eL = np.array([sa, 0.0, ca])
        F_aero = -D * ev + L * eL + np.array([0.0, Y, 0.0])

        # --- Propulsion: electric pusher prop, thrust along body x ---
        thr = _clamp(ctrl.throttle, 0.0, 1.0)
        T = thr * min(uav.thrust_static, uav.prop_power / max(V, 6.0))
        F_aero[0] += T

        # --- Weight ---
        F_grav_b = R.T @ np.array([0.0, 0.0, -uav.mass * G])

        # --- Moments ---
        p, qrate, r = self.omega
        v_damp = max(V, 2.0)
        Cm = (uav.Cm0 + uav.Cma * alpha
              + uav.Cmq * (qrate * cbar / (2 * v_damp))
              + uav.Cmde * de)
        Cl = (uav.Clb * beta
              + uav.Clp * (p * b / (2 * v_damp))
              + uav.Clda * da + uav.Cldr * dr)
        Cn = (uav.Cnb * beta
              + uav.Cnr * (r * b / (2 * v_damp))
              + uav.Cnda * da + uav.Cndr * dr)
        M_aero = qbar * S * np.array([b * Cl, cbar * Cm, b * Cn])

        F = F_aero + F_grav_b
        M = M_aero

        # --- Ground model (damping scaled to a 20 kg airframe) ---
        weight_on_wheels = 0.0
        if self.on_ground:
            F_w = R @ F_aero
            lift_up = F_w[2]
            weight_on_wheels = max(0.0, uav.mass * G - lift_up)

            v_w = R @ self.vel_b
            vh = math.sqrt(v_w[0] ** 2 + v_w[1] ** 2)

            mu = 0.55 if ctrl.brakes else 0.035
            if vh > 0.05 and weight_on_wheels > 0:
                fdir = -v_w[:2] / vh
                F_fric_w = np.array([fdir[0] * mu * weight_on_wheels,
                                     fdir[1] * mu * weight_on_wheels, 0.0])
            else:
                F_fric_w = np.zeros(3)
            F += R.T @ F_fric_w

            # Lateral tire grip.
            hx, hy = R[0, 0], R[1, 0]
            hnorm = math.sqrt(hx * hx + hy * hy) + 1e-9
            hx, hy = hx / hnorm, hy / hnorm
            v_along = v_w[0] * hx + v_w[1] * hy
            v_lat = np.array([v_w[0] - v_along * hx,
                              v_w[1] - v_along * hy, 0.0])
            F_lat_w = (-uav.mass * 3.0 * v_lat
                       * min(1.0, weight_on_wheels / (uav.mass * G)))
            F += R.T @ F_lat_w

            pitch, roll, _yaw = self._euler_from_R(R)
            # Nosewheel steering + yaw damping (scaled for 20 kg).
            M[2] += dr * weight_on_wheels * 0.35 - r * 60.0
            # Pitch spring toward taxi attitude; pilot overcomes it to rotate.
            M[1] += (pitch - 0.02) * uav.mass * G * 0.6 - qrate * 60.0
            # Gear holds wings level.
            M[0] += -roll * uav.mass * G * 1.0 - p * 80.0

        self._last = {
            "alpha": alpha, "beta": beta, "V": V, "rho": rho, "qbar": qbar,
            "CL": CL, "CD": CD, "thrust": T, "stalled": stalled,
            "de_rad": de, "da_rad": da, "dr_rad": dr,
            "weight_on_wheels": weight_on_wheels,
        }
        return F, M

    @staticmethod
    def _euler_from_R(R):
        pitch = math.asin(_clamp(R[2, 0], -1.0, 1.0))
        roll = math.asin(_clamp(R[2, 1], -1.0, 1.0))
        yaw = math.atan2(R[1, 0], R[0, 0])
        return pitch, roll, yaw

    def _substep(self, h: float, ctrl: ControlState):
        uav = self.uav
        F, M = self._forces_moments(ctrl)

        # Angular dynamics: I*w_dot = M - w x (I w)
        I = np.array([uav.Ixx, uav.Iyy, uav.Izz])
        Iw = I * self.omega
        w_dot = (M - np.cross(self.omega, Iw)) / I
        self.omega = self.omega + w_dot * h

        # Translational dynamics in the rotating body frame.
        v_dot = F / uav.mass - np.cross(self.omega, self.vel_b)
        self.vel_b = self.vel_b + v_dot * h

        # Attitude: q_dot = 0.5 * q otimes [0, w]; renormalize every substep.
        qd = 0.5 * quat_multiply(self.quat, np.array([0.0, *self.omega]))
        self.quat = quat_normalize(self.quat + qd * h)

        # Position in the world frame.
        R = quat_to_matrix(self.quat)
        v_w = R @ self.vel_b
        self.pos = self.pos + v_w * h

        # Ground contact projection / liftoff / touchdown.
        if self.on_ground:
            Fw_up = (R @ (F - R.T @ np.array([0.0, 0.0, -uav.mass * G])))[2]
            if Fw_up > uav.mass * G and v_w[2] >= -0.01:
                self.on_ground = False
            else:
                self.pos[2] = GEAR_HEIGHT
                v_w = R @ self.vel_b
                if v_w[2] < 0:
                    self.vel_b = self.vel_b - R.T @ np.array([0.0, 0.0, v_w[2]])
        else:
            if self.pos[2] <= GEAR_HEIGHT:
                v_w = R @ self.vel_b
                if v_w[2] < 0:
                    self.pos[2] = GEAR_HEIGHT
                    self.on_ground = True

        # Hard safety net: nothing leaves this function non-finite.
        for arr in (self.pos, self.vel_b, self.quat, self.omega):
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError("non-finite state in _substep")
