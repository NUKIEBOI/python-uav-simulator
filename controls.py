"""Pilot + payload control state.

Flight channels (same conventions as the flight model):
  elevator  -1..+1   (+1 = pull back / nose up)
  aileron   -1..+1   (+1 = roll right)
  rudder    -1..+1   (+1 = nose right)
  throttle   0..+1
  trim      -1..+1   (+ = nose-up trim)

Payload channels:
  gimbal_pan   -1..+1  (scaled to +/-166 deg; + = look right)
  gimbal_tilt  -1..+1  (scaled to -11 deg .. +95 deg; + = look down)
  scan_sweep   bool    (when on, the gimbal follows the auto-scan pattern)
  log_detections bool  (detection list panel visible)

The autopilot writes the four flight channels directly when engaged; the
pilot's sticks are ignored for flight (gimbal stays manual unless scanning).
"""

from dataclasses import dataclass


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# Gimbal travel, radians (mirrors sensors.GimbalCamera limits).
PAN_LIMIT = 2.90
TILT_MIN = -0.20
TILT_MAX = 1.65


@dataclass
class ControlState:
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0
    trim: float = 0.0
    brakes: bool = False
    gimbal_pan: float = 0.0
    gimbal_tilt: float = 0.0   # -1..+1 normalized; see gimbal_tilt_rad()
    scan_sweep: bool = False
    log_detections: bool = True

    # --- Trim -----------------------------------------------------------
    def trim_up(self, amount: float = 0.05):
        self.trim = clamp(self.trim + amount, -1.0, 1.0)

    def trim_down(self, amount: float = 0.05):
        self.trim = clamp(self.trim - amount, -1.0, 1.0)

    def reset_trim(self):
        self.trim = 0.0

    def elevator_total(self, trim_authority: float = 0.6) -> float:
        return clamp(self.elevator + self.trim * trim_authority, -1.0, 1.0)

    # --- Gimbal -----------------------------------------------------------
    def gimbal_pan_rad(self) -> float:
        return clamp(self.gimbal_pan, -1.0, 1.0) * PAN_LIMIT

    def gimbal_tilt_rad(self) -> float:
        """Normalized -1..+1 mapped onto [TILT_MIN, TILT_MAX]."""
        n = clamp(self.gimbal_tilt, -1.0, 1.0)
        return TILT_MIN + (n + 1.0) / 2.0 * (TILT_MAX - TILT_MIN)

    def nudge_gimbal(self, dpan: float, dtilt: float):
        self.gimbal_pan = clamp(self.gimbal_pan + dpan, -1.0, 1.0)
        self.gimbal_tilt = clamp(self.gimbal_tilt + dtilt, -1.0, 1.0)

    # --- Housekeeping ---------------------------------------------------
    def reset(self):
        self.elevator = 0.0
        self.aileron = 0.0
        self.rudder = 0.0
        self.throttle = 0.0
        self.trim = 0.0
        self.brakes = False
        self.gimbal_pan = 0.0
        self.gimbal_tilt = 0.0
        self.scan_sweep = False
        self.log_detections = True

    def normalized(self) -> "ControlState":
        c = ControlState()
        c.elevator = clamp(self.elevator, -1, 1)
        c.aileron = clamp(self.aileron, -1, 1)
        c.rudder = clamp(self.rudder, -1, 1)
        c.throttle = clamp(self.throttle, 0, 1)
        c.trim = clamp(self.trim, -1, 1)
        c.brakes = bool(self.brakes)
        c.gimbal_pan = clamp(self.gimbal_pan, -1, 1)
        c.gimbal_tilt = clamp(self.gimbal_tilt, -1, 1)
        c.scan_sweep = bool(self.scan_sweep)
        c.log_detections = bool(self.log_detections)
        return c
