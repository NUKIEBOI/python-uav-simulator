"""Gimballed camera payload: scan-and-detect simulation.

HEADLESS: NumPy only. The camera is a 2-axis gimbal under the fuselage with
a configurable field of view and max range. Each step, every known ground
object is projected into the camera frame; objects inside the FOV and range
are "detected", with type, world position, range, bearing (relative to the
UAV nose), and a confidence that falls with range.

IMPORTANT HONESTY NOTE (also stated in the README): this is a *geometric*
detection model -- it answers "is the object inside the camera frustum?"
It stands in for real computer vision, which would add false positives,
missed detections, lighting/weather effects, and classification uncertainty.
The geometry, gimbal math, and detection-report plumbing are all real; the
"camera" just never has to deal with actual pixels.

Gimbal conventions (body frame: x fwd, y right, z up):
  * pan: rotation about body +z. Positive pan looks RIGHT.
  * tilt: rotation about body +y. Positive tilt looks DOWN
    (consistent with the flight model, where +y pitch is nose-down).
  * Camera boresight at pan=0, tilt=0 points straight ahead (+x).
"""

import math

import numpy as np

from flight_model import quat_to_matrix


# ---------------------------------------------------------------------------
# Target field: 12 detectable ground objects near (600, 300).
# ---------------------------------------------------------------------------

def build_target_field():
    """Return the canonical list of detectable objects.

    Each object: {"id", "type", "pos": (x, y, z), "size": approx. meters,
    "color": (r, g, b) for the 3D view}. The simulator, the headless test,
    and the unit tests all share this one definition.
    """
    objs = []
    # 4 colored landing pads (3x3 m squares, flat on the ground).
    pads = [
        ("pad-red", (520.0, 220.0), (0.85, 0.15, 0.15)),
        ("pad-blue", (680.0, 240.0), (0.15, 0.35, 0.85)),
        ("pad-green", (560.0, 400.0), (0.15, 0.70, 0.20)),
        ("pad-yellow", (720.0, 420.0), (0.90, 0.80, 0.15)),
    ]
    for oid, (x, y), color in pads:
        objs.append({"id": oid, "type": "landing_pad",
                     "pos": (x, y, 0.05), "size": 3.0, "color": color})
    # 6 box "vehicles" (4 x 2 x 1.5 m).
    veh_xy = [(480.0, 300.0), (620.0, 180.0), (760.0, 320.0),
              (540.0, 480.0), (700.0, 520.0), (640.0, 360.0)]
    for i, (x, y) in enumerate(veh_xy, 1):
        objs.append({"id": f"veh-{i:02d}", "type": "vehicle",
                     "pos": (x, y, 0.75), "size": 4.0,
                     "color": (0.35, 0.38, 0.30)})
    # 2 "person" markers (small, hard to spot -- the real challenge).
    for i, (x, y) in enumerate([(590.0, 260.0), (670.0, 460.0)], 1):
        objs.append({"id": f"person-{i:02d}", "type": "person",
                     "pos": (x, y, 0.5), "size": 0.6,
                     "color": (0.80, 0.55, 0.30)})
    return objs


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class GimbalCamera:
    def __init__(self, fov_h_deg=60.0, fov_v_deg=45.0, max_range_m=300.0):
        self.fov_h = math.radians(fov_h_deg)
        self.fov_v = math.radians(fov_v_deg)
        self.max_range = float(max_range_m)
        # Gimbal travel limits (radians).
        self.pan_limit = 2.90    # ~+/-166 deg
        self.tilt_min = -0.20    # slightly above horizontal
        self.tilt_max = 1.65     # near straight down

    # -- geometry ---------------------------------------------------------
    @staticmethod
    def _rot_z(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _rot_y(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def camera_rotation(self, R_body, pan, tilt):
        """World-from-camera rotation for the given gimbal angles."""
        return R_body @ self._rot_z(pan) @ self._rot_y(tilt)

    # -- detection ----------------------------------------------------------
    def detect(self, fm, pan, tilt, objects):
        """Project every object into the camera frame; return detections.

        Each detection: {"id", "type", "pos", "range_m", "bearing_deg",
        "confidence"}. Bearing is measured from the UAV nose (+x), positive
        to the right. Deterministic: same geometry always gives the same
        report.
        """
        R_body = quat_to_matrix(fm.quat)
        R_cam = self.camera_rotation(R_body, pan, tilt)
        # Gimbal ball sits slightly below/ahead of the CG.
        cam_pos = fm.pos + R_body @ np.array([0.6, 0.0, -0.35])

        detections = []
        for obj in objects:
            d_w = np.array(obj["pos"], float) - cam_pos
            rng = float(np.linalg.norm(d_w))
            if rng > self.max_range or rng < 0.5:
                continue
            d_cam = R_cam.T @ d_w
            if d_cam[0] < 0.5:                      # behind the camera
                continue
            bearing = math.atan2(d_cam[1], d_cam[0])
            elev = math.atan2(d_cam[2], math.hypot(d_cam[0], d_cam[1]))
            if abs(bearing) > self.fov_h / 2 or abs(elev) > self.fov_v / 2:
                continue
            # Bearing relative to the UAV nose for the operator display.
            d_body = R_body.T @ d_w
            bearing_nose = math.degrees(math.atan2(d_body[1], d_body[0]))
            # Confidence falls with range; bigger objects read slightly
            # better. (A real CV system would be far noisier.)
            conf = 1.15 - rng / self.max_range
            conf *= max(0.75, min(1.10, 0.75 + 0.25 * (obj["size"] / 2.0)))
            conf = round(max(0.05, min(0.99, conf)), 2)
            detections.append({
                "id": obj["id"],
                "type": obj["type"],
                "pos": tuple(float(v) for v in obj["pos"]),
                "range_m": round(rng, 1),
                "bearing_deg": round(bearing_nose, 1),
                "confidence": conf,
            })
        # Nearest first -- like a real operator display would sort.
        detections.sort(key=lambda d: d["range_m"])
        return detections


def sweep_angles(t):
    """Auto-scan gimbal pattern: slow pan sweep with a drifting tilt.

    Returns (pan_rad, tilt_rad). The pan sweeps +/-~57 deg on a 14 s period
    while the tilt breathes around ~50 deg down on a 53 s period, so the
    footprint walks a wedge across the ground -- the classic "lawnmower
    stare" used when searching an area. Deterministic in t.
    """
    pan = 1.0 * math.sin(2.0 * math.pi * t / 14.0)
    tilt = 0.90 + 0.35 * math.sin(2.0 * math.pi * t / 53.0 + 1.0)
    return pan, tilt
