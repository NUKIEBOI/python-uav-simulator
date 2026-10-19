"""Panda3D visualization for the fixed-wing UAV simulator.

Importing this module does NOT start the application or open a window: it
only defines the UAVSim class and helpers. The app starts when main() is
called (see main.py), which keeps `python main.py --test` headless.

Coordinate conversion: physics uses x-forward / y-right / z-up while Panda3D
uses x-right / y-forward / z-up. sim_to_panda() maps positions; orientation
uses sim_quat_to_hpr() (same derivation as the companion airplane sim).

Onboard cameras (the point of this sim -- you fly looking THROUGH them):
  * chase ....... third-person follow view (no feed dressing)
  * fpv ......... nose camera: letterboxed feed, reticle, telemetry, REC
  * payload ..... gimbal ball camera: same feed dressing PLUS detection
                  markers projected onto the feed, and it follows YOUR
                  pan/tilt inputs (or the auto-scan sweep)
  * tower ....... fixed ground camera tracking the UAV
Cycle with C. The FPV and payload feeds are dressed like real UAV video:
letterbox bars, center reticle, corner telemetry, blinking REC.
"""

import math
import random

import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    MouseButton,
    NodePath,
    PNMImage,
    Point3,
    TextNode,
    Texture,
    TextureStage,
    TransparencyAttrib,
    Vec4,
    WindowProperties,
)

from uav_params import SurveillanceUAV
from flight_model import FlightModel, quat_to_matrix, quat_multiply, quat_from_axis_angle
from autopilot import Autopilot
from battery import Battery
from controls import ControlState
from sensors import GimbalCamera, build_target_field, sweep_angles

DEG = 180.0 / math.pi
MS_TO_KTS = 1.94384
M_TO_FT = 3.28084
MS_TO_FPM = 196.85

# Sun travel direction in Panda coords, derived from the light's Hpr(35,-50,0)
# (see _build_world). SUN_DIR points from the scene toward the sun.
_SUN_TRAVEL = np.array([math.sin(math.radians(35)) * math.cos(math.radians(-50)),
                        math.cos(math.radians(35)) * math.cos(math.radians(-50)),
                        math.sin(math.radians(-50))])
SUN_DIR = -_SUN_TRAVEL / np.linalg.norm(_SUN_TRAVEL)


# ---------------------------------------------------------------------------
# Procedural geometry helpers (no external assets)
# ---------------------------------------------------------------------------

def make_box_node(name, sx, sy, sz, color):
    """Colored box GeomNode centered on the origin (caller frame)."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    cwriter = GeomVertexWriter(vdata, "color")
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    verts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
             (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    for v in verts:
        vwriter.addData3(*v)
        cwriter.addData4(color[0], color[1], color[2], 1.0)
    tris = GeomTriangles(Geom.UHStatic)
    for a, b, c, d in [(0, 1, 2, 3), (4, 6, 5, 7), (0, 4, 5, 1),
                       (2, 6, 7, 3), (0, 3, 7, 4), (1, 5, 6, 2)]:
        tris.addVertices(a, b, c)
        tris.addVertices(a, c, d)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_cone_node(name, r_base, height, color, segments=10):
    """Cone with its base ring at z=0 and apex at z=height."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    vw.addData3(0, 0, 0)
    cw.addData4(*color, 1.0)
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        vw.addData3(r_base * math.cos(a), r_base * math.sin(a), 0)
        cw.addData4(*color, 1.0)
    vw.addData3(0, 0, height)
    cw.addData4(*color, 1.0)
    apex = segments + 1
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        a, b = 1 + i, 1 + (i + 1) % segments
        tris.addVertices(0, b, a)      # base fan
        tris.addVertices(a, b, apex)   # side
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_sphere_node(name, radius, sectors=28, rings=14):
    """UV-mapped sphere (equirect-ish: v=1 at the top), viewed from inside."""
    fmt = GeomVertexFormat.getV3t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    uw = GeomVertexWriter(vdata, "texcoord")
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vw.addData3(x, y, z)
            uw.addData2(s / sectors, 1.0 - r / rings)
    tris = GeomTriangles(Geom.UHStatic)
    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + 1
            c = a + sectors + 1
            d = c + 1
            tris.addVertices(a, c, b)
            tris.addVertices(b, c, d)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_cylinder_node(name, radius, height, color, segments=14):
    """Upright cylinder: side wall + top cap, base ring at z=0."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    for zz in (0.0, height):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            vw.addData3(radius * math.cos(a), radius * math.sin(a), zz)
            cw.addData4(*color, 1.0)
    top_center = 2 * segments
    vw.addData3(0, 0, height)
    cw.addData4(*color, 1.0)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        a0, b0 = i, (i + 1) % segments
        a1, b1 = segments + i, segments + (i + 1) % segments
        tris.addVertices(a0, a1, b1)
        tris.addVertices(a0, b1, b0)
        tris.addVertices(top_center, a1, b1)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


# --- Lofted airframe geometry -------------------------------------------
# Smooth, real-airplane shapes (tapered fuselage, airfoil wings) instead of
# stacked boxes.

# Simple cambered airfoil profile: (chord fraction, thickness fraction).
AIRFOIL = [(0.00, 0.000), (0.05, 0.030), (0.15, 0.048), (0.30, 0.058),
           (0.50, 0.055), (0.70, 0.042), (0.90, 0.020), (1.00, 0.003),
           (1.00, -0.003), (0.90, -0.014), (0.70, -0.028), (0.50, -0.035),
           (0.30, -0.034), (0.15, -0.026), (0.05, -0.013), (0.00, 0.000)]


def _ellipse_loop(hw, hh, zc, segs=16):
    """Elliptical cross-section loop: returns [(x, z), ...]."""
    return [(hw * math.cos(2.0 * math.pi * i / segs),
             zc + hh * math.sin(2.0 * math.pi * i / segs))
            for i in range(segs)]


def _airfoil_loop(le, chord, base, chord_axis=1):
    """Airfoil cross-section loop.

    le: leading-edge position along the chord axis; chord extends toward
    -chord_axis from there. base: offset along the thickness axis.
    chord_axis=1 -> loop points are (chordwise_y, thickness_z);
    chord_axis=0 -> loop points are (thickness_x, chordwise_y).
    """
    pts = []
    for cf, tf in AIRFOIL:
        c = le - cf * chord
        t = base + tf * chord
        pts.append((c, t) if chord_axis == 1 else (t, c))
    return pts


def make_loft_node(name, sections, color, axis="y"):
    """Loft closed cross-section loops into a smooth solid.

    sections: list of (pos, [(u, v), ...]) with equal-length loops.
    axis: 'y' -> vertex (u, pos, v); 'x' -> (pos, u, v); 'z' -> (u, v, pos).
    Stitches consecutive loops and caps both ends.
    """
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    n = len(sections[0][1])
    for pos, loop in sections:
        for (u, v) in loop:
            if axis == "y":
                vw.addData3(u, pos, v)
            elif axis == "x":
                vw.addData3(pos, u, v)
            else:
                vw.addData3(u, v, pos)
            cw.addData4(*color, 1.0)
    tris = GeomTriangles(Geom.UHStatic)
    for s in range(len(sections) - 1):
        a0 = s * n
        b0 = (s + 1) * n
        for i in range(n):
            j = (i + 1) % n
            tris.addVertices(a0 + i, b0 + i, b0 + j)
            tris.addVertices(a0 + i, b0 + j, a0 + j)
    for s, flip in ((0, True), (len(sections) - 1, False)):
        pos, loop = sections[s]
        cu = sum(p[0] for p in loop) / n
        cv = sum(p[1] for p in loop) / n
        ci = vdata.getNumRows()
        if axis == "y":
            vw.addData3(cu, pos, cv)
        elif axis == "x":
            vw.addData3(pos, cu, cv)
        else:
            vw.addData3(cu, cv, pos)
        cw.addData4(*color, 1.0)
        base = s * n
        for i in range(n):
            j = (i + 1) % n
            if flip:
                tris.addVertices(ci, base + j, base + i)
            else:
                tris.addVertices(ci, base + i, base + j)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def sim_to_panda(pos_sim):
    """Map sim (x fwd, y right, z up) to Panda3D (x right, y fwd, z up)."""
    x, y, z = pos_sim
    return (y, x, z)


def sim_quat_to_hpr(quat):
    """(heading, pitch, roll) degrees in Panda3D convention from sim quat."""
    R = quat_to_matrix(quat)
    Rp = np.array([
        [R[1, 1], R[1, 0], R[1, 2]],
        [R[0, 1], R[0, 0], R[0, 2]],
        [R[2, 1], R[2, 0], R[2, 2]],
    ])
    M = Rp.T
    pitch = math.asin(max(-1.0, min(1.0, M[2, 1])))
    heading = math.atan2(-M[0, 1], M[1, 1])
    roll = math.atan2(-M[2, 0], M[2, 2])
    return heading * DEG, pitch * DEG, roll * DEG


HELP_TEXT = """\
CONTROLS -- flight                     CONTROLS -- payload & cameras
  Up / W ..... nose down (push)          Z / X ..... gimbal tilt down / up
  Down / S ... nose up (pull)            N / M ..... gimbal pan left / right
  Left / A ... roll left                 V ......... auto-scan sweep on/off
  Right / D .. roll right                L ......... detection log on/off
                                             U ......... autopilot engage/disengage
  Q / E ...... rudder left / right
  Shift ...... throttle up
  Ctrl ....... throttle down            C ......... camera: chase/FPV/payload/
  T / G ...... trim nose-up / nose-down           tower/top-down/free
  B .......... wheel brakes             P / Space . pause
  H .......... toggle this help         R ......... reset flight
                                        Esc ....... quit

Free cam (C until "free"): WASD flies the camera, Q/E go down/up, hold
Shift for 4x speed, scroll wheel adjusts speed, hold right-mouse to look.
Flight keys steer the camera there; the UAV keeps doing what it was doing.

Takeoff: full throttle (hold Shift), rotate at ~12 m/s (pull S gently),
climb out, then press U for autopilot. Press V to start the scan sweep and
watch the payload feed pick up targets.
"""

# Default autonomous mission: a circuit over the target field at 80 m.
DEFAULT_WAYPOINTS = [
    (600.0, 100.0, 80.0),
    (850.0, 500.0, 80.0),
    (400.0, 500.0, 80.0),
    (350.0, 100.0, 80.0),
]


class UAVSim(ShowBase):
    """Panda3D fixed-wing UAV simulator application."""

    CAM_NAMES = ("chase", "fpv", "payload", "tower", "topdown", "free")

    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.uav = SurveillanceUAV()
        self.fm = FlightModel(self.uav)
        self.ctrl = ControlState()
        self.ap = Autopilot(cruise_speed=self.uav.cruise_speed)
        self.ap.set_waypoints(DEFAULT_WAYPOINTS)
        self.battery = Battery()
        self.camera_payload = GimbalCamera()
        self.objects = build_target_field()

        self.paused = False
        self.cam_mode = 0
        self.prop_angle = 0.0
        self.keys = {}
        self.detections = []       # latest detection report
        self.detect_log = []       # recent detections for the log panel
        self._seen_ids = set()
        self._rec_t = 0.0
        self._pan, self._tilt = 0.0, 0.0
        # World animation state.
        self.clouds = []
        self.radar_dome = None
        self.sock_pivot = None
        # Free-camera state (C -> "free" mode).
        self.fc_h = 0.0
        self.fc_p = 0.0
        self.fc_speed = 30.0
        self._fc_rmb = False
        self._fc_mx = 0.0
        self._fc_my = 0.0
        self._fc_settle = 0

        self._build_world()
        self._build_uav()
        self._build_targets()
        self._build_hud()
        self._build_feed_overlay()
        self._bind_keys()

        self.taskMgr.add(self._update, "sim-update")

    # -- world ------------------------------------------------------------
    def _tex_from_image(self, img, name):
        tex = Texture(name)
        tex.load(img)
        return tex

    def _make_ground_texture(self, size=1024, world=4000.0):
        rnd = random.Random(7)  # deterministic
        img = PNMImage(size, size)
        for j in range(size):
            for i in range(size):
                patch = (math.sin(i * 0.021) * math.sin(j * 0.017)
                         + 0.5 * math.sin(i * 0.053 + 1.7) * math.sin(j * 0.047 + 0.6))
                n = rnd.uniform(-0.035, 0.035)
                g = 0.40 + 0.10 * patch + n
                img.setXel(i, j, 0.30 + 0.6 * n, g, 0.24 + 0.5 * n)

        # (The runway is its own crisp card now, aligned with the takeoff
        # axis -- see _build_world. It is no longer painted into the grass.)
        # Target-field outline (the scan area), drawn faintly. NOTE: the
        # texture is in Panda frame (x right, y fwd); the field is sim
        # x∈[480,760], y∈[180,520], i.e. Panda x∈[180,520], y∈[480,760].
        # Image rows are flipped vertically on load, hence the (1-t).
        def px_x(x):
            return int((x + world / 2) / world * (size - 1))

        def px_y(y):
            return int((1.0 - (y + world / 2) / world) * (size - 1))

        fx0, fx1, fy0, fy1 = 100, 600, 400, 840
        for i in range(px_x(fx0), px_x(fx1) + 1):
            for j in (px_y(fy0), px_y(fy1)):
                img.setXel(i, j, 0.95, 0.55, 0.15)
        for j in range(px_y(fy1), px_y(fy0) + 1):
            for i in (px_x(fx0), px_x(fx1)):
                img.setXel(i, j, 0.95, 0.55, 0.15)
        tex = Texture("ground")
        tex.load(img)
        return tex

    def _make_runway_texture(self):
        """Crisp concrete runway: edge lines, centerline, thresholds,
        touchdown zones, and designation numbers. Runway runs along +Y
        (the UAV's takeoff axis)."""
        W, H = 128, 2048
        RW_W, RW_L = 30.0, 1240.0  # meters
        rnd = random.Random(7)
        img = PNMImage(W, H)

        def mx(x):
            return int((x + RW_W / 2) / RW_W * (W - 1))

        def my(y):
            # Image top (j=0) maps to v=1 = far/north end (+Y).
            return int((RW_L / 2 - y) / RW_L * (H - 1))

        white = (0.90, 0.90, 0.90)
        for j in range(H):
            for i in range(W):
                n = rnd.uniform(-0.022, 0.022)
                img.setXel(i, j, 0.33 + n, 0.33 + n, 0.34 + n)

        def hline(y0_m, y1_m, x0_m, x1_m, col=white):
            for j in range(my(y1_m), my(y0_m) + 1):
                for i in range(mx(x0_m), mx(x1_m) + 1):
                    if 0 <= i < W and 0 <= j < H:
                        img.setXel(i, j, *col)

        # Edge lines.
        hline(-600, 600, 13.2, 13.9)
        hline(-600, 600, -13.9, -13.2)
        # Centerline dashes: 30 m dash, 30 m gap.
        y = -585.0
        while y < 585.0:
            hline(y, y + 30.0, -0.45, 0.45)
            y += 60.0
        # Threshold piano keys at both ends (8 bars).
        for end in (-1, 1):
            y_base = end * 606.0
            for bar in range(8):
                x = -12.6 + bar * 3.4
                hline(min(y_base, y_base - end * 18.0),
                      max(y_base, y_base - end * 18.0), x, x + 2.2)
        # Touchdown zone pairs.
        for end in (-1, 1):
            for d in (150.0, 300.0, 450.0):
                yc = end * (606.0 - d)
                hline(yc - 6.0, yc + 6.0, 6.5, 8.3)
                hline(yc - 6.0, yc + 6.0, -8.3, -6.5)

        # Designation numbers, 3x5 blocky font.
        font = {"0": ["111", "101", "101", "101", "111"],
                "2": ["111", "001", "111", "100", "111"],
                "7": ["111", "001", "010", "010", "010"],
                "9": ["111", "101", "111", "001", "111"]}

        def draw_number(text, y_center_m, rot180):
            cell = 8
            total_w = len(text) * 4 * cell - cell
            top = my(y_center_m) - (5 * cell) // 2
            cx = W // 2 - total_w // 2
            for di, ch in enumerate(text):
                for r in range(5):
                    for c in range(3):
                        if font[ch][r][c] == "1":
                            rr, cc = (4 - r, 2 - c) if rot180 else (r, c)
                            for dj in range(cell):
                                for di2 in range(cell):
                                    i = cx + di * 4 * cell + cc * cell + di2
                                    j = top + rr * cell + dj
                                    if 0 <= i < W and 0 <= j < H:
                                        img.setXel(i, j, *white)

        # "27" at the south end reads for a northbound takeoff;
        # "09" at the north end reads for a southbound one.
        draw_number("27", -520.0, rot180=False)
        draw_number("09", 520.0, rot180=True)
        return self._tex_from_image(img, "runway")

    def _make_concrete_texture(self, size=256, slab=64):
        """Poured-concrete slabs with seams, for aprons and taxiways."""
        rnd = random.Random(21)
        img = PNMImage(size, size)
        for j in range(size):
            for i in range(size):
                n = rnd.uniform(-0.02, 0.02)
                seam = (i % slab < 2) or (j % slab < 2)
                base = 0.36 if seam else 0.47
                img.setXel(i, j, base + n, base + n, base + 0.01 + n)
        return self._tex_from_image(img, "concrete")

    def _make_taxiway_texture(self):
        """Asphalt strip with a yellow centerline, tiled along its length."""
        W, H = 64, 512
        rnd = random.Random(33)
        img = PNMImage(W, H)
        for j in range(H):
            for i in range(W):
                n = rnd.uniform(-0.02, 0.02)
                img.setXel(i, j, 0.30 + n, 0.30 + n, 0.31 + n)
        # Yellow centerline (continuous, like a real taxiway).
        for j in range(H):
            for i in range(W // 2 - 2, W // 2 + 2):
                img.setXel(i, j, 0.75, 0.62, 0.15)
        tex = self._tex_from_image(img, "taxiway")
        tex.setWrapU(Texture.WMRepeat)
        tex.setWrapV(Texture.WMRepeat)
        return tex

    def _make_sky_texture(self):
        """Vertical gradient: deep zenith blue -> warm horizon haze."""
        img = PNMImage(8, 256)
        for j in range(256):
            t = j / 255.0  # 0 = image top = zenith
            if t < 0.5:
                k = t / 0.5
                r = 0.10 + (0.83 - 0.10) * k
                g = 0.30 + (0.87 - 0.30) * k
                b = 0.70 + (0.91 - 0.70) * k
            else:
                k = (t - 0.5) / 0.5
                r = 0.83 + (0.32 - 0.83) * k
                g = 0.87 + (0.36 - 0.87) * k
                b = 0.91 + (0.40 - 0.91) * k
            for i in range(8):
                img.setXel(i, j, r, g, b)
        return self._tex_from_image(img, "sky")

    def _make_glow_texture(self, size=96):
        """Soft radial glow (white core -> transparent), for sun and lights."""
        img = PNMImage(size, size, 4)
        for j in range(size):
            for i in range(size):
                dx = (i / (size - 1)) * 2.0 - 1.0
                dy = (j / (size - 1)) * 2.0 - 1.0
                d = math.sqrt(dx * dx + dy * dy)
                a = max(0.0, 1.0 - d) ** 2.2
                img.setXel(i, j, 1.0, 1.0, 1.0)
                img.setAlpha(i, j, a)
        return self._tex_from_image(img, "glow")

    def _make_cloud_texture(self, size=128):
        """Puffy cloud puff from composited radial blobs."""
        rnd = random.Random(1234)
        img = PNMImage(size, size, 4)
        blobs = [(rnd.uniform(0.25, 0.75), rnd.uniform(0.35, 0.65),
                  rnd.uniform(0.15, 0.30)) for _ in range(9)]
        for j in range(size):
            for i in range(size):
                x, y = i / size, j / size
                a = 0.0
                for bx, by, br in blobs:
                    d2 = ((x - bx) ** 2 + (y - by) ** 2) / (br * br)
                    if d2 < 4.0:
                        a += math.exp(-d2 * 1.6) * 0.75
                a = min(1.0, a)
                # Slightly darker base for depth.
                shade = 0.92 + 0.08 * (1.0 - min(1.0, a))
                img.setXel(i, j, shade, shade, 1.0)
                img.setAlpha(i, j, min(1.0, a * 1.15))
        return self._tex_from_image(img, "cloud")

    def _build_world(self):
        # Ground with field patchwork.
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        ground = self.render.attachNewNode(cm.generate())
        ground.setTexture(self._make_ground_texture())
        ground.setPos(0, 0, -0.05)
        # Solid fallback: if the texture ever fails to show, grass still does.
        gbase = NodePath(make_box_node("ground_base", 4000, 4000, 0.1,
                                       (0.32, 0.46, 0.26)))
        gbase.setPos(0, 0, -0.12)
        gbase.reparentTo(self.render)

        # Runway as its own card (crisp markings), along the takeoff axis
        # (+Y in Panda frame = +X in sim frame). The old texture-painted
        # runway was misaligned with the takeoff direction; this one isn't.
        rcm = CardMaker("runway")
        rcm.setFrame(-15, 15, -620, 620)
        runway = self.render.attachNewNode(rcm.generate())
        runway.setTexture(self._make_runway_texture())
        runway.setPos(0, 0, 0.06)
        rwbase = NodePath(make_box_node("runway_base", 30.5, 1240.5, 0.08,
                                        (0.36, 0.36, 0.37)))
        rwbase.setPos(0, 0, 0.0)
        rwbase.reparentTo(self.render)

        # Sky dome: gradient sphere, unlit, drawn behind everything.
        sky = self.render.attachNewNode(make_sphere_node("sky", 8000.0))
        sky.setTexture(self._make_sky_texture())
        sky.setTwoSided(True)
        sky.setLightOff()
        sky.setFogOff()
        sky.setDepthWrite(False)
        sky.setBin("background", -10)

        # Sun glow sprite, repositioned every frame relative to the camera.
        glow_tex = self._make_glow_texture()
        self.glow_tex = glow_tex
        scm = CardMaker("sun")
        scm.setFrame(-650, 650, -650, 650)
        self.sun_card = self.render.attachNewNode(scm.generate())
        self.sun_card.setTexture(glow_tex)
        self.sun_card.setColor(1.0, 0.93, 0.80, 1.0)
        self.sun_card.setBillboardPointEye()
        self.sun_card.setTransparency(TransparencyAttrib.MAlpha)
        self.sun_card.setLightOff()
        self.sun_card.setFogOff()
        self.sun_card.setDepthWrite(False)
        self.sun_card.setBin("background", -5)

        # Drifting clouds.
        cloud_tex = self._make_cloud_texture()
        rnd = random.Random(99)
        for _ in range(14):
            ccm = CardMaker("cloud")
            w = rnd.uniform(380, 750)
            ccm.setFrame(-w / 2, w / 2, -w / 3.2, w / 3.2)
            c = self.render.attachNewNode(ccm.generate())
            c.setTexture(cloud_tex)
            c.setTransparency(TransparencyAttrib.MAlpha)
            c.setLightOff()
            c.setBillboardPointEye()
            ang = rnd.uniform(0, 2 * math.pi)
            rad = rnd.uniform(600, 2700)
            c.setPos(rad * math.cos(ang), rad * math.sin(ang),
                     rnd.uniform(380, 680))
            c.setDepthWrite(False)
            self.clouds.append((c, rnd.uniform(2.0, 5.0)))

        # Lighting: warm sun + cool sky fill (no real-time shadows, no
        # auto-shaders: the fixed-function path is the most driver-compatible).
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.0, 0.94, 0.82, 1.0))
        self.sun_np = self.render.attachNewNode(sun)
        self.sun_np.setHpr(35, -50, 0)
        self.render.setLight(self.sun_np)
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.42, 0.47, 0.58, 1.0))
        self.render.setLight(self.render.attachNewNode(amb))

        # Distance haze.
        fog = Fog("haze")
        fog.setColor(0.80, 0.84, 0.89)
        fog.setExpDensity(0.00040)
        self.render.setFog(fog)

        self._build_airbase(rnd)

    def _build_airbase(self, rnd):
        """Military airbase: parallel taxiway, apron, control tower,
        hangars, hardened shelters, rotating radar, parked jets, and
        support vehicles. (Panda frame: x right, y fwd.)"""
        concrete = self._make_concrete_texture()
        concrete.setWrapU(Texture.WMRepeat)
        concrete.setWrapV(Texture.WMRepeat)

        def pave(name, x0, x1, y0, y1, z=0.04, rep_u=1, rep_v=1, tex=None):
            cm = CardMaker(name)
            cm.setFrame(x0, x1, y0, y1)
            n = self.render.attachNewNode(cm.generate())
            n.setTexture(concrete if tex is None else tex)
            n.setTexScale(TextureStage.getDefault(), rep_u, rep_v)
            n.setPos(0, 0, z)
            # Solid fallback under the textured card.
            base = NodePath(make_box_node(name + "_base", x1 - x0 + 0.5,
                                          y1 - y0 + 0.5, 0.06,
                                          (0.42, 0.42, 0.43)))
            base.setPos((x0 + x1) / 2.0, (y0 + y1) / 2.0, z - 0.05)
            base.reparentTo(self.render)
            return n

        # Parallel taxiway east of the runway + three connectors.
        pave("taxiway", 82, 98, -620, 620, tex=self._make_taxiway_texture(),
             rep_v=12)
        for yc in (-400, 0, 400):
            pave("connector_%d" % yc, 14, 84, yc - 7, yc + 7)
        # Main apron.
        pave("apron", 110, 290, -180, 180, rep_u=3, rep_v=6)

        gray = (0.60, 0.61, 0.64)
        dark = (0.36, 0.37, 0.40)

        # Control tower.
        tower_parts = [
            ("tshaft", 9, 9, 30, gray, (200, -240, 15)),
            ("tcab", 14, 14, 5.5, (0.13, 0.19, 0.27), (200, -240, 32.5)),
            ("troof", 15.5, 15.5, 1.0, dark, (200, -240, 35.8)),
            ("tmast", 0.5, 0.5, 9, dark, (200, -240, 40.5)),
        ]
        for name, ex, ey, ez, color, pos in tower_parts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.reparentTo(self.render)

        # Hangar row.
        for hy in (-100, -20, 60):
            walls = NodePath(make_box_node("hangar", 46, 32, 12, gray))
            walls.setPos(205, hy, 6)
            walls.reparentTo(self.render)
            roof = NodePath(make_box_node("hangar_roof", 49, 35, 1.4, dark))
            roof.setPos(205, hy, 12.7)
            roof.reparentTo(self.render)
            door = NodePath(make_box_node("hangar_door", 0.5, 30, 9,
                                          (0.15, 0.16, 0.19)))
            door.setPos(205 - 23.2, hy, 4.5)
            door.reparentTo(self.render)

        # Hardened aircraft shelters (earth tone).
        for hx, hy in ((262, -140), (262, 140)):
            sh = NodePath(make_box_node("has", 26, 20, 9, (0.46, 0.43, 0.35)))
            sh.setPos(hx, hy, 4.5)
            sh.reparentTo(self.render)
            shr = NodePath(make_box_node("has_roof", 28, 22, 1.2,
                                         (0.38, 0.35, 0.28)))
            shr.setPos(hx, hy, 9.6)
            shr.reparentTo(self.render)

        # Radar: pedestal + rotating dome.
        ped = NodePath(make_box_node("radar_ped", 7, 7, 10, gray))
        ped.setPos(330, 260, 5)
        ped.reparentTo(self.render)
        self.radar_dome = NodePath(make_sphere_node("radome", 4.6))
        self.radar_dome.setColor(0.90, 0.90, 0.92)
        self.radar_dome.setPos(330, 260, 14.6)
        self.radar_dome.reparentTo(self.render)

        # Parked jets on the apron.
        for jx, jy in ((150, -120), (150, -40), (150, 40)):
            self._build_static_jet(jx, jy, 90.0)

        # Fuel truck.
        truck = self.render.attachNewNode("truck")
        tparts = [
            ("cab", 2.6, 2.8, 2.6, (0.36, 0.37, 0.29), (0, 3.6, 1.7)),
            ("tank", 2.8, 6.8, 2.8, (0.72, 0.72, 0.68), (0, -1.2, 1.9)),
            ("chassis", 2.4, 10.5, 0.7, dark, (0, 0, 0.75)),
        ]
        for name, ex, ey, ez, color, pos in tparts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(truck)
        for wx, wy in ((-1.2, 3.6), (1.2, 3.6), (-1.2, -0.5), (1.2, -0.5),
                       (-1.2, -3.2), (1.2, -3.2)):
            wheel = NodePath(make_box_node("wheel", 0.5, 0.9, 0.9,
                                           (0.08, 0.08, 0.09)))
            wheel.setPos(wx, wy, 0.45)
            wheel.reparentTo(truck)
        truck.setPos(235, -160, 0)
        truck.setH(15.0)

        # Apron floodlight poles.
        for fx, fy in ((115, -175), (285, -175), (115, 175), (285, 175)):
            pole = NodePath(make_box_node("flood_pole", 0.4, 0.4, 20, dark))
            pole.setPos(fx, fy, 10)
            pole.reparentTo(self.render)
            head = NodePath(make_box_node("flood_head", 3.2, 1.0, 0.8,
                                          (0.2, 0.2, 0.22)))
            head.setPos(fx, fy, 20.3)
            head.reparentTo(self.render)
            fcm = CardMaker("flood_glow")
            fcm.setFrame(-2.2, 2.2, -2.2, 2.2)
            g = self.render.attachNewNode(fcm.generate())
            g.setTexture(self.glow_tex)
            g.setColor(1.0, 0.92, 0.70, 0.9)
            g.setBillboardPointEye()
            g.setTransparency(TransparencyAttrib.MAlpha)
            g.setLightOff()
            g.setDepthWrite(False)
            g.setPos(fx, fy, 19.7)

        # Operations building.
        ops = NodePath(make_box_node("ops", 34, 16, 9, gray))
        ops.setPos(160, -205, 4.5)
        ops.reparentTo(self.render)
        opsr = NodePath(make_box_node("ops_roof", 36, 18, 1.0, dark))
        opsr.setPos(160, -205, 9.5)
        opsr.reparentTo(self.render)

        # Fuel farm: three cylindrical tanks.
        for i, tx in enumerate((146, 162, 178)):
            tank = NodePath(make_cylinder_node("fueltank", 6.0, 10.0,
                                               (0.78, 0.77, 0.72)))
            tank.setPos(tx, -320, 0)
            tank.reparentTo(self.render)
            band = NodePath(make_cylinder_node("tankband", 6.15, 1.2,
                                               (0.72, 0.10, 0.10)))
            band.setPos(tx, -320, 8.0)
            band.reparentTo(self.render)

        # Approach light masts past both runway thresholds.
        for end in (-1, 1):
            for k in range(5):
                ay = end * (660.0 + k * 30.0)
                mast = NodePath(make_box_node("amast", 0.35, 0.35, 7, dark))
                mast.setPos(0, ay, 3.5)
                mast.reparentTo(self.render)
                bar = NodePath(make_box_node("ambar", 5.0, 0.35, 0.35, dark))
                bar.setPos(0, ay, 7.0)
                bar.reparentTo(self.render)
                acm = CardMaker("aglow")
                acm.setFrame(-1.4, 1.4, -1.4, 1.4)
                ag = self.render.attachNewNode(acm.generate())
                ag.setTexture(self.glow_tex)
                ag.setColor(1.0, 1.0, 1.0, 0.95)
                ag.setBillboardPointEye()
                ag.setTransparency(TransparencyAttrib.MAlpha)
                ag.setLightOff()
                ag.setDepthWrite(False)
                ag.setPos(0, ay, 7.0)

        # Windsock near the south end.
        pole = NodePath(make_box_node("pole", 0.18, 0.18, 6.5,
                                      (0.75, 0.75, 0.78)))
        pole.setPos(28, -560, 3.25)
        pole.reparentTo(self.render)
        self.sock_pivot = self.render.attachNewNode("sock_pivot")
        self.sock_pivot.setPos(28, -560, 6.4)
        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("sock", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        segs, sock_len = 10, 2.6
        for ring, (yy, rr) in enumerate(((0.0, 0.50), (sock_len, 0.16))):
            for i in range(segs):
                a = 2.0 * math.pi * i / segs
                vw.addData3(rr * math.cos(a), yy, rr * math.sin(a))
                if ring == 0:
                    cw.addData4(0.95, 0.45, 0.10, 1.0)
                else:
                    cw.addData4(0.94, 0.94, 0.94, 1.0)
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(segs):
            a, b = i, (i + 1) % segs
            c, d = segs + i, segs + (i + 1) % segs
            tris.addVertices(a, c, b)
            tris.addVertices(b, c, d)
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        sock = NodePath(GeomNode("sock"))
        sock.node().addGeom(geom)
        sock.setTwoSided(True)
        sock.reparentTo(self.sock_pivot)

        # Trees: pines scattered clear of the runway, the base complex,
        # and the UAV target field (Panda x 100-600, y 400-840).
        trunk_c = (0.30, 0.22, 0.15)
        leaf_c = [(0.16, 0.34, 0.16), (0.20, 0.40, 0.18)]
        placed = 0
        while placed < 60:
            x = rnd.uniform(-1900, 1900)
            y = rnd.uniform(-1900, 1900)
            if abs(x) < 55 and abs(y) < 700:
                continue
            if x > 40 and abs(y) < 720:
                continue
            if math.hypot(x - 330, y - 260) < 70:
                continue
            if math.hypot(x, y) < 230:
                continue
            if 60 < x < 640 and 360 < y < 880:
                continue  # keep the scan field clear
            s = rnd.uniform(0.8, 1.7)
            trunk = NodePath(make_box_node("trunk", 0.5 * s, 0.5 * s,
                                           2.2 * s, trunk_c))
            trunk.setPos(x, y, 1.1 * s)
            trunk.reparentTo(self.render)
            lc = leaf_c[placed % 2]
            for rr, hh, zz in ((2.6, 4.2, 2.0), (1.8, 3.0, 4.6)):
                cone = NodePath(make_cone_node("pine", rr * s, hh * s, lc))
                cone.setPos(x, y, zz * s)
                cone.reparentTo(self.render)
            placed += 1

        # Runway edge lights: small posts with a warm glow.
        for y in range(-600, 601, 60):
            for x in (-16.5, 16.5):
                post = NodePath(make_box_node("elpost", 0.14, 0.14, 0.9,
                                              (0.2, 0.2, 0.22)))
                post.setPos(x, y, 0.45)
                post.reparentTo(self.render)
                lcm = CardMaker("elglow")
                lcm.setFrame(-0.8, 0.8, -0.8, 0.8)
                g = self.render.attachNewNode(lcm.generate())
                g.setTexture(self.glow_tex)
                g.setColor(1.0, 0.85, 0.55, 0.85)
                g.setBillboardPointEye()
                g.setTransparency(TransparencyAttrib.MAlpha)
                g.setLightOff()
                g.setDepthWrite(False)
                g.setPos(x, y, 1.1)

    def _build_static_jet(self, x, y, heading):
        """Simple parked military jet silhouette (procedural boxes)."""
        jet = self.render.attachNewNode("static_jet")
        gray = (0.44, 0.45, 0.48)
        dark = (0.14, 0.14, 0.16)
        glass = (0.10, 0.14, 0.20)
        parts = [
            ("jfus", 1.7, 11.0, 1.7, gray, (0, 0, 1.7)),
            ("jcanopy", 0.95, 2.6, 0.75, glass, (0, 2.4, 2.75)),
            ("jtail", 2.6, 1.0, 0.12, gray, (0, -4.7, 1.8)),
            ("jfin", 0.14, 1.3, 2.1, gray, (0, -4.7, 2.9)),
            ("jnozzle", 1.0, 0.9, 1.0, dark, (0, -5.7, 1.7)),
            ("jgear_n", 0.3, 0.3, 1.0, dark, (0, 3.4, 0.5)),
            ("jgear_l", 0.3, 0.3, 1.0, dark, (-1.1, -0.6, 0.5)),
            ("jgear_r", 0.3, 0.3, 1.0, dark, (1.1, -0.6, 0.5)),
        ]
        for name, ex, ey, ez, color, pos in parts:
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(jet)
        # Swept wings and tailplanes (tips angled back).
        for side in (-1, 1):
            wing = NodePath(make_box_node("jwing", 5.4, 1.7, 0.14, gray))
            wing.setPos(side * 3.1, -0.9, 1.55)
            wing.setH(-side * 28.0)
            wing.setTwoSided(True)
            wing.reparentTo(jet)
            stab = NodePath(make_box_node("jstab", 2.4, 0.9, 0.10, gray))
            stab.setPos(side * 1.5, -4.7, 1.8)
            stab.setH(-side * 22.0)
            stab.setTwoSided(True)
            stab.reparentTo(jet)
        # Pointed nose.
        nose = NodePath(make_cone_node("jnose", 0.85, 2.4, gray))
        nose.setP(-90)  # apex forward (+Y)
        nose.setPos(0, 5.5, 1.7)
        nose.reparentTo(jet)
        jet.setPos(x, y, 0)
        jet.setH(heading)

    # -- UAV model ----------------------------------------------------------
    def _build_uav(self):
        """Predator-style fixed-wing UAV from lofted airframe parts:
        smooth fuselage, airfoil wing and V-tail (Panda axes in parts)."""
        gray = (0.72, 0.74, 0.78)
        dark = (0.16, 0.17, 0.20)
        glass = (0.12, 0.18, 0.26)

        self.uav_node = self.render.attachNewNode("uav")

        def part(node, pos=(0, 0, 0)):
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(self.uav_node)
            return node

        # Fuselage: smooth tapered loft, tail to rounded nose.
        fus = [(y, _ellipse_loop(hw, hh, zc)) for (y, hw, hh, zc) in (
            (-1.70, 0.09, 0.09, 0.06), (-1.10, 0.19, 0.19, 0.05),
            (-0.30, 0.27, 0.29, 0.03), (0.60, 0.30, 0.32, 0.01),
            (1.30, 0.28, 0.30, -0.01), (1.80, 0.20, 0.20, -0.04),
            (2.05, 0.10, 0.10, -0.06))]
        part(NodePath(make_loft_node("fuselage", fus, gray)))
        # Glass nose sensor.
        nose_glass = NodePath(make_sphere_node("sensor_nose", 0.14))
        part(nose_glass, (0, 1.95, -0.10))
        nose_glass.setColor(*glass)

        # High wing: tapered airfoil lofts.
        for side in (-1, 1):
            s = side
            wing = [(s * 0.20, _airfoil_loop(0.72, 0.78, 0.30)),
                    (s * 2.25, _airfoil_loop(0.55, 0.55, 0.36))]
            part(NodePath(make_loft_node("wing", wing, gray, axis="x")))
        # V-tail: two airfoil panels angled like an inverted V.
        for side, roll in ((-1, 38), (1, -38)):
            fin = [(0.55, _airfoil_loop(-1.50, 0.62, 0.0, chord_axis=0)),
                   (1.45, _airfoil_loop(-1.75, 0.40, 0.0, chord_axis=0))]
            node = part(NodePath(make_loft_node("fin", fin, gray, axis="z")),
                        (0, 0, 0.12))
            node.setR(roll)
        # Gimbal ball under the nose.
        ball = NodePath(make_sphere_node("gimbal_ball", 0.16))
        part(ball, (0, 1.15, -0.38))
        ball.setColor(*dark)
        # Landing gear.
        for name, pos in (("gear_l", (-0.35, 0.5, -0.45)),
                          ("gear_r", (0.35, 0.5, -0.45)),
                          ("gear_n", (0, 1.5, -0.45))):
            part(NodePath(make_box_node(name, 0.12, 0.12, 0.45, dark)), pos)
        # Pusher prop at the tail, spinning about Panda y (forward axis).
        self.prop = self.uav_node.attachNewNode("prop")
        for nm, ex, ez in (("pb1", 0.09, 1.5), ("pb2", 1.5, 0.09)):
            blade = NodePath(make_box_node(nm, ex, 0.06, ez, dark))
            blade.setTwoSided(True)
            blade.reparentTo(self.prop)
        self.prop.setPos(0, -1.55, 0.10)

    # -- target objects -------------------------------------------------------
    def _build_targets(self):
        """Visuals for the 12 detectable objects (shared def in sensors.py)."""
        for obj in self.objects:
            x, y, z = obj["pos"]
            px, py = y, x  # sim -> panda
            color = obj["color"]
            if obj["type"] == "landing_pad":
                node = NodePath(make_box_node(obj["id"], 3.0, 3.0, 0.12, color))
                node.setPos(px, py, 0.06)
            elif obj["type"] == "vehicle":
                node = NodePath(make_box_node(obj["id"], 2.0, 4.2, 1.5, color))
                node.setPos(px, py, 0.75)
            else:  # person
                node = NodePath(make_box_node(obj["id"], 0.5, 0.5, 1.1, color))
                node.setPos(px, py, 0.55)
            node.setTwoSided(True)
            node.reparentTo(self.render)

    # -- HUD ------------------------------------------------------------------
    def _text(self, pos, scale=0.05, align=TextNode.ALeft, fg=(1, 1, 1, 1)):
        return OnscreenText(text="", pos=pos, scale=scale, align=align, fg=fg,
                            bg=(0, 0, 0, 0.45), mayChange=True)

    def _build_hud(self):
        self.hud_left = self._text((-1.72, 0.88), 0.055)
        self.hud_warn = OnscreenText(text="", pos=(0, 0.80), scale=0.075,
                                    align=TextNode.ACenter, fg=(1, 0.15, 0.15, 1),
                                    mayChange=True)
        self.hud_log = self._text((-1.72, -0.58), 0.042, fg=(1.0, 0.95, 0.6, 1))
        self.help_text = OnscreenText(text=HELP_TEXT, pos=(-1.70, 0.30),
                                     scale=0.040, align=TextNode.ALeft,
                                     fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.55),
                                     mayChange=False)
        self._build_attitude_indicator()

    def _build_attitude_indicator(self):
        """Compact AI: blue/brown ball that pitches and rolls under a fixed
        symbol, tucked at bottom-center."""
        root = NodePath("ai")
        root.reparentTo(self.aspect2d)
        root.setPos(0, 0, -0.74)
        self.ai_root = root
        S = 0.15  # half-size

        frame = CardMaker("ai_frame")
        frame.setFrame(-S, S, -S, S)
        fn = root.attachNewNode(frame.generate())
        fn.setColor(0.05, 0.05, 0.08, 0.85)
        fn.setBin("fixed", 10)
        fn.setDepthTest(False)

        self.ai_ball = root.attachNewNode("ai_ball")
        for nm, y0, y1, col in (("ai_sky", 0.0, 2 * S, (0.20, 0.45, 0.85, 1.0)),
                                ("ai_gnd", -2 * S, 0.0, (0.55, 0.38, 0.22, 1.0))):
            cm = CardMaker(nm)
            cm.setFrame(-S * 0.92, S * 0.92, y0, y1)
            n = self.ai_ball.attachNewNode(cm.generate())
            n.setColor(*col)
            n.setBin("fixed", 11)
            n.setDepthTest(False)
        # Horizon line.
        hcm = CardMaker("ai_horizon")
        hcm.setFrame(-S * 0.92, S * 0.92, -0.006, 0.006)
        hn = self.ai_ball.attachNewNode(hcm.generate())
        hn.setColor(1, 1, 1, 0.9)
        hn.setBin("fixed", 11)
        hn.setDepthTest(False)

        # Fixed aircraft symbol (stays put while the ball moves).
        sym = root.attachNewNode("ai_sym")
        for x0, x1 in [(-0.10, -0.025), (0.025, 0.10)]:
            wcm = CardMaker("ai_wing")
            wcm.setFrame(x0, x1, -0.010, 0.010)
            wn = sym.attachNewNode(wcm.generate())
            wn.setColor(1, 0.85, 0.2, 1)
            wn.setBin("fixed", 12)
            wn.setDepthTest(False)
        dot_cm = CardMaker("ai_dot")
        dot_cm.setFrame(-0.010, 0.010, -0.010, 0.010)
        dn = sym.attachNewNode(dot_cm.generate())
        dn.setColor(1, 0.85, 0.2, 1)
        dn.setBin("fixed", 12)
        dn.setDepthTest(False)

    # -- camera-feed dressing ---------------------------------------------------
    def _bar(self, y0, y1):
        cm = CardMaker("bar")
        cm.setFrame(-2.2, 2.2, y0, y1)
        n = self.aspect2d.attachNewNode(cm.generate())
        n.setColor(0, 0, 0, 1)
        n.setBin("fixed", 40)
        n.setDepthTest(False)
        return n

    def _build_feed_overlay(self):
        """Letterbox bars, reticle, REC, feed telemetry, detection markers.

        Shown only in the fpv / payload views to mimic a real UAV video feed.
        """
        self.feed_nodes = []
        # Letterbox bars (top/bottom ~16% each).
        for y0, y1 in ((0.68, 1.0), (-1.0, -0.68)):
            self.feed_nodes.append(self._bar(y0, y1))
        # Center reticle: thin cross.
        for (x0, x1, y0, y1) in ((-0.045, 0.045, -0.004, 0.004),
                                 (-0.004, 0.004, -0.06, 0.06)):
            cm = CardMaker("ret")
            cm.setFrame(x0, x1, y0, y1)
            n = self.aspect2d.attachNewNode(cm.generate())
            n.setColor(1, 1, 1, 0.85)
            n.setBin("fixed", 41)
            n.setDepthTest(False)
            self.feed_nodes.append(n)
        # REC indicator: red square + text, top-left of the feed.
        cm = CardMaker("rec_dot")
        cm.setFrame(-1.62, -1.56, 0.56, 0.62)
        dot = self.aspect2d.attachNewNode(cm.generate())
        dot.setColor(1, 0.1, 0.1, 1)
        dot.setBin("fixed", 41)
        dot.setDepthTest(False)
        self.feed_nodes.append(dot)
        self.rec_dot = dot
        self.rec_text = OnscreenText(text="REC", pos=(-1.52, 0.575), scale=0.05,
                                    align=TextNode.ALeft, fg=(1, 0.15, 0.15, 1),
                                    mayChange=False)
        self.rec_text.setBin("fixed", 41)
        self.feed_nodes.append(self.rec_text)
        # Feed telemetry, bottom-left inside the bars.
        self.feed_tel = OnscreenText(text="", pos=(-1.62, -0.60), scale=0.045,
                                    align=TextNode.ALeft, fg=(0.6, 1.0, 0.6, 1),
                                    mayChange=True)
        self.feed_tel.setBin("fixed", 41)
        self.feed_nodes.append(self.feed_tel)
        self.feed_label = OnscreenText(text="", pos=(1.62, 0.575), scale=0.05,
                                      align=TextNode.ARight, fg=(1, 1, 1, 0.9),
                                      mayChange=True)
        self.feed_label.setBin("fixed", 41)
        self.feed_nodes.append(self.feed_label)
        # Detection marker pool (payload view): "[id]" labels placed by
        # projecting each detection onto the feed each frame.
        self.markers = []
        for _ in range(12):
            m = OnscreenText(text="", pos=(0, 0), scale=0.045,
                             align=TextNode.ACenter, fg=(1.0, 0.85, 0.2, 1),
                             mayChange=True)
            m.setBin("fixed", 42)
            m.setDepthTest(False)
            self.markers.append(m)
            self.feed_nodes.append(m)
        self._set_feed_visible(False)

    def _set_feed_visible(self, vis):
        for n in self.feed_nodes:
            n.show() if vis else n.hide()

    # -- input ------------------------------------------------------------------
    def _bind_keys(self):
        a = self.accept
        # Held keys tracked in self.keys. Holding Shift/Ctrl renames the
        # other key events at the OS level ("w" arrives as "shift-w"), so
        # the modified variants need bindings too -- otherwise the pilot
        # can't steer while holding a throttle key.
        for k in ("w", "s", "a", "d", "arrow_up", "arrow_down", "arrow_left",
                  "arrow_right", "q", "e", "shift", "control",
                  "z", "x", "n", "m"):
            a(k, self._key_down, [k])
            a(k + "-up", self._key_up, [k])
            if k not in ("shift", "control"):
                a("shift-" + k, self._key_down, [k])
                a("shift-" + k + "-up", self._key_up, [k])
                a("control-" + k, self._key_down, [k])
                a("control-" + k + "-up", self._key_up, [k])

        def tap(key, fn):
            a(key, fn)
            a("shift-" + key, fn)
            a("control-" + key, fn)

        # Toggles (same controls as before; shift/ctrl variants added so
        # they fire even while a throttle key is held).
        tap("v", self._toggle_scan)
        tap("l", self._toggle_log)
        tap("c", self._cycle_camera)
        tap("p", self._toggle_pause)
        tap("space", self._toggle_pause)   # Space also pauses (P still does)
        tap("r", self._reset)
        tap("h", self._toggle_help)
        tap("b", self._toggle_brakes)
        tap("t", lambda: self.ctrl.trim_up())
        tap("g", lambda: self.ctrl.trim_down())
        tap("u", self._toggle_autopilot)    # autopilot engage/disengage
        tap("escape", self._quit)
        # Scroll wheel: free-cam speed (ignored in other camera modes).
        a("wheel_up", self._fc_speed_up)
        a("wheel_down", self._fc_speed_down)

    def _key_down(self, k):
        self.keys[k] = True

    def _key_up(self, k):
        self.keys[k] = False

    def _toggle_scan(self):
        self.ctrl.scan_sweep = not self.ctrl.scan_sweep

    def _toggle_log(self):
        self.ctrl.log_detections = not self.ctrl.log_detections

    def _toggle_brakes(self):
        self.ctrl.brakes = not self.ctrl.brakes

    def _toggle_help(self):
        self.help_text.toggleVis()

    def _toggle_pause(self):
        self.paused = not self.paused

    def _toggle_autopilot(self):
        if self.ap.engaged:
            self.ap.disengage()
        else:
            self.ap.engage()

    def _cycle_camera(self):
        self.cam_mode = (self.cam_mode + 1) % len(self.CAM_NAMES)
        if self.CAM_NAMES[self.cam_mode] == "free":
            # Place the free camera 45 m behind and 12 m above the UAV,
            # looking at it. WASD/QE/Shift then fly the camera itself.
            px, py, pz = sim_to_panda(self.fm.pos)
            h, _p, _r = sim_quat_to_hpr(self.fm.quat)
            hr = math.radians(h)
            campos = np.array([px + math.sin(hr) * 45.0,
                               py - math.cos(hr) * 45.0,
                               pz + 12.0])
            if self.camera is not None:
                self.camera.setPos(*campos)
            d = np.array([px, py, pz]) - campos
            n = max(1e-6, float(np.linalg.norm(d)))
            self.fc_h = math.degrees(math.atan2(-d[0], d[1]))
            self.fc_p = math.degrees(math.asin(max(-1.0, min(1.0, d[2] / n))))
            self.fc_speed = 30.0
            self._fc_rmb = False
        else:
            self._fc_set_cursor_locked(False)
            self._fc_rmb = False

    def _reset(self):
        self.fm.reset()
        self.ctrl.reset()
        self.battery.reset()
        self.ap.disengage()
        self.ap.set_waypoints(DEFAULT_WAYPOINTS)
        self.detections = []
        self.detect_log = []
        self._seen_ids = set()
        self.paused = False

    def _quit(self):
        import sys
        sys.exit(0)

    def _apply_held_keys(self, dt):
        k = self.keys
        c = self.ctrl
        if self.CAM_NAMES[self.cam_mode] == "free":
            # Free cam: WASD/QE/Shift fly the camera (see _move_camera), so
            # the flight surfaces ease to neutral and the throttle holds.
            # The gimbal still answers (it doesn't conflict with the camera).
            c.elevator += (0.0 - c.elevator) * min(1.0, dt * 6.0)
            c.aileron += (0.0 - c.aileron) * min(1.0, dt * 6.0)
            c.rudder += (0.0 - c.rudder) * min(1.0, dt * 6.0)
        else:
            # Flight sticks (airplane convention: Up/W = push/nose-down).
            pitch_in = (1.0 if (k.get("arrow_up") or k.get("w")) else 0.0) \
                - (1.0 if (k.get("arrow_down") or k.get("s")) else 0.0)
            roll_in = (1.0 if (k.get("arrow_right") or k.get("d")) else 0.0) \
                - (1.0 if (k.get("arrow_left") or k.get("a")) else 0.0)
            c.elevator = -pitch_in   # +1 elevator = nose up; key "up" pushes down
            c.aileron = roll_in
            c.rudder = (1.0 if k.get("e") else 0.0) - (1.0 if k.get("q") else 0.0)
            if k.get("shift"):
                c.throttle = min(1.0, c.throttle + 0.6 * dt)
            if k.get("control"):
                c.throttle = max(0.0, c.throttle - 0.6 * dt)
        # Gimbal: Z/X tilt down/up, N/M pan left/right.
        dtilt = (0.9 * dt if k.get("z") else 0.0) - (0.9 * dt if k.get("x") else 0.0)
        dpan = (0.9 * dt if k.get("m") else 0.0) - (0.9 * dt if k.get("n") else 0.0)
        c.nudge_gimbal(dpan, dtilt)

    # -- per-frame update ---------------------------------------------------------
    def _update(self, task):
        dt = min(self.clock.getDt(), 0.05)
        if not self.paused:
            self._apply_held_keys(dt)
            # Autopilot drives the flight channels when engaged.
            if self.ap.engaged:
                self.ap.update(self.fm, self.ctrl, dt)
            # Battery follows throttle; critical -> emergency auto-land.
            self.battery.step(dt, self.ctrl.throttle)
            if self.battery.critical_battery and not self.ap.emergency_land:
                self.ap.trigger_emergency_land()
            self.fm.step(dt, self.ctrl)

            # Gimbal angles: manual sticks or the auto-scan sweep.
            if self.ctrl.scan_sweep:
                pan, tilt = sweep_angles(self.fm.time)
            else:
                pan = self.ctrl.gimbal_pan_rad()
                tilt = self.ctrl.gimbal_tilt_rad()
            self._pan, self._tilt = pan, tilt

            # Payload detection (cheap: 12 objects).
            self.detections = self.camera_payload.detect(self.fm, pan, tilt,
                                                        self.objects)
            for d in self.detections:
                if d["id"] not in self._seen_ids:
                    self._seen_ids.add(d["id"])
                    self.detect_log.append(d)
                    self.detect_log = self.detect_log[-8:]

            # Prop spin follows throttle.
            self.prop_angle += self.ctrl.throttle * 45.0 * dt
            self.prop.setR(math.degrees(self.prop_angle) % 360.0)

        self._move_model()
        self._move_camera(dt)
        self._update_hud(dt)
        # Windsock sway + radar sweep.
        t = self.fm.time
        if self.sock_pivot is not None:
            self.sock_pivot.setH(18.0 * math.sin(t * 0.6) + 8.0 * math.sin(t * 1.7))
        if self.radar_dome is not None:
            self.radar_dome.setH((t * 25.0) % 360.0)
        # Clouds drift and wrap.
        for node, speed in self.clouds:
            x = node.getX() + speed * dt
            if x > 2900:
                x = -2900
            node.setX(x)
        return task.cont

    def _move_model(self):
        if self.camera is None:
            return  # headless: nothing to move
        self.uav_node.setPos(*sim_to_panda(self.fm.pos))
        self.uav_node.setHpr(*sim_quat_to_hpr(self.fm.quat))

    def _gimbal_world(self):
        """Gimbal ball world position + camera-frame quaternion (sim frame)."""
        R = quat_to_matrix(self.fm.quat)
        pos = self.fm.pos + R @ np.array([0.6, 0.0, -0.35])
        q = quat_multiply(quat_multiply(self.fm.quat,
                                        quat_from_axis_angle([0, 0, 1], self._pan)),
                          quat_from_axis_angle([0, 1, 0], self._tilt))
        return pos, q

    def _move_camera(self, dt=0.0):
        if self.camera is None:
            return
        mode = self.CAM_NAMES[self.cam_mode]
        R = quat_to_matrix(self.fm.quat)
        uav_p = sim_to_panda(self.fm.pos)

        # Sun glow follows the camera at a fixed offset toward the sun.
        cx, cy, cz = self.camera.getPos()
        self.sun_card.setPos(cx + SUN_DIR[0] * 7000.0,
                             cy + SUN_DIR[1] * 7000.0,
                             cz + SUN_DIR[2] * 7000.0)

        if mode == "chase":
            back = sim_to_panda(self.fm.pos + R @ np.array([-16.0, 0.0, 5.5]))
            self.camera.setPos(*back)
            self.camera.lookAt(*uav_p)
            self._set_feed_visible(False)
        elif mode == "fpv":
            nose = sim_to_panda(self.fm.pos + R @ np.array([2.2, 0.0, 0.35]))
            self.camera.setPos(*nose)
            self.camera.setHpr(*sim_quat_to_hpr(self.fm.quat))
            self._set_feed_visible(True)
            self.feed_label.setText("NOSE CAM")
        elif mode == "payload":
            gpos, gq = self._gimbal_world()
            self.camera.setPos(*sim_to_panda(gpos))
            self.camera.setHpr(*sim_quat_to_hpr(gq))
            self._set_feed_visible(True)
            self.feed_label.setText("PAYLOAD CAM")
            self._place_markers(gpos, gq)
        elif mode == "tower":
            self.camera.setPos(200, -240, 38)
            self.camera.lookAt(*uav_p)
            self._set_feed_visible(False)
        elif mode == "topdown":
            h, _p, _r = sim_quat_to_hpr(self.fm.quat)
            hr = math.radians(h)
            back = np.array([-math.sin(hr), math.cos(hr), 0.0])
            target = np.array(uav_p) - back * 18.0 + np.array([0, 0, 55.0])
            cur = np.array(self.camera.getPos())
            smooth = cur + (target - cur) * min(1.0, dt * 3.0)
            self.camera.setPos(*smooth)
            self.camera.lookAt(*uav_p)
            self._set_feed_visible(False)
        else:  # free: fly the camera anywhere with WASD/QE.
            # The flight keys steer the camera here (see _apply_held_keys:
            # surfaces ease to neutral, throttle/gimbal untouched).
            self._set_feed_visible(False)
            hr, pr = math.radians(self.fc_h), math.radians(self.fc_p)
            fwd = np.array([-math.sin(hr) * math.cos(pr),
                            math.cos(hr) * math.cos(pr),
                            math.sin(pr)])
            right = np.array([math.cos(hr), math.sin(hr), 0.0])
            k = self.keys
            speed = self.fc_speed * (4.0 if k.get("shift") else 1.0)
            move = (fwd * ((1.0 if (k.get("arrow_up") or k.get("w")) else 0.0) -
                           (1.0 if (k.get("arrow_down") or k.get("s")) else 0.0)) +
                    right * ((1.0 if (k.get("arrow_right") or k.get("d")) else 0.0) -
                             (1.0 if (k.get("arrow_left") or k.get("a")) else 0.0)) +
                    np.array([0.0, 0.0, 1.0]) *
                    ((1.0 if k.get("e") else 0.0) -
                     (1.0 if k.get("q") else 0.0)))
            self.camera.setPos(*(np.array(self.camera.getPos()) +
                                 move * speed * dt))
            # Right-mouse drag: free look. Driven by cursor *movement*
            # (immune to centering bias, so no drift); the cursor is hidden
            # + re-centered near the screen edges so the drag can't wander
            # onto other monitors.
            mw = self.mouseWatcherNode
            if mw is not None and mw.hasMouse():
                rmb = mw.isButtonDown(MouseButton.three())
                mx, my = mw.getMouseX(), mw.getMouseY()
                if rmb and self.win is not None:
                    if not self._fc_rmb:
                        self._fc_set_cursor_locked(True)
                        self.win.movePointer(0, self.win.getXSize() // 2,
                                             self.win.getYSize() // 2)
                        self._fc_mx, self._fc_my = 0.0, 0.0
                        self._fc_settle = 2
                    elif self._fc_settle > 0:
                        # Absorb frames while the re-center lands.
                        self._fc_settle -= 1
                        self._fc_mx, self._fc_my = mx, my
                    else:
                        self.fc_h -= (mx - self._fc_mx) * 100.0
                        self.fc_p = max(-85.0, min(85.0,
                                                   self.fc_p + (my - self._fc_my) * 80.0))
                        self._fc_mx, self._fc_my = mx, my
                        if abs(mx) > 0.9 or abs(my) > 0.9:
                            self.win.movePointer(0, self.win.getXSize() // 2,
                                                 self.win.getYSize() // 2)
                            self._fc_settle = 2
                elif self._fc_rmb:
                    self._fc_set_cursor_locked(False)
                self._fc_rmb = rmb
            self.camera.setHpr(self.fc_h, self.fc_p, 0.0)

        if mode != "payload":
            for m in self.markers:
                m.setText("")

    def _fc_set_cursor_locked(self, locked):
        """Hide/show the OS cursor (position is managed by the drag code)."""
        win = self.win
        if win is None:
            return
        # NB: win.getProperties() returns a const object - mutate a fresh one.
        props = WindowProperties()
        props.setCursorHidden(locked)
        win.requestProperties(props)

    def _fc_speed_up(self):
        if self.CAM_NAMES[self.cam_mode] == "free":
            self.fc_speed = min(500.0, self.fc_speed * 1.25)

    def _fc_speed_down(self):
        if self.CAM_NAMES[self.cam_mode] == "free":
            self.fc_speed = max(1.0, self.fc_speed / 1.25)

    def _place_markers(self, gpos, gq):
        """Project current detections onto the payload feed as [id] labels."""
        # Camera-space transform in Panda coords.
        R = quat_to_matrix(gq)
        Rp = np.array([[R[1, 1], R[1, 0], R[1, 2]],
                       [R[0, 1], R[0, 0], R[0, 2]],
                       [R[2, 1], R[2, 0], R[2, 2]]])
        cam_p = np.array(sim_to_panda(gpos))
        lens = self.camera.node().getLens()
        aspect = self.getAspectRatio()
        for i, m in enumerate(self.markers):
            if i < len(self.detections):
                d = self.detections[i]
                wp = np.array(sim_to_panda(d["pos"]))
                rel = Rp.T @ (wp - cam_p)
                p2 = lens.project(Point3(*rel))
                # Visible feed band is y in [-0.68, 0.68] (letterbox bars
                # cover the rest), so scale NDC into it.
                if p2 is not None:
                    m.setText(f"[{d['id']}]")
                    m.setPos(p2[0] * aspect, p2[1] * 0.68)
                else:
                    m.setText("")
            else:
                m.setText("")

    def _update_hud(self, dt):
        if self.camera is None:
            return
        fm, uav = self.fm, self.uav
        pitch, roll, yaw = fm.attitude_euler()
        V = fm.airspeed()
        hdg = (math.degrees(yaw) % 360.0)
        R = quat_to_matrix(fm.quat)
        vs = float((R @ fm.vel_b)[2])

        ap_txt = self.ap.status if self.ap.engaged else "MANUAL"
        gimbal_txt = (f"GMB {math.degrees(self._pan):+4.0f}/"
                      f"{math.degrees(self._tilt):+4.0f}" +
                      (" SCAN" if self.ctrl.scan_sweep else ""))
        self.hud_left.setText(
            f"SPD {V * MS_TO_KTS:5.0f} kt\n"
            f"ALT {fm.pos[2] * M_TO_FT:6.0f} ft\n"
            f"HDG {hdg:03.0f}\n"
            f"VS {vs * MS_TO_FPM:+5.0f} fpm\n"
            f"BAT {self.battery.percent:3.0f}% {self.battery.voltage:4.1f}V\n"
            f"{ap_txt}\n"
            f"DET {len(self._seen_ids)}/12  {gimbal_txt}")
        warn = []
        if fm.info().get("stalled"):
            warn.append("STALL")
        if self.battery.critical_battery:
            warn.append("CRITICAL BATTERY - AUTO-LAND")
        elif self.battery.low_battery:
            warn.append("LOW BATTERY")
        if fm.on_ground:
            warn.append("ON GROUND")
        self.hud_warn.setText("  ".join(warn))

        if self.ctrl.log_detections and self.detect_log:
            lines = [f"{d['id']:10s} {d['type']:11s} {d['range_m']:5.0f}m "
                     f"conf {d['confidence']:.2f}" for d in self.detect_log]
            self.hud_log.setText("\n".join(lines))
        else:
            self.hud_log.setText("")

        # Feed dressing follows the active feed view.
        mode = self.CAM_NAMES[self.cam_mode]
        if mode in ("fpv", "payload"):
            self._rec_t += dt
            vis = (self._rec_t % 1.0) < 0.6
            self.rec_dot.show() if vis else self.rec_dot.hide()
            self.rec_text.show() if vis else self.rec_text.hide()
            self.feed_tel.setText(
                f"HDG {hdg:03.0f}  ALT {fm.pos[2] * M_TO_FT:4.0f}FT  "
                f"SPD {V * MS_TO_KTS:3.0f}KT")

        # Attitude indicator: the ball carries the horizon; roll is
        # left-positive, so ball rotation = -roll. Nose-up pushes the
        # horizon down -> ball shifts down.
        self.ai_ball.setR(-roll * DEG)
        self.ai_ball.setZ(-pitch * 0.55)


def main():
    app = UAVSim()
    app.run()


if __name__ == "__main__":
    main()
