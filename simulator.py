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
    NodePath,
    PNMImage,
    Point3,
    TextNode,
    Texture,
    Vec4,
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
  Ctrl ....... throttle down            C ......... camera: chase/FPV/payload/tower
  T / G ...... trim nose-up / nose-down P ......... pause
  B .......... wheel brakes             R ......... reset flight
  H .......... toggle this help         Esc ....... quit

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

    CAM_NAMES = ("chase", "fpv", "payload", "tower")

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

        self._build_world()
        self._build_uav()
        self._build_targets()
        self._build_hud()
        self._build_feed_overlay()
        self._bind_keys()

        self.taskMgr.add(self._update, "sim-update")

    # -- world ------------------------------------------------------------
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

        def to_px(x):
            return int((x + world / 2) / world * (size - 1))

        # Runway along x: [-600, 600] x [-10, 10].
        rx0, rx1 = to_px(-600), to_px(600)
        ry0, ry1 = to_px(-10), to_px(10)
        for j in range(ry0, ry1 + 1):
            for i in range(rx0, rx1 + 1):
                img.setXel(i, j, 0.26, 0.26, 0.28)
        for j in (ry0 + 1, ry1 - 1):
            for i in range(rx0, rx1 + 1):
                img.setXel(i, j, 0.9, 0.9, 0.9)
        yc = to_px(0)
        x = -585.0
        while x < 585.0:
            for k in range(int(30 / world * size)):
                i = to_px(x) + k
                if rx0 <= i <= rx1:
                    for dj in (-1, 0, 1):
                        img.setXel(i, yc + dj, 0.92, 0.92, 0.92)
            x += 60.0
        # Target-field outline (the scan area), drawn faintly.
        for cx, cy in [((400, 800), (100, 550))]:
            (x0, x1), (y0, y1) = cx, cy
            for i in range(to_px(x0), to_px(x1) + 1):
                for j in (to_px(y0), to_px(y1)):
                    img.setXel(i, j, 0.95, 0.55, 0.15)
            for j in range(to_px(y0), to_px(y1) + 1):
                for i in (to_px(x0), to_px(x1)):
                    img.setXel(i, j, 0.95, 0.55, 0.15)
        tex = Texture("ground")
        tex.load(img)
        return tex

    def _build_world(self):
        cm = CardMaker("ground")
        cm.setFrame(-2000, 2000, -2000, 2000)
        ground = self.render.attachNewNode(cm.generate())
        ground.setTexture(self._make_ground_texture())
        ground.setPos(0, 0, -0.05)

        sky_img = PNMImage(4, 256)
        for j in range(256):
            t = j / 255.0
            r = 0.78 + (0.20 - 0.78) * t
            g = 0.84 + (0.45 - 0.84) * t
            b = 0.90 + (0.95 - 0.90) * t
            for i in range(4):
                sky_img.setXel(i, j, r, g, b)
        sky_tex = Texture("sky")
        sky_tex.load(sky_img)
        sky = self.render.attachNewNode(make_box_node("sky", 9000, 9000, 9000, (1, 1, 1)))
        sky.setTexture(sky_tex)
        sky.setTwoSided(True)
        sky.setFogOff()
        sky.setDepthWrite(False)
        sky.setBin("background", -10)

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.0, 0.96, 0.88, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(35, -50, 0)
        self.render.setLight(sun_np)
        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.55, 0.60, 0.68, 1.0))
        self.render.setLight(self.render.attachNewNode(amb))

        fog = Fog("haze")
        fog.setColor(0.78, 0.83, 0.89)
        fog.setExpDensity(0.00045)
        self.render.setFog(fog)

    # -- UAV model ----------------------------------------------------------
    def _build_uav(self):
        """Predator-style fixed-wing UAV from boxes (Panda axes in parts)."""
        gray = (0.72, 0.74, 0.78)
        dark = (0.16, 0.17, 0.20)
        glass = (0.12, 0.18, 0.26)

        self.uav_node = self.render.attachNewNode("uav")
        # (name, ex right, ey fwd, ez up, color, pos)
        parts = [
            ("fuselage", 0.55, 3.0, 0.55, gray, (0, 0.0, 0.0)),
            ("nose", 0.42, 0.7, 0.42, gray, (0, 1.75, -0.05)),
            ("sensor_nose", 0.30, 0.25, 0.22, glass, (0, 2.05, -0.12)),
            ("wing", 4.2, 0.70, 0.08, gray, (0, 0.35, 0.32)),
            ("tailboom", 0.18, 1.4, 0.18, gray, (0, -1.9, 0.05)),
            ("fin_l", 1.05, 0.45, 0.07, gray, (0, 0, 0)),   # placed below
            ("fin_r", 1.05, 0.45, 0.07, gray, (0, 0, 0)),
            ("gimbal_ball", 0.30, 0.30, 0.30, dark, (0, 1.15, -0.38)),
            ("gear_l", 0.12, 0.12, 0.45, dark, (-0.35, 0.5, -0.45)),
            ("gear_r", 0.12, 0.12, 0.45, dark, (0.35, 0.5, -0.45)),
            ("gear_n", 0.12, 0.12, 0.45, dark, (0, 1.5, -0.45)),
        ]
        for name, ex, ey, ez, color, pos in parts:
            if name in ("fin_l", "fin_r"):
                continue  # handled below (angled V-tail)
            node = NodePath(make_box_node(name, ex, ey, ez, color))
            node.setPos(*pos)
            node.setTwoSided(True)
            node.reparentTo(self.uav_node)
        # V-tail: two panels angled like an inverted/regular V.
        for name, roll_deg in (("fin_l", 35), ("fin_r", -35)):
            node = NodePath(make_box_node(name, 1.05, 0.45, 0.07, gray))
            node.setPos(0, -2.55, 0.30)
            node.setR(roll_deg)
            node.setTwoSided(True)
            node.reparentTo(self.uav_node)
        # Pusher prop at the tail, spinning about Panda y (forward axis).
        self.prop = self.uav_node.attachNewNode("prop")
        for nm, ex, ez in (("pb1", 0.09, 1.5), ("pb2", 1.5, 0.09)):
            blade = NodePath(make_box_node(nm, ex, 0.06, ez, dark))
            blade.setTwoSided(True)
            blade.reparentTo(self.prop)
        self.prop.setPos(0, -1.45, 0.12)

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
        self.hud_right = self._text((1.72, 0.88), 0.055, TextNode.ARight)
        self.hud_warn = OnscreenText(text="", pos=(0, 0.80), scale=0.075,
                                    align=TextNode.ACenter, fg=(1, 0.15, 0.15, 1),
                                    mayChange=True)
        self.hud_ap = OnscreenText(text="", pos=(0, 0.70), scale=0.05,
                                  align=TextNode.ACenter, fg=(0.4, 1.0, 0.4, 1),
                                  mayChange=True)
        self.hud_log = self._text((-1.72, 0.45), 0.042, fg=(1.0, 0.95, 0.6, 1))
        self.help_text = OnscreenText(text=HELP_TEXT, pos=(-1.70, 0.55),
                                     scale=0.040, align=TextNode.ALeft,
                                     fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.55),
                                     mayChange=False)

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
        # Held keys tracked in self.keys.
        for k in ("w", "s", "a", "d", "arrow_up", "arrow_down", "arrow_left",
                  "arrow_right", "q", "e", "shift", "control",
                  "z", "x", "n", "m"):
            a(k, self._key_down, [k])
            a(k + "-up", self._key_up, [k])
        # Toggles.
        a("v", self._toggle_scan)
        a("l", self._toggle_log)
        a("c", self._cycle_camera)
        a("p", self._toggle_pause)
        a("r", self._reset)
        a("h", self._toggle_help)
        a("b", self._toggle_brakes)
        a("t", lambda: self.ctrl.trim_up())
        a("g", lambda: self.ctrl.trim_down())
        a("u", self._toggle_autopilot)    # autopilot engage/disengage
        a("escape", self._quit)

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
        self._move_camera()
        self._update_hud(dt)
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

    def _move_camera(self):
        if self.camera is None:
            return
        mode = self.CAM_NAMES[self.cam_mode]
        R = quat_to_matrix(self.fm.quat)
        uav_p = sim_to_panda(self.fm.pos)

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
        else:  # tower
            self.camera.setPos(60, -90, 14)
            self.camera.lookAt(*uav_p)
            self._set_feed_visible(False)

        if mode != "payload":
            for m in self.markers:
                m.setText("")

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

        self.hud_left.setText(
            f"SPD {V * MS_TO_KTS:3.0f} KT   ALT {fm.pos[2] * M_TO_FT:4.0f} FT   "
            f"HDG {hdg:03.0f}   VS {vs * MS_TO_FPM:+5.0f} FPM")
        ap_txt = self.ap.status if self.ap.engaged else "MANUAL"
        self.hud_right.setText(
            f"BAT {self.battery.percent:3.0f}% {self.battery.voltage:4.1f}V   "
            f"{ap_txt}   DET {len(self._seen_ids)}/12")
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
        self.hud_ap.setText(
            f"GMB pan {math.degrees(self._pan):+4.0f}  tilt "
            f"{math.degrees(self._tilt):+4.0f}" +
            ("   SCAN" if self.ctrl.scan_sweep else ""))

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


def main():
    app = UAVSim()
    app.run()


if __name__ == "__main__":
    main()
