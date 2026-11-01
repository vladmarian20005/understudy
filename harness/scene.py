"""The calibrated setup: both arms' bases and joint offsets, the overhead camera, the gripper, the table.

The world frame is the table: x to the right in the overhead image, y up the image (away from the camera), z up from
the tabletop, in metres. scene.json holds the measured numbers.
"""
import json
import os

import cv2
import numpy as np

from . import kin

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.json")
SIDES = ("left", "right")
READY_Y, READY_Z = 0.26, 0.28  # m: the ready pose's fingertips, in front of the base and above the table: closer in or
# higher, the arm has no room to move 5 cm down or back


class Camera:
    """A pinhole camera with radial distortion. cam_to_world is the camera's pose in the world."""

    def __init__(self, width, height, K, dist, cam_to_world):
        self.width, self.height = int(width), int(height)
        self.K = np.asarray(K, dtype=float)
        self.dist = np.asarray(dist, dtype=float)
        self.cam_to_world = np.asarray(cam_to_world, dtype=float)
        self.world_to_cam = np.linalg.inv(self.cam_to_world)

    def project(self, pts):
        """World points (N, 3) to pixels (N, 2); points behind the camera come back as NaN."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        pc = pts @ self.world_to_cam[:3, :3].T + self.world_to_cam[:3, 3]
        px, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        px = px.reshape(-1, 2)
        px[pc[:, 2] <= 1e-3] = np.nan
        return px

    def to_table(self, px, z=0.0):
        """Pixels (N, 2) to world points on the plane at height z (m)."""
        px = np.atleast_2d(np.asarray(px, dtype=float))
        n = cv2.undistortPoints(px.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        rays = np.column_stack([n, np.ones(len(n))]) @ self.cam_to_world[:3, :3].T
        o = self.cam_to_world[:3, 3]
        t = (z - o[2]) / rays[:, 2]
        return o + rays * t[:, None]


class Scene:
    def __init__(self, path=SCENE):
        d = json.load(open(path))
        g = d["gripper"]
        self.stroke_mm = float(g["stroke_mm"])
        self.arms = {}
        self.ports = {}
        for s in SIDES:
            a = d["arms"][s]
            self.arms[s] = kin.Arm(s, a["base"], a["offsets_deg"], a["signs"], g["tcp"], g["jaw_deg"],
                                   kin.motor_range_deg(a["calibration"]))
            self.ports[s] = a["port"]
        self.calibration_ids = {s: d["arms"][s]["calibration"] for s in SIDES}
        self.leaders = d.get("leaders", {})  # side -> port and calibration of the leader arm that can take it over
        self.camera = Camera(**d["camera"])
        self.table_x = tuple(d["table"]["x"])
        self.table_y = tuple(d["table"]["y"])
        self._ready = {}

    def base_xy(self, side):
        return self.arms[side].base[:2, 3].copy()

    def forward(self, side):
        """The arm's forward direction on the table (its base x axis), unit 2-vector."""
        f = self.arms[side].base[:2, 0]
        return f / np.linalg.norm(f)

    def home_tip(self, side):
        """Where the fingertips park: 25 cm in front of the base, 10 cm up (at the steepest pitch that reaches)."""
        p = self.base_xy(side) + 0.25 * self.forward(side)
        return np.array([p[0], p[1], 0.10])

    def ready(self, side):
        """The joints of the pose both arms start from and park in: raised, the gripper level and pointing straight
        ahead (+y), the wrist camera on top. READY_Y in front of the base, READY_Z up; the fingertips sit off the
        wrist's axis, so their x is whatever makes the gripper point straight ahead. Mostly out of the overhead camera's
        way: it hides about 20 cm2 of the table 15-45 cm out, at its sides."""
        if side not in self._ready:
            arm = self.arms[side]
            b = self.base_xy(side)
            x, q = b[0], None
            for _ in range(5):
                q = arm.ik([x, b[1] + READY_Y, READY_Z], 0, seed=q, roll=arm.upright_roll())
                a = -arm.tool(q)[1][:, 2]
                x -= np.arctan2(a[0], a[1]) * READY_Y  # turning about the base by the heading's error
            self._ready[side] = q
        return self._ready[side].copy()

    def middle(self):
        """The point on the table halfway between the bases, 25 cm forward: where handovers meet (closer in, the taker
        can't hover above its grasp point)."""
        f = self.forward("left") + self.forward("right")
        f /= np.linalg.norm(f)
        return (self.base_xy("left") + self.base_xy("right")) / 2 + 0.25 * f

    def on_table(self, xy, margin=0.0):
        return (self.table_x[0] + margin <= xy[0] <= self.table_x[1] - margin
                and self.table_y[0] + margin <= xy[1] <= self.table_y[1] - margin)
