"""The simulator: both arms as position servos, boxes on the table, the parallel grippers, and rendered cameras.

It plays the same role as the servo buses and the camera server. robotd runs on it unchanged with --sim.
- Servos move toward their goal at a capped speed, and sag under gravity like the real ones: each settles short of
  its goal by compliance x the gravity torque on it (gravity.py), about 1 cm at the fingertips when extended.
- The table and the objects' tops are solid: an arm whose fingertips (or held object) would go into them stops there.
  Its encoders then fall behind its goal, which is how robotd feels contact on the real arms too.
- A closing gripper stops on an object that lies between its fingers, which is then held and moves with the gripper.
  An object whose jaws open falls flat on the table, unless the other gripper still holds it.
- Cameras are drawn with OpenCV from the same camera model as the real overhead camera, so pixels and table cm match
  the real setup.
- Faults on demand (faults={"miss": p, ...}), so a brain's recovery can be tried before the real arms fail:
  miss = a closing gripper misses the object (per grasp), slip = a held object falls (per second carried),
  camera = a camera drops out for CAMERA_OUTAGE_S (per second), bus = a servo read fails (per read).
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import gravity
from .scene import Camera

JOINT_SPEED = 180.0  # deg/s
GRIP_SPEED = 120.0  # mm/s
FINGER_LEN = 0.04  # m: how far the fingers reach back from the fingertips
PAD = 0.012  # m: half the finger pad's width
MIN_OVERLAP = 0.003  # m: the fingers must reach this far below an object's top to hold it
# deg per unit gravity torque (gravity.torque), per joint: the real arms sag 1-1.5 cm at the fingertips when extended
SAG_COMPLIANCE = (0.0, 1200.0, 1300.0, 900.0, 0.0)
FAULTS = ("miss", "slip", "camera", "bus")
CAMERA_OUTAGE_S = 1.5

# name, size L x W x H (m), colour (BGR), centre x, y (m), yaw (deg)
SCENES = {
    "brush": [("brush", (0.20, 0.03, 0.025), (40, 140, 230), -0.08, 0.24, 10.0)],
    "blocks": [("red block", (0.04, 0.04, 0.04), (50, 50, 210), -0.12, 0.22, 0.0),
               ("blue block", (0.04, 0.04, 0.04), (210, 120, 40), 0.10, 0.26, 30.0)],
}


@dataclass
class SimObject:
    name: str
    size: tuple
    color: tuple
    pos: np.ndarray  # centre, world m
    yaw: float  # deg
    R: np.ndarray = None  # full rotation while held
    held: dict = field(default_factory=dict)  # side -> object pose in the tool frame (4x4)

    def __post_init__(self):
        if self.R is None:
            self.R = rot_z(self.yaw)

    def T(self):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.pos
        return T

    def corners(self):
        L, W, H = self.size
        c = np.array([[sx * L / 2, sy * W / 2, sz * H / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return c @ self.R.T + self.pos

    def extent(self, d, R=None):
        """The object's width along direction d (m), at its rotation or R."""
        R = self.R if R is None else R
        return float(sum(abs(d @ R[:, i]) * s for i, s in enumerate(self.size)))


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class SimArmIO:
    """One simulated arm: 5 joints (deg) and the gripper opening (mm)."""

    def __init__(self, world, side, q0):
        self.world, self.side = world, side
        self.q = np.asarray(q0, dtype=float).copy()
        self.goal = self.q.copy()
        self.torque_on = False
        self.stalled = False  # the gripper is closed on something
        self.blocked = False  # the arm is pressing on the table or an object

    def read(self):
        if self.torque_on and self.world.fault("bus"):
            raise IOError(f"{self.side}: no status packet (simulated bus fault)")
        return self.q.copy()

    def write(self, goal):
        self.goal = np.asarray(goal, dtype=float).copy()

    def torque(self, on):
        if on:
            self.goal = self.q.copy()
        self.torque_on = bool(on)

    def temperatures(self):
        return [35.0] * 6

    def step(self, dt):
        if not self.torque_on:
            return
        arm = self.world.scene.arms[self.side]
        target = self.goal[:5] - self.world.sag * gravity.torque(arm, self.goal)
        q_next = self.q[:5] + np.clip(target - self.q[:5], -JOINT_SPEED * dt, JOINT_SPEED * dt)
        self.blocked = self.world.solid_in_the_way(self.side, self.q, q_next)
        if not self.blocked:
            self.q[:5] = q_next
        w_goal = self.goal[5]
        w = self.q[5] + np.clip(w_goal - self.q[5], -GRIP_SPEED * dt, GRIP_SPEED * dt)
        stop = self.world.grip_stop(self.side, self.q[5], w)
        self.q[5] = w if stop is None else stop
        self.stalled = stop is not None


class SimLeaderIO:
    """A leader arm for tests: wherever q is set."""

    def __init__(self, q0):
        self.q = np.asarray(q0, dtype=float).copy()

    def read(self):
        return self.q.copy()


class SimWorld:
    def __init__(self, scene, objects="brush", noise=0.0, seed=0, sag=True, faults=None):
        self.scene = scene
        self.sag = np.array(SAG_COMPLIANCE) if sag else np.zeros(5)
        self.faults = dict(faults or {})
        unknown = set(self.faults) - set(FAULTS)
        if unknown:
            raise ValueError(f"unknown faults {', '.join(sorted(unknown))}: there are {', '.join(FAULTS)}")
        self.fault_rng = np.random.default_rng(seed + 1)
        self.on_fault = lambda kind, what: None
        self.t = 0.0
        self.camera_down_until = {}
        self.rng = np.random.default_rng(seed)
        self.noise = noise  # m: error added to what perception reports
        self.objects = []
        for name, size, color, x, y, yaw in SCENES[objects] if isinstance(objects, str) else objects:
            self.add(name, size, color, x, y, yaw)
        self.io = {}
        for s in scene.arms:
            q, _ = scene.arms[s].steepest(scene.home_tip(s), lowest=0)
            self.io[s] = SimArmIO(self, s, np.append(q, 40.0))

    def add(self, name, size, color, x, y, yaw=0.0):
        self.objects.append(SimObject(name, tuple(size), tuple(color), np.array([x, y, size[2] / 2]), yaw))

    # ---- grippers -----------------------------------------------------------------------------------------------
    def tool(self, side):
        """Fingertip point, jaw direction, approach direction and the tool transform for an arm."""
        arm = self.scene.arms[side]
        p, R = arm.tool(self.io[side].q)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return p, R @ arm.jaw, -R[:, 2], T

    def _between(self, side, o, w):
        """Whether o lies between this gripper's fingers when they are w mm apart (and how far off-centre it is)."""
        p, j, a, _ = self.tool(side)
        k = np.cross(a, j)
        d = o.pos - p
        e_j, e_k = o.extent(j), o.extent(k)
        along_j, along_k = d @ j, d @ k
        top = max(c[2] for c in o.corners())
        bottom = min(c[2] for c in o.corners())
        fingers_low, fingers_high = sorted([p[2], (p - a * FINGER_LEN)[2]])
        overlap = min(fingers_high, top) - max(fingers_low, bottom)
        inside = abs(along_j) < w / 2000 and e_j <= w / 1000 + 0.002 and abs(along_k) <= e_k / 2 + PAD
        return inside and overlap >= MIN_OVERLAP and top - fingers_low >= MIN_OVERLAP, along_j, e_j

    def solid_in_the_way(self, side, q_now, q_next):
        """Whether moving this arm's joints from q_now to q_next would push its fingertips, or the object it holds,
        into the table or the top of an object."""
        arm = self.scene.arms[side]
        w = self.io[side].q[5] / 2000

        def fingers(q):
            p, R = arm.tool(q)
            j = R @ arm.jaw
            return [p + j * w, p - j * w], p, R

        tips, p, R = fingers(q_next)
        if min(t[2] for t in tips) < 0:
            return True
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        for o in self.objects:
            if side in o.held:
                rel = o.held[side]
                L, Wd, H = o.size
                c = np.array([[sx * L / 2, sy * Wd / 2, sz * H / 2, 1] for sx in (-1, 1) for sy in (-1, 1)
                              for sz in (-1, 1)])
                if np.min((T @ rel @ c.T)[2]) < 0:
                    return True
                continue
            if o.held:
                continue
            top = max(c[2] for c in o.corners())
            before = fingers(q_now)[0]
            for t, b in zip(tips, before):
                local = o.R.T @ (t - o.pos)
                inside = abs(local[0]) < o.size[0] / 2 and abs(local[1]) < o.size[1] / 2
                if inside and t[2] < top and b[2] >= top - 0.001:
                    return True
        return False

    def fault(self, kind, dt=None, what=None):
        """Whether a simulated fault of this kind happens now: per event, or per second when dt is given."""
        p = self.faults.get(kind, 0.0)
        if p <= 0:
            return False
        hit = self.fault_rng.random() < (p * dt if dt is not None else p)
        if hit:
            self.on_fault(kind, what or kind)
        return hit

    def camera_down(self, name, dt):
        """Whether a camera is in a simulated drop-out (starting one at the camera fault rate)."""
        if self.t < self.camera_down_until.get(name, -1.0):
            return True
        if self.fault("camera", dt, f"{name} dropped out for {CAMERA_OUTAGE_S:.1f} s"):
            self.camera_down_until[name] = self.t + CAMERA_OUTAGE_S
            return True
        return False

    def place(self, name, xy, yaw=None):
        """Put an object (not held) somewhere on the table: for setting up a test."""
        o = next((o for o in self.objects if o.name == name), None)
        if o is None or o.held:
            raise ValueError(f"no free object {name!r}")
        o.yaw = o.yaw if yaw is None else float(yaw)
        o.R = rot_z(o.yaw)
        o.pos = np.array([xy[0], xy[1], o.size[2] / 2])

    def grip_stop(self, side, w_now, w_next):
        """Where a moving gripper stops (mm), or None. Closing on an object grasps it; opening lets it go."""
        for o in self.objects:
            if side in o.held:
                # where the object is now, carried by these fingers (its own pose is only updated after the joints)
                _, j, _, T = self.tool(side)
                e = o.extent(j, (T @ o.held[side])[:3, :3]) * 1000
                if w_next > e + 3:
                    self.release(side, o)
                    return None
                return max(w_next, e)
        if w_next >= w_now:
            return None
        for o in self.objects:
            ok, along_j, e_j = self._between(side, o, w_now)
            if ok and w_next <= e_j * 1000 <= w_now:
                if self.fault("miss", what=f"{side} missed the {o.name}"):
                    return None
                p, j, a, T = self.tool(side)
                if not o.held:
                    o.pos = o.pos - j * along_j  # the fingers centre it as they close
                o.held[side] = np.linalg.inv(T) @ o.T()
                return e_j * 1000
        return None

    def release(self, side, o):
        del o.held[side]
        if not o.held:
            ax = o.R[:, int(np.argmax(o.size))]
            yaw = math.degrees(math.atan2(ax[1], ax[0]))
            o.R = rot_z(yaw)
            o.yaw = yaw
            o.pos = np.array([o.pos[0], o.pos[1], o.size[2] / 2])

    def step(self, dt):
        self.t += dt
        for io in self.io.values():
            io.step(dt)
        for o in self.objects:
            if len(o.held) == 1 and self.fault("slip", dt, f"the {o.name} slipped out of {next(iter(o.held))}"):
                self.release(next(iter(o.held)), o)
                continue
            if o.held:
                side = next(iter(o.held))
                T = self.tool(side)[3] @ o.held[side]
                o.R, o.pos = T[:3, :3], T[:3, 3]

    def holder(self, o):
        return next(iter(o.held), None)

    # ---- what the cameras and perception see ------------------------------------------------------------------------
    def snapshot(self):
        return dict(q={s: io.q.copy() for s, io in self.io.items()},
                    objects=[(o.name, o.size, o.color, o.pos.copy(), o.R.copy()) for o in self.objects])

    def visible_objects(self):
        """Ground truth for perception, with the configured noise: name, centre (m), size, yaw, colour."""
        out = []
        for o in self.objects:
            n = self.rng.normal(0, self.noise, 2) if self.noise else np.zeros(2)
            ax = o.R[:, int(np.argmax(o.size[:2]))]
            out.append(dict(name=o.name, pos=o.pos + np.append(n, 0), size=o.size,
                            yaw=math.degrees(math.atan2(ax[1], ax[0])) % 180, color=o.color, held=list(o.held)))
        return out


def wrist_camera(scene, side, q, width=640, height=480, hfov=100.0):
    """A camera on the gripper looking along the approach, the jaw line across the image."""
    arm = scene.arms[side]
    p, R = arm.tool(q)
    a, j = -R[:, 2], R @ arm.jaw
    origin = p - a * 0.10 + np.cross(a, j) * 0.03
    y = np.cross(a, j)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([j, y, a])
    T[:3, 3] = origin
    f = width / 2 / math.tan(math.radians(hfov) / 2)
    return Camera(width, height, [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], [0, 0, 0, 0, 0], T)


def render(scene, snap, cam, own=None):
    """Draw the table, the objects and the arms as `cam` sees them. own = the arm carrying the camera (not drawn)."""
    img = np.full((cam.height, cam.width, 3), 38, np.uint8)
    corners = np.array([[x, y, 0] for x, y in [(scene.table_x[0], scene.table_y[0]), (scene.table_x[1], scene.table_y[0]),
                                                (scene.table_x[1], scene.table_y[1]), (scene.table_x[0], scene.table_y[1])]])
    px = cam.project(corners)
    if not np.isnan(px).any():
        cv2.fillConvexPoly(img, px.astype(np.int32), (58, 56, 54))
    items = []  # (depth, draw function)
    for name, size, color, pos, R in snap["objects"]:
        L, W, H = size
        c = np.array([[sx * L / 2, sy * W / 2, sz * H / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        pts = c @ R.T + pos
        depth = float(np.mean((pts - cam.cam_to_world[:3, 3]) @ cam.cam_to_world[:3, 2]))
        items.append((depth, ("box", pts, color)))
    for s, q in snap["q"].items():
        if s == own:
            continue
        arm = scene.arms[s]
        pts = arm.points(q)
        p, R = arm.tool(q)
        a, j = -R[:, 2], R @ arm.jaw
        w = q[5] / 2000
        fingers = [(p + j * w, p + j * w - a * FINGER_LEN), (p - j * w, p - j * w - a * FINGER_LEN)]
        depth = float(np.mean((pts - cam.cam_to_world[:3, 3]) @ cam.cam_to_world[:3, 2]))
        items.append((depth, ("arm", pts, fingers)))
    for _, item in sorted(items, key=lambda it: -it[0]):
        if item[0] == "box":
            px = cam.project(item[1])
            if np.isnan(px).any():
                continue
            hull = cv2.convexHull(px.astype(np.int32))
            cv2.fillConvexPoly(img, hull, item[2])
            cv2.polylines(img, [hull], True, tuple(int(v * 0.6) for v in item[2]), 1, cv2.LINE_AA)
        else:
            _, pts, fingers = item
            px = cam.project(pts)
            for i in range(len(pts) - 1):
                if np.isnan(px[i]).any() or np.isnan(px[i + 1]).any():
                    continue
                d = float((pts[i] - cam.cam_to_world[:3, 3]) @ cam.cam_to_world[:3, 2])
                t = max(2, int(cam.K[0, 0] * 0.045 / max(d, 0.05)))
                cv2.line(img, tuple(px[i].astype(int)), tuple(px[i + 1].astype(int)), (205, 205, 200), t, cv2.LINE_AA)
            for f0, f1 in fingers:
                fp = cam.project(np.array([f0, f1]))
                if not np.isnan(fp).any():
                    cv2.line(img, tuple(fp[0].astype(int)), tuple(fp[1].astype(int)), (245, 245, 245), 3, cv2.LINE_AA)
    return img
