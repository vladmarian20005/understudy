"""SO-101 kinematics: forward kinematics from the URDF, the parallel gripper's tool point, and IK.

Joint angles come from lerobot in degrees (its DEGREES mode: 0 at the middle of the calibrated range). The URDF angle
is sign * (degrees + offset), in radians, with the offsets and signs measured per arm (scene.json).

The tool frame is gripper_link. The fingertips are at `tcp` in that frame, the approach (wrist to fingertips) is its -z,
and the jaws close along its x axis turned by jaw_deg about z.
- pitch: how far the approach points below the arm base's horizontal, in degrees (90 = straight down). It's measured
  in the base frame because a 5-joint arm on a slightly tilted base (2-6 degrees, measured) can't point exactly at the
  world's vertical, only at its own
- yaw: the direction of the jaw line on the table, in degrees, mod 180 (0 = along the table's x)
Lengths are metres in this module.
"""
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "assets", "so101_new_calib.urdf")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CALIB_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower")
MOTOR_STEPS = 4096
LIMIT_MARGIN_DEG = 3.0
ANGLE_WEIGHT = 0.05  # m per rad: 1 degree of pitch or yaw error weighs like 0.9 mm in the IK
POS_TOL = 0.0015  # m
ANGLE_TOL = math.radians(1.5)


class Unreachable(Exception):
    pass


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def load_chain(urdf=URDF):
    """The fixed origin transform of each of the 5 joints (base_link to gripper_link) and their URDF limits (rad)."""
    root = ET.parse(urdf).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    origins, limits, link = [], [], "base_link"
    for name in JOINTS:
        j = joints[name]
        if j.find("parent").get("link") != link or j.find("axis").get("xyz").split() != ["0", "0", "1"]:
            raise ValueError(f"unexpected URDF chain at {name}")
        o = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = rot_rpy(*[float(v) for v in o.get("rpy").split()])
        T[:3, 3] = [float(v) for v in o.get("xyz").split()]
        origins.append(T)
        lim = j.find("limit")
        limits.append((float(lim.get("lower")), float(lim.get("upper"))))
        link = j.find("child").get("link")
    if link != "gripper_link":
        raise ValueError("the URDF chain doesn't end at gripper_link")
    return origins, np.array(limits)


_CHAIN = None


def chain():
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = load_chain()
    return _CHAIN


def motor_range_deg(calibration_id):
    """The calibrated motor range of the 5 arm joints in lerobot degrees, or None without a calibration file."""
    path = os.path.join(CALIB_DIR, f"{calibration_id}.json")
    if not os.path.exists(path):
        return None
    cal = json.load(open(path))
    half = [(cal[j]["range_max"] - cal[j]["range_min"]) / 2 * 360 / (MOTOR_STEPS - 1) for j in JOINTS]
    return np.array([[-h, h] for h in half])


def wrap180(a):
    """An angle in degrees wrapped to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def wrap90(a):
    """A line direction difference in degrees wrapped to [-90, 90): lines have no sign."""
    return (a + 90.0) % 180.0 - 90.0


class Arm:
    def __init__(self, side, base, offsets_deg=(0,) * 5, signs=(1,) * 5, tcp=(0, 0, -0.12), jaw_deg=90.0,
                 motor_range=None):
        self.side = side
        self.base = np.asarray(base, dtype=float)
        self.offsets = np.radians(np.asarray(offsets_deg, dtype=float))
        self.signs = np.asarray(signs, dtype=float)
        self.tcp = np.asarray(tcp, dtype=float)
        self.jaw = np.array([math.cos(math.radians(jaw_deg)), math.sin(math.radians(jaw_deg)), 0.0])
        self.origins, urdf_limits = chain()
        lim = np.sort(np.stack([self.to_deg(urdf_limits[:, 0]), self.to_deg(urdf_limits[:, 1])], axis=1), axis=1)
        if motor_range is not None:
            lim = np.stack([np.maximum(lim[:, 0], motor_range[:, 0]), np.minimum(lim[:, 1], motor_range[:, 1])], 1)
        self.limits = lim + np.array([LIMIT_MARGIN_DEG, -LIMIT_MARGIN_DEG])

    # ---- angles --------------------------------------------------------------------------------------------------
    def to_urdf(self, q_deg):
        return self.signs * (np.radians(np.asarray(q_deg, dtype=float)[:5]) + self.offsets)

    def to_deg(self, q_urdf):
        return np.degrees(np.asarray(q_urdf, dtype=float) * self.signs - self.offsets)

    def canonical_roll(self, q_deg, toward=(0.0, 1.0, 0.0)):
        """The same grasp with the wrist rolled so the jaw axis points along `toward` (world; default +y, away from
        the bases), where the joint allows. Half a turn of the roll leaves the fingertip point and the jaw line where
        they are, so for the symmetric parallel gripper both rolls are the same grasp; choosing by the jaw's direction
        keeps the demonstrations consistent (a policy averaging two rolls 180 deg apart lands on neither) and each
        wrist camera on the same side. Grasps here are across a long object lying mostly along x, so the jaw axis is
        mostly along y and the choice is clear."""
        q = np.asarray(q_deg, dtype=float).copy()
        j = self.tool(q)[1] @ self.jaw
        if j @ np.asarray(toward, dtype=float) >= 0:
            return q
        for turn in sorted((-180.0, 180.0), key=lambda d: abs(q[4] + d)):  # only the roll changes
            if self.limits[4, 0] <= q[4] + turn <= self.limits[4, 1]:
                alt = q.copy()
                alt[4] += turn
                return alt
        return q

    def within_limits(self, q_deg, slack=0.0):
        q = np.asarray(q_deg, dtype=float)[:5]
        return bool(np.all(q >= self.limits[:, 0] - slack) and np.all(q <= self.limits[:, 1] + slack))

    # ---- forward kinematics ----------------------------------------------------------------------------------------
    def _frames_urdf(self, qu):
        T, out = self.base, []
        for Tj, a in zip(self.origins, qu):
            c, s = math.cos(a), math.sin(a)
            R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
            T = T @ Tj @ R
            out.append(T)
        return out

    def _tool_urdf(self, qu):
        T = self._frames_urdf(qu)[-1]
        return T[:3, :3] @ self.tcp + T[:3, 3], T[:3, :3]

    def frames(self, q_deg):
        """World transforms of the 5 joint frames, each after its joint's rotation (the last is gripper_link)."""
        return self._frames_urdf(self.to_urdf(q_deg))

    def tool(self, q_deg):
        """Fingertip position (world, m) and gripper_link's rotation."""
        return self._tool_urdf(self.to_urdf(q_deg))

    def pose(self, q_deg):
        """Fingertip position (m), pitch and yaw (degrees)."""
        p, R = self.tool(q_deg)
        return p, *self.angles(R)

    def angles(self, R):
        a = self.base[:3, :3].T @ -R[:, 2]
        pitch = math.degrees(math.atan2(-a[2], math.hypot(a[0], a[1])))
        j = R @ self.jaw
        yaw = math.degrees(math.atan2(j[1], j[0])) % 180.0
        return pitch, yaw

    def points(self, q_deg):
        """Points along the arm (world, m): shoulder, upper arm, elbow, wrist, wrist roll, fingertips. Consecutive
        points make the segments used for the table and arm-to-arm checks."""
        F = self.frames(q_deg)
        p_tip = F[-1][:3, :3] @ self.tcp + F[-1][:3, 3]
        return np.array([F[0][:3, 3], F[1][:3, 3], F[2][:3, 3], F[3][:3, 3], F[4][:3, 3], p_tip])

    # ---- inverse kinematics ----------------------------------------------------------------------------------------
    def _residual(self, qu, p, pitch, yaw, perp=None):
        pt, R = self._tool_urdf(qu)
        r = list(pt - p)
        if pitch is not None:
            a = self.base[:3, :3].T @ -R[:, 2]
            h = math.hypot(a[0], a[1])
            u = a[:2] / h if h > 1e-6 else np.zeros(2)
            want = np.array([u[0] * math.cos(pitch), u[1] * math.cos(pitch), -math.sin(pitch)])
            r += list(ANGLE_WEIGHT * (a - want))
        if perp is not None:
            r.append(ANGLE_WEIGHT * float((R @ self.jaw) @ perp))
        elif yaw is not None:
            j = R @ self.jaw
            d = math.radians(wrap90(math.degrees(math.atan2(j[1], j[0]) - yaw)))
            r.append(ANGLE_WEIGHT * d)
        return np.array(r)

    def _errors(self, qu, p, pitch, yaw, perp=None):
        pt, R = self._tool_urdf(qu)
        pe = float(np.linalg.norm(pt - p))
        cur_pitch, cur_yaw = self.angles(R)
        ae = 0.0
        if pitch is not None:
            ae = max(ae, abs(math.radians(cur_pitch) - pitch))
        if perp is not None:
            ae = max(ae, abs(math.asin(min(1.0, abs(float((R @ self.jaw) @ perp))))))
        elif yaw is not None:
            ae = max(ae, abs(math.radians(wrap90(cur_yaw - math.degrees(yaw)))))
        return pe, ae

    def _solve(self, q0_deg, p, pitch, yaw, iters, perp=None):
        lo, hi = self.to_urdf(self.limits[:, 0]), self.to_urdf(self.limits[:, 1])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        qu = np.clip(self.to_urdf(q0_deg), lo, hi)
        eps, lam = 1e-5, 0.004
        for _ in range(iters):
            r = self._residual(qu, p, pitch, yaw, perp)
            pe, ae = self._errors(qu, p, pitch, yaw, perp)
            if pe < POS_TOL and ae < ANGLE_TOL:
                return qu, pe, ae
            J = np.empty((len(r), 5))
            for i in range(5):
                d = np.zeros(5)
                d[i] = eps
                J[:, i] = (self._residual(qu + d, p, pitch, yaw, perp) - r) / eps
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(r)), -r)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            qu = np.clip(qu + dq, lo, hi)
        pe, ae = self._errors(qu, p, pitch, yaw, perp)
        return qu, pe, ae

    def steepest(self, p, yaw=None, seed=None, lowest=30, step=5):
        """The steepest pitch (90 down to `lowest`) that reaches p: (joint degrees, pitch). Straight down only reaches
        low points 15-25 cm in front of the base; higher or farther needs a tilted approach."""
        err = None
        for pitch in range(90, lowest - 1, -step):
            try:
                return self.ik(p, pitch, yaw, seed, restarts=1 if pitch > lowest else 4), float(pitch)
            except Unreachable as e:
                err = e
        raise err

    def ik(self, p, pitch=None, yaw=None, seed=None, iters=80, restarts=4, perp=None):
        """Joint degrees (5) that put the fingertips at p (world, m) with the given pitch and yaw (degrees, either can
        be None = free). perp, a direction, replaces yaw: the jaw line is set square to it (to close across a tilted
        object). Tries the seed first, then a few standard postures, then random ones. Raises Unreachable."""
        p = np.asarray(p, dtype=float)
        pr = None if pitch is None else math.radians(pitch)
        yr = None if yaw is None else math.radians(yaw)
        # the pan that faces the target, as a fallback seed
        local = np.linalg.inv(self.base) @ np.append(p, 1.0)
        pan = self.to_deg([math.atan2(local[1], local[0]), 0, 0, 0, 0])[0]
        seeds = [] if seed is None else [np.asarray(seed, dtype=float)[:5]]
        seeds += [np.array([pan, 0, 0, 60, 0]), np.array([pan, -30, 30, 80, 0]), np.array([pan, 20, -40, 60, 90]),
                  np.array([pan, 0, 0, 60, -90])]
        rng = np.random.default_rng(int(abs(p[0] * 1e4) + abs(p[1] * 1e6)) % 2 ** 31)
        seeds += [rng.uniform(self.limits[:, 0], self.limits[:, 1]) for _ in range(restarts)]
        best = None
        for s in seeds:
            qu, pe, ae = self._solve(s, p, pr, yr, iters if best is None else iters // 2,
                                     None if perp is None else np.asarray(perp, dtype=float))
            if best is None or pe + ANGLE_WEIGHT * ae < best[1] + ANGLE_WEIGHT * best[2]:
                best = (qu, pe, ae)
            if pe < POS_TOL and ae < ANGLE_TOL:
                return self.to_deg(qu)
        raise Unreachable(f"{self.side} can't reach ({p[0] * 100:.1f}, {p[1] * 100:.1f}, {p[2] * 100:.1f}) cm"
                          + (f" at pitch {pitch:.0f}" if pitch is not None else "")
                          + f": {best[1] * 1000:.0f} mm off, {math.degrees(best[2]):.0f} deg off")
