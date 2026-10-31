"""Motion plans: phases of fingertip goals for one or both arms, turned into checked 50 Hz joint trajectories.

A phase moves the arms named in it at the same time and ends when the slower one arrives. Each arm's goal is a
`Goal`: a fingertip point, pitch and yaw (None keeps the current value), a gripper opening, or joint angles.
Fingertips move in straight lines by default. Every tick of every phase is checked before anything moves: inverse
kinematics, joint limits, the table, the workspace, the gap between the two arms, and that an object an arm holds
(planner.holding, robotd's record of what each gripper holds) stays clear of the table.

A trajectory row is the 5 joint angles (lerobot degrees) and the gripper opening (mm).
"""
import math
from dataclasses import dataclass

import numpy as np

from . import kin

TICK = 0.02
SPEEDS = {"slow": (0.04, 40.0), "normal": (0.10, 80.0), "fast": (0.18, 140.0)}  # fingertip m/s, joint and pitch deg/s
GRIP_SPEED = 100.0  # mm/s
MIN_PHASE = 0.3  # s
MAX_JOINT_STEP = 6.0  # deg per tick: more means the IK jumped to another posture
TIP_FLOOR = 0.003  # m: the fingertips never go lower
ARM_FLOOR = 0.02  # m: the rest of the arm stays this high
ARM_GAP = 0.015  # m between the two arms' outlines
RADII = (0.025, 0.025, 0.025, 0.03, 0.035)  # m: shoulder-upper arm, upper arm-elbow, forearm, wrist, gripper
CARRY_FLOOR = 0.005  # m: a held object's lowest corner stays this high, except in a phase that lowers the floor
# m: one that starts lower (just picked up, or being set down) may go this much lower still: lifted by one end, a long
# object's other end pivots on the table, and the plan's few mm into it are the table pushing it up
PIVOT_SLACK = 0.003
Z_MAX = 0.40


class PlanError(Exception):
    def __init__(self, kind, why, phase=None):
        super().__init__(why)
        self.kind, self.why, self.phase = kind, why, phase


@dataclass
class Goal:
    p: object = None  # fingertip, world m (3,)
    pitch: object = None  # deg; None keeps the current pitch; "steep" = the steepest that reaches the goal
    yaw: float = None  # deg mod 180; None keeps the current yaw
    width: float = None  # gripper opening, mm; None keeps it
    joints: object = None  # 5 joint degrees, instead of p/pitch/yaw
    line: bool = True  # straight fingertip line; False interpolates the joints
    floor: float = TIP_FLOOR  # m: the lowest the fingertips may go in this phase


@dataclass
class Phase:
    name: str
    goals: dict  # side -> Goal
    speed: str = "normal"
    gap: float = ARM_GAP
    stop_on_contact: bool = False  # the controller ends the phase when a moving arm's fingertips are held up


@dataclass
class Trajectory:
    name: str
    rows: dict  # side -> (N, 6) array; both have the same N
    contact: bool = False  # stop on contact (Phase.stop_on_contact)

    @property
    def duration(self):
        return len(next(iter(self.rows.values()))) * TICK


def smooth(s):
    """Minimum-jerk timing: 0 to 1 with zero speed and acceleration at both ends."""
    return s * s * s * (10 - 15 * s + 6 * s * s)


def seg_dist(p1, q1, p2, q2):
    """The shortest distance between segments p1-q1 and p2-q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def arms_gap(pa, pb):
    """The gap between two arms' outlines (capsules around consecutive points), m."""
    best = math.inf
    for i in range(len(pa) - 1):
        for j in range(len(pb) - 1):
            best = min(best, seg_dist(pa[i], pa[i + 1], pb[j], pb[j + 1]) - RADII[i] - RADII[j])
    return best


def held_corners(T_tool, held):
    """The 8 corners (world, m) of an object held as `held` (robotd's record: size, rel) by a gripper at T_tool."""
    L, W, H = held["size"]
    c = np.array([[sx * L / 2, sy * W / 2, sz * H / 2, 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (T_tool @ held["rel"] @ c.T).T[:, :3]


class Planner:
    def __init__(self, scene, holding=None):
        self.scene = scene
        self.holding = holding if holding is not None else {}

    def _endpoint(self, arm, q0, goal):
        """The goal's pose with the current values filled in: (p, pitch, yaw, width, q_end)."""
        p0, pitch0, yaw0 = arm.pose(q0[:5])
        if goal.joints is not None:
            q_end = np.asarray(goal.joints, dtype=float)
            p1, pitch1, yaw1 = arm.pose(q_end)
        else:
            p1 = p0 if goal.p is None else np.asarray(goal.p, dtype=float)
            yaw1 = yaw0 if goal.yaw is None else float(goal.yaw) % 180
            if goal.pitch == "steep":
                q_end, pitch1 = arm.steepest(p1, yaw1, seed=q0[:5])
            else:
                pitch1 = pitch0 if goal.pitch is None else float(goal.pitch)
                q_end = arm.ik(p1, pitch1, yaw1, seed=q0[:5])
        width1 = q0[5] if goal.width is None else float(np.clip(goal.width, 0, self.scene.stroke_mm))
        return (p0, pitch0, yaw0), (p1, pitch1, yaw1), width1, q_end

    def _arm_rows(self, side, q0, goal, speed, duration=None):
        arm = self.scene.arms[side]
        v, w = SPEEDS[speed]
        (p0, pitch0, yaw0), (p1, pitch1, yaw1), width1, q_end = self._endpoint(arm, q0, goal)
        dyaw = kin.wrap90(yaw1 - yaw0)
        need = max(MIN_PHASE, abs(width1 - q0[5]) / GRIP_SPEED, float(np.max(np.abs(q_end - q0[:5]))) / w)
        if goal.line and goal.joints is None:
            need = max(need, float(np.linalg.norm(p1 - p0)) / v, abs(pitch1 - pitch0) / w, abs(dyaw) / w)
        if duration is None:
            return need
        n = max(1, int(math.ceil(duration / TICK)))
        rows = np.empty((n, 6))
        q = q0[:5].copy()
        for k in range(n):
            s = smooth((k + 1) / n)
            if goal.line and goal.joints is None:
                p = p0 + (p1 - p0) * s
                q_new = arm.ik(p, pitch0 + (pitch1 - pitch0) * s, (yaw0 + dyaw * s) % 180, seed=q, iters=40)
                if np.max(np.abs(q_new - q)) > MAX_JOINT_STEP:
                    raise kin.Unreachable(f"{side} would swing to another posture near "
                                          f"({p[0] * 100:.1f}, {p[1] * 100:.1f}, {p[2] * 100:.1f}) cm")
                q = q_new
            else:
                q = q0[:5] + (q_end - q0[:5]) * s
            rows[k, :5] = q
            rows[k, 5] = q0[5] + (width1 - q0[5]) * s
        return rows

    def plan(self, q_now, phases):
        """Trajectories for the phases, in order, from the current joints {side: (6,)}. Raises PlanError."""
        q = {s: np.asarray(v, dtype=float).copy() for s, v in q_now.items()}
        out = []
        for ph in phases:
            try:
                need = max(self._arm_rows(s, q[s], g, ph.speed) for s, g in ph.goals.items())
                rows = {s: self._arm_rows(s, q[s], g, ph.speed, need) for s, g in ph.goals.items()}
            except kin.Unreachable as e:
                raise PlanError("unreachable", f"{ph.name}: {e}", ph.name)
            n = len(next(iter(rows.values())))
            for s in q:
                if s not in rows:
                    rows[s] = np.tile(q[s], (n, 1))
            traj = Trajectory(ph.name, rows, ph.stop_on_contact)
            self.check(traj, ph)
            out.append(traj)
            q = {s: r[-1].copy() for s, r in rows.items()}
        return out

    def check(self, traj, phase):
        sides = list(traj.rows)
        n = len(traj.rows[sides[0]])
        # an arm may start past a joint limit or low (folded up, or resting on the table after torque off): it can
        # move back in, but no further out than it started
        start = {s: np.asarray(traj.rows[s][0], dtype=float)[:5] for s in sides}
        lo = {s: np.minimum(self.scene.arms[s].limits[:, 0], start[s]) - 0.5 for s in sides}
        hi = {s: np.maximum(self.scene.arms[s].limits[:, 1], start[s]) + 0.5 for s in sides}
        pts0 = {s: self.scene.arms[s].points(traj.rows[s][0]) for s in sides}
        floors = {s: min(phase.goals[s].floor if s in phase.goals else TIP_FLOOR, pts0[s][-1][2]) for s in sides}
        arm_floors = {s: min(ARM_FLOOR, float(np.min(pts0[s][1:-1, 2]))) for s in sides}
        carry_floor = {}  # a held object may start low (just picked up, or set down) but not go lower
        for k in list(range(0, n, 2)) + [n - 1]:
            pts = {}
            for s in sides:
                arm = self.scene.arms[s]
                q = traj.rows[s][k]
                q5 = np.asarray(q, dtype=float)[:5]
                if np.any(q5 < lo[s]) or np.any(q5 > hi[s]):
                    raise PlanError("check", f"{phase.name}: {s} would pass a joint limit", phase.name)
                pts[s] = arm.points(q)
                tip = pts[s][-1]
                if tip[2] < floors[s] - 1e-4:
                    raise PlanError("check", f"{phase.name}: {s} fingertips would go {tip[2] * 100:.1f} cm from "
                                    f"the table", phase.name)
                if np.min(pts[s][1:-1, 2]) < arm_floors[s]:
                    raise PlanError("check", f"{phase.name}: {s} arm would come within 2 cm of the table", phase.name)
                if not self.scene.on_table(tip[:2]) or tip[2] > Z_MAX:
                    raise PlanError("check", f"{phase.name}: {s} fingertips would leave the workspace at "
                                    f"({tip[0] * 100:.0f}, {tip[1] * 100:.0f}, {tip[2] * 100:.0f}) cm", phase.name)
                held = self.holding.get(s)
                if held is not None:
                    p_t, R_t = arm.tool(q)
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = R_t, p_t
                    low = float(np.min(held_corners(T, held)[:, 2]))
                    # a touch-down lowers it until it rests on something, aiming below the table to find it
                    floor = carry_floor.setdefault(s, -0.04 if phase.stop_on_contact
                                                   else min(CARRY_FLOOR, floors[s], low - PIVOT_SLACK))
                    if low < floor - 1e-4:
                        raise PlanError("check", f"{phase.name}: the object {s} holds would hit the table "
                                        f"({low * 100:.1f} cm)", phase.name)
            if len(sides) == 2:
                gap = arms_gap(pts[sides[0]], pts[sides[1]])
                if gap < phase.gap:
                    raise PlanError("check", f"{phase.name}: the arms would come within {max(gap, 0) * 100:.1f} cm "
                                    f"of each other (keep {phase.gap * 100:.1f} cm) at t={k * TICK:.2f} s",
                                    phase.name)
