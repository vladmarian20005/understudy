"""The motions robotd runs: move (keyframes for one or both arms, the gripper's opening included) and home.

Each runs as one robotd command, one at a time across both arms, and returns the same result shape (Skills.run).
A failure raises Fail with one of the failure kinds; robotd reports it as the result's outcome, fail and why.
Units at this boundary: cm, degrees, mm for the gripper opening. Inside: metres.
"""
import inspect
import math
import time

import numpy as np

from .motion import SPEEDS, Goal, Phase, PlanError

FAIL_KINDS = ("slip", "unreachable", "check", "blocked", "manual", "torque_off", "busy", "timeout")
HOLD_MIN_MM = 3.0  # an opening above this after closing means something is between the fingers
HOME_SETTLE_S = 0.6


class Fail(Exception):
    def __init__(self, kind, why):
        super().__init__(why)
        self.kind, self.why = kind, why


class Stopped(Exception):
    def __init__(self, why, kind=None):
        super().__init__(why)
        self.why, self.kind = why, kind


def cm(v):
    return [round(float(x) * 100, 1) for x in v]


def obj_pose(center, z, yaw):
    T = np.eye(4)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [center[0], center[1], z]
    return T


class Skills:
    def __init__(self, core):
        self.core = core
        self.moved = False

    # ---- plumbing -------------------------------------------------------------------------------------------------
    @property
    def scene(self):
        return self.core.scene

    def q(self):
        return self.core.ctl.snapshot()["goal"]

    def measured(self, side):
        return self.core.ctl.snapshot()["q"][side]

    def width(self, side):
        return float(self.measured(side)[5])

    def tool_T(self, side, q=None):
        q = self.measured(side) if q is None else q
        p, R = self.scene.arms[side].tool(q)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def play(self, *phases):
        try:
            trajs = self.core.planner.plan(self.q(), list(phases))
        except PlanError as e:
            raise Fail(e.kind, e.why)
        ok, why = self.core.ctl.play(trajs, self.core.cancel)
        self.moved = True
        if not ok and why.startswith("contact"):
            raise Fail("check", f"unexpected {why}")
        if not ok:
            if why == "torque off":
                raise Fail("torque_off", "the arms are off")
            if why == "busy":
                raise Fail("busy", "another motion is running")
            if why.startswith(("blocked", "hot", "control error")):
                raise Stopped(why, "blocked")
            if why.startswith("manual"):
                raise Stopped(why.split(": ", 1)[1], "manual")
            if why.startswith("the plan starts"):
                raise Fail("check", why)
            raise Stopped(why)

    # ---- the motions ------------------------------------------------------------------------------------------------
    def check_holding(self, arm, kind):
        w = self.width(arm)
        if w < HOLD_MIN_MM:
            self.core.holding[arm] = None
            raise Fail(kind, f"{arm} lost the object")
        return {"width_mm": round(w, 1)}

    def move(self, left=None, right=None, speed="normal", check_only=False, avoid=None):
        """Free-form motion. For each arm a keyframe or a list of them; the k-th keyframes of both arms make one phase,
        and both finish each phase together. A keyframe (table cm, degrees, mm): xy [x, y], z, rel [dx, dy, dz] (from
        where the previous keyframe left it), yaw (the jaw line, mod 180), pitch (90 straight down, 0 level; left
        out: as steep as reaches there), width (the opening). What's left out stays as it was. roll (degrees) turns
        only the wrist from where it is, which tilts what the gripper holds about the fingers' direction; it goes in
        an arm's first keyframe, with nothing but width. check_only plans and checks without moving. avoid: standing
        things the arms keep clear of, [{"name", "xy" [x, y] cm, "r" cm, "h" cm}] (robotd doesn't see the table's
        objects as obstacles otherwise)."""
        frames = {}
        for side, kf in (("left", left), ("right", right)):
            if kf is not None:
                frames[side] = [kf] if isinstance(kf, dict) else list(kf)
        if not frames:
            raise Fail("check", "move needs keyframes for left, right or both")
        if speed not in SPEEDS:
            raise Fail("check", f"speed is one of {', '.join(SPEEDS)}")
        q0 = self.q()
        tips = {s: self.tool_T(s, q0[s])[:3, 3] for s in frames}

        def goal(side, kf, line, k):
            unknown = set(kf) - {"xy", "z", "rel", "yaw", "pitch", "width", "roll"}
            if unknown:
                raise Fail("check", f"unknown keyframe keys {sorted(unknown)}: use xy, z, rel, yaw, pitch, width, roll")
            if "roll" in kf:
                if set(kf) - {"roll", "width"} or k:
                    raise Fail("check", "roll goes in an arm's first keyframe, with nothing but width")
                q = np.array(q0[side][:5], dtype=float)
                q[4] += float(kf["roll"])
                tips[side] = self.scene.arms[side].tool(q)[0]
                return Goal(joints=q, width=kf.get("width"), line=False)
            p = tips[side].copy()
            if "xy" in kf:
                p[:2] = np.array(kf["xy"], dtype=float) / 100
            if "z" in kf:
                p[2] = float(kf["z"]) / 100
            if "rel" in kf:
                p += np.array(kf["rel"], dtype=float) / 100
            moves = bool({"xy", "z", "rel", "yaw", "pitch"} & set(kf))
            tips[side] = p
            return Goal(p=p if moves else None, yaw=kf.get("yaw"), width=kf.get("width"),
                        pitch=kf.get("pitch", "steep") if moves else None, line=line)

        def phases(line):
            for s in frames:
                tips[s] = self.tool_T(s, q0[s])[:3, 3]
            n = max(len(f) for f in frames.values())
            return [Phase(f"move {k + 1}", {s: goal(s, f[k], line, k) for s, f in frames.items() if k < len(f)},
                          speed=speed, avoid=obstacles) for k in range(n)]

        try:
            obstacles = tuple((str(o["name"]), float(o["xy"][0]) / 100, float(o["xy"][1]) / 100, float(o["r"]) / 100,
                               float(o["h"]) / 100) for o in (avoid or ()))
        except (KeyError, TypeError, ValueError, IndexError):
            raise Fail("check", 'avoid is a list of {"name", "xy": [x, y], "r", "h"} (cm)')

        try:
            try:
                trajs = self.core.planner.plan(q0, phases(True))
                line = True
            except PlanError as e:  # a straight fingertip line that leaves the reach: swing the joints instead
                if e.kind != "unreachable":
                    raise
                trajs = self.core.planner.plan(q0, phases(False))
                line = False
        except PlanError as e:
            raise Fail(e.kind, e.why)
        plan = dict(phases=len(trajs), duration_s=round(sum(t.duration for t in trajs), 2),
                    ends={s: cm(self.scene.arms[s].tool(trajs[-1].rows[s][-1])[0]) for s in frames})
        if check_only:
            return dict(checked=plan)
        self.play(*phases(line))
        out = dict(plan)
        for s in frames:
            if self.core.holding.get(s):
                out.update(self.check_holding(s, "slip"))
        return out

    def home(self, arms=("left", "right"), width=None):
        """Park the arms in the ready pose (raised, level, pointing ahead, camera on top: Scene.ready); width (mm) also
        sets the grippers' opening."""
        arms = [arms] if isinstance(arms, str) else list(arms)
        goals = {}
        for s in arms:  # the same home pose every time, wrist included: every episode starts alike
            goals[s] = Goal(joints=self.scene.ready(s), line=False, width=width)
        self.play(Phase("home", goals))
        self.wait(HOME_SETTLE_S)  # holding still is when the controller learns the arms' sag
        return {}

    def wait(self, seconds):
        """Hold still for a while, on the controller's clock (simulated time in the simulator)."""
        end = self.core.ctl.now() + seconds
        while self.core.ctl.now() < end:
            if self.core.cancel.is_set():
                raise Stopped("stop requested")
            time.sleep(0.005)

    # ---- the runner ------------------------------------------------------------------------------------------------
    def run(self, name, args):
        """Run one motion and return its result."""
        core = self.core
        t0, ev0 = core.clock(), core.ev_id
        self.moved = False
        outcome, fail, why, detail = "done", None, None, {}
        fn = getattr(self, name, None)
        try:
            if name.startswith("_") or name not in MOTIONS or fn is None:
                raise Fail("check", f"unknown command {name!r}")
            inspect.signature(fn).bind(**args)
            detail = fn(**args) or {}
        except Fail as e:
            outcome, fail, why = ("failed" if self.moved else "refused"), e.kind, e.why
        except Stopped as e:
            outcome, fail, why = "stopped", e.kind, e.why
        except TypeError as e:
            outcome, why = "refused", f"bad arguments for {name}: {e}"
        if outcome != "done":
            core.event(f"{name} {outcome}: {why}", "warn")
        return dict(op=name, outcome=outcome, fail=fail, why=why, t=[round(t0, 2), round(core.clock(), 2)],
                    arms=core.arms_state(), events=core.events_since(ev0), **detail)


MOTIONS = ("move", "home")
