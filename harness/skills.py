"""The motion primitives the agents call: reach, grasp, move_to, release, handover and home, plus pick and place.

Each runs as one robotd command, one at a time across both arms, and returns the same result shape (Skills.run).
A failure raises Fail with one of the failure kinds; robotd reports it as the result's outcome, fail and why.
Units at this boundary: cm, degrees, mm for the gripper opening. Inside: metres.
"""
import inspect
import math

import numpy as np

from . import kin, perception
from .motion import Goal, Phase, PlanError

FAIL_KINDS = ("miss", "finger_on_object", "no_contact", "slip", "dropped", "unreachable", "check", "blocked",
              "manual", "torque_off", "busy", "timeout", "not_seen")
TRAVEL_Z = 0.10
HOLD_MIN_MM = 3.0  # an opening above this after closing means something is between the fingers
SQUEEZE_MM = 6.0  # after a grasp the goal is this far inside the object, not fully closed, so the servo doesn't cook
END_MARGIN = 0.04  # long objects are held this far in from their end
MEET_Z = 0.10  # m: where mid-air passes happen; lower and the arms crowd each other, higher and they can't back out
PULL_OUT = 0.03  # m: after letting go, the fingers slide back off the object along the approach
TAKE_HOVER = 0.05  # m: the taker waits this far above its grasp point, then closes in
LONG = 2.5  # length / width ratio that makes an object "long": held at an end instead of the middle


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
        self.reached = {}  # side -> where reach left the arm ready to grasp
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
        if not ok:
            if why == "torque off":
                raise Fail("torque_off", "the arms are off")
            if why == "busy":
                raise Fail("busy", "another motion is running")
            if why.startswith(("blocked", "hot", "control error")):
                raise Stopped(why, "blocked")
            if why.startswith("the plan starts"):
                raise Fail("check", why)
            raise Stopped(why)

    def detect(self):
        dets = self.core.perception.detect()
        self.core.last_objects = dets
        return dets

    def resolve(self, target):
        d = perception.find(self.detect(), target)
        if d is None:
            raise Fail("not_seen", f"no single object matches {target!r}")
        return d

    def hold(self, side):
        h = self.core.holding.get(side)
        if h is None:
            raise Fail("check", f"{side} isn't holding anything")
        return h

    def other(self, side):
        return "right" if side == "left" else "left"

    # ---- grasp geometry -------------------------------------------------------------------------------------------
    def grasp_spots(self, side, d):
        """Where to pinch a detected object, best first: (point m, jaw yaw, opening mm, tip height m). A long object
        is held near an end, the one closer to this arm first, so the rest stays free for a pass; its middle is the
        last resort (an end can sit too close to the base to reach)."""
        L, W, H = (v / 100 for v in d["size_cm"])
        c = np.array(d["center_cm"]) / 100
        u = np.array([math.cos(math.radians(d["yaw"])), math.sin(math.radians(d["yaw"]))])
        points = [c]
        if L >= LONG * W:
            s = L / 2 - min(END_MARGIN, L / 4)
            ends = sorted([c + u * s, c - u * s], key=lambda e: np.linalg.norm(e - self.scene.base_xy(side)))
            points = ends + [c]
        z = float(np.clip(H * 0.4, 0.008, 0.03))
        return [(p, (d["yaw"] + 90) % 180, min(self.scene.stroke_mm, W * 1000 + 25), z) for p in points]

    def approach(self, side, point, yaw, hover_z, perp=None):
        """The steepest pitch at which both the grasp point and the point hover_z above it (back along the approach)
        are reachable, with the jaw line at yaw or square to perp. Returns (pitch, hover point, yaw, hover joints)."""
        arm = self.scene.arms[side]
        seed = self.q()[side][:5]
        for pitch in range(90, 29, -5):
            try:
                q = arm.ik(point, pitch, yaw, seed=seed, restarts=1, perp=perp)
                a = -arm.tool(q)[1][:, 2]
                yaw_q = arm.pose(q)[2]
                hover = point - a * (hover_z - point[2]) / max(-a[2], 0.3)
                q_hover = arm.ik(hover, pitch, yaw_q, seed=q, restarts=1)
                return float(pitch), hover, yaw_q, q_hover
            except kin.Unreachable:
                continue
        raise Fail("unreachable", f"{side} can't get above ({point[0] * 100:.1f}, {point[1] * 100:.1f}) cm")

    # ---- primitives ------------------------------------------------------------------------------------------------
    def reach(self, arm, target=None, xy=None, z=10.0, yaw=None, width=None, offset=None):
        if target is not None:
            d = self.resolve(target)
            if d["held_by"]:
                raise Fail("check", f"{d['id']} is held by {d['held_by']}")
            spots = self.grasp_spots(arm, d)
        elif xy is not None:
            d = None
            spots = [(np.array(xy, dtype=float) / 100, self.scene.arms[arm].pose(self.q()[arm])[2], None, 0.015)]
        else:
            raise Fail("check", "reach needs a target or xy")
        err, want_yaw = None, yaw
        for point, auto_yaw, opening, gz in spots:
            if offset:
                point = point + np.array(offset, dtype=float) / 100
            g = np.array([point[0], point[1], gz])
            yaw = auto_yaw if want_yaw is None else want_yaw
            try:
                pitch, hover, _, q_hover = self.approach(arm, g, yaw, z / 100)
                break
            except Fail as e:
                err = e
        else:
            raise err
        width = opening if width is None else width
        self.play(Phase("reach", {arm: Goal(joints=q_hover, line=False,
                                            width=self.scene.stroke_mm if width is None else width)}))
        self.reached[arm] = dict(obj=d, point=g, hover=hover, pitch=pitch, yaw=yaw)
        return {}

    def grasp(self, arm, width=None, offset=None):
        r = self.reached.pop(arm, None)
        if r is None:
            raise Fail("check", f"{arm} has to reach an object before grasping")
        g, hover = r["point"].copy(), r["hover"].copy()
        if offset:
            shift = np.append(np.array(offset, dtype=float) / 100, 0)
            g, hover = g + shift, hover + shift
            self.play(Phase("shift", {arm: Goal(p=hover)}, speed="slow"))
        opening = self.width(arm) if width is None else width
        self.play(Phase("down", {arm: Goal(p=g, width=opening)}, speed="slow"),
                  Phase("close", {arm: Goal(width=0)}))
        w = self.width(arm)
        if w < HOLD_MIN_MM:
            self.play(Phase("open", {arm: Goal(width=opening)}), Phase("up", {arm: Goal(p=hover)}))
            raise Fail("miss", f"{arm} closed on nothing")
        self.play(Phase("squeeze", {arm: Goal(width=max(0.0, w - SQUEEZE_MM))}))
        d = r["obj"]
        if d is not None:
            T_obj = obj_pose(np.array(d["center_cm"]) / 100, d["size_cm"][2] / 200, d["yaw"])
            size = [v / 100 for v in d["size_cm"]]
            name = d["name"]
        else:
            T_obj, size, name = self.tool_T(arm), [w / 1000] * 3, "object"
        self.core.holding[arm] = dict(name=name, size=size, rel=np.linalg.inv(self.tool_T(arm)) @ T_obj,
                                      grasp_z=float(g[2]), width=w)
        self.play(Phase("lift", {arm: Goal(p=hover)}))
        return self.check_holding(arm, "slip")

    def check_holding(self, arm, kind):
        w = self.width(arm)
        if w < HOLD_MIN_MM:
            self.core.holding[arm] = None
            raise Fail(kind, f"{arm} lost the object")
        return {"width_mm": round(w, 1)}

    def move_to(self, arm, xy, z=10.0, yaw=None, pitch=None):
        p = np.array([xy[0] / 100, xy[1] / 100, z / 100])
        self.play(Phase("move", {arm: Goal(p=p, pitch="steep" if pitch is None else pitch, yaw=yaw, line=False)}))
        if self.core.holding.get(arm):
            return self.check_holding(arm, "slip")
        return {}

    def release(self, arm, set_down=True):
        h = self.core.holding.get(arm)
        w = self.width(arm)
        opening = min(self.scene.stroke_mm, w + 30)
        tip = self.tool_T(arm, self.q()[arm])[:3, 3]
        if set_down and h is not None:
            low = np.array([tip[0], tip[1], h["grasp_z"] + 0.002])
            self.play(Phase("lower", {arm: Goal(p=low)}, speed="slow"),
                      Phase("let go", {arm: Goal(width=opening)}),
                      Phase("up", {arm: Goal(p=[tip[0], tip[1], max(tip[2], TRAVEL_Z)])}))
        else:
            self.play(Phase("let go", {arm: Goal(width=opening)}))
            self.pull_out(arm)
        self.core.holding[arm] = None
        return {}

    def pull_out(self, arm):
        """Slide the open fingers back off the object along the approach, keeping the wrist's angle (steeper if the
        arm can't keep it that high)."""
        T = self.tool_T(arm, self.q()[arm])
        p = T[:3, 3] + T[:3, 2] * PULL_OUT
        try:
            self.play(Phase("pull out", {arm: Goal(p=p)}, speed="slow"))
        except Fail as e:
            if e.kind != "unreachable":
                raise
            self.play(Phase("pull out", {arm: Goal(p=p, pitch="steep", line=False)}, speed="slow"))

    def home(self, arms=("left", "right")):
        arms = [arms] if isinstance(arms, str) else list(arms)
        goals = {}
        for s in arms:
            q, _ = self.scene.arms[s].steepest(self.scene.home_tip(s), seed=self.q()[s][:5], lowest=0)
            goals[s] = Goal(joints=q, line=False)
        self.play(Phase("home", goals))
        return {}

    def pick(self, arm, target, width=None, offset=None):
        self.reach(arm, target=target, width=width, offset=offset)
        return self.grasp(arm)

    def place(self, arm, xy=None, on=None):
        """Put the held object down with its centre at xy (cm), or on another object."""
        if on is not None:
            xy = self.resolve(on)["center_cm"]
        if xy is None:
            raise Fail("check", "place needs xy or on")
        h = self.hold(arm)
        arm_k = self.scene.arms[arm]
        want = np.array(xy, dtype=float) / 100
        yaw = arm_k.pose(self.q()[arm])[2]
        tip = want.copy()
        for _ in range(3):  # where the fingertips go so that the object's centre lands on xy
            try:
                q, _ = arm_k.steepest(np.append(tip, TRAVEL_Z), yaw, seed=self.q()[arm][:5])
            except kin.Unreachable as e:
                raise Fail("unreachable", str(e))
            centre = (self.tool_T(arm, np.append(q, 0.0)) @ h["rel"])[:2, 3]
            tip = tip + (want - centre)
        self.move_to(arm, cm(tip), yaw=yaw)
        return self.release(arm, set_down=True)

    def handover(self, giver, taker, mode="air", at=None):
        h = self.hold(giver)
        if self.core.holding.get(taker):
            raise Fail("check", f"{taker} is already holding something")
        if mode == "table":
            return self._handover_table(giver, taker)
        if mode != "air":
            raise Fail("check", f"mode must be air or table, not {mode!r}")
        L = max(h["size"])
        m = np.append(self.scene.middle(), MEET_Z) if at is None else np.array(at, dtype=float) / 100
        toward = np.sign(self.scene.base_xy(taker)[0] - self.scene.base_xy(giver)[0])  # +1: taker on the right
        axis = np.array([toward, 0.0, 0.0])
        i_long = int(np.argmax(h["size"]))
        s_g = float(np.linalg.inv(h["rel"])[i_long, 3])  # where the giver holds it, along the object from its centre
        s_t = -(np.sign(s_g) or 1.0) * (L / 2 - min(END_MARGIN, L / 4))  # the taker takes the other end
        g_tip = m - axis * abs(s_t - s_g) / 2  # the two grips straddle the meeting point
        # The giver holds it across its jaws, so its jaw line is across the pass direction. Of the wrist angles that
        # reach the meeting point and point the free end at the taker, take the one that holds the object most level:
        # then the taker's jaws, set across it from above, are square to it.
        arm_g = self.scene.arms[giver]
        yaw_meet = (math.degrees(math.atan2(axis[1], axis[0])) + 90) % 180

        def taker_spot(q):
            T = self.tool_T(giver, np.append(q, self.width(giver))) @ h["rel"]
            return T[:3, 3] + T[:3, i_long] * s_t, T

        options = []
        for pitch in range(90, 29, -5):
            try:
                q = arm_g.ik(g_tip, pitch, yaw_meet, seed=self.q()[giver][:5], restarts=1)
            except kin.Unreachable:
                continue
            flipped = q.copy()
            flipped[4] += -180.0 if q[4] > 0 else 180.0
            for qq in (q, flipped):
                if not arm_g.within_limits(qq):
                    continue
                t_pt, T_obj = taker_spot(qq)
                if (t_pt - g_tip) @ axis > 0:
                    options.append((abs(T_obj[2, i_long]), -pitch, qq, t_pt, T_obj))
        options.sort(key=lambda o: (round(o[0], 2), o[1]))
        err = Fail("unreachable", f"{giver} can't hold it at the meeting point with the free end toward {taker}")
        for _, _, q_g, t_pt, T_obj in options:
            try:  # the taker's jaws close square to the object as it is held, tilted or not
                _, hover_t, _, q_t = self.approach(taker, t_pt, None, t_pt[2] + TAKE_HOVER, perp=T_obj[:3, i_long])
                break
            except Fail as e:
                err = e
        else:
            raise err
        self.play(Phase("meet", {giver: Goal(joints=q_g, line=False),
                                 taker: Goal(joints=q_t, width=self.scene.stroke_mm, line=False)}))
        self.play(Phase("take", {taker: Goal(p=t_pt)}, speed="slow"), Phase("close", {taker: Goal(width=0)}))
        w = self.width(taker)
        if w < HOLD_MIN_MM:
            self.play(Phase("open", {taker: Goal(width=self.scene.stroke_mm)}), Phase("up", {taker: Goal(p=hover_t)}))
            raise Fail("miss", f"{taker} closed on nothing; {giver} still holds it")
        self.play(Phase("squeeze", {taker: Goal(width=max(0.0, w - SQUEEZE_MM))}))
        T_obj = self.tool_T(giver) @ h["rel"]
        self.play(Phase("let go", {giver: Goal(width=min(self.scene.stroke_mm, self.width(giver) + 25))}))
        self.core.holding[giver] = None
        self.pull_out(giver)
        self.home(giver)
        if self.width(taker) < HOLD_MIN_MM:
            self.core.holding[taker] = None
            raise Fail("dropped", f"the object fell when {giver} let go")
        bottom = T_obj[2, 3] - h["size"][2] / 2
        self.core.holding[taker] = dict(h, rel=np.linalg.inv(self.tool_T(taker)) @ T_obj, width=self.width(taker),
                                        grasp_z=float(self.tool_T(taker)[2, 3] - bottom))
        return {"width_mm": round(self.width(taker), 1)}

    def _handover_table(self, giver, taker):
        spot = self.scene.middle()
        self.place(giver, xy=cm(spot))
        self.home(giver)
        dets = [d for d in self.detect() if not d["held_by"]]
        if not dets:
            raise Fail("not_seen", "nothing on the table after setting it down")
        d = min(dets, key=lambda d: np.linalg.norm(np.array(d["center_cm"]) / 100 - spot))
        return self.pick(taker, d["id"])

    # ---- the runner ------------------------------------------------------------------------------------------------
    def run(self, name, args):
        """Run one primitive and return its result."""
        core = self.core
        t0, ev0 = core.clock(), core.ev_id
        self.moved = False
        outcome, fail, why, detail = "done", None, None, {}
        fn = getattr(self, name, None)
        try:
            if name.startswith("_") or name not in PRIMITIVES or fn is None:
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
        side = args.get("arm") or args.get("giver")
        res = dict(op=name, outcome=outcome, fail=fail, why=why, t=[round(t0, 2), round(core.clock(), 2)],
                   arms=core.arms_state(), events=core.events_since(ev0), **detail)
        if side is not None:
            res["arm"] = side
        if name == "handover":
            res.update(giver=args.get("giver"), taker=args.get("taker"))
        return res


PRIMITIVES = ("reach", "grasp", "move_to", "release", "handover", "home", "pick", "place")
