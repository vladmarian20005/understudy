"""The motion primitives the agents call: reach, grasp, move_to, release, handover and home, plus pick and place.

Each runs as one robotd command, one at a time across both arms, and returns the same result shape (Skills.run).
A failure raises Fail with one of the failure kinds; robotd reports it as the result's outcome, fail and why.
Units at this boundary: cm, degrees, mm for the gripper opening. Inside: metres.

Touch-down: near the table nothing trusts a computed height. A grasp goes down past the table until the fingertips
rest on it (the controller feels it), then rises to the grasp height above where it touched; a finger that lands on
the object's top instead is a finger_on_object failure. Setting an object down lowers it until it rests on the table,
then opens. The arms' gravity sag is compensated in the controller; touch-down takes care of what's left.
"""
import inspect
import itertools
import math
import time

import numpy as np

from . import kin, perception
from .motion import Goal, Phase, PlanError, held_corners

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
TOUCH_BELOW = 0.015  # m: a touch-down aims this far below where contact is expected
CARRY_CLEAR = 0.02  # m: a held object is carried with its lowest corner at least this high
SWING_UP = 0.06  # m: how much higher it goes when a direct swing would drag it
TOUCH_FLOOR = -0.006  # m: and never lower than this under the table
ON_TABLE = 0.008  # m: a grasp that touches down with its lowest finger higher than this landed on the object
LIFT_OFF = 0.005  # m: after touching the table, the lowest finger rises at least this far before closing
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

    def touch_down(self, arm, p, speed="slow", **goal):
        """Lower arm's fingertips toward p (m) until they rest on something. Returns where they touched, as
        dict(tip=, finger=) heights in m (finger = the lowest finger), or None if they reached p without touching."""
        p = np.array(p, dtype=float)
        p[2] = max(p[2], TOUCH_FLOOR)

        def plan(**extra):
            return self.core.planner.plan(self.q(), [Phase("touch down", {arm: Goal(p=p, floor=TOUCH_FLOOR, **goal,
                                                                                    **extra)},
                                                           speed=speed, stop_on_contact=True)])

        try:
            try:
                trajs = plan()
            except PlanError as e:  # the current pitch doesn't reach that low: as steep as reaches there
                if e.kind != "unreachable" or "pitch" in goal:
                    raise
                try:
                    trajs = plan(pitch="steep")
                except PlanError as e2:  # close to the base a straight line can't be followed: a short joint move
                    if e2.kind != "unreachable":
                        raise
                    trajs = plan(pitch="steep", line=False)
        except PlanError as e:
            raise Fail(e.kind, e.why)
        ok, why = self.core.ctl.play(trajs, self.core.cancel)
        self.moved = True
        if not ok and why == f"contact: {arm}":
            c = self.core.ctl.contact.get(arm)
            self.core.event(f"{arm} touched down: fingertips at {c['tip'] * 100:.1f} cm, lowest finger at "
                            f"{c['finger'] * 100:.1f} cm")
            return c
        if not ok:
            if why.startswith(("blocked", "hot", "control error")):
                raise Stopped(why, "blocked")
            raise Stopped(why)
        return None

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
        is held near an end, the one on this arm's side first (sideways, so it's the same end however the object is
        turned: a policy learns one grasp, not two), so the rest stays free for a pass; its middle is the last resort
        (an end can sit too close to the base to reach)."""
        L, W, H = (v / 100 for v in d["size_cm"])
        c = np.array(d["center_cm"]) / 100
        u = np.array([math.cos(math.radians(d["yaw"])), math.sin(math.radians(d["yaw"]))])
        points = [c]
        if L >= LONG * W:
            s = L / 2 - min(END_MARGIN, L / 4)
            ends = sorted([c + u * s, c - u * s], key=lambda e: abs(e[0] - self.scene.base_xy(side)[0]))
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
                q_hover = arm.canonical_roll(arm.ik(hover, pitch, yaw_q, seed=q, restarts=1))
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
            for hz in (z, 0.7 * z, 0.5 * z):  # close in front of the base a high hover doesn't reach: lower, same spot
                try:
                    pitch, hover, _, q_hover = self.approach(arm, g, yaw, hz / 100)
                    break
                except Fail as e:
                    err = e
            else:
                continue
            break
        else:
            raise err
        width = opening if width is None else width
        self.play(Phase("reach", {arm: Goal(joints=q_hover, line=False,
                                            width=self.scene.stroke_mm if width is None else width)}))
        self.reached[arm] = dict(obj=d, point=g, hover=hover, pitch=pitch, yaw=yaw, q_hover=q_hover)
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
        line = (g - hover) / (hover[2] - g[2])  # down the approach, per metre of height
        c = self.touch_down(arm, g + line * TOUCH_BELOW, width=opening)
        if c is not None and c["finger"] > ON_TABLE:
            self.play(Phase("up", {arm: Goal(p=hover, pitch=r["pitch"])}))
            raise Fail("finger_on_object", f"{arm} touched down on top of the object at {c['finger'] * 100:.1f} cm")
        if c is not None:  # on the table: back up the approach to the grasp height, measured from where it touched
            z = max(c["tip"] + LIFT_OFF, c["finger"] + g[2])
            g = g - line * (z - g[2])
            self.play(Phase("rise", {arm: Goal(p=g, pitch=r["pitch"], floor=TOUCH_FLOOR)}, speed="slow"))
        self.play(Phase("close", {arm: Goal(width=0)}))
        w = self.width(arm)
        if w < HOLD_MIN_MM:
            self.play(Phase("open", {arm: Goal(width=opening)}), Phase("up", {arm: Goal(p=hover, pitch=r["pitch"])}))
            raise Fail("miss", f"{arm} closed on nothing")
        self.play(Phase("squeeze", {arm: Goal(width=max(0.0, w - SQUEEZE_MM))}))
        d = r["obj"]
        if d is not None:
            T_obj = obj_pose(np.array(d["center_cm"]) / 100, d["size_cm"][2] / 200, d["yaw"])
            # closing fingers centre what they grip between them: the object's middle is now on the jaws' centre line
            T = self.tool_T(arm)
            j = T[:3, :3] @ self.scene.arms[arm].jaw
            T_obj[:3, 3] -= j * ((T_obj[:3, 3] - T[:3, 3]) @ j)
            size = [v / 100 for v in d["size_cm"]]
            name = d["name"]
        else:
            T_obj, size, name = self.tool_T(arm), [w / 1000] * 3, "object"
        self.core.holding[arm] = dict(name=name, size=size, rel=np.linalg.inv(self.tool_T(arm)) @ T_obj,
                                      grasp_z=float(g[2]), width=w)
        self.lift(arm, hover, r["pitch"], r.get("q_hover"))
        return self.check_holding(arm, "slip")

    def lift(self, arm, p, pitch, q_back=None):
        """Up to p at the approach's pitch; if that line is just out of reach, back to the joints it came down from
        (q_back), then as steep as reaches. A fresh solution for the way up can have the wrist turned the other way
        round, which swings a long held object's far end through the table."""
        try:
            self.play(Phase("lift", {arm: Goal(p=p, pitch=pitch)}))
            return
        except Fail as e:
            if e.kind != "unreachable":
                raise
        if q_back is not None:
            try:
                self.play(Phase("lift", {arm: Goal(joints=q_back, line=False)}))
                return
            except Fail as e:
                if e.kind not in ("unreachable", "check"):
                    raise
        self.play(Phase("lift", {arm: Goal(p=p, pitch="steep", line=False)}))

    def check_holding(self, arm, kind):
        w = self.width(arm)
        if w < HOLD_MIN_MM:
            self.core.holding[arm] = None
            raise Fail(kind, f"{arm} lost the object")
        return {"width_mm": round(w, 1)}

    def solve(self, arm, p, yaw=None, pitch=None, turn=None):
        """Joints for the fingertips at p (m), steepest pitch unless given. The same jaw line half a turn of the wrist
        round flips a held object end for end: turn="keep" keeps the wrist's turn, "flip" turns it round, None keeps
        it where the joint allows."""
        arm_k = self.scene.arms[arm]
        q_now = self.q()[arm][:5]
        yaw = arm_k.pose(q_now)[2] if yaw is None else yaw
        try:
            q = arm_k.ik(p, pitch, yaw, seed=q_now) if pitch is not None else arm_k.steepest(p, yaw, seed=q_now)[0]
        except kin.Unreachable as e:
            raise Fail("unreachable", str(e))
        want_flip = turn == "flip"
        if (abs(q[4] - q_now[4]) > 90) != want_flip:
            turned = q.copy()
            turned[4] += -180.0 if turned[4] > q_now[4] or (want_flip and turned[4] > 0) else 180.0
            if arm_k.within_limits(turned):
                q = turned
            elif turn is not None:
                raise Fail("unreachable", f"{arm} can't get there with its wrist turned that way")
        return q

    def move_to(self, arm, xy, z=10.0, yaw=None, pitch=None):
        q = self.solve(arm, np.array([xy[0] / 100, xy[1] / 100, z / 100]), yaw, pitch)
        self.play(Phase("move", {arm: Goal(joints=q, line=False)}))
        if self.core.holding.get(arm):
            return self.check_holding(arm, "slip")
        return {}

    def release(self, arm, set_down=True):
        h = self.core.holding.get(arm)
        w = self.width(arm)
        opening = min(self.scene.stroke_mm, w + 30)
        tip = self.tool_T(arm, self.q()[arm])[:3, 3]
        if set_down and h is not None:  # lower until the object's low end rests on the table, then open
            low = float(np.min(held_corners(self.tool_T(arm, self.q()[arm]), h)[:, 2]))
            self.touch_down(arm, [tip[0], tip[1], tip[2] - low - TOUCH_BELOW])
            self.play(Phase("let go", {arm: Goal(width=opening)}))
            self.core.holding[arm] = None  # it rests on the table now: the way up mustn't carry it along
            self.play(Phase("up", {arm: Goal(p=[tip[0], tip[1], max(tip[2], TRAVEL_Z)], pitch="steep", line=False)}))
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

    def home(self, arms=("left", "right"), width=None):
        """Park the arms; width (mm) also sets the grippers' opening."""
        arms = [arms] if isinstance(arms, str) else list(arms)
        goals = {}
        for s in arms:  # the same home pose every time, wrist included: every episode starts alike
            q, _ = self.scene.arms[s].steepest(self.scene.home_tip(s), lowest=0)
            goals[s] = Goal(joints=self.scene.arms[s].canonical_roll(q), line=False, width=width)
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
        want = np.array(xy, dtype=float) / 100
        yaw_now = self.scene.arms[arm].pose(self.q()[arm])[2]
        # the jaw angle doesn't matter for setting it down: try the current one first, then others, each with the
        # wrist either way round (the held object points the other way); best one it can lower straight down from
        def pose(down):
            for yaw, turn in itertools.product([yaw_now + d for d in (0, 30, -30, 60, -60, 90)], ("keep", "flip")):
                try:
                    return self.place_pose(arm, h, want, yaw % 180, turn, down=down)
                except Fail:
                    continue
            return None

        poses = pose(True) or pose(False)
        if poses is None:
            raise Fail("unreachable", f"{arm} can't put its centre at ({xy[0]:.1f}, {xy[1]:.1f}) cm")
        q, q_high = poses
        # swing over high and come down onto the spot, so the object doesn't sweep low on the way; if even that
        # would drag it, lift it straight up first
        over = [Phase("move", {arm: Goal(joints=q_high, line=False)})]
        if q_high is not q:
            over.append(Phase("down", {arm: Goal(joints=q, line=False)}))
        try:
            self.play(*over)
        except Fail as e:
            if "would hit the table" not in e.why:
                raise
            up = self.tool_T(arm, self.q()[arm])[:3, 3] + [0, 0, SWING_UP]
            for goal in (dict(), dict(pitch="steep"), dict(pitch="steep", line=False)):  # straight up if it can
                try:
                    self.play(Phase("up", {arm: Goal(p=up, **goal)}), *over)
                    break
                except Fail as e2:
                    if e2.kind not in ("unreachable", "check"):
                        raise
            else:
                raise Fail("check", f"{arm} can't lift the object clear of the table to carry it over")
        self.check_holding(arm, "slip")
        return self.release(arm, set_down=True)

    def place_pose(self, arm, h, want, yaw, turn, down=False):
        """Joints that hold the object with its centre over want (m) and its low end CARRY_CLEAR above the table,
        and the highest of SWING_UP, half that or the same pose to swing over in, for one jaw angle and wrist turn.
        down: also only where the set-down can lower it straight down at the same pitch (close to the base a shallow
        pitch doesn't reach low, and a steeper one swings a held object's far end down into the table).
        Raises Fail when they don't exist."""
        def centred(z):
            tip = want.copy()
            for _ in range(6):  # where the fingertips go so that the object's centre lands on want
                q = self.solve(arm, np.append(tip, z), yaw, turn=turn)
                off = want - (self.tool_T(arm, np.append(q, 0.0)) @ h["rel"])[:2, 3]
                if np.linalg.norm(off) < 0.005:
                    return q
                tip = tip + off
            raise Fail("unreachable", "the object's centre doesn't converge on the spot")

        q = centred(TRAVEL_Z)
        # a long object held at one end hangs tilted: carry it high enough that its low end clears the table
        low = float(np.min(held_corners(self.tool_T(arm, np.append(q, 0.0)), h)[:, 2]))
        z = TRAVEL_Z + max(0.0, CARRY_CLEAR - low)
        if z > TRAVEL_Z:
            q = centred(z)
        if down:
            T = self.tool_T(arm, np.append(q, 0.0))
            low = float(np.min(held_corners(T, h)[:, 2]))
            tip = T[:3, 3] - [0.0, 0.0, low + TOUCH_BELOW]  # where release's touch-down aims
            tip[2] = max(tip[2], TOUCH_FLOOR)
            _, pitch, yaw_q = self.scene.arms[arm].pose(q)
            try:
                self.scene.arms[arm].ik(tip, pitch, yaw_q, seed=q)
            except kin.Unreachable as e:
                raise Fail("unreachable", str(e))
        for up in (SWING_UP, SWING_UP / 2):  # close in front of the base, high poses don't reach
            try:
                return q, centred(z + up)
            except Fail:
                pass
        return q, q

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
        c = self.touch_down(taker, t_pt)
        if c is not None:  # a finger came down on the object instead of beside it
            self.play(Phase("up", {taker: Goal(p=hover_t, pitch="steep")}))
            raise Fail("finger_on_object", f"{taker} touched the object at {c['finger'] * 100:.1f} cm; {giver} "
                       "still holds it")
        self.play(Phase("close", {taker: Goal(width=0)}))
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
