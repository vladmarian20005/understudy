"""The arms' actions in a plan step (grasp, move, tilt, open, grip, nudge, park), both arms at the same time.

An action is one arm's part of a step: a generator of stages. A stage is the keyframes that arm goes through next
(and what to do just before, like capping the grip force); after it the action gets both arms' state back and
decides its next stage, or stops with its result, or raises Failed. A step runs its two actions stage by stage
together: the k-th stages of both arms go to robotd as one move, so both arms work at once and robotd still checks
the pair (reach, the table, the gap between the arms, what each holds) before anything moves.

Everything is in table cm. Positions an action is given are for the grip point, the spot between the finger pads
2 cm back from the fingertips, where the jaws pinch and where what the gripper holds sits; robotd's keyframes are for
the fingertips, so each one is shifted along the fingers' direction.
"""
import time
from dataclasses import dataclass

import numpy as np

from harness import robot

from . import tools

GRIP_BACK = tools.GRIP_BACK_CM
GAP_CM = 4.0  # a grasp comes down this far in front of the object, then goes in
OVER_CM = 4.0  # and starts this much above the grasp height
WIDE_MM = 20.0  # the jaws open this much wider than the object
HOLD_MA = 120.0  # the gripper motor draws more than this while it squeezes something (bench: ~280 on a ball,
# 50-80 empty)
PITCHES = (80, 70, 60, 50, 40, 30, 20, 10, 0)  # tried steepest first when a grasp isn't told its pitch
STEEP = 55  # a grasp this steep comes straight down onto the object; flatter ones come in from the front
TALL_CM = 10.0  # an object this tall is gripped from the side, as level as reaches
EDGE = 0.85  # jaws stopping narrower than this share of the object's width caught its edge (31 Oct: a ball held
# at 50 of 65 mm, 77%, slipped out on the lift)
CORRECT = 2  # times a grip point that ends short of where it was sent is sent again
MISS_CM = 0.7  # short by more than this
MAX_FIX_CM = 2.5  # at most this much past (more is something in the way)
REGRIPS = 2  # then they open, go in deeper and close again, this many times at most
# where each grip point parks: raised, level and out to the side, so the overhead camera sees the table (31 Oct: home
# hid 430 cm2 of it, the left arm right over the balls; this hides about 50, all near the bases)
# 1 Nov: with the original gripper (6b0e032) (+-26, 12, 28) no longer reaches level (5 mm, 16 deg off), so every park
# and every step with one was refused; 4 cm farther out does (checked with check_only on both sims)
PARK = {"left": [-30.0, 12.0, 28.0], "right": [30.0, 12.0, 28.0]}


class Failed(Exception):
    pass


@dataclass
class Stage:
    keyframes: list
    speed: str = "normal"
    before: object = None


def heading(side, xy):
    """Unit table direction from the arm's base to xy (cm)."""
    d = np.asarray(xy[:2], dtype=float) - tools.scene().base_xy(side) * 100
    return d / max(np.linalg.norm(d), 1e-6)


def tips(grip, head, pitch):
    """Where the fingertips are when the grip point is at grip (cm), the fingers pointing along head, tilted pitch."""
    p = np.radians(pitch)
    a = np.array([np.cos(p) * head[0], np.cos(p) * head[1], -np.sin(p)])
    return np.asarray(grip, dtype=float) + a * GRIP_BACK


def kf(grip, head, pitch, yaw=None, **extra):
    t = tips(grip, head, pitch)
    k = {"xy": [round(float(t[0]), 2), round(float(t[1]), 2)], "z": round(float(t[2]), 2), "pitch": float(pitch)}
    if yaw is not None:
        k["yaw"] = float(yaw) % 180
    k.update(extra)
    return k


def arm_heading(side, arm):
    """The way an arm's fingers point now, flat on the table."""
    f, _, _ = tools.nudge_frame(side, arm["joints"])
    return f[:2] / max(np.linalg.norm(f[:2]), 1e-6)


def grip_point(side, arm):
    _, g, _ = tools.frame(side, arm["joints"])
    return g * 100


def sent_pitch(side, arm):
    return tools._pitch.get(side, arm["pitch"])


def current(side, n=3):
    """The gripper motor's current, the median of a few reads (it jitters)."""
    vals = []
    for _ in range(n):
        vals.append(robot.status()["arms"][side].get("grip_ma", 0))
        time.sleep(0.07)
    return float(np.median(vals))


def find(name):
    """A remembered object by name: exact, else the only one whose name contains it."""
    objs = tools._memory["objects"]
    hit = [o for o in objs if o["name"] == name] or [o for o in objs if name.lower() in o["name"].lower()]
    if len(hit) != 1:
        raise Failed(f"no single object called {name!r}: there are {[o['name'] for o in objs]}")
    return hit[0]


def target_of(name, arms):
    """The top of an object, cm: where it is held now if an arm holds it, else as it stands on the table."""
    for s, a in arms.items():
        if (a.get("held") or {}).get("name") == name:
            return np.array(a["held"]["top_cm"], dtype=float), a["held"]
    o = find(name)
    return np.array(o["center_cm"] + [o["size_cm"][2]], dtype=float), None


# ---- actions: generators of stages, each gets both arms' state after its stage ------------------------------------
def reach(side, g, head, p, yaw, speed="slow"):
    """The grip point to g, and there for real: a stretched arm's joints settle a few degrees short of their goals
    (31 Oct: every grasp 2-3 cm short of the object, both arms), so it's sent again past g by what's still missing,
    up to CORRECT times. Measured from the joints' encoders."""
    arms = yield Stage([kf(g, head, p, yaw)], speed)
    for _ in range(CORRECT):
        miss = np.asarray(g, dtype=float) - grip_point(side, arms[side])
        if np.linalg.norm(miss) < MISS_CM:
            break
        miss *= min(1.0, MAX_FIX_CM / np.linalg.norm(miss))
        arms = yield Stage([kf(np.asarray(g) + miss, head, p, yaw)], speed)
    return arms


def grasp(side, arms, object, height=None, pitch=None, lift=5.0, offset=None):
    """Grip point onto the object at height (cm above the table; default its middle), fingers along the line from
    the arm's base, jaws square to it: open wide, come down in front of it, go in, close with the capped force, hold,
    lift. Fails (open again, backed off) when the jaws close on nothing or it slips on the lift."""
    o = find(object)
    c = np.array(o["center_cm"], dtype=float)
    H, wide = float(o["size_cm"][2]), float(min(o["size_cm"][:2]))
    h = float(o.get("z_cm", H / 2)) if height is None else float(height)
    head = heading(side, c)
    right = np.array([head[1], -head[0]])
    g = np.array([c[0], c[1], h])
    if offset:
        f, r, u = (list(offset) + [0, 0, 0])[:3]
        g[:2] += f * head + r * right
        g[2] += u
    yaw = np.degrees(np.arctan2(head[1], head[0])) + 90
    width = min(tools.scene().stroke_mm, wide * 10 + WIDE_MM)
    head3 = np.append(head, 0.0)
    approach_starts = {}  # pitch -> where the grip point starts its approach

    def approach(p):
        """Steep: the open jaws come straight down round it from above its top. Flatter: down in front of it, then
        in along the fingers (31 Oct: a ball 22 cm from the base had no room in front for that)."""
        if p >= STEEP:
            approach_starts[p] = g + [0, 0, max(OVER_CM, H - h + 3)]
            return [kf(approach_starts[p], head, p, yaw, width=width), kf(g, head, p, yaw)]
        down = g - head3 * (wide / 2 + GAP_CM)
        approach_starts[p] = down + [0, 0, OVER_CM]
        return [kf(approach_starts[p], head, p, yaw, width=width), kf(down, head, p, yaw), kf(g, head, p, yaw)]

    # tall: from the side, as level as reaches (31 Oct: tilted 30-75 deg at a 26 cm tube, the wrist all wrong);
    # otherwise the steepest that reaches
    order = PITCHES if H < TALL_CM else tuple(reversed(PITCHES))
    for p in ([pitch] if pitch is not None else order):
        frames = approach(p)
        if robot.move(**{side: frames}, check_only=True).get("outcome") == "done":
            break
    else:
        raise Failed(f"{side} can't reach a grasp on {object} at {h:.0f} cm high"
                     + ("" if pitch is None else f" at pitch {pitch}"))
    held_pitch = sent_pitch(side, arms[side])
    tools._pitch[side] = float(p)
    # Coming down from well above (after letting go over a tall thing): across at this height first, then down, so
    # the arm doesn't sweep through what it was above (1 Nov: it knocked the tube it had just filled into the next
    # ball). A start that can't be reached that way goes straight, as before.
    now = grip_point(side, arms[side])
    start = approach_starts[p]
    if now[2] - start[2] > 5 and np.linalg.norm(now[:2] - start[:2]) > 3:
        # at the pitch it has: turning the wrist steep right above the tall thing swings the fingers down into it
        across = kf([start[0], start[1], now[2]], head, held_pitch, None)
        if robot.move(**{side: [across] + frames}, check_only=True).get("outcome") == "done":
            frames = [across] + frames
    yield Stage(frames[:-1])
    yield from reach(side, g, head, p, yaw)
    regrips = 0
    while True:
        yield Stage([{"width": 0}], "slow", lambda: robot.grip_force(side, tools.GRIP_PCT))
        time.sleep(tools.GRIP_SETTLE_S)
        w = robot.status()["arms"][side]["width_mm"]
        if w >= EDGE * wide * 10 or regrips == REGRIPS:
            break
        # Narrower than the object: the jaws caught its near edge, it sits farther along the fingers than the
        # numbers said (31 Oct: a 75 mm tube gripped at 46-56 mm four times, 2-3 cm short). The chord says how far.
        r = wide / 2
        deeper = r if w < tools.HOLD_MIN_MM else float(np.sqrt(max(r * r - (w / 20) ** 2, 0.0)))
        g = g + head3 * min(deeper + 0.5, r)
        regrips += 1
        yield Stage([{"width": width}], "slow")
        yield from reach(side, g, head, p, yaw)
    if w < tools.HOLD_MIN_MM:
        yield Stage([{"width": width}, frames[0]])
        raise Failed(f"{side} closed on nothing at {object} (jaws shut to {w:.0f} mm, {regrips} re-grips): opened "
                     "and backed off")
    robot.hold(side, o["name"], o["size_cm"], [float(g[0]), float(g[1])], o.get("z_cm", H / 2), o.get("yaw", 0.0))
    yield Stage([{"width": max(0.0, w - tools.GRIP_HOLD_MM)}, {"rel": [0, 0, float(lift)], "pitch": float(p)}])
    ma, w2 = current(side), robot.status()["arms"][side]["width_mm"]
    if ma < HOLD_MA or w2 < tools.HOLD_MIN_MM:
        robot.hold(side)
        yield Stage([{"width": width}])
        raise Failed(f"{side} lost {object} on the lift (jaws closed at {w:.0f} mm on a {wide * 10:.0f} mm object, "
                     f"now {w2:.0f} mm, motor {ma:.0f} mA; {regrips} re-grips): opened")
    return dict(holding=o["name"], jaws_mm=round(w, 1), squeeze_ma=round(ma), pitch=p, regrips=regrips)


def move(side, arms, to=None, above=None, by=3.0, rel=None, pitch=None, yaw=None, speed=None):
    """The grip point (what the gripper holds) to `to` [x, y, z], or `by` cm above the top of the object `above`
    (held by an arm or standing), or shifted by rel [dx, dy, dz]; at pitch (default: the one it has) and yaw (default:
    kept)."""
    a = arms[side]
    p = sent_pitch(side, a) if pitch is None else float(pitch)
    g0 = grip_point(side, a)
    if rel is not None:
        goal = g0 + np.array(rel, dtype=float)
    elif to is not None:
        goal = np.array(to, dtype=float)
    elif above is not None:
        top, _ = target_of(above, arms)
        mine = a.get("held")
        goal = top + [0, 0, float(by) + (mine["size_cm"][2] / 2 if mine else 0.0)]
    else:
        raise Failed("move needs to, above or rel")
    head = heading(side, goal) if np.linalg.norm(goal[:2] - tools.scene().base_xy(side) * 100) > 8 else \
        arm_heading(side, a)

    def jaw(p):  # near level the jaw line can only lie square to where the fingers point
        return np.degrees(np.arctan2(head[1], head[0])) + 90 if yaw is None and p < 30 else yaw

    # Not told a pitch: the one it has if that reaches, else the nearest flatter one that does (flatter reaches
    # higher and farther), else steeper. 1 Nov: a ball gripped at 80 then sent 3 cm above a 26 cm tube was refused at
    # 80 and 30 and went in at 0; the plan had left pitch out, so the whole run stalled on it.
    tries = [p] if pitch is not None else [p] + [q for q in reversed(PITCHES) if q < p] + [q for q in PITCHES if q > p]
    # Going well up (onto something tall): up first where it is, then across, so neither the gripper nor what it
    # holds sweeps through the thing it's going above (1 Nov: a ball carried low-to-high in one line knocked the tube
    # over three times out of three).
    rise = goal[2] - g0[2] > 5 and np.linalg.norm(goal[:2] - g0[:2]) > 3
    up = np.array([g0[0], g0[1], goal[2]])

    def ok(frames):
        return robot.move(**{side: frames}, check_only=True).get("outcome") == "done"

    found = None
    for lift_first in ([True, False] if rise else [False]):  # straight across only when going up first can't
        for q in tries:
            frames = ([kf(up, head, q, jaw(q))] if lift_first else []) + [kf(goal, head, q, jaw(q))]
            if (len(tries) == 1 and not lift_first) or ok(frames):
                found = (q, lift_first)
                break
        if found:
            break
    if found is None:
        raise Failed(f"{side} can't reach {[round(float(v), 1) for v in goal]} cm at any pitch")
    p, rise = found
    tools._pitch[side] = p
    if rise:
        yield Stage([kf(up, head, p, jaw(p))], speed or "normal")
    arms = yield from reach(side, goal, head, p, jaw(p), speed or "normal")
    held = (arms or {}).get(side, {}).get("held")
    if above is not None and held:
        # What it holds needn't sit at the grip point (a ball gripped steep and turned level ends up 1-2 cm along the
        # fingers): line what it holds up over the target by where robotd has it, once.
        off = np.array(held["center_cm"][:2], dtype=float) - grip_point(side, arms[side])[:2]
        if 0.4 < np.linalg.norm(off) < 4:
            arms = yield from reach(side, goal - np.append(off, 0.0), head, p, jaw(p), "slow")
    return {} if pitch is not None or p == tries[0] else dict(pitch=p)


def tilt(side, arms, deg):
    """Turn the wrist by deg: what the gripper holds tilts about the fingers' direction."""
    yield Stage([{"roll": float(deg)}], "slow")
    return {}


def open_(side, arms, width=None):
    yield Stage([{"width": tools.scene().stroke_mm if width is None else float(width)}])
    robot.hold(side)
    return {}


def grip(side, arms):
    """Close with the capped force where the gripper is, and hold what's between the jaws."""
    yield Stage([{"width": 0}], "slow", lambda: robot.grip_force(side, tools.GRIP_PCT))
    time.sleep(tools.GRIP_SETTLE_S)
    a = robot.status()["arms"][side]
    if a["width_mm"] < tools.HOLD_MIN_MM:
        yield Stage([{"width": tools.scene().stroke_mm}])
        raise Failed(f"{side} closed on nothing: opened again")
    o = tools.at_grip(side, a)
    if o is not None:
        g = grip_point(side, a)
        robot.hold(side, o["name"], o["size_cm"], [float(g[0]), float(g[1])], o.get("z_cm", o["size_cm"][2] / 2),
                   o.get("yaw", 0.0))
    yield Stage([{"width": max(0.0, a["width_mm"] - tools.GRIP_HOLD_MM)}])
    return dict(holding=o["name"] if o else "something", jaws_mm=round(a["width_mm"], 1), squeeze_ma=round(current(side)))


def nudge(side, arms, forward=0.0, right=0.0, up=0.0):
    """A small step in the gripper's own directions (see tools.nudge), keeping its tilt."""
    a = arms[side]
    f, r, u = tools.nudge_frame(side, a["joints"])
    d = float(forward or 0) * f + float(right or 0) * r + float(up or 0) * u
    yield Stage([{"rel": [round(float(v), 2) for v in d], "pitch": sent_pitch(side, a)}], "slow")
    return {}


def park(side, arms, open=False):
    """Raised, level and out to the side, where the overhead camera sees the table past it; open: jaws open too."""
    x, y, z = PARK[side]
    tools._pitch[side] = 0.0
    yield Stage([dict({"xy": [x, y], "z": z, "pitch": 0.0}, **({"width": tools.scene().stroke_mm} if open else {}))])
    return {}


ACTIONS = {"grasp": grasp, "move": move, "tilt": tilt, "open": open_, "grip": grip, "nudge": nudge, "park": park}


def start(side, spec, arms):
    """A step's action for one arm, from the plan's {"do": name, ...}."""
    spec = dict(spec)
    name = spec.pop("do", None)
    if name not in ACTIONS:
        raise Failed(f"no action {name!r}: use {', '.join(ACTIONS)}")
    try:
        return ACTIONS[name](side, arms, **spec)
    except TypeError as e:
        raise Failed(f"bad arguments for {name}: {e}")


def run(step):
    """Run one plan step: {"left": action, "right": action}, both at once. Returns ({side: result}, stopped), where a
    result has "failed" when that arm's action couldn't finish, and stopped is robotd's reason when a person or a
    safety reflex stopped the arms (the run should end)."""
    arms = robot.status()["arms"]
    gens, results, stage = {}, {}, {}
    for side in ("left", "right"):
        if step.get(side):
            try:
                gens[side] = start(side, step[side], arms)
                stage[side] = next(gens[side])
            except StopIteration as e:
                results[side] = e.value or {}
            except Failed as e:
                results[side] = dict(failed=str(e))
    while stage:
        for st in stage.values():
            if st.before:
                st.before()
        frames = {s: st.keyframes for s, st in stage.items() if st.keyframes}
        speed = "slow" if any(st.speed == "slow" for st in stage.values()) else \
            "fast" if all(st.speed == "fast" for st in stage.values()) else "normal"
        r = robot.move(frames.get("left"), frames.get("right"), speed) if frames else dict(outcome="done")
        if r.get("outcome") != "done":
            why = f"{r.get('outcome')}: {r.get('why')}"
            for s in stage:
                results[s] = dict(failed=why)
            stopped = r.get("why") if r.get("outcome") == "stopped" and r.get("fail") in (None, "manual") else None
            return results, stopped
        arms = robot.status()["arms"]
        nxt = {}
        for s in stage:
            try:
                nxt[s] = gens[s].send(arms)
            except StopIteration as e:
                results[s] = e.value or {}
            except Failed as e:
                results[s] = dict(failed=str(e))
        stage = nxt
    return results, None
