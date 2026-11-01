"""The arms' actions in a plan step (grasp, move, place, put_in, tilt, open, grip, nudge, park), both arms at once.

An action is one arm's part of a step: a generator of stages. A stage is the keyframes that arm goes through next
(and what to do just before, like capping the grip force); after it the action gets both arms' state back and
decides its next stage, or stops with its result, or raises Failed. A step runs its two actions stage by stage
together: the k-th stages of both arms go to robotd as one move, so both arms work at once and robotd still checks
the pair (reach, the table, the gap between the arms, what each holds) before anything moves.

Everything is in table cm. Positions an action is given are for the grip point: where the middle of what the gripper
holds sits (a grasp puts it as deep in the jaws as it fits, against the fixed finger: tools.grip_offset), or with
nothing held a spot between the fingers GRIP_BACK_CM in. robotd's keyframes are for the fingertips, so each one is
shifted from the grip point along and across the fingers.

The original SO-101 gripper: the wrist camera on top (the roll stays upright: robotd keeps the roll joint when no
yaw is given), a fixed finger on the right of where the fingers point and a jaw that swings in from the left about a
hinge 8 cm back. A wide-open jaw pushes what it closes on out toward the fingertips, so a grasp goes deep.
"""
import time
from dataclasses import dataclass

import numpy as np

from harness import kin, robot

from . import tools, wrist
from .reach import GRASP_PITCHES as PITCHES, TALL_CM
from .rig import scene

GAP_CM = 4.0  # a grasp comes down this far in front of the object, then goes in
OVER_CM = 4.0  # and starts this much above the grasp height
WIDE_MM = 20.0  # the jaws open this much wider than the object
TIGHT_MM = 8.0  # or only this much, when something stands where the swinging jaw opens
JAW_CM = 1.5  # the swinging jaw's thickness
HOLD_MA = 120.0  # the gripper motor draws more than this while it squeezes something (idle: 50-80)
STEEP = 55  # a grasp this steep comes straight down onto the object; flatter ones come in from the front
TALL_PITCH = 20  # a tall thing is gripped from the side, no steeper than this: steeper, it hangs tilted and can't be set
# down upright
EDGE = 0.85  # jaws stopping narrower than this share of the object's width caught its edge, and it slips on the lift
CORRECT = 2  # times a grip point that ends short of where it was sent is sent again
MISS_CM = 0.7  # short by more than this
MAX_FIX_CM = 2.5  # at most this much past (more is something in the way)
REGRIPS = 2  # then they open, go in deeper and close again, this many times at most
LOOKS = 2  # wrist-camera looks before going in (each can move the aim and the open jaws)
LOOK_FIX_CM = 0.5  # a look that finds the object this far or farther from the aim moves the aim
DEEP_CM = 4.5  # a grasp puts the object's middle this far in from the fingertips, at most: held at the fingertips it
# slips and tips
PALM_CM = 7.5  # and its back no deeper than this: the jaws' palm
PALM_GAP_CM = 0.3  # less this, so the jaw doesn't press it against the palm
TIP_ABOVE_CM = 1.1  # a steep grasp keeps the fingertips this high above the table: any lower, the jaw pushes the object out
FIT_GAP_CM = 0.5  # the fixed finger comes down this far beside the object: at 3 mm a 2 mm error puts it on the object, at
# 8 mm the jaw shoves the object across and it slips out
WIDE_CM = 7.2  # wider than this, an object goes only WIDE_DEEP_CM in and the fixed finger only WIDE_GAP_CM beside it: the
WIDE_GAP_CM = 0.3  # jaw has less room round it, and deep it would be near the end of its travel, with nothing left to
WIDE_DEEP_CM = 3.2  # squeeze with
SET_ABOVE_CM = 0.5  # place lets go with the bottom of what it holds this far above where it lands
DROP_TALL_CM = 1.0  # open won't let go of a tall thing whose lowest point is higher than this over the table: it falls
CLEAR_CM = 2.0  # and needs this much room between it and anything standing next to where it lands
BACK_OFF_CM = 4.0  # after letting go the fingers back out this far along their own line
LONG = 1.8  # an object this many times longer than wide is gripped across its length when taken from above
AVOID_H_CM = 10.0  # standing things this tall are obstacles for the arms (shorter ones they pass over)
AVOID_PAD_CM = 1.5  # and this much wider than they look: the camera's error
RIM_CLEAR_CM = 3.0  # an arm going over something tall (put_in) keeps clear of it up to this far below its rim
VIA_CM = 5.0  # coming in over it from its own side: from this far outside it (the gripper's girth and the gap)
TRAVEL_MM = 20.0  # the jaws travel nearly shut: the open jaw sticks out and hits things
MIN_FORCE, MAX_FORCE = 5.0, 60.0  # % a grasp or grip may ask for
SLIP_MM = 6.0  # jaws this much narrower after the lift than where they stopped on the object: it slipped out
_near = set()  # what an action works on right now (the object a grasp goes for, the one just let go of): no obstacle
_over = set()  # what a move goes above: an obstacle only up to RIM_CLEAR_CM below its rim


class Failed(Exception):
    pass


@dataclass
class Stage:
    keyframes: list
    speed: str = "normal"
    before: object = None


def obstacles(moving=("left", "right")):
    """The standing things the arms keep clear of (robotd checks every motion against them): everything remembered
    10 cm or taller, except what a moving arm holds, what was dropped into something, and what an action is working
    on right now (_near). Something tall an arm holds while the other moves is in the way too, from the table up to its
    top. What a move goes above (_over) is in the way up to RIM_CLEAR_CM below its rim."""
    arms = robot.status()["arms"]
    held = {(a.get("held") or {}).get("name") for s, a in arms.items() if s in moving}
    out = [dict(name=o["name"], xy=[float(v) for v in o["center_cm"][:2]], r=float(max(o["size_cm"][:2])) / 2 + AVOID_PAD_CM,
                h=float(o["size_cm"][2]) - (RIM_CLEAR_CM if o["name"] in _over else 0.0)) for o in tools._memory["objects"]
           if float(o["size_cm"][2]) >= AVOID_H_CM and o["name"] not in held | _near and not o.get("dropped_into")
           and (tools._memory["looks"] - o["seen"] <= 1 or o.get("placed"))]  # not a ghost of where it was
    for s, a in arms.items():
        h = a.get("held")
        if s not in moving and h and h["name"] not in _near and float(max(h["size_cm"])) >= AVOID_H_CM:
            top = rim_cm(h) - RIM_CLEAR_CM if h["name"] in _over else float(max(h["top_cm"][2], h["bottom_cm"][2]))
            out = [o for o in out if o["name"] != h["name"]] + [dict(
                name=h["name"], xy=[float(v) for v in h["center_cm"][:2]], h=top,
                r=float(max(np.hypot(*np.subtract(h["top_cm"][:2], h["center_cm"][:2])),
                            min(h["size_cm"][:2]) / 2)) + AVOID_PAD_CM)]
    return out


def edge_drop(held):
    """How far a held box's end edge dips below the middle of that end, cm, from how far it leans."""
    ax = np.subtract(held["top_cm"], held["bottom_cm"], dtype=float)
    return float(max(held["size_cm"][:2])) / 2 * np.linalg.norm(ax[:2]) / max(float(np.linalg.norm(ax)), 1e-6)


def rim_cm(held):
    """How high the lowest point of the upper end's edge of what an arm holds is (a tilted mouth dips on one side)."""
    return float(max(held["top_cm"][2], held["bottom_cm"][2])) - edge_drop(held)


def lowest_cm(held):
    """How high the lowest point of what an arm holds is (robotd's box: its two ends and its width)."""
    return float(min(held["top_cm"][2], held["bottom_cm"][2])) - edge_drop(held)


def rmove(left=None, right=None, speed=None, check_only=None):
    """robot.move with the standing things to keep clear of."""
    moving = tuple(s for s, f in (("left", left), ("right", right)) if f)
    return robot.move(left, right, speed, check_only, avoid=obstacles(moving) or None)


def heading(side, xy):
    """Unit table direction from the arm's base to xy (cm)."""
    d = np.asarray(xy[:2], dtype=float) - scene().base_xy(side) * 100
    return d / max(np.linalg.norm(d), 1e-6)


def tips(grip, head, pitch, off):
    """Where the fingertips are when the grip point is at grip (cm), the fingers pointing along head, tilted pitch;
    off: the grip point from the fingertips, (cm back along the fingers, cm across to their right)."""
    p = np.radians(pitch)
    a = np.array([np.cos(p) * head[0], np.cos(p) * head[1], -np.sin(p)])
    right = np.array([head[1], -head[0], 0.0])
    return np.asarray(grip, dtype=float) + a * off[0] - right * off[1]


def fixed_face_cm(depth):
    """How far right of the fingertips' point the fixed finger's inner face is, at a depth (cm) in from the
    fingertips: it slants out a little toward the palm (the mount's CAD: 3.5 cm at the tip, 4.0 at 5 cm in)."""
    return 3.5 + 0.11 * depth


def grip_depth(wide, h, pitch):
    """How deep (cm in from the fingertips) a grasp puts the middle of an object wide cm across, held at h cm above
    the table: as deep as it fits before the palm, and for a steep grasp no deeper than keeps the fingertips off the
    table."""
    d = min(DEEP_CM, PALM_CM - PALM_GAP_CM - wide / 2, WIDE_DEEP_CM if wide > WIDE_CM else DEEP_CM)
    if pitch > 10:  # a low thing keeps them a third of its height up
        d = min(d, (h - min(TIP_ABOVE_CM, h / 3)) / np.sin(np.radians(pitch)))
    return max(d, 0.3)


def kf(grip, head, pitch, yaw=None, off=None, **extra):
    t = tips(grip, head, pitch, off or (tools.GRIP_BACK_CM, 0.0))
    k = {"xy": [round(float(t[0]), 2), round(float(t[1]), 2)], "z": round(float(t[2]), 2), "pitch": float(pitch)}
    if yaw is not None:
        k["yaw"] = float(yaw) % 180
    k.update(extra)
    return k


def arm_heading(side, arm):
    """The way an arm's fingers point now, flat on the table."""
    f, _, _ = tools.nudge_frame(side, arm["joints"])
    return f[:2] / max(np.linalg.norm(f[:2]), 1e-6)


def grip_point(side, arm, off=None):
    """Where a grip point is now, cm: off (back, across) from the fingertips, default the arm's."""
    _, g, _ = tools.frame(side, arm["joints"], off)
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
def reach(side, g, head, p, yaw, speed="slow", off=None):
    """The grip point to g, and there for real: a stretched arm's joints settle a few degrees short of their goals, so
    it's sent again past g by what's still missing, up to CORRECT times. Measured from the joints' encoders. off: the
    grip point's offset (default the arm's)."""
    off = off or tools.grip_offset(side)
    arms = yield Stage([kf(g, head, p, yaw, off)], speed)
    for _ in range(CORRECT):
        miss = np.asarray(g, dtype=float) - grip_point(side, arms[side], off)
        if np.linalg.norm(miss) < MISS_CM:
            break
        miss *= min(1.0, MAX_FIX_CM / np.linalg.norm(miss))
        again = kf(np.asarray(g) + miss, head, p, yaw, off)
        if rmove(**{side: [again]}, check_only=True).get("outcome") != "done":
            break  # past what it reaches: it stays as close as it got
        arms = yield Stage([again], speed)
    return arms


def grasp(side, arms, object, height=None, pitch=None, lift=5.0, offset=None, force=None):
    """The object's middle (at height cm above the table; default its middle) deep into the jaws, the fingers along
    the line from the arm's base, the fixed finger just beside it: open wide, come down in front of it (steep: over
    it), go in, close with the capped force, hold, lift. Fails (open again, backed off) when the jaws close on nothing
    or it slips on the lift."""
    o = find(object)
    _near.add(o["name"])
    c = np.array(o["center_cm"], dtype=float)
    H, wide = float(o["size_cm"][2]), float(min(o["size_cm"][:2]))
    h = float(o.get("z_cm", H / 2)) if height is None else float(height)
    head = heading(side, c)
    right = np.array([head[1], -head[0]])
    shift = np.zeros(3)
    if offset:
        f, r, u = (list(offset) + [0, 0, 0])[:3]
        shift = np.append(f * head + r * right, u)
    g = np.array([c[0], c[1], h]) + shift

    def aim(xy):
        """The object is at xy (cm), as a wrist camera saw it: grasp there, and remember it there."""
        nonlocal c, g
        c = np.array(xy[:2], dtype=float)
        g = np.array([c[0], c[1], h]) + shift
        o["center_cm"] = [float(c[0]), float(c[1])]
    yaw = None  # the wrist's roll stays upright (camera on top): the jaws close level, square to the fingers' line
    L = float(max(o["size_cm"][:2]))
    long_yaw = None  # from above: the jaws across a long thing, square to a box's faces (else they catch on its corners)
    if L > LONG * wide:
        long_yaw = (float(o.get("yaw", 0.0)) + 90) % 180
    elif str(o.get("shape", "")).startswith("flat"):
        across = (np.degrees(np.arctan2(head[1], head[0])) + 90) % 180  # where the jaws lie by themselves
        face = min(((float(o.get("yaw", 0.0)) + k) % 180 for k in (0, 90)), key=lambda y: abs(kin.wrap90(y - across)))
        long_yaw = face if abs(kin.wrap90(face - across)) > 8 else None
    width = min(scene().stroke_mm, wide * 10 + WIDE_MM)
    pct = grip_pct(force)
    head3 = np.append(head, 0.0)

    def fit(p):
        """Where the object sits in the jaws at pitch p: (cm in from the fingertips, cm right of their point)."""
        d = grip_depth(wide, h, p)
        return d, fixed_face_cm(d) - wide / 2 - (WIDE_GAP_CM if wide > WIDE_CM else FIT_GAP_CM)

    def approach(p, off):
        """Steep: the open jaws come straight down round it from above its top. Flatter: down in front of it with
        the fingertips clear of it, then in along the fingers."""
        if p >= STEEP:  # the jaws open only once there: travelling open, the swinging jaw hits things
            above = max(OVER_CM, H - h + 1.5 + off[0] * np.sin(np.radians(p)))  # the fingertips clear of its top
            return [kf(g + [0, 0, above], head, p, yaw, off), {"width": width}, kf(g, head, p, yaw, off)]
        down = g - head3 * (wide / 2 + GAP_CM + off[0] * np.cos(np.radians(p)))
        return [kf(down + [0, 0, OVER_CM], head, p, yaw, off), {"width": width}, kf(down, head, p, yaw, off),
                kf(g, head, p, yaw, off)]

    # tall: from the side, as level as reaches; otherwise the steepest that reaches
    order = PITCHES if H < TALL_CM else tuple(p for p in reversed(PITCHES) if p <= TALL_PITCH)
    near = []  # what stands beside it where the swinging jaw (left of the fingers) opens: (name, cm left of its middle
    for other in tools._memory["objects"]:  # to the other's near side)
        if other["name"] == o["name"] or other.get("dropped_into") or float(other["size_cm"][2]) < h - 2.0:
            continue
        v = np.asarray(other["center_cm"], dtype=float) - c
        r = float(max(other["size_cm"][:2])) / 2
        if v @ head3[:2] < -(r + 2.0) or v @ head3[:2] > r + 2.0:
            continue  # in front of or behind the fingers
        near.append((other["name"], float(-(v @ right)) - r))
    why = "out of reach"
    for p in ([pitch] if pitch is not None else order):
        off = fit(p)
        yaw = long_yaw if long_yaw is not None and p >= STEEP else None
        if yaw is None and near:  # how far left of the object's middle the open jaw reaches
            for w_try in (width, min(width, wide * 10 + TIGHT_MM)):
                reach_left = w_try / 10 - fixed_face_cm(off[0]) + off[1] + JAW_CM
                hit = [n for n, room in near if -2.0 < room < reach_left]
                if not hit:
                    width = w_try
                    break
            else:
                raise Failed(f"{side}'s swinging jaw opens on the left of its fingers and would hit {hit[0]} next to "
                             f"{object}: take {object} with the other arm, or move {hit[0]} first")
        if yaw is not None:  # the wrist turns to lie the jaws across it: which side the fixed finger ends up on is the
            off = (off[0], 0.0)  # roll robotd picks, so it goes in centred
        frames = approach(p, off)
        checked = rmove(**{side: frames}, check_only=True)
        if checked.get("outcome") == "done":
            break
        if checked.get("why") and ("would" in checked["why"] or "can't reach" not in why):
            why = checked["why"]  # something in the way says more than a pitch that doesn't reach
    else:  # the last pitch's reason: often not the reach
        raise Failed(f"{side} can't grasp {object} at {h:.0f} cm high" + ("" if pitch is None else f" at pitch {pitch}")
                     + f": {why}" + ("" if H < TALL_CM or pitch is not None else
                                     f" (a thing this tall only level-ish, up to {TALL_PITCH}: try the other arm, or it "
                                     "farther from this arm's base)"))
    tools._pitch[side] = float(p)
    yield Stage(frames[:-1])
    looked = []  # how far each wrist look moved the object, cm

    def look_and_aim():
        """From where the jaws wait open, the wrist camera finds the object; when it is LOOK_FIX_CM or more from where
        the grasp aims, aim there and wait open at the new spot. False when it can't be seen or the new spot can't be
        reached (the grasp then goes where it aimed)."""
        nonlocal frames
        seen = wrist.locate(side, o) if tools.FIND else None  # SAM's on in a run of Muse's; a rehearsal has no looks
        looked.append(None if seen is None else seen[1])
        if seen is None or seen[1] < LOOK_FIX_CM:
            return False
        old = (c.copy(), g.copy())
        aim(seen[0])
        again = approach(p, off)
        if rmove(**{side: again}, check_only=True).get("outcome") != "done":
            aim(old[0])
            return False
        frames = again
        yield Stage(frames[:-1], "slow")
        return True

    for _ in range(LOOKS):  # the last centimetres by the wrist camera: a few cm away, it sees what the overhead missed
        if not (yield from look_and_aim()):
            break
    yield from reach(side, g, head, p, yaw, off=off)
    regrips = 0
    while True:
        yield Stage([{"width": 0}], "slow", lambda: robot.grip_force(side, pct))
        time.sleep(tools.GRIP_SETTLE_S)
        w = robot.status()["arms"][side]["width_mm"]
        if w >= EDGE * wide * 10 or regrips == REGRIPS:
            break
        if w < tools.HOLD_MIN_MM:  # caught nothing: back out to where the jaws waited, look again, aim there
            regrips += 1
            yield Stage([{"width": width}], "slow")
            yield Stage([[f for f in frames[:-1] if "xy" in f][-1]], "slow")
            yield from look_and_aim()
            yield from reach(side, g, head, p, yaw, off=off)
            continue
        # Narrower than the object: the jaws caught its near edge, it sits farther along the fingers than the
        # numbers said. The chord says how far.
        r = wide / 2
        deeper = r if w < tools.HOLD_MIN_MM else float(np.sqrt(max(r * r - (w / 20) ** 2, 0.0)))
        g = g + head3 * min(deeper + 0.5, r)
        regrips += 1
        yield Stage([{"width": width}], "slow")
        yield from reach(side, g, head, p, yaw, off=off)
    if w < tools.HOLD_MIN_MM:
        yield Stage([{"width": width}, frames[0]])
        raise Failed(f"{side} closed on nothing at {object} (jaws shut to {w:.0f} mm, {regrips} re-grips): opened "
                     "and backed off")
    off = (off[0], fixed_face_cm(off[0]) - wide / 2)  # the jaw pushed it against the fixed finger
    tools._grip[side] = off
    o.pop("placed", None)  # it moves with the arm now
    gp = grip_point(side, robot.status()["arms"][side])  # where the jaws hold it across, as the arm really is: robotd
    # carries it from there, not from where the grasp aimed. Its height stays its own: a grasp can close above its middle
    robot.hold(side, o["name"], o["size_cm"], [float(gp[0]), float(gp[1])], o.get("z_cm", H / 2), o.get("yaw", 0.0))
    yield Stage([{"width": max(0.0, w - tools.GRIP_HOLD_MM)}, {"rel": [0, 0, float(lift)], "pitch": float(p)}])
    ma, w2 = current(side), robot.status()["arms"][side]["width_mm"]
    # the jaws keep closing at the cap: narrower after the lift than where they stopped on it means it slipped out
    if ma < HOLD_MA or w2 < tools.HOLD_MIN_MM or w2 < w - SLIP_MM:
        robot.hold(side)
        tools._grip.pop(side, None)
        yield Stage([{"width": width}])
        raise Failed(f"{side} lost {object} on the lift (jaws closed at {w:.0f} mm on a {wide * 10:.0f} mm object, "
                     f"now {w2:.0f} mm, motor {ma:.0f} mA; {regrips} re-grips): opened")
    return dict(holding=o["name"], jaws_mm=round(w, 1), object_mm=round(wide * 10), force=pct, squeeze_ma=round(ma),
                pitch=p, regrips=regrips, depth_cm=round(off[0], 1),
                **({"wrist_moved_it_cm": looked} if any(looked) else {}))


def grip_pct(force):
    """How hard the jaws squeeze, % of the motor's torque: the default, or what Muse asked for (a soft thing squashes
    at the default)."""
    return tools.GRIP_PCT if force is None else float(np.clip(force, MIN_FORCE, MAX_FORCE))


def below_grip(side, a):
    """How far the bottom of what an arm holds is below its grip point, cm (0 holding nothing)."""
    held = a.get("held")
    return float(grip_point(side, a)[2] - held["bottom_cm"][2]) if held else 0.0


def reaching_pitch(side, goal, head, p, yaw, off=None):
    """p if the grip point reaches goal at that pitch, else the nearest pitch that does, else p (the move then says
    why)."""
    off = off or tools.grip_offset(side)
    for q in sorted(range(0, 91, 10), key=lambda q: (abs(q - p), q)):  # the nearest first, then on to level or down
        q = float(q) if abs(q - p) > 2 else float(p)
        if rmove(**{side: [kf(goal, head, q, yaw, off)]}, check_only=True).get("outcome") == "done":
            return q
    return p


def holds_tall(arm):
    held = arm.get("held")
    return bool(held) and float(max(held["size_cm"])) >= TALL_CM


def move(side, arms, to=None, above=None, by=3.0, rel=None, pitch=None, yaw=None, speed=None):
    """The grip point (what the gripper holds) to `to` [x, y, z], or what it holds (or the grip point) so its bottom is
    `by` cm above the top of the object `above` (held by an arm or standing), or shifted by rel [dx, dy, dz]; at pitch
    (default: the one it has, or the nearest that reaches there) and yaw (default: the wrist's roll as it is)."""
    a = arms[side]
    p = sent_pitch(side, a) if pitch is None else float(pitch)
    g0 = grip_point(side, a)
    if rel is not None:
        goal = g0 + np.array(rel, dtype=float)
    elif to is not None:
        goal = np.array(to, dtype=float)
    elif above is not None:
        held_by = [x["held"]["name"] for x in arms.values() if (x.get("held") or {}).get("name") == above]
        name = held_by[0] if held_by else find(above)["name"]
        _over.add(name)
        top, _ = target_of(above, arms)
        goal = top + [0, 0, float(by) + below_grip(side, a)]
    else:
        raise Failed("move needs to, above or rel")
    head = heading(side, goal) if np.linalg.norm(goal[:2] - scene().base_xy(side) * 100) > 8 else \
        arm_heading(side, a)
    out = {}
    free = pitch is None and not holds_tall(a)  # turning the wrist tips a tall held thing over
    via = None
    if above is not None:
        q, via = over(side, g0, goal, head, p, yaw, free, name)
    else:
        q = reaching_pitch(side, goal, head, p, yaw) if free else p
    if q != p:
        out["pitch"] = f"{p:.0f} doesn't reach there: moved at {q:.0f}"
        p = q
    tools._pitch[side] = p
    if via is not None:
        yield Stage([kf(via, head, p, yaw, tools.grip_offset(side))], speed or "normal")
    yield from reach(side, goal, head, p, yaw, speed or "normal")
    return out


def over(side, g0, goal, head, p, yaw, free, name):
    """How a move comes to goal above `name`: the pitch (p; free: the nearest that works) and a point to go through
    first, the first of these the arm can do: up where the gripper is, then across (never through what it goes above);
    in from this arm's side of it at the goal's height; straight."""
    off = tools.grip_offset(side)
    vias = [np.append(g0[:2], goal[2])] if goal[2] > g0[2] + 1.0 else []
    ob = next((o for o in obstacles((side,)) if o["name"] == name), None)
    if ob is not None:
        c = np.asarray(ob["xy"], dtype=float)
        for d in range(1, 31):
            xy = goal[:2] - head * d
            if np.linalg.norm(xy - c) >= ob["r"] + VIA_CM:
                vias.append(np.append(xy, goal[2]))
                break
    pitches = sorted(range(0, 91, 10), key=lambda q: (abs(q - p), q)) if free else [p]
    for q in pitches:  # the nearest first, then on to level or down
        q = float(q) if abs(q - p) > 2 else float(p)
        for v in vias + [None]:
            keys = ([kf(v, head, q, yaw, off)] if v is not None else []) + [kf(goal, head, q, yaw, off)]
            if rmove(**{side: keys}, check_only=True).get("outcome") == "done":
                return q, v
    return p, None  # nothing works: the move says why


def place(side, arms, at=None, on=None, speed=None):
    """Set down what the arm holds: carried over `at` [x, y] (on the table) or onto the object `on` (on its top), then
    lowered until its bottom is SET_ABOVE_CM above where it lands, let go, and the fingers back off along their own
    line (out of a side grasp, up out of one from above)."""
    a = arms[side]
    if not a.get("held"):
        raise Failed(f"{side} holds nothing to place")
    if at is not None:
        xy, floor = np.array(at[:2], dtype=float), 0.0
        mine = float(max(a["held"]["size_cm"][:2])) / 2
        here = np.asarray(a["held"]["center_cm"][:2], dtype=float)
        for o in tools._memory["objects"]:  # setting something down right beside a standing thing knocks it over
            if o["name"] == a["held"]["name"] or o.get("dropped_into") or o.get("held_by"):
                continue
            if np.linalg.norm(np.asarray(o["center_cm"], dtype=float) - here) < mine:
                continue  # seen inside what it holds (the camera can't see under it): it goes down with it
            need = mine + float(max(o["size_cm"][:2])) / 2 + CLEAR_CM
            gap = float(np.linalg.norm(xy - np.asarray(o["center_cm"], dtype=float)))
            if gap < need:
                raise Failed(f"[{xy[0]:.0f}, {xy[1]:.0f}] is {gap:.0f} cm from the middle of {o['name']}: setting "
                             f"{a['held']['name']} down there needs {need:.0f}; pick a free spot")
    elif on is not None:
        _near.add(find(on)["name"])
        top, _ = target_of(on, arms)
        xy, floor = top[:2], float(top[2])
    else:
        raise Failed("place needs at [x, y] or on NAME")
    drop = below_grip(side, a)
    g0 = grip_point(side, a)
    low = np.append(xy, floor + drop + SET_ABOVE_CM)
    head = heading(side, low) if np.linalg.norm(xy - scene().base_xy(side) * 100) > 8 else arm_heading(side, a)
    p = sent_pitch(side, a) if holds_tall(a) else reaching_pitch(side, low, head, sent_pitch(side, a), None)
    for _ in range(6):  # turned to another pitch, what it holds hangs lower (robotd holds it as a box): a bit higher
        r = rmove(**{side: [kf(low, head, p, None, tools.grip_offset(side))]}, check_only=True)
        if r.get("outcome") == "done" or "hit the table" not in str(r.get("why")):
            break
        low = low + [0, 0, 0.5]
    high = np.append(xy, max(g0[2], low[2] + 3.0))
    tools._pitch[side] = p
    arms = yield from reach(side, high, head, p, None, speed or "normal")
    arms = yield from reach(side, low, head, p, None, "slow")
    return (yield from let_go(side, arms[side]))


def put_in(side, arms, into, by=3.0):
    """Let what the arm holds drop into the object `into` (an opening on its top): held with its bottom by cm above the
    rim (a stretched arm sags a cm or two below where it's sent), let go, back off. Check the images after: something
    can land on the rim."""
    a = arms[side]
    if not a.get("held"):
        raise Failed(f"{side} holds nothing to put in")
    moved = yield from move(side, arms, above=into, by=by)
    yield Stage([], before=lambda: time.sleep(0.3))  # still before it opens
    held_by = [x["held"]["name"] for x in robot.status()["arms"].values() if (x.get("held") or {}).get("name") == into]
    note = yield from let_go(side, robot.status()["arms"][side], into=held_by[0] if held_by else find(into)["name"])
    return dict(moved, **note)


def let_go(side, arm, into=None):
    """Open, forget what was held (it's remembered where it was let go, or as dropped into `into`), and back the
    fingers out along their own line and up."""
    yield Stage([{"width": scene().stroke_mm}], "slow")
    if arm.get("held"):
        _near.add(arm["held"]["name"])
    # out far enough to clear it before the jaws close again
    clear = tools.grip_offset(side)[0] + (float(max(arm["held"]["size_cm"][:2])) / 2 if arm.get("held") else 0.0) + 1.5
    out = max(BACK_OFF_CM, clear)
    left_at(side, arm, into)
    robot.hold(side)
    tools._grip.pop(side, None)
    f, _, _ = tools.nudge_frame(side, arm["joints"])
    p0 = sent_pitch(side, arm)
    # back out along the fingers and up; close to a base that can be out of reach: then more up, the wrist tipped down
    ways = ((out, 2.0, 0), (out, 5.0, 0), (out - 1.0, 5.0, 20), (out - 2.0, 7.0, 40))
    if into:  # it fell into something below: straight up is clear
        ways = ((0.0, 4.0, 0), (1.0, 4.0, 0)) + ways
    for back, up, tip in ways:
        p = np.radians(p0)
        d = -np.array([np.cos(p) * f[0], np.cos(p) * f[1], -np.sin(p)]) * back + [0, 0, up]
        kf_ = {"rel": [round(float(v), 2) for v in d], "pitch": float(min(90.0, p0 + tip))}
        if rmove(**{side: [kf_]}, check_only=True).get("outcome") == "done":
            tools._pitch[side] = kf_["pitch"]
            yield Stage([kf_], "slow")
            yield Stage([{"width": TRAVEL_MM}])  # nearly shut while it travels: the open jaw sticks out
            return {}
    return {"note": "let go, but the fingers can't back out from here: they are still round it; move up or park "
                    "before the other arm comes near"}


def tilt(side, arms, deg):
    """Turn the wrist by deg: what the gripper holds tilts about the fingers' direction."""
    yield Stage([{"roll": float(deg)}], "slow")
    return {}


def left_at(side, arm, into=None):
    """What the arm held is remembered where it was let go, so the next look knows it by its name; into: it was dropped
    into that object (the camera may not see it again: the state says where it went)."""
    held = arm.get("held")
    for o in tools._memory["objects"] if held else []:
        if o["name"] == held["name"]:
            o.update(center_cm=[round(float(v), 1) for v in held["center_cm"][:2]], seen=tools._memory["looks"],
                     placed=True)
            if into:
                o["dropped_into"] = into


def open_(side, arms, width=None):
    a = arms[side]
    if holds_tall(a) and "bottom_cm" in a["held"] and lowest_cm(a["held"]) > DROP_TALL_CM:
        raise Failed(f"{side} holds {a['held']['name']} {lowest_cm(a['held']):.0f} cm above the table: let go there, a "
                     f"tall thing falls over; set it down with place (at a free spot, or on something)")
    yield Stage([{"width": scene().stroke_mm if width is None else float(width)}])
    left_at(side, arms[side])
    robot.hold(side)
    tools._grip.pop(side, None)
    return {}


def grip(side, arms, force=None):
    """Close with the capped force where the gripper is, and hold what's between the jaws."""
    pct = grip_pct(force)
    yield Stage([{"width": 0}], "slow", lambda: robot.grip_force(side, pct))
    time.sleep(tools.GRIP_SETTLE_S)
    a = robot.status()["arms"][side]
    if a["width_mm"] < tools.HOLD_MIN_MM:
        yield Stage([{"width": scene().stroke_mm}])
        raise Failed(f"{side} closed on nothing: opened again")
    o = tools.at_grip(side, a)
    if o is not None:
        g = grip_point(side, a)
        robot.hold(side, o["name"], o["size_cm"], [float(g[0]), float(g[1])], o.get("z_cm", o["size_cm"][2] / 2),
                   o.get("yaw", 0.0))
    yield Stage([{"width": max(0.0, a["width_mm"] - tools.GRIP_HOLD_MM)}])
    return dict(holding=o["name"] if o else "something", jaws_mm=round(a["width_mm"], 1), force=pct,
                squeeze_ma=round(current(side)))


def nudge(side, arms, forward=0.0, right=0.0, up=0.0):
    """A small step in the gripper's own directions (see tools.nudge), keeping its tilt."""
    a = arms[side]
    f, r, u = tools.nudge_frame(side, a["joints"])
    d = float(forward or 0) * f + float(right or 0) * r + float(up or 0) * u
    yield Stage([{"rel": [round(float(v), 2) for v in d], "pitch": sent_pitch(side, a)}], "slow")
    return {}


def park(side, arms, open=False):
    """The ready pose (Scene.ready): raised, level, pointing ahead, the wrist camera on top, where the overhead camera
    sees the table past it; the wrist turns back upright first if a tilt left it turned. open: jaws open too."""
    q = scene().ready(side)
    x, y, z = scene().arms[side].tool(q)[0] * 100
    tools._pitch[side] = 0.0
    frames = []
    turn = float(q[4] - arms[side]["joints"][4])
    if abs(turn) > 2:
        frames.append({"roll": round(turn, 1)})
    shut = {} if arms[side].get("held") else {"width": TRAVEL_MM}
    frames.append(dict({"xy": [round(float(x), 2), round(float(y), 2)], "z": round(float(z), 2), "pitch": 0.0},
                       **({"width": scene().stroke_mm} if open else shut)))
    yield Stage(frames)
    return {}


ACTIONS = {"grasp": grasp, "move": move, "place": place, "put_in": put_in, "tilt": tilt, "open": open_, "grip": grip,
           "nudge": nudge, "park": park}


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
    _near.clear()
    _over.clear()
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
        r = rmove(frames.get("left"), frames.get("right"), speed) if frames else dict(outcome="done")
        if r.get("outcome") == "refused" and len(frames) == 2:  # nothing moved: is it one arm's part, or the pair?
            alone = {s: rmove(**{s: f}, check_only=True) for s, f in frames.items()}
            bad = [s for s, c in alone.items() if c.get("outcome") != "done"]
            if bad and len(bad) < 2:  # that arm's action fails; the other goes on by itself
                for s in bad:
                    results[s] = dict(failed=f"refused: {alone[s].get('why')}")
                    stage.pop(s)
                continue
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
