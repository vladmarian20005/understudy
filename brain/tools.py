"""What the brain sees and remembers: a fresh look at the table, the objects by name across looks, each arm's grip point,
and the state as text for the model.

Objects come from the overhead camera, which can't see under an arm: an object an arm covers keeps the position it
was last seen whole at, until the camera sees it whole again or it hasn't been seen for FORGET_LOOKS looks.
"""
import base64
import json
import re

import cv2
import numpy as np

from harness import overlay, robot
from harness.skills import HOLD_MIN_MM

from . import sam, wrist
from .reach import GRIP_BACK_CM, TALL_CM, reach_facts
from .rig import WRISTS, scene

# +1 if a nudge along the jaw line's direction moves the gripper toward the right of its wrist image. Both -1: image
# right is +x when an arm reaches +y.
RIGHT_SIGN = {"left": -1, "right": -1}
# grip closes with the gripper's torque capped at GRIP_PCT, so the jaws stop on whatever pushes back harder and can't
# squeeze harder than that.
GRIP_PCT = 30.0
GRIP_HOLD_MM = 100.0  # then the jaws are held this far inside where they stopped, still capped: all the way, so they
# keep squeezing at the cap. The original gripper's jaw swings in and pushes what it holds out toward the fingertips:
# held just inside where it stopped, it stops pushing and the thing slides out
GRIP_SETTLE_S = 0.4  # the jaws come to rest on it before the opening is read
GRIP_NEAR_CM = 4.0  # an object this close to the grip point is the one being gripped
SHRUNK = 0.6  # a detection this much smaller than the object was seen whole is an arm covering part of it
MATCH_CM = 5.0  # a detection this close to a remembered object is that object
ALIKE_CM = 3.0  # found by another word, it also has to be about as wide (this close)
PLACED_CM = 3.0  # something an arm set down stays where it was set unless the camera sees it farther than this
FORGET_LOOKS = 10

_memory = dict(looks=0, objects=[], counts={})  # objects as last seen whole, each with the look it was seen at
# what SAM looks for on the overhead image, set from the task (brain/plan.py); empty: robotd's colour detector
FIND = []
# side -> the pitch it was last sent to. A nudge keeps that, not the measured one: an arm stretched out sags a few
# degrees steeper than it's sent, and sending it the measured pitch ratchets it steeper at every nudge.
_pitch = {}
# side -> where the middle of what it holds sits in its jaws: (cm back from the fingertips along the fingers, cm to the
# right of the fingertips' point across them). A grasp puts each object as deep as it fits and against the fixed
# finger, so it isn't at the default grip point.
_grip = {}


def grip_offset(side):
    return _grip.get(side, (GRIP_BACK_CM, 0.0))


def reset():
    """Forget the objects: a new run starts from what the camera sees."""
    _memory.update(looks=0, objects=[], counts={})
    _pitch.clear()
    _grip.clear()


def new_name(kind, alone):
    """A name for an object seen for the first time: its kind, numbered when there's more than one of that kind; never
    a name another object has (the colour detector's and SAM's can meet)."""
    taken = {o["name"] for o in _memory["objects"]}
    while True:
        k = _memory["counts"][kind] = _memory["counts"].get(kind, 0) + 1
        name = kind if k == 1 and alone else f"{kind} {k}"
        if name not in taken:
            return name


def kind_of(d):
    """What a detection is: the word SAM found it by, else the colour detector's name without its number."""
    return d.get("found_as") or re.sub(r"\s*#?\d+$", "", d["name"]) or d["name"]


def alike(o, d):
    """Whether a detection found by another word can be the remembered object: the same build (round, standing, flat)
    and about as wide."""
    build = lambda x: str(x.get("shape", "")).split(",")[0].split(" ")[0]
    return build(o) == build(d) and abs(min(o["size_cm"][:2]) - min(d["size_cm"][:2])) < ALIKE_CM


def remember(objects, held=()):
    """Update the remembered objects with a look's detections. Each keeps its name for the whole run: a detection is
    the remembered object nearest it within MATCH_CM, whatever word found it this time (else a plan's names stop
    resolving), the same kind first; held: names of objects a gripper holds now (they move with it, so they aren't
    matched to the table)."""
    _memory["looks"] += 1
    n, kept = _memory["looks"], []
    pool = [o for o in _memory["objects"] if o["name"] not in held]
    objects = sorted((d for d in objects if not d.get("held_by")),  # in a gripper: not on the table
                     key=lambda d: d["center_cm"][0])
    for d in objects:
        kind = kind_of(d)
        dist = lambda o: float(np.linalg.norm(np.subtract(o["center_cm"], d["center_cm"])))
        reach = MATCH_CM * (2 if d.get("hidden") else 1)  # partly behind an arm: its position is rougher
        old = [o for o in pool if dist(o) < reach and (o.get("kind", o["name"]) == kind or alike(o, d))]
        o = min(old, key=lambda o: (o.get("kind", o["name"]) != kind, dist(o)), default=None)
        if o is not None:
            d = dict(d, kind=o.get("kind", kind), name=o["name"], **({"dropped_into": o["dropped_into"]}
                                                                    if "dropped_into" in o else {}))
            pool.remove(o)
            _memory["objects"].remove(o)
            if np.prod(d["size_cm"][:2]) < SHRUNK * np.prod(o["size_cm"][:2]) or d.get("hidden"):  # partly covered:
                kept.append(o)  # where, and when, it was last seen whole
                continue
            if o["size_cm"][2] > d["size_cm"][2]:  # it doesn't get shorter: a blob that isn't round enough reads as
                # the default height
                d = dict(d, size_cm=d["size_cm"][:2] + o["size_cm"][2:], z_cm=o.get("z_cm", o["size_cm"][2] / 2))
            if o.get("placed") and dist(o) < PLACED_CM:  # an arm set it down there: that's where it is, better than
                d = dict(d, center_cm=o["center_cm"], placed=True)  # the camera's guess
            if o.get("corrected"):  # the model's size from the images: better than one slanted camera's outline
                d = dict(d, size_cm=o["size_cm"], z_cm=o["z_cm"], shape=o["shape"], corrected=True)
            if o.get("measured"):  # its height came from a wrist camera: better than the overhead's "at least"
                d = dict(d, size_cm=d["size_cm"][:2] + o["size_cm"][2:], z_cm=o["z_cm"], shape=o["shape"],
                         measured=True)
        else:
            r_d = float(max(d["size_cm"][:2])) / 2
            r_t = lambda t: float(max(t["size_cm"][:2])) / 2
            part = [t for t in pool + kept
                    if (float(t["size_cm"][2]) >= TALL_CM and d.get("shape") != "round" and dist(t) < r_t(t) + r_d - 0.5)
                    or (dist(t) < r_t(t) and r_d < r_t(t))]
            if part:  # overlapping a tall standing thing and not round, or smaller and inside something's footprint: a
                continue  # piece of it
            d = dict(d, kind=kind, name=new_name(kind, sum(kind_of(e) == kind for e in objects) == 1))
        kept.append(dict(d, seen=n))
    stale = [o for o in _memory["objects"] if n - o["seen"] < FORGET_LOOKS]  # not seen this look: an arm covers it?
    _memory["objects"] = kept + stale


def correct(fixes):
    """The model's corrections of listed objects, by name: {"size_cm": [L, W, H], "center_cm": [x, y]}, from what the
    images show: one slanted camera misreads small things' heights and can list a sliver of something as the whole. A
    corrected size holds over later
    looks; a corrected spot holds until the camera sees the object PLACED_CM or more away. Returns what was done."""
    out = []
    for name, fix in (fixes or {}).items():
        o = next((o for o in _memory["objects"] if o["name"] == name), None)
        if o is None or not isinstance(fix, dict):
            out.append(f"{name}: no such object")
            continue
        if fix.get("size_cm"):
            size = [float(v) for v in fix["size_cm"]]
            L, W, H = (size + size[-1:] * 3)[:3]
            tall = H >= TALL_CM or H >= 1.5 * max(L, W)
            shape = o.get("shape", "") if str(o.get("shape", "")).startswith("round") else "upright" if tall else "flat"
            o.update(size_cm=[L, W, H], z_cm=H / 2, corrected=True, shape=f"{shape.split(' ')[0]} (size as corrected)")
        if fix.get("center_cm"):
            o.update(center_cm=[float(v) for v in fix["center_cm"][:2]], placed=True, seen=_memory["looks"])
        out.append(f"{name}: " + ", ".join(f"{k} {fix[k]}" for k in ("size_cm", "center_cm") if fix.get(k)))
    return out


def frame(side, joints, grip=None):
    """An arm's fingertips (m), grip point (m; grip: its (back, across) from them, default the arm's), and the table
    directions (unit, 3D) of a nudge's forward, right and up: right along the jaw line, toward the right of the wrist
    image; forward square to it, the way the fingers point; up straight up."""
    arm = scene().arms[side]
    p, R = arm.tool(joints)
    approach = -R[:, 2]
    flat = approach[:2]
    heading = flat if np.linalg.norm(flat) > 0.25 else p[:2] - scene().base_xy(side)  # straight down: outward
    jaw = (R @ arm.jaw)[:2]
    if np.linalg.norm(jaw) > 0.2:
        right = RIGHT_SIGN[side] * jaw / np.linalg.norm(jaw)
    else:  # the jaw line on end (the wrist rolled over at a level tilt): right of the heading
        right = np.array([heading[1], -heading[0]]) / np.linalg.norm(heading)
    forward = np.array([-right[1], right[0]])
    if forward @ heading < 0:
        forward = -forward
    axes = np.append(forward, 0.0), np.append(right, 0.0), np.array([0.0, 0.0, 1.0])
    back, across = grip or grip_offset(side)
    side_of = np.array([heading[1], -heading[0], 0.0]) / np.linalg.norm(heading)  # right of where the fingers point
    return p, p - approach * back / 100 + side_of * across / 100, axes


def nudge_frame(side, joints):
    return frame(side, joints)[2]


def state_text(seen):
    """The arms' poses and the objects, each with its offset from each grip point, as compact text."""
    arms, grips = {}, {}
    for s, a in seen.get("arms", {}).items():
        arms[s] = {k: a[k] for k in ("tip_cm", "pitch", "yaw", "width_mm") if k in a}
        _, g, axes = frame(s, a["joints"])
        arms[s]["grip_point_cm"] = [round(float(v) * 100, 1) for v in g]
        if "grip_ma" in a:  # the gripper motor squeezes something above about 120 mA, idles at 50-80
            arms[s]["grip_ma"] = a["grip_ma"]
        if a.get("held"):
            arms[s]["holding"] = {k: a["held"][k] for k in ("name", "center_cm", "top_cm", "bottom_cm", "size_cm")}
        grips[s] = (g * 100, axes)
    held = {a["held"]["name"]: s for s, a in seen.get("arms", {}).items() if a.get("held")}
    objs = []
    for o in _memory["objects"]:
        if o["name"] in held:
            objs.append(dict(name=o["name"], held_by=held[o["name"]]))
            continue
        c = np.array(o["center_cm"] + [o.get("z_cm", o["size_cm"][2] / 2)], dtype=float)
        offsets = {s: [round(float((c - g) @ ax), 1) for ax in axes] for s, (g, axes) in grips.items()}
        if str(o.get("shape", "")).startswith("upright"):
            o = dict(o, shape=o["shape"].replace("upright", "standing upright") + f": its top is at z {o['size_cm'][2]:.0f}")
        objs.append(dict({k: o[k] for k in ("name", "color", "center_cm", "size_cm", "shape", "dropped_into") if k in o},
                         seen="now" if o["seen"] == _memory["looks"] else
                         f"{_memory['looks'] - o['seen']} looks ago (covered by an arm, or gone)", from_grip=offsets,
                         reach=reach_facts(o)))
    return json.dumps(dict(arms=arms, objects=objs))


def at_grip(side, arm):
    """The remembered object nearest the arm's grip point, if one is within GRIP_NEAR_CM of it."""
    _, g, _ = frame(side, arm["joints"])
    near = [(np.linalg.norm(np.array(o["center_cm"] + [o.get("z_cm", o["size_cm"][2] / 2)]) - g * 100), o)
            for o in _memory["objects"]]
    d, o = min(near, key=lambda t: t[0], default=(None, None))
    return o if d is not None and d < GRIP_NEAR_CM else None


def label(o):
    """An object's name, and for a standing one how tall it is: perspective makes a tall thing look like it lies."""
    if str(o.get("shape", "")).startswith("upright"):
        return f"{o['name']} (standing, {o['size_cm'][2]:.0f}{'+' if 'least' in o['shape'] else ''} cm tall)"
    return o["name"]


def look(cams=("overhead",)):
    """Fresh images and positions. With FIND set, the objects are SAM's (by name, any colour) and the overhead image
    gets the grid, the grippers and their names drawn on; otherwise robotd's colour detector and overlay."""
    if not FIND:
        seen = robot.look(list(cams))
        remember(seen.get("objects", []))
    else:
        seen = robot.look(list(cams), annotate=False)
        arms = seen.get("arms", {})
        img = cv2.imdecode(np.frombuffer(base64.b64decode(seen["images"]["overhead"]), np.uint8), cv2.IMREAD_COLOR)
        held = {s: a["held"] for s, a in arms.items() if a.get("held")}
        outline = []  # points over a held thing's outline: SAM's pieces of it aren't things on the table
        for h in held.values():
            r = float(min(h["size_cm"][:2])) / 2 * 0.8
            for k in np.linspace(0, 1, 7):
                c = np.add(h["bottom_cm"], np.subtract(h["top_cm"], h["bottom_cm"]) * k)
                outline += [(c + [dx, dy, 0]).tolist() for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r))]
        held_px = robot.to_pixels(outline) if held else []
        q = {s: list(a["joints"]) + [a["width_mm"]] for s, a in arms.items()}
        try:
            objs = sam.objects(scene(), img, FIND, q, held_px)
        except Exception as e:  # SAM down: what the colour detector sees, rather than nothing
            print(f"SAM failed ({e}): colour detector instead", flush=True)
            objs = seen.get("objects", [])
        remember(objs, {h["name"] for h in held.values()})
        for s, h in held.items():  # what a gripper holds is where the gripper has it
            for o in _memory["objects"]:
                if o["name"] == h["name"]:
                    o.update(center_cm=h["center_cm"][:2], z_cm=h["center_cm"][2], seen=_memory["looks"])
        fresh = [dict(id="", name=label(o), px=o["px"]) for o in _memory["objects"]
                 if o["seen"] == _memory["looks"] and o["name"] not in {h["name"] for h in held.values()}]
        ok, buf = cv2.imencode(".jpg", overlay.draw(img, scene(), fresh, arms), [cv2.IMWRITE_JPEG_QUALITY, 85])
        seen["images"]["overhead"] = base64.b64encode(buf.tobytes()).decode()
        seen["objects"] = objs
    sides = {c: s for s, c in WRISTS.items()}
    for cam, b64 in list(seen.get("images", {}).items()):
        if cam != "overhead":
            seen["images"][cam] = wrist.wrist_image(b64, sides[cam], seen.get("arms", {}).get(sides[cam]),
                                                    grip_offset(sides[cam]))
    return seen
