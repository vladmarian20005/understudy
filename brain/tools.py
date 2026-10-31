"""The tools a model drives the arms with: robotd's checked motions as OpenAI function schemas, and the code that runs
them.

No task skills: the model picks things up the way a person with a joystick and a camera on the gripper would, by
sight. move brings a gripper near something (table cm, read off the overhead grid), nudge steps it in the gripper's
own directions while the model watches the wrist camera, and grip and open work the jaws. Every call comes back as
text: the outcome, why it failed, where an arm was sent and where it got to, each arm's pose, and each object with
its offset from each gripper's grip point (where the jaws pinch) in that gripper's nudge directions. With it come a
fresh overhead image and the wrist camera of each arm that moved, with the line through the fingertips and the arm's
pose drawn on.

Objects come from the overhead camera, which can't see under an arm: an object an arm covers keeps the position it
was last seen whole at, until the camera sees it whole again or it hasn't been seen for FORGET_LOOKS looks.
"""
import base64
import json
import time

import cv2
import numpy as np

from harness import robot
from harness.scene import Scene
from harness.skills import HOLD_MIN_MM

ARM = {"type": "string", "enum": ["left", "right"]}
XY = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "table cm [x, y]"}
KEYFRAME = {
    "type": "object",
    "properties": {
        "xy": XY,
        "z": {"type": "number", "description": "fingertip height above the table, cm"},
        "rel": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                "description": "[dx, dy, dz] table cm from where the previous keyframe left the fingertips"},
        "yaw": {"type": "number", "description": "jaw line angle on the table, degrees (0 = along x); left out: kept"},
        "pitch": {"type": "number", "description": "90 straight down, 0 level; left out: the steepest that reaches"},
        "width": {"type": "number", "description": "gripper opening, mm (0-84)"},
    },
}
WRISTS = {"left": "left_wrist", "right": "right_wrist"}
WRIST_PX = (640, 480)  # wrist images go to the model at this size
# The line through the two fingertips in each wrist image, as fractions of its width and height: the jaws close
# below it. Measured 31 Oct from the pink fingertips; the jaws move in parallel, so it holds at any opening.
TIP_LINE = {"left_wrist": ((0.153, 0.454), (0.942, 0.479)), "right_wrist": ((0.163, 0.442), (0.833, 0.475))}
# The jaws' centre line, x as a fraction of the width: halfway between the open fingertips. Far from the bases the
# numbers are 1-2 cm off (31 Oct: from_grip said centred, a finger pushed the ball 1.5 cm aside), and the jaws have
# under 1 cm to spare round a ball: sideways the wrist image decides.
JAW_CENTRE = {"left_wrist": 0.548, "right_wrist": 0.498}
# +1 if a nudge along the jaw line's direction moves the gripper toward the right of its wrist image. Both -1 (31 Oct:
# nudged on the real arms, the table moved the other way in the image): image right is +x when an arm reaches +y.
RIGHT_SIGN = {"left": -1, "right": -1}
GRIP_BACK_CM = 2.0  # the grip point, where the jaws pinch: this far back from the fingertips along the fingers
SHORT_CM = 1.0  # an arm that ends farther than this from where it was sent is reported short
PITCH_GIVE = (0, -5, 5, -10, 10, -15, 15)  # deg: a nudge keeps the tilt if it can, else takes the nearest that reaches
# grip closes with the gripper's torque capped at GRIP_PCT, so the jaws stop on whatever pushes back harder and can't
# squeeze harder than that. 31 Oct: closing in position steps at the 70% the servo allows crushed a tennis ball 25 mm
# before anything noticed, and the fingers creaked.
GRIP_PCT = 30.0
GRIP_HOLD_MM = 4.0  # then the jaws are held this far inside where they stopped, still capped
GRIP_SETTLE_S = 0.4  # the jaws come to rest on it before the opening is read
GRIP_NEAR_CM = 4.0  # an object this close to the grip point is the one being gripped
SHRUNK = 0.6  # a detection this much smaller than the object was seen whole is an arm covering part of it
MATCH_CM = 5.0  # a detection this close to a remembered object of the same name is that object
FORGET_LOOKS = 10

_scene = None
_memory = dict(looks=0, objects=[])  # objects as last seen whole, each with the look it was seen at
# side -> the pitch it was last sent to. A nudge keeps that, not the measured one: an arm stretched out sags a few
# degrees steeper than it's sent, and sending it the measured pitch ratcheted it steeper at every nudge (31 Oct:
# 45 -> 66, where it no longer reached past a ball's middle).
_pitch = {}


def fn(name, description, props=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props or {}, "required": list(required)}}}


TOOLS = [
    fn("look", "Fresh images and positions without moving. overhead (default) is the table with the grid; wrists "
       "adds both wrist cameras.", {"cams": {"type": "string", "enum": ["overhead", "wrists"]}}),
    fn("move", "Coarse motion to table coordinates, checked before anything moves. left and right are lists of "
       "keyframes; the k-th keyframes of both arms are one phase and both arms finish each phase together. Values "
       "left out stay as they are, except pitch: give it, or the arm takes the steepest pitch that reaches, which "
       "can tilt the gripper a lot. check_only plans it without moving.",
       {"left": {"type": "array", "items": KEYFRAME}, "right": {"type": "array", "items": KEYFRAME},
        "speed": {"type": "string", "enum": ["slow", "normal", "fast"]}, "check_only": {"type": "boolean"}}),
    fn("nudge", "Step one gripper a little in its own directions, keeping its tilt and jaw angle: forward is the way "
       "the fingers point (flat, along the table), right is toward the right of its wrist image, up is straight "
       "up. cm, negative for back, left, down. The same directions as each object's offset from the grip point, "
       "so nudging by an object's offset brings the grip point onto it.",
       {"arm": ARM, "forward": {"type": "number"}, "right": {"type": "number"}, "up": {"type": "number"}},
       ["arm"]),
    fn("grip", "Close the jaws with a limited force until they stop on what's between them, and hold it. Says the "
       "opening they stopped at (and the object's width, when one is at the grip point); on nothing they open again.",
       {"arm": ARM}, ["arm"]),
    fn("open", "Open the jaws (to let go, or before reaching for something).",
       {"arm": ARM, "width": {"type": "number", "description": "opening, mm (default fully open, 84)"}}, ["arm"]),
    fn("home", "Park one or both arms out of the way (the start pose).",
       {"arms": {"type": "string", "enum": ["both", "left", "right"]},
        "width": {"type": "number", "description": "gripper opening, mm"}}),
    fn("done", "Finish: the task is done, or can't be done. Say honestly which, and why.",
       {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ["success", "summary"]),
]
NAMES = {t["function"]["name"] for t in TOOLS}


def scene():
    global _scene
    if _scene is None:
        _scene = Scene()
    return _scene


def reset():
    """Forget the objects: a new run starts from what the camera sees."""
    _memory.update(looks=0, objects=[])
    _pitch.clear()


def remember(objects):
    """Update the remembered objects with a look's detections."""
    _memory["looks"] += 1
    n, kept = _memory["looks"], []
    for d in objects:
        old = [o for o in _memory["objects"] if o["name"] == d["name"]
               and np.linalg.norm(np.subtract(o["center_cm"], d["center_cm"])) < MATCH_CM]
        o = min(old, key=lambda o: np.linalg.norm(np.subtract(o["center_cm"], d["center_cm"])), default=None)
        if o is not None:
            _memory["objects"].remove(o)
            if np.prod(d["size_cm"][:2]) < SHRUNK * np.prod(o["size_cm"][:2]):  # partly covered: where it was whole
                kept.append(o)
                continue
        if o is not None and o["size_cm"][2] > d["size_cm"][2]:  # it doesn't get shorter: a blob that isn't round
            # enough reads as the default height (31 Oct: a ball's middle put at 1.2 cm instead of 3.6)
            d = dict(d, size_cm=d["size_cm"][:2] + o["size_cm"][2:], z_cm=o.get("z_cm", o["size_cm"][2] / 2))
        kept.append(dict(d, seen=n))
    stale = [o for o in _memory["objects"] if n - o["seen"] < FORGET_LOOKS]  # not seen this look: an arm covers it?
    _memory["objects"] = kept + stale


def frame(side, joints):
    """An arm's fingertips (m), grip point (m), and the table directions (unit, 3D) of a nudge's forward, right and up:
    right along the jaw line, toward the right of the wrist image; forward square to it, the way the fingers point;
    up straight up."""
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
    return p, p - approach * GRIP_BACK_CM / 100, axes


def nudge_frame(side, joints):
    return frame(side, joints)[2]


def state_text(seen):
    """The arms' poses and the objects, each with its offset from each grip point, as compact text."""
    arms, grips = {}, {}
    for s, a in seen.get("arms", {}).items():
        arms[s] = {k: a[k] for k in ("tip_cm", "pitch", "yaw", "width_mm") if k in a}
        _, g, axes = frame(s, a["joints"])
        grips[s] = (g * 100, axes)
    objs = []
    for o in _memory["objects"]:
        c = np.array(o["center_cm"] + [o.get("z_cm", o["size_cm"][2] / 2)], dtype=float)
        offsets = {s: [round(float((c - g) @ ax), 1) for ax in axes] for s, (g, axes) in grips.items()}
        objs.append(dict({k: o[k] for k in ("name", "color", "center_cm", "size_cm") if k in o},
                         seen="now" if o["seen"] == _memory["looks"] else
                         f"{_memory['looks'] - o['seen']} looks ago (covered by an arm, or gone)", from_grip=offsets))
    return json.dumps(dict(arms=arms, objects=objs))


def wrist_image(b64, cam, arm):
    """A wrist image for the model: smaller, with the line through the fingertips and the arm's pose drawn on."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, WRIST_PX, interpolation=cv2.INTER_AREA)
    w, h = WRIST_PX
    if cam in TIP_LINE:
        (x0, y0), (x1, y1) = TIP_LINE[cam]
        k = (y1 - y0) / (x1 - x0)  # the line across the whole image
        a, b = (0, int((y0 - k * x0) * h)), (w, int((y0 + k * (1 - x0)) * h))
        cv2.line(img, a, b, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "fingertips", (8, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        xc = int(JAW_CENTRE[cam] * w)
        cv2.line(img, (xc, a[1] + int(k * xc)), (xc, h), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "centre", (xc + 4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    if arm:
        x, y, z = arm["tip_cm"]
        s = f"fingertips z {z:.1f} cm   pitch {arm['pitch']:.0f}   jaws {arm['width_mm']:.0f} mm"
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(img, s, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def look(cams=("overhead",)):
    seen = robot.look(list(cams))
    remember(seen.get("objects", []))
    sides = {c: s for s, c in WRISTS.items()}
    for cam, b64 in list(seen.get("images", {}).items()):
        if cam != "overhead":
            seen["images"][cam] = wrist_image(b64, cam, seen.get("arms", {}).get(sides.get(cam)))
    return seen


def targets(tips, frames):
    """Where each arm's fingertips are sent (cm), by robotd's move rules: xy, then z, then rel, keyframe by keyframe."""
    out = {}
    for s, kfs in frames.items():
        p = np.array(tips[s], dtype=float)
        for kf in kfs:
            if "xy" in kf:
                p[:2] = kf["xy"]
            if "z" in kf:
                p[2] = kf["z"]
            if "rel" in kf:
                p += np.array(kf["rel"], dtype=float)
        out[s] = p
    return out


def at_grip(side, arm):
    """The remembered object nearest the arm's grip point, if one is within GRIP_NEAR_CM of it."""
    _, g, _ = frame(side, arm["joints"])
    near = [(np.linalg.norm(np.array(o["center_cm"] + [o.get("z_cm", o["size_cm"][2] / 2)]) - g * 100), o)
            for o in _memory["objects"]]
    d, o = min(near, key=lambda t: t[0], default=(None, None))
    return o if d is not None and d < GRIP_NEAR_CM else None


def grip(side, arm):
    """Close with the torque capped at GRIP_PCT until the jaws stop on what's between them, and hold it there; open
    again on nothing. Says the opening they stopped at, the object's width when one is at the grip point, and the peak
    current (how hard they squeezed)."""
    def now():
        return robot.status()["arms"][side]

    w0 = now()["width_mm"]
    robot.grip_force(side, GRIP_PCT)
    r = robot.move(**{side: [{"width": 0}]}, speed="slow")
    if r.get("outcome") != "done":
        return dict(r, width_mm=now()["width_mm"])
    time.sleep(GRIP_SETTLE_S)
    a = now()
    w, out = a["width_mm"], dict(r, width_mm=round(a["width_mm"], 1))
    if "grip_peak_ma" in a:
        out["squeeze_ma"] = a["grip_peak_ma"]
    o = at_grip(side, arm)
    if o:
        out["object_width_mm"] = round(min(o["size_cm"][:2]) * 10)
    if w < HOLD_MIN_MM:
        robot.move(**{side: [{"width": w0}]}, speed="slow")
        return dict(out, why=f"closed on nothing (at {w:.1f} mm): opened again")
    robot.move(**{side: [{"width": max(0.0, w - GRIP_HOLD_MM)}]}, speed="slow")
    return out


def motion(name, args, arms):
    """Run one motion tool. Returns robotd's result, the arms it moved (their wrist cameras come back), and where it
    sent each one's fingertips (cm), for the ones a position was asked of."""
    if name == "home":
        _pitch.clear()
        which = args.get("arms", "both")
        return robot.home(None if which == "both" else which, args.get("width")), [], {}
    if name == "open":
        side = args["arm"]
        width = scene().stroke_mm if args.get("width") is None else args["width"]
        return robot.move(**{side: [{"width": width}]}, speed="slow"), [side], {}
    if name == "grip":
        return grip(args["arm"], arms[args["arm"]]), [args["arm"]], {}
    tilt = None
    if name == "nudge":
        side = args["arm"]
        f, r, u = nudge_frame(side, arms[side]["joints"])
        d = sum(float(args.get(k) or 0) * v for k, v in (("forward", f), ("right", r), ("up", u)))
        rel, pitch = [round(float(v), 2) for v in d], _pitch.get(side, arms[side]["pitch"])
        for give in PITCH_GIVE:  # near the edge of the reach the same tilt can be just out of it: the nearest one in
            kf = {"rel": rel, "pitch": float(np.clip(pitch + give, 0, 90))}
            if robot.move(**{side: [kf]}, check_only=True).get("outcome") == "done":
                break
        else:  # nothing reaches: the move says why at the tilt it has
            kf = {"rel": rel, "pitch": pitch}
        if kf["pitch"] != pitch:
            tilt = f"pitch {pitch:.0f} doesn't reach there: tilted to {kf['pitch']:.0f}"
        frames = {side: [kf]}
        speed = "slow"
    else:  # move
        if args.get("check_only"):
            return robot.move(args.get("left"), args.get("right"), args.get("speed"), True), [], {}
        frames = {s: args[s] for s in ("left", "right") if args.get(s)}
        speed = args.get("speed")
    if not frames:
        return dict(outcome="refused", fail="check", why="move needs keyframes for left, right or both"), [], {}
    r = robot.move(frames.get("left"), frames.get("right"), speed)
    if r.get("outcome") != "refused":
        for s, kfs in frames.items():
            pitched = [kf["pitch"] for kf in kfs if kf.get("pitch") is not None]
            if pitched:
                _pitch[s] = float(pitched[-1])
            elif any(set(kf) & {"xy", "z", "rel"} for kf in kfs):
                _pitch.pop(s, None)  # moved at the steepest pitch that reaches: what that is, only the arm knows
    if tilt and r.get("outcome") == "done":
        r["tilt"] = tilt
    sent = targets({s: arms[s]["tip_cm"] for s in frames}, frames) if r.get("outcome") != "refused" else {}
    return r, list(frames), sent


def run(name, args):
    """Run one tool call. Returns (text for the model, the look that followed it: images and positions, robotd's
    result for a motion or None)."""
    if name == "look":
        seen = look(["overhead"] + (list(WRISTS.values()) if args.get("cams") == "wrists" else []))
        return state_text(seen), seen, None
    if name not in NAMES - {"done"}:
        return f"unknown tool {name!r}", None, None
    if name in ("nudge", "grip", "open") and args.get("arm") not in WRISTS:
        return f"bad arguments for {name}: arm is left or right", None, None
    if name == "move" and any(not isinstance(args.get(s) or [], list) for s in ("left", "right")):
        return "bad arguments for move: left and right are lists of keyframes", None, None
    r, moved, sent = motion(name, args, robot.status()["arms"])
    out = {k: r[k] for k in ("outcome", "fail", "why", "tilt", "width_mm", "object_width_mm", "squeeze_ma", "checked")
           if r.get(k) is not None}
    seen = look(["overhead"] + [WRISTS[s] for s in moved])
    for s, p in sent.items():
        got = np.array(seen["arms"][s]["tip_cm"])
        off = float(np.linalg.norm(got - p))
        if off > SHORT_CM:
            out.setdefault("short", {})[s] = f"sent to {np.round(p, 1).tolist()}, got to {got.tolist()}: {off:.1f} cm"
    out["events"] = [e["text"] for e in r.get("events", []) if e.get("kind") != "debug"][-6:]
    return json.dumps(out) + "\nnow: " + state_text(seen), seen, r
