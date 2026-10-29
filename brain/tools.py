"""The tools a model drives the arms with: robotd's primitives as OpenAI function schemas, and the code that runs them.

Every call comes back as text (the outcome, why it failed, what robotd noticed, where the arms and objects are) plus
a fresh overhead image with the grid and grippers drawn on, so one turn is one call and the model always acts on
what's there now.
"""
import json

from harness import robot

ARM = {"type": "string", "enum": ["left", "right"]}
XY = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "table cm [x, y]"}
KEYFRAME = {
    "type": "object",
    "properties": {
        "xy": XY,
        "z": {"type": "number", "description": "fingertip height above the table, cm"},
        "rel": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                "description": "[dx, dy, dz] cm from where the previous keyframe left the fingertips"},
        "yaw": {"type": "number", "description": "jaw line angle on the table, degrees (0 = along x)"},
        "pitch": {"type": "number", "description": "90 straight down, 0 level; left out: the steepest that reaches"},
        "width": {"type": "number", "description": "gripper opening, mm (0-84)"},
    },
}


def fn(name, description, props=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props or {}, "required": list(required)}}}


TOOLS = [
    fn("look", "Fresh images and positions without moving. overhead (default) is the table with the grid; wrists "
       "adds both wrist cameras (what each gripper sees, close up).",
       {"cams": {"type": "string", "enum": ["overhead", "wrists", "all"]}}),
    fn("pick", "Pick an object up with one arm: reach over it, go down until the fingers touch the table, close, "
       "lift. A long object is held near its end on that arm's side. Fails (miss, finger_on_object, unreachable) "
       "say why.",
       {"arm": ARM, "target": {"type": "string", "description": "object id (#1), name or colour"},
        "width": {"type": "number", "description": "opening before closing, mm (default: the object's width + 25)"}},
       ["arm", "target"]),
    fn("place", "Put the held object down: its centre at xy (or on another object), lowered until it rests on the "
       "table, then let go and rise.",
       {"arm": ARM, "xy": XY, "on": {"type": "string", "description": "object id, name or colour to put it on"},
        "yaw": {"type": "number", "description": "its long side's angle on the table, degrees (optional)"}},
       ["arm"]),
    fn("handover", "Pass the held object from one arm to the other in mid-air: the giver brings it to the middle, "
       "the taker grips the other end, the giver lets go.",
       {"giver": ARM, "taker": ARM}, ["giver", "taker"]),
    fn("move", "Free-form motion, checked before anything moves. left and right are lists of keyframes; the k-th "
       "keyframes of both arms are one phase and both arms finish each phase together. Values left out stay as "
       "they are. Use it to lift, carry, tilt, push or get out of the way; picking up and letting go are "
       "pick/grasp and place/release. check_only plans it without moving.",
       {"left": {"type": "array", "items": KEYFRAME}, "right": {"type": "array", "items": KEYFRAME},
        "speed": {"type": "string", "enum": ["slow", "normal", "fast"]}, "check_only": {"type": "boolean"}}),
    fn("reach", "Bring an open gripper over an object (or over xy) ready to grasp, without closing.",
       {"arm": ARM, "target": {"type": "string"}, "xy": XY,
        "z": {"type": "number", "description": "hover height, cm (default 10)"},
        "yaw": {"type": "number", "description": "jaw line angle, degrees (default: across the object)"},
        "width": {"type": "number", "description": "opening, mm"}}, ["arm"]),
    fn("grasp", "After reach: go down until the fingers touch the table, close, and say whether it's holding.",
       {"arm": ARM, "width": {"type": "number"}}, ["arm"]),
    fn("release", "Open the gripper. set_down (default true) first lowers a held object until it rests on the "
       "table.", {"arm": ARM, "set_down": {"type": "boolean"}}, ["arm"]),
    fn("home", "Park one or both arms out of the way (the start pose).",
       {"arms": {"type": "string", "enum": ["both", "left", "right"]},
        "width": {"type": "number", "description": "gripper opening, mm"}}),
    fn("done", "Finish: the task is done, or can't be done. Say honestly which, and why.",
       {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ["success", "summary"]),
]
NAMES = {t["function"]["name"] for t in TOOLS}


def state_text(seen):
    """What a look returned, as compact text: the arms and the objects."""
    arms = {s: {k: a[k] for k in ("tip_cm", "yaw", "width_mm", "holding", "object") if k in a}
            for s, a in seen.get("arms", {}).items()}
    objs = [{k: o[k] for k in ("id", "name", "color", "center_cm", "size_cm", "yaw", "held_by") if k in o}
            for o in seen.get("objects", [])]
    return json.dumps(dict(arms=arms, objects=objs))


def look(cams="overhead"):
    names = {"overhead": ["overhead"], "wrists": ["overhead", "left_wrist", "right_wrist"],
             "all": ["overhead", "left_wrist", "right_wrist"]}.get(cams, ["overhead"])
    return robot.look(names)


def run(name, args):
    """Run one tool call. Returns (text for the model, the look that followed it: images and positions, robotd's
    result for a motion or None)."""
    if name == "look":
        seen = look(args.get("cams", "overhead"))
        return state_text(seen), seen, None
    if name == "home":
        arms = args.get("arms", "both")
        r = robot.home(None if arms == "both" else arms, args.get("width"))
    elif name == "move":
        r = robot.move(args.get("left"), args.get("right"), args.get("speed"), args.get("check_only"))
    elif name in ("pick", "place", "handover", "reach", "grasp", "release"):
        try:
            r = getattr(robot, name)(**args)
        except TypeError as e:
            return f"bad arguments for {name}: {e}", None, None
    else:
        return f"unknown tool {name!r}", None, None
    out = {k: r[k] for k in ("outcome", "fail", "why", "width_mm", "checked") if r.get(k) is not None}
    out["events"] = [e["text"] for e in r.get("events", []) if e.get("kind") != "debug"][-6:]
    seen = look()
    return json.dumps(out) + "\nnow: " + state_text(seen), seen, r
