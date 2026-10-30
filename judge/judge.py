"""The judge: did a recorded episode do its task? Muse Spark looks at the overhead camera at the start, around the
handover and at the end, reads what robotd recorded (and SAM's vision section, if there is one), and returns the
label's judge section.

It judges on its own evidence: it isn't shown robotd's end check or the brain's claim of success, so when it agrees
with robotd's check (which final() requires before an episode trains), that agreement means something.
"""
import json
import time

from . import frames as fr

# One list for command results and judge.failure.kind (the contracts doc)
FAILURES = ["miss", "finger_on_object", "no_contact", "slip", "dropped", "unreachable", "check", "blocked", "manual",
            "torque_off", "busy", "timeout", "wrong_place", "not_seen"]

SYSTEM = """You are the judge for a two-arm robot (SO-101 arms with parallel grippers) on a table. You get one recorded
episode: its task, what the robot did step by step, and frames from the overhead camera. Decide whether the task
was done, from the frames above all.

The frames have the table's grid drawn on in cm: x to the right, y away from the arms, a line every 5 cm. The left
arm's base is at about (-18, -1) and the right arm's at (18, -1). When the task has a target, a red cross marks it,
with a red ring of 3 cm around it: the object has to end inside the ring and not in a gripper.

Rules:
- success is true only if the last frame shows the goal met. A step log that says "done" is not enough: look.
- If it failed, name the first thing that went wrong, with one of these kinds:
  miss (the fingers closed on nothing), finger_on_object, no_contact, slip (lost while carrying), dropped (fell during
  a handover), unreachable, check (a plan was refused), blocked, manual (a person took over), torque_off, busy,
  timeout, wrong_place (it ended away from the target), not_seen (the object can't be found in the frames).
- confidence: how sure you are, 0-1. Use below 0.7 when the frames don't show it clearly (the object hidden by a
  gripper, out of view, blurred): those episodes go to a person.
- reason: one plain sentence a person reads in mission control, about what the frames show.
Answer by calling verdict once."""

VERDICT = {"type": "function", "function": {
    "name": "verdict", "description": "Your decision on the episode.",
    "parameters": {"type": "object", "properties": {
        "success": {"type": "boolean"},
        "confidence": {"type": "number", "description": "0-1"},
        "failure": {"type": "string", "enum": ["none"] + FAILURES,
                    "description": "the first thing that went wrong; none when it succeeded"},
        "failure_t": {"type": "number", "description": "when it went wrong, s since the episode began"},
        "failure_arm": {"type": "string", "enum": ["left", "right", "none"]},
        "reason": {"type": "string", "description": "one sentence about what the frames show"},
    }, "required": ["success", "confidence", "failure", "reason"]}}}


class JudgeError(Exception):
    pass


def pick_frames(label):
    """[(frame index, what it is)]: the start, just after the handover (or the middle), and the end."""
    n = int(label.get("frames") or 0)
    fps = float(label.get("fps") or 30)
    if n <= 0:
        return []
    picks = [(0, "start")]
    passes = [e for e in (label.get("robot") or {}).get("events", []) if e.get("kind") == "pass"]
    if passes and passes[0].get("t") is not None:
        picks.append((round(passes[0]["t"] * fps), "just after the handover"))
    else:
        picks.append((n // 2, "middle"))
    picks.append((n - 1, "end"))
    out, seen = [], set()
    for i, why in picks:
        i = min(max(int(i), 0), n - 1)
        if i not in seen:
            seen.add(i)
            out.append((i, why))
    return sorted(out)


def describe(label):
    """What the judge is told in words. Left out on purpose: robotd's end check, its idea of where the object
    ended, and the brain's own verdict (the agents section)."""
    robot = label.get("robot") or {}
    steps = [{k: s[k] for k in ("op", "arm", "giver", "taker", "t", "outcome", "fail", "why", "width_mm") if k in s}
             for s in robot.get("steps", [])]
    facts = {
        "task": label.get("task"),
        "who": (label.get("source") or {}).get("kind"),
        "giver": robot.get("giver"), "taker": robot.get("taker"),
        "target_cm": robot.get("target_cm"),
        "duration_s": robot.get("duration_s"),
        "a person took over (interventions)": robot.get("interventions", 0),
        "steps": steps,
    }
    text = "Episode:\n" + json.dumps(facts, indent=1)
    vision = label.get("vision")
    if vision:
        text += ("\n\nWhat SAM 3.1 segmentation saw in the overhead video (pixels in the 640x480 frame, times in s):\n"
                 + json.dumps({k: vision.get(k) for k in ("objects", "events")}, indent=1))
    return text


def verdict_to_judge(args, reply_model, frames_used, tokens_in, tokens_out, seconds):
    """The model's verdict call -> the label's judge section, cleaned up."""
    success = bool(args.get("success"))
    try:
        conf = float(args.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)
    failure = None
    kind = args.get("failure")
    if not success and kind in FAILURES:
        arm = args.get("failure_arm")
        failure = {"kind": kind, "t": args.get("failure_t"), "arm": arm if arm in ("left", "right") else None}
    return {"model": reply_model, "success": success, "confidence": round(conf, 2), "failure": failure,
            "reason": str(args.get("reason", "")).strip(), "frames": frames_used,
            "tokens": {"in": tokens_in, "out": tokens_out}, "latency_s": round(seconds, 1)}


def judge(label, images, llm, log=print):
    """images: [(frame index, what it is, BGR image already annotated)]. Returns (judge section, cost in $)."""
    fps = float(label.get("fps") or 30)
    parts = [{"type": "text", "text": describe(label)}]
    for i, why, img in images:
        parts += [{"type": "text", "text": f"Frame {i} (t = {i / fps:.1f} s, {why}):"},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fr.jpeg_b64(img)}"}}]
    messages = [{"role": "user", "content": parts}]
    tokens_in = tokens_out = 0
    seconds = cost = 0.0
    model = getattr(llm, "model", "")
    for attempt in range(2):
        reply = llm.chat(SYSTEM, messages, [VERDICT])
        tokens_in += reply.tokens_in
        tokens_out += reply.tokens_out
        seconds += reply.seconds
        cost += reply.cost
        model = reply.model or model
        call = next((c for c in reply.calls if c.name == "verdict" and not c.error), None)
        if call is not None:
            return verdict_to_judge(call.args, model, [i for i, _, _ in images], tokens_in, tokens_out, seconds), cost
        log(f"  the judge answered without a usable verdict: {reply.text.strip()[:200]!r}")
        messages.append(reply.message)
        for c in reply.calls:  # every tool call needs its answer before the next user turn
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": c.error or f"there's no tool {c.name!r}: call verdict"})
        messages.append({"role": "user", "content": "Call verdict with your decision."})
    raise JudgeError("the model didn't call verdict")
