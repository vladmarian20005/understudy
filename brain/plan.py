"""Plan, act, check: Muse thinks the task through once, the arms run the plan fast, Muse checks where it matters.

1. Look: both arms park out of the camera's way, Muse names what the task involves, SAM finds those things on the
   table (any colour, with real heights), and the overhead image gets the grid and their names.
2. Plan: one call that thinks hard: the task (and, from a video, what the person did, with frames from it), the
   table, what the arms can reach. The answer is a list of steps, each one action per arm; both arms of a step move
   at the same time.
3. Act: brain/act.py runs a step with no model call in between: grasps, moves, tilts, letting go.
4. Check: after a step that asks for it, or one that failed, a quick call with fresh images: go on, run these steps
   first, re-plan the rest, or finish.
Muse decides everything about the task; the code only does the motions it's told, checked by robotd.
"""
import base64
import json
import os
import re
import time

import cv2
import numpy as np

from harness import robot

from . import act, tools
from .prompt import RIG

SHAPES = ["cylinder", "box", "cup", "bowl", "ball"]  # plain shape words SAM also looks for, in case a name misses
MAX_STEPS = 80
MAX_CALLS = 50

SYSTEM = RIG + """
You plan and run a task on these arms. You first make the whole plan; then it runs step by step, with no call to you
in between unless a step asks for a check or fails. Then you see fresh images and decide.

Objects come from a segmentation model on the overhead image, by name: center_cm [x, y], size_cm [L, W, H], and shape:
"round" (as tall as wide), "upright" (a standing thing; its height comes from the image and is good to a few cm;
"at least this tall" when its top is cut off by the image's edge) or "flat" (its height is a guess). An object an arm
holds is listed under that arm: its center, top and bottom now, top being the end that was up when it was picked up.
It moves with the gripper. Each arm also has its grip_point_cm (the spot between the finger pads, 2 cm back from the
fingertips, where the jaws pinch and what it holds sits) and grip_ma (the gripper motor's current: above about 120 mA
it is squeezing something, 50-80 idle).

Actions, one per arm per step; a step's left and right actions run at the same time:
- {"do": "grasp", "object": NAME, "height": cm, "pitch": deg, "lift": cm, "offset": [forward, right, up]}
  The grip point onto the object at height (cm above the table where the jaws close; default the object's middle),
  coming in along the line from that arm's base: jaws open wider than the object, down in front of it, in, close with
  a capped force, hold, lift by lift cm (default 5). pitch: 90 fingers straight down, 0 level; left out: the
  steepest that reaches, or for an object 10 cm or taller the most level that reaches (from the side): leave it
  out unless you have a reason. The jaws open to 84 mm: what they close on must be narrower than about 80 mm there.
  If the jaws close narrower than the object (they caught its edge), the grasp goes deeper and grips again by
  itself, and every approach is corrected for an arm that stops short. It fails cleanly (jaws open, backed
  off) when they close on nothing or it slips on the lift. offset shifts the grip point in the gripper's own
  directions (forward the way the fingers point, right, up), cm, to correct a miss.
- {"do": "move", "to": [x, y, z]}: the grip point, and what it holds, to that point.
  {"do": "move", "above": NAME, "by": cm}: what it holds (or the grip point) so its bottom is `by` cm above the top
  of NAME, where NAME is now, also while the other arm holds it.
  {"do": "move", "rel": [dx, dy, dz]}: shifted.
  Options: "pitch" (default: the one it has; holding something, keep it unless you mean to turn it), "speed".
- {"do": "tilt", "deg": D}: turn the wrist by D degrees: what it holds tilts about the direction the fingers point.
  Tilt after moving it into place: a move afterwards can turn the wrist back.
- {"do": "open"}: let go. {"do": "grip"}: close where it is (grasp already grips).
- {"do": "nudge", "forward": cm, "right": cm, "up": cm}: a small step in the gripper's own directions.
- {"do": "park"}: raised, level, out to the side, so the camera sees the table.

Rules:
- Every step is checked before anything moves: reach, the table, a held object's clearance above it, and a 1.5 cm
  gap between the arms. A refused step changes nothing.
- Use both arms in the same step whenever they don't depend on each other: it is much faster. When both work near
  the middle, keep one a little to the side until the other is in place.
- Before you plan a pose, check it against the reach and height facts above: a target one arm can't reach gets
  refused. A standing object's top is at its height; held upright and lifted, it is higher still.
- To put something into something: hold it above the opening ("move above" by 2-4 cm), check, then open.
- Add a "check" to a step whose outcome you can't be sure of (a grasp, letting go into something, a tilt): say what
  should be true after it. Skip it for plain moves.
- Reply with JSON only, no other text.
"""

PLAN_ASK = """Make the plan. Reply with JSON:
{"understanding": "what has to happen and how the arms will do it, 2-4 sentences",
 "objects": {"<word the task uses>": "<object name from the list>"},
 "steps": [{"left": ACTION or null, "right": ACTION or null, "check": "what should be true after it, or omit",
            "why": "a few words"}],
 "done_when": "what the finished table looks like"}"""

CHECK_ASK = """Reply with JSON:
{"ok": true or false, "why": "what you see, one sentence",
 "next": [only the steps to run now, same format as the plan's; omit if none: the rest of the plan stays],
 "replace_rest": true if those steps replace the rest of the plan (re-plan), false to run them before the rest,
 "done": {"success": true or false, "summary": "..."} only when the task is finished or can't be done}"""


def parse(text):
    """The JSON object in a reply (it may come with a fence or a stray word)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def image_part(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def video_frames(path, n=6, width=480):
    """n frames spread over a video, as JPEG base64, with their times."""
    cap = cv2.VideoCapture(path)
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30
    out = []
    for k in range(n):
        i = int((k + 0.5) * count / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append((round(i / fps, 1), base64.b64encode(buf.tobytes()).decode()))
    return out


def watch(path):
    """What the person in a video does: (summary, objects), from watch's cached Muse reading, or read now."""
    stem = os.path.splitext(os.path.basename(path))[0]
    here = os.path.join(os.path.dirname(__file__), "..")
    for cache in (os.path.join(here, "watch", "sam_out", "videos", stem, "muse.json"),
                  os.path.join(here, "sam_out", "videos", stem, "muse.json")):
        if os.path.exists(cache):
            d = json.load(open(cache))
            return d.get("summary", ""), d.get("objects", [])
    import sys
    sys.path.insert(0, os.path.join(here, "watch"))
    from sam_pipeline.credentials import api_key
    from sam_pipeline.muse_client import MuseClient, PROMPT
    c = MuseClient(api_key(), "https://api.meta.ai/v1")
    r = c.watch(c.upload(__import__("pathlib").Path(path)), PROMPT)
    return r.summary, r.objects


class Run:
    def __init__(self, llm, log=print, run_dir=None):
        self.llm, self.log, self.run_dir = llm, log, run_dir
        self.messages, self.calls, self.turn = [], 0, 0
        self.stats = dict(tokens_in=0, tokens_out=0, cost=0.0, llm_seconds=0.0, steps=0, calls=0)
        self.record = []

    def save(self, **rec):
        rec = dict(t=round(time.time(), 2), **rec)
        self.record.append(rec)
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "turns.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def save_images(self, seen, tag):
        if not self.run_dir:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        for cam, b64 in (seen or {}).get("images", {}).items():
            with open(os.path.join(self.run_dir, f"{tag}_{cam}.jpg"), "wb") as f:
                f.write(base64.b64decode(b64))

    def ask(self, content, effort=None, keep_images=1):
        """Send one user message (text and images) in the running conversation; returns the parsed JSON."""
        for m in self.messages[:-1] if keep_images else self.messages:  # older images go: the history stays cheap
            if isinstance(m.get("content"), list):
                m["content"] = [p for p in m["content"] if p.get("type") != "image_url"]
        self.messages.append({"role": "user", "content": content})
        for attempt in range(2):
            self.calls += 1
            r = self.llm.ask(SYSTEM, self.messages, effort)
            for k, v in (("tokens_in", r.tokens_in), ("tokens_out", r.tokens_out), ("cost", r.cost),
                         ("llm_seconds", r.seconds)):
                self.stats[k] += v
            try:
                out = parse(r.text)
                self.messages.append(r.message)
                self.save(muse=r.text, seconds=round(r.seconds, 1))
                return out
            except ValueError as e:
                self.messages.append(r.message)
                self.messages.append({"role": "user", "content": f"That wasn't valid JSON ({e}). JSON only."})
        raise RuntimeError("Muse didn't answer with JSON")

    def look(self, sides=(), tag=None):
        seen = tools.look(["overhead"] + [tools.WRISTS[s] for s in sides])
        if tag:
            self.save_images(seen, tag)
        return seen

    def seen_parts(self, text, seen):
        parts = [{"type": "text", "text": text + "\n\nNow: " + tools.state_text(seen)}]
        for cam, b64 in seen.get("images", {}).items():
            parts += [{"type": "text", "text": f"{cam} camera:"}, image_part(b64)]
        return parts

    def go(self, task, video=None, max_steps=MAX_STEPS, plan_only=False):
        """Do the task (or the video's). plan_only: look and plan without moving anything, then stop."""
        t0 = time.time()
        tools.reset()
        tools.FIND = []
        if not plan_only:
            self.log("parking both arms out of the camera's way")
            for side in ("left", "right"):  # nothing held from a run that was stopped
                robot.hold(side)
            _, stopped = act.run({"left": {"do": "park", "open": True}, "right": {"do": "park", "open": True}})
            if stopped:
                return self.end(False, f"a person stopped the arms: {stopped}", t0)
        summary, video_objects, frames = "", [], []
        if video:
            summary, video_objects = watch(video)
            frames = video_frames(video)
            task = task or ("Do with the two arms what the person in the video does: " + summary)
            self.log(f"video: {summary}")
        seen = self.look()
        names = self.ask([{"type": "text", "text":
                           f"Task: {task}\nObjects named in the video: {video_objects}\nList the physical things on "
                           "the table this task involves, as the short concrete nouns a segmentation model knows "
                           "('tennis ball', 'cup', 'cylinder'): {\"find\": [...]}"},
                          image_part(seen["images"]["overhead"])], effort="low").get("find", [])
        tools.FIND = list(dict.fromkeys([n.lower() for n in names + video_objects] + SHAPES))
        tools.reset()  # from here on the objects are SAM's: forget the colour detector's
        seen = self.look(tag="00")
        self.log(f"SAM looks for {tools.FIND}; found " + ", ".join(
            f"{o['name']} at {o['center_cm']}" for o in tools._memory["objects"]))
        intro = f"Task: {task}"
        if video:
            intro += ("\n\nThis comes from a video of a person doing it, filmed facing them: left and right are the "
                      "person's own (their right hand is on the image's left), and their left hand is the robot's left "
                      f"arm. What they do, in order: {summary}\nFrames from the video follow, in order.")
        parts = [{"type": "text", "text": intro}]
        for t, b64 in frames:
            parts += [{"type": "text", "text": f"video at {t} s:"}, image_part(b64)]
        parts += self.seen_parts("The robot's table (both arms parked).", seen) + [{"type": "text", "text": PLAN_ASK}]
        self.log("planning (Muse thinks it through)")
        plan = self.ask(parts, keep_images=0)
        steps = list(plan.get("steps") or [])
        self.log(f"plan: {plan.get('understanding', '')}")
        for k, s in enumerate(steps, 1):
            self.log(f"  {k}. {describe(s)}")
        done_when = plan.get("done_when", "the task is done")
        if plan_only:
            return self.end(False, "planned only: nothing moved", t0)
        i = 0
        while self.stats["steps"] < max_steps and self.calls < MAX_CALLS:
            if i >= len(steps):  # the plan ran out: is it done?
                seen = self.look(("left", "right"), tag=f"{self.stats['steps']:02d}_end")
                v = self.ask(self.seen_parts(f"All planned steps ran. Done when: {done_when}. Is it? "
                                             "Finish with done, or give the next steps.", seen) +
                             [{"type": "text", "text": CHECK_ASK}], effort="low")
                if v.get("done") or not v.get("next"):
                    d = v.get("done") or dict(success=bool(v.get("ok")), summary=v.get("why", ""))
                    return self.end(bool(d.get("success")), str(d.get("summary", "")), t0)
                steps, i = steps + list(v["next"]), i
                continue
            step = steps[i]
            self.stats["steps"] += 1
            n = self.stats["steps"]
            ts = time.time()
            results, stopped = act.run(step)
            self.log(f"[{n}] {describe(step)} -> {json.dumps(results)} ({time.time() - ts:.1f} s)")
            self.save(step=step, results=results, seconds=round(time.time() - ts, 1))
            if stopped:
                return self.end(False, f"a person stopped the arms: {stopped}", t0)
            failed = any("failed" in r for r in results.values())
            if not (step.get("check") or failed):
                i += 1
                continue
            sides = [s for s in ("left", "right") if step.get(s)]
            seen = self.look(sides, tag=f"{n:02d}")
            ahead = "; ".join(f"{k}. {describe(s)}" for k, s in enumerate(steps[i + 1:i + 6], 1)) or "none"
            v = self.ask(self.seen_parts(
                f"Step {n} ran: {describe(step)}\nResults: {json.dumps(results)}\n"
                f"Check: {step.get('check') or 'it failed: what now?'}\nNext in the plan: {ahead}"
                + ("" if len(steps) <= i + 6 else f" (and {len(steps) - i - 6} more)"), seen)
                + [{"type": "text", "text": CHECK_ASK}], effort="low")
            self.log(f"    check: {'ok' if v.get('ok') else 'not ok'}: {v.get('why', '')}")
            if v.get("done"):
                return self.end(bool(v["done"].get("success")), str(v["done"].get("summary", "")), t0)
            nxt = list(v.get("next") or [])
            if nxt:
                self.log("    next: " + " | ".join(describe(s) for s in nxt) + (" (re-planned)" if v.get(
                    "replace_rest") else ""))
            steps = steps[:i + 1] + nxt + ([] if v.get("replace_rest") else steps[i + 1:])
            i += 1
        return self.end(False, f"stopped after {self.stats['steps']} steps and {self.calls} calls", t0)

    def end(self, success, summary, t0):
        self.log(f"{'success' if success else 'failure'}: {summary}")
        return dict(self.stats, success=success, summary=summary, calls=self.calls,
                    cost=round(self.stats["cost"], 4), llm_seconds=round(self.stats["llm_seconds"], 1),
                    seconds=round(time.time() - t0, 1), model=getattr(self.llm, "model", ""), record=self.record)


def describe(step):
    def one(a):
        a = dict(a)
        return a.pop("do", "?") + ("" if not a else " " + json.dumps(a, separators=(",", ":")))
    arms = [f"{s}: {one(step[s])}" for s in ("left", "right") if step.get(s)]
    return " || ".join(arms) + (f"  [check: {step['check']}]" if step.get("check") else "")
