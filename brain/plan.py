"""Plan, act, check: Muse thinks the task through once, the arms run the plan fast, Muse checks where it matters.

1. Look: both arms go to the ready pose, out of the camera's way, Muse names what the task involves, SAM finds those things on the
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

from harness import robot

from . import act, reach, tools, wrist
from . import video as video_lib
from .prompt import CHECK_ASK, PLAN_ASK, find_ask, plan_system, video_intro
from .rig import scene

SHAPES = ["cylinder", "can", "box", "cup", "bowl", "ball"]  # plain shape words SAM also looks for, in case a name misses
MAX_STEPS = 80
MAX_CALLS = 50
KEEP = 6  # Muse sees the naming and planning exchanges and the last KEEP checks: the whole history costs too much


def parse(text):
    """The JSON object in a reply (it may come with a fence or a stray word)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def image_part(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


class Run:
    def __init__(self, llm, log=print, run_dir=None):
        self.llm, self.log, self.run_dir = llm, log, run_dir
        self.system = plan_system(scene())
        self.plan = None  # the plan as Muse first made it, with its task
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
        if len(self.messages) > 4 + 2 * KEEP:  # it alternates user, Muse: what's cut starts and ends a pair
            del self.messages[4:len(self.messages) - 2 * KEEP]
        for m in self.messages[:-1] if keep_images else self.messages:  # older images go: the history stays cheap
            if isinstance(m.get("content"), list):
                m["content"] = [p for p in m["content"] if p.get("type") != "image_url"]
        self.messages.append({"role": "user", "content": content})
        for attempt in range(2):
            self.calls += 1
            r = self.llm.ask(self.system, self.messages, effort)
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
        holding = "; ".join(f"{s} holds " + ((a.get("held") or {}).get("name") or "nothing")
                            for s, a in sorted(seen.get("arms", {}).items()))
        parts = [{"type": "text", "text": text + f"\n\nNow {holding}. State: " + tools.state_text(seen)}]
        for cam, b64 in seen.get("images", {}).items():
            parts += [{"type": "text", "text": f"{cam} camera:"}, image_part(b64)]
        return parts

    def go(self, task, video=None, max_steps=MAX_STEPS, plan_only=False):
        """Do the task (or the video's). plan_only: look and plan without moving anything, then stop."""
        t0 = time.time()
        tools.reset()
        tools.FIND = []
        if not plan_only:
            self.log("both arms to the ready pose: raised, level, pointing ahead, cameras on top")
            for side in ("left", "right"):  # nothing held from a run that was stopped
                robot.hold(side)
            r = robot.home(None, act.TRAVEL_MM)
            if r.get("outcome") != "done":
                return self.end(False, f"the arms couldn't get to the ready pose: {r.get('why')}", t0)
        summary, video_objects, frames = "", [], []
        if video:
            summary, video_objects = video_lib.reading(video)
            frames = video_lib.frames(video)
            task = task or ("Do with the two arms what the person in the video does: " + summary)
            self.log(f"video: {summary}")
        seen = self.look()
        names = self.ask([{"type": "text", "text": find_ask(task, video_objects)},
                          image_part(seen["images"]["overhead"])], effort="low").get("find", [])
        words = [w for n in names for w in (n if isinstance(n, list) else [n])]  # each thing's words, or a flat list
        tools.FIND = list(dict.fromkeys([str(n).lower() for n in words + video_objects] + SHAPES))
        tools.reset()  # from here on the objects are SAM's: forget the colour detector's
        seen = self.look(tag="00")
        self.log(f"SAM looks for {tools.FIND}; found " + ", ".join(
            f"{o['name']} at {o['center_cm']}" for o in tools._memory["objects"]))
        if not plan_only:  # the arms are in the ready pose, their cameras looking level across the table
            heights = wrist.measure_heights(tools._memory["objects"])
            if heights:
                self.log("measured from the wrist cameras: " + ", ".join(f"{n} {h} cm tall" for n, h in heights.items()))
                seen = self.look(tag="00")
        intro = f"Task: {task}"
        if video:
            intro += video_intro(summary, video_lib.held(video))
        parts = [{"type": "text", "text": intro}]
        for t, b64 in frames:
            parts += [{"type": "text", "text": f"video at {t} s:"}, image_part(b64)]
        parts += self.seen_parts("The robot's table (both arms parked).", seen) + [
            {"type": "text", "text": reach.spots_text()}, {"type": "text", "text": PLAN_ASK}]
        self.log("planning (Muse thinks it through)")
        plan = self.ask(parts, keep_images=0)
        self.corrections(plan)
        self.plan = dict(plan, task=task, model=getattr(self.llm, "model", ""))
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
                self.corrections(v)
                if v.get("done") or not v.get("next"):
                    d = v.get("done") or dict(success=bool(v.get("ok")), summary=v.get("why", ""))
                    return self.end(bool(d.get("success")), str(d.get("summary", "")), t0)
                steps, i = steps + list(v["next"]), i
                continue
            step = steps[i]
            self.stats["steps"] += 1
            n = self.stats["steps"]
            self.starting(n, step)
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
            self.save(state=tools.state_text(seen))  # what Muse is shown, for reading a run afterwards
            ahead = "; ".join(f"{k}. {describe(s)}" for k, s in enumerate(steps[i + 1:i + 6], 1)) or "none"
            v = self.ask(self.seen_parts(
                f"Step {n} ran: {describe(step)}\nResults: {json.dumps(results)}\n"
                f"Check: {step.get('check') or 'it failed: what now?'}\nNext in the plan: {ahead}"
                + ("" if len(steps) <= i + 6 else f" (and {len(steps) - i - 6} more)"), seen)
                + [{"type": "text", "text": CHECK_ASK}], effort="low")
            self.log(f"    check: {'ok' if v.get('ok') else 'not ok'}: {v.get('why', '')}")
            self.corrections(v)
            if v.get("done"):
                return self.end(bool(v["done"].get("success")), str(v["done"].get("summary", "")), t0)
            nxt = list(v.get("next") or [])
            if nxt:
                self.log("    next: " + " | ".join(describe(s) for s in nxt) + (" (re-planned)" if v.get(
                    "replace_rest") else ""))
            steps = steps[:i + 1] + nxt + ([] if v.get("replace_rest") else steps[i + 1:])
            i += 1
        return self.end(False, f"stopped after {self.stats['steps']} steps and {self.calls} calls", t0)

    def starting(self, n, step):
        """Called just before step n runs: nothing here; a dashboard shows it running."""

    def corrections(self, reply):
        if reply.get("correct"):
            self.log("    corrected: " + "; ".join(tools.correct(reply["correct"])))

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
