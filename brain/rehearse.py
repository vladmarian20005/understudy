"""Replay a saved plan over and over and record each run, with no model calls: data a policy can learn the task from.

    python -m brain "<task>" --plan-only --save-plan plans/mine.json
    python -m harness.robotd --sim --hd --objects cubes --no-camera-streams --port 5701 --dataset mine
    ROBOTD_PORT=5701 python -m brain.rehearse plans/mine.json --objects "A,B,C" \\
        --box "A=x0,x1,y0,y1" --box "B=x0,x1,y0,y1" --box "C=x0,x1,y0,y1" --on "C,A,B" --episodes 20 --seed 1

--objects go left to right, --on is the chain from the top down: what worked means is that each stands on the next. Look-alike
objects are told apart by where they start, so each has its own --box, and a reset always puts it back in that box.

Outside the recording: each object (the top one first) picked up and set at a random spot in its box, then robotd's home
for both arms at WIDTH_MM: every episode's start. Recorded: the plan's steps through brain/act.py by object name, and the
arm lifting clear, with no look in between; a closing park stays out, so no episode ends at home. Then home and a look:
--on says from where the objects are whether it worked. A failed run stays in the dataset, labelled, not trainable.

With --policy outputs/<run>, the same resets and the same check, but a trained policy does the task and nothing is
recorded: how well it learned. A trial stops as soon as the task is done, so the policy doesn't start over.

On the real arms: the same command against robotd without --sim, plus --size L,W,H --prompt "<what they look like>": robotd's
colour detector can't tell look-alikes apart and one overhead camera can't measure height, so SAM finds the piles (one
prompt, one mask per pile; perception only, key in .env) and --objects and --on say which is which. A person there has to
switch torque on (key a in robotd) and stay at the keys (space stops the arms), set the objects out at first, each in its
box and square to the arm's line, and put one back when a reset can't (it waits for Enter). Ctrl+C here ends after the
episode under way.
"""
import argparse
import json
import signal
import sys
import time

import cv2
import numpy as np

from harness import robot
from harness.perception import TablePerception

from . import act, sam, tools

WIDTH_MM = 40.0  # the gripper opening of every start: what the ready pose starts with in the simulator
XY_CM, Z_CM = 1.5, 0.5  # on: centres this close across, and one object's height above give or take this
SPOT_CM = 3.0  # a reset put an object where it meant to when it's this close; two masks this close are one pile
ATTEMPTS = 3  # resets tried before a person is asked
LOW_CM = 15.0  # an arm with its fingertips lower than this is still at something
TURNED = 0.15  # a grasp on an object standing on another goes this share of its width right of it: it can end up about
# 20 deg turned in the jaws, and then its corner sticks out this far past where the fixed finger comes down
TIPS_CLEAR_CM = 0.6  # and the fingertips stay this far above what it stands on
PILE_CM = 2.0  # a mask's centre this close to the lowest object's, at some level's height, is part of a pile on it
known = {}  # object name -> what's known about it, as last seen (one under another can't be seen)


def far(p, q, cm):
    return np.hypot(*np.subtract(p, q)) >= cm


def by_sam(img, a):
    """The objects from SAM's masks of one prompt, at their known size: a mask's centre on the plane through the middle
    of what it shows. A mask whose centre comes near the chain's lowest object at some level's height is part of a pile on
    it (the camera sees a tower from the side, and SAM may give each object's sliver its own mask): the biggest of those is
    the pile's top. The masks left over are the objects still on the table, left to right, so how many they are says how
    high the pile is, and the ones in it that don't show stay where they were."""
    L, W, H = a.size
    tp = TablePerception(tools.scene())
    centre = lambda c, z: [round(float(v) * 100, 1) for v in tp.footprint(c, z / 100)[0]]
    one = lambda n, c, z: dict(name=n, center_cm=centre(c, z), z_cm=z, size_cm=[L, W, H], yaw=0.0)
    outlines = [max(cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                    key=cv2.contourArea) for _, m in sam.segment(img, [a.prompt])[a.prompt]]
    outlines.sort(key=lambda c: -cv2.contourArea(c))
    base, pile, rest = known.get(a.on[-1]), [], []
    for c in outlines:
        if base and min(np.hypot(*np.subtract(centre(c, (k + 0.5) * H), base["center_cm"]))
                        for k in range(len(a.on))) < PILE_CM:
            pile.append(c)
        elif all(far(centre(c, H / 2), centre(r, H / 2), SPOT_CM) for r in rest):
            rest.append(c)
    up = min(len(a.objects) - 1 - len(rest), len(a.on) - 1) if pile else 0
    if up <= 0:  # nothing stands on anything
        all_there = sorted(pile[:1] + rest, key=lambda c: centre(c, H / 2)[0])
        return {n: one(n, c, H / 2) for n, c in zip(a.objects, all_there)}
    tower = a.on[-1 - up:]  # from the top down
    out = {n: dict(known[n], z_cm=(up - k + 0.5) * H) for k, n in enumerate(tower) if k and n in known}
    out[tower[0]] = one(tower[0], pile[0], (up + 0.5) * H)
    others = [n for n in a.objects if n not in tower]
    out.update({n: one(n, c, H / 2) for n, c in zip(others, sorted(rest, key=lambda c: centre(c, H / 2)[0]))})
    return out


def see(a):
    """Fresh positions, into what brain/act.py reads. Returns the objects never seen."""
    seen = robot.look(["overhead"], annotate=False)
    found = by_sam(robot.image(seen), a) if a.size else {o["name"]: o for o in seen["objects"]}
    known.update({n: found[n] for n in a.objects if n in found})
    tools._memory["objects"] = [dict(o) for o in known.values()]
    return [n for n in a.objects if n not in known]


def landed(held, on):
    """What a place just set down on another object. act.py has it where it was let go; its height comes from what it
    stands on, with no look in the recording (on the real arms a look is a SAM call: idle frames). act.py reads an
    object's top from size_cm, so the next place lands on top of this one."""
    mem = {o["name"]: o for o in tools._memory["objects"]}
    H = known[held]["size_cm"][2]
    top = mem[on]["size_cm"][2] + H
    mem[held].update(size_cm=mem[held]["size_cm"][:2] + [top], z_cm=top - H / 2)
    known[held] = dict(known[held], center_cm=mem[held]["center_cm"], z_cm=top - H / 2)


def holding(a):
    return (robot.status()["arms"][a.arm].get("held") or {}).get("name")


def home():
    tools._pitch.clear()
    tools._grip.clear()
    return robot.home(None, WIDTH_MM).get("outcome") == "done"


def spots(rng, a, order):
    """A random spot per object in its own box, in the order they move: apart from the spots before it and the objects
    still waiting."""
    for k in range(10000):
        s = {n: [round(rng.uniform(a.box[n][0], a.box[n][1]), 1), round(rng.uniform(a.box[n][2], a.box[n][3]), 1)]
             for n in order}
        if all(far(s[m], s[n], a.gap) and (k > 5000 or far(s[m], known[n]["center_cm"], a.gap))  # one that fell may be
               for i, m in enumerate(order) for n in order[i + 1:]):  # in the way of every spot: then it's bumped
            return s
    return None


def reset(rng, a):
    """The objects at new random spots and both arms home. Returns None, or why it didn't work."""
    if robot.status()["arms"][a.arm]["tip_cm"][2] < LOW_CM:  # a reset that went wrong: off what it let go of first
        act.run({a.arm: {"do": "open"}})
        act.run({a.arm: {"do": "move", "rel": [0, 0, a.lift]}})
    if not home() or see(a):
        return "the arms can't get home, or an object can't be seen"
    order = sorted(a.objects, key=lambda n: -known[n]["z_cm"])
    target = spots(rng, a, order)
    if target is None:
        return "no spots apart enough"
    off = []  # a pick that failed counts even if the object happens to stand near its spot: it may be turned so that the
    for n in order:  # jaws can't take it
        o = known[n]
        spec = {"do": "grasp", "object": n}
        H = o["size_cm"][2]
        if o["z_cm"] > H:  # it stands on something: the jaws close round it, above that
            spec.update(height=o["z_cm"] - H / 2 + TIPS_CLEAR_CM + act.DEEP_CM, lift=3, offset=[0, TURNED * H, 0])
        for step in (spec, {"do": "place", "at": target[n]}):
            r, stopped = act.run({a.arm: step})
            if stopped:
                raise SystemExit("a person stopped the arms")
            if step is spec and "failed" in r[a.arm]:
                off.append(n)
    if not home():
        return "the arms can't get home"
    see(a)
    off += [n for n in a.objects if n not in off and (far(known[n]["center_cm"], target[n], SPOT_CM)
                                                      or abs(known[n]["z_cm"] - known[n]["size_cm"][2] / 2) > Z_CM)]
    return f"{off} didn't get to their spots" if off else None


def worked(a):
    """Whether each object of the --on chain stands on the next, and by how much: xy and height apart, from the top."""
    xy, dz, ok = [], [], True
    for t, l in zip(a.on, a.on[1:]):
        top, low = known[t], known[l]
        xy.append(round(float(np.hypot(*np.subtract(top["center_cm"], low["center_cm"]))), 2))
        dz.append(round(float(top["z_cm"] - low["z_cm"]), 2))
        ok = ok and xy[-1] <= XY_CM and abs(dz[-1] - low["size_cm"][2]) <= Z_CM and not top.get("held_by")
    return ok, xy, dz


def trial(a, task):
    """One run of a trained policy from an episode's start. Returns (worked, xy_cm, dz_cm, seconds)."""
    begun = robot.policy_start(a.policy, a.max_s, task)
    if not begun.get("ok"):
        raise SystemExit("; ".join(begun.get("errors", [])))
    seconds, grip_free = 0.0, known[a.on[0]]["size_cm"][0] * 10 + 10
    while True:
        s = robot.status()
        if not s["policy"]["running"]:
            break
        seconds = s["policy"]["t"]
        see(a)
        if worked(a)[0] and s["arms"][a.arm]["width_mm"] > grip_free:
            robot.policy_stop("the task is done")
            break
        time.sleep(0.2)
    act.run({a.arm: {"do": "open"}})
    act.run({a.arm: {"do": "move", "rel": [0, 0, a.lift], "pitch": a.lift_pitch}})
    home()
    see(a)
    return (*worked(a), round(seconds, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan", help="a JSON file with the task, the plan's steps and its objects (our name -> the plan's)")
    ap.add_argument("--objects", required=True, help="the objects a reset moves, left to right, names with commas")
    ap.add_argument("--box", required=True, action="append", help="NAME=x0,x1,y0,y1 cm: where a reset sets that object")
    ap.add_argument("--on", required=True, help="success is this chain, from the top down: names with commas")
    ap.add_argument("--gap", type=float, default=8.5, help="cm between spots, for the jaws")
    ap.add_argument("--arm", default="left", help="the arm that resets and lifts clear")
    ap.add_argument("--lift", type=float, default=5.0, help="cm the arm lifts clear at the end of an episode")
    ap.add_argument("--lift-pitch", type=float, default=60.0, help="at this pitch, the same in every episode")
    ap.add_argument("--size", help="L,W,H cm of every object: find them with SAM (the real arms)")
    ap.add_argument("--prompt", help="--size: what SAM looks for")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--minutes", type=float, help="start no episode after this long")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", help="a trained policy does the task instead of the plan, and nothing is recorded")
    ap.add_argument("--max-s", type=float, default=30.0, help="--policy: the longest a trial runs")
    a = ap.parse_args()
    a.objects, a.on = a.objects.split(","), a.on.split(",")
    a.box = {b.split("=")[0]: [float(v) for v in b.split("=")[1].split(",")] for b in a.box}
    a.size = [float(v) for v in a.size.split(",")] if a.size else None
    if a.size and not a.prompt:
        ap.error("--size needs --prompt")
    plan = json.load(open(a.plan))
    text = json.dumps(plan["steps"])
    for ours, theirs in plan.get("objects", {}).items():
        text = text.replace(json.dumps(theirs), json.dumps(ours))
    steps = all_steps = json.loads(text)
    while steps and all((steps[-1].get(k) or {}).get("do", "park") == "park" for k in ("left", "right")):
        steps = steps[:-1]  # going home isn't part of an episode
    lift = {a.arm: {"do": "move", "rel": [0, 0, a.lift], "pitch": a.lift_pitch}}
    if not robot.status().get("armed"):
        raise SystemExit("the arms are off: switch torque on (key a in robotd)")
    rng, wanted = np.random.default_rng(a.seed), []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: wanted.append(True))
    t0, good, done = time.time(), 0, 0
    tools.reset()
    while done < a.episodes and not wanted and (not a.minutes or time.time() - t0 < a.minutes * 60):
        t = time.time()
        for _ in range(ATTEMPTS):
            why = reset(rng, a)
            if not why:
                break
            print(f"reset: {why}", flush=True)
        if why:
            if not sys.stdin.isatty():
                raise SystemExit(f"needs a person: {why}")
            input(f"{why}. Put the objects back in their boxes, then Enter ")
            continue
        start = {n: (known[n]["center_cm"], known[n].get("yaw")) for n in a.objects}
        t_reset = time.time() - t
        if a.policy:
            ok, xy, dz, seconds = trial(a, plan["task"])
            done, good = done + 1, good + ok
            print(json.dumps(dict(trial=done, ok=ok, xy_cm=xy, dz_cm=dz, seconds=seconds, start=start)), flush=True)
            print(f"{good}/{done} worked", flush=True)
            continue
        begin = robot.episode_begin(plan["task"], {"kind": "rehearsal", "plan": a.plan, "model": plan.get("model")},
                                    target_cm=known[a.on[-1]]["center_cm"])
        if not begin.get("ok"):
            raise SystemExit("; ".join(begin.get("errors", [])))
        results, stopped = [], None
        for step in steps + [lift]:
            spec, held = step.get(a.arm) or {}, holding(a)
            r, stopped = act.run(step)
            results.append(r)
            if stopped:
                break
            if spec.get("do") == "place" and spec.get("on") and held and not holding(a):
                landed(held, spec["on"])
        end = robot.episode_end(valid=not stopped, note=stopped)
        for step in all_steps[len(steps):]:
            act.run(step)
        home()
        see(a)
        ok, xy, dz = worked(a)
        robot.episode_label(begin["episode"], "judge", dict(success=ok, confidence=1.0, by="rehearse: object positions",
                                                            xy_cm=xy, dz_cm=dz, results=results, start=start))
        done, good = done + 1, good + ok
        print(json.dumps(dict(episode=begin["episode"], ok=ok, xy_cm=xy, dz_cm=dz, frames=end.get("frames"),
                              sim_s=end.get("duration_s"), reset_s=round(t_reset, 1), wall_s=round(time.time() - t, 1),
                              start=start, results=results)), flush=True)
        print(f"{good}/{done} worked, {done / (time.time() - t0) * 3600:.0f} episodes an hour", flush=True)
        if stopped:
            raise SystemExit(f"a person stopped the arms: {stopped}")


if __name__ == "__main__":
    main()
