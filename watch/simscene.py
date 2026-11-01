"""A simulator scene from a video: Muse says what the task's objects are (kind, real size, colour, how many), the code
lays them out on the robot's table where the arms reach and the overhead camera sees them, and writes a scene file
the HD simulator loads (harness/mjsim.py), so the arms rehearse on the things the video shows instead of a fixed scene.

    python -m watch.simscene <folder> [--out scene.json]     folder: the video mode's output (clip.mp4, muse.json,
                                                              score.json), e.g. watch/sam_out/videos/<name>
    python -m harness.robotd --sim --hd --objects scene.json  the simulator on it

Kinds the simulator has: "ball" (a sphere), "tube" (an open-topped container: can, cup, jar, bowl) and "box" (any
other solid thing, as a block). Where things go follows the video: what the person's left hand handles goes on the
left arm's side, the right hand's on the right, containers and shared things in the middle where both arms reach.
"""
import argparse
import json
import os

import numpy as np

KINDS = {"ball": "ball", "container": "tube", "box": "box"}
LIMITS = {"ball": ([2.0], [9.0]), "tube": ([3.0, 3.0], [15.0, 30.0]), "box": ([1.0, 1.0, 1.0], [25.0, 25.0, 25.0])}
DEFAULT_SIZE = {"ball": [6.7], "tube": [7.8, 21.0], "box": [6.0, 6.0, 6.0]}
MAX_EACH, MAX_ALL = 6, 10
# the table area both the arms and the overhead camera cover (cm); tall things lean outward in the overhead image, so
# they stay nearer
X_RANGE, Y_RANGE, TALL_Y_MAX, TALL_CM = (-24, 24), (15, 36), 32, 15
REACH_CM = (17, 36)  # from an arm's base: nearer is cramped, farther needs a flat gripper
GAP_CM = 2.5  # between footprints
ANCHOR = {"left": (-12, 26), "right": (12, 26), "middle": (3, 29)}

ASK = """These are frames of a video of a person's hands doing a task, with what a video model said happens. The
task's objects are to be rebuilt in a robot simulator that has three kinds of thing: "ball" (round, it rolls),
"container" (open at the top: a can, cup, jar, bowl, an open box) and "box" (any other solid thing, as a block).

What happens: {summary}
Objects named, with how many of each a segmentation model tracked: {named}

For each object the task involves, reply with its real-world size (estimate from familiar things: a tennis ball is
6.7 cm across, a can of 3 tennis balls 7.8 cm wide and 21 cm tall), its main colour, how many are on the table at the
start, and which of the person's own hands handles it (left, right, both, or none).
Reply with JSON only:
{{"objects": [{{"name": "<short concrete noun, as named above>", "kind": "ball" | "container" | "box",
  "size_cm": [diameter] for a ball | [outer diameter, height] for a container | [length, width, height] for a box,
  "color": "#rrggbb", "count": N, "hand": "left" | "right" | "both" | "none"}}]}}"""


def read(folder):
    """What the video mode found: (summary, [(name, instances)], clip path or None)."""
    muse = json.load(open(os.path.join(folder, "muse.json")))
    named = [(n, 1) for n in muse.get("objects", [])]
    score_path = os.path.join(folder, "score.json")
    if os.path.exists(score_path):
        objs = [o for o in (json.load(open(score_path)).get("vision") or {}).get("objects", [])
                if not o["name"].endswith(" hand")]
        named = [(o.get("muse_name") or o["name"], int(o.get("instances") or 1)) for o in objs] or named
    clip = os.path.join(folder, "clip.mp4")
    return muse.get("summary", ""), named, clip if os.path.exists(clip) else None


def describe(summary, named, clip, llm):
    """Muse's account of the objects: [{name, kind, size_cm, color, count, hand}] (kinds as the simulator names
    them)."""
    from brain.plan import image_part, parse, video_frames

    content = [{"type": "text", "text": ASK.format(summary=summary or "(not described)", named=", ".join(
        f"{n} ({k})" for n, k in named) or "(none)")}]
    for t, b64 in video_frames(clip, n=4) if clip else []:
        content += [{"type": "text", "text": f"video at {t} s:"}, image_part(b64)]
    reply = llm.ask("You describe objects in videos for a robot simulator. JSON only.",
                    [{"role": "user", "content": content}], "low")
    return clean(parse(reply.text).get("objects") or [])


def clean(objects):
    """Kinds the simulator has, sizes it can build, counts it can lay out."""
    out = []
    for o in objects:
        kind = KINDS.get(str(o.get("kind", "box")).lower(), "box")
        lo, hi = LIMITS[kind]
        size = [float(v) for v in (o.get("size_cm") or [])][:len(lo)]
        size = size if len(size) == len(lo) else DEFAULT_SIZE[kind]
        size = [round(min(max(v, a), b), 1) for v, a, b in zip(size, lo, hi)]
        color = str(o.get("color") or "#b4b4b4")
        if not (color.startswith("#") and len(color) == 7):
            color = "#b4b4b4"
        hand = str(o.get("hand") or "none").lower()
        out.append(dict(name=str(o.get("name") or kind).strip().lower(), kind=kind, size_cm=size, color=color,
                        count=max(1, min(int(o.get("count") or 1), MAX_EACH)),
                        hand=hand if hand in ("left", "right") else "middle"))
    return out


def footprint(o):
    s = o["size_cm"]
    return s[0] / 2 if o["kind"] in ("ball", "tube") else float(np.hypot(s[0], s[1])) / 2


def height(o):
    s = o["size_cm"]
    return s[0] if o["kind"] == "ball" else s[-1]


def layout(objects, bases=None):
    """Every instance a spot on the table: its arm's side (both arms' reach for the middle), clear of the others,
    in the camera's view; containers and the biggest things first. Returns (placed, notes)."""
    if bases is None:
        from harness.scene import Scene
        sc = Scene()
        bases = {s: sc.base_xy(s) * 100 for s in ("left", "right")}
    xs = np.arange(X_RANGE[0], X_RANGE[1] + 0.1, 1.0)
    ys = np.arange(Y_RANGE[0], Y_RANGE[1] + 0.1, 1.0)
    grid = np.array([(x, y) for x in xs for y in ys])
    dist = {s: np.linalg.norm(grid - b, axis=1) for s, b in bases.items()}
    ok_for = {s: (d >= REACH_CM[0]) & (d <= REACH_CM[1]) for s, d in dist.items()}
    ok_for["middle"] = ok_for["left"] & ok_for["right"]
    placed, notes, total = [], [], 0
    order = sorted(objects, key=lambda o: (o["kind"] != "tube", -footprint(o)))
    for o in order:
        for k in range(o["count"]):
            if total >= MAX_ALL:
                notes.append(f"left out {o['name']} {k + 1}: at most {MAX_ALL} things")
                continue
            r = footprint(o)
            side = "middle" if o["kind"] == "tube" else o["hand"]  # things go into a container from either side
            free = ok_for[side].copy()
            if height(o) > TALL_CM:
                free &= grid[:, 1] <= TALL_Y_MAX
            for p in placed:
                free &= np.linalg.norm(grid - p["xy_cm"], axis=1) >= r + p["r"] + GAP_CM
            if not free.any() and side != "middle":  # its side is full: anywhere either arm reaches
                free = (ok_for["left"] | ok_for["right"]).copy()
                for p in placed:
                    free &= np.linalg.norm(grid - p["xy_cm"], axis=1) >= r + p["r"] + GAP_CM
            if not free.any():
                notes.append(f"no room for {o['name']} {k + 1}")
                continue
            cand = grid[free]
            xy = cand[np.argmin(np.linalg.norm(cand - ANCHOR[side], axis=1))]
            name = o["name"] if o["count"] == 1 else f"{o['name']} {k + 1}"
            placed.append(dict(name=name, kind=o["kind"], size_cm=o["size_cm"], color=o["color"],
                               xy_cm=[float(xy[0]), float(xy[1])], yaw=0.0, r=r))
            total += 1
    for p in placed:
        del p["r"]
    return placed, notes


def build(folder, llm, out=None):
    """Read the video mode's folder, ask Muse, lay it out, write the scene file. Returns (path, scene)."""
    summary, named, clip = read(folder)
    objects = describe(summary, named, clip, llm)
    placed, notes = layout(objects)
    if not placed:
        raise ValueError("Muse named nothing the simulator can build")
    scene = dict(version=1, kind="sim_scene", source=os.path.abspath(folder), described=objects, notes=notes,
                 objects=placed)
    path = out or os.path.join(folder, "scene.json")
    json.dump(scene, open(path, "w"), indent=1)
    return path, scene


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="the video mode's output folder (clip.mp4, muse.json, score.json)")
    ap.add_argument("--out", help="where the scene goes (default: <folder>/scene.json)")
    ap.add_argument("--llm", help="backend (default: LLM in .env, else muse)")
    a = ap.parse_args()
    from brain import llm
    path, scene = build(a.folder, llm.backend(a.llm), a.out)
    for o in scene["objects"]:
        print(f"  {o['name']:<22} {o['kind']:<5} {o['size_cm']} cm  at {o['xy_cm']}  {o['color']}")
    for n in scene["notes"]:
        print(f"  note: {n}")
    print(f"wrote {path}\nrun it: python -m harness.robotd --sim --hd --objects {path}")


if __name__ == "__main__":
    main()
