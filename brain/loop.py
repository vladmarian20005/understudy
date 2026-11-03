"""Muse drives the arms one small step at a time and looks after every step: always a closed loop, each step as small
as "right arm, move a bit to the front".

Every turn Muse gets the overhead image, both wrist images and the arms' numbers, and answers with one action for one
arm: a step in the gripper's own directions, a move to a table point, a tilt, a wrist roll, open, close (until the
jaws stop on something and the motor current rises), home, or find (segmentation on the overhead image). robotd checks
every motion before it moves. Nothing happens that Muse didn't ask for.
"""
import base64
import json
import os
import re
import time

import numpy as np

from harness import robot

from . import see

SYSTEM = """You drive two robot arms (SO-101) on a table, one small step at a time. After every step you get fresh
images and numbers and choose the next step. Nothing moves unless you say so, and nothing happens by itself: when
something goes wrong, look at what you did and change it.

The table, in cm: x to the right, y away from the arms, z up from the table top. The left arm's base is at (-18, -1),
the right arm's at (18, -1). The overhead image has a 5 cm grid (x along the bottom, y up the left side) and each
gripper drawn on it: left magenta, right cyan (a dot at the fingertips, a cross on the table under them, its height).

Each gripper has a fixed finger and a jaw that swings in against it from a hinge 8 cm back. A camera on top of the
gripper looks along the fingers: in the wrist image the fingers come in from the bottom corners, the fixed finger on
the right, the swinging jaw on the left. The cyan line runs through the fingertips; the "grip" cross is the grip
point, 3 cm back from the fingertips and 3.5 cm left of the fixed finger: where the middle of what it takes should be.
Something is in the grip when it sits between the fingers below the fingertips line, on the cross. From afar the wrist
camera can't judge distance (a thing far beyond the fingertips also looks between them): use the numbers to get close.
Close up, use both: from_grip says how far, the wrist image shows whether it is really between the fingers. Go by
whichever says it is farther (twice a clear tube looked deep in the wrist image while it sat at the fingertips; the
jaws pushed it out).

The jaws open up to 98 mm at the fingertips, less deeper in. While closing, the swinging jaw pushes what it holds
toward the fixed finger and out toward the fingertips, so take things deep: their middle on the grip cross, not at
the fingertips.

Both arms start in the ready pose: 28 cm up, gripper level, pointing straight ahead (+y), jaws nearly shut.
Reach: about 15-45 cm from each base, a little past x = 0, not the far side. Level, 30 cm out from a base, the grip
point goes up to about 34 cm; farther out the top is lower (40 cm out: about 26). Low, near the table, a level gripper
reaches only 35-45 cm out from its base; tilted down 30-45 it reaches low from about 28 cm out.

Every turn you get, per arm: the fingertips and the grip point (x, y, z), pitch (0 level, 90 straight down), the jaw
opening (mm), the gripper motor's current (mA: about 5 idle, 40-300 while squeezing something, more with more force)
and what the gripper is doing. Per thing found: its centre and size on the table, and from_grip: where its middle is
from each grip point, as [forward, right, up] cm in that gripper's own directions, the same ones "move" takes. A
thing's numbers are from when it was last found: an arm can push it, and a held thing moves with the gripper. Find
again when it matters.

Actions, one per reply, for one arm:
- {"arm": A, "do": "move", "forward": cm, "right": cm, "up": cm, "speed": S}: a step in the gripper's own directions
  (forward: the way the fingers point, kept flat; right: to the right of that; up: straight up; negative: the other
  way; any left out are 0). It keeps the gripper's pitch and roll. At most 10 cm.
- {"arm": A, "do": "move_to", "x": cm, "y": cm, "z": cm, "pitch": deg, "speed": S}: the grip point (and what it
  holds) to a table point, for bigger moves through free space. pitch: as it is now if left out.
- {"arm": A, "do": "pitch", "deg": D}: tilt the gripper about its grip point: 0 level, 90 down.
- {"arm": A, "do": "roll", "deg": D}: turn the wrist by D degrees from where it is now (relative: 0 does nothing)
  about the fingers: what it holds tilts sideways. Positive tips its top to the left (-x), negative to the right
  (+x). It stays turned until you roll back (roll 25 is undone by roll -25) or go home. Never with a container in
  the jaws: it stays upright (refused).
- {"arm": A, "do": "open", "mm": W}: open to W mm (0-98).
- {"arm": A, "do": "close", "force": F, "gentle": true or false}: two ways:
  firm (the default for anything narrower than 75 mm, like a ball): close at once and keep squeezing at F % of the
  motor's torque (default 25): the jaws stop on it and the current rises to 150-250 mA. This is what held the balls.
  gentle (the default for wide things, like a tube; or "gentle": true): close step by step until the jaws meet a load,
  then hold just past it ("squeeze_mm", default 2): firm, not crushing, so a container's opening stays round (81 mm
  tube held at 75; squeezed hard it pinched shut and the balls couldn't drop in). The current then is about 25-40 mA.
  The result says where the jaws stopped and the current.
- {"arm": A, "do": "home"}: back to the ready pose, out of the overhead camera's way.
- {"do": "find", "things": ["tennis ball", "clear tube"]}: segmentation looks for these on the overhead image and
  gives their centres and sizes. Short concrete nouns, with colour; a thing it misses may be found by another word
  (a white tube only as "white can").
- {"do": "done", "success": true or false, "summary": "..."}: the task is finished, or can't be done.
S, the speed: "very_slow" for the last centimetres before touching, "slow" (the default) near things, "normal" or
"fast" only through free space.

How to pick something up:
1. find it. Open the jaws 15-20 mm wider than it (at most 98).
2. move_to a point on the arm's side of it, 8-10 cm before it, at the height of its middle: a standing thing at
   about half its height, a ball at its radius. Tall things: level, never from above.
   Low things (a ball, a cube): tilted down 30-45, or level if they are 35 cm or more from the base. If it's refused,
   tilt more or start closer.
3. Line up: move right or left until from_grip right is about 0, up or down until from_grip up is about 0. For left
   and right trust from_grip: with the jaws open wide, a thing lined up right sits left of the middle between the
   fingers in the wrist image, on the grip cross (which is 3.5 cm left of the fixed finger), not midway between them
   (going by the image's middle, an arm went left and right around a ball for ten steps).
4. Go forward: 3 cm steps while it's far, then 1-2 cm, the last ones very_slow. For a wide thing (a tube) within 6 cm
   steps are cut to 1.5 cm and it is found again after each ("pushed" in the result: it moved; the result says
   whether that is as deep as it goes, so close there, or to back off and line up again). Re-centre after each step,
   until
   it is fully inside the gripper, never at the fingertips: from_grip forward at most 1 cm (close refuses before
   that), and in the wrist image it sits below the fingertips line with both fingers reaching past its middle along
   its sides. If the numbers say it's in but the wrist image shows it at or beyond the fingertips line, go on forward
   1-2 cm at a time (a stretched arm sags short). If it moves or tips, stop and back off: never push it.
5. close. "closed on nothing": it wasn't deep enough. Jaws stopped on it with the current up (100+ mA): go to 6,
   even if the jaws are narrower than its width: soft things (tennis balls, plastic tubes) give 15-30 mm.
6. move up 2 cm, slowly: the lift decides. It came up with the jaws (between them in the wrist image, the current
   still up, the jaws not narrower): it is held, keep it and go on. Only if it stayed down or slipped out: back off
   5 cm with the jaws as they are, then open, then line up again deeper (a held tennis ball was let go because
   it looked shallow; an open jaw backing out knocked a standing tube over).

The two grasps that worked, step by step: do them the same way.
- A standing tube: open 98; move_to 8 cm before it on the arm's side, at about half its height (11-12 cm for a 24 cm
  tube: higher up, at 14-16 cm, the jaws pushed it away every time), pitch 0; centre it (move right/left);
  forward 3, 3, then small steps until from_grip forward is about 0 or a push says it is as deep as it goes; close
  gentle (it met the 81 mm tube at 69-75 mm and held it there, the opening still round); move up 2: it came up.
- A ball on the table: open 90; move_to 9 cm before it on the line from the arm's base, z = the ball's radius (3.5),
  pitch 35, so the grip point is at its middle; centre it; forward 3, 2, then 2 and 1.5 very_slow, keeping from_grip
  up within 1, until from_grip forward is about 0; close firm, force 25 (61 mm, 221 mA); move up 2, then pitch 0
  and straight up.
  Never come at a ball from above its middle: the jaws slide over it (two misses that way), and every touch
  rolls it away.

Putting something into a container (go up right after picking it up, then toward the container):
1. Right after the lift test, go straight up (move up, gripper level), then move back toward your own base and up,
   to about (-15, 25) for the left arm, z 35: the arm reaches highest near its base (35-36 cm there, only 31 cm
   over x 2, y 29). Wait there, high and on your own side, until the container is in place. The rim of a container
   the other arm holds is its "top end" in that arm's numbers; "at least" means the camera cut its top off and the
   real rim can be up to 5 cm higher.
2. Then across, high (z 35, or as high as it reaches there: over the middle about 31-33), to right above the opening:
   move_to the top end's x and y. After every move
   of the holding arm the container is measured on the overhead image ("measured" in that step's result) and its
   top end in the numbers is where it was seen: go by it (by the arm's numbers alone the opening was 4.5 cm off,
   and a ball dropped on the rim bounced off the table).
3. Check in the overhead image that what it holds is inside the rim's circle, not over the rim. Then lower straight
   down in 2 cm steps until its bottom is 3-5 cm above the rim (check the grip point's height after each step: a
   stretched arm stops short), and open only 15 mm wider than what it holds (a jaw opened wide swings out into the
   rim). open is refused while what it holds, seen on the overhead image, is more than 1.5 cm from right over the
   opening, or its bottom more than 6 cm above the rim (let go 9.5 cm over it, the ball missed): do what the
   refusal says, then open. Don't go lower: lower down near the
   tube the arms tangled; a ball dropped from 3-5 cm still goes in. Near the container's top, below 8 cm over
   it (11 when its top is "at least"), moves go only straight up or down (refused otherwise), except small steps
   right over the opening to line up: line up above it while high.
4. After opening: straight up 8-10 cm first, the jaws as they are, then away, level. Never down or sideways while the
   gripper is next to the container (the open jaw caught the tube after the drop).
Never carry it low past the container: it knocks it.

Holding a container for the other arm to fill: take it level by its middle, lift it only 1-2 cm, and keep it
standing straight up the whole time: never roll it (tilted, balls dropped right over
the opening missed it; roll is refused while it holds one). Bring it to the middle, a little toward the other
arm, where that arm still reaches high over it: for the right arm holding a tube for the left, the grip point at
about (-3, 25, 13) (its opening then about 5 cm farther out from the right base than the grip point: the measured
top end says where). Keep it still while the other arm works.
Take turns, in this order (both at once, the arms got tangled):
1. The holding arm takes the container and lifts it 1-2 cm, and it waits there, upright, on its own side.
2. The other arm picks up the thing to put in and goes straight up with it, high.
3. Only then does the container come to the middle, upright. The other arm comes over it from above and drops the
   thing in, then goes up and back to its own side.
4. Before the other arm goes for the next thing, the container goes back to its side.
Never both arms low in the middle. A move is refused if a gripper would come within 10 cm of the other gripper, or
into the container the other arm holds below its top.

Rules:
- One arm at a time. Keep the arms at least 10 cm apart.
- Near things: steps of 3 cm or less, and look after each. A fallen or rolled-away thing costs far more than a slow
  step.
- "refused" means nothing moved: read why (out of reach, the table, too close to the other arm) and change something.
  "short": the arm stopped before where it was sent.
- Never open while holding something unless it is right where it should go: it drops.
- Keep count in your plan of what you have put where ("ball 1 in the tube"). A clear or deep container hides what is
  inside it: when your count says all are in and find shows none left on the table, finish the task's last steps and
  call done (after both balls were in, the arms kept searching for a second ball).
- The task's things are on the table under the overhead camera's grid. Things the wrist cameras see beyond it (the
  desk behind, the room) aren't part of the task and are out of reach: if one of the task's things falls off the
  grid, call done with success false and say where it went (after a miss the left arm reached for balls lying on the
  desk behind until it was stopped).
- Reply with JSON only:
{"see": "what the images and numbers show, 1-2 sentences; for the arm you'll move: where the thing is from its jaws",
 "plan": "the steps still to do, short",
 "arm": "left" or "right", "do": "...", ...the action's fields}
"""

MAX_TURNS = 80
KEEP = 6  # Muse sees the task and the last KEEP exchanges; only the newest carries images
MAX_STEP_CM = 10.0
SPEEDS = ("very_slow", "slow", "normal", "fast")
FORCE = 25.0  # % torque close squeezes with unless Muse says (real arms: 12 % draws ~40 mA on a cube, 30 % ~260)
MIN_FORCE, MAX_FORCE = 10.0, 60.0
STROKE_MM = 98.0
OPEN_WAIT_S = 2.0  # open waits at most this long for the jaws to get there
SETTLE_S = 0.5
LOAD_MA = 60.0  # close stops at this motor current: it has met something (idle about 5, squeezing hard 150-250)
SQUEEZE_MM = 2.0  # then is held this far past where it met the load: firm, not crushing (a tube crushed
MAX_SQUEEZE_MM = 15.0  # 81 -> 58 mm pinched shut and the balls stuck; held at 75 its opening stayed round)
STALL_MM = 0.5  # a step that closes the jaws less than this: they have met a load
CLOSE_STEP_MM = 4.0  # it feels its way in steps of this
NEAR_MM = 10.0  # starting this much wider than the thing between the jaws
MAX_CLOSE_STEPS = 60
STEP_SETTLE_S = 0.1  # the jaws come to rest on what they closed on before they're read
HOLD_MIN_MM = 3.0  # stopped wider than this: something is between the jaws
HOLD_MA = 15.0  # and the motor pushes on it with more than this (idle 0-6; a light hold past the load 25-40)
SLIP_MM = 6.0  # this much narrower than where they stopped: it's slipping out
FINAL_CM = 6.0  # forward moves this close to the thing ahead go in steps of at most FINAL_STEP_CM, very slowly,
FINAL_STEP_CM = 1.5  # and the thing is found again after each: if it moved more than PUSH_CM, it is being pushed
PUSH_CM = 1.0
WIDE_MM = 75.0  # only a thing this wide stops short in the jaws (they narrow behind the fingertips): a narrower one
# that moves is being pushed (a ball rolls)
GRIP_BACK_TIPS_CM = 5.0  # a pushed thing whose middle is less than this ahead of the grip point is as deep as it goes
LOST_CM = 8.0  # a detection farther than this from where it was is another thing
LEVEL_CM = 1.5  # nor while a thing shorter than TALL_CM has its middle farther than this above or below it
DEEP_CM = 1.0  # close refuses while a thing's middle is farther than this in front of the grip point
SQUASH = 0.8  # jaws stopped narrower than this share of a thing's width are squashing it (or hold only its edge)
JAWS_CM = 5.0  # a thing this close across the jaws is the one between them
MAX_TILT = 2.0  # deg from upright a held tall thing may be rolled: it stays upright (tilted 15-25 deg, two
# balls dropped right over the measured opening missed it)
ARMS_CM = 10.0  # a move may not bring a grip point nearer than this to the other one (unless it moves away)
HELD_CLEAR_CM = 7.0  # nor this near to the axis of the tall thing the other arm holds, below its top
FILL_XY = np.array([-3.0, 25.0])  # cm: where a held container stands to be filled (its holder's grip point)
IN_PLACE_CM = 6.0  # an arm holding something comes within WAIT_CM of the tall thing the other arm holds only once
WAIT_CM = 15.0  # that arm's grip point is this near FILL_XY (in place for filling)
DROP_MAX_CM = 6.0  # open over the opening lets go only with what it holds at most this far above the rim
OVER_TOP_CM = 4.5  # right over the opening (within this of the top end) a straight lowering may go down to
DROP_OVER_CM = 3.0  # this far over the listed top: that is the drop
OPEN_OVER_CM = 1.5  # and open lets go there only with what it holds, seen on the overhead image, this near to right
# over the opening (dropped 4.5 cm off it, the ball hit the rim and bounced off the table)
ABOVE_CM = 8.0  # within NEAR_TOP_CM of the top of a tall thing the other arm holds, below this much over it, only
NEAR_TOP_CM = 12.0  # straight up or down (or away without going down): it is come at from above, never from under
LINE_UP_CM = 5.0  # but at least this high over its top, steps toward right over the opening line up for the drop
STRAIGHT_CM = 1.5  # a move with less sideways than this is straight up or down
FIND_PAST_TOP_CM = 5.0  # find skips what it sees on a held tall thing up to this far past its listed top
UNSURE_TOP_CM = 3.0  # and this much higher over a held thing whose top the camera cut off
TALL_CM = 10.0  # a held thing this long gets its ends listed
SHORT_CM = 1.0  # an arm that ends farther than this from where it was sent is short


def seg_dist(p, a, b):
    """How far point p is from the segment a-b (cm)."""
    ab = b - a
    k = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + k * ab)))


def tube_ends(grip, ax, tilt, ahead, across, below, length):
    """(bottom, top) cm of a held tall thing: the grip point and the gripper's (forward, right, up) directions, its tilt
    from upright (deg, + tips its top to the left), its axis ahead cm in front of the grip point and across cm across
    the jaws, below cm of it under the grip point."""
    phi = np.radians(tilt)
    axis = np.cos(phi) * ax[2] - np.sin(phi) * ax[1]
    mid = grip + ahead * ax[0] + across * (np.cos(phi) * ax[1] + np.sin(phi) * ax[2])
    return mid - below * axis, mid + (length - below) * axis


def parse(text):
    """The JSON object in a reply (it may come with a fence or a stray word)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def image_part(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def r1(v):
    return [round(float(x), 1) for x in v]


def video_task(task, summary):
    """The task with what the person in its video does (Muse's summary of the video)."""
    return (task + "\n" if task else "") + "Do with the two arms what the person in the video does (their left " \
        f"hand is the left arm): {summary}"


def watch(path):
    """What the person in a video does, from watch's cached Muse reading (watch/sam_out), else read now."""
    stem = os.path.splitext(os.path.basename(path))[0]
    here = os.path.join(os.path.dirname(__file__), "..")
    cache = os.path.join(here, "watch", "sam_out", "videos", stem, "muse.json")
    if os.path.exists(cache):
        return json.load(open(cache)).get("summary", "")
    import pathlib
    import sys
    sys.path.insert(0, os.path.join(here, "watch"))
    from sam_pipeline.credentials import api_key
    from sam_pipeline.muse_client import MuseClient, PROMPT
    c = MuseClient(api_key(), "https://api.meta.ai/v1")
    return c.watch(c.upload(pathlib.Path(path)), PROMPT).summary


class Run:
    def __init__(self, llm, log=print, run_dir=None, effort="low"):
        self.llm, self.log, self.run_dir, self.effort = llm, log, run_dir, effort
        self.messages, self.record = [], []
        self.stats = dict(tokens_in=0, tokens_out=0, cost=0.0, llm_seconds=0.0, calls=0, steps=0)
        self.pitch = {s: 0.0 for s in see.SIDES}  # the pitch each arm was last sent to (a stretched arm sags steeper
        # than sent: sending back the measured one ratchets it down)
        self.closed = {s: None for s in see.SIDES}  # where the jaws stopped on the last close; None: open since
        self.held = {s: None for s in see.SIDES}  # what each arm closed on: {name, length (cm), below (cm of it under
        # the grip point), roll0 (the wrist roll then)}: where its ends are now follows from the roll
        self.deepest = {}  # side -> the thing a push showed is as deep in the jaws as it goes
        self.things, self.found_at, self.labels = [], None, []

    # ---- the record --------------------------------------------------------------------------------------------
    def save(self, **rec):
        rec = dict(t=round(time.time(), 2), **rec)
        self.record.append(rec)
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "turns.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def save_images(self, images, n):
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            for cam, b64 in images.items():
                with open(os.path.join(self.run_dir, f"{n:02d}_{cam}.jpg"), "wb") as f:
                    f.write(base64.b64decode(b64))

    # ---- what Muse is told -------------------------------------------------------------------------------------
    def gripper(self, side, a):
        w, ma, w0 = a["width_mm"], a.get("grip_ma"), self.closed[side]
        if w0 is None:
            return "holding nothing"
        if w < HOLD_MIN_MM:
            return "closed on nothing"
        if ma is not None and ma < HOLD_MA:
            return f"closed at {w:.0f} mm but the motor isn't pushing ({ma} mA): it may hold nothing"
        if w < w0 - SLIP_MM:
            return f"squeezing something, but {w0 - w:.0f} mm narrower than where the jaws stopped: slipping out?"
        return f"squeezing something {w:.0f} mm wide"

    def state(self, arms):
        lines, grips = ["Arms now:"], {}
        for s in see.SIDES:
            a = arms[s]
            tip, grip, ax = see.axes(s, a["joints"])
            grips[s] = (grip, ax)
            roll = self.roll(s, a)
            lines.append(f"- {s}: fingertips {r1(tip)}, grip point {r1(grip)}, pitch {round(a['pitch'])}, roll "
                         f"{round(roll)} (0 upright), jaws "
                         f"{a['width_mm']:.0f} mm" + ("" if "grip_ma" not in a else f", motor {a['grip_ma']} mA") +
                         f"; {self.gripper(s, a)}")
            h = self.held[s]
            if h and self.closed[s] is not None and a["width_mm"] >= HOLD_MIN_MM:
                line = f"  {s} holds {h['name']}"
                ends = self.ends(s, a)
                if ends:
                    line += (f", {h['length']:.0f}{' or more' if h.get('at_least') else ''} cm long: its top end "
                             f"(the opening, if it has one) is now at {'least at ' if h.get('at_least') else ''}"
                             f"{r1(ends[1])}, its bottom end at {r1(ends[0])}, tilted "
                             f"{abs(round(roll - h['roll0'] + h.get('tilt', 0.0)))} deg from upright ("
                             + ("measured on the overhead image" if h.get("measured") else
                                "by the arm's numbers: not seen on the overhead image since it last moved") + ")")
                lines.append(line)
        if self.things:
            ago = self.stats["steps"] - self.found_at
            lines.append("Things found " + ("just now" if ago == 0 else f"{ago} steps ago") + ":")
            holders = {h["name"]: s for s, h in self.held.items() if h and self.closed[s] is not None}
            for o in self.things:
                if o["name"] in holders:
                    lines.append(f"- {o['name']}: held by the {holders[o['name']]} arm (see above)")
                    continue
                c = np.array(o["center_cm"] + [o["z_cm"]], dtype=float)
                rel = {s: r1([(c - g) @ v for v in ax]) for s, (g, ax) in grips.items()}
                lines.append(f"- {o['name']}: centre {r1(o['center_cm'])}, size {r1(o['size_cm'])} (long, short, "
                             f"tall), {o['shape']}; from_grip left {rel['left']}, right {rel['right']}")
        else:
            lines.append("Nothing found yet: use find.")
        return "\n".join(lines)

    def ask(self, content):
        """One user message in the running conversation; the parsed JSON reply."""
        if len(self.messages) > 2 * KEEP:  # the task stays, then Muse's reply from KEEP turns back onward
            del self.messages[1:len(self.messages) - 2 * KEEP + 1]
        for m in self.messages:  # only the newest message carries images
            if isinstance(m.get("content"), list):
                m["content"] = [p for p in m["content"] if p.get("type") != "image_url"]
        self.messages.append({"role": "user", "content": content})
        for attempt in range(2):
            self.stats["calls"] += 1
            r = self.llm.ask(SYSTEM, self.messages, self.effort)
            for k, v in (("tokens_in", r.tokens_in), ("tokens_out", r.tokens_out), ("cost", r.cost),
                         ("llm_seconds", r.seconds)):
                self.stats[k] += v
            self.messages.append(r.message)
            try:
                out = parse(r.text)
                self.save(muse=r.text, seconds=round(r.seconds, 1))
                return out
            except ValueError as e:
                self.messages.append({"role": "user", "content": f"That wasn't valid JSON ({e}). JSON only."})
        raise RuntimeError("Muse didn't answer with JSON")

    # ---- the actions -------------------------------------------------------------------------------------------
    def motion(self, side, kf, speed, target=None):
        r = robot.move(**{side: [kf]}, speed=speed)
        if r.get("outcome") != "done" and "speed is one of" in str(r.get("why")):  # robotd started before very_slow
            r = robot.move(**{side: [kf]}, speed="slow")
        out = self.outcome(r)
        if r.get("outcome") in ("done", "failed", "stopped"):
            tip = robot.status()["arms"][side]["tip_cm"]
            out["fingertips_now"] = tip
            if target is not None and np.linalg.norm(np.subtract(tip, target)) > SHORT_CM:
                out["short"] = f"{np.linalg.norm(np.subtract(tip, target)):.1f} cm from where it was sent {r1(target)}"
        return out

    @staticmethod
    def outcome(r):
        if r.get("outcome") == "done":
            return {"outcome": "done"}
        return {"outcome": r.get("outcome", "refused"), "why": r.get("why") or r.get("fail"), "fail": r.get("fail")}

    def act(self, a, seen):
        """Do one action; after a move of an arm holding a tall thing, and on find, measure that thing again."""
        out = self.step(a, seen)
        moved = a.get("do") in ("move", "move_to", "pitch", "roll") and out.get("outcome") == "done"
        for s in see.SIDES:
            h = self.held[s]
            if h and h["length"] >= TALL_CM and (a.get("do") == "find" or (moved and a.get("arm") == s)):
                out.update(self.measure(s))
        return out

    def step(self, a, seen):
        do, side = a.get("do"), a.get("arm")
        if do == "find":
            names = [str(n) for n in a.get("things") or []]
            if not names or "raw_overhead" not in seen:
                return {"outcome": "refused", "why": "find needs things (a list of names) and the overhead camera"}
            try:
                self.things = see.find(seen["raw_overhead"], names, seen["arms"], self.held_ends(seen["arms"]))
            except Exception as e:  # the service busy or down: Muse decides what then
                return {"outcome": "failed", "why": f"segmentation failed: {e}"}
            self.found_at, self.labels = self.stats["steps"], self.things
            return {"outcome": "done", "found": [o["name"] for o in self.things] or "nothing"}
        if side not in see.SIDES:
            return {"outcome": "refused", "why": 'say which arm: "arm": "left" or "right"'}
        speed = a.get("speed") or "slow"
        if speed not in SPEEDS:
            return {"outcome": "refused", "why": f"speed is one of {', '.join(SPEEDS)}"}
        arm = seen["arms"][side]
        if do == "move":
            tip, g, (f, r, u) = see.axes(side, arm["joints"])
            fw, note = float(a.get("forward") or 0), None
            o, ahead_cm, _ = self.ahead(side, arm)
            close_in = (o is not None and fw > 0 and ahead_cm < FINAL_CM  # wide things only: a ball goes in the way
                        and min(o["size_cm"][:2]) * 10 >= WIDE_MM)  # that worked, no checks
            if close_in:  # the last centimetres: small, very slow, and look whether it moved (3 cm steps
                # pushed a tube 7 cm away instead of taking it into the jaws)
                if fw > FINAL_STEP_CM:
                    note = f"forward cut to {FINAL_STEP_CM} cm: this close to {o['name']} it goes in small steps"
                    fw = FINAL_STEP_CM
                speed = "very_slow"
            d = fw * f + float(a.get("right") or 0) * r + float(a.get("up") or 0) * u
            if np.linalg.norm(d) > MAX_STEP_CM:
                return {"outcome": "refused", "why": f"at most {MAX_STEP_CM:.0f} cm: use move_to for bigger moves"}
            clash = self.clash(side, seen["arms"], g + d)
            if clash:
                return {"outcome": "refused", "why": clash}
            out = self.motion(side, {"rel": r1(d), "pitch": self.pitch[side]}, speed, np.array(arm["tip_cm"]) + d)
            if note:
                out["note"] = note
            if close_in and out["outcome"] == "done":
                moved = self.refind(o)
                if moved is not None and moved > PUSH_CM:
                    _, now_cm, _ = self.ahead(side, robot.status()["arms"][side])
                    wide = min(o["size_cm"][:2]) * 10 >= WIDE_MM
                    if wide and now_cm < GRIP_BACK_TIPS_CM:  # a wide thing already between the fingertips: as deep
                        # as a thing this wide goes (an 81 mm tube was pushed at the same depth every time;
                        # earlier runs closed right there and held it)
                        self.deepest[side] = o["name"]
                        out["pushed"] = (f"{o['name']} moved {moved:.1f} cm: it is as deep in the jaws as it goes "
                                         "(the jaws narrow behind the fingertips). Don't push further: close now")
                    else:
                        out["pushed"] = (f"{o['name']} moved {moved:.1f} cm with that step: the gripper is pushing "
                                         "it, not taking it in. Back off 3 cm, line up again (its middle on the "
                                         "grip cross, the jaws wide open around it), then come in again")
            return out
        if do in ("move_to", "pitch"):
            if do == "move_to":
                try:
                    g = np.array([float(a["x"]), float(a["y"]), float(a["z"])])
                except (KeyError, TypeError, ValueError):
                    return {"outcome": "refused", "why": "move_to needs x, y and z (cm)"}
                pitch = float(a["pitch"]) if a.get("pitch") is not None else self.pitch[side]
                head = g[:2] - see.scene().base_xy(side) * 100  # the fingers point away from the base
                clash = self.clash(side, seen["arms"], g)
                if clash:
                    return {"outcome": "refused", "why": clash}
            else:
                _, g, (f, _, _) = see.axes(side, arm["joints"])
                pitch, head = float(a.get("deg", 0)), f[:2]
            tip = see.tip_for(g, head, pitch)
            out = self.motion(side, {"xy": r1(tip[:2]), "z": round(float(tip[2]), 1), "pitch": pitch}, speed, tip)
            if out["outcome"] == "done":
                self.pitch[side] = pitch
            return out
        if do == "roll":
            deg, h = float(a.get("deg", 0)), self.held[side]
            if h and h["length"] >= TALL_CM and self.closed[side] is not None:
                tilt = self.roll(side, arm) + deg - h["roll0"]
                if abs(tilt) > MAX_TILT:
                    return {"outcome": "refused", "why": f"that would tilt {h['name']} {abs(tilt):.0f} deg from "
                            f"upright: it stays straight up (now "
                            f"{self.roll(side, arm) - h['roll0']:.0f}, roll {h['roll0'] - self.roll(side, arm):.0f} "
                            "stands it upright)"}
            return self.motion(side, {"roll": deg}, speed)
        if do == "open":
            off = self.off_opening(side, seen)
            if off:
                return {"outcome": "refused", "why": off}
            mm = min(max(float(a.get("mm", STROKE_MM)), 0.0), STROKE_MM)
            out = self.outcome(robot.move(**{side: [{"width": mm}]}, speed="slow"))
            self.closed[side], self.held[side] = None, None
            self.deepest.pop(side, None)
            end = time.time() + OPEN_WAIT_S  # the jaws are still swinging out when the move ends (47 of 98 mm)
            while time.time() < end and abs(robot.status()["arms"][side]["width_mm"] - mm) > 3:
                time.sleep(0.1)
            return out
        if do == "close":
            short = self.not_in(side, arm)
            if short:
                return {"outcome": "refused", "why": short}
            pct = min(max(float(a.get("force") or FORCE), MIN_FORCE), MAX_FORCE)
            robot.grip_force(side, pct)  # firm: what it squeezes with; gentle: a ceiling, the jaws stop well before
            o, ahead_cm, _ = self.ahead(side, arm)
            wide = o is not None and ahead_cm <= GRIP_BACK_TIPS_CM and min(o["size_cm"][:2]) * 10 >= WIDE_MM
            gentle = a.get("gentle") if a.get("gentle") is not None else wide
            if not gentle:  # firm: close and keep squeezing at the cap (held where they stopped, the jaw stops
                out = self.outcome(robot.move(**{side: [{"width": 0}]}, speed="slow"))  # pushing and things slide
                time.sleep(SETTLE_S)  # out)
                if robot.status()["arms"][side]["width_mm"] >= HOLD_MIN_MM:
                    robot.move(**{side: [{"width": 0}]}, speed="slow")
                return self.after_close(side, arm, out, pct, "firm")
            o = self.between(side, arm)
            if o is not None:  # quickly to just wider than it, then feel the way in
                near = round(min(o["size_cm"][:2]) * 10) + NEAR_MM
                if near < arm["width_mm"] - 1:
                    robot.move(**{side: [{"width": near}]}, speed="slow")
            met, before = None, None
            for _ in range(MAX_CLOSE_STEPS):  # a load: the jaws stop closing (2 mm steps: the current stays near
                now = robot.status()["arms"][side]  # 30 mA, it only climbs when squeezed hard, on the real arms)
                w = now["width_mm"]
                if now.get("grip_ma", 0) >= LOAD_MA or (before is not None and before - w < STALL_MM):
                    met = w
                    break
                if w < HOLD_MIN_MM:
                    break
                before = w
                robot.move(**{side: [{"width": max(0.0, w - CLOSE_STEP_MM)}]}, speed="slow")
                time.sleep(STEP_SETTLE_S)
            out = {"outcome": "done"}
            if met is not None:  # hold a little past where it met the load
                squeeze = min(max(float(a.get("squeeze_mm") or SQUEEZE_MM), 0.0), MAX_SQUEEZE_MM)
                robot.move(**{side: [{"width": max(0.0, met - squeeze)}]}, speed="slow")
                time.sleep(SETTLE_S)
                out.update(met_load_at_mm=round(met, 1), squeeze_mm=squeeze)
            return self.after_close(side, arm, out, pct, "gentle")
        if do == "home":
            out = self.outcome(robot.home([side]))
            if out["outcome"] == "done":
                self.pitch[side] = 0.0
            return out
        return {"outcome": "refused", "why": f"no action {do!r}"}

    def after_close(self, side, arm, out, pct, how):
        """Read where the jaws stopped, note what they hold, and say so."""
        now = robot.status()["arms"][side]
        w = now["width_mm"]
        self.closed[side] = w
        out.update(how=how, force=pct, jaws_stopped_mm=round(w, 1))
        o, ahead_cm, _ = self.ahead(side, arm)  # what it closed on: between the jaws, or a wide thing that stopped
        if o is not None and ahead_cm > GRIP_BACK_TIPS_CM:  # short in them
            o = None
        if o is not None and w >= HOLD_MIN_MM:
            wide = round(min(o["size_cm"][:2]) * 10)
            out["between_the_jaws"] = f"{o['name']}, about {wide} mm wide"
            _, g, _ = see.axes(side, arm["joints"])
            self.held[side] = dict(name=o["name"], word=o.get("word", o["name"]), length=o["size_cm"][2],
                                   roll0=self.roll(side, arm), below=float(g[2] - (o["z_cm"] - o["size_cm"][2] / 2)),
                                   at_least="least" in str(o["shape"]), radius=min(o["size_cm"][:2]) / 2,
                                   ahead=float(np.clip(ahead_cm, 0.0, GRIP_BACK_TIPS_CM)))  # a wide thing stops
            # short in the jaws, its axis in front of the grip point (a tube's 4 cm)
            if w < SQUASH * wide:
                out["note"] = f"the jaws are {wide - w:.0f} mm narrower than it: soft, or not at its widest; " \
                              "the lift tells whether it holds"
        if "grip_ma" in now:
            out.update(current_ma=now["grip_ma"], peak_ma=now.get("grip_peak_ma"))
        return out

    def ahead(self, side, arm):
        """The found thing in line with the jaws, nearest the grip point, with how far its middle is in front of the
        grip point (cm); (None, 0) if none."""
        _, g, ax = see.axes(side, arm["joints"])
        near = []
        for o in self.things:
            c = np.array(o["center_cm"] + [o["z_cm"]], dtype=float)
            fwd, right, up = [float((c - g) @ v) for v in ax]
            if abs(right) <= JAWS_CM and abs(up) <= o["size_cm"][2] / 2 + 2 and -JAWS_CM < fwd < 3 * JAWS_CM:
                near.append((abs(fwd) + abs(right), fwd, up, o))
        if not near:
            return None, 0.0, 0.0
        _, fwd, up, o = min(near, key=lambda t: t[0])
        return o, fwd, up

    def refind(self, o):
        """Find o again by its word; update it in place. How far it moved (cm), or None if not seen."""
        try:
            seen = see.look()
            found = [t for t in see.find(seen["raw_overhead"], [o.get("word", o["name"])], seen["arms"],
                                         self.held_ends(seen["arms"]))]
        except Exception:  # the service busy: no news, Muse goes on from the images
            return None
        if not found:
            return None
        new = min(found, key=lambda t: np.linalg.norm(np.subtract(t["center_cm"], o["center_cm"])))
        moved = float(np.linalg.norm(np.subtract(new["center_cm"], o["center_cm"])))
        if moved < LOST_CM:
            o.update(center_cm=new["center_cm"], size_cm=new["size_cm"], z_cm=new["z_cm"], shape=new["shape"],
                     px=new["px"])
            return moved
        return None

    def between(self, side, arm):
        o, fwd, _ = self.ahead(side, arm)
        return o if o is not None and fwd <= JAWS_CM / 2 else None

    def not_in(self, side, arm):
        """Why the jaws mustn't close yet: a thing between the fingers whose middle is still in front of the grip point
        (fully inside the gripper, not at the fingertips); None if nothing found is there."""
        o, fwd, up = self.ahead(side, arm)
        if o is None:
            return None
        if self.deepest.get(side) == o["name"] and fwd < GRIP_BACK_TIPS_CM:  # as deep as it goes (a push said so)
            return None
        if fwd > DEEP_CM:
            return (f"{o['name']}'s middle is still {fwd:.1f} cm in front of the grip point: it must be fully inside "
                    f"the gripper, not at the fingertips. Go forward about {fwd:.0f} cm first (or find it again if it "
                    "moved)")
        if o["shape"] == "round" and abs(up) > LEVEL_CM:  # a ball taken above or below its middle (only a round
            # thing's height is measured: a flat one's is a guess, and one partly under the gripper reads flat)
            return (f"{o['name']}'s middle is {abs(up):.1f} cm {'below' if up < 0 else 'above'} the grip point: the "
                    f"jaws would slip over it. Move {'down' if up < 0 else 'up'} about {abs(up):.0f} cm first ("
                    "a ball taken 2.5 cm above its middle was missed)")
        return None

    # ---- the loop ----------------------------------------------------------------------------------------------
    def ends(self, side, a):
        """(bottom, top) cm of a tall thing the arm holds, turned with the wrist since it was taken; None if none."""
        h = self.held[side]
        if not h or h["length"] < TALL_CM or self.closed[side] is None or a["width_mm"] < HOLD_MIN_MM:
            return None
        _, grip, ax = see.axes(side, a["joints"])
        return tube_ends(grip, ax, self.roll(side, a) - h["roll0"] + h.get("tilt", 0.0), h.get("ahead", 0.0),
                         h.get("across", 0.0), h["below"], h["length"])

    def measure(self, side):
        """Measure the tall thing side holds on a fresh overhead image and put its ends where it was seen; what to tell
        Muse ({} if it holds none)."""
        seen = see.look()
        a, h = seen["arms"][side], self.held[side]
        before = self.ends(side, a)
        if before is None or "raw_overhead" not in seen:
            return {}
        _, grip, ax = see.axes(side, a["joints"])
        tilt = self.roll(side, a) - h["roll0"]
        try:
            fit = see.fit_held(seen["raw_overhead"], h.get("word", h["name"]), h.get("radius", 4.0),
                               lambda t, f, c: tube_ends(grip, ax, tilt + t, f, c, h["below"], h["length"]))
        except Exception as e:  # the service busy: it stays where it was listed
            fit, why = None, f"segmentation failed: {e}"
        else:
            why = "an arm over it?"
        if fit is None:
            h["measured"] = False
            return {"not_measured": f"the {h['name']} wasn't seen clearly on the overhead image ({why}): its ends "
                                    "are from the arm's numbers"}
        h.update(tilt=fit["tilt"], ahead=fit["ahead"], across=fit["across"], measured=True)
        top = self.ends(side, a)[1]
        return {"measured": f"the {h['name']}'s top end, seen on the overhead image, is at {r1(top)} "
                            f"({np.linalg.norm(top - before[1]):.1f} cm from where it was listed)"}

    def off_opening(self, side, seen):
        """Why side mustn't let go yet: it holds something over the opening of what the other arm holds, but not right
        over it (what it holds found on the overhead image, else its grip point); None if fine or not a drop."""
        other = "right" if side == "left" else "left"
        arms = seen["arms"]
        ends = self.ends(other, arms[other])
        if ends is None or self.closed[side] is None or arms[side]["width_mm"] < HOLD_MIN_MM:
            return None
        _, g, (f, r, _) = see.axes(side, arms[side]["joints"])
        top = ends[1]
        if float(np.linalg.norm(g[:2] - top[:2])) > NEAR_TOP_CM or g[2] < top[2]:
            return None
        at, how = g[:2], "by the arm's numbers"
        if self.held[side] and "raw_overhead" in seen:
            try:
                px = see.scene().camera.project([g / 100])[0]
                xy = see.held_at(seen["raw_overhead"], self.held[side].get("word", self.held[side]["name"]), g[2], px)
            except Exception:  # the service busy: by the numbers
                xy = None
            if xy is not None:
                at, how = xy, "seen on the overhead image"
        d = top[:2] - at
        what = self.held[side]["name"] if self.held[side] else "what it holds"
        if float(np.linalg.norm(d)) > OPEN_OVER_CM:
            where = "measured on the overhead image" if self.held[other].get("measured") else "by the arm's numbers"
            return (f"{what} ({how}) is {np.linalg.norm(d):.1f} cm from right over the opening of the "
                    f"{self.held[other]['name']} (its top end {r1(top)}, {where}): it would miss. Move forward "
                    f"{float(d @ f[:2]):.1f}, right {float(d @ r[:2]):.1f} first, then open")
        over = g[2] - (self.held[side].get("radius", 3.5) if self.held[side] else 3.5) - top[2]
        if over > DROP_MAX_CM:
            return (f"{what}'s bottom is {over:.1f} cm above the rim (at most {DROP_MAX_CM:.0f}): move down "
                    f"{over - 4:.1f} first, straight down, then open")
        return None

    def clash(self, side, arms, goal):
        """Why side's grip point mustn't go to goal (cm): too near the other gripper, or into the tall thing the other
        arm holds below its top, or what side holds carried into the other gripper (robotd doesn't see held things as
        obstacles). None if clear."""
        other = "right" if side == "left" else "left"
        _, g0, _ = see.axes(side, arms[side]["joints"])
        _, g2, _ = see.axes(other, arms[other]["joints"])
        if np.linalg.norm(goal - g2) < ARMS_CM and np.linalg.norm(goal - g2) < np.linalg.norm(g0 - g2):
            return (f"{side}'s grip point would come {np.linalg.norm(goal - g2):.0f} cm from the {other} gripper "
                    f"(at least {ARMS_CM:.0f}): take turns, one arm in the middle at a time")
        ends = self.ends(other, arms[other])
        unsure = UNSURE_TOP_CM if self.held[other] and self.held[other].get("at_least") else 0
        straight_over = (ends is not None and float(np.linalg.norm(goal[:2] - ends[1][:2])) < OVER_TOP_CM
                         and float(np.linalg.norm(goal[:2] - g0[:2])) < STRAIGHT_CM)  # lowering onto the opening
        if ends and straight_over and goal[2] >= ends[1][2] + DROP_OVER_CM:
            pass
        elif ends and goal[2] < ends[1][2] + 3 + unsure and seg_dist(goal, *ends) < HELD_CLEAR_CM:
            return (f"{side} would go into the {self.held[other]['name']} the {other} arm holds (its top is at "
                    f"{r1(ends[1])}): come over it from above its top end")
        if ends and self.closed[side] is not None and arms[side]["width_mm"] >= HOLD_MIN_MM:
            off = float(np.linalg.norm(g2[:2] - FILL_XY))
            if off > IN_PLACE_CM and float(np.linalg.norm(goal[:2] - ends[1][:2])) < WAIT_CM:
                return (f"{side} holds something: it waits on its own side until the {self.held[other]['name']} is in "
                        f"place (the {other} grip point at about {r1(FILL_XY)}, now {off:.0f} cm from there)")
        if ends:  # near its top and below a safe height, only straight up or down, or away without going down
            # (come to it from above, never from under)
            top, high = ends[1], ends[1][2] + ABOVE_CM + unsure
            d_now, d_goal = (float(np.linalg.norm(p[:2] - top[:2])) for p in (g0, goal))
            sideways = float(np.linalg.norm(goal[:2] - g0[:2])) > STRAIGHT_CM
            low = (d_now < NEAR_TOP_CM and g0[2] < high) or (d_goal < NEAR_TOP_CM and goal[2] < high)
            away = d_goal > d_now + 1 and goal[2] >= g0[2] - 0.5
            line_up = d_goal < OVER_TOP_CM and min(g0[2], goal[2]) >= top[2] + LINE_UP_CM
            if sideways and low and not away and not line_up:
                return (f"near the top of the {self.held[other]['name']} the {other} arm holds ({r1(top)}), {side} "
                        f"moves only straight up or down below {high:.0f} cm: first straight up to {high:.0f} cm or "
                        "higher, then across to right above its top end, then straight down")
        mine = self.ends(side, arms[side])
        if mine:
            b, t = mine[0] + goal - g0, mine[1] + goal - g0
            if g2[2] < t[2] + 3 and seg_dist(g2, b, t) < HELD_CLEAR_CM:
                return (f"the {self.held[side]['name']} {side} holds would hit the {other} gripper: move the {other} "
                        "arm up and away first")
        return None

    def held_ends(self, arms):
        """Each held tall thing's (bottom, top) for find, the top taken 5 cm farther along it: a camera that cut its top
        off listed it short (a ball inside a held tube, seen through its opening, was found as on the table)."""
        out = []
        for s in see.SIDES:
            e = self.ends(s, arms[s])
            if e:
                ax = (e[1] - e[0]) / max(float(np.linalg.norm(e[1] - e[0])), 1e-6)
                out.append((e[0], e[1] + ax * FIND_PAST_TOP_CM))
        return out

    @staticmethod
    def roll(side, a):
        """The wrist's roll from upright (camera on top), deg."""
        return float(a["joints"][4] - see.scene().ready(side)[4])

    def go(self, task, max_turns=MAX_TURNS, home=True, held=None):
        """home: start from the ready pose (else from where the arms are); held: {side: {name, length, below, roll0}}
        for what an arm already holds."""
        t0 = time.time()
        if home:
            self.log("both arms to the ready pose")
            r = robot.home(None)
            if r.get("outcome") != "done":
                return self.end(False, f"the arms couldn't get to the ready pose: {r.get('why')}", t0)
        for s, a in robot.status()["arms"].items():  # jaws already squeezing something from before the run
            if a["width_mm"] >= HOLD_MIN_MM and a.get("grip_ma", 0) >= HOLD_MA:
                self.closed[s] = a["width_mm"]
            self.pitch[s] = float(round(a["pitch"] / 5) * 5)
        self.held.update(held or {})
        arms = robot.status()["arms"]
        for s in held or {}:  # it holds it even if the squeeze went slack (a robotd stop sets the goals to where the
            self.closed[s] = arms[s]["width_mm"]  # jaws are: the current drops)
        last = None
        for n in range(max_turns):
            seen = see.look(self.labels)
            self.labels = []
            self.save_images(seen["images"], n)
            text = (f"Task: {task}\nStart by looking at the images, then find what the task involves.\n\n"
                    if n == 0 else "")
            if last:
                text += f"Step {self.stats['steps']}: {last['said']} -> {json.dumps(last['result'])}\n"
            text += self.state(seen["arms"])
            parts = [{"type": "text", "text": text}]
            for cam in ("overhead",) + tuple(see.WRISTS.values()):
                if cam in seen["images"]:
                    parts += [{"type": "text", "text": f"{cam} camera:"}, image_part(seen["images"][cam])]
            a = self.ask(parts)
            self.log(f"[{n}] sees: {a.get('see', '')}\n     plan: {a.get('plan', '')}")
            if a.get("do") == "done":
                return self.end(bool(a.get("success")), str(a.get("summary", "")), t0)
            said = json.dumps({k: v for k, v in a.items() if k not in ("see", "plan")})
            ts = time.time()
            try:
                result = self.act(a, seen)
            except (TypeError, ValueError) as e:
                result = {"outcome": "refused", "why": f"bad arguments: {e}"}
            self.stats["steps"] += 1
            self.log(f"     {said} -> {json.dumps(result)} ({time.time() - ts:.1f} s)")
            self.save(step=a, result=result, seconds=round(time.time() - ts, 1))
            if result.get("outcome") == "stopped" and result.get("fail") in (None, "manual"):
                return self.end(False, f"a person stopped the arms: {result.get('why')}", t0)
            last = dict(said=said, result=result)
        return self.end(False, f"stopped after {max_turns} turns", t0)

    def end(self, success, summary, t0):
        self.log(f"{'success' if success else 'failure'}: {summary}")
        return dict(self.stats, success=success, summary=summary, cost=round(self.stats["cost"], 4),
                    llm_seconds=round(self.stats["llm_seconds"], 1), seconds=round(time.time() - t0, 1),
                    model=getattr(self.llm, "model", ""), record=self.record)


def run(task, llm, max_turns=MAX_TURNS, log=print, run_dir=None, effort="low", home=True, held=None):
    """Do the task; the run's numbers, its record and the conversation (watch/delegate.py calls this)."""
    r = Run(llm, log, run_dir, effort)
    t0 = time.time()
    try:
        out = r.go(task, max_turns, home, held)
    except robot.RobotdDown as e:
        out = r.end(False, f"robotd stopped answering: {e}", t0)
    except KeyboardInterrupt:
        robot.stop("brain interrupted")
        out = r.end(False, "interrupted", t0)
    return dict(out, turns=out["calls"], transcript=r.messages)
