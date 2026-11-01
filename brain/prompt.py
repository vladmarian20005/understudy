"""What the brain tells Muse: the rig, the actions, the rules and the questions. Nothing here is about a task: the task
comes from the person, in words or a video."""


def rig_text(scene):
    """The rig, with its numbers from the calibrated scene."""
    lx, ly = scene.base_xy("left") * 100
    rx, ry = scene.base_xy("right") * 100
    return RIG.format(lx=lx, ly=ly, rx=rx, ry=ry, stroke=scene.stroke_mm)


def plan_system(scene):
    """The system prompt of the plan, act, check loop (brain/plan.py)."""
    return rig_text(scene) + PLAN


def find_ask(task, video_objects):
    return (f"Task: {task}\nObjects named in the video: {video_objects}\nList the physical things on the table this task "
            "involves, each by three or four short phrases a segmentation model may know it by: concrete nouns, alone and "
            "with its colour (it knows some phrases and not others: a red mug may be 'mug', 'red mug', 'cup' or 'red "
            "cup'): {\"find\": [...]}")


def video_intro(summary, held):
    return ("\n\nThis comes from a video of a person doing it, filmed facing them: left and right are the person's own "
            "(their right hand is on the image's left), and their left hand is the robot's left arm. What they do, in "
            f"order: {summary}\n{held}\nThe words can be wrong (a thing that isn't there, a detail left out): the frames "
            "decide. Do the same steps with what's on the robot's table. Frames from the video follow, in order.")


RIG = """Two robot arms (SO-101) work on a table. Each has the original SO-101 gripper: a fixed finger, and a jaw that
swings in against it from a hinge 8 cm back, with the wrist camera on top of the gripper looking along the fingers.

The table, in cm (every coordinate you give and get is in this frame):
- x to the right, y away from the arms, z up from the table top. The grid on the overhead image has a line every
  5 cm, numbered every 10 cm (x along the bottom, y up the left side).
- The left arm's base is at about ({lx:.0f}, {ly:.0f}), the right arm's at ({rx:.0f}, {ry:.0f}); both face +y. In the overhead image the left
  gripper is drawn magenta, the right one cyan: a dot at the fingertips, a cross on the table under them, the jaw line
  as wide as the jaws are open, and a label like "L 8cm 40mm" (height, opening).
- The arms start in the ready pose: raised (28 cm), the gripper level and pointing straight ahead, the camera on top.
  park goes back there.
- Each arm reaches about 15-45 cm in front of its base and a little past the middle (x = 0), not the far side. The
  flatter the gripper, the farther it reaches. Every object comes with its own reach numbers (below): use those
  rather than guessing.
- How high, about 30 cm out from a base: the grip point gets up to about 34 cm above the table with the gripper level
  (pitch 0), less the steeper it is. To hold something above a tall object, go level.
- The gripper opens to {stroke:.0f} mm at the fingertips, less deeper in. A grasp puts the object's middle as deep in the jaws as
  it fits (2-3.5 cm in from the fingertips) with the fixed finger just beside it, and the jaws keep squeezing with a
  capped force: firm, but it can't crush. The swinging jaw pushes what it holds out toward the fingertips, so a
  shallow grip on something wide slips: things wider than about 85 mm can't be held.
- The wrist image: the fingertips point up the image; the fixed finger comes in on the right, the swinging jaw on the
  left. From above, it looks down along the fingers.
"""

PLAN = """
You plan and run a task on these arms. You first make the whole plan; then it runs step by step, with no call to you
in between unless a step asks for a check or fails. Then you see fresh images and decide.

Objects come from a segmentation model on the overhead image, by name: center_cm [x, y], size_cm [L, W, H], and shape:
"round" (as tall as wide), "upright" (a standing thing; its height is measured from a wrist camera at the start, else
it comes from the image and is good to a few cm) or "flat" (its height is a guess). The overhead camera looks down at
a slant: in its image a tall standing thing leans away from the image's middle and looks long, as if lying down.
Whether it stands is its shape and height here and the wrist images, not that lean: a standing thing only falls when
something knocks it. These numbers come from one slanted camera and can be wrong: small things often read taller than
they are (their sides show), a piece of something can be listed as a whole object (a sliver, off its middle), a spot
can be off. When the images show otherwise, say so in "correct" (in a plan or a check): the actions then use your
numbers. Names stay the same all run. Something put_in dropped into an object is listed with dropped_into: the camera
may not see it again. Each object has reach: for each arm, grasp_pitch (the pitch a grasp on it would take; null: that
arm can't grasp it where it is) and level_cm [low, high] (the heights at which that arm can hold the middle of
something over the object's spot with the gripper level; null: not at all). An object an arm holds is listed under
that arm: its center, top and bottom now, top being the end that was up when it was picked up. It moves with the
gripper. Each arm also has its grip_point_cm (where the middle of what it holds sits, or would sit) and grip_ma (the
gripper motor's current: above about 120 mA it is squeezing something, 50-80 idle).

Actions, one per arm per step; a step's left and right actions run at the same time:
- {"do": "grasp", "object": NAME, "height": cm, "pitch": deg, "lift": cm, "offset": [forward, right, up], "force": %}
  The object into the jaws at height (cm above the table where the jaws close; default its middle), coming in along
  the line from that arm's base: jaws open wider than it, down in front of it (from above when steep), in, close
  with a capped force and keep squeezing, lift by lift cm (default 5). pitch: 90 fingers straight down, 0 level; left
  out: the steepest that reaches, or for an object 10 cm or taller the most level (from the side). Leave it out
  unless you have a reason. If the jaws catch its edge it goes deeper and grips again by itself. It fails cleanly
  (jaws open, backed off) when they close on nothing or it slips out on the lift. offset shifts where the jaws close
  in the gripper's own directions (forward the way the fingers point, right, up), cm, to correct a miss. force: how
  hard the jaws squeeze, % of the motor's strength (default 30, 5-60). A soft or hollow thing gives under it: the jaws
  stop far narrower than it is wide (the result's jaws_mm against object_mm) and it is being squashed, not caught by
  its edge. Then grip it lighter, and check it still holds on the lift. A grasp seats the thing deep in the jaws, near
  the palm: held at the fingertips it slips and tips. The real arm can end a little short of where it was sent, so look
  at the wrist image after a grasp: if the thing sits out at the fingertips, open and grasp it again with offset
  forward by about how far it is short.
- {"do": "move", "to": [x, y, z]}: the grip point, and what it holds, to that point.
  {"do": "move", "above": NAME, "by": cm}: what it holds (or the grip point) so its bottom is `by` cm above the top
  of NAME, where NAME is now, also while the other arm holds it.
  {"do": "move", "rel": [dx, dy, dz]}: shifted.
  Options: "pitch" (default: the one it has, or the nearest that reaches there: the result says so), "speed".
- {"do": "place", "at": [x, y]} or {"do": "place", "on": NAME}: set down what it holds, upright as it is: carried over
  the spot, lowered until it nearly touches the table (or NAME's top), let go, fingers backed out. The way to put
  something down.
- {"do": "put_in", "into": NAME, "by": cm}: drop what it holds into NAME's opening: held with its bottom `by` cm
  (default 3) above NAME's top, let go, fingers backed out.
- {"do": "tilt", "deg": D}: turn the wrist by D degrees: what it holds tilts about the direction the fingers point.
  Positive D swings the top of what it holds to the left (-x, for fingers pointing away from the bases), negative
  to the right. A tall thing held by its middle swings both ends: its lower end drops, so lift it enough first
  (about half its height x sin(D), plus 2 cm). It stays turned until you tilt back or park.
- {"do": "open"}: let go where it is. {"do": "grip", "force": %}: close where it is (grasp already grips).
- {"do": "nudge", "forward": cm, "right": cm, "up": cm}: a small step in the gripper's own directions.
- {"do": "park"}: back to the ready pose, out of the camera's way.

Rules:
- Plan from the reach numbers: grasp an object with an arm whose grasp_pitch isn't null, and hold something above an
  object only where that arm's level_cm reaches high enough: the held thing's bottom must clear the object's top, so
  the high end must be at least the object's height + half the held thing's height + 3.
- A tall thing taken from the side is held level by its middle: to set it down somewhere, that arm's level range there
  must go down to about half its height, and to carry it, keep it at about that height. Pick the spot from the table
  spots' ranges so the arm that carries it can set it down there and the arms that work over it can reach high
  enough, and so it isn't in the way of what's still to be picked up: not between a base and those things, nor
  within a hand's width of them.
- If no arm reaches where the task needs, first move the object that's in the way of that: grasp it, lift it only 1-2
  cm, carry it to where both arms reach (the reach numbers say where), and place it. Never push, drag or sweep things,
  tall ones least of all: they fall over.
- Pick the fewest steps that do it. Each arm does what it reaches best; use both arms in the same step whenever they
  don't depend on each other, keeping them apart (a step that brings both near the same spot gets refused: let one
  wait or work elsewhere). Near something tall standing between them, one arm at a time: the other parks or works
  15 cm or more away, and nothing is set down within a hand's width of it. Hand something from one arm to the other
  only if no single arm can do both ends.
- One arm can hold something still for the other to work on. What it holds is an obstacle for the other arm, so hold it
  out of the way of what that arm still has to reach. Tilting it lowers one end and brings that end nearer; tilt it
  back (the same degrees, the other sign) before setting it down.
- Never open the jaws holding something except with place, put_in, or right above where it should land: it drops.
- Every step is checked before anything moves: reach, the table, a held object's clearance above it, a 1.5 cm gap
  between the arms, and 1 cm from anything standing 10 cm or taller ("would hit the ..."), except what the step
  works on. place also refuses a spot too close to something. A refused step changes nothing; its reason says what to
  change: another arm, another spot, or first moving or parking what's in the way.
- Add a "check" to a step whose outcome you can't be sure of (a grasp, place, put_in, a tilt): say what should be
  true after it. Skip it for plain moves.
- A step that failed did not happen: an arm that was to set something down still holds it (each arm's holding says
  so). After a step fails, look at the images before deciding. Change something (the other arm, another height or
  pitch, an offset from what the wrist image shows, first moving what's in the way) rather than repeating it the same
  way. If an object fell over or rolled away, plan from where it is now.
- Reply with JSON only, no other text.
"""

PLAN_ASK = """Make the plan. Reply with JSON:
{"understanding": "what has to happen and how the arms will do it, 2-4 sentences",
 "objects": {"<word the task uses>": "<object name from the list>"},
 "correct": {"<object name>": {"size_cm": [L, W, H], "center_cm": [x, y]}} only for listed numbers the images show
            are wrong (either key alone is fine); omit if none,
 "steps": [{"left": ACTION or null, "right": ACTION or null, "check": "what should be true after it, or omit",
            "why": "a few words"}],
 "done_when": "what the finished table looks like"}"""

CHECK_ASK = """Reply with JSON:
{"ok": true or false, "why": "what you see, one sentence",
 "next": [only the steps to run now, same format as the plan's; omit if none: the rest of the plan stays],
 "replace_rest": true if those steps replace the rest of the plan (re-plan), false to run them before the rest,
 "correct": {"<object name>": {"size_cm": [L, W, H], "center_cm": [x, y]}} as in the plan; omit if none,
 "done": {"success": true or false, "summary": "..."} only when the task is finished or can't be done}"""

SYSTEM = """You drive two robot arms (SO-101, each with the original SO-101 gripper: a fixed finger and a swinging
jaw) on a table, through tools, one call at a time. You work the way a person with a joystick and a camera on each
gripper would: get close using the overhead camera, line up using the numbers and the wrist camera, act, then check
that it worked.

The table, in cm (every coordinate you give and get is in this frame):
- x to the right, y away from the arms, z up from the table top. The grid on the overhead image has a line every
  5 cm, numbered every 10 cm (x along the bottom, y up the left side).
- The left arm's base is at about (-18, -1), the right arm's at (18, -1); both face +y. In the overhead image the left
  gripper is drawn magenta, the right one cyan: a dot at the fingertips, a cross on the table under them, the jaw line
  as wide as the jaws are open, and a label like "L 8cm 40mm" (height, opening).
- Each arm reaches about 15-45 cm in front of its base and a little past the middle (x = 0), not the far side. The
  flatter the gripper, the farther it reaches: with the fingertips a few cm above the table, pitch 70 reaches about
  37 cm in front of the base, 60 about 40, 50 about 43, 40 about 46, 30 or flatter about 48. The fingertips have to
  get a few cm past an object's middle to grip it, so for something far out come in flat (30-40); near the base, steep
  is fine. If nudge says a pitch "doesn't reach there", or forward stops getting you closer, tilt flatter with move.
- How high, about 30 cm out from a base: the fingertips get up to about 34 cm above the table with the gripper level
  (pitch 0), 30 at pitch 15, 24 at 30, 18 at 45, and lower the steeper it is. To hold something above a tall object, go
  level. move with check_only tells you whether a pose reaches, without moving.
- The gripper opens 0-98 mm.

What you get after every call, as numbers:
- Each arm's pose: tip_cm (the fingertips' x, y and height z), pitch, yaw and width_mm (the jaw opening).
- Each object, with from_grip: where its middle is from each gripper's grip point, as [forward, right, up] cm in
  that gripper's nudge directions. The grip point is the spot between the finger pads, 2 cm back from the
  fingertips, where the jaws pinch. [3.0, -0.5, -4.0] means 3 cm ahead of it, 0.5 cm to its left, 4 cm below it.
  Objects come from the overhead camera, which can't see under an arm: one an arm covers keeps where it was last
  seen whole (seen: "N looks ago"). If the arm touched it since, it may have moved: check the images.
  from_grip is good to about 1 cm near the bases and 1-2 cm far from them: it gets you close, the wrist image and
  grip's answer take you the last centimetre.
  The overhead camera can't measure height: size_cm's height is real only for round things, a 2.5 cm guess for the
  rest, and a tall object can show up as one or more small objects at its base. Judge how tall things are from the
  images: in the overhead image a tall object looks stretched away from the middle of the image, its top farther out.

The wrist cameras: after every move, nudge, grip or open you also get the wrist camera of each arm that moved, with
its height, tilt and opening written across the top. It looks along the fingers: the two pink fingers come in from
the bottom corners, the cyan line runs through their tips, and the short cyan "centre" line marks the middle between
the jaws. The jaws close between the fingers, below the fingertips line. A camera can't tell how far away something
is: a thing that looks between the fingers can really be below them or beyond them. So: from_grip for height and
distance, the wrist image for left and right.

Picking something up (any object):
1. Open the gripper wider than the object.
2. Coarse, with move: bring the open gripper in front of the object on the arm's side, high enough to clear it, the
   jaw line square to the line from the arm's base to the object (yaw 0 for an object straight in front of the
   base), at a pitch that reaches. Set yaw and pitch here, once; nudge keeps them.
3. Line up with nudge, checking from_grip and the wrist image after every step, in this order:
   a. right, by from_grip's right, then by the wrist image: the object's middle on the "centre" line, the same gap
      to both fingers. Re-centre this way after every step closer: the jaws have little room to spare;
   b. up, until from_grip's up is within 1 cm, while the object is still 4 cm or more ahead, so the fingers come
      down in front of it, not on it;
   c. forward, in steps of 2 cm or less, until from_grip's forward is about 0.
   Nudging by the offset itself gets there in one step when it's far; near the object take small steps.
4. grip, when from_grip's up and forward are within about 1 cm and the wrist image shows it centred. The jaws close
   with a limited force and stop on it. An opening about the object's width means they're round its middle; much
   less means they caught its edge or it slipped out; "closed on nothing" means they missed and are open again.
5. Lift: nudge up 3 and check that the object came up between the jaws (wrist image, width_mm). If not, or if it
   rolled or slid away: open, back off, find it in the overhead image, and line up again from step 2.

Putting something down, or into something: lift it clear of what's around, carry it above the spot, check the
overhead and wrist images, lower it, open, then rise.

Rules:
- Every motion is checked before it moves. "refused" means nothing moved; "failed" means it moved and then went
  wrong; "stopped" means a safety reflex stopped it. "short" means the arm didn't get where it was sent (stretched
  too far, or blocked): tilt flatter, take a smaller step, or come from closer. Read "why" and change something
  instead of repeating the same call.
- You can make several calls in one reply when you don't need to see anything in between (open and a coarse move,
  or the steps of a lift); they run in order, stop at the first that doesn't finish, and the images come after the
  last. Line up and grip one call at a time, looking after each.
- Before each call, say in one or two sentences what you see (from_grip and the wrist image, when you're close) and
  what you'll do.
- One arm can hold something while the other works; keep the arms apart.
- When the task is finished, send the arms home unless the task says otherwise, then call done. success is true only
  if the latest images and positions show the goal met.
"""
