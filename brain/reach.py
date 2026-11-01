"""What the arms reach, from the kinematics alone: the numbers Muse plans from. robotd's check of each motion decides."""
import numpy as np

from harness import kin
from harness.scene import SIDES

from .rig import scene

GRIP_BACK_CM = 3.0  # the grip point with nothing held, or the default for a grasp's reach: this far back along the fingers
GRASP_PITCHES = (80, 70, 60, 50, 40, 30, 20, 10, 0)  # tried steepest first for a short thing, most level first for a tall one
TALL_CM = 10.0  # a thing this tall is gripped from the side
SPOTS = [(x, y) for y in (20, 25, 30, 35) for x in (-10, -5, 0, 5, 10)]  # where the plan learns how high each arm holds
_level_cache = {}
_reach_cache = {}


def level_range(side, xy):
    """The lowest and highest (cm) an arm's grip point gets over xy (cm) with the gripper level and the camera on top,
    or None: near a base a level gripper reaches high but not low, far out it reaches low but not high."""
    key = (side, round(float(xy[0])), round(float(xy[1])))
    if key not in _level_cache:
        arm = scene().arms[side]
        head = np.asarray(xy, dtype=float) - scene().base_xy(side) * 100
        head /= max(np.linalg.norm(head), 1e-6)
        ok = lambda z: _reaches(arm, np.append(np.asarray(xy, dtype=float) + head * GRIP_BACK_CM, z), 0)
        hits = [z for z in range(4, 47, 3) if ok(z)]
        if not hits:
            _level_cache[key] = None
        else:
            lo, hi = hits[0], hits[-1]
            lo = next((z for z in (lo - 2, lo - 1) if z >= 2 and ok(z)), lo)
            hi = next((z for z in (hi + 2, hi + 1) if ok(z)), hi)
            _level_cache[key] = (lo, hi)
    return _level_cache[key]


def _reaches(arm, tip_cm, pitch):
    try:
        arm.ik(np.asarray(tip_cm, dtype=float) / 100, pitch, roll=arm.upright_roll(), restarts=0, iters=60)
        return True
    except kin.Unreachable:
        return False


def reach_facts(o):
    """For each arm, about (the kinematics alone; the motion's own check decides): the pitch a grasp on the object
    would take (None: out of reach), and from how low to how high a level grip point gets over it (where the middle of
    what the gripper holds would be). Kept per spot and size: an object that doesn't move isn't worked out again."""
    c, H = o["center_cm"], float(o["size_cm"][2])
    key = (round(c[0]), round(c[1]), round(H), round(float(min(o["size_cm"][:2]))))
    if key not in _reach_cache:
        out = {}
        h = float(o.get("z_cm", H / 2))
        for side in SIDES:
            arm = scene().arms[side]
            head = np.asarray(c, dtype=float) - scene().base_xy(side) * 100
            head /= max(np.linalg.norm(head), 1e-6)
            grasp = None
            for p in (GRASP_PITCHES if H < TALL_CM else tuple(reversed(GRASP_PITCHES))):
                a = np.array([np.cos(np.radians(p)) * head[0], np.cos(np.radians(p)) * head[1], -np.sin(np.radians(p))])
                if _reaches(arm, np.append(c, h) + a * 2.5, p):
                    grasp = p
                    break
            r = level_range(side, c)
            out[side] = dict(grasp_pitch=grasp, level_cm=None if r is None else list(r))
        _reach_cache[key] = out
    return _reach_cache[key]


def spots_text():
    """From how low to how high each arm holds the middle of something with the gripper level, over a grid of table
    spots."""
    cells = []
    for xy in SPOTS:
        r = {s: level_range(s, xy) for s in SIDES}
        cells.append(f"({xy[0]}, {xy[1]}): " + " / ".join("-" if r[s] is None else f"{r[s][0]}-{r[s][1]}" for s in SIDES))
    return ("Heights (cm) at which each arm can hold the middle of something with the gripper level, over table spots "
            "(left / right; - is out of reach): " + "; ".join(cells))
