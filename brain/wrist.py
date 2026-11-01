"""The wrist cameras: where a point shows in one, finding an object through one, measuring a standing thing's height from
the side, and the image the model gets, with the fingertips' line and the grip point drawn on. The lens comes from
scene.json: as calibrated on the real arms (harness/calib_board.py), from the mount's CAD in the simulators, which render
from those same numbers."""
import base64

import cv2
import numpy as np

from harness import robot

from . import sam
from .rig import WRISTS, real, scene

WRIST_PX = (640, 480)  # wrist images go to the model at this size
OVERLAY = (255, 255, 0)  # BGR
LOOK_MAX_CM = 6.0  # a wrist camera's sighting farther than this from where an object was thought to be is another thing
LOOK_MIN_FRAC = 0.002  # masks smaller than this share of the wrist image are specks


def lens(side, size):
    """side's wrist camera for an image of size (w, h): its pose in gripper_link, K and distortion."""
    return scene().wrist_lens(side, size, measured=real())


def cam_px(T_cam, K, dist, p):
    """A point (in T_cam's frame) on the image (pixels, floats), or None behind the lens."""
    pc = (np.asarray(p, dtype=float) - T_cam[:3, 3]) @ T_cam[:3, :3]
    if pc[2] <= 1e-3:
        return None
    return cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].ravel()


def cam_ray(T_cam, K, dist, px):
    """The direction a pixel looks along, in T_cam's frame."""
    n = cv2.undistortPoints(np.array(px, dtype=float).reshape(1, 1, 2), K, dist).ravel()
    return T_cam[:3, :3] @ np.array([n[0], n[1], 1.0])


def wrist_px(side, p, size=WRIST_PX):
    """A point in gripper_link (m) on side's wrist image (pixels), or None behind the lens."""
    uv = cam_px(*lens(side, size), p)
    return None if uv is None else (int(uv[0]), int(uv[1]))


def wrist_camera(side, joints, size=WRIST_PX):
    """side's wrist camera in the world, (pose 4x4, K, distortion) for an image of size."""
    arm = scene().arms[side]
    tip, R = arm.tool(joints)
    T_gl = np.eye(4)
    T_gl[:3, :3], T_gl[:3, 3] = R, tip - R @ arm.tcp
    T, K, dist = lens(side, size)
    return T_gl @ T, K, dist


def masks(img, word):
    """SAM's masks of word on an image; none when SAM can't be reached: a look that finds nothing changes nothing."""
    try:
        return sam.segment(img, [word]).get(word, [])
    except (RuntimeError, OSError, SystemExit) as e:
        print(f"SAM failed on a wrist image ({e})", flush=True)
        return []


def measure_heights(objects):
    """Standing objects measured from the side: the overhead's outline reads them centimetres short and can cut them off
    at the image's edge. In the ready pose each wrist camera looks level across the table: SAM finds the object there,
    and the top of its outline, at the object's distance, is its height. Updates the objects; returns {name: height cm}."""
    todo = [o for o in objects if str(o.get("shape", "")).startswith("upright")]
    if not todo:
        return {}
    seen = robot.look(list(WRISTS.values()), annotate=False)
    views = []  # (how far off its image's middle, side, object, its base's pixel, its distance)
    for side, cam in WRISTS.items():
        if cam not in seen.get("images", {}):
            continue
        img = cv2.imdecode(np.frombuffer(base64.b64decode(seen["images"][cam]), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        cam = wrist_camera(side, seen["arms"][side]["joints"], (w, h))
        for o in todo:
            c = np.append(np.asarray(o["center_cm"], dtype=float) / 100, 0.0)
            uv = cam_px(*cam, c)
            if uv is None or (c - cam[0][:3, 3]) @ cam[0][:3, 2] < 0.08:
                continue
            u, v = uv
            if 0.15 * w < u < 0.85 * w and v > 0:
                views.append((abs(u - w / 2), side, o, (u, v), img, cam))
    out = {}
    for _, side, o, (u, v), img, cam in sorted(views, key=lambda t: t[0]):
        lens, f = cam[0][:3, 3], cam[1][0, 0]
        if o["name"] in out:
            continue
        word = o.get("found_as") or o.get("kind") or o["name"]
        h, w = img.shape[:2]
        horiz = np.linalg.norm(np.asarray(o["center_cm"], dtype=float) / 100 - lens[:2])
        r_px = f * float(min(o["size_cm"][:2])) / 200 / max(horiz, 0.05)
        cols = slice(max(int(u - r_px / 2), 0), min(int(u + r_px / 2) + 1, w))
        top = None
        for conf, m in masks(img, word):
            rows = np.nonzero(m[:, cols].any(axis=1))[0]
            if not len(rows) or rows.max() < v - 0.2 * (v - rows.min()) or rows.min() <= 1:
                continue  # not reaching down to its base, or cut off at the top here too
            top = rows.min() if top is None else min(top, rows.min())
        if top is None:
            continue
        ray = cam_ray(*cam, (u, top))
        reach = horiz + (1 if ray[2] < 0 else -1) * float(min(o["size_cm"][:2])) / 200  # looking down: the far rim
        z = (lens[2] + ray[2] * reach / np.linalg.norm(ray[:2])) * 100
        H = o["size_cm"][2]  # seen whole from above, the overhead's height is good to a few cm: a far-off reading is
        # another thing's outline
        if z > H if "least" in str(o.get("shape", "")) else H / 1.5 < z < H * 1.5:
            o.update(size_cm=o["size_cm"][:2] + [round(float(z), 1)], z_cm=round(float(z) / 2, 1), measured=True,
                     shape=f"upright (its height measured from the {side} wrist camera)")
            out[o["name"]] = round(float(z), 1)
    return out


def locate(side, o):
    """Where object o really is, seen from side's wrist camera a few cm away (the overhead can be centimetres off). SAM's
    masks of it, by the word that found it; the one
    whose middle comes nearest where o should show; the ray through that middle (the calibrated lens) meets the level
    of o's middle. Returns ([x, y] cm, how far that is from where o was thought to be, cm), or None when no mask of it
    is within LOOK_MAX_CM."""
    cam_name = WRISTS[side]
    seen = robot.look([cam_name], annotate=False)
    b64 = seen.get("images", {}).get(cam_name)
    if not b64:
        return None
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    cam = wrist_camera(side, seen["arms"][side]["joints"], (w, h))
    lens = cam[0][:3, 3]
    z = float(o.get("z_cm", o["size_cm"][2] / 2)) / 100
    c = np.asarray(o["center_cm"][:2], dtype=float) / 100
    word = o.get("found_as") or o.get("kind") or o["name"]
    best = None
    for conf, m in masks(img, word):
        ys, xs = np.nonzero(m)
        if not LOOK_MIN_FRAC * w * h < len(xs) < 0.5 * w * h:
            continue
        ray = cam_ray(*cam, (float(xs.mean()), float(ys.mean())))
        if ray[2] > -1e-6:  # level or up: not toward the table
            continue
        p = lens + ray * (z - lens[2]) / ray[2]
        d = float(np.linalg.norm(p[:2] - c)) * 100
        if d < LOOK_MAX_CM and (best is None or d < best[1]):
            best = (p[:2] * 100, d)
    return None if best is None else ([round(float(v), 1) for v in best[0]], round(best[1], 1))


def wrist_image(b64, side, arm, grip):
    """A wrist image for the model: smaller, with the line through the fingertips, the grip point (where the middle of
    what it holds sits, or would; grip: (cm back from the fingertips, cm across)) and the arm's pose drawn on."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, WRIST_PX, interpolation=cv2.INTER_AREA)
    w, h = WRIST_PX
    tcp = scene().arms[side].tcp
    a, b = wrist_px(side, scene().fixed_tip), wrist_px(side, tcp)
    if a and b:
        k = (b[1] - a[1]) / (b[0] - a[0]) if abs(b[0] - a[0]) > 1 else 0.0  # the line across the whole image
        l0, l1 = (0, int(a[1] - k * a[0])), (w, int(a[1] + k * (w - a[0])))
        cv2.line(img, l0, l1, OVERLAY, 1, cv2.LINE_AA)
        cv2.putText(img, "fingertips", (8, l0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OVERLAY, 1, cv2.LINE_AA)
    g = wrist_px(side, tcp + np.array([-grip[1] / 100, 0.0, grip[0] / 100]))  # gripper_link: -x is the fixed finger's side
    if g:
        cv2.drawMarker(img, g, OVERLAY, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(img, "grip", (g[0] + 8, g[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OVERLAY, 1, cv2.LINE_AA)
    if arm:
        x, y, z = arm["tip_cm"]
        s = f"fingertips z {z:.1f} cm   pitch {arm['pitch']:.0f}   jaws {arm['width_mm']:.0f} mm"
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(img, s, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()
