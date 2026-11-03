"""What Muse sees every turn: the overhead image with the grid and both grippers drawn on, each wrist image with the
fingertips line, the grip point and the arm's numbers across the top, and the things segmentation found."""
import base64

import cv2
import numpy as np

from harness import overlay, robot
from harness.scene import Scene

from . import sam

SIDES = ("left", "right")
WRISTS = {"left": "left_wrist", "right": "right_wrist"}
WRIST_PX = (640, 480)  # wrist images go to the model at this size
GRIP_BACK_CM = 3.0  # the grip point, where the jaws pinch: this far back from the fingertips along the fingers
# A held tall thing measured on the overhead image: its drawn outline moved over these until it best covers what
# segmentation finds (a held tube's opening was 4.5 cm from where the arm's numbers put it, the ball missed)
FIT_TILT = np.arange(-10.0, 15.1, 2.5)  # deg more tilt than the wrist roll gives (it sags in a gentle grip)
FIT_AHEAD = np.arange(-2.0, 8.1, 0.5)  # cm: its axis this far in front of the grip point (a wide thing stops short)
FIT_ACROSS = np.arange(-4.0, 4.1, 0.5)  # cm: and this far across the jaws
FIT_IOU = 0.6  # an outline covering it less than this isn't it (hidden behind the other arm, or not found)
FIT_PX = 4  # outlines are compared at a quarter of the image's size
FIT_WORDS = ("cylinder", "transparent cylinder")  # asked for besides its own word: held in the air, segmentation
# missed a clear tube by its name in 3 of 4 images; the fit keeps only an outline shaped like it

_scene = None
_real = None


def scene():
    global _scene
    if _scene is None:
        _scene = Scene()
    return _scene


def real_rig():
    global _real
    if _real is None:
        _real = not robot.status().get("sim", True)
    return _real


def axes(side, joints):
    """The fingertips (cm), the grip point (cm), and the table directions (unit) of the gripper's forward (the way the
    fingers point, flat), right (right of that) and up."""
    p, R = scene().arms[side].tool(joints)
    approach = -R[:, 2]
    flat = approach[:2]
    if np.linalg.norm(flat) < 0.25:  # pointing down: forward is away from the base
        flat = p[:2] - scene().base_xy(side)
    f = flat / np.linalg.norm(flat)
    forward, right, up = np.array([f[0], f[1], 0.0]), np.array([f[1], -f[0], 0.0]), np.array([0.0, 0.0, 1.0])
    return p * 100, p * 100 - approach * GRIP_BACK_CM, (forward, right, up)


def tip_for(grip, head, pitch):
    """Where the fingertips go (cm) for the grip point at grip (cm), the fingers pointing along head (x, y) and tilted
    down by pitch (deg)."""
    h = np.asarray(head, dtype=float) / np.linalg.norm(head)
    c, s = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    return np.asarray(grip, dtype=float) + GRIP_BACK_CM * np.array([c * h[0], c * h[1], -s])


def wrist_px(side, p, size=WRIST_PX):
    """A point in gripper_link (m) on side's wrist image (pixels), or None behind the lens. The sims render from the
    mount's CAD lens, the real arms use the one calibrated with the board."""
    T, K, dist = scene().wrist_lens(side, size, measured=real_rig())
    pc = (np.asarray(p, dtype=float) - T[:3, 3]) @ T[:3, :3]
    if pc[2] <= 1e-3:
        return None
    uv = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].ravel()
    return int(uv[0]), int(uv[1])


def decode(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def encode(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def wrist_image(b64, side, arm):
    """A wrist image for the model: the line through the fingertips, the grip point, and the arm's numbers on top."""
    img = cv2.resize(decode(b64), WRIST_PX, interpolation=cv2.INTER_AREA)
    w, h = WRIST_PX
    tcp = scene().arms[side].tcp
    a, b = wrist_px(side, scene().fixed_tip), wrist_px(side, tcp)
    if a and b:
        k = (b[1] - a[1]) / (b[0] - a[0]) if abs(b[0] - a[0]) > 1 else 0.0
        l0, l1 = (0, int(a[1] - k * a[0])), (w, int(a[1] + k * (w - a[0])))
        cv2.line(img, l0, l1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "fingertips", (8, l0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    g = wrist_px(side, tcp + np.array([0.0, 0.0, GRIP_BACK_CM / 100]))
    if g:
        cv2.drawMarker(img, g, (255, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(img, "grip", (g[0] + 8, g[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    z = arm["tip_cm"][2]
    s = f"{side.upper()}  tips z {z:.1f} cm  pitch {round(arm['pitch'])}  jaws {arm['width_mm']:.0f} mm"
    if "grip_ma" in arm:
        s += f"  {arm['grip_ma']} mA"
    cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(img, s, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return encode(img)


def look(labels=()):
    """Fresh images and arm numbers: {"arms": robotd's arms, "images": {overhead, left_wrist, right_wrist}} with the
    drawings on. labels: things to name on the overhead image ({name, px})."""
    seen = robot.look(["overhead"] + list(WRISTS.values()), annotate=False)
    arms = seen.get("arms", {})
    images = seen.get("images", {})
    if "overhead" in images:
        raw = decode(images["overhead"])
        seen["raw_overhead"] = raw
        marks = [dict(id="", name=o["name"], px=o["px"]) for o in labels]
        images["overhead"] = encode(overlay.draw(raw, scene(), marks, arms))
    for side, cam in WRISTS.items():
        if cam in images and side in arms:
            images[cam] = wrist_image(images[cam], side, arms[side])
    return seen


def cylinder_px(bottom, top, radius):
    """A cylinder's ends' rims (cm) on the overhead image, pixels (48, 2): their hull is its outline."""
    ax = np.subtract(top, bottom) / np.linalg.norm(np.subtract(top, bottom))
    u = np.cross(ax, [0.0, 0.0, 1.0]) if abs(ax[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    return scene().camera.project(np.vstack([bottom + ring, top + ring]) / 100)


def fit_held(img, word, radius, ends_for):
    """Where a held tall thing really is, from its outline on the overhead image: {iou, tilt, ahead, across}, the
    corrections whose cylinder (ends_for(tilt, ahead, across) -> bottom, top cm; radius cm) best covers what
    segmentation finds as word; None if nothing found covers it well enough."""
    h, w = img.shape[0] // FIT_PX, img.shape[1] // FIT_PX
    masks = [cv2.resize((m > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
             for found in sam.segment(img, [word, *FIT_WORDS]).values() for _, m in found]
    best = None
    for tilt in FIT_TILT:
        for ahead in FIT_AHEAD:
            for across in FIT_ACROSS:
                px = cylinder_px(*ends_for(tilt, ahead, across), radius) / FIT_PX
                if np.isnan(px).any():
                    continue
                sil = np.zeros((h, w), np.uint8)
                cv2.fillPoly(sil, [cv2.convexHull(px.astype(np.float32)).astype(np.int32)], 1)
                sil = sil > 0
                for m in masks:
                    iou = float((sil & m).sum() / max((sil | m).sum(), 1))
                    if best is None or iou > best["iou"]:
                        best = dict(iou=round(iou, 2), tilt=float(tilt), ahead=float(ahead), across=float(across),
                                    px=px * FIT_PX)
    if best is None or best["iou"] < FIT_IOU:
        return None
    px = best.pop("px")
    edge = any(best[k] in (g[0], g[-1]) for k, g in (("tilt", FIT_TILT), ("ahead", FIT_AHEAD), ("across", FIT_ACROSS)))
    cut = (px < 0).any() or (px[:, 0] >= img.shape[1]).any() or (px[:, 1] >= img.shape[0]).any()
    return None if edge or cut else best  # part of it out of the image, or the best at the end of what was tried:
    # not sure where it is (a tube standing at the image's top edge came out 6 cm off)


def held_at(img, word, z, near_px):
    """A held round thing's centre (x, y cm) from its outline on the overhead image, at height z (cm): the outline of
    word nearest near_px (where the arm's numbers put it); None if none is near."""
    best = None
    for _, m in sam.segment(img, [word]).get(word, []):
        contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        d = float(np.hypot(cx - near_px[0], cy - near_px[1]))
        if d < 2 * r and (best is None or d < best[0]):
            best = (d, (cx, cy))
    if best is None:
        return None
    return scene().camera.to_table([best[1]], z=z / 100)[0, :2] * 100


def find(img, names, arms, held=()):
    """The named things on the overhead image (SAM): [{name, center_cm, size_cm, z_cm, shape, px}], several of one
    kind numbered from left to right. held: (bottom, top) cm of each thing a gripper holds: what's seen on it or inside
    it isn't on the table (a ball dropped into a held tube)."""
    q = {s: list(a["joints"]) + [a["width_mm"]] for s, a in arms.items()}
    pts = [list(np.add(b, np.subtract(t, b) * k)) for b, t in held for k in np.linspace(0, 1, 9)]
    found = sam.objects(scene(), img, names, q, robot.to_pixels(pts) if pts else ())
    out = []
    for name in dict.fromkeys(o["found_as"] for o in found):
        same = sorted((o for o in found if o["found_as"] == name), key=lambda o: o["center_cm"][0])
        for k, o in enumerate(same, 1):
            out.append(dict(name=name if len(same) == 1 else f"{name} {k}", word=name, center_cm=o["center_cm"],
                            size_cm=o["size_cm"], z_cm=o["z_cm"], shape=o["shape"], px=o["px"]))
    return out
