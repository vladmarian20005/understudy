"""Calibrate the rig from a ChArUco board shown full screen on a 13" iPad lying flat on the table: the overhead camera's
pose, each wrist camera's lens, and each arm's base and joint offsets, from views at many heights (a fit from table
touches alone leaves the arms 4-7 cm short and low of their kinematics when raised).

    python -m harness.calib_board board                      the board's PNG, at the iPad's pixels: AirDrop it, open it
                                                             in Photos, full screen, brightness up, auto-lock off
    python -m harness.calib_board capture --square MM --thick MM     (robotd running and armed; a hand near space)
    python -m harness.calib_board fit logs/calib/<stamp> [--tip left=U,V --tip right=U,V] [--write]
    python -m harness.calib_board check                      (robotd restarted on the new scene.json) each closed
                                                             gripper 2 cm over board corners, and the images

--square: one square's side as measured on the screen (calipers across 4 squares, / 4); --thick: the screen's height
over the table (the iPad, and its case if it lies in one). The board lies with its long side along the table's x.

capture: each arm in turn (the other waits at ready) looks at the board with its wrist camera from about 18 poses, at
fingertip heights 10-30 cm and pitches 15-75, slowly, each planned by robotd first; the joints, the wrist image and the
overhead image are saved at each. fit: the wrist lens from those views (OpenCV), then per arm its base, the offsets of
shoulder lift, elbow, wrist flex and roll, and the camera's pose on the gripper (near the mount's CAD; --tip: where the
fixed fingertip shows in that arm's wrist image, which pins the camera to the finger), all by least squares on the
board corners' pixels; the overhead camera's pose from the board. The table frame stays where it was: the table top is
the board's plane less --thick, the origin midway between the bases and the line between them at the same angle as
before. --write puts it in scene.json (the old one kept as scene.json.bak); restart robotd and the viewer to use it.
"""
import argparse
import json
import math
import os
import shutil
import time

import cv2
import numpy as np

from . import kin
from .scene import SCENE, Scene

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "logs", "calib")
SQUARES = (7, 5)  # across, down: the four corner squares are black, so the pattern's outer edge is easy to measure
SQUARE_PX, MARKER_PX = 368, 276  # on the screen; 368 px is 35.4 mm at the 13" iPad's 264 ppi
SCREEN_PX = (2752, 2064)  # 13" iPad Pro (M4); the 13" Air's 2732 x 2048 shows it 0.7 % smaller, which --square measures
DICT = cv2.aruco.DICT_4X4_50  # the biggest cells for the overhead camera, where a square is about 30 px
WRISTS = {"left": "left_wrist", "right": "right_wrist"}
TIP_TOP_Y = -0.0042  # m: the fixed finger's tip as the camera sees it, the corner of its top face (the camera's side, -y)
# and its inner face (scene.json's fixed_tip), which --tip gives in the image
# capture
TIP_Z_CM = (10, 14, 18, 22, 26, 30)  # fingertip heights over the table
PITCHES = (15, 30, 45, 60, 75)
YAW_OFFS = (0, 15, -15)  # the jaw line turned from square to the reach: the camera rolls off the top, the board turns
AIMS = ((0, 0), (-7, 0), (7, 0), (0, -5), (0, 5))  # cm from the board's middle, where the camera's axis should meet it
VIEWS = 18  # per arm
SEEN = 0.75  # a pose is kept if the CAD camera would see this share of the corners
CLEAR_CM = 7.0  # every point of the arm stays this far over the screen
SETTLE_S = 1.2  # held still this long before the joints and images are read
MIN_CORNERS = 8
# fit
MOUNT_DEG, MOUNT_MM, ROLL_DEG, TIP_PX, TILT_DEG = 3.0, 3.0, 2.0, 4.0, 2.0  # how far the fit may stray from the CAD and the
# measured roll offsets, and a base from level, before it pays like one pixel; and how close the fixed fingertip's
# projection should come to --tip; and how well PnP gives a camera's pose from one view
POSE_MM, POSE_DEG = 1.5, 0.3


def board(square_mm=SQUARE_PX * 25.4 / 264):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard(SQUARES, square_mm / 1000, square_mm * MARKER_PX / SQUARE_PX / 1000, d)


def write_png(path):
    w, h = SQUARES[0] * SQUARE_PX, SQUARES[1] * SQUARE_PX
    img = board().generateImage((w, h), marginSize=0, borderBits=1)
    canvas = np.full((SCREEN_PX[1], SCREEN_PX[0]), 255, np.uint8)
    x0, y0 = (SCREEN_PX[0] - w) // 2, (SCREEN_PX[1] - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    cv2.imwrite(path, canvas)
    return path


def detect(b, img):
    """The board's known points in an image: (object points (N, 3) m in the board's frame, pixels (N, 2)), or None.
    The chessboard corners between markers, and each marker's own four corners: on a slanted view through the wide
    wrist lens the chessboard corners often don't interpolate, the markers do."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    cc, ci, mc, mi = cv2.aruco.CharucoDetector(b, cv2.aruco.CharucoParameters(), dp).detectBoard(gray)
    obj, px = [], []
    if cc is not None and len(cc):
        o, p = b.matchImagePoints(cc, ci)
        obj.append(o.reshape(-1, 3))
        px.append(p.reshape(-1, 2))
    ids, pts = list(b.getIds().ravel()), b.getObjPoints()
    for c, i in zip(mc or (), [] if mi is None else mi.ravel()):
        if i in ids:
            obj.append(np.asarray(pts[ids.index(i)]).reshape(4, 3))
            px.append(np.asarray(c).reshape(4, 2))
    if not obj or sum(len(o) for o in obj) < MIN_CORNERS:
        return None
    return np.vstack(obj).astype(np.float64), np.vstack(px).astype(np.float64)


# ---- geometry ----------------------------------------------------------------------------------------------------------
def T_of(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rodrigues(v):
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))[0]


def fk(base, offsets_deg, signs, q_deg):
    """gripper_link's pose (4x4) for lerobot joint degrees, given the base's pose and the offsets."""
    T = np.asarray(base, dtype=float)
    for Tj, a in zip(kin.chain()[0], signs * np.radians(np.asarray(q_deg[:5], dtype=float) + offsets_deg)):
        c, s = math.cos(a), math.sin(a)
        T = T @ Tj @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    return T


def project(T_cam_pts, K, dist, pts):
    """Points (N, 3) in some frame to pixels, through a camera whose pose in that frame is T_cam_pts (OpenCV's model,
    k1 k2 p1 p2 k3, in numpy: smooth enough for the fit's finite differences)."""
    R, t = T_cam_pts[:3, :3], T_cam_pts[:3, 3]
    pc = (np.asarray(pts, dtype=float) - t) @ R
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    k1, k2, p1, p2, k3 = (list(np.ravel(dist)) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]])


# ---- capture -----------------------------------------------------------------------------------------------------------
def board_in_table(sc, b, img):
    """The board's pose in the table frame from the overhead image and the current camera calibration."""
    d = detect(b, img)
    if d is None:
        raise SystemExit("the overhead camera doesn't see the board: is the screen on, and in view?")
    obj, px = d
    cam = sc.camera
    ok, rv, tv = cv2.solvePnP(obj, px, cam.K, cam.dist)
    return cam.cam_to_world @ T_of(rodrigues(rv), tv.ravel()), len(px)


def poses(sc, side, T_wb, q_now, thick_cm, size):
    """About VIEWS keyframes from which side's wrist camera sees the whole board, spread over heights, pitches, rolls
    and parts of the image, as the current calibration predicts."""
    arm = sc.arms[side]
    pts_w = (T_wb[:3, :3] @ board().getChessboardCorners().T).T + T_wb[:3, 3]
    mid = pts_w.mean(axis=0)
    cam, K, _ = sc.wrist_lens(side, size, measured=False)
    base = arm.base[:3, 3]
    cands = []
    for z in TIP_Z_CM:
        for pitch in PITCHES:
            for ax, ay in AIMS:
                aim = mid[:2] + np.array([ax, ay]) / 100
                tip = aim - 0.15 * (aim - base[:2]) / np.linalg.norm(aim - base[:2])
                p = math.radians(pitch)
                for _ in range(4):  # the fingertips where the camera's axis meets the board at aim
                    h = (tip - base[:2]) / np.linalg.norm(tip - base[:2])
                    lens = np.append(tip, z / 100) - 0.100 * np.append(h * math.cos(p), -math.sin(p)) \
                        + 0.068 * np.append(h * math.sin(p), math.cos(p))
                    down = p + math.radians(25)
                    d = (lens[2] - thick_cm / 100) / math.tan(down) if down < math.radians(89) else 0.0
                    tip = tip + (aim - (lens[:2] + d * h))
                h = (tip - base[:2]) / np.linalg.norm(tip - base[:2])
                for yo in YAW_OFFS:
                    yaw = (math.degrees(math.atan2(h[1], h[0])) + 90 + yo) % 180
                    try:
                        q = arm.ik([tip[0], tip[1], z / 100], pitch, yaw, seed=q_now, restarts=1)
                    except kin.Unreachable:
                        continue
                    if not arm.within_limits(q):
                        continue
                    pts = arm.points(q)
                    over = (np.abs(pts[:, 0] - mid[0]) < 0.16) & (np.abs(pts[:, 1] - mid[1]) < 0.13)
                    if np.any(pts[over, 2] < thick_cm / 100 + CLEAR_CM / 100) or np.any(pts[:, 2] < 0.04):
                        continue
                    if (side == "left" and tip[0] > 0.03) or (side == "right" and tip[0] < -0.03):
                        continue
                    Tc = fk(arm.base, np.degrees(arm.offsets), arm.signs, q) @ cam
                    pc = (np.linalg.inv(Tc) @ np.column_stack([pts_w, np.ones(len(pts_w))]).T)[:3].T
                    if np.any(pc[:, 2] < 0.05):
                        continue
                    px = project(Tc, K, np.zeros(5), pts_w)
                    inside = (px[:, 0] > 0.05 * size[0]) & (px[:, 0] < 0.95 * size[0]) & \
                             (px[:, 1] > 0.05 * size[1]) & (px[:, 1] < 0.95 * size[1])
                    if inside.mean() < SEEN:
                        continue
                    c = px[inside].mean(axis=0) / size
                    cands.append(dict(kf=dict(xy=[round(tip[0] * 100, 1), round(tip[1] * 100, 1)], z=z,
                                              pitch=pitch, yaw=round(yaw, 1)), q=q,
                                      f=np.array([z / 8, pitch / 20, yo / 15, c[0] * 3, c[1] * 3])))
    if not cands:
        return []
    chosen = [min(cands, key=lambda c: np.abs(c["q"] - q_now).sum())]
    while len(chosen) < min(VIEWS, len(cands)):  # farthest first: as different as the candidates allow
        chosen.append(max(cands, key=lambda c: min(np.linalg.norm(c["f"] - o["f"]) for o in chosen)))
    order, left, q = [], chosen, q_now  # then in the order that travels least
    while left:
        nxt = min(left, key=lambda c: np.abs(c["q"] - q).sum())
        order.append(nxt)
        left = [c for c in left if c is not nxt]
        q = nxt["q"]
    return order


def capture(a):
    from . import robot
    from .cameras import ZmqCameras
    if not robot.status().get("armed"):
        raise SystemExit("the arms are off: switch torque on (key a in robotd)")
    sc = Scene()
    b = board(a.square)
    cams = ZmqCameras()
    time.sleep(1.5)
    run = os.path.join(OUT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run)
    print(f"both arms to ready; saving to {os.path.relpath(run, ROOT)}", flush=True)
    r = robot.home()
    if r.get("outcome") != "done":
        raise SystemExit(f"couldn't get to ready: {r.get('why')}")
    time.sleep(SETTLE_S)
    imgs = cams.grab()
    cv2.imwrite(os.path.join(run, "overhead_ready.png"), imgs["overhead"])
    T_wb, n = board_in_table(sc, b, imgs["overhead"])
    print(f"overhead sees {n} corners; the board's middle at ({T_wb[0, 3] * 100:.1f}, {T_wb[1, 3] * 100:.1f}) cm, "
          f"its plane {T_wb[2, 3] * 100:.1f} cm up (you said {a.thick / 10:.1f})", flush=True)
    json.dump(dict(square_mm=a.square, thick_mm=a.thick, T_wb_old=T_wb.tolist(), scene=json.load(open(SCENE)),
                   sizes={s: list(imgs[WRISTS[s]].shape[1::-1]) for s in WRISTS}),
              open(os.path.join(run, "meta.json"), "w"), indent=1)
    log = open(os.path.join(run, "views.jsonl"), "a")
    sides = [a.arm] if a.arm else ["left", "right"]
    for side in sides:
        q_now = np.array(robot.status()["arms"][side]["joints"])
        size = imgs[WRISTS[side]].shape[1::-1]
        plan = poses(sc, side, T_wb, q_now, a.thick / 10, size)
        print(f"{side}: {len(plan)} views", flush=True)
        for k, c in enumerate(plan):
            kf = c["kf"]
            chk = robot.move(**{side: [kf]}, speed="slow", check_only=True)
            if chk.get("outcome") != "done":
                print(f"  {k + 1}. {kf}: refused ({chk.get('why')})", flush=True)
                continue
            r = robot.move(**{side: [kf]}, speed="slow")
            if r.get("outcome") != "done":
                print(f"  {k + 1}. {kf}: {r.get('outcome')} ({r.get('why')})", flush=True)
                if r.get("outcome") == "stopped":
                    raise SystemExit("stopped")
                continue
            time.sleep(SETTLE_S)
            st = robot.status()["arms"]
            imgs = cams.grab()
            name = f"{side}_{k:02d}"
            cv2.imwrite(os.path.join(run, name + ".png"), imgs[WRISTS[side]])
            cv2.imwrite(os.path.join(run, name + "_overhead.png"), imgs["overhead"])
            d = detect(b, imgs[WRISTS[side]])
            log.write(json.dumps(dict(side=side, view=name, kf=kf, joints=st[side]["joints"],
                                      other=st["right" if side == "left" else "left"]["joints"],
                                      tip_cm=st[side]["tip_cm"], corners=0 if d is None else len(d[1]))) + "\n")
            log.flush()
            print(f"  {k + 1}. {kf}: {0 if d is None else len(d[1])} corners", flush=True)
        robot.home([side])
    print(f"done: python -m harness.calib_board fit {os.path.relpath(run, ROOT)}")


# ---- fit ---------------------------------------------------------------------------------------------------------------
def lens(dets, size):
    """K and distortion for one wrist camera from its board views."""
    obj = [d[0].astype(np.float32) for d in dets]
    px = [d[1].astype(np.float32) for d in dets]
    for _ in range(2):  # then again without the points that miss by far more than the rest (a marker the fingers clip)
        rms, K, dist, rv, tv = cv2.calibrateCamera(obj, px, tuple(size), None, None)
        keep = []
        for o, p, r, t in zip(obj, px, rv, tv):
            e = np.linalg.norm(cv2.projectPoints(o, r, t, K, dist)[0].reshape(-1, 2) - p, axis=1)
            keep.append(e < max(3 * rms, 2.0))
        obj = [o[k] for o, k in zip(obj, keep) if k.sum() >= MIN_CORNERS]
        px = [p[k] for p, k in zip(px, keep) if k.sum() >= MIN_CORNERS]
    return rms, K, dist.ravel()


def fit_rig(arms, sc_old, cam0, fixed_tip, tips):
    """arms: side -> dict(views=[{joints, obj, px}], K, dist). Least squares over both arms' views at once on the
    camera's pose in each view (PnP on the board): each base in the board's frame, the offsets of joints 2-5, the camera
    on the gripper, and the board's scale, one for both (the arms' link lengths are the metric reference, so it checks
    the measured square). Fitting the pixels instead let the left camera's narrow lens (f 986 px at 1024) pin the
    camera's direction and leave its distance loose (a 26 deg shoulder offset). The bases are bolted to
    the table, so they stay near level (TILT_DEG): from views with little pan between them a tilted base and a shoulder
    offset look alike. First with the CAD mount and the board as measured, from the base that puts the CAD camera where
    each view says; then all of it. Returns ({side: (base in the board's frame, offsets_deg, camera in gripper_link,
    per-view (mm, deg) off)}, scale)."""
    from scipy.spatial.transform import Rotation
    sides, init = list(arms), {}
    for s in sides:
        arm, a = sc_old.arms[s], arms[s]
        off0 = np.degrees(arm.offsets)
        for v in a["views"]:
            ok, rv, tv = cv2.solvePnP(v["obj"], v["px"], a["K"], a["dist"], flags=cv2.SOLVEPNP_IPPE)
            ok, rv, tv = cv2.solvePnP(v["obj"], v["px"], a["K"], a["dist"], rv, tv, True)
            v["T_bc"] = np.linalg.inv(T_of(rodrigues(rv), tv.ravel()))
        ests = [v["T_bc"] @ np.linalg.inv(cam0) @ np.linalg.inv(fk(np.eye(4), off0, arm.signs, v["joints"]))
                for v in a["views"]]
        init[s] = (T_of(Rotation.from_matrix([e[:3, :3] for e in ests]).mean().as_matrix(),
                        np.mean([e[:3, 3] for e in ests], axis=0)), off0)

    def unpack(x, s):  # degrees and mm: every parameter about 1 for a typical error
        y, (base0, off0) = x[16 * sides.index(s):16 * sides.index(s) + 16], init[s]
        base = T_of(rodrigues(np.radians(y[0:3])) @ base0[:3, :3], base0[:3, 3] + y[3:6] / 1000)
        off = off0.copy()
        off[1:5] += y[6:10]
        cam = T_of(cam0[:3, :3] @ rodrigues(np.radians(y[10:13])), cam0[:3, 3] + y[13:16] / 1000)
        return base, off, cam, y

    def off_by(x, s, v):
        base, off, cam, _ = unpack(x, s)
        T = fk(base, off, sc_old.arms[s].signs, v["joints"]) @ cam
        return ((T[:3, 3] - v["T_bc"][:3, 3] * (1 + x[-1] / 100)) * 1000,
                np.degrees(cv2.Rodrigues(T[:3, :3].T @ v["T_bc"][:3, :3])[0].ravel()))

    def residuals(x):
        r = []
        for s in sides:
            base, _, cam, y = unpack(x, s)
            for v in arms[s]["views"]:
                mm, deg = off_by(x, s, v)
                r += [mm / POSE_MM, deg / POSE_DEG]
            r += [np.degrees(np.cross([0.0, 0.0, -1.0], base[:3, 2])[:2]) / TILT_DEG,  # table up is the board's -z
                  y[10:13] / MOUNT_DEG, y[13:16] / MOUNT_MM, [y[9] / ROLL_DEG]]
            if s in tips:
                r.append((project(cam, arms[s]["K"], arms[s]["dist"], fixed_tip[None])[0] - tips[s]) / TIP_PX)
        return np.concatenate(r)

    x = np.zeros(16 * len(sides) + 1)
    first = np.zeros(len(x), bool)
    for i in range(len(sides)):
        first[16 * i:16 * i + 10] = True
    x = _stage(residuals, x, first)
    x = _stage(residuals, x, np.ones(len(x), bool))
    out = {}
    for s in sides:
        base, off, cam, _ = unpack(x, s)
        per = [off_by(x, s, v) for v in arms[s]["views"]]
        out[s] = (base, off, cam, [(float(np.linalg.norm(mm)), float(np.linalg.norm(deg))) for mm, deg in per])
    return out, 1 + x[-1] / 100


def _stage(fn, x, free):
    """x with its free entries fitted by least squares (robust: a bad view counts less)."""
    from scipy.optimize import least_squares

    def f(y):
        z = x.copy()
        z[free] = y
        return fn(z)
    z = x.copy()
    z[free] = least_squares(f, x[free], jac="3-point", diff_step=1e-4, loss="soft_l1", f_scale=3.0).x
    return z


def table_frame(bases_b, thick_m, old_scene):
    """The table frame in the board's: z up from the table top (the board's plane less thick; the board's z points into
    the screen), the origin midway between the bases, and the line from the left base to the right at the angle it had
    in the old calibration."""
    ol = np.array(old_scene["arms"]["left"]["base"])[:3, 3]
    orr = np.array(old_scene["arms"]["right"]["base"])[:3, 3]
    th0 = math.atan2(orr[1] - ol[1], orr[0] - ol[0])
    z = np.array([0.0, 0.0, -1.0])
    lb, rb = bases_b["left"][:3, 3], bases_b["right"][:3, 3]
    e = (rb - lb) - z * ((rb - lb) @ z)
    e /= np.linalg.norm(e)
    x = math.cos(th0) * e - math.sin(th0) * np.cross(z, e)
    m = (lb + rb) / 2
    origin = np.array([m[0], m[1], thick_m])
    T_bw = T_of(np.column_stack([x, np.cross(z, x), z]), origin)
    return np.linalg.inv(T_bw)


def fit(a):
    run = a.run
    meta = json.load(open(os.path.join(run, "meta.json")))
    meta["square_mm"] = a.square or meta["square_mm"]
    b = board(meta["square_mm"])
    old = meta["scene"]
    tmp = os.path.join(run, "scene_old.json")
    json.dump(old, open(tmp, "w"))
    sc_old, sc = Scene(tmp), Scene()
    cam0 = sc.wrist_lens("left", (640, 480), measured=False)[0]  # the same mount on both arms
    fixed_tip = sc.fixed_tip + [0.0, TIP_TOP_Y, 0.0]
    T_wb_old = np.array(meta["T_wb_old"])
    tips = {}
    for t in a.tip or []:
        s, uv = t.split("=")
        tips[s] = np.array([float(v) for v in uv.split(",")])
    views = [json.loads(l) for l in open(os.path.join(run, "views.jsonl"))]
    arms = {}
    for side in WRISTS:
        vs = []
        for v in views:
            if v["side"] != side:
                continue
            d = detect(b, cv2.imread(os.path.join(run, v["view"] + ".png")))
            if d is not None:
                vs.append(dict(v, obj=d[0], px=d[1]))
        if len(vs) < 6:
            print(f"{side}: only {len(vs)} views with the board: skipped")
            continue
        size = meta["sizes"][side]
        rms, K, dist = lens([(v["obj"], v["px"]) for v in vs], size)
        print(f"{side} wrist lens: {len(vs)} views, rms {rms:.2f} px, f {K[0, 0]:.1f} {K[1, 1]:.1f} "
              f"(CAD {sc.wrist_lens(side, size, measured=False)[1][0, 0]:.1f}), centre {K[0, 2]:.1f} {K[1, 2]:.1f}, dist {np.round(dist, 3).tolist()}")
        arms[side] = dict(views=vs, K=K, dist=dist, size=size)
    if len(arms) < 2:
        raise SystemExit("need both arms")
    fits, scale = fit_rig(arms, sc_old, cam0, fixed_tip, tips)
    square = meta["square_mm"] * scale
    print(f"the board's squares come out {square:.2f} mm (measured {meta['square_mm']:.2f})")
    out = dict(arms={}, bases_b={})
    for side, (base_b, off, cam, per) in fits.items():
        dR = np.degrees(np.linalg.norm(cv2.Rodrigues(cam0[:3, :3].T @ cam[:3, :3])[0]))
        tip = project(cam, arms[side]["K"], arms[side]["dist"], fixed_tip[None])[0]
        mm, deg = np.array(per).T
        print(f"{side} arm: views off by {np.median(mm):.1f} mm, {np.median(deg):.2f} deg median (worst {mm.max():.1f} mm, "
              f"{deg.max():.1f} deg); offsets "
              f"{np.round(np.degrees(sc_old.arms[side].offsets), 2).tolist()} -> {np.round(off, 2).tolist()}; camera "
              f"{dR:.1f} deg and {np.linalg.norm(cam[:3, 3] - cam0[:3, 3]) * 1000:.1f} mm from the CAD; fixed "
              f"fingertip drawn at {np.round(tip).tolist()}" + (f" (clicked {tips[side].tolist()})" if side in tips else ""))
        out["bases_b"][side] = base_b
        out["arms"][side] = dict(arms[side], off=off, cam=cam)
    b = board(square)
    T_wb = table_frame(out["bases_b"], meta["thick_mm"] / 1000, old)
    shift = T_wb[:3, 3] - T_wb_old[:3, 3]
    print(f"the board in the new table frame is {np.round(shift * 100, 1).tolist()} cm from where the old "
          "calibration put it")
    new = json.loads(json.dumps(old))
    for side, r in out["arms"].items():
        base_w = T_wb @ out["bases_b"][side]
        tilt = np.degrees(math.acos(min(1.0, base_w[2, 2])))
        ob = np.array(old["arms"][side]["base"])
        print(f"{side} base: at {np.round(base_w[:3, 3] * 100, 1).tolist()} cm (was "
              f"{np.round(ob[:3, 3] * 100, 1).tolist()}), tilted {tilt:.1f} deg (was "
              f"{np.degrees(math.acos(min(1.0, ob[2, 2]))):.1f})")
        sa = sc_old.arms[side]
        new["arms"][side]["base"] = np.round(base_w, 6).tolist()
        new["arms"][side]["offsets_deg"] = [round(float(v), 3) for v in r["off"]]
        new["arms"][side]["wrist"] = dict(size=r["size"], K=np.round(r["K"], 3).tolist(),
                                          dist=np.round(r["dist"], 5).tolist(),
                                          cam_in_gripper=np.round(r["cam"], 6).tolist())
        arm_new = kin.Arm(side, base_w, r["off"], sa.signs, sa.tcp, 0.0)
        errs = []
        for v in r["views"]:  # where the fingertips really were, against where the old calibration put them
            errs.append((arm_new.tool(v["joints"])[0] - sa.tool(v["joints"])[0]) * 100)
        errs = np.array(errs)
        hi = [e for e, v in zip(errs, r["views"]) if v["kf"]["z"] >= 20]
        print(f"{side} fingertips, new minus old over its views: mean {np.round(errs.mean(axis=0), 1).tolist()} cm"
              + (f", at 20 cm up or more {np.round(np.mean(hi, axis=0), 1).tolist()}" if hi else ""))
    det = None
    for f in sorted(os.listdir(run)):
        if f.endswith("overhead.png") or f == "overhead_ready.png":
            d = detect(b, cv2.imread(os.path.join(run, f)))
            if d is not None and (det is None or len(d[1]) > len(det[1])):
                det = d
    cam = sc_old.camera
    ok, rv, tv = cv2.solvePnP(det[0], det[1], cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    ok, rv, tv = cv2.solvePnP(det[0], det[1], cam.K, cam.dist, rv, tv, True)
    T_wc = T_wb @ np.linalg.inv(T_of(rodrigues(rv), tv.ravel()))
    e = project(T_wc, cam.K, cam.dist, (T_wb[:3, :3] @ det[0].T).T + T_wb[:3, 3]) - det[1]
    print(f"overhead: {len(det[1])} corners, rms {np.sqrt((e ** 2).sum(axis=1).mean()):.2f} px; camera at "
          f"{np.round(T_wc[:3, 3] * 100, 1).tolist()} cm (was {np.round(cam.cam_to_world[:3, 3] * 100, 1).tolist()})")
    new["camera"]["cam_to_world"] = np.round(T_wc, 6).tolist()
    new["measured"] = (old.get("measured", "") + f"; {time.strftime('%Y-%m-%d')}: bases, offsets of joints 2-5, the "
                       f"overhead camera's pose and the wrist cameras (lens, pose on the gripper) from a ChArUco board "
                       f"on an iPad, {sum(len(r['views']) for r in out['arms'].values())} wrist views at 10-30 cm "
                       f"(harness/calib_board.py, {os.path.basename(os.path.normpath(run))})")
    path = os.path.join(run, "scene_new.json")
    json.dump(new, open(path, "w"), indent=2)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    if a.write:
        shutil.copy(SCENE, SCENE + ".bak")
        shutil.copy(path, SCENE)
        print("and harness/scene.json (old one: scene.json.bak): restart robotd and the viewer")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("board")
    c = sub.add_parser("capture")
    c.add_argument("--square", type=float, required=True, help="one square, mm")
    c.add_argument("--thick", type=float, required=True, help="the screen's height over the table, mm")
    c.add_argument("--arm", choices=list(WRISTS), help="only this arm")
    f = sub.add_parser("fit")
    f.add_argument("run")
    f.add_argument("--tip", action="append", help="side=U,V: the fixed fingertip in that arm's wrist image")
    f.add_argument("--square", type=float, help="one square, mm, if not as at the capture")
    f.add_argument("--write", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "board":
        print(write_png(os.path.join(OUT, "charuco_ipad13.png")))
    elif a.cmd == "capture":
        capture(a)
    elif a.cmd == "fit":
        fit(a)


if __name__ == "__main__":
    main()
