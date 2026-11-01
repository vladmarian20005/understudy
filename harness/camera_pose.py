"""Re-solve the overhead camera's pose after it has been moved, with the arms as the target. Each arm in turn closes
its gripper and holds its fingertips 2 cm over a few spots on the table; you click the fingertips in the window, and
solvePnP fits the camera's pose to the clicks. The lens (K, dist) stays as calibrated. Writes
harness/scene.json (the old one is kept as scene.json.bak); restart robotd and the viewer to use it.

    python -m harness.camera_pose            (robotd running and armed, the table clear; keep a hand near space)

Click the very end of the closed fingers, where they point at the table. The red cross is where the old pose puts
them. Keys: u = undo the last click, s = skip this spot, q = quit without writing.
"""
import json
import os
import shutil

import cv2
import numpy as np

from . import overlay, robot
from .cameras import ZmqCameras
from .scene import SCENE, Scene

SPOTS = {"left": [(-22, 14), (-10, 20), (-24, 30), (-8, 32)], "right": [(22, 14), (10, 20), (24, 30), (8, 32)]}
Z_CM = 2.0
SCALE = 1.5
WINDOW = "camera pose: click the fingertips"


def go(arm, xy):
    """The arm's closed fingertips Z_CM over xy: up first, across, then down. The fingertip position, cm, or None."""
    r = robot.move(**{arm: [{"rel": [0, 0, 6]}, {"xy": list(xy), "z": 8, "width": 0}, {"xy": list(xy), "z": Z_CM}]},
                   speed="slow")
    if r.get("outcome") != "done":
        print(f"  {arm} {xy}: {r.get('outcome')} ({r.get('why')}), skipped")
        return None
    return robot.status()["arms"][arm]["tip_cm"]


def click(cams, scene, tip_cm, label):
    """Wait for a click on the fingertips. Returns the pixel (original image), "undo", "skip" or "quit"."""
    clicked = {}
    cv2.setMouseCallback(WINDOW, lambda e, u, v, *_: clicked.update(px=(u / SCALE, v / SCALE))
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    guess = scene.camera.project(np.array([tip_cm]) / 100)[0]
    while "px" not in clicked:
        img = cams.grab(["overhead"])["overhead"].copy()
        if np.all(np.isfinite(guess)):
            cv2.drawMarker(img, (int(guess[0]), int(guess[1])), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        overlay.text(img, f"{label}: click the end of the closed fingers   (u undo, s skip, q quit)", (10, 24),
                     (0, 220, 255), 0.6)
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("u"), ord("s"), ord("q")):
            return {ord("u"): "undo", ord("s"): "skip", ord("q"): "quit"}[k]
    return clicked["px"]


def solve(scene, pairs):
    obj = np.array([p for p, _ in pairs], dtype=np.float64) / 100
    img = np.array([u for _, u in pairs], dtype=np.float64)
    w2c = scene.camera.world_to_cam
    rvec, tvec = cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3].copy()
    ok, rvec, tvec = cv2.solvePnP(obj, img, scene.camera.K, scene.camera.dist, rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("solvePnP failed")
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    proj = cv2.projectPoints(obj, rvec, tvec, scene.camera.K, scene.camera.dist)[0].reshape(-1, 2)
    return np.linalg.inv(w2c), np.linalg.norm(proj - img, axis=1)


def main():
    scene = Scene()
    if not robot.status().get("armed"):
        raise SystemExit("the arms are off: press a in robotd")
    cams = ZmqCameras()
    cams.wait()
    cv2.namedWindow(WINDOW)
    pairs = []
    robot.home(None, 0)
    for arm, spots in SPOTS.items():
        i = 0
        while i < len(spots):
            tip = go(arm, spots[i])
            if tip is None:
                i += 1
                continue
            got = click(cams, scene, tip, f"{arm} arm, spot {i + 1}/{len(spots)}")
            if got == "quit":
                robot.home(None, 84)
                raise SystemExit("quit: nothing written")
            if got == "undo":
                if pairs:
                    pairs.pop()
                i = max(i - 1, 0)
                continue
            if got != "skip":
                pairs.append((tip, got))
                print(f"  {arm} tip ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) cm -> pixel ({got[0]:.0f}, {got[1]:.0f})")
            i += 1
        robot.home(arm, 84)
    cv2.destroyAllWindows()
    if len(pairs) < 6:
        raise SystemExit(f"only {len(pairs)} clicks: need 6 or more, nothing written")
    c2w, err = solve(scene, pairs)
    moved = np.linalg.norm(c2w[:3, 3] - scene.camera.cam_to_world[:3, 3]) * 100
    print(f"camera moved {moved:.1f} cm; fit error per click (px): {np.round(err, 1).tolist()}, "
          f"RMS {np.sqrt(np.mean(err ** 2)):.1f} px (about {np.sqrt(np.mean(err ** 2)) * 0.11:.1f} cm on the table)")
    d = json.load(open(SCENE))
    d["camera"]["cam_to_world"] = [[round(float(v), 6) for v in row] for row in c2w]
    shutil.copy(SCENE, SCENE + ".bak")
    with open(SCENE, "w") as f:
        f.write(json.dumps(d, indent=2) + "\n")
    print(f"wrote {os.path.relpath(SCENE)} (old one in scene.json.bak): restart robotd and the viewer")


if __name__ == "__main__":
    main()
