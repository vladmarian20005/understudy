"""The rig at a glance: the overhead camera with the table's cm grid, both grippers and the objects robotd sees, the two
wrist cameras, and the arms' state. The mouse over the overhead image reads out the table cm under it (on the table
top), so an object can be put exactly where a task says. Read only: it never moves anything.

    ./understudy.sh view          (or python -m harness.viewer; the cameras running, robotd optional)

The cameras come straight from the camera server, so robotd isn't slowed down; the arms come from robotd's status and
the objects from its detect, a few times a second. q or Esc quits.
"""
import os
import time

import cv2
import numpy as np

from . import overlay, robot
from .cameras import CameraDown, ZmqCameras
from .scene import Scene

SCALE = 1.5  # the overhead image, enlarged
PANEL_H = 200
BRAIN_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "brain_live.log")
BRAIN_FRESH_S = 120  # a brain log older than this is a finished run: not shown
STATUS_S, DETECT_S = 0.1, 0.5
BG = (32, 32, 32)


def tile(cams, name, size):
    try:
        img = cams.grab([name])[name]
    except CameraDown as e:
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        overlay.text(img, str(e), (10, 50), (80, 80, 255), 0.5)
        return img
    return cv2.resize(img, size)


def brain_lines():
    """The running brain's latest thought and call, from the log it's tee'd to: [] when no run is going."""
    try:
        if time.time() - os.path.getmtime(BRAIN_LOG) > BRAIN_FRESH_S:
            return []
        lines = open(BRAIN_LOG).read().splitlines()
    except OSError:
        return []
    said = next((s for s in reversed(lines) if s.startswith("[")), "")
    did = next((s.strip() for s in reversed(lines) if s.strip().startswith("->")), "")
    return [f"Muse {said}", f"     {did}"] if said else []


def panel(width, arms, st, objects, mouse_cm):
    img = np.full((PANEL_H, width, 3), BG, np.uint8)
    for j, s in enumerate(brain_lines()):
        overlay.text(img, s[:170], (10, 150 + 20 * j), (120, 200, 255), 0.45)
    if st is None:
        overlay.text(img, "robotd isn't answering: ./understudy.sh", (10, 24), (80, 80, 255), 0.55)
    else:
        sk = st.get("skill") or {}
        doing = f"{sk.get('name')} running" if sk.get("running") else st.get("mode", "")
        overlay.text(img, f"robotd: {'armed' if st.get('armed') else 'TORQUE OFF'}, {doing}", (10, 22),
                     (120, 230, 120) if st.get("armed") else (80, 80, 255), 0.55)
        for i, (side, a) in enumerate(arms.items()):
            x, y, z = a["tip_cm"]
            held = f"holding {a['object'] or 'something'}" if a["holding"] else f"open {a['width_mm']:.0f} mm"
            overlay.text(img, f"{side:5}  tip x {x:6.1f}  y {y:5.1f}  z {z:5.1f} cm   pitch {a['pitch']:5.1f}  "
                              f"yaw {a['yaw']:6.1f}   {held}   {a['mode']}   {a.get('temp_c', 0):.0f} C   "
                              f"grip {a.get('grip_ma', 0):.0f} mA (peak {a.get('grip_peak_ma', 0):.0f})",
                         (10, 48 + 22 * i), overlay.COLORS[side], 0.5)
        for j, e in enumerate((st.get("events") or [])[-2:]):
            overlay.text(img, f"{time.strftime('%H:%M:%S', time.localtime(e['t']))}  {e['text']}"[:150],
                         (10, 100 + 20 * j), (200, 200, 200), 0.45)
    objs = "   ".join(f"{o['id']} {o['name']} ({o['center_cm'][0]:.1f}, {o['center_cm'][1]:.1f})" for o in objects)
    overlay.text(img, ("objects: " + objs) if objects else "objects: none seen", (10, PANEL_H - 8), (255, 255, 255),
                 0.45)
    if mouse_cm is not None:
        overlay.text(img, f"mouse: x {mouse_cm[0]:.1f}  y {mouse_cm[1]:.1f} cm", (width - 260, 22), (0, 220, 255),
                     0.55)
    return img


def main():
    scene = Scene()
    cams = ZmqCameras()
    mouse = {"px": None}

    def on_mouse(event, u, v, flags, param):
        mouse["px"] = (u / SCALE, v / SCALE)

    cv2.namedWindow("understudy")
    cv2.setMouseCallback("understudy", on_mouse)
    st, objects, t_status, t_detect = None, [], 0.0, 0.0
    while True:
        now = time.monotonic()
        if now - t_status > STATUS_S:
            t_status = now
            try:
                st = robot.call("status", timeout_s=0.5, since=(st or {}).get("last_event", 0) - 2)
            except robot.RobotdDown:
                st = None
        if st is not None and now - t_detect > DETECT_S:
            t_detect = now
            try:
                objects = robot.call("detect", timeout_s=1.0).get("objects", [])
            except robot.RobotdDown:
                objects = []
        arms = (st or {}).get("arms") or {}

        try:
            top = cams.grab(["overhead"])["overhead"]
            top = overlay.draw(top, scene, objects, arms)
        except CameraDown as e:
            top = np.full((480, 640, 3), 40, np.uint8)
            overlay.text(top, str(e), (10, 50), (80, 80, 255), 0.5)
        mouse_cm = None
        if mouse["px"] is not None and mouse["px"][0] < top.shape[1] and mouse["px"][1] < top.shape[0]:
            p = scene.camera.to_table(np.array([mouse["px"]], dtype=float))[0]
            if np.all(np.isfinite(p)):
                mouse_cm = (p[0] * 100, p[1] * 100)
                u, v = int(mouse["px"][0]), int(mouse["px"][1])
                cv2.drawMarker(top, (u, v), (0, 220, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        top = cv2.resize(top, None, fx=SCALE, fy=SCALE)
        h = top.shape[0]
        side = (round(h / 2 * 4 / 3), h // 2)
        wrists = np.vstack([tile(cams, "left_wrist", side), tile(cams, "right_wrist", (side[0], h - side[1]))])
        overlay.text(wrists, "left wrist", (10, 22), overlay.COLORS["left"], 0.6)
        overlay.text(wrists, "right wrist", (10, side[1] + 22), overlay.COLORS["right"], 0.6)
        frame = np.hstack([top, wrists])
        frame = np.vstack([frame, panel(frame.shape[1], arms, st, objects, mouse_cm)])
        cv2.imshow("understudy", frame)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    cams.close()


if __name__ == "__main__":
    main()
