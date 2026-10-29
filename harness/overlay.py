"""What an agent sees on the overhead image: the table's cm grid, every object's id, and both grippers (where they are,
how high, their jaw line and opening). Coordinates read off the grid are the ones the primitives take.

The grid is the table frame: x to the right, y away from the arms' bases, in cm; a line every 5 cm, numbered every
10 cm along the near and left edges. A gripper is drawn at its fingertips, with a thin line down to the table point
under it (where it would touch down), its jaw line as wide as the jaws are open, and a label "L 8cm 40mm" (height,
opening; "holding" when it holds something).
"""
import cv2
import numpy as np

COLORS = {"left": (255, 0, 255), "right": (255, 255, 0)}  # BGR: left magenta, right cyan
GRID = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX
STEP_CM = 5


def text(img, s, org, color=(255, 255, 255), scale=0.4):
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def visible(img, uv):
    return np.all(np.isfinite(uv)) and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0]


def polyline(img, cam, pts_m, color, thick=1):
    uv = cam.project(pts_m)
    ok = np.all(np.isfinite(uv), axis=1)
    if ok.sum() > 1:
        cv2.polylines(img, [uv[ok].astype(np.int32)], False, color, thick, cv2.LINE_AA)


def grid(img, scene):
    cam = scene.camera
    x0, x1 = (round(v * 100) for v in scene.table_x)
    y0, y1 = (round(v * 100) for v in scene.table_y)
    lines = img.copy()
    for x in range(x0 - x0 % STEP_CM, x1 + 1, STEP_CM):
        polyline(lines, cam, [[x / 100, y / 100, 0] for y in np.linspace(y0, y1, 12)], GRID, 2 if x == 0 else 1)
    for y in range(y0 - y0 % STEP_CM + STEP_CM, y1 + 1, STEP_CM):
        polyline(lines, cam, [[x / 100, y / 100, 0] for x in np.linspace(x0, x1, 12)], GRID, 2 if y == 0 else 1)
    cv2.addWeighted(lines, 0.45, img, 0.55, 0, img)
    # numbers: x along the lowest grid row that shows, y along the leftmost column that shows
    for x in range(x0 - x0 % 10, x1 + 1, 10):
        for y in range(y0 - y0 % STEP_CM + STEP_CM, y1 + 1, STEP_CM):
            uv = cam.project([[x / 100, y / 100, 0]])[0]
            if visible(img, uv) and uv[1] < img.shape[1] - 4:
                text(img, f"{x}", (uv[0] + 2, uv[1] - 3), GRID, 0.35)
                break
    for y in range(10, y1 + 1, 10):
        for x in range(x0 - x0 % STEP_CM, x1 + 1, STEP_CM):
            uv = cam.project([[x / 100, y / 100, 0]])[0]
            if visible(img, uv) and uv[0] > 2:
                text(img, f"y{y}", (uv[0] + 2, uv[1] - 3), GRID, 0.35)
                break


def gripper(img, scene, side, arm):
    cam, color = scene.camera, COLORS[side]
    tip = np.array(arm["tip_cm"], dtype=float) / 100
    floor = tip * [1, 1, 0]
    polyline(img, cam, [tip, floor], color, 1)
    u, v = cam.project([floor])[0]
    if np.all(np.isfinite([u, v])):
        cv2.drawMarker(img, (int(u), int(v)), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    a = np.radians(arm["yaw"])
    half = max(arm["width_mm"], 15.0) / 2000 * np.array([np.cos(a), np.sin(a), 0.0])
    polyline(img, cam, [tip - half, tip + half], color, 3)
    uv = cam.project([tip])[0]
    if np.all(np.isfinite(uv)):
        cv2.circle(img, (int(uv[0]), int(uv[1])), 4, color, -1, cv2.LINE_AA)
        state = "holding" if arm["holding"] else f"{arm['width_mm']:.0f}mm"
        text(img, f"{side[0].upper()} {arm['tip_cm'][2]:.0f}cm {state}", (uv[0] + 8, uv[1] + 14), color)


def draw(img, scene, objects, arms):
    """The overlay on a copy of the overhead image. objects: perception's detections; arms: robotd's arms_state."""
    img = img.copy()
    grid(img, scene)
    for d in objects:
        u, v = d["px"]
        cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        text(img, f"{d['id']} {d['name']}", (u + 6, v - 6), scale=0.45)
    for side, arm in arms.items():
        gripper(img, scene, side, arm)
    text(img, "table cm: x right, y away from the bases, grid 5 cm", (6, 14), scale=0.4)
    return img
