"""What's on the table: objects with an id, a name, centre and size in cm, long-axis yaw, colour and pixel.

The simulator reports its own objects (with optional noise). On the real arms, objects come from the overhead camera
(see cameras.py); either way the result has the same shape:
    {"id": "#1", "name": "brush", "center_cm": [x, y], "size_cm": [L, W, H], "yaw": deg, "color": "orange",
     "px": [u, v], "held_by": "left" | None}
Ids are numbered left to right in the image and are only valid until the next detection.
"""
import cv2
import numpy as np

COLOURS = [(10, "red"), (22, "orange"), (35, "yellow"), (85, "green"), (130, "blue"), (160, "purple"), (180, "red")]


def colour_name(bgr):
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    if v < 50:
        return "black"
    if s < 50:
        return "white" if v > 170 else "grey"
    return next(name for top, name in COLOURS if h <= top)


def number(dets):
    dets.sort(key=lambda d: d["px"][0])
    for i, d in enumerate(dets, 1):
        d["id"] = f"#{i}"
    return dets


class SimPerception:
    def __init__(self, world, scene):
        self.world, self.scene = world, scene

    def detect(self):
        dets = []
        for o in self.world.visible_objects():
            top = o["pos"] + [0, 0, o["size"][2] / 2]
            px = self.scene.camera.project(top)[0]
            dets.append(dict(name=o["name"], center_cm=[round(float(v) * 100, 1) for v in o["pos"][:2]],
                             z_cm=round(float(o["pos"][2]) * 100, 1),
                             size_cm=[round(float(v) * 100, 1) for v in o["size"]], yaw=round(float(o["yaw"]), 1),
                             color=colour_name(o["color"]), px=[int(round(px[0])), int(round(px[1]))],
                             held_by=o["held"][0] if o["held"] else None))
        return number(dets)


def find(dets, target):
    """The detection a target names: "#2" or "2", or a name or colour that matches exactly one object."""
    t = str(target).strip().lower()
    if t.lstrip("#").isdigit():
        return next((d for d in dets if d["id"] == "#" + t.lstrip("#")), None)
    hits = [d for d in dets if t in d["name"].lower() or t == d["color"]]
    return hits[0] if len(hits) == 1 else None


SAT_MIN, VAL_MIN = 80, 60  # coloured objects; the table and the arms are grey
MIN_AREA_PX = 120
ARM_BAND = 0.05  # m: half-width of the band masked out around each arm
OBJECT_H = 0.025  # m: assumed height, since one overhead camera can't see it
HELD_NEAR = 0.05  # m: an object this close to a holding gripper's fingertips is in that gripper


class TablePerception:
    """Coloured objects on the table, from the overhead camera. The arms are masked out using their joint angles, and
    each object's outline is projected onto the table through the calibrated camera model."""

    def __init__(self, scene, cameras=None, ctl=None, holding=None):
        self.scene, self.cameras, self.ctl = scene, cameras, ctl
        self.holding = holding if holding is not None else {}

    def arm_mask(self, shape, q):
        cam = self.scene.camera
        mask = np.zeros(shape[:2], np.uint8)
        for s, qs in q.items():
            pts = self.scene.arms[s].points(qs)
            px = cam.project(pts)
            for i in range(len(pts) - 1):
                if np.isnan(px[i]).any() or np.isnan(px[i + 1]).any():
                    continue
                d = float((pts[i] - cam.cam_to_world[:3, 3]) @ cam.cam_to_world[:3, 2])
                t = max(4, int(2 * ARM_BAND * cam.K[0, 0] / max(d, 0.05)))
                cv2.line(mask, tuple(px[i].astype(int)), tuple(px[i + 1].astype(int)), 255, t)
        return mask

    def detect(self, img=None, q=None):
        if img is None:
            img = self.cameras.grab(["overhead"])["overhead"]
        if q is None:
            q = self.ctl.snapshot()["q"]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] >= SAT_MIN) & (hsv[:, :, 2] >= VAL_MIN)).astype(np.uint8) * 255
        mask[self.arm_mask(img.shape, q) > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tips = {s: self.scene.arms[s].tool(qs)[0] for s, qs in q.items()}
        dets = []
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA_PX:
                continue
            m = cv2.moments(c)
            px = [m["m10"] / m["m00"], m["m01"] / m["m00"]]
            table = self.scene.camera.to_table(c.reshape(-1, 2).astype(float), z=OBJECT_H / 2)[:, :2]
            box = cv2.boxPoints(cv2.minAreaRect(table.astype(np.float32)))
            edges = [box[(i + 1) % 4] - box[i] for i in range(2)]
            L, W = sorted([float(np.linalg.norm(e)) for e in edges], reverse=True)
            long_edge = max(edges, key=lambda e: float(np.linalg.norm(e)))
            centre = box.mean(axis=0)
            one = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(one, [c], -1, 255, -1)
            colour = colour_name([int(v) for v in cv2.mean(img, mask=one)[:3]])
            held = next((s for s, h in self.holding.items() if h and np.linalg.norm(tips[s][:2] - centre) < HELD_NEAR),
                        None)
            dets.append(dict(name=f"{colour} object", center_cm=[round(float(v) * 100, 1) for v in centre],
                             z_cm=round(OBJECT_H * 50, 1), size_cm=[round(L * 100, 1), round(W * 100, 1),
                                                                    round(OBJECT_H * 100, 1)],
                             yaw=round(float(np.degrees(np.arctan2(long_edge[1], long_edge[0]))) % 180, 1),
                             color=colour, px=[int(round(px[0])), int(round(px[1]))], held_by=held))
        return number(dets)
