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
