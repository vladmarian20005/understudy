"""What the page shows live: the three cameras as JPEG frames (the overhead one with the viewer's grid, grippers and
objects drawn on) and robotd's state. One thread does the work for every open page, a few times a second; it only
reads (status, detect and the camera server), so robotd isn't slowed down."""
import base64
import threading
import time

import cv2
import numpy as np

from harness import overlay, robot
from harness.cameras import ORDER, CameraDown, ZmqCameras
from harness.scene import Scene

FPS = 12
STATUS_S, DETECT_S = 0.2, 0.5


def placeholder(text, size=(640, 480)):
    img = np.full((size[1], size[0], 3), 28, np.uint8)
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, text, ((size[0] - w) // 2, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1,
                cv2.LINE_AA)
    return img


def jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class Live:
    def __init__(self, cams=None, scene=None, fps=FPS):
        self.cams = cams if cams is not None else ZmqCameras()
        self.scene = scene if scene is not None else Scene()
        self.fps = fps
        self.status = None  # robotd's status reply, or None when it isn't answering
        self.objects = []
        self.frames = {n: jpeg(placeholder("starting")) for n in ORDER}
        self.seq = 0
        self.cond = threading.Condition()
        self.running = True
        threading.Thread(target=self._loop, daemon=True, name="ui-live").start()

    def close(self):
        self.running = False

    # ---- the loop -------------------------------------------------------------------------------------------------
    def _poll(self, now, last):
        if now - last["status"] > STATUS_S:
            last["status"] = now
            try:
                since = (self.status or {}).get("last_event", 0) - 4
                self.status = robot.call("status", timeout_s=0.5, since=since)
            except robot.RobotdDown:
                self.status, self.objects = None, []
        if self.status is not None and now - last["detect"] > DETECT_S:
            last["detect"] = now
            try:
                self.objects = robot.call("detect", timeout_s=1.0).get("objects", [])
            except robot.RobotdDown:
                self.objects = []

    def _frame(self, name):
        if name == "overhead":
            try:
                img = self.cams.grab(["overhead"])["overhead"]
                arms = (self.status or {}).get("arms") or {}
                return jpeg(overlay.draw(img, self.scene, self.objects, arms))
            except CameraDown:
                return jpeg(placeholder("overhead camera offline"))
        try:
            b64, _ = self.cams.latest[name]
            self.cams.grab([name])  # raises when the frame is stale
            return base64.b64decode(b64)  # already a JPEG: pass it through
        except (CameraDown, KeyError, TypeError):
            return jpeg(placeholder(f"{name.replace('_', ' ')} camera offline"))

    def _loop(self):
        last = {"status": 0.0, "detect": 0.0}
        while self.running:
            t0 = time.monotonic()
            try:
                self._poll(t0, last)
                frames = {n: self._frame(n) for n in ORDER}
            except Exception as e:  # never let a bad frame kill the page's cameras
                frames = {n: jpeg(placeholder(f"error: {type(e).__name__}")) for n in ORDER}
            with self.cond:
                self.frames = frames
                self.seq += 1
                self.cond.notify_all()
            time.sleep(max(0.0, 1 / self.fps - (time.monotonic() - t0)))

    # ---- for the server -------------------------------------------------------------------------------------------
    def wait_frame(self, name, after, timeout=2.0):
        """The newest JPEG of a camera once there is one newer than seq `after`: (seq, bytes)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after or not self.running, timeout=timeout)
            return self.seq, self.frames.get(name, b"")

    def snapshot(self):
        st = self.status
        if st is None:
            return {"robotd": False}
        arms = {}
        for side, a in (st.get("arms") or {}).items():
            arms[side] = {k: a.get(k) for k in ("tip_cm", "holding", "object", "width_mm", "mode")}
        return {
            "robotd": True, "sim": bool(st.get("sim")), "armed": bool(st.get("armed")), "mode": st.get("mode"),
            "skill": (st.get("skill") or {}).get("name") if (st.get("skill") or {}).get("running") else None,
            "episode": st.get("episode"), "dataset": st.get("dataset"), "arms": arms,
            "objects": [{k: o.get(k) for k in ("id", "name", "center_cm", "held_by")} for o in self.objects],
            "events": [{"t": e.get("t"), "text": e.get("text")} for e in (st.get("events") or [])[-5:]],
            "cameras": {n: bool(h.get("ok")) for n, h in self.cams.health().items()},
        }
