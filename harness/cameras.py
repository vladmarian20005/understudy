"""The real cameras, read from the camera server: each camera publishes JPEG frames on its own ZMQ port in lerobot's
ZMQCamera format ({"timestamps": {name: t}, "images": {name: base64 JPEG}}).

A thread per camera keeps the newest JPEG; grab() decodes on demand and refuses a frame older than MAX_AGE, so nothing
records or plans from a frozen image.
"""
import base64
import json
import os
import threading
import time

import cv2
import numpy as np
import zmq

BASE_PORT = int(os.environ.get("CAMERA_PORT", 5555))
ORDER = ("overhead", "left_wrist", "right_wrist")  # camera i is on BASE_PORT + i
MAX_AGE = 1.0  # s


class CameraDown(Exception):
    pass


class ZmqCameras:
    def __init__(self, names=ORDER, base_port=BASE_PORT):
        self.latest = {n: (None, 0.0) for n in names}
        self.lock = threading.Lock()
        self.running = True
        self.threads = []
        for n in names:
            t = threading.Thread(target=self._listen, args=(n, base_port + ORDER.index(n)), daemon=True,
                                 name=f"camera-{n}")
            t.start()
            self.threads.append(t)

    def _listen(self, name, port):
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 500)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://127.0.0.1:{port}")
        while self.running:
            try:
                msg = json.loads(s.recv_string())
            except zmq.Again:
                continue
            images = msg.get("images", {})
            b64 = images.get(name) or next(iter(images.values()), None)
            if b64:
                with self.lock:
                    self.latest[name] = (b64, time.monotonic())
        s.close()

    def wait(self, timeout=10.0):
        """Block until every camera has sent a frame. Raises CameraDown."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if all(v[0] is not None for v in self.latest.values()):
                return
            time.sleep(0.05)
        missing = [n for n, v in self.latest.items() if v[0] is None]
        raise CameraDown(f"no frames from {', '.join(missing)}: is the camera server running?")

    def grab(self, names=ORDER, snap=None):
        out = {}
        now = time.monotonic()
        for n in names:
            with self.lock:
                b64, t = self.latest[n]
            if b64 is None or now - t > MAX_AGE:
                raise CameraDown(f"{n}: no frame for {now - t:.1f} s")
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise CameraDown(f"{n}: a frame didn't decode")
            out[n] = img
        return out

    def close(self):
        self.running = False
