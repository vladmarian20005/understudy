"""The real cameras, read from the camera server (cameras/server.py). Each camera publishes JPEG frames on its own ZMQ
port, the first port plus its index in cameras/cameras.json, as
{"timestamps": {name: t}, "images": {name: base64 JPEG}, "seq": n, "repeat": bool}.

A thread per camera keeps the newest JPEG. Frames the server re-sends while a camera stalls ("repeat") don't count as
fresh, and grab() refuses a camera with no fresh frame for MAX_AGE, so nothing records or plans from a frozen image.

    python -m harness.cameras            wait for every camera, print its size and frame rate
    python -m harness.cameras --view     then show them side by side (q quits)
"""
import argparse
import base64
import collections
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import zmq

RIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cameras", "cameras.json")
MAX_AGE = 1.0  # s
FPS_WINDOW = 2.0  # s


def rig(path=RIG):
    """The camera names in port order and the first port, from cameras.json (CAMERA_PORT overrides the port)."""
    with open(path) as f:
        cfg = json.load(f)
    return tuple(cfg["cameras"]), int(os.environ.get("CAMERA_PORT", cfg.get("port", 5555)))


ORDER, BASE_PORT = rig()


class CameraDown(Exception):
    pass


class ZmqCameras:
    def __init__(self, names=ORDER, base_port=BASE_PORT):
        self.latest = {n: (None, 0.0) for n in names}  # newest fresh JPEG and when it came (monotonic)
        self.arrivals = {n: collections.deque(maxlen=120) for n in names}
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
            if b64 and not msg.get("repeat"):
                now = time.monotonic()
                with self.lock:
                    self.latest[name] = (b64, now)
                    self.arrivals[name].append(now)
        s.close()

    def wait(self, timeout=10.0):
        """Block until every camera has sent a frame. Raises CameraDown."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if all(v[0] is not None for v in self.latest.values()):
                return
            time.sleep(0.05)
        missing = [n for n, v in self.latest.items() if v[0] is None]
        raise CameraDown(f"no frames from {', '.join(missing)}: is the camera server running? "
                         "(./understudy.sh cameras)")

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

    def health(self):
        """Per camera: ok, fresh frames per second over the last FPS_WINDOW, and seconds since the last fresh one."""
        now = time.monotonic()
        out = {}
        with self.lock:
            for n, (b64, t) in self.latest.items():
                recent = sum(1 for a in self.arrivals[n] if now - a <= FPS_WINDOW)
                age = None if b64 is None else now - t
                out[n] = dict(ok=age is not None and age <= MAX_AGE, fps=round(recent / FPS_WINDOW, 1),
                              age_s=None if age is None else round(age, 2))
        return out

    def close(self):
        self.running = False


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # the server runs as root
        pass
    return True


def view(cams, height=360):
    """Every camera side by side at the same height, with its frame rate. q or Esc quits."""
    while True:
        health, tiles = cams.health(), []
        for n in cams.latest:
            try:
                img = cams.grab([n])[n]
                img = cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height))
                text, colour = f"{n}  {health[n]['fps']:.0f} fps", (60, 220, 60)
            except CameraDown as e:
                img = np.full((height, height * 4 // 3, 3), 40, np.uint8)
                text, colour = str(e), (60, 60, 230)
            cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            tiles.append(img)
        cv2.imshow("cameras", np.hstack(tiles))
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Check the camera server's streams.")
    ap.add_argument("--view", action="store_true", help="then show the cameras (q quits)")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for every camera")
    ap.add_argument("--server-pid", type=int, help="stop waiting if this process (the camera server) exits")
    a = ap.parse_args()
    cams = ZmqCameras()
    try:
        end = time.monotonic() + a.timeout
        while True:
            try:
                cams.wait(timeout=0.5)
                break
            except CameraDown as e:
                if a.server_pid and not alive(a.server_pid):
                    print("  the camera server exited", file=sys.stderr)
                    return 1
                if time.monotonic() >= end:
                    print(f"  {e}", file=sys.stderr)
                    return 1
        time.sleep(FPS_WINDOW)  # a full window of frames for the rates
        low = False
        for n, h in cams.health().items():
            try:
                img = cams.grab([n])[n]
            except CameraDown as e:
                print(f"  {e}", file=sys.stderr)
                return 1
            low |= h["fps"] < 25
            print(f"  {n}: {img.shape[1]}x{img.shape[0]}, {h['fps']:.0f} fps" + (" (low)" if h["fps"] < 25 else ""))
        if low:
            print("  low frame rate: check the USB cables and the slots in cameras/cameras.json", file=sys.stderr)
        if a.view:
            view(cams)
        return 0
    finally:
        cams.close()


if __name__ == "__main__":
    sys.exit(main())
