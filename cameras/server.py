"""The camera server: the rig's USB cameras, captured with libuvc and published on ZMQ, one port per camera.

    ./understudy.sh cameras [--view]                  the usual way: starts this as root and checks every stream
    sudo cameras/.uvcenv/bin/python cameras/server.py   by hand
    python cameras/server.py --fake                     moving test patterns: no cameras, no root

Why libuvc and root instead of OpenCV:
- USB bandwidth. macOS reserves each camera's largest isochronous slot whatever mode it streams, and two of ours ask
  for ~3000 bytes per microframe: more than one USB controller holds for all three. libuvc lets us pick each slot.
- Drop-outs. A camera that falls off USB for a moment is reopened here; an OpenCV capture stays dead.
macOS only lets libusb claim a UVC camera from a root process, and the camera is gone from AVFoundation meanwhile.

The cameras, their modes, USB slots, rotations and UVC controls are in cameras.json. Camera i of that file publishes
on tcp://127.0.0.1:<port + i> (a subset keeps the same ports), one JSON message per frame:
    {"timestamps": {name: capture time, Unix s}, "images": {name: base64 JPEG}, "seq": n, "repeat": false}
That's lerobot's ZMQCamera format plus two keys. The JPEGs are ordinary ones, true colours in any viewer. (lerobot's
own image server JPEG-encodes RGB arrays, so its ZMQCamera would hand these out with red and blue swapped. The harness
reads them with harness/cameras.py.)
When a camera stalls, its last frame is re-sent with "repeat": true for up to --repeat-s, then nothing, so a reader
that checks frame age stops instead of using a frozen image. Controls are re-applied whenever cameras.json changes, so
exposure can be tuned while it runs.

stdout: status, and stats every 10 s. stderr: drop-outs and problems, for the terminal (they're in stdout too).
It stops on Ctrl+C, when the --parent-pid process exits, or when the stop file appears, since a user can't signal a
root process: touch cameras/.stop
"""
import argparse
import base64
import json
import logging
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "cameras.json")
STOP_FILE = os.path.join(HERE, ".stop")
STALL_S = 0.5  # no new frame for this long: start re-sending the last one
REPEAT_S = 5.0  # still none after this: stop sending the camera
READ_TIMEOUT_S = 0.25
REOPEN_AFTER = 3  # read timeouts in a row before the camera is closed and opened again
JPEG_QUALITY = 90
STATS_S = 10.0
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, -90: cv2.ROTATE_90_COUNTERCLOCKWISE}

quit_event = threading.Event()
wake = threading.Event()  # set by a capture thread when it has a new frame for the publisher
log_file = sys.stdout


def log(msg):
    print(f"[cameras {time.strftime('%H:%M:%S')}] {msg}", file=log_file, flush=True)


def warn(msg):
    """Something the person at the terminal should see; it goes to the log too."""
    log(msg)
    print(f"[cameras] {msg}", file=sys.stderr, flush=True)


def load(path=CONFIG):
    with open(path) as f:
        return json.load(f)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


# ---- USB bandwidth ------------------------------------------------------------------------------------------------
def libuvc_request(width, height, fps, factor):
    """Bytes per microframe libuvc requests for an MJPEG stream with this bandwidth_factor, with its integer arithmetic
    (stream.c in the pupil-labs fork). It then takes the camera's first alternate setting at least that big."""
    per_frame = int(width * height // 8 * factor)
    per_second = per_frame * (10_000_000 // round(10_000_000 / fps) + 1)
    return per_second // 1000 // 8 + 12


def bandwidth_factor(cam):
    """The bandwidth_factor that makes libuvc choose cam["slot"]: its request lands halfway between the next smaller
    slot and that one."""
    (w, h), fps, slot = cam["size"], cam["fps"], cam["slot"]
    if slot not in cam["slots"]:
        raise ValueError(f"slot {slot} isn't one of the camera's slots {cam['slots']}")
    below = max([s for s in cam["slots"] if s < slot], default=0)
    per_unit = (w * h // 8) * (10_000_000 // round(10_000_000 / fps) + 1) / 1000 / 8
    factor = ((below + slot) / 2 - 12) / per_unit
    asked = libuvc_request(w, h, fps, factor)
    if not below < asked <= slot:
        raise ValueError(f"no bandwidth factor gives slot {slot}: {factor:.3f} asks for {asked}")
    return factor


# ---- frame sources ------------------------------------------------------------------------------------------------
class UvcCamera:
    """A real camera through libuvc. read() returns a BGR frame, or None for one damaged in transit, and raises when no
    frame comes."""

    def __init__(self, name, cam):
        self.name, self.cam, self.cap = name, cam, None

    def open(self):
        import uvc

        devices = uvc.device_list()
        dev = next((d for d in devices if self.cam["usb"] in d["name"]), None)
        if dev is None:
            seen = ", ".join(d["name"] for d in devices) or "none"
            raise RuntimeError(f"{self.cam['usb']} isn't on USB (UVC cameras there: {seen})")
        cap = uvc.Capture(dev["uid"])
        try:
            (w, h), fps = self.cam["size"], self.cam["fps"]
            mode = next((m for m in cap.available_modes if tuple(m)[:3] == (w, h, fps)
                         and "MJP" in " ".join(str(v) for v in tuple(m)[3:]).upper()), None)
            if mode is None:
                raise RuntimeError(f"no MJPEG {w}x{h} at {fps} fps among {cap.available_modes}")
            cap.bandwidth_factor = bandwidth_factor(self.cam)
            cap.frame_mode = mode
            self.cap = cap
            self.defaults()
            cap.get_frame(timeout=2.0)  # starts the stream
        except Exception:
            self.cap = None
            cap.close()
            raise

    def defaults(self):
        """A steady frame rate (auto exposure may not stretch frames in dim light), and auto exposure unless
        cameras.json sets the mode: a camera another app left on manual can come up nearly black."""
        configured = self.cam.get("controls", {})
        for c in self.cap.controls:
            if c.display_name == "Auto Exposure Priority":
                try:
                    c.value = 0
                except Exception:
                    pass
            elif c.display_name == "Auto Exposure Mode" and c.display_name not in configured and c.value == 1:
                for mode in (8, 2):  # UVC: 8 aperture priority, 2 full auto, 1 manual
                    try:
                        c.value = mode
                    except Exception:
                        continue
                    if c.value == mode:
                        log(f"{self.name}: exposure was manual, now auto ({mode})")
                        break

    def set_controls(self, wanted):
        by_name = {c.display_name: c for c in self.cap.controls}
        done = []
        for name, value in wanted.items():
            c = by_name.get(name)
            if c is None:
                warn(f"{self.name}: no UVC control {name!r}; it has {', '.join(by_name)}")
                continue
            try:
                if not c.min_val <= int(value) <= c.max_val:
                    raise ValueError(f"{value} is outside {c.min_val}..{c.max_val}")
                c.value = int(value)
                done.append(f"{name}={c.value}")
            except Exception as e:
                warn(f"{self.name}: {name}: {e}")
        if done:
            log(f"{self.name}: " + ", ".join(done))

    def read(self):
        frame = self.cap.get_frame(timeout=READ_TIMEOUT_S)
        try:
            img = frame.bgr  # decodes the MJPEG
        except Exception:
            return None
        return img if getattr(frame, "data_fully_received", True) else None

    def close(self):
        if self.cap is not None:
            try:
                self.cap.close()  # hands the camera back to macOS
            except Exception:
                pass
            self.cap = None


class FakeCamera:
    """A moving test pattern at the camera's mode and rate, with a red top-left corner (before rotation) to check the
    colour order. gone = (start, seconds) makes it vanish that long after t0, like a camera dropping off USB."""

    def __init__(self, name, cam, gone=None, t0=None):
        self.name, self.cam, self.gone = name, cam, gone
        self.t0 = time.monotonic() if t0 is None else t0
        self.n, self.next_t = 0, 0.0

    def _gone(self):
        return self.gone is not None and self.gone[0] <= time.monotonic() - self.t0 < sum(self.gone)

    def open(self):
        if self._gone():
            raise RuntimeError("not on USB (--fake-gone)")

    def read(self):
        if self._gone():
            time.sleep(READ_TIMEOUT_S)
            raise TimeoutError("no frame (--fake-gone)")
        now = time.monotonic()
        self.next_t = max(self.next_t + 1.0 / self.cam["fps"], now)
        time.sleep(self.next_t - now)
        w, h = self.cam["size"]
        img = np.full((h, w, 3), 90, np.uint8)
        img[:h // 8, :w // 8] = (0, 0, 255)
        x = self.n * 4 % w
        img[:, x:x + 6] = 220
        cv2.putText(img, f"{self.name} {self.n}", (w // 8 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self.n += 1
        return img

    def set_controls(self, wanted):
        pass

    def close(self):
        pass


# ---- capture and publish ------------------------------------------------------------------------------------------
class Feed:
    """One camera's capture thread: opens the camera (again after a drop-out), reads frames, rotates and encodes them,
    and keeps the newest one for the publisher."""

    def __init__(self, name, cam, port, source, config_path=CONFIG):
        self.name, self.cam, self.port, self.source, self.config_path = name, cam, port, source, config_path
        self.rotation = ROTATIONS[cam.get("rotate", 0)]
        self.lock = threading.Lock()
        self.b64, self.t, self.fresh = None, 0.0, False
        self.streaming = False  # it has delivered a frame since the server started
        self.stats = dict(frames=0, damaged=0, repeated=0)
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"capture-{name}")

    def run(self):
        waiting = None
        while not quit_event.is_set():
            try:
                self.source.open()
            except Exception as e:
                if str(e) != waiting:
                    waiting = str(e)
                    (warn if self.streaming else log)(f"{self.name}: waiting for the camera: {e}")
                quit_event.wait(1.0)
                continue
            waiting = None
            (w, h), cam = self.cam["size"], self.cam
            log(f"{self.name}: {cam['usb']} {w}x{h} MJPEG at {cam['fps']} fps, USB slot {cam['slot']}, "
                f"rotated {cam.get('rotate', 0)}, on tcp://127.0.0.1:{self.port}")
            try:
                self.stream()
            finally:
                self.source.close()

    def stream(self):
        misses, mtime, next_check = 0, None, 0.0  # mtime None: apply the controls once the camera is open
        while not quit_event.is_set():
            if time.monotonic() >= next_check:
                next_check = time.monotonic() + 0.5
                m = self._mtime()
                if m != mtime:
                    mtime = m
                    self._apply_controls()
            try:
                img = self.source.read()
            except Exception:
                misses += 1
                if misses >= REOPEN_AFTER:
                    log(f"{self.name}: no frames for {misses * READ_TIMEOUT_S:.2f} s, reopening the camera")
                    return
                continue
            misses = 0
            if img is None:
                self.stats["damaged"] += 1
                continue
            if self.rotation is not None:
                img = cv2.rotate(img, self.rotation)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
            with self.lock:
                self.b64, self.t, self.fresh = b64, time.time(), True
            self.streaming = True
            self.stats["frames"] += 1
            wake.set()

    def _mtime(self):
        try:
            return os.path.getmtime(self.config_path)
        except OSError:
            return None

    def _apply_controls(self):
        try:
            controls = load(self.config_path)["cameras"][self.name].get("controls", {})
        except Exception as e:
            warn(f"{self.name}: can't read the controls in {self.config_path} ({e}); using the ones it started with")
            controls = self.cam.get("controls", {})
        self.source.set_controls(controls)

    def take_stats(self, secs):
        s, self.stats = self.stats, dict(frames=0, damaged=0, repeated=0)
        return f"{self.name} {s['frames'] / secs:.1f} fps, {s['damaged']} damaged, {s['repeated']} repeated"


def publish(feeds, socks, repeat_s=REPEAT_S):
    """Send every new frame. A camera with no new frame for STALL_S has its last one re-sent at its frame rate, flagged
    "repeat", until repeat_s after its last real frame; then it goes quiet until it's back."""
    state = {f.name: dict(seq=0, sent=0.0, stalled=False, quiet=False) for f in feeds}
    while not quit_event.is_set():
        wake.wait(0.01)
        wake.clear()
        now = time.time()
        for f in feeds:
            s = state[f.name]
            with f.lock:
                b64, t, fresh = f.b64, f.t, f.fresh
                f.fresh = False
            if b64 is None:
                continue
            if fresh:
                if s["stalled"]:
                    warn(f"{f.name}: frames back after {now - s['since']:.1f} s")
                    s["stalled"] = s["quiet"] = False
                repeat = False
            else:
                if now - t < STALL_S or now - s["sent"] < 1.0 / f.cam["fps"]:
                    continue
                if not s["stalled"]:
                    s.update(stalled=True, since=t)
                    warn(f"{f.name}: no frames for {now - t:.1f} s, re-sending its last one")
                if now - t > repeat_s:
                    if not s["quiet"]:
                        s["quiet"] = True
                        warn(f"{f.name}: still gone after {repeat_s:.0f} s, stopped sending it: check its USB cable")
                    continue
                repeat = True
                f.stats["repeated"] += 1
            s["seq"] += 1
            s["sent"] = now
            msg = {"timestamps": {f.name: t}, "images": {f.name: b64}, "seq": s["seq"], "repeat": repeat}
            try:
                socks[f.name].send_string(json.dumps(msg), zmq.NOBLOCK)
            except zmq.Again:
                pass


def parse_gone(specs):
    gone = {}
    for spec in specs:
        name, _, window = spec.partition("=")
        start, _, secs = window.partition(":")
        gone[name] = (float(start), float(secs))
    return gone


def main():
    global log_file
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cameras", help="comma-separated names from cameras.json (default: all of them)")
    ap.add_argument("--port", type=int, help="the first camera's port (default: cameras.json)")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--fake", action="store_true", help="test patterns instead of the cameras: no USB, no root")
    ap.add_argument("--fake-gone", action="append", default=[], metavar="NAME=START:SECONDS",
                    help="with --fake: that camera vanishes START s after the server starts, for SECONDS")
    ap.add_argument("--repeat-s", type=float, default=REPEAT_S, help="how long a stalled camera's last frame is re-sent")
    ap.add_argument("--parent-pid", type=int, help="exit when this process (the launcher) does")
    ap.add_argument("--stop-file", default=STOP_FILE, help="exit when this file appears")
    a = ap.parse_args()

    cfg = load(a.config)
    names = list(cfg["cameras"])
    wanted = [n.strip() for n in a.cameras.split(",") if n.strip()] if a.cameras else names
    unknown = [n for n in wanted if n not in names]
    if unknown:
        sys.exit(f"no camera {', '.join(unknown)} in {a.config}: it has {', '.join(names)}")
    port0 = a.port or cfg.get("port", 5555)
    if not a.fake:
        if os.geteuid() != 0:
            sys.exit("the cameras need root: ./understudy.sh cameras, or sudo cameras/.uvcenv/bin/python "
                     "cameras/server.py (or --fake for test patterns)")
        # libuvc prints its clock corrections straight to fd 1: keep the log on a copy of it, send fd 1 to /dev/null
        log_file = os.fdopen(os.dup(1), "w")
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        logging.getLogger("uvc").setLevel(logging.CRITICAL)  # it logs every damaged frame; the stats count them
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: quit_event.set())
    if os.path.exists(a.stop_file):
        os.remove(a.stop_file)

    ctx = zmq.Context()
    socks = {}
    for n in wanted:
        port = port0 + names.index(n)
        s = socks[n] = ctx.socket(zmq.PUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SNDHWM, 2)
        try:
            s.bind(f"tcp://127.0.0.1:{port}")
        except zmq.ZMQError as e:
            for s in socks.values():
                s.close()
            ctx.term()
            sys.exit(f"can't publish {n} on port {port} ({e}): is another camera server running? "
                     "pgrep -fl cameras/server.py")

    gone, t0 = parse_gone(a.fake_gone), time.monotonic()
    feeds = []
    for n in wanted:
        cam = cfg["cameras"][n]
        source = FakeCamera(n, cam, gone.get(n), t0) if a.fake else UvcCamera(n, cam)
        feeds.append(Feed(n, cam, port0 + names.index(n), source, a.config))
    sender = threading.Thread(target=publish, args=(feeds, socks, a.repeat_s), daemon=True, name="publisher")
    for f in feeds:
        f.thread.start()
    sender.start()
    log(f"{'test patterns' if a.fake else 'libuvc'} for {', '.join(wanted)}, pid {os.getpid()}")

    ready, t_stats, nag = False, time.monotonic(), time.monotonic()
    try:
        while not quit_event.wait(0.2):
            if a.parent_pid and not alive(a.parent_pid):
                log("the launcher exited")
                break
            if os.path.exists(a.stop_file):
                os.remove(a.stop_file)
                log("stop file seen")
                break
            now = time.monotonic()
            if not ready and all(f.streaming for f in feeds):
                ready = True
                log(f"ready: {', '.join(wanted)}")
            elif not ready and now - nag >= 15:
                nag = now
                warn("still waiting for " + ", ".join(f.name for f in feeds if not f.streaming))
            if now - t_stats >= STATS_S:
                log(" | ".join(f.take_stats(now - t_stats) for f in feeds))
                t_stats = now
    finally:
        quit_event.set()
        for f in feeds:
            f.thread.join(timeout=2)  # each closes its camera on the way out
        sender.join(timeout=1)
        for s in socks.values():
            s.close()
        ctx.term()
        log("stopped, cameras released")


if __name__ == "__main__":
    main()
