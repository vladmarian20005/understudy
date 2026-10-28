"""robotd: the only process that drives the arms.

    python -m harness.robotd --sim            simulated arms, objects and cameras (armed at start)
    python -m harness.robotd                  the real arms: press a to switch torque on

It owns the 50 Hz controller, the planner, perception and the primitives, and serves one JSON request at a time on
ZMQ REP (tcp://127.0.0.1:5560, ROBOTD_PORT). No request can switch torque on: only the a key can.
Keys: a = torque on (hold where they are), space = stop, x twice = torque off, q = quit.
"""
import argparse
import base64
import collections
import itertools
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import zmq

from . import sim
from .controller import Controller
from .motion import Planner
from .perception import SimPerception
from .scene import SIDES, Scene
from .skills import Skills

PORT = int(os.environ.get("ROBOTD_PORT", 5560))
CAMERAS = ("overhead", "left_wrist", "right_wrist")


class SimCameras:
    def __init__(self, scene, world):
        self.scene, self.world = scene, world

    def grab(self, names=CAMERAS, snap=None):
        snap = snap or self.world.snapshot()
        out = {}
        for n in names:
            if n == "overhead":
                out[n] = sim.render(self.scene, snap, self.scene.camera)
            else:
                side = n.split("_")[0]
                out[n] = sim.render(self.scene, snap, sim.wrist_camera(self.scene, side, snap["q"][side]), own=side)
        return out


class Core:
    def __init__(self, sim_mode=True, speed=1.0, objects="brush", noise=0.0, arm=True, quiet=False):
        self.scene = Scene()
        self.quiet = quiet
        self.events = collections.deque(maxlen=500)
        self.ev_id = 0
        self.ev_lock = threading.Lock()
        self.holding = {s: None for s in SIDES}
        self.last_objects = []
        self.cancel = threading.Event()
        self.skill = {}
        self.skill_ids = itertools.count(1)
        self.skill_lock = threading.Lock()
        self.recorder = None
        self.sim = sim_mode
        if sim_mode:
            self.world = sim.SimWorld(self.scene, objects, noise)
            ios = self.world.io
            self.perception = SimPerception(self.world, self.scene)
            self.cameras = SimCameras(self.scene, self.world)
        else:
            from .bus import BusArmIO

            self.world = None
            ios = {s: BusArmIO(self.scene.ports[s], self.scene.calibration_ids[s], self.scene.stroke_mm)
                   for s in SIDES}
            self.perception = None
            self.cameras = None
        self.ctl = Controller(ios, self.world, speed, on_event=self.event).start()
        self.planner = Planner(self.scene)
        self.skills = Skills(self)
        if sim_mode and arm:
            self.ctl.arm()

    def close(self):
        self.ctl.close()

    # ---- time and events ------------------------------------------------------------------------------------------
    def clock(self):
        """Seconds since episode_begin while recording, else Unix time."""
        if self.recorder is not None and self.recorder.active():
            return self.ctl.now() - self.recorder.t0
        return time.time()

    def event(self, text, kind="info"):
        with self.ev_lock:
            self.ev_id += 1
            e = dict(id=self.ev_id, t=round(self.clock(), 2), kind=kind, text=text)
            self.events.append(e)
        if not self.quiet:
            print(f"[robotd {time.strftime('%H:%M:%S')}] {text}", flush=True)
        return e

    def events_since(self, since):
        with self.ev_lock:
            return [e for e in self.events if e["id"] > since]

    # ---- state ----------------------------------------------------------------------------------------------------
    def arms_state(self):
        snap = self.ctl.snapshot()
        out = {}
        for s in SIDES:
            q = snap["q"][s]
            p, pitch, yaw = self.scene.arms[s].pose(q)
            h = self.holding[s]
            out[s] = dict(tip_cm=[round(float(v) * 100, 1) for v in p], pitch=round(pitch, 1), yaw=round(yaw, 1),
                          width_mm=round(float(q[5]), 1), holding=h is not None, object=h["name"] if h else None,
                          mode=snap["mode"][s], joints=[round(float(v), 1) for v in q[:5]])
            if s in snap["temps"]:
                out[s]["temp_c"] = max(snap["temps"][s])
        return out

    def busy(self):
        return bool(self.skill.get("running"))

    # ---- primitives -----------------------------------------------------------------------------------------------
    def start_skill(self, name, args):
        with self.skill_lock:
            if self.busy():
                return dict(ok=False, errors=[f"{self.skill['name']} is still running"], fail="busy")
            if not self.ctl.armed():
                return dict(ok=False, errors=["the arms are off: Vlad switches torque on (key a)"], fail="torque_off")
            self.cancel.clear()
            sid = next(self.skill_ids)
            self.skill = dict(id=sid, name=name, args=args, running=True, result=None)

        def go():
            try:
                res = self.skills.run(name, args)
            except Exception as e:  # a bug: hold the arms and report it
                self.ctl.stop(f"internal error in {name}: {e!r}")
                res = dict(op=name, outcome="failed", fail=None, why=f"internal error: {e!r}",
                           arms=self.arms_state(), events=[])
                self.event(f"{name} crashed: {e!r}", "error")
            self.skill.update(running=False, result=res)

        threading.Thread(target=go, daemon=True, name=f"skill-{name}").start()
        return dict(ok=True, id=sid)

    def look(self, cams=("overhead",), annotate=True):
        dets = self.perception.detect() if self.perception else []
        self.last_objects = dets
        images = {}
        if self.cameras is not None:
            for n, img in self.cameras.grab([c for c in cams if c in CAMERAS]).items():
                if n == "overhead" and annotate:
                    img = img.copy()
                    for d in dets:
                        u, v = d["px"]
                        cv2.putText(img, d["id"], (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images[n] = base64.b64encode(buf.tobytes()).decode()
        return dict(ok=True, objects=dets, images=images, arms=self.arms_state())

    # ---- the API --------------------------------------------------------------------------------------------------
    def handle(self, req):
        cmd = req.get("cmd")
        if cmd == "status":
            sk = dict(self.skill)
            return dict(ok=True, armed=self.ctl.armed(), busy=self.busy(), skill=sk, arms=self.arms_state(),
                        events=self.events_since(req.get("since", 0)), last_event=self.ev_id,
                        episode=self.recorder.status() if self.recorder else None)
        if cmd == "look":
            return self.look(req.get("cams", ["overhead"]), req.get("annotate", True))
        if cmd == "detect":
            dets = self.perception.detect() if self.perception else []
            self.last_objects = dets
            return dict(ok=True, objects=dets)
        if cmd == "skill":
            return self.start_skill(req.get("name"), req.get("args") or {})
        if cmd == "stop":
            self.cancel.set()
            self.ctl.stop(req.get("reason", "stop requested"))
            return dict(ok=True)
        if cmd in ("episode_begin", "episode_end", "episode_label"):
            if self.recorder is None:
                return dict(ok=False, errors=["no recorder in this robotd"])
            return getattr(self.recorder, cmd[len("episode_"):])(**{k: v for k, v in req.items() if k != "cmd"})
        return dict(ok=False, errors=[f"unknown cmd {cmd!r}"])


def to_json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"can't send {type(o).__name__}")


def serve(core, port=PORT, stop_event=None):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{port}")
    sock.setsockopt(zmq.RCVTIMEO, 200)
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                msg = sock.recv_string()
            except zmq.Again:
                continue
            try:
                rep = core.handle(json.loads(msg))
            except Exception as e:
                rep = dict(ok=False, errors=[f"{type(e).__name__}: {e}"])
            sock.send_string(json.dumps(rep, default=to_json))
    finally:
        sock.close()


def keyboard(core, quit_event):
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    last_x = 0.0
    try:
        tty.setcbreak(fd)
        while not quit_event.is_set():
            ch = sys.stdin.read(1)
            if ch == "a":
                core.ctl.arm()
            elif ch == " ":
                core.cancel.set()
                core.ctl.stop("space pressed")
            elif ch == "x":
                if time.time() - last_x < 1.5:
                    core.ctl.disarm()
                else:
                    print("[robotd] press x again within 1.5 s for torque off: hold the arms, they will drop",
                          flush=True)
                last_x = time.time()
            elif ch == "q":
                quit_event.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="simulated arms, objects and cameras")
    ap.add_argument("--speed", type=float, default=1.0, help="simulator speed (1 = real time, 0 = flat out)")
    ap.add_argument("--objects", default="brush", help=f"simulator scene: {', '.join(sim.SCENES)}")
    ap.add_argument("--noise", type=float, default=0.0, help="simulated perception error, cm")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    core = Core(a.sim, a.speed, a.objects, a.noise / 100)
    quit_event = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, args=(core, quit_event), daemon=True).start()
    print(f"[robotd] {'simulator' if a.sim else 'real arms'} on tcp://127.0.0.1:{a.port}"
          + ("" if core.ctl.armed() else ": press a to switch torque on"), flush=True)
    server = threading.Thread(target=serve, args=(core, a.port, quit_event), daemon=True)
    server.start()
    try:
        quit_event.wait()
    except KeyboardInterrupt:
        pass
    if not a.sim:
        print("[robotd] quitting: torque off in 1 s, hold the arms", flush=True)
        time.sleep(1)
        core.ctl.disarm()
    core.close()


if __name__ == "__main__":
    main()
