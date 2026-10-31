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

from . import overlay, sim
from .cameras import CameraDown, Publisher, sizes as camera_sizes
from .controller import Controller
from .motion import Planner
from .perception import ARM_COLOURS, SimPerception, TablePerception
from .policy import TASK, PolicyRunner
from .recorder import DATASETS, Recorder
from .scene import SIDES, Scene
from .skills import Skills, obj_pose

PORT = int(os.environ.get("ROBOTD_PORT", 5560))
SAG_DIR = os.path.expanduser("~/.cache/understudy")  # the arms' learned gravity sag, kept between runs
CAMERAS = ("overhead", "left_wrist", "right_wrist")


class SimCameras:
    """The simulator's cameras, rendered at the rig's sizes (cameras/cameras.json)."""

    def __init__(self, scene, world):
        self.scene, self.world = scene, world
        self.sizes = camera_sizes()
        self.last_t = {}

    def grab(self, names=CAMERAS, snap=None):
        snap = snap or self.world.snapshot()
        out = {}
        down = []
        for n in names:  # every camera's fault clock moves on, even when another one is down
            dt = self.world.t - self.last_t.get(n, self.world.t)
            self.last_t[n] = self.world.t
            if self.world.camera_down(n, dt):
                down.append(n)
        if down:
            raise CameraDown(f"{', '.join(down)}: no frame (simulated drop-out)")
        for n in names:
            if n == "overhead":
                out[n] = sim.render(self.scene, snap, self.scene.camera)
            else:
                side = n.split("_")[0]
                w, h = self.sizes.get(n, (640, 480))
                cam = sim.wrist_camera(self.scene, side, snap["q"][side], width=w, height=h)
                out[n] = sim.render(self.scene, snap, cam, own=side)
        return out

    def health(self):
        return {n: dict(ok=True, fps=30.0, age_s=0.0) for n in CAMERAS}


class Core:
    def __init__(self, sim_mode=True, speed=1.0, objects="brush", noise=0.0, arm=True, quiet=False,
                 dataset="understudy_pass", datasets=DATASETS, vision=False, stream_cameras=False, sag=True,
                 faults=None):
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
            self.world = sim.SimWorld(self.scene, objects, noise, sag=sag, faults=faults)
            self.world.on_fault = lambda kind, what: self.event(f"simulated fault: {what}", "warn")
            ios = self.world.io
            self.cameras = SimCameras(self.scene, self.world)
            self.perception = SimPerception(self.world, self.scene)
        else:
            from .bus import BusArmIO
            from .cameras import ZmqCameras

            self.world = None
            self.cameras = ZmqCameras()
            self.cameras.wait()
            ios = {s: BusArmIO(self.scene.ports[s], self.scene.calibration_ids[s], self.scene.stroke_mm)
                   for s in SIDES}
        sag_file = None if sim_mode else os.path.join(SAG_DIR, "_".join(self.scene.calibration_ids[s] for s in SIDES)
                                                       + ".json")
        self.ctl = Controller(ios, self.world, speed, on_event=self.event).set_arms(self.scene.arms, sag_file)
        if not sim_mode:
            self.ctl.set_leaders(self.connect_leaders())
        self.ctl.start()
        if vision or not sim_mode:  # objects from the overhead image, as on the real arms
            self.perception = TablePerception(self.scene, self.cameras, self.ctl, self.holding,
                                              ignore_colours=() if sim_mode else ARM_COLOURS)
        self.planner = Planner(self.scene, self.holding)
        self.skills = Skills(self)
        self.policy = PolicyRunner(self)
        self.publisher = None
        if sim_mode and stream_cameras:  # the simulator's cameras on the rig's ports, for viewers and the UI
            try:
                self.publisher = Publisher(lambda names: self.cameras.grab(names))
            except zmq.ZMQError as e:
                self.event(f"camera ports busy ({e}): the simulator's cameras aren't streamed", "warn")
        if dataset:
            self.recorder = Recorder(self, dataset, datasets)
        if sim_mode and arm:
            self.ctl.arm()

    def connect_leaders(self):
        """The leader arms, for takeover; one that isn't there only turns takeover off for its follower."""
        from .bus import LeaderIO

        leaders = {}
        for s, cfg in self.scene.leaders.items():
            try:
                leaders[s] = LeaderIO(cfg["port"], cfg["calibration"], self.scene.stroke_mm)
            except Exception as e:
                self.event(f"no {s} leader ({e}): taking over the {s} arm by hand is off", "warn")
        return leaders

    @property
    def interventions(self):
        """Times Vlad took an arm over with its leader."""
        return self.ctl.takeovers

    def close(self):
        if self.policy.active():
            self.policy.stop("robotd closing")
        if self.publisher is not None:
            self.publisher.close()
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
            if h is not None:  # where what it holds is now: it moves with the gripper
                p_t, R_t = self.scene.arms[s].tool(q)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R_t, p_t
                T = T @ h["rel"]
                ends = [T @ [0, 0, k * h["size"][2] / 2, 1] for k in (1, -1)]
                out[s]["held"] = dict(name=h["name"], center_cm=[round(float(v) * 100, 1) for v in T[:3, 3]],
                                      top_cm=[round(float(v) * 100, 1) for v in ends[0][:3]],
                                      bottom_cm=[round(float(v) * 100, 1) for v in ends[1][:3]],
                                      size_cm=[round(float(v) * 100, 1) for v in h["size"]])
            if s in snap["temps"]:
                out[s]["temp_c"] = max(snap["temps"][s])
            if s in snap["grip_ma"]:  # the gripper motor's current now, and its peak since the last grip_force
                out[s]["grip_ma"] = round(snap["grip_ma"][s])
                out[s]["grip_peak_ma"] = round(snap["grip_peak"].get(s, 0.0))
        return out

    def busy(self):
        return bool(self.skill.get("running"))

    # ---- primitives -----------------------------------------------------------------------------------------------
    def start_skill(self, name, args):
        with self.skill_lock:
            if self.busy():
                return dict(ok=False, errors=[f"{self.skill['name']} is still running"], fail="busy")
            if self.policy.active():
                return dict(ok=False, errors=["a policy is driving the arms: policy_stop first"], fail="busy")
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
            if self.recorder is not None:
                self.recorder.on_step(res)
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
                    img = overlay.draw(img, self.scene, dets, self.arms_state())
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images[n] = base64.b64encode(buf.tobytes()).decode()
        return dict(ok=True, objects=dets, images=images, arms=self.arms_state())

    # ---- the API --------------------------------------------------------------------------------------------------
    def handle(self, req):
        cmd = req.get("cmd")
        if cmd == "status":
            sk = dict(self.skill)
            mode = "policy" if self.policy.active() else "skill" if self.busy() else "idle"
            return dict(ok=True, mode=mode, armed=self.ctl.armed(), busy=self.busy(), skill=sk, arms=self.arms_state(),
                        events=self.events_since(req.get("since", 0)), last_event=self.ev_id,
                        episode=self.recorder.status() if self.recorder else None, cameras=self.cameras.health(),
                        policy=self.policy.status(), interventions=self.interventions, sim=self.sim)
        if cmd == "look":
            return self.look(req.get("cams", ["overhead"]), req.get("annotate", True))
        if cmd == "detect":
            dets = self.perception.detect() if self.perception else []
            self.last_objects = dets
            return dict(ok=True, objects=dets)
        if cmd == "skill":
            return self.start_skill(req.get("name"), req.get("args") or {})
        if cmd == "grip_force":  # cap a gripper's torque, % of the motor's, and start a new current peak
            if req.get("arm") not in SIDES:
                return dict(ok=False, errors=["arm is left or right"])
            self.ctl.grip_force(req["arm"], float(req.get("pct", 30)))
            return dict(ok=True)
        if cmd == "hold":  # what a gripper holds, as it stood on the table (none: nothing): it moves with the gripper
            side = req.get("arm")
            if side not in SIDES:
                return dict(ok=False, errors=["arm is left or right"])
            if not req.get("name"):
                self.holding[side] = None
                return dict(ok=True)
            try:
                size = [float(v) / 100 for v in req["size_cm"]]
                T_obj = obj_pose(np.array(req["center_cm"][:2], dtype=float) / 100, float(req["z_cm"]) / 100,
                                 float(req.get("yaw") or 0.0))
            except (KeyError, TypeError, ValueError) as e:
                return dict(ok=False, errors=[f"hold needs name, size_cm, center_cm and z_cm: {e}"])
            p_t, R_t = self.scene.arms[side].tool(self.ctl.snapshot()["q"][side])
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R_t, p_t
            self.holding[side] = dict(name=str(req["name"]), size=size, rel=np.linalg.inv(T) @ T_obj,
                                      grasp_z=float(p_t[2]), width=float(self.ctl.snapshot()["q"][side][5]))
            return dict(ok=True)
        if cmd == "stop":
            self.cancel.set()
            if self.policy.active():
                self.policy.stop(req.get("reason", "stop requested"))
            self.ctl.stop(req.get("reason", "stop requested"))
            return dict(ok=True)
        if cmd == "to_table":  # overhead pixels -> table cm, at a height above the table (default 0)
            px = np.asarray(req.get("px", []), dtype=float).reshape(-1, 2)
            pts = self.scene.camera.to_table(px, z=float(req.get("z_cm", 0.0)) / 100) if len(px) else []
            return dict(ok=True, cm=[[round(float(v) * 100, 2) for v in p[:2]] for p in pts])
        if cmd == "to_pixels":  # table cm (x, y, optional height) -> overhead pixels
            pts = [list(p) + [0.0] * (3 - len(p)) for p in req.get("cm", [])]
            px = self.scene.camera.project(np.asarray(pts, dtype=float) / 100) if pts else []
            return dict(ok=True, px=[[round(float(v), 1) for v in p] for p in px])
        if cmd == "policy_start":
            return self.policy.start(req.get("path", ""), req.get("max_s"), req.get("task") or TASK)
        if cmd == "policy_stop":
            return self.policy.stop(req.get("reason", "policy_stop"))
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


class Trace:
    """Every request and reply as JSON lines, for replaying real-arm sessions in tests. status is kept at most once
    a second (clients poll it every 50 ms)."""

    def __init__(self, path):
        self.f = open(path, "a")
        self.last_status = 0.0

    def write(self, req, rep):
        now = time.time()
        if req.get("cmd") == "status":
            if now - self.last_status < 1.0:
                return
            self.last_status = now
        self.f.write(json.dumps({"t": round(now, 3), "req": req, "rep": rep}, default=to_json) + "\n")
        self.f.flush()


def serve(core, port=PORT, stop_event=None, trace=None):
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
            if trace is not None:
                trace.write(json.loads(msg), rep)
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
                try:
                    core.ctl.arm()
                except Exception as e:  # a bus error part way: nothing left half on, and the keys keep working
                    print(f"[robotd] torque on failed ({e}): switching torque off, hold the arms", flush=True)
                    try:
                        core.ctl.disarm()
                    except Exception as e2:
                        print(f"[robotd] torque off failed too ({e2}): quit with Ctrl+C", flush=True)
            elif ch == " ":
                core.cancel.set()
                if core.policy.active():
                    core.policy.stop("space pressed")
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
    ap.add_argument("--dataset", default="understudy_pass", help="episodes go to datasets/<name>")
    ap.add_argument("--vision", action="store_true", help="simulator: find objects in the rendered overhead image")
    ap.add_argument("--no-camera-streams", action="store_true",
                    help="simulator: don't publish the rendered cameras on the camera ports")
    ap.add_argument("--no-sag", action="store_true", help="simulator: servos that don't sag under gravity")
    ap.add_argument("--faults", default="", metavar="KIND=P,...",
                    help=f"simulator: fault probabilities, kinds {', '.join(sim.FAULTS)} (e.g. miss=0.1,slip=0.02)")
    ap.add_argument("--trace", metavar="FILE", help="append every request and reply to FILE as JSON lines")
    ap.add_argument("--sag-off", action="store_true", help="no gravity feed-forward and no sag learning")
    a = ap.parse_args()
    faults = {k: float(v) for k, v in (f.split("=") for f in a.faults.split(",") if f)}
    core = Core(a.sim, a.speed, a.objects, a.noise / 100, dataset=a.dataset, vision=a.vision,
                stream_cameras=not a.no_camera_streams, sag=not a.no_sag, faults=faults)
    if a.sag_off:
        core.ctl.freeze_sag()
    quit_event = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, args=(core, quit_event), daemon=True).start()
    print(f"[robotd] {'simulator' if a.sim else 'real arms'} on tcp://127.0.0.1:{a.port}"
          + ("" if core.ctl.armed() else ": press a to switch torque on"), flush=True)
    trace = Trace(a.trace) if a.trace else None
    server = threading.Thread(target=serve, args=(core, a.port, quit_event, trace), daemon=True)
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
