"""The camera server in --fake mode (test patterns: no USB, no root), on ports away from the real ones."""
import base64
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest
import zmq

from harness import cameras
from harness.cameras import CameraDown, ZmqCameras

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER = os.path.join(ROOT, "cameras", "server.py")
spec = importlib.util.spec_from_file_location("camera_server", SERVER)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.fixture
def start(tmp_path):
    procs = []

    def run(port, *args):
        p = subprocess.Popen([sys.executable, SERVER, "--fake", "--port", str(port), "--stop-file",
                              str(tmp_path / "stop"), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        procs.append(p)
        return p

    yield run
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


def subscriber(port):
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, 100)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.connect(f"tcp://127.0.0.1:{port}")
    return s


def first_message(port, timeout=10):
    s = subscriber(port)
    end = time.monotonic() + timeout
    try:
        while time.monotonic() < end:
            try:
                return json.loads(s.recv_string())
            except zmq.Again:
                pass
        raise AssertionError(f"nothing on port {port}")
    finally:
        s.close()


def red(px):
    return px[2] > 200 and px[0] < 60 and px[1] < 60


def test_every_slot_in_the_rig_gets_a_bandwidth_factor():
    for cam in server.load()["cameras"].values():
        for slot in cam["slots"]:
            below = max([s for s in cam["slots"] if s < slot], default=0)
            f = server.bandwidth_factor(dict(cam, slot=slot))
            assert below < server.libuvc_request(*cam["size"], cam["fps"], f) <= slot


def test_frames_come_in_true_colour_at_the_rig_sizes(start):
    start(5870)
    cams = ZmqCameras(base_port=5870)
    try:
        cams.wait(timeout=15)
        time.sleep(cameras.FPS_WINDOW + 0.3)
        imgs = cams.grab()
        assert imgs["overhead"].shape == (480, 640, 3)
        assert imgs["left_wrist"].shape == (768, 1024, 3)
        assert imgs["right_wrist"].shape == (480, 640, 3)
        assert red(imgs["overhead"][5, 5])
        assert red(imgs["right_wrist"][-5, -5])  # rotated 180: the red corner is bottom right
        health = cams.health()
        assert all(h["ok"] and h["fps"] >= 25 for h in health.values()), health
    finally:
        cams.close()


def test_a_stalled_camera_is_repeated_then_dropped_then_back(start):
    start(5880, "--cameras", "overhead", "--fake-gone", "overhead=1.0:2.0", "--repeat-s", "1.2")
    s = subscriber(5880)
    msgs = []
    end = time.monotonic() + 12
    try:
        while time.monotonic() < end:
            try:
                m = json.loads(s.recv_string())
            except zmq.Again:
                continue
            msgs.append((time.monotonic(), m["repeat"], m["seq"], m["timestamps"]["overhead"]))
            repeats = [i for i, x in enumerate(msgs) if x[1]]
            if repeats and sum(1 for x in msgs[repeats[-1]:] if not x[1]) >= 10:
                break
    finally:
        s.close()
    repeats = [x for x in msgs if x[1]]
    assert repeats, "no repeated frames during the stall"
    assert all(r[3] == repeats[0][3] for r in repeats), "repeats carry the last real frame's capture time"
    gaps = np.diff([x[0] for x in msgs])
    assert gaps.max() > 0.5, "it didn't go quiet after --repeat-s"
    assert not msgs[-1][1], "fresh frames didn't come back"
    assert [x[2] for x in msgs] == sorted(x[2] for x in msgs)


def test_repeated_frames_dont_count_as_fresh(monkeypatch):
    monkeypatch.setattr(cameras, "MAX_AGE", 0.3)
    stop = threading.Event()

    def publish():
        p = zmq.Context.instance().socket(zmq.PUB)
        p.setsockopt(zmq.LINGER, 0)
        p.bind("tcp://127.0.0.1:5891")
        b64 = base64.b64encode(cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()).decode()
        t0 = time.monotonic()
        while not stop.is_set():
            repeat = time.monotonic() - t0 > 0.5
            p.send_string(json.dumps({"timestamps": {"overhead": 0}, "images": {"overhead": b64}, "repeat": repeat}))
            time.sleep(0.02)
        p.close()

    threading.Thread(target=publish, daemon=True).start()
    cams = ZmqCameras(names=("overhead",), base_port=5891)
    try:
        cams.wait(timeout=5)
        time.sleep(1.0)
        with pytest.raises(CameraDown):
            cams.grab(["overhead"])
        assert not cams.health()["overhead"]["ok"]
    finally:
        stop.set()
        cams.close()


def test_a_second_server_on_the_same_ports_refuses(start):
    start(5884, "--cameras", "overhead")
    first_message(5884)
    second = start(5884, "--cameras", "overhead")
    _, err = second.communicate(timeout=15)
    assert second.returncode != 0 and "another camera server" in err


def test_it_stops_on_the_stop_file_and_with_its_launcher(start, tmp_path):
    p = start(5886, "--cameras", "overhead")
    first_message(5886)
    (tmp_path / "stop").touch()
    assert p.wait(5) == 0
    launcher = subprocess.Popen(["sleep", "60"])
    q = start(5886, "--cameras", "overhead", "--parent-pid", str(launcher.pid))
    first_message(5886)
    launcher.kill()
    launcher.wait()
    assert q.wait(5) == 0


def test_robotd_reports_camera_health():
    from harness.robotd import Core

    core = Core(sim_mode=True, speed=0, quiet=True, dataset=None)
    try:
        assert core.handle(dict(cmd="status"))["cameras"]["overhead"]["ok"]
    finally:
        core.close()


def test_a_camera_drop_out_marks_the_episode_invalid(tmp_path):
    from harness.robotd import Core

    core = Core(sim_mode=True, speed=0, quiet=True, dataset="drop", datasets=str(tmp_path))
    grab, calls = core.cameras.grab, [0]

    def flaky(names, snap=None):
        calls[0] += 1
        if 5 <= calls[0] < 10:
            raise CameraDown("overhead: no frame for 1.2 s")
        return grab(names, snap)

    core.cameras.grab = flaky
    try:
        assert core.handle(dict(cmd="episode_begin", task="Pass the brush to the other arm"))["ok"]
        assert core.start_skill("pick", dict(arm="left", target="brush"))["ok"]
        while core.busy():
            time.sleep(0.005)
        end = core.handle(dict(cmd="episode_end"))
        assert end["ok"], end
        robot = json.load(open(end["sidecar"]))["robot"]
        assert not robot["valid"] and robot["camera_gaps"] == 5 and "camera drop-out" in robot["note"]
    finally:
        core.close()
