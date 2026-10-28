"""Blocking Python client for robotd: one function per command, each returns robotd's reply.

    from harness import robot
    seen = robot.look()                       # objects + the overhead image
    r = robot.pick("left", "#1")              # waits until the motion is done
    if r["outcome"] != "done": print(r["fail"], r["why"])

Motion commands return the result described in the contracts doc: op, arm, outcome (done | refused | failed |
stopped), fail, why, t, arms, events, plus width_mm after a grasp. Arguments left as None aren't sent.
"""
import base64
import json
import os
import time

import zmq

PORT = int(os.environ.get("ROBOTD_PORT", 5560))
TIMEOUT_S = 120


class RobotdDown(Exception):
    pass


def call(cmd, timeout_s=10.0, **kw):
    """Send one request to robotd and return its reply."""
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    s.setsockopt(zmq.SNDTIMEO, 1000)
    s.connect(f"tcp://127.0.0.1:{PORT}")
    try:
        s.send_string(json.dumps(dict(cmd=cmd, **kw)))
        return json.loads(s.recv_string())
    except zmq.Again:
        raise RobotdDown(f"robotd isn't answering on port {PORT}: start it with python -m harness.robotd --sim")
    finally:
        s.close()


def _motion(name, **args):
    args = {k: v for k, v in args.items() if v is not None}
    r = call("skill", name=name, args=args)
    if not r.get("ok"):
        return dict(op=name, outcome="refused", fail=r.get("fail", "busy"), why="; ".join(r.get("errors", [])))
    end = time.time() + TIMEOUT_S
    while time.time() < end:
        sk = call("status")["skill"]
        if sk.get("id") == r["id"] and not sk.get("running"):
            return sk["result"]
        time.sleep(0.05)
    call("stop", reason=f"{name} timed out")
    return dict(op=name, outcome="stopped", fail="timeout", why=f"{name} didn't finish in {TIMEOUT_S} s")


def reach(arm, target=None, xy=None, z=None, yaw=None, width=None, offset=None):
    return _motion("reach", arm=arm, target=target, xy=xy, z=z, yaw=yaw, width=width, offset=offset)


def grasp(arm, width=None, offset=None):
    return _motion("grasp", arm=arm, width=width, offset=offset)


def move_to(arm, xy, z=None, yaw=None, pitch=None):
    return _motion("move_to", arm=arm, xy=xy, z=z, yaw=yaw, pitch=pitch)


def release(arm, set_down=None):
    return _motion("release", arm=arm, set_down=set_down)


def handover(giver, taker, mode=None, at=None):
    return _motion("handover", giver=giver, taker=taker, mode=mode, at=at)


def home(arms=None):
    return _motion("home", arms=arms)


def pick(arm, target, width=None, offset=None):
    return _motion("pick", arm=arm, target=target, width=width, offset=offset)


def place(arm, xy=None, on=None):
    return _motion("place", arm=arm, xy=xy, on=on)


def look(cams=("overhead",), annotate=True):
    return call("look", cams=list(cams), annotate=annotate)


def image(reply, cam="overhead"):
    """A camera image from a look() reply, as a BGR numpy array."""
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(base64.b64decode(reply["images"][cam]), np.uint8), cv2.IMREAD_COLOR)


def detect():
    return call("detect")


def status(since=0):
    return call("status", since=since)


def stop(reason="stop requested"):
    return call("stop", reason=reason)


def episode_begin(task, source=None, giver=None, taker=None, target_cm=None):
    return call("episode_begin", task=task, source=source or {"kind": "pingpong"}, giver=giver, taker=taker,
                target_cm=target_cm)


def episode_end(valid=True, note=None):
    return call("episode_end", timeout_s=120, valid=valid, note=note)


def episode_label(episode, section, data):
    return call("episode_label", episode=episode, section=section, data=data)


def policy_start(path, max_s=None, task=None):
    """Start a trained policy (a checkpoint or training run directory) driving both arms; returns at once."""
    return call("policy_start", timeout_s=120, path=path, max_s=max_s, task=task)


def policy_stop(reason="policy_stop"):
    return call("policy_stop", reason=reason)


def run_policy(path, max_s, done=None, task=None):
    """Run a policy until max_s, or until done(status) returns true; then stop it. Returns the run's summary."""
    r = policy_start(path, max_s, task)
    if not r.get("ok"):
        return dict(ok=False, why="; ".join(r.get("errors", [])))
    while True:
        st = status()
        if not st["policy"]["running"]:
            return dict(ok=True, **(st["policy"]["last"] or {}))
        if done is not None and done(st):
            return policy_stop("done")
        time.sleep(0.1)
