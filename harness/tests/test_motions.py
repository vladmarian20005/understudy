import threading
import time

import numpy as np
import pytest

from harness import robot
from harness.robotd import Core, serve


def run(core, name, **args):
    r = core.start_skill(name, args)
    assert r["ok"], r
    while core.busy():
        time.sleep(0.005)
    return core.skill["result"]


@pytest.fixture
def core():
    c = Core(sim_mode=True, speed=0, quiet=True)
    yield c
    c.close()


def brush(core):
    return core.world.objects[0]


def tip(core, side):
    return core.scene.arms[side].tool(core.ctl.snapshot()["q"][side])[0] * 100


def test_move_plays_keyframes_for_both_arms_in_step(core):
    left0 = tip(core, "left")
    r = run(core, "move", left={"rel": [0, 0, -5]}, right={"rel": [0, 0, -5]}, check_only=True)
    assert r["outcome"] == "done" and r["checked"]["phases"] == 1
    assert np.allclose(tip(core, "left"), left0, atol=0.2)  # checked, not moved
    assert abs(r["checked"]["ends"]["left"][2] - (left0[2] - 5)) < 0.5
    r = run(core, "move", left=[{"xy": [-10, 25], "z": 12}, {"rel": [0, 5, 0], "yaw": 0}],
            right={"xy": [10, 25], "z": 8, "width": 20})
    assert r["outcome"] == "done" and r["phases"] == 2, r["why"]
    assert np.allclose(tip(core, "left"), [-10, 30, 12], atol=0.5)
    assert np.allclose(tip(core, "right"), [10, 25, 8], atol=0.5)
    assert abs(r["arms"]["left"]["yaw"] % 180) < 2 or abs(r["arms"]["left"]["yaw"] % 180 - 180) < 2
    assert abs(r["arms"]["right"]["width_mm"] - 20) < 1


def test_move_refuses_what_it_cant_do_without_moving(core):
    before = tip(core, "left")
    for args, kind, words in [(dict(left={"z": -5}), "check", "table"),
                              (dict(left={"xy": [60, 25]}), "unreachable", "reach"),
                              (dict(left={"up": 3}), "check", "unknown keyframe"), (dict(), "check", "keyframes"),
                              (dict(left={"z": 12}, speed="warp"), "check", "speed")]:
        r = run(core, "move", **args)
        assert r["outcome"] == "refused" and r["fail"] == kind and words in r["why"], (args, r["why"])
    assert np.allclose(tip(core, "left"), before, atol=0.2)


def hold(core, side, width):
    """Tell robotd the side's gripper holds a brush between its fingertips, its jaws width mm apart."""
    assert run(core, "move", **{side: {"width": width}})["outcome"] == "done"
    tip = core.arms_state()[side]["tip_cm"]
    r = core.handle(dict(cmd="hold", arm=side, name="brush", size_cm=[20, 3, 2.5], center_cm=tip[:2], z_cm=tip[2]))
    assert r["ok"], r


def test_move_carries_a_held_object(core):
    hold(core, "left", 30)
    before = core.arms_state()["left"]["held"]["center_cm"]
    r = run(core, "move", left={"rel": [5, 0, -5]})
    assert r["outcome"] == "done" and r["width_mm"] > 20, r["why"]
    after = r["arms"]["left"]["held"]["center_cm"]
    assert np.allclose(np.subtract(after, before), [5, 0, -5], atol=0.5)  # it moved with the gripper


def test_jaws_shut_on_what_it_held_is_a_slip(core):
    hold(core, "left", 0)
    r = run(core, "move", left={"rel": [0, 0, -5]})
    assert (r["outcome"], r["fail"]) == ("failed", "slip") and not core.holding["left"]


def test_refusals(core):
    for name, args in [("pick", dict(arm="left", target="brush")), ("fly", {})]:
        r = run(core, name, **args)
        assert (r["outcome"], r["fail"]) == ("refused", "check") and "unknown command" in r["why"]
    r = run(core, "home", side="left")
    assert r["outcome"] == "refused" and "bad arguments" in r["why"]


def test_result_shape(core):
    r = run(core, "move", left={"rel": [0, 0, -5]})
    assert set(r) >= {"op", "outcome", "fail", "why", "t", "arms", "events"}
    assert set(r["arms"]["left"]) >= {"tip_cm", "yaw", "width_mm", "holding", "mode"}
    assert r["t"][1] >= r["t"][0]


def test_stop_and_busy():
    core = Core(sim_mode=True, speed=1.0, quiet=True)
    try:
        assert core.start_skill("move", {"left": {"xy": [-10, 30], "z": 8}, "speed": "slow"})["ok"]
        r = core.start_skill("home", {})
        assert not r["ok"] and r["fail"] == "busy"
        time.sleep(0.5)
        core.handle({"cmd": "stop"})
        while core.busy():
            time.sleep(0.01)
        assert core.skill["result"]["outcome"] == "stopped"
    finally:
        core.close()


def test_client_over_zmq(core, monkeypatch):
    port = 5871
    stop = threading.Event()
    threading.Thread(target=serve, args=(core, port, stop), daemon=True).start()
    monkeypatch.setattr(robot, "PORT", port)
    try:
        seen = robot.look()
        assert seen["objects"][0]["name"] == "brush"
        assert robot.image(seen).shape == (480, 640, 3)
        r = robot.move(left={"rel": [0, 0, -5]})
        assert r["outcome"] == "done", r["why"]
        st = robot.status()
        assert not st["arms"]["left"]["holding"] and not st["busy"]
        assert robot.call("nope")["ok"] is False
    finally:
        stop.set()
