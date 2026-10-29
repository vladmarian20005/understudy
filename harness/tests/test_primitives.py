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


def test_pick_and_place(core):
    r = run(core, "pick", arm="left", target="brush")
    assert r["outcome"] == "done" and r["width_mm"] > 20
    assert r["arms"]["left"]["holding"] and brush(core).held
    r = run(core, "place", arm="left", xy=[-15, 20])
    assert r["outcome"] == "done", r["why"]
    assert not brush(core).held and not core.holding["left"]
    assert np.linalg.norm(brush(core).pos[:2] * 100 - [-15, 20]) < 4


def test_ping_pong_pass(core):
    """One episode: the left arm picks the brush, passes it to the right in the air, the right puts it down."""
    assert run(core, "pick", arm="left", target="brush")["outcome"] == "done"
    r = run(core, "handover", giver="left", taker="right")
    assert r["outcome"] == "done", r["why"]
    assert list(brush(core).held) == ["right"] and core.holding["right"] and not core.holding["left"]
    assert brush(core).pos[2] > 0.04  # it never touched the table
    r = run(core, "place", arm="right", xy=[12, 24])
    assert r["outcome"] == "done", r["why"]
    assert np.linalg.norm(brush(core).pos[:2] * 100 - [12, 24]) < 5
    assert run(core, "home")["outcome"] == "done"


def test_a_set_down_near_the_base_picks_a_pose_it_can_lower_from(core):
    """Close in front of a base the first pose found held the brush at a shallow pitch that can't go lower, and the
    touch-down's steeper fallback swung the brush's far end under the table (2 of 400 sim episodes)."""
    run(core, "home")
    core.world.place("brush", [0.117, 0.22], yaw=-18)
    for name, args in [("pick", dict(arm="right", target="brush")), ("handover", dict(giver="right", taker="left")),
                       ("place", dict(arm="left", xy=[-13.1, 20.0]))]:
        r = run(core, name, **args)
        assert r["outcome"] == "done", (name, r["why"])
    assert np.linalg.norm(brush(core).pos[:2] * 100 - [-13.1, 20.0]) < 3


def test_table_handover(core):
    run(core, "pick", arm="left", target="brush")
    r = run(core, "handover", giver="left", taker="right", mode="table")
    assert r["outcome"] == "done", r["why"]
    assert list(brush(core).held) == ["right"]


def test_miss(core):
    r = run(core, "pick", arm="left", target="brush", offset=[0, 5])
    assert r["outcome"] == "failed" and r["fail"] == "miss"
    assert not brush(core).held


def test_refusals(core):
    r = run(core, "reach", arm="left", xy=[40, 44])
    assert (r["outcome"], r["fail"]) == ("refused", "unreachable")
    r = run(core, "grasp", arm="left")
    assert (r["outcome"], r["fail"]) == ("refused", "check")
    r = run(core, "pick", arm="left", target="#9")
    assert (r["outcome"], r["fail"]) == ("refused", "not_seen")
    r = run(core, "reach", arm="left", colour="red")
    assert r["outcome"] == "refused" and "bad arguments" in r["why"]
    r = run(core, "fly", arm="left")
    assert r["outcome"] == "refused"
    r = run(core, "release", arm="right", set_down=True)
    assert r["outcome"] == "done"  # nothing held: it just opens


def test_result_shape(core):
    r = run(core, "reach", arm="left", target="brush")
    assert set(r) >= {"op", "arm", "outcome", "fail", "why", "t", "arms", "events"}
    assert set(r["arms"]["left"]) >= {"tip_cm", "yaw", "width_mm", "holding", "mode"}
    assert r["t"][1] >= r["t"][0]


def test_stop_and_busy():
    core = Core(sim_mode=True, speed=1.0, quiet=True)
    try:
        assert core.start_skill("pick", {"arm": "left", "target": "brush"})["ok"]
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
        r = robot.pick("left", "brush")
        assert r["outcome"] == "done", r["why"]
        st = robot.status()
        assert st["arms"]["left"]["holding"] and not st["busy"]
        assert robot.call("nope")["ok"] is False
    finally:
        stop.set()
