"""The loop against the simulator, with a scripted model: each action reaches robotd, the result and fresh images come
back in the next turn, and the run ends on done."""
import itertools
import threading

import numpy as np
import pytest

from brain import loop, see
from brain.llm import Scripted
from harness import robot
from harness.robotd import Core, serve

PORTS = itertools.count(5938)


@pytest.fixture
def sim(tmp_path, monkeypatch):
    core = Core(sim_mode=True, speed=0, quiet=True, dataset="loop", datasets=str(tmp_path))
    stop = threading.Event()
    port = next(PORTS)
    threading.Thread(target=serve, args=(core, port, stop), daemon=True).start()
    monkeypatch.setattr(robot, "PORT", port)
    monkeypatch.setattr(see, "_real", False)
    monkeypatch.setattr(see, "fit_held", lambda *a, **k: None)  # no segmentation service in the tests
    monkeypatch.setattr(see, "held_at", lambda *a, **k: None)
    yield core
    stop.set()
    core.close()


def text_of(message):
    c = message["content"]
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if p.get("type") == "text")


def test_each_action_runs_and_muse_sees_its_result(sim, monkeypatch):
    monkeypatch.setattr(see, "find", lambda img, names, arms, held=(): [dict(
        name="ball", center_cm=[-14.0, 30.0], size_cm=[6.5, 6.5, 6.5], z_cm=3.3, shape="round", px=[100, 100])])
    llm = Scripted([
        {"see": "", "plan": "", "do": "find", "things": ["ball"]},
        {"arm": "left", "do": "open", "mm": 80},
        {"arm": "left", "do": "move", "forward": 3, "up": -5},
        {"arm": "left", "do": "move_to", "x": -14, "y": 28, "z": 10, "pitch": 30},
        {"arm": "left", "do": "pitch", "deg": 20},
        {"arm": "left", "do": "roll", "deg": 15},
        {"arm": "left", "do": "close", "force": 20},
        {"arm": "left", "do": "move", "forward": 30},
        {"arm": "left", "do": "home"},
        {"do": "done", "success": True, "summary": "ok"},
    ])
    out = loop.run("test the actions", llm, max_turns=20)
    assert out["success"] and out["summary"] == "ok"
    steps = [r for r in out["record"] if "step" in r]
    assert [s["step"]["do"] for s in steps] == ["find", "open", "move", "move_to", "pitch", "roll", "close", "move",
                                                "home"]
    for s in steps:
        assert s["result"]["outcome"] == ("refused" if s["step"].get("forward") == 30 else "done"), s
    assert "at most 10 cm" in steps[7]["result"]["why"]
    assert "jaws_stopped_mm" in steps[6]["result"]
    tip = robot.status()["arms"]["left"]["tip_cm"]
    assert abs(tip[2] - 28) < 2  # home: the ready pose
    seen = [text_of(m[-1]) for m in llm.seen]
    assert "Task: test the actions" in seen[0] and "Nothing found yet" in seen[0]
    assert "- ball: centre [-14.0, 30.0]" in seen[1] and "from_grip left" in seen[1]
    assert '"do": "move"' in seen[3] and '"outcome": "done"' in seen[3]
    assert "squeezing" in seen[7] or "closed on nothing" in seen[7]
    # only the newest message carries images
    for m in llm.seen[-1][:-1]:
        assert not any(p.get("type") == "image_url" for p in m["content"]) if isinstance(m["content"], list) else True


def test_a_move_keeps_the_pitch_it_was_sent(sim):
    llm = Scripted([
        {"arm": "right", "do": "move_to", "x": 14, "y": 25, "z": 12, "pitch": 40},
        {"arm": "right", "do": "move", "right": 2},
        {"do": "done", "success": True, "summary": ""},
    ])
    loop.run("pitch", llm, max_turns=5)
    assert abs(robot.status()["arms"]["right"]["pitch"] - 40) < 4


def test_close_refuses_while_the_thing_is_at_the_fingertips(sim, monkeypatch):
    llm = Scripted([{"do": "done", "success": True, "summary": ""}])
    r = loop.Run(llm)
    arm = robot.status()["arms"]["left"]
    _, g, (f, _, _) = see.axes("left", arm["joints"])
    at = lambda fwd: [dict(name="tube", center_cm=[float(g[0] + f[0] * fwd), float(g[1] + f[1] * fwd)],
                           size_cm=[8, 8, 2 * float(g[2])], z_cm=float(g[2]), shape="upright", px=[0, 0])]
    r.things = at(4.0)
    assert "4.0 cm in front of the grip point" in r.not_in("left", arm)
    r.things = at(0.5)
    assert r.not_in("left", arm) is None


def test_a_held_tall_thing_stays_upright(sim):
    r = loop.Run(Scripted([]))
    arm = robot.status()["arms"]["right"]
    r.closed["right"] = 60.0
    r.held["right"] = dict(name="tube", length=24.0, below=12.0, roll0=r.roll("right", arm))
    seen = {"arms": robot.status()["arms"]}
    out = r.act({"arm": "right", "do": "roll", "deg": 15}, seen)
    assert out["outcome"] == "refused" and "stays straight up" in out["why"]
    assert "top end" in r.state(seen["arms"])


def test_an_arm_may_not_go_into_what_the_other_holds(sim):
    r = loop.Run(Scripted([]))
    arms = robot.status()["arms"]
    r.closed["right"] = 60.0
    r.held["right"] = dict(name="tube", length=24.0, below=12.0, roll0=r.roll("right", arms["right"]))
    b, t = r.ends("right", arms["right"])
    assert "go into the tube" in r.clash("left", arms, b + np.array([-4.0, 0.0, 2.0]))
    assert r.clash("left", arms, t + np.array([-2.0, 0.0, 8.0])) is None  # well over its top: fine
    _, g2, _ = see.axes("right", arms["right"]["joints"])
    assert "from the right gripper" in r.clash("left", arms, g2 + np.array([-6.0, 0.0, 5.0]))


def test_the_tube_is_come_at_from_above(sim):
    r = loop.Run(Scripted([]))
    arms = robot.status()["arms"]
    r.closed["right"] = 60.0
    r.held["right"] = dict(name="tube", length=24.0, below=12.0, roll0=r.roll("right", arms["right"]))
    b, t = r.ends("right", arms["right"])
    from_side = t + np.array([-8.0, 0.0, 2.0])  # beside its top, a little above it: sliding in from there is refused
    assert "only straight up or down" in r.clash("left", arms, from_side) or r.clash("left", arms, from_side)


def test_the_drop_over_the_opening_is_allowed_and_the_tube_waits(sim):
    r = loop.Run(Scripted([]))
    arms = robot.status()["arms"]
    r.closed["right"] = 60.0
    r.held["right"] = dict(name="tube", length=24.0, below=12.0, roll0=r.roll("right", arms["right"]), at_least=True)
    b, t = r.ends("right", arms["right"])
    # the left already right over the opening, lowering straight down: the drop
    real_axes = see.axes
    over = t + np.array([0.5, 0.0, 6.0])
    see_axes = lambda side, joints: (over + [0, 3, 0], over, (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]),
                                                              np.array([0, 0, 1.0]))) if side == "left" \
        else real_axes(side, joints)
    loop.see.axes = see_axes
    try:
        assert r.clash("left", arms, over + np.array([0.0, 0.0, -2.0])) is None
    finally:
        loop.see.axes = real_axes


def test_a_held_tube_sits_where_it_was_measured(sim):
    r = loop.Run(Scripted([]))
    arms = robot.status()["arms"]
    r.closed["right"] = 60.0
    r.held["right"] = dict(name="tube", length=24.0, below=12.0, roll0=r.roll("right", arms["right"]))
    _, t0 = r.ends("right", arms["right"])
    r.held["right"]["ahead"] = 4.0  # a wide thing stops short in the jaws: its axis in front of the grip point
    _, t1 = r.ends("right", arms["right"])
    _, _, (f, _, _) = see.axes("right", arms["right"]["joints"])
    assert np.allclose(t1 - t0, 4.0 * f)


def test_open_over_the_opening_waits_until_it_is_right_over_it(sim, monkeypatch):
    r = loop.Run(Scripted([]))
    arms = robot.status()["arms"]
    r.closed["right"], r.closed["left"] = 60.0, 62.0
    r.held["right"] = dict(name="tube", length=24.0, below=12.0, roll0=r.roll("right", arms["right"]))
    t = r.ends("right", arms["right"])[1]
    seen = {"arms": dict(arms, left=dict(arms["left"], width_mm=62.0))}
    real_axes = see.axes
    for off, refused in ((4.0, True), (1.0, False)):
        g = t + np.array([off, 0.0, 6.0])  # the left's grip point over the tube's top, off by off cm to the right
        monkeypatch.setattr(see, "axes", lambda side, joints, g=g: (
            g + [0, 3, 0], g, (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])))
            if side == "left" else real_axes(side, joints))
        why = r.off_opening("left", seen)
        assert (why is not None) == refused, why
        if refused:
            assert "right -4.0" in why
    g = t + np.array([0.0, 0.0, 12.0])  # right over it but its bottom 8.5 cm above the rim: let go lower
    monkeypatch.setattr(see, "axes", lambda side, joints: (
        g + [0, 3, 0], g, (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])))
        if side == "left" else real_axes(side, joints))
    assert "move down 4.5" in r.off_opening("left", seen)
