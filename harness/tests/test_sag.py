"""Gravity sag and touch-down, in the simulator with its servos sagging like the real ones."""
import time

import numpy as np
import pytest

from harness import gravity
from harness.motion import Goal, Phase, held_corners
from harness.robotd import Core


@pytest.fixture
def core():
    c = Core(sim_mode=True, speed=0, quiet=True, dataset=None)
    yield c
    c.close()


def run(core, name, **args):
    assert core.start_skill(name, args)["ok"]
    while core.busy():
        time.sleep(0.005)
    return core.skill["result"]


def tip_error_mm(core, side):
    snap = core.ctl.snapshot()
    arm = core.scene.arms[side]
    return float(np.linalg.norm(arm.tool(snap["goal"][side])[0] - arm.tool(snap["q"][side])[0]) * 1000)


def test_the_simulated_arms_sag_like_the_real_ones(core):
    core.ctl.sag = {s: gravity.SagModel(np.zeros(5)) for s in core.ctl.sag}  # no compensation
    core.ctl._learn_sag = lambda: None
    run(core, "home")
    assert 5 < tip_error_mm(core, "left") < 20  # the real arms: 1-1.5 cm when extended


def test_the_sag_is_learned_while_holding_and_compensated(core):
    run(core, "home")
    for s in ("left", "right"):
        assert tip_error_mm(core, s) < 2.0
        c = core.ctl.sag[s].c
        assert c[1] == pytest.approx(1200, rel=0.15) and c[2] == pytest.approx(1300, rel=0.15)
    # the model learned at home carries over to a pose it never held
    p = np.append(core.scene.middle(), 0.12)
    assert run(core, "move_to", arm="left", xy=list(p[:2] * 100), z=12)["outcome"] == "done"
    assert tip_error_mm(core, "left") < 3.0


def test_the_learned_sag_survives_a_restart(tmp_path):
    from harness.controller import Controller

    path = str(tmp_path / "sag.json")
    a = Core(sim_mode=True, speed=0, quiet=True, dataset=None)
    try:
        run(a, "home")
        a.ctl.sag_file = path
        a.ctl.save_sag()
        learned = a.ctl.sag["left"].c.copy()
    finally:
        a.close()
    ctl = Controller(a.world.io).set_arms(a.scene.arms, path)
    assert np.allclose(ctl.sag["left"].c, np.round(learned, 1))


def test_a_grasp_touches_the_table_first(core):
    run(core, "home")
    r = run(core, "pick", arm="left", target="brush")
    assert r["outcome"] == "done", r["why"]
    assert any("left touched down" in e["text"] for e in core.events_since(0))
    brush = core.world.objects[0]
    believed = (core.skills.tool_T("left") @ core.holding["left"]["rel"])[:3, 3]
    assert np.linalg.norm(believed - brush.pos) < 0.003  # it knows where the brush is in its hand


def test_a_finger_on_top_of_the_object_is_reported(core):
    run(core, "home")
    d = core.perception.detect()[0]
    jaw = np.radians(d["yaw"] + 90)
    off = [2.6 * np.cos(jaw), 2.6 * np.sin(jaw)]  # shifted along the jaws: one finger lands on the brush
    r = run(core, "pick", arm="left", target="brush", offset=off)
    assert r["outcome"] == "failed" and r["fail"] == "finger_on_object", r
    assert not core.world.objects[0].held


def test_setting_down_stops_when_the_object_meets_the_table(core):
    run(core, "home")
    assert run(core, "pick", arm="left", target="brush")["outcome"] == "done"
    r = run(core, "place", arm="left", xy=[-14, 28])
    assert r["outcome"] == "done", r["why"]
    brush = core.world.objects[0]
    assert not brush.held and np.hypot(*(brush.pos[:2] * 100 - [-14, 28])) < 1.5
    assert sum("left touched down" in e["text"] for e in core.events_since(0)) == 2  # grasp, then set-down


def test_a_held_object_may_not_be_swung_into_the_table(core):
    run(core, "home")
    assert run(core, "pick", arm="left", target="brush")["outcome"] == "done"
    h = core.holding["left"]
    T = core.skills.tool_T("left")
    assert np.min(held_corners(T, h)[:, 2]) > 0.01
    tip = T[:3, 3]
    with pytest.raises(Exception, match="would hit the table"):
        core.planner.plan(core.ctl.snapshot()["goal"], [Phase("low", {"left": Goal(p=[tip[0], tip[1], 0.012])})])


def test_an_arm_that_stops_short_is_reported_blocked(core):
    run(core, "home")
    core.world.solid_in_the_way = lambda side, q0, q1: side == "left"
    r = run(core, "move_to", arm="left", xy=[-10, 30], z=8)
    assert r["outcome"] == "stopped" and r["fail"] == "blocked", r
