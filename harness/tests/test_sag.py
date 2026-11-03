"""Gravity sag, in the simulator with its servos sagging like the real ones."""
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


def test_holding_still_trims_away_what_the_sag_model_misses(core):
    core.ctl.sag = {s: gravity.SagModel(np.zeros(5)) for s in core.ctl.sag}  # a model that compensates nothing
    core.ctl.sag_learning = True
    learn = gravity.SagModel.learn
    gravity.SagModel.learn = lambda *a: None  # and doesn't learn: only the trim can close the gap
    try:
        run(core, "home")
        end = core.ctl.now() + 1.0
        while core.ctl.now() < end:
            time.sleep(0.005)
        assert tip_error_mm(core, "left") < 1.5  # 5-20 mm without it (above)
        assert 0 < np.max(np.abs(core.ctl.trim["left"])) <= 3.0
    finally:
        gravity.SagModel.learn = learn


def test_the_sag_is_learned_while_holding_and_compensated(core):
    run(core, "home")
    for s in ("left", "right"):
        assert tip_error_mm(core, s) < 2.0
        c = core.ctl.sag[s].c
        assert c[1] == pytest.approx(1200, rel=0.15) and c[2] == pytest.approx(1300, rel=0.15)
    # the model learned at home carries over to a pose it never held
    p = np.append(core.scene.middle(), 0.12)
    assert run(core, "move", left={"xy": list(p[:2] * 100), "z": 12})["outcome"] == "done"
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


def grip_brush(core):
    """The left gripper closes on the brush from above: open over it, down to 1 cm, close."""
    d = core.perception.detect()[0]
    for kf in ({"xy": d["center_cm"], "z": 6, "yaw": d["yaw"] + 90, "pitch": 90, "width": 60}, {"z": 1}, {"width": 0}):
        assert run(core, "move", left=kf)["outcome"] == "done"
    assert list(core.world.objects[0].held) == ["left"]


def test_a_held_object_may_not_be_swung_into_the_table(core):
    run(core, "home")
    tip = core.arms_state()["left"]["tip_cm"]  # robotd is told it holds a brush across its fingertips
    assert core.handle(dict(cmd="hold", arm="left", name="brush", size_cm=[20, 3, 2.5], center_cm=tip[:2],
                            z_cm=tip[2]))["ok"]
    h = core.holding["left"]
    T = core.skills.tool_T("left")
    assert np.min(held_corners(T, h)[:, 2]) > 0.01
    tip = T[:3, 3]
    with pytest.raises(Exception, match="would hit the table"):
        core.planner.plan(core.ctl.snapshot()["goal"],
                          [Phase("low", {"left": Goal(p=[tip[0], tip[1], 0.012], pitch="steep")})])


def test_an_arm_that_stops_short_is_reported_blocked(core):
    run(core, "home")
    core.world.solid_in_the_way = lambda side, q0, q1: side == "left"
    r = run(core, "move", left={"xy": [-10, 30], "z": 8})
    assert r["outcome"] == "stopped" and r["fail"] == "blocked", r


def test_no_sag_is_learned_while_squeezing_something(core):
    run(core, "home")
    grip_brush(core)
    n = core.ctl.sag["left"].samples
    t0 = core.ctl.now()
    while core.ctl.now() - t0 < 1.5:  # holds the brush still
        time.sleep(0.01)
    assert core.ctl.sag["left"].samples == n
