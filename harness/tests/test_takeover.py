"""Leader takeover: moving a leader arm takes its follower over, stops whatever robotd was doing, and counts."""
import json
import time

import numpy as np
import pytest

from harness import controller, sim
from harness.robotd import Core


def with_leaders(tmp_path, speed):
    c = Core(sim_mode=True, speed=speed, quiet=True, dataset="takeover", datasets=str(tmp_path))
    q = c.ctl.snapshot()["q"]
    c.leaders = {s: sim.SimLeaderIO(q[s]) for s in q}
    c.ctl.set_leaders(c.leaders)
    return c


@pytest.fixture
def core(tmp_path):
    c = with_leaders(tmp_path, speed=0)
    yield c
    c.close()


@pytest.fixture
def realtime(tmp_path):
    c = with_leaders(tmp_path, speed=1.0)  # so the leader moves in the middle of the motion
    yield c
    c.close()


def wait(cond, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def sim_wait(core, seconds):
    t0 = core.ctl.now()
    assert wait(lambda: core.ctl.now() - t0 >= seconds)


def test_moving_a_leader_takes_over_and_stops_the_skill(realtime):
    core = realtime
    sim_wait(core, 0.1)
    assert core.start_skill("move_to", dict(arm="left", xy=[-10, 30], z=12))["ok"]
    sim_wait(core, 0.2)
    core.leaders["left"].q[0] += 20  # Vlad swings the left leader's base
    assert wait(lambda: not core.busy())
    r = core.skill["result"]
    assert r["outcome"] == "stopped" and r["fail"] == "manual", r
    assert core.handle(dict(cmd="status"))["arms"]["left"]["mode"] == "manual"
    assert core.interventions == 1
    # the follower trails the leader
    target = core.leaders["left"].q
    assert wait(lambda: abs(core.ctl.snapshot()["q"]["left"][0] - target[0]) < 1.0)
    # nothing else moves while Vlad has an arm
    assert core.start_skill("home", {})["ok"]
    assert wait(lambda: not core.busy())
    assert core.skill["result"]["fail"] == "manual"


def test_resting_the_leader_gives_the_arm_back(core):
    sim_wait(core, 0.1)
    core.leaders["left"].q[1] += 15
    assert wait(lambda: core.ctl.mode["left"] == "manual")
    sim_wait(core, controller.RELEASE_S + 0.3)  # the leader rests
    assert core.ctl.mode["left"] == "hold"
    assert any("gave the left arm back" in e["text"] for e in core.events_since(0))
    # it holds where it was left, and robotd can move it again
    assert core.start_skill("home", {})["ok"]
    assert wait(lambda: not core.busy())
    assert core.skill["result"]["outcome"] == "done"


def test_a_takeover_stops_a_policy_and_is_an_intervention_in_the_label(core, monkeypatch):
    import torch

    from harness import policy

    monkeypatch.setattr(policy, "resolve", lambda path: path)
    monkeypatch.setattr(policy, "load", lambda path: (type("P", (), {"reset": lambda self: None})(), None, None,
                                                       torch.device("cpu")))
    import lerobot.utils.control_utils as cu

    monkeypatch.setattr(cu, "predict_action", lambda obs, *a, **k: torch.tensor(obs["observation.state"]))
    assert core.handle(dict(cmd="episode_begin", task="t"))["ok"]
    assert core.handle(dict(cmd="policy_start", path="fake", max_s=20))["ok"]
    sim_wait(core, 0.3)
    core.leaders["right"].q[0] -= 20
    assert wait(lambda: not core.policy.active())
    assert "won't stream" in core.policy.last["why"]
    sim_wait(core, controller.RELEASE_S + 0.3)
    end = core.handle(dict(cmd="episode_end"))
    assert json.load(open(end["sidecar"]))["robot"]["interventions"] == 1


def test_a_leader_that_drifts_while_torque_is_off_doesnt_take_over(core):
    core.ctl.disarm()
    core.leaders["left"].q[0] += 30
    sim_wait(core, 0.2)
    core.ctl.arm()
    sim_wait(core, 0.2)
    assert core.ctl.mode["left"] == "hold" and core.interventions == 0
