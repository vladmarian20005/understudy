"""The policy runner and the controller's streaming mode, with a stand-in policy (no trained model needed)."""
import time

import numpy as np
import pytest
import torch

from harness import controller, policy
from harness.motion import TICK
from harness.recorder import vector
from harness.robotd import Core


@pytest.fixture
def core():
    c = Core(sim_mode=True, speed=0, quiet=True, dataset=None)
    yield c
    c.close()


def fake_policy(monkeypatch, core, action_fn):
    """Make policy_start load a stand-in whose action is action_fn(state vector)."""
    monkeypatch.setattr(policy, "resolve", lambda path: path)
    monkeypatch.setattr(policy, "load", lambda path: (_Fake(), None, None, torch.device("cpu")))
    import lerobot.utils.control_utils as cu

    def predict(obs, pol, device, pre, post, use_amp, task=None, robot_type=None):
        for k, v in obs.items():
            if "images" in k:
                assert v.shape == (480, 640, 3) and v.dtype == np.uint8
        return torch.tensor(action_fn(obs["observation.state"]), dtype=torch.float32)

    monkeypatch.setattr(cu, "predict_action", predict)


class _Fake:
    def reset(self):
        pass


def wait(cond, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_streaming_is_rate_limited(core):
    q0 = core.ctl.snapshot()["goal"]["left"].copy()
    target = q0.copy()
    target[0] += 30
    assert core.ctl.stream({"left": target})
    t0 = core.ctl.now()
    assert wait(lambda: core.ctl.now() - t0 >= 5 * TICK)
    moved = core.ctl.snapshot()["goal"]["left"][0] - q0[0]
    ticks = round((core.ctl.now() - t0) / TICK)
    assert 0 < moved <= controller.STREAM_DEG_S * TICK * (ticks + 1) + 1e-6
    assert wait(lambda: abs(core.ctl.snapshot()["q"]["left"][0] - target[0]) < 0.5)
    core.ctl.stop("test")
    assert core.ctl.target == {}


def test_a_policy_holding_still_runs_until_its_time_is_up(core, monkeypatch):
    fake_policy(monkeypatch, core, lambda state: state)  # "stay where you are"
    r = core.handle(dict(cmd="policy_start", path="fake", max_s=2.0))
    assert r["ok"], r
    assert not core.start_skill("home", {})["ok"]  # skills wait while a policy drives
    assert wait(lambda: not core.policy.active())
    last = core.handle(dict(cmd="status"))["policy"]["last"]
    assert "2 s are up" in last["why"]
    assert last["steps"] == pytest.approx(2.0 * 30, abs=2)  # 30 fps in simulated time


def test_the_policy_moves_the_arms(core, monkeypatch):
    start = vector(core.ctl.snapshot()["q"], core.scene.stroke_mm)
    goal = start.copy()
    goal[0] += 15  # left shoulder pan
    goal[11] = 80  # right gripper opens to 80 %
    fake_policy(monkeypatch, core, lambda state: goal)
    assert core.handle(dict(cmd="policy_start", path="fake", max_s=3.0))["ok"]
    assert wait(lambda: not core.policy.active())
    now = vector(core.ctl.snapshot()["q"], core.scene.stroke_mm)
    assert now[0] == pytest.approx(goal[0], abs=1.0)
    assert now[11] == pytest.approx(80, abs=2.0)


def test_an_unsafe_pose_stops_the_policy(core, monkeypatch):
    arm = core.scene.arms["left"]
    q = core.ctl.snapshot()["q"]["left"]
    dive_q = None
    for d in range(5, 90, 5):  # shoulder lift forward until the fingertips are well under the table
        cand = q.copy()
        cand[1] += d
        if arm.within_limits(cand) and arm.tool(cand)[0][2] < -0.02:
            dive_q = cand
            break
    assert dive_q is not None

    def dive(state):
        a = state.copy()
        a[:5] = dive_q[:5]
        return a

    fake_policy(monkeypatch, core, dive)
    assert core.handle(dict(cmd="policy_start", path="fake", max_s=5.0))["ok"]
    assert wait(lambda: not core.policy.active())
    why = core.policy.last["why"]
    assert "refused a pose" in why
    assert all(m == "hold" for m in core.ctl.snapshot()["mode"].values())


def test_stop_ends_a_policy(core, monkeypatch):
    fake_policy(monkeypatch, core, lambda state: state)
    assert core.handle(dict(cmd="policy_start", path="fake"))["ok"]
    assert core.handle(dict(cmd="status"))["policy"]["running"]
    core.handle(dict(cmd="stop", reason="test"))
    assert not core.policy.active()
    assert core.start_skill("home", {})["ok"]


def test_check_pose_accepts_home_and_refuses_collisions(core):
    snap = core.ctl.snapshot()["q"]
    assert policy.check_pose(core.scene, snap) is None
    # both fingertips at the same point in the middle: the arms would collide
    mid = core.scene.middle()
    qs = {}
    for s in ("left", "right"):
        q, _ = core.scene.arms[s].steepest(np.array([mid[0], mid[1], 0.10]), lowest=0)
        qs[s] = np.append(q, 40.0)
    assert "of each other" in policy.check_pose(core.scene, qs)


def test_small_overshoots_are_clamped_and_large_ones_refused(core):
    q = {s: v.copy() for s, v in core.ctl.snapshot()["q"].items()}
    lim = core.scene.arms["left"].limits
    q["left"][3] = lim[3, 1] + 4  # a few degrees past the wrist limit: what a trained policy does
    assert policy.check_pose(core.scene, q) is None
    assert q["left"][3] == pytest.approx(lim[3, 1])
    q["left"][4] = lim[4, 1] + 20
    why = policy.check_pose(core.scene, q)
    assert "left wrist_roll would go to" in why
