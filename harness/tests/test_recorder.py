import json
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

from harness.recorder import FPS, STILL, Recorder, final
from harness.robotd import Core


def run(core, name, **args):
    assert core.start_skill(name, args)["ok"]
    while core.busy():
        time.sleep(0.005)
    return core.skill["result"]


@pytest.fixture
def core(tmp_path):
    c = Core(sim_mode=True, speed=0, quiet=True, dataset="test_pass", datasets=str(tmp_path))
    yield c
    c.close()


def episode(core, target):
    r = core.handle(dict(cmd="episode_begin", task="Pass the brush to the other arm", giver="left", taker="right",
                         target_cm=target))
    assert r["ok"], r
    for name, args in [("pick", dict(arm="left", target="brush")), ("handover", dict(giver="left", taker="right")),
                       ("place", dict(arm="right", xy=target)), ("home", {})]:
        assert run(core, name, **args)["outcome"] == "done"
    end = core.handle(dict(cmd="episode_end"))
    assert end["ok"], end
    return r["episode"], end


def test_one_episode_loads_in_lerobot(core):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ep, end = episode(core, [12, 24])
    assert ep == 0 and 0 < end["frames"] < end["duration_s"] * 30
    ds = LeRobotDataset("local/test_pass", root=core.recorder.root)
    assert ds.num_episodes == 1 and ds.num_frames == end["frames"]
    item = ds[ds.num_frames // 2]
    assert tuple(item["observation.state"].shape) == (12,) and tuple(item["action"].shape) == (12,)
    assert tuple(item["observation.images.left_overhead"].shape) == (3, 480, 640)
    assert item["task"] == "Pass the brush to the other arm"


def written(rows):
    """How many frames the recorder writes for ticks of (left arm's reading, left arm's goal, right arm's reading), each
    one 6 values of which the last is the gripper in mm; the stroke is 100 mm, so mm are % here."""
    core = SimpleNamespace(ctl=SimpleNamespace(listeners=[]), world=None, scene=SimpleNamespace(stroke_mm=100.0))
    rec = Recorder(core)
    rec.ep, rec.queue = dict(recording=True, index=0, last_written=None, skipped=0, frame_t=[]), queue.Queue()
    rec.t0, rec.next_t = 0.0, 0.0
    for k, (state, goal, right) in enumerate(rows):
        rec._tick(k / FPS, dict(left=np.array(state, dtype=float), right=np.array(right, dtype=float)),
                  dict(left=np.array(goal, dtype=float), right=np.zeros(6)))
    assert rec.queue.qsize() + rec.ep["skipped"] == len(rows)
    return rec.queue.qsize()


def joints(j=0, v=0.0):
    return [v if i == j else 10.0 for i in range(6)]


def test_idle_frames_are_not_written():
    still = joints()
    rest = [0.0] * 6
    assert written([(still, still, rest)] * 60) == 1, "the first frame, and nothing while nothing moves"
    tick = 0.0879  # one encoder tick of a real arm joint
    jitter = [(joints(1, 10 + tick * (k % 3 - 1)), still, rest) for k in range(60)]
    assert written(jitter) == 1, "encoders flickering a tick either way aren't motion"
    assert written([(joints(1, 10 + k * 1.0), still, rest) for k in range(5)] + [(joints(1, 14.0), still, rest)] * 30) == 5, \
        "a joint settling while its goal has stopped is kept"
    assert written([(still, joints(1, 10 + k * 1.0), rest) for k in range(5)]) == 5, "a goal moving ahead of the joint"
    assert written([(joints(5, 40 - k * 2.0), joints(5, 40 - k * 2.0), rest) for k in range(10)]) == 10, "the gripper"
    assert written([(still, still, joints(0, k * 1.0)) for k in range(5)]) == 5, "the other arm"


def test_slow_motion_adds_up_and_gets_written():
    n = written([(joints(2, 10 + k * 0.1), joints(2, 10 + k * 0.1), [0.0] * 6) for k in range(31)])
    assert n == pytest.approx(3.0 / STILL, abs=2), "3 deg at a third of the threshold a tick still gets its frames"


def test_an_episode_has_no_idle_frames(core):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _, end = episode(core, [12, 24])
    label = json.load(open(end["sidecar"]))
    assert label["robot"]["idle_frames_skipped"] > 0 and label["frames"] == end["frames"]
    data = LeRobotDataset("local/test_pass", root=core.recorder.root).hf_dataset
    state = np.stack([np.asarray(v) for v in data["observation.state"]])
    action = np.stack([np.asarray(v) for v in data["action"]])
    moved = np.maximum(np.abs(np.diff(state, axis=0)).max(axis=1), np.abs(np.diff(action, axis=0)).max(axis=1))
    assert moved.min() >= STILL - 1e-4, "every frame moved at least the threshold from the one before it"


def test_label_file_and_final_decision(core):
    ep, end = episode(core, [12, 24])
    label = json.load(open(end["sidecar"]))
    robot = label["robot"]
    assert [s["op"] for s in robot["steps"]] == ["pick", "handover", "place", "home"]
    assert [e["kind"] for e in robot["events"]] == ["grasp", "pass", "release"]
    assert robot["check"]["passed"] and robot["object"]["name"] == "brush"
    assert robot["steps"][0]["t"][0] >= 0 and robot["steps"][-1]["t"][1] <= end["duration_s"] + 0.1
    r = core.handle(dict(cmd="episode_label", episode=ep, section="judge",
                         data=dict(model="muse-spark", success=True, confidence=0.9, failure=None, reason="ok")))
    assert r["ok"] and r["final"]["train"]
    r = core.handle(dict(cmd="episode_label", episode=ep, section="vision", data=dict(model="sam-3.1", objects=[])))
    assert r["ok"] and json.load(open(end["sidecar"]))["vision"]["model"] == "sam-3.1"


def test_episodes_append(core):
    episode(core, [12, 24])
    # the brush is on the right now: pass it back
    r = core.handle(dict(cmd="episode_begin", task="Pass the brush to the other arm", giver="right", taker="left",
                         target_cm=[-12, 24]))
    assert r["ok"] and r["episode"] == 1
    for name, args in [("pick", dict(arm="right", target="brush")), ("handover", dict(giver="right", taker="left")),
                       ("place", dict(arm="left", xy=[-12, 24]))]:
        res = run(core, name, **args)
        assert res["outcome"] == "done", res["why"]
    assert core.handle(dict(cmd="episode_end"))["ok"]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    assert LeRobotDataset("local/test_pass", root=core.recorder.root).num_episodes == 2


def test_an_episode_nobody_ends_is_ended_as_invalid(core, monkeypatch):
    """A brain that dies mid-episode doesn't leave robotd recording frames until the disk is full."""
    from harness import recorder
    monkeypatch.setattr(recorder, "MAX_EPISODE_S", 2.0)
    assert core.handle(dict(cmd="episode_begin", task="t"))["ok"]
    end = time.monotonic() + 30
    while core.recorder.active() or core.recorder.ep is not None:
        assert time.monotonic() < end
        time.sleep(0.05)
    label = json.load(open(core.recorder._label_path(0)))
    assert label["robot"]["valid"] is False and "no episode_end" in label["robot"]["note"]
    assert label["frames"] <= 2 * 30 + 2
    assert core.handle(dict(cmd="episode_begin", task="t"))["ok"]  # and the next one can start


def test_commands_in_the_wrong_order(core):
    assert not core.handle(dict(cmd="episode_end"))["ok"]
    assert core.handle(dict(cmd="episode_begin", task="t"))["ok"]
    assert not core.handle(dict(cmd="episode_begin", task="t"))["ok"]
    assert not core.handle(dict(cmd="episode_label", episode=99, section="judge", data={}))["ok"]


def test_final_rules():
    good = dict(robot=dict(valid=True, check=dict(passed=True)), judge=dict(success=True, confidence=0.9))
    assert final(good)["train"]
    assert not final(dict(good, robot=dict(valid=False, check=dict(passed=True))))["train"]
    disagree = final(dict(good, robot=dict(valid=True, check=dict(passed=False))))
    assert not disagree["train"] and disagree["review"]
    unsure = final(dict(good, judge=dict(success=True, confidence=0.5)))
    assert not unsure["train"] and unsure["review"]
