"""The judge with a scripted model: which frames it picks, what it's told (and not told), how a verdict becomes the
label's judge section, and a whole episode recorded in the simulator, read back from lerobot and judged."""
import json
import time

import numpy as np
import pytest

from brain.llm import Scripted
from judge import frames as fr
from judge.judge import JudgeError, describe, judge, pick_frames

LABEL = {
    "episode": 3, "task": "Pass the brush to the other arm", "fps": 30, "frames": 600,
    "source": {"kind": "agent"},
    "robot": {"giver": "left", "taker": "right", "target_cm": [12, 24], "valid": True, "duration_s": 20.0,
              "steps": [{"op": "pick", "arm": "left", "t": [0.1, 5.0], "outcome": "done"},
                        {"op": "handover", "giver": "left", "taker": "right", "t": [5.0, 9.5], "outcome": "done"}],
              "events": [{"t": 5.0, "kind": "grasp", "arm": "left"},
                         {"t": 9.5, "kind": "pass", "from": "left", "to": "right"}],
              "object": {"name": "brush", "end_cm": [12.4, 23.8], "held_by": None},
              "check": {"passed": True, "distance_cm": 0.4}},
    "agents": {"success": True, "summary": "passed the brush"},
}


def blank(n=3):
    return [(i, "frame", np.zeros((480, 640, 3), np.uint8)) for i in range(n)]


def test_frames_are_the_start_the_handover_and_the_end():
    assert pick_frames(LABEL) == [(0, "start"), (285, "just after the handover"), (599, "end")]
    no_pass = dict(LABEL, robot=dict(LABEL["robot"], events=[]))
    assert pick_frames(no_pass) == [(0, "start"), (300, "middle"), (599, "end")]
    assert pick_frames(dict(LABEL, frames=1)) == [(0, "start")]
    assert pick_frames(dict(LABEL, frames=0)) == []


def test_the_judge_isnt_told_the_answer():
    text = describe(LABEL)
    assert "Pass the brush" in text and "handover" in text and "[\n  12,\n  24\n ]" in text
    for leak in ("passed", "distance_cm", "end_cm", "passed the brush"):
        assert leak not in text
    with_vision = dict(LABEL, vision={"model": "sam-3.1", "objects": [{"name": "brush", "visible": 0.9}],
                                      "events": [{"t": 9.4, "kind": "pass", "from": "left", "to": "right"}]})
    assert "SAM 3.1" in describe(with_vision) and '"visible": 0.9' in describe(with_vision)


def test_a_verdict_becomes_the_judge_section():
    llm = Scripted([[("verdict", dict(success=True, confidence=1.4, failure="dropped", failure_arm="left",
                                      reason="The brush lies inside the ring, no gripper on it."))]])
    j, cost = judge(LABEL, blank(), llm, log=lambda s: None)
    assert j["success"] and j["confidence"] == 1.0 and j["failure"] is None  # success drops the failure
    assert j["frames"] == [0, 1, 2] and j["reason"].startswith("The brush") and cost == 0
    sent = llm.seen[0][0]["content"]
    assert sum(p["type"] == "image_url" for p in sent) == 3


def test_a_failure_keeps_its_kind_and_unknown_kinds_are_dropped():
    llm = Scripted([[("verdict", dict(success=False, confidence=0.8, failure="dropped", failure_t=9.2,
                                      failure_arm="right", reason="It fell."))]])
    j, _ = judge(LABEL, blank(), llm, log=lambda s: None)
    assert j["failure"] == {"kind": "dropped", "t": 9.2, "arm": "right"}
    llm = Scripted([[("verdict", dict(success=False, confidence=0.5, failure="exploded", reason="?"))]])
    assert judge(LABEL, blank(), llm, log=lambda s: None)[0]["failure"] is None


def test_it_asks_again_once_then_gives_up():
    llm = Scripted(["It looks fine to me.", [("verdict", dict(success=True, confidence=0.9, failure="none",
                                                                reason="ok"))]])
    assert judge(LABEL, blank(), llm, log=lambda s: None)[0]["success"]
    second = llm.seen[1]
    assert second[-1]["content"] == "Call verdict with your decision."
    with pytest.raises(JudgeError):
        judge(LABEL, blank(), Scripted(["no", [("look", {})]]), log=lambda s: None)


def test_it_wont_send_verdicts_to_robotd_on_another_dataset(monkeypatch):
    from harness import robot
    from judge.__main__ import check_dataset

    monkeypatch.setattr(robot, "status", lambda since=0: {"ok": True, "dataset": "understudy"})
    check_dataset("understudy")
    with pytest.raises(SystemExit, match="restart it with --dataset sim_pass"):
        check_dataset("sim_pass")
    monkeypatch.setattr(robot, "status", lambda since=0: {"ok": True})  # a robotd that doesn't say: trusted
    check_dataset("sim_pass")


# ---- a real episode, through the simulator and lerobot ----------------------------------------------------------
def run(core, name, **args):
    assert core.start_skill(name, args)["ok"]
    while core.busy():
        time.sleep(0.005)
    return core.skill["result"]


@pytest.fixture
def core(tmp_path):
    pytest.importorskip("lerobot")
    from harness.robotd import Core

    c = Core(sim_mode=True, speed=0, quiet=True, dataset="judge_test", datasets=str(tmp_path))
    yield c
    c.close()


def test_a_recorded_episode_is_read_back_judged_and_trains(core):
    from harness.scene import Scene

    target = [12, 24]
    r = core.handle(dict(cmd="episode_begin", task="Pass the brush to the other arm", giver="left", taker="right",
                         target_cm=target))
    assert r["ok"], r
    for name, args in [("pick", dict(arm="left", target="brush")), ("handover", dict(giver="left", taker="right")),
                       ("place", dict(arm="right", xy=target)), ("home", {})]:
        assert run(core, name, **args)["outcome"] == "done"
    end = core.handle(dict(cmd="episode_end"))
    assert end["ok"], end
    label = json.load(open(end["sidecar"]))

    picks = pick_frames(label)
    assert len(picks) == 3 and picks[1][1] == "just after the handover"
    raw = fr.read(core.recorder.root, label["episode"], [i for i, _ in picks])
    assert [img.shape for img in raw] == [(480, 640, 3)] * 3 and raw[0].dtype == np.uint8
    assert np.abs(raw[0].astype(int) - raw[-1].astype(int)).mean() > 0.5  # the brush moved
    scene = Scene()
    images = [(i, why, fr.annotate(img, scene, target, why)) for (i, why), img in zip(picks, raw)]
    assert not np.array_equal(images[0][2], raw[0])  # grid and target drawn on

    llm = Scripted([[("verdict", dict(success=True, confidence=0.92, failure="none",
                                      reason="The brush lies in the ring and no gripper holds it."))]])
    verdict, _ = judge(label, images, llm, log=lambda s: None)
    stored = core.handle(dict(cmd="episode_label", episode=label["episode"], section="judge", data=verdict))
    assert stored["ok"] and stored["final"]["train"] and stored["final"]["why"] == "valid; judge and check agree"
