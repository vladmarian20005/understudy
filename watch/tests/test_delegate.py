"""Delegate: which scores the brain can be given, the task and target it gets, and a whole run against the simulator
with a scripted model (and, with lerobot, recorded and labelled with its score)."""
import json
import sys
import threading

import pytest

from brain.llm import Scripted
from harness import robot
from harness.robotd import Core, serve
from watch import delegate

SCORE = {
    "version": 1, "kind": "score", "task": "Pass the paintbrush to the other arm",
    "object": {"name": "paintbrush", "prompt": "paintbrush"},
    "source": {"kind": "ego4d", "video_uid": "abc", "start_s": 10.0, "end_s": 16.0,
               "narration": "#C C transfers the paintbrush from his left hand to his right hand"},
    "frame": "ego", "fps": 30.0, "frames": 180, "size_px": [1280, 960],
    "giver": "left", "taker": "right",
    "handover": {"t": 1.93, "t_end": 2.27, "both_s": 0.34, "at_px": [655, 590], "at_cm": None},
    "target_cm": None,
    "quality": {"pass_seen": True},
}


def test_only_scores_with_a_pass_and_both_sides_are_delegated():
    delegate.check(SCORE)
    with pytest.raises(delegate.ScoreError, match="no pass"):
        delegate.check(dict(SCORE, quality={"pass_seen": False}))
    delegate.check(dict(SCORE, quality={"pass_seen": False}), allow_unseen=True)
    for giver, taker in [(None, "right"), ("left", "left"), ("up", "right")]:
        with pytest.raises(delegate.ScoreError, match="who gives"):
            delegate.check(dict(SCORE, giver=giver, taker=taker))


def test_a_file_that_isnt_a_score_is_refused(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"kind": "label", "version": 1}))
    with pytest.raises(delegate.ScoreError, match="version 1 score"):
        delegate.load(path)


def test_the_target_is_asked_for_else_the_scores_else_in_front_of_the_taker():
    assert delegate.target_for(SCORE) == [12.0, 24.0]
    assert delegate.target_for(dict(SCORE, taker="left")) == [-12.0, 24.0]
    assert delegate.target_for(dict(SCORE, target_cm=[10, 20])) == [10.0, 20.0]
    assert delegate.target_for(dict(SCORE, target_cm=[10, 20]), override=[-5, 30]) == [-5.0, 30.0]


def test_the_task_gives_the_arms_job_and_the_person_as_a_hint():
    task = delegate.task_for(SCORE, [12, 24])
    assert task.startswith("Pick up the brush with the left arm, hand it over in mid-air to the right arm")
    assert "x 12, y 24" in task and "send both arms home" in task
    assert "the paintbrush from their left hand to their right hand" in task and "0.3 s" in task
    older = {k: v for k, v in SCORE.items() if k not in ("object", "handover")}
    older["vision"] = {"objects": [{"name": "pen"}]}  # a score written before "object" existed
    assert "passed the pen from their left hand" in delegate.task_for(older, [12, 24])
    overhead = dict(SCORE, handover=dict(SCORE["handover"], at_cm=[1.2, 26.0]))
    assert "around (1, 26) cm" in delegate.task_for(overhead, [12, 24])


def test_the_label_keeps_where_the_task_came_from_but_not_ego4ds_words():
    src = delegate.source_for(SCORE, "sam_out/scores/c1/score.json", "muse-spark-1.3")
    assert src["kind"] == "agent" and src["score"]["video_uid"] == "abc" and src["score"]["giver"] == "left"
    assert "narration" not in json.dumps(src)


# ---- against the simulator ----------------------------------------------------------------------------------------
@pytest.fixture
def sim(tmp_path, monkeypatch):
    core = Core(sim_mode=True, speed=0, quiet=True, dataset="delegate", datasets=str(tmp_path / "datasets"))
    stop = threading.Event()
    threading.Thread(target=serve, args=(core, 5899, stop), daemon=True).start()
    monkeypatch.setattr(robot, "PORT", 5899)
    monkeypatch.setattr(delegate, "HERE", str(tmp_path))  # its log goes to tmp_path/logs/delegate
    yield core
    stop.set()
    core.close()


def scripted_pass():
    return Scripted([
        [("pick", dict(arm="left", target="brush"))],
        [("handover", dict(giver="left", taker="right"))],
        [("place", dict(arm="right", xy=[12, 24]))],
        [("home", {})],
        [("done", dict(success=True, summary="passed it and put it on the target"))],
    ])


def run_main(monkeypatch, tmp_path, *args):
    path = tmp_path / "score.json"
    path.write_text(json.dumps(SCORE))
    llm = scripted_pass()
    monkeypatch.setattr("brain.llm.backend", lambda name=None: llm)
    monkeypatch.setattr(sys, "argv", ["delegate", str(path), *args])
    delegate.main()
    return llm


def test_a_score_drives_the_arms_through_the_brain(sim, monkeypatch, tmp_path, capsys):
    llm = run_main(monkeypatch, tmp_path)
    first = llm.seen[0][0]["content"][0]["text"]
    assert first.startswith("Task: Pick up the brush with the left arm")  # the brain got the score's task
    brush = robot.detect()["objects"][0]
    assert abs(brush["center_cm"][0] - 12) < 3 and abs(brush["center_cm"][1] - 24) < 3 and not brush["held_by"]
    out = capsys.readouterr().out
    assert "success: passed it" in out
    assert len(list((tmp_path / "logs" / "delegate").glob("*.json"))) == 1


def test_recorded_the_episode_is_labelled_with_its_score_and_target(sim, monkeypatch, tmp_path):
    pytest.importorskip("lerobot")
    run_main(monkeypatch, tmp_path, "--record")
    label = json.load(open(next((tmp_path / "datasets" / "delegate" / "sidecar").glob("*.json"))))
    assert label["source"]["kind"] == "agent" and label["source"]["score"]["video_uid"] == "abc"
    assert label["robot"]["target_cm"] == [12.0, 24.0] and label["robot"]["giver"] == "left"
    assert label["robot"]["check"]["passed"]  # with a target, robotd's own check can run
    assert label["agents"]["success"]
