"""The agent loop against the simulator, with a scripted model: tool calls reach robotd, results and images come
back, and the run ends on done, on a model that stops calling tools, or on too many turns."""
import itertools
import json
import threading

import numpy as np
import pytest

from brain import agent, tools
from brain.llm import Scripted
from harness import robot
from harness.robotd import Core, serve


PORTS = itertools.count(5898)  # one per test: the last test's server can still hold its port while the next binds


@pytest.fixture
def sim(tmp_path, monkeypatch):
    core = Core(sim_mode=True, speed=0, quiet=True, dataset="agent", datasets=str(tmp_path))
    stop = threading.Event()
    port = next(PORTS)
    threading.Thread(target=serve, args=(core, port, stop), daemon=True).start()
    monkeypatch.setattr(robot, "PORT", port)
    yield core
    stop.set()
    core.close()


def images(messages):
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")


def test_a_scripted_pick_runs_through_the_tools(sim):
    x, y = robot.detect()["objects"][0]["center_cm"]
    llm = Scripted([
        [("open", dict(arm="left"))],
        [("move", dict(left=[{"xy": [x, y - 3], "z": 8, "pitch": 60}]))],
        [("nudge", dict(arm="left", forward=2, up=-3))],
        [("grip", dict(arm="left"))],
        [("open", dict(arm="left"))],
        [("home", {})],
        [("done", dict(success=True, summary="went through the motions"))],
    ])
    out = agent.run("pick up the brush", llm, log=lambda s: None)
    assert out["success"] and out["turns"] == 7
    assert [c["name"] for c in out["calls"]] == ["open", "move", "nudge", "grip", "open", "home", "done"]
    assert all('"outcome": "done"' in c["result"] for c in out["calls"][:6])
    assert '"width_mm"' in out["calls"][3]["result"]
    # every motion came back with the positions, the overhead image and the moved arm's wrist camera; home's with
    # the overhead only, and older images and older results' positions were dropped
    last = llm.seen[-1]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert len(tool_msgs) == 6 and [("now: " in m["content"]) for m in tool_msgs] == [False] * 3 + [True] * 3
    cams = [[p["text"] for p in m["content"] if p.get("type") == "text" and p["text"].endswith("camera:")]
            for m in last if m["role"] == "user" and isinstance(m.get("content"), list)]
    assert cams[-2:] == [["overhead camera:", "left_wrist camera:"], ["overhead camera:"]]
    assert images(last) == 3


def test_a_nudge_goes_where_it_says_and_keeps_the_tilt_and_jaws(sim):
    robot.move([{"xy": [-14, 24], "z": 8, "pitch": 60, "yaw": 20}])
    a0 = robot.status()["arms"]["left"]
    f, r, u = tools.nudge_frame("left", a0["joints"])
    assert abs(f @ r) < 1e-6 and abs(u @ f) < 1e-6 and f[1] > 0.5  # forward points away from the base
    text, seen, res = tools.run("nudge", dict(arm="left", forward=2, right=-1, up=-2))
    assert res["outcome"] == "done" and "short" not in text
    a1 = seen["arms"]["left"]
    moved = np.array(a1["tip_cm"]) - np.array(a0["tip_cm"])
    assert np.linalg.norm(moved - (2 * f - r - 2 * u)) < 0.4
    assert abs(a1["pitch"] - a0["pitch"]) < 3 and abs(a1["joints"][4] - a0["joints"][4]) < 1  # the wrist's roll kept
    assert set(seen["images"]) == {"overhead", "left_wrist"}


def test_mistakes_come_back_as_text_and_silence_ends_the_run(sim):
    llm = Scripted([
        [("fly", {})],
        [("pick", dict(arm="left", target="brush"))],
        [("nudge", dict(side="left", up=2))],
        [("move", dict(left=[{"z": -4}]))],
        "I think I'm done.", "Really.", "Yes.",
    ])
    out = agent.run("do something", llm, log=lambda s: None)
    assert not out["success"] and out["summary"] == "the model stopped calling tools"
    results = [c["result"] for c in out["calls"]]
    assert "no tool 'fly'" in results[0]
    assert "no tool 'pick'" in results[1]  # no task skills: picking up is the model's job
    assert "bad arguments for nudge" in results[2]
    assert '"outcome": "refused"' in results[3] and "table" in results[3]


def test_nudging_by_from_grip_brings_the_grip_point_onto_the_object(sim):
    x, y = robot.detect()["objects"][0]["center_cm"]
    robot.move([{"xy": [x - 2, y - 6], "z": 9, "pitch": 60, "yaw": 0, "width": 84}])
    tools.reset()
    seen = tools.look()
    fwd, right, up = json.loads(tools.state_text(seen))["objects"][0]["from_grip"]["left"]
    assert fwd > 3 and abs(right) > 1 and up < -3
    text, seen, r = tools.run("nudge", dict(arm="left", forward=fwd, right=right))
    assert r["outcome"] == "done"
    fwd2, right2, _ = json.loads(text.split("now: ", 1)[1])["objects"][0]["from_grip"]["left"]
    assert abs(fwd2) < 0.6 and abs(right2) < 0.6


def test_a_grip_on_nothing_opens_again_instead_of_pressing(sim):
    robot.move([{"xy": [-18, 22], "z": 10, "width": 60}])
    text, seen, r = tools.run("grip", dict(arm="left"))
    assert "closed on nothing" in text and seen["arms"]["left"]["width_mm"] > 50


def test_an_object_an_arm_covers_keeps_where_it_was_seen_whole():
    tools.reset()
    ball = dict(name="yellow object", center_cm=[-17.0, 39.0], size_cm=[7.0, 7.0, 7.0], z_cm=3.5)
    tools.remember([ball])
    tools.remember([dict(ball, center_cm=[-15.0, 37.5], size_cm=[4.0, 3.0, 7.0])])  # half under the arm
    tools.remember([])  # all of it under the arm
    (o,) = tools._memory["objects"]
    assert o["center_cm"] == [-17.0, 39.0] and o["seen"] == 1
    tools.remember([dict(ball, center_cm=[-5.0, 30.0])])  # seen whole somewhere else: it rolled
    assert [o["center_cm"] for o in tools._memory["objects"]] == [[-5.0, 30.0], [-17.0, 39.0]]
    for _ in range(tools.FORGET_LOOKS):
        tools.remember([dict(ball, center_cm=[-5.0, 30.0])])
    assert [o["center_cm"] for o in tools._memory["objects"]] == [[-5.0, 30.0]]


def test_the_stop_key_ends_the_run(sim, monkeypatch):
    monkeypatch.setattr(tools, "run", lambda name, args: (
        '{"outcome": "stopped"}', None, dict(outcome="stopped", fail=None, why="stop requested")))
    out = agent.run("keep going", Scripted([[("nudge", dict(arm="left", up=2))]] * 5), log=lambda s: None)
    assert out["turns"] == 1 and not out["success"] and "stop requested" in out["summary"]


def test_calls_in_one_reply_run_in_order_and_stop_at_the_first_that_fails(sim):
    llm = Scripted([
        [("move", dict(left=[{"z": -4}])), ("nudge", dict(arm="left", up=2))],
        [("done", dict(success=False, summary="stopped"))],
    ])
    out = agent.run("two at once", llm, log=lambda s: None)
    results = [c["result"] for c in out["calls"]]
    assert '"outcome": "refused"' in results[0] and results[1] == "skipped: move didn't finish"


def test_a_nudge_keeps_the_pitch_the_arm_was_sent_to_not_the_one_it_sags_to(sim, monkeypatch):
    tools.reset()
    tools.run("move", dict(left=[{"xy": [-14, 24], "z": 8, "pitch": 50}]))
    sent, move, status = [], robot.move, robot.status

    def spy(left=None, right=None, speed=None, check_only=None):
        sent.extend(left or [])
        return move(left, right, speed, check_only)

    def sagged(since=0):  # the arm reads 8 deg steeper than it was sent
        st = status(since)
        st["arms"]["left"]["pitch"] += 8
        return st

    monkeypatch.setattr(robot, "move", spy)
    monkeypatch.setattr(robot, "status", sagged)
    tools.run("nudge", dict(arm="left", forward=1))
    tools.run("nudge", dict(arm="left", forward=1))
    assert sent and all(kf["pitch"] == 50 for kf in sent)


def test_an_object_seen_round_once_keeps_its_height():
    tools.reset()
    ball = dict(name="yellow object", center_cm=[-11.0, 38.0], size_cm=[7.0, 7.0, 7.0], z_cm=3.5)
    tools.remember([ball])
    tools.remember([dict(ball, size_cm=[7.2, 6.6, 2.5], z_cm=1.2)])  # this frame's blob wasn't round enough
    (o,) = tools._memory["objects"]
    assert o["size_cm"] == [7.2, 6.6, 7.0] and o["z_cm"] == 3.5 and o["seen"] == 2


def test_old_results_lose_their_positions_but_keep_their_outcome():
    msgs = [{"role": "tool", "content": f'{{"outcome": "done", "n": {i}}}\nnow: {{"arms": {{}}}}'} for i in range(6)]
    agent.shorten_old_results(msgs, keep=2)
    assert [("now: " in m["content"]) for m in msgs] == [False] * 4 + [True] * 2
    assert msgs[0]["content"] == '{"outcome": "done", "n": 0}'


def test_the_run_stops_after_max_turns(sim):
    out = agent.run("look forever", Scripted([[("look", {})]] * 5), max_turns=3, log=lambda s: None)
    assert out["turns"] == 3 and not out["success"] and "3 turns" in out["summary"]


def test_tool_schemas_are_valid():
    for t in tools.TOOLS:
        f = t["function"]
        assert set(f) == {"name", "description", "parameters"} and f["parameters"]["type"] == "object"
        assert set(f["parameters"]["required"]) <= set(f["parameters"]["properties"])
    json.dumps(tools.TOOLS)
