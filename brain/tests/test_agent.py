"""The agent loop against the simulator, with a scripted model: tool calls reach robotd, results and images come
back, and the run ends on done, on a model that stops calling tools, or on too many turns."""
import json
import threading

import pytest

from brain import agent, tools
from brain.llm import Scripted
from harness import robot
from harness.robotd import Core, serve


@pytest.fixture
def sim(tmp_path, monkeypatch):
    core = Core(sim_mode=True, speed=0, quiet=True, dataset="agent", datasets=str(tmp_path))
    stop = threading.Event()
    threading.Thread(target=serve, args=(core, 5898, stop), daemon=True).start()
    monkeypatch.setattr(robot, "PORT", 5898)
    yield core
    stop.set()
    core.close()


def images(messages):
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")


def test_a_scripted_pass_runs_through_the_tools(sim):
    llm = Scripted([
        [("pick", dict(arm="left", target="brush"))],
        [("handover", dict(giver="left", taker="right"))],
        [("place", dict(arm="right", xy=[14, 26]))],
        [("home", {})],
        [("done", dict(success=True, summary="passed"))],
    ])
    out = agent.run("pass the brush to the right arm", llm, log=lambda s: None)
    assert out["success"] and out["turns"] == 5
    assert [c["name"] for c in out["calls"]] == ["pick", "handover", "place", "home", "done"]
    assert all('"outcome": "done"' in c["result"] for c in out["calls"][:4])
    brush = robot.detect()["objects"][0]
    assert brush["center_cm"][0] > 8 and not brush["held_by"]
    # every call came back with the positions and a fresh overhead image; older images were dropped
    last = llm.seen[-1]
    assert last[-1]["role"] == "user" and images(last) == agent.KEEP_IMAGES
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert len(tool_msgs) == 4 and all("now: " in m["content"] for m in tool_msgs)


def test_mistakes_come_back_as_text_and_silence_ends_the_run(sim):
    llm = Scripted([
        [("fly", {})],
        [("pick", dict(arm="left", target="brush", force=3))],
        [("move", dict(left=[{"z": -4}]))],
        "I think I'm done.", "Really.", "Yes.",
    ])
    out = agent.run("do something", llm, log=lambda s: None)
    assert not out["success"] and out["summary"] == "the model stopped calling tools"
    results = [c["result"] for c in out["calls"]]
    assert "no tool 'fly'" in results[0]
    assert "bad arguments for pick" in results[1]
    assert '"outcome": "refused"' in results[2] and "table" in results[2]
    assert robot.detect()["objects"][0]["held_by"] is None


def test_the_run_stops_after_max_turns(sim):
    out = agent.run("look forever", Scripted([[("look", {})]] * 5), max_turns=3, log=lambda s: None)
    assert out["turns"] == 3 and not out["success"] and "3 turns" in out["summary"]


def test_tool_schemas_are_valid():
    for t in tools.TOOLS:
        f = t["function"]
        assert set(f) == {"name", "description", "parameters"} and f["parameters"]["type"] == "object"
        assert set(f["parameters"]["required"]) <= set(f["parameters"]["properties"])
    json.dumps(tools.TOOLS)
