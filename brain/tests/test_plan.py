"""The plan, act, check loop against the simulator, with a scripted model and the simulator's detector in SAM's place."""
import itertools
import threading

import numpy as np
import pytest

from brain import plan, tools, wrist
from brain.llm import Scripted
from harness import robot
from harness.robotd import Core, serve

PORTS = itertools.count(5898)  # one per test: the last test's server can still hold its port while the next binds
FIND = {"find": ["brush"]}
GRASP = {"left": {"do": "grasp", "object": "brush"}, "right": None, "check": "left holds it"}
PLACE = {"left": {"do": "place", "at": [-4, 32]}, "right": None}
OK = {"ok": True, "why": "as it should be"}
DONE = {"done": {"success": True, "summary": "it's there"}}


def plan_of(*steps):
    return {"understanding": "pick it up and set it down", "steps": list(steps), "done_when": "it lies at (-4, 32)"}


@pytest.fixture
def sim(tmp_path, monkeypatch):
    core = Core(sim_mode=True, speed=0, quiet=True, dataset="plan", datasets=str(tmp_path))
    stop = threading.Event()
    port = next(PORTS)
    threading.Thread(target=serve, args=(core, port, stop), daemon=True).start()
    monkeypatch.setattr(robot, "PORT", port)
    monkeypatch.setattr(tools.sam, "objects", lambda *a, **k: robot.detect()["objects"])
    monkeypatch.setattr(wrist, "masks", lambda img, word: [])
    yield core
    stop.set()
    core.close()
    tools.FIND = []


def brush():
    return robot.detect()["objects"][0]


def test_a_plan_runs_and_checks_where_it_asks(sim):
    llm = Scripted([FIND, plan_of(GRASP, PLACE), OK, DONE])
    out = plan.Run(llm, log=lambda s: None).go("put the brush at (-4, 32)")
    assert out["success"] and out["steps"] == 2 and out["calls"] == 4
    at, steps = brush(), [r for r in out["record"] if "step" in r]
    assert abs(at["center_cm"][0] + 4) < 3 and abs(at["center_cm"][1] - 32) < 3 and not at["held_by"], steps


def test_the_prompt_goes_with_every_call_and_the_task_with_the_plan(sim):
    seen = []
    llm = Scripted([FIND, plan_of(), DONE])
    ask = llm.ask
    llm.ask = lambda system, messages, effort=None: seen.append(system) or ask(system, messages, effort)
    run = plan.Run(llm, log=lambda s: None)
    run.go("put the brush at (-4, 32)")
    assert len(seen) == 3 and all("Actions, one per arm per step" in s for s in seen)
    assert run.plan["task"] == "put the brush at (-4, 32)" and run.plan["model"] == "scripted"


def test_plan_only_moves_nothing(sim):
    before = robot.status()["arms"]["left"]["joints"]
    out = plan.Run(Scripted([FIND, plan_of(GRASP)]), log=lambda s: None).go("anything", plan_only=True)
    assert not out["success"] and out["steps"] == 0
    assert np.allclose(robot.status()["arms"]["left"]["joints"], before, atol=1)


def test_a_failed_step_is_shown_to_the_model_and_its_new_steps_replace_the_rest(sim):
    nothing = {"left": {"do": "grasp", "object": "no such thing"}, "right": None}
    llm = Scripted([FIND, plan_of(nothing, PLACE), {"ok": False, "why": "no such thing", "next": [GRASP],
                                                     "replace_rest": True}, DONE])
    out = plan.Run(llm, log=lambda s: None).go("put the brush at (-4, 32)")
    asked = llm.seen[2][-1]["content"][0]["text"]
    assert "it failed: what now?" in asked and "no single object" in asked
    assert out["success"] and out["steps"] == 2 and brush()["held_by"] == "left"


def test_the_model_can_correct_what_the_camera_got_wrong(sim):
    fix = {"correct": {"brush": {"size_cm": [20.0, 3.0, 2.5], "center_cm": [-8.0, 24.0]}}}
    run = plan.Run(Scripted([FIND, dict(plan_of(), **fix), DONE]), log=lambda s: None)
    run.go("anything")
    o = next(o for o in tools._memory["objects"] if o["name"] == "brush")
    assert o["size_cm"] == [20.0, 3.0, 2.5] and o["center_cm"] == [-8.0, 24.0]


def test_the_run_stops_at_the_call_limit(sim, monkeypatch):
    monkeypatch.setattr(plan, "MAX_CALLS", 3)
    again = {"left": {"do": "move", "rel": [0, 0, -1]}, "right": None, "check": "lower"}
    out = plan.Run(Scripted([FIND, plan_of(*[again] * 5), OK, OK, OK]), log=lambda s: None).go("anything")
    assert not out["success"] and "calls" in out["summary"]
