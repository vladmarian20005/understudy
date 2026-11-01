"""The contract between robotd and the brain, UI and recorder consumers ("Understudy contracts" doc): every command's
reply shape, the failure kinds, and the episode label file. A change here is a change Andrei's side has to know about."""
import json
import time

import pytest

from harness import sim
from harness.robotd import Core, Trace
from harness.skills import FAIL_KINDS

RESULT_KEYS = {"op", "outcome", "fail", "why", "t", "arms", "events"}
ARM_KEYS = {"tip_cm", "pitch", "yaw", "width_mm", "holding", "object", "mode", "joints"}
OUTCOMES = {"done", "refused", "failed", "stopped"}


@pytest.fixture
def core(tmp_path):
    c = Core(sim_mode=True, speed=0, quiet=True, dataset="contract", datasets=str(tmp_path))
    yield c
    c.close()


def call(core, cmd, **kw):
    """One request as it travels: through JSON both ways."""
    return json.loads(json.dumps(core.handle(json.loads(json.dumps(dict(cmd=cmd, **kw)))), default=str))


def skill(core, name, **args):
    r = call(core, "skill", name=name, args=args)
    if not r["ok"]:
        return r
    while call(core, "status")["busy"]:
        time.sleep(0.005)
    return call(core, "status")["skill"]["result"]


def test_status(core):
    st = call(core, "status", since=0)
    assert {"ok", "mode", "armed", "busy", "skill", "arms", "events", "last_event", "episode", "cameras", "policy",
            "interventions", "sim"} <= set(st)
    assert st["mode"] in ("idle", "skill", "policy") and st["sim"] is True and st["interventions"] == 0
    for side in ("left", "right"):
        assert ARM_KEYS <= set(st["arms"][side])
        assert len(st["arms"][side]["tip_cm"]) == 3
    assert all({"id", "t", "kind", "text"} <= set(e) for e in st["events"])
    assert set(st["cameras"]) == {"overhead", "left_wrist", "right_wrist"}
    assert all({"ok", "fps", "age_s"} <= set(h) for h in st["cameras"].values())
    assert st["policy"]["running"] is False


def test_look_and_detect(core):
    r = call(core, "look", cams=["overhead", "left_wrist", "right_wrist"])
    assert r["ok"] and set(r["images"]) == {"overhead", "left_wrist", "right_wrist"}
    obj = r["objects"][0]
    assert {"id", "name", "center_cm", "size_cm", "yaw", "color", "px", "held_by"} <= set(obj)
    assert obj["id"].startswith("#")
    assert call(core, "detect")["objects"][0]["id"] == obj["id"]


def test_every_motion_result_has_the_same_shape(core):
    for name, args in [("home", {}), ("reach", dict(arm="left", target="brush")), ("grasp", dict(arm="left")),
                       ("move_to", dict(arm="left", xy=[-12, 26], z=12)), ("release", dict(arm="left")),
                       ("pick", dict(arm="left", target="brush")), ("handover", dict(giver="left", taker="right")),
                       ("place", dict(arm="right", xy=[14, 26])), ("home", {})]:
        r = skill(core, name, **args)
        assert RESULT_KEYS <= set(r), (name, r)
        assert r["outcome"] == "done", (name, r["why"])
        assert r["fail"] is None and r["why"] is None
        assert len(r["t"]) == 2 and r["t"][1] >= r["t"][0]
        assert set(r["arms"]) == {"left", "right"}


def test_failures_name_a_kind_from_the_list(core):
    r = skill(core, "grasp", arm="left")  # no reach first
    assert r["outcome"] == "refused" and r["fail"] in FAIL_KINDS
    r = skill(core, "move_to", arm="left", xy=[40, 44], z=5)  # out of reach
    assert r["outcome"] == "refused" and r["fail"] == "unreachable"
    r = skill(core, "pick", arm="left", target="nothing like this")
    assert r["fail"] == "not_seen"
    r = skill(core, "no_such_command")
    assert r["outcome"] == "refused"


def test_a_second_command_while_one_runs_is_busy(tmp_path):
    c = Core(sim_mode=True, speed=1.0, quiet=True, dataset=None)  # real time: the first one is still running
    try:
        assert call(c, "skill", name="move_to", args=dict(arm="left", xy=[-10, 30], z=12))["ok"]
        r = call(c, "skill", name="home", args={})
        assert r["ok"] is False and r["fail"] == "busy"
        while call(c, "status")["busy"]:
            time.sleep(0.01)
    finally:
        c.close()


def test_the_episode_label_file(core, tmp_path):
    b = call(core, "episode_begin", task="Pass the brush to the other arm", source={"kind": "unspecified"},
             giver="left", taker="right", target_cm=[14, 26])
    assert b["ok"] and isinstance(b["episode"], int)
    for name, args in [("pick", dict(arm="left", target="brush")), ("handover", dict(giver="left", taker="right")),
                       ("place", dict(arm="right", xy=[14, 26])), ("home", {})]:
        assert skill(core, name, **args)["outcome"] == "done"
    e = call(core, "episode_end")
    assert e["ok"] and {"episode", "frames", "duration_s", "sidecar"} <= set(e)
    label = json.load(open(e["sidecar"]))
    assert {"version", "dataset", "episode", "task", "fps", "frames", "source", "robot"} <= set(label)
    robot = label["robot"]
    assert {"started", "duration_s", "giver", "taker", "target_cm", "valid", "note", "interventions", "steps",
            "events", "object", "check", "log", "camera_gaps"} <= set(robot)
    assert robot["valid"] and robot["check"]["passed"]
    assert all(s["outcome"] in OUTCOMES for s in robot["steps"])
    judge = dict(model="muse-spark", success=True, confidence=0.9, failure=None, reason="on target", frames=[0])
    r = call(core, "episode_label", episode=b["episode"], section="judge", data=judge)
    assert r["ok"] and r["final"] == {"success": True, "train": True, "review": False,
                                      "why": "valid; judge and check agree"}
    assert call(core, "episode_label", episode=b["episode"], section="other", data={})["ok"] is False


def test_policy_commands_refuse_cleanly(core):
    r = call(core, "policy_start", path="/no/such/checkpoint")
    assert r["ok"] is False and "can't load" in r["errors"][0]
    assert call(core, "policy_stop")["ok"] is False


def test_unknown_commands_are_refused(core):
    r = call(core, "fly")
    assert r["ok"] is False and r["errors"]


def test_the_trace_keeps_requests_and_replies(core, tmp_path):
    tr = Trace(str(tmp_path / "trace.jsonl"))
    for req in [dict(cmd="status"), dict(cmd="status"), dict(cmd="detect")]:
        tr.write(req, core.handle(req))
    lines = [json.loads(l) for l in open(tmp_path / "trace.jsonl")]
    assert [l["req"]["cmd"] for l in lines] == ["status", "detect"]  # status at most once a second
    assert all({"t", "req", "rep"} <= set(l) for l in lines)


def test_simulated_faults_reach_the_failure_kinds(tmp_path):
    c = Core(sim_mode=True, speed=0, quiet=True, dataset=None, faults={"miss": 1.0})
    try:
        r = skill(c, "pick", arm="left", target="brush")
        assert r["outcome"] == "failed" and r["fail"] == "miss"
        assert any("simulated fault" in e["text"] for e in c.events_since(0))
    finally:
        c.close()
    with pytest.raises(ValueError):
        sim.SimWorld(Core.__init__.__globals__["Scene"](), faults={"meteor": 1.0})


def test_a_simulated_camera_drop_out_invalidates_the_episode(tmp_path):
    c = Core(sim_mode=True, speed=0, quiet=True, dataset="drop", datasets=str(tmp_path), faults={"camera": 0.2})
    try:
        assert call(c, "episode_begin", task="t")["ok"]
        for name, args in [("pick", dict(arm="left", target="brush")), ("place", dict(arm="left", xy=[-14, 28]))]:
            assert skill(c, name, **args)["outcome"] == "done"
        e = call(c, "episode_end")
        robot = json.load(open(e["sidecar"]))["robot"]
        assert robot["valid"] is False and robot["camera_gaps"] > 0 and "camera drop-out" in robot["note"]
    finally:
        c.close()


def test_pixels_and_table_cm_convert_both_ways(core):
    cm = [[10.0, 25.0], [-15.0, 30.0, 4.0]]
    px = call(core, "to_pixels", cm=cm)["px"]
    assert len(px) == 2 and all(0 <= u < 640 and 0 <= v < 480 for u, v in px)
    back = call(core, "to_table", px=px[:1])["cm"]
    assert back[0] == pytest.approx([10.0, 25.0], abs=0.05)
    assert call(core, "to_table", px=px[1:], z_cm=4.0)["cm"][0] == pytest.approx([-15.0, 30.0], abs=0.05)


def test_home_can_set_the_grippers(core):
    assert skill(core, "home", width=80)["outcome"] == "done"
    st = call(core, "status")
    assert all(a["width_mm"] == pytest.approx(80, abs=1) for a in st["arms"].values())
