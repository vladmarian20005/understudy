import numpy as np
import pytest

from harness import kin, sim
from harness.perception import TablePerception, find
from harness.scene import Scene


@pytest.fixture(scope="module")
def scene():
    return Scene()


@pytest.mark.parametrize("objects", ["blocks", "brush"])
def test_overhead_detector_finds_the_simulated_objects(scene, objects):
    """The real-camera detector, run on the simulator's overhead render (same camera model), finds every object
    within a centimetre and gets its size, direction and colour."""
    world = sim.SimWorld(scene, objects)
    snap = world.snapshot()
    img = sim.render(scene, snap, scene.camera)
    dets = TablePerception(scene).detect(img, snap["q"])
    truth = world.visible_objects()
    assert len(dets) == len(truth)
    for o in truth:
        d = min(dets, key=lambda d: np.hypot(*(np.array(d["center_cm"]) - o["pos"][:2] * 100)))
        assert np.hypot(*(np.array(d["center_cm"]) - o["pos"][:2] * 100)) < 1.0
        assert abs(d["size_cm"][0] - o["size"][0] * 100) < 1.5 and abs(d["size_cm"][1] - o["size"][1] * 100) < 1.5
        if o["size"][0] > 2 * o["size"][1]:
            assert abs(kin.wrap90(d["yaw"] - o["yaw"])) < 5
    assert {d["color"] for d in dets} == {"red", "blue"} if objects == "blocks" else {"orange"}


def test_arms_are_not_objects(scene):
    world = sim.SimWorld(scene, objects=[])
    snap = world.snapshot()
    assert TablePerception(scene).detect(sim.render(scene, snap, scene.camera), snap["q"]) == []


def test_find():
    dets = [dict(id="#1", name="red object", color="red"), dict(id="#2", name="brush", color="orange"),
            dict(id="#3", name="blue object", color="blue")]
    assert find(dets, "#2")["name"] == "brush" and find(dets, "1")["id"] == "#1"
    assert find(dets, "brush")["id"] == "#2" and find(dets, "red")["id"] == "#1"
    assert find(dets, "#4") is None and find(dets, "object") is None


def test_ping_pong_with_camera_perception():
    """The whole pass in the simulator with objects found in the overhead image, as on the real arms."""
    import time

    from harness.robotd import Core

    core = Core(sim_mode=True, speed=0, quiet=True, dataset=None, vision=True)

    def run(name, **args):
        assert core.start_skill(name, args)["ok"]
        while core.busy():
            time.sleep(0.005)
        return core.skill["result"]

    try:
        assert core.perception.detect()[0]["color"] == "orange"
        for name, args in [("pick", dict(arm="left", target="orange")), ("handover", dict(giver="left", taker="right")),
                           ("place", dict(arm="right", xy=[12, 24])), ("home", {})]:
            r = run(name, **args)
            assert r["outcome"] == "done", (name, r["why"])
        brush = core.world.objects[0]
        assert np.hypot(*(brush.pos[:2] * 100 - [12, 24])) < 4
    finally:
        core.close()
