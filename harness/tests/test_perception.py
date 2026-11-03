import numpy as np
import pytest

from harness import kin, sim
from harness.perception import TablePerception
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


def test_robotd_with_camera_perception_finds_the_brush_where_it_is():
    """robotd --vision: objects found in the overhead image, as on the real arms, not read from the simulator."""
    from harness.robotd import Core

    core = Core(sim_mode=True, speed=0, quiet=True, dataset=None, vision=True)
    try:
        d = core.handle(dict(cmd="detect"))["objects"][0]
        brush = core.world.objects[0]
        assert d["color"] == "orange" and np.hypot(*(np.array(d["center_cm"]) - brush.pos[:2] * 100)) < 1.0
    finally:
        core.close()
