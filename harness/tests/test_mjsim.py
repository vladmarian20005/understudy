import time

import cv2
import mujoco
import numpy as np
import pytest

from harness import mjsim
from harness.robotd import Core
from harness.scene import SIDES, Scene


@pytest.fixture(scope="module")
def world():
    w = mjsim.MjWorld(Scene(), "balls")
    yield w
    w.close()


def test_the_cad_arms_move_like_the_kinematics(world):
    """Same joints in, same gripper_link out: the MuJoCo arms stand at scene.json's bases and follow kin.py."""
    m, d = world.model, mujoco.MjData(world.model)
    rng = np.random.default_rng(0)
    for s in SIDES:
        arm = world.scene.arms[s]
        for _ in range(10):
            q = rng.uniform(arm.limits[:, 0], arm.limits[:, 1])
            d.qpos[world.qadr[s]] = arm.to_urdf(q)
            mujoco.mj_kinematics(m, d)
            T = arm.frames(q)[-1]
            b = m.body(f"{s}/gripper").id
            assert np.linalg.norm(d.xpos[b] - T[:3, 3]) < 1e-4
            assert np.abs(d.xmat[b].reshape(3, 3) - T[:3, :3]).max() < 1e-3


def test_the_overhead_render_lands_where_the_calibrated_camera_projects():
    scene = Scene()
    pts = [(0.0, 0.30), (-0.15, 0.40), (0.20, 0.40), (0.0, 0.05), (-0.2, 0.2)]
    w = mjsim.MjWorld(scene, [(f"m{i}", "box", (0.012, 0.012, 0.002), (0, 255, 0), x, y, 0.0)
                              for i, (x, y) in enumerate(pts)])
    hsv = cv2.cvtColor(w.render("overhead", scene.camera.width, scene.camera.height), cv2.COLOR_BGR2HSV)
    mask = (np.abs(hsv[..., 0].astype(int) - 60) < 10) & (hsv[..., 1] > 150) & (hsv[..., 2] > 60)
    _, _, _, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    for x, y in pts:
        p = scene.camera.project([[x, y, 0.002]])[0]
        assert min(np.linalg.norm(c - p) for c in centres[1:]) < 2.0  # px
    w.close()


def test_the_gripper_opening_follows_the_moving_jaw(world):
    assert world.grip_opening(world.grip_angle(0.0)) == pytest.approx(0.0, abs=0.5)
    assert np.all(np.diff(world.grip_mm) >= 0)
    assert world.grip_opening(mjsim.OPEN_DEG) == pytest.approx(world.scene.stroke_mm, abs=1.5)
    for mm in (10.0, 40.0, 67.0):
        assert world.grip_opening(world.grip_angle(mm)) == pytest.approx(mm, abs=0.2)


def test_the_cameras_render_at_the_rig_sizes(world):
    snap = world.snapshot()
    for name, (w, h) in (("overhead", (640, 480)), ("left_wrist", (1024, 768)), ("right_wrist", (640, 480))):
        img = world.render(name, w, h, snap)
        assert img.shape == (h, w, 3) and img.std() > 10  # a picture, not a blank


def run(core, name, **args):
    r = core.start_skill(name, args)
    assert r["ok"], r
    while core.busy():
        time.sleep(0.005)
    return core.skill["result"]


def test_robotd_picks_a_ball_by_friction_and_sets_it_down():
    core = Core(sim_mode=True, speed=0, quiet=True, hd=True, objects="balls", dataset=None)
    try:
        ball = core.world.objects[0]
        r = run(core, "pick", arm="left", target=ball.name)
        assert r["outcome"] == "done", r.get("why")
        assert ball.held == ["left"] and r["width_mm"] == pytest.approx(67, abs=4)
        assert core.ctl.snapshot()["grip_ma"]["left"] > 200  # squeezing
        assert run(core, "move_to", arm="left", xy=[-10, 28], z=20)["outcome"] == "done"
        assert ball.held == ["left"] and ball.pos[2] > 0.12  # carried, not dragged
        assert run(core, "place", arm="left", xy=[-15, 20])["outcome"] == "done"
        assert not ball.held and ball.pos[2] == pytest.approx(0.0335, abs=0.003)
        assert np.linalg.norm(ball.pos[:2] * 100 - [-15, 20]) < 5  # the opening jaw can flick it: it rolls a bit
    finally:
        core.close()
