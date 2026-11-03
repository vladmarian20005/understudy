import time
from concurrent.futures import ThreadPoolExecutor

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



def test_cameras_render_from_several_threads_at_once(world):
    """robotd renders for a request while the camera streams render: OpenGL from two threads used to deadlock."""
    with ThreadPoolExecutor(4) as pool:
        imgs = list(pool.map(lambda n: world.render(n, 640, 480), ["overhead", "left_wrist", "right_wrist"] * 4))
    assert all(img.shape == (480, 640, 3) for img in imgs)


def run(core, name, **args):
    r = core.start_skill(name, args)
    assert r["ok"], r
    while core.busy():
        time.sleep(0.005)
    return core.skill["result"]


def settle(core, seconds):
    t0 = core.ctl.now()
    while core.ctl.now() - t0 < seconds:
        time.sleep(0.005)


def test_a_ball_is_held_by_friction_when_the_jaws_close_on_it():
    """The loop's grasp, by moves: in front of the ball on the line from the base, tilted 35 deg, then the grip point
    onto its middle, and close. It comes up with the gripper and falls when the jaws open."""
    core = Core(sim_mode=True, speed=0, quiet=True, hd=True, objects="balls", dataset=None)
    try:
        ball = core.world.objects[0]
        core.world.place(ball.name, (-0.26, 0.34))  # where a tilted gripper reaches it, clear of the other balls
        settle(core, 0.5)
        mid = ball.pos * 100
        head = mid[:2] - core.scene.base_xy("left") * 100
        head /= np.linalg.norm(head)
        c, s = np.cos(np.radians(35)), np.sin(np.radians(35))
        for back, width in ((6, 90), (0, None)):  # the fingertips are 3 cm past the grip point
            tip = mid - np.append(head * back, 0) + 3.0 * np.array([c * head[0], c * head[1], -s])
            kf = dict(xy=list(tip[:2]), z=float(tip[2]), pitch=35, **({"width": width} if width else {}))
            assert run(core, "move", left=kf, speed="slow")["outcome"] == "done"
        assert run(core, "move", left={"width": 0})["outcome"] == "done"
        settle(core, 0.5)
        assert ball.held == ["left"] and core.ctl.snapshot()["grip_ma"]["left"] > 200  # squeezing
        assert run(core, "move", left={"rel": [0, 0, 10]})["outcome"] == "done"
        assert ball.held == ["left"] and ball.pos[2] > 0.12  # carried, not dragged
        assert run(core, "move", left={"width": 90})["outcome"] == "done"
        settle(core, 1.5)
        assert not ball.held and ball.pos[2] == pytest.approx(0.0335, abs=0.003)
    finally:
        core.close()


# ---- keeping things where the arms reach (2 Nov) ---------------------------------------------------------------------
def test_a_rolling_ball_stops_at_the_fence_and_the_arms_pass_through_it():
    w = mjsim.MjWorld(Scene(), "balls")
    try:
        o = next(o for o in w.objects if o.name == "ball 3")
        w.data.qvel[o.dadr:o.dadr + 3] = [0.0, 2.0, 0.0]  # rolling away from the arms at 2 m/s
        for _ in range(100):
            w.step(0.02)
        assert o.pos[1] < mjsim.FENCE[3]  # stopped inside, not rolled off past the arms' reach
        fence = [g for g in range(w.model.ngeom) if w.model.geom(g).name.startswith("fence")]
        arm = [g for g in range(w.model.ngeom) if w.model.geom_contype[g] == mjsim.ARM_TYPE]
        assert len(fence) == 4 and arm
        for f in fence:  # nothing of the arms collides with it
            for g in arm:
                assert not (w.model.geom_contype[f] & w.model.geom_conaffinity[g]
                            or w.model.geom_contype[g] & w.model.geom_conaffinity[f])
    finally:
        w.close()


def test_reset_puts_back_only_what_no_arm_can_reach():
    core = Core(sim_mode=True, speed=0, quiet=True, dataset=None, hd=True, objects="balls")
    try:
        r = core.handle(dict(cmd="sim_reset", name="tube"))  # where the scene put it: in reach
        assert not r["ok"] and "within the" in r["errors"][0] and "handle it with the arms" in r["errors"][0]
        core.world.place("tube", (0.40, 0.44))  # knocked past both arms' reach
        r = core.handle(dict(cmd="sim_reset", name="tube"))
        assert r["ok"] and r["object"] == "tube" and r["now_cm"] == [4.0, 30.0]
        tube = next(o for o in core.world.objects if o.name == "tube")
        assert np.linalg.norm(tube.pos[:2] - [0.04, 0.30]) < 0.01
        core.world.place("ball 1", (0.40, 0.44))  # the brain's name for it may differ: kind and where it was seen
        r = core.handle(dict(cmd="sim_reset", name="ball 4", kind="ball", near_cm=[39, 43]))
        assert r["ok"] and r["object"] == "ball 1"
        assert not core.handle(dict(cmd="sim_reset", name="cup"))["ok"]
        # knocked over within reach: the arms can't stand it up, so it may be reset
        tube = next(o for o in core.world.objects if o.name == "tube")
        with core.world.lock:
            core.world.data.qpos[tube.qadr:tube.qadr + 7] = [0.05, 0.28, 0.045, 0.7071068, 0.7071068, 0.0, 0.0]
            mujoco.mj_forward(core.world.model, core.world.data)
        r = core.handle(dict(cmd="sim_reset", name="tube"))
        assert r["ok"] and "lying 90 deg from upright" in r["was"]
        assert tube.R[2, 2] > 0.99  # standing again
    finally:
        core.close()
