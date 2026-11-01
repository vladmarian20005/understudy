import math

import numpy as np
import pytest

from harness import kin
from harness.scene import Scene


@pytest.fixture(scope="module")
def scene():
    return Scene()


def test_chain_loads():
    origins, limits = kin.load_chain()
    assert len(origins) == 5 and limits.shape == (5, 2)


def test_angle_conversion_round_trips(scene):
    arm = scene.arms["left"]
    q = np.array([10.0, -20.0, 30.0, 40.0, -50.0])
    assert np.allclose(arm.to_deg(arm.to_urdf(q)), q)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_reaches_random_poses_found_by_fk(scene, side):
    """FK of random in-limit postures gives reachable poses; for those in the working area (over the table in front
    of the arms, fingertips pointing down), IK must find a posture with the same pose."""
    arm = scene.arms[side]
    rng = np.random.default_rng(0)
    solved = 0
    for _ in range(200):
        q = rng.uniform(arm.limits[:, 0], arm.limits[:, 1])
        p, pitch, yaw = arm.pose(q)
        if not (0.01 < p[2] < 0.3 and p[1] > 0.08 and 0 <= pitch <= 90 and scene.on_table(p[:2])):
            continue
        q2 = arm.ik(p, pitch, yaw)
        p2, pitch2, yaw2 = arm.pose(q2)
        assert np.linalg.norm(p2 - p) < 0.002
        assert abs(pitch2 - pitch) < 2
        assert abs(kin.wrap90(yaw2 - yaw)) < 2
        assert arm.within_limits(q2, slack=1e-6)
        solved += 1
    assert solved >= 10


@pytest.mark.parametrize("side", ["left", "right"])
def test_steep_grasps_cover_the_middle_of_the_table(scene, side):
    """Low grasps around the handover spot, at any jaw direction, with the gripper at least 45 degrees down."""
    arm = scene.arms[side]
    mid = scene.middle()
    for dx in (-0.08, 0.0, 0.08):
        for yaw in (0, 45, 90, 135):
            q, pitch = arm.steepest([mid[0] + dx, mid[1], 0.02], yaw)
            p, pitch2, y = arm.pose(q)
            assert p[2] == pytest.approx(0.02, abs=0.002) and pitch >= 45 and abs(pitch2 - pitch) < 2
            assert abs(kin.wrap90(y - yaw)) < 2


def test_straight_down_reaches_low_points_near_the_base(scene):
    arm = scene.arms["left"]
    p = np.append(scene.base_xy("left") + 0.2 * scene.forward("left"), 0.02)
    q, pitch = arm.steepest(p, None)
    assert pitch == 90


def test_far_point_is_unreachable(scene):
    with pytest.raises(kin.Unreachable):
        scene.arms["left"].ik([0.4, 0.45, 0.02], 90, 0)


def test_home_is_reachable(scene):
    for s in ("left", "right"):
        q, pitch = scene.arms[s].steepest(scene.home_tip(s), lowest=0)
        assert scene.arms[s].within_limits(q) and pitch >= 30


def test_camera_round_trip(scene):
    cam = scene.camera
    pts = np.array([[0.0, 0.2, 0.0], [-0.2, 0.3, 0.0], [0.15, 0.05, 0.03]])
    px = cam.project(pts)
    assert np.all((px[:, 0] > 0) & (px[:, 0] < cam.width) & (px[:, 1] > 0) & (px[:, 1] < cam.height))
    back = np.array([cam.to_table(px[i], z=pts[i, 2])[0] for i in range(len(pts))])
    assert np.allclose(back, pts, atol=1e-4)


def test_table_frame_matches_the_image(scene):
    """x grows to the right in the image and y grows up the image."""
    px = scene.camera.project([[0, 0.2, 0], [0.1, 0.2, 0], [0, 0.3, 0]])
    assert px[1, 0] > px[0, 0] and px[2, 1] < px[0, 1]
    assert math.isclose(scene.arms["left"].base[0, 3], -scene.arms["right"].base[0, 3], abs_tol=0.01)


def symmetric(arm):
    """The arm with a parallel gripper's tool point, on the roll axis."""
    arm.tcp = np.array([0.0, 0.0, -0.12])
    return arm


def test_half_a_turn_of_the_wrist_is_the_same_grasp():
    from harness.scene import Scene

    arm = symmetric(Scene().arms["left"])
    for q in ([10.0, -20.0, 15.0, 70.0, 120.0], [10.0, -20.0, 15.0, 70.0, -60.0], [-5, 10, -30, 80, 30]):
        q = np.array(q)
        c = arm.canonical_roll(q)
        assert (arm.tool(c)[1] @ arm.jaw)[1] >= 0  # the jaw axis points away from the bases
        p1, pitch1, yaw1 = arm.pose(q)
        p2, pitch2, yaw2 = arm.pose(c)
        assert np.allclose(p1, p2) and pitch1 == pytest.approx(pitch2) and yaw1 == pytest.approx(yaw2)
        assert c[4] == pytest.approx(q[4]) or abs(c[4] - q[4]) == pytest.approx(180)


def test_the_roll_is_canonical_even_with_another_joint_at_its_limit():
    from harness.scene import Scene

    arm = symmetric(Scene().arms["right"])
    q = np.array([10.0, -20.0, 15.0, arm.limits[3, 1] + 0.2, -104.7])  # wrist flex just past its limit
    c = arm.canonical_roll(q)
    assert (arm.tool(c)[1] @ arm.jaw)[1] >= 0


def test_an_off_axis_tool_point_keeps_its_roll():
    """The original gripper's tool point sits between its jaws, off the roll axis: half a turn would move it."""
    from harness.scene import Scene

    arm = Scene().arms["left"]
    arm.tcp = np.array([0.027, 0.0, -0.104])
    q = np.array([10.0, -20.0, 15.0, 70.0, -60.0])
    assert np.array_equal(arm.canonical_roll(q), q)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_ready_pose_is_level_points_ahead_with_the_camera_on_top(scene, side):
    arm = scene.arms[side]
    q = scene.ready(side)
    p, R = arm.tool(q)
    assert arm.within_limits(q)
    assert abs(arm.angles(R)[0]) < 2  # level
    a = -R[:, 2]
    assert math.degrees(abs(math.atan2(a[0], a[1]))) < 6  # pointing ahead, +y
    assert (R @ kin.CAMERA)[2] > 0.95  # the camera on top
    assert p[2] == pytest.approx(0.28, abs=0.002)


@pytest.mark.parametrize("side", ["left", "right"])
def test_holding_the_roll_keeps_the_camera_on_top_level_or_steep(scene, side):
    """At a level pitch the yaw can't see the roll; holding the roll joint keeps the camera on the gripper's top side:
    up when level, away from the base pointing down."""
    arm = scene.arms[side]
    up = arm.upright_roll()
    x = -0.10 if side == "left" else 0.10
    for p, pitch in (([x, 0.28, 0.20], 0), ([x, 0.25, 0.08], 45), ([x, 0.22, 0.03], 90)):
        q = arm.ik(np.array(p), pitch, roll=up)
        _, R = arm.tool(q)
        assert q[4] == pytest.approx(up)
        cam, approach = R @ kin.CAMERA, -R[:, 2]
        upward = np.array([0, 0, 1.0]) - approach * approach[2]  # up, square to where the fingers point
        if np.linalg.norm(upward) < 0.2:  # straight down: away from the base
            upward = np.append(np.asarray(p[:2]) - arm.base[:2, 3], 0.0)
        assert cam @ upward / np.linalg.norm(upward) > 0.9, (pitch, cam)
        assert abs((R @ arm.jaw)[2]) < 0.1  # the jaws close level
