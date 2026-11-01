import numpy as np
import pytest

from harness import motion
from harness.motion import Goal, Phase, PlanError, Planner
from harness.scene import Scene


@pytest.fixture(scope="module")
def scene():
    return Scene()


@pytest.fixture(scope="module")
def home(scene):
    return {s: np.append(scene.arms[s].steepest(scene.home_tip(s), lowest=0)[0], 40.0) for s in ("left", "right")}


def test_segment_distance():
    a, b = np.array([0, 0, 0.0]), np.array([1, 0, 0.0])
    assert motion.seg_dist(a, b, np.array([0.5, 1, 0]), np.array([0.5, 2, 0])) == pytest.approx(1.0)
    assert motion.seg_dist(a, b, np.array([2, 0, 0.0]), np.array([3, 0, 0.0])) == pytest.approx(1.0)
    assert motion.seg_dist(a, b, np.array([0.5, -1, 0.0]), np.array([0.5, 1, 0.0])) == pytest.approx(0.0)


def test_straight_line_ends_on_target(scene, home):
    target = np.append(scene.middle() + [-0.06, 0], 0.05)
    [traj] = Planner(scene).plan(home, [Phase("go", {"left": Goal(p=target, pitch=60, yaw=0, width=60)})])
    rows = traj.rows["left"]
    p, pitch, yaw = scene.arms["left"].pose(rows[-1])
    assert np.linalg.norm(p - target) < 0.002 and abs(pitch - 60) < 2 and rows[-1, 5] == pytest.approx(60)
    assert np.max(np.abs(np.diff(rows[:, :5], axis=0))) < motion.MAX_JOINT_STEP
    # the fingertips stay on the line from start to end
    start = scene.arms["left"].pose(home["left"])[0]
    d = target - start
    for q in rows[::10]:
        tip = scene.arms["left"].pose(q)[0]
        off = (tip - start) - d * ((tip - start) @ d) / (d @ d)
        assert np.linalg.norm(off) < 0.003
    assert np.allclose(traj.rows["right"], home["right"])


def test_both_arms_share_a_phase(scene, home):
    goals = {s: Goal(p=scene.home_tip(s) + [0, 0.05, 0], pitch="steep") for s in ("left", "right")}
    [traj] = Planner(scene).plan(home, [Phase("both", goals)])
    assert len(traj.rows["left"]) == len(traj.rows["right"]) > 1


def test_phases_chain(scene, home):
    tip = scene.home_tip("left")
    trajs = Planner(scene).plan(home, [Phase("down", {"left": Goal(p=tip - [0, 0, 0.05])}),
                                       Phase("up", {"left": Goal(p=tip)})])
    assert [t.name for t in trajs] == ["down", "up"]
    assert np.allclose(trajs[0].rows["left"][-1], trajs[1].rows["left"][0], atol=1.0)


def test_table_check(scene, home):
    p = np.append(scene.middle() + [-0.06, 0], 0.0)
    with pytest.raises(PlanError) as e:
        Planner(scene).plan(home, [Phase("dig", {"left": Goal(p=p, pitch=60, floor=0.01)})])
    assert e.value.kind == "check" and "table" in e.value.why


def test_arms_may_not_meet(scene, home):
    right_tip = scene.arms["right"].pose(home["right"])[0]
    with pytest.raises(PlanError) as e:
        Planner(scene).plan(home, [Phase("crash", {"left": Goal(p=right_tip + [0, 0, 0.02], pitch=45)})])
    assert e.value.kind in ("check", "unreachable")


def test_unreachable(scene, home):
    with pytest.raises(PlanError) as e:
        Planner(scene).plan(home, [Phase("far", {"left": Goal(p=[0.4, 0.44, 0.05])})])
    assert e.value.kind == "unreachable"


def test_joint_move(scene, home):
    q = home["left"][:5] + [10, 0, 0, 0, 20]
    [traj] = Planner(scene).plan(home, [Phase("j", {"left": Goal(joints=q, line=False)})])
    assert np.allclose(traj.rows["left"][-1, :5], q)


def test_a_goal_without_yaw_keeps_the_wrist_roll(scene):
    start = {s: np.append(scene.ready(s), 40.0) for s in ("left", "right")}
    for goal in (Goal(p=[-0.12, 0.30, 0.15], pitch=0), Goal(p=[-0.12, 0.25, 0.06], pitch=60, line=False)):
        [traj] = Planner(scene).plan(start, [Phase("go", {"left": goal})])
        assert np.allclose(traj.rows["left"][:, 4], start["left"][4], atol=0.01)
