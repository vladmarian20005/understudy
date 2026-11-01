import threading
import time

import numpy as np
import pytest

from harness import sim
from harness.bus import BusArmIO
from harness.controller import Controller
from harness.motion import Goal, Phase, Planner
from harness.scene import Scene


@pytest.fixture(scope="module")
def scene():
    return Scene()


def rig(scene, objects="blocks", speed=0):
    world = sim.SimWorld(scene, objects, sag=False)  # the bare controller: sag has its own tests (test_sag.py)
    ctl = Controller(world.io, world, speed=speed).start()
    ctl.arm()
    return world, ctl, Planner(scene)


def run(ctl, planner, phases):
    snap = ctl.snapshot()
    ok, why = ctl.play(planner.plan(snap["goal"], phases))
    assert ok, why


def test_pick_carry_drop(scene):
    world, ctl, planner = rig(scene)
    block = world.objects[0]
    x, y, _ = block.pos
    run(ctl, planner, [Phase("above", {"left": Goal(p=[x, y, 0.08], pitch="steep", yaw=90, width=70, line=False)}),
                       Phase("down", {"left": Goal(p=[x, y, 0.015], pitch="steep", yaw=90)}),
                       Phase("close", {"left": Goal(width=0)}),
                       Phase("lift", {"left": Goal(p=[x, y, 0.10], pitch="steep")})])
    assert "left" in block.held
    assert block.pos[2] > 0.08
    assert ctl.snapshot()["q"]["left"][5] == pytest.approx(40, abs=1)  # stopped on the 4 cm block
    run(ctl, planner, [Phase("carry", {"left": Goal(p=[x + 0.05, y, 0.10], pitch="steep")}),
                       Phase("open", {"left": Goal(width=70)})])
    assert not block.held and block.pos[2] == pytest.approx(0.02)
    assert block.pos[0] == pytest.approx(x + 0.05, abs=0.01)
    ctl.close()


def test_closing_on_air_misses(scene):
    world, ctl, planner = rig(scene)
    x, y, _ = world.objects[0].pos
    run(ctl, planner, [Phase("beside", {"left": Goal(p=[x, y - 0.08, 0.02], pitch="steep", yaw=90, width=70,
                                                         line=False)}),
                       Phase("close", {"left": Goal(width=0)})])
    assert not world.objects[0].held and ctl.snapshot()["q"]["left"][5] < 1
    ctl.close()


def test_stop_holds_where_it_is(scene):
    world, ctl, planner = rig(scene, speed=1.0)
    p = scene.home_tip("left")
    trajs = planner.plan(ctl.snapshot()["goal"], [Phase("slow", {"left": Goal(p=p + [0.08, 0.05, 0], pitch="steep")},
                                                        speed="slow")])
    threading.Timer(0.4, ctl.stop, args=("test stop",)).start()
    ok, why = ctl.play(trajs)
    assert not ok and why == "test stop"
    snap = ctl.snapshot()
    assert snap["mode"]["left"] == "hold"
    time.sleep(0.1)
    assert np.allclose(ctl.snapshot()["goal"]["left"], snap["goal"]["left"])
    ctl.close()


def test_play_refuses_without_torque_or_from_elsewhere(scene):
    world = sim.SimWorld(scene)
    ctl = Controller(world.io, world, speed=0).start()
    planner = Planner(scene)
    trajs = planner.plan(ctl.snapshot()["goal"], [Phase("x", {"left": Goal(width=60)})])
    assert ctl.play(trajs) == (False, "torque off")
    ctl.arm()
    q = ctl.snapshot()["goal"]
    q["left"] = q["left"] + [20, 0, 0, 0, 0, 0]
    ok, why = ctl.play(planner.plan(q, [Phase("x", {"left": Goal(width=60)})]))
    assert not ok and "starts" in why
    ctl.close()


def test_blocked_arm_reflex(scene):
    world, ctl, planner = rig(scene)
    world.io["left"].step = lambda dt: None  # the arm can't move
    ok, why = ctl.play(planner.plan(ctl.snapshot()["goal"],
                                    [Phase("go", {"left": Goal(p=scene.home_tip("left") + [0.1, 0.05, 0],
                                                               pitch="steep")})]))
    assert not ok and why.startswith("blocked")
    ctl.close()


def test_render_shows_objects_where_the_camera_model_puts_them(scene):
    world = sim.SimWorld(scene, "blocks")
    img = sim.render(scene, world.snapshot(), scene.camera)
    assert img.shape == (480, 640, 3)
    for o in world.objects:
        px = scene.camera.project(o.pos + [0, 0, o.size[2] / 2])[0].astype(int)
        assert np.abs(img[px[1], px[0]].astype(int) - o.color).max() < 30
    wrist = sim.render(scene, world.snapshot(), sim.wrist_camera(scene, "left", world.io["left"].q), own="left")
    assert wrist.shape == (480, 640, 3)


class FakeBus:
    def __init__(self):
        self.calls = []
        self.pos = {"shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0, "wrist_flex": 4.0,
                    "wrist_roll": 5.0, "gripper": 50.0}

    def sync_read(self, name, normalize=True, num_retry=0):
        self.calls.append(("read", name))
        return dict(self.pos) if name == "Present_Position" else {k: 30 for k in self.pos}

    def sync_write(self, name, values, num_retry=0):
        self.calls.append(("sync_write", name, dict(values)))

    def write(self, name, motor, value, num_retry=0):
        self.calls.append(("write", name, motor))

    def enable_torque(self, num_retry=0):
        self.calls.append(("torque", True))

    def disable_torque(self, num_retry=0):
        self.calls.append(("torque", False))

    def configure_motors(self):
        self.calls.append(("configure",))


def test_real_arm_goes_live_where_it_is():
    """Torque on only after the goal has been set to the present position: no jump to an old goal."""
    bus = FakeBus()
    io = BusArmIO("port", "x", 84.0, bus=bus)
    assert io.read()[5] == pytest.approx(42.0)
    io.torque(True)
    names = [c[:2] for c in bus.calls]
    i_goal = names.index(("sync_write", "Goal_Position"))
    assert names[-1] == ("torque", True) and i_goal == len(names) - 2
    assert bus.calls[i_goal][2] == bus.pos
    assert ("torque", False) in names[:i_goal]
    io.write([10, 20, 30, 40, 50, 84])
    assert bus.calls[-1][2]["gripper"] == pytest.approx(100)


def test_real_gripper_opening_follows_the_swinging_jaw():
    """With the jaw's travel known (101 deg on the left arm), a 67 mm opening sits at 53 % of the servo's range, not 68 %:
    the opening grows fast near closed and slowly near open."""
    io = BusArmIO("port", "x", 98.0, bus=FakeBus(), opening=Scene().opening)
    io.travel = 101.4
    assert io.mm(0) == pytest.approx(0) and io.mm(100) == pytest.approx(97.9, abs=0.3)
    assert io.pct(67) == pytest.approx(52.6, abs=0.5)
    assert io.mm(io.pct(40)) == pytest.approx(40, abs=0.1)


def test_closing_jaws_square_up_a_skewed_brush(scene):
    """Flat pads turn a free bar square as they close and stop at its width, not at its length across the jaw line
    (what a policy's slightly-off grasp sees on the real gripper)."""
    world = sim.SimWorld(scene, "brush", sag=False)
    ctl = Controller(world.io, world, speed=0).start()
    ctl.arm()
    brush = world.objects[0]
    x, y, _ = brush.pos
    yaw0 = brush.yaw
    planner = Planner(scene)
    run(ctl, planner, [Phase("above", {"left": Goal(p=[x + 0.04, y, 0.08], pitch="steep", yaw=yaw0 + 90 + 12, width=60)}),
                       Phase("down", {"left": Goal(p=[x + 0.04, y, 0.012], pitch="steep")}),
                       Phase("close", {"left": Goal(width=0)})])
    assert "left" in brush.held
    assert ctl.snapshot()["q"]["left"][5] == pytest.approx(30, abs=3)  # the 3 cm width
    turned = np.degrees(np.arctan2(brush.R[1, 0], brush.R[0, 0])) - yaw0
    assert (turned + 90) % 180 - 90 == pytest.approx(12, abs=1.5)
    ctl.close()


def test_a_long_held_object_stays_held_while_the_wrist_turns(scene):
    """The fingers' grip on a long object is checked against where the object is now, not a tick ago."""
    world = sim.SimWorld(scene, "brush", sag=False)
    ctl = Controller(world.io, world, speed=0).start()
    ctl.arm()
    brush = world.objects[0]
    x, y, _ = brush.pos
    planner = Planner(scene)
    run(ctl, planner, [Phase("above", {"left": Goal(p=[x + 0.06, y, 0.08], pitch="steep", yaw=brush.yaw + 90, width=60)}),
                       Phase("down", {"left": Goal(p=[x + 0.06, y, 0.012], pitch="steep")}),
                       Phase("close", {"left": Goal(width=0)})])
    assert "left" in brush.held
    q = ctl.snapshot()["goal"]["left"].copy()
    q[0] += 25  # swing the base and the wrist fast, with the brush held at its end
    q[4] += 40
    run(ctl, planner, [Phase("swing", {"left": Goal(joints=q[:5], line=False)}, speed="fast")])
    assert "left" in brush.held
    ctl.close()
