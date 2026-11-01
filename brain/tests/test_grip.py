"""The brain's own numbers: how deep a grasp goes, names that stay put, the wrist overlay, the reach facts, the prompt. No
robotd."""
import numpy as np

from brain import act, reach, tools, wrist
from brain.prompt import plan_system
from brain.rig import scene


def test_a_grasp_goes_deep_but_not_into_the_table_or_the_end_of_the_jaws_travel():
    assert act.grip_depth(3.0, 15.0, 0) == act.DEEP_CM  # a narrow thing from the side: as deep as it goes
    tube = act.grip_depth(7.8, 13.0, 0)
    assert 2.6 < tube < act.DEEP_CM  # a 78 mm tube: deep, not at the fingertips, but less: the jaws keep travel to
    # squeeze with
    ball = act.grip_depth(6.7, 3.35, 80)
    assert ball < act.DEEP_CM and 3.35 - ball * np.sin(np.radians(80)) >= act.TIP_ABOVE_CM - 1e-6  # fingertips up
    brush = act.grip_depth(3.0, 1.25, 80)
    assert 0.3 <= brush and 1.25 - brush * np.sin(np.radians(80)) > 0.3  # a low thing: still off the table


def test_one_object_keeps_one_name_whatever_word_found_it():
    tools.reset()
    tools.remember([dict(name="ball 1", center_cm=[-10, 22], size_cm=[6.7] * 3),
                    dict(name="tube", center_cm=[12, 36], size_cm=[7.8, 7.8, 26])])
    first = {o["name"] for o in tools._memory["objects"]}
    tools.remember([dict(name="tennis ball", found_as="tennis ball", center_cm=[-10.6, 22.4], size_cm=[6.7] * 3),
                    dict(name="cylinder", found_as="cylinder", center_cm=[12.4, 36.9], size_cm=[7.8, 7.8, 18]),
                    dict(name="ball", found_as="ball", center_cm=[5, 30], size_cm=[6.7] * 3)])
    names = [o["name"] for o in tools._memory["objects"]]
    assert len(names) == 3 and len(set(names)) == 3 and "tube" in names  # three things, three names
    near = min(tools._memory["objects"], key=lambda o: abs(o["center_cm"][0] + 10.6))
    assert near["name"] in first  # the ball found as "tennis ball" kept the name it had
    tube = next(o for o in tools._memory["objects"] if o["name"] == "tube")
    assert tube["size_cm"][2] == 26  # it doesn't get shorter


def test_the_grip_point_is_drawn_between_the_fingers_below_the_fingertips_line(monkeypatch):
    monkeypatch.setattr(wrist, "real", lambda: False)  # the mount's CAD, as the simulators render it
    side = "left"
    tcp = scene().arms[side].tcp
    tip, grip = wrist.wrist_px(side, tcp), wrist.wrist_px(side, tcp + np.array([0.0, 0.0, 0.03]))
    fixed = wrist.wrist_px(side, scene().fixed_tip)
    w, h = wrist.WRIST_PX
    assert all(0 <= p[0] < w and 0 <= p[1] < h for p in (tip, grip, fixed))
    assert grip[1] > tip[1]  # deeper in is lower in the image (the jaws come in at the bottom)
    assert fixed[0] > tip[0]  # the fixed finger is on the image's right


def test_level_reach_is_a_range_and_the_middle_reaches_high():
    lo, hi = reach.level_range("left", (0, 25))
    assert lo < 10 and hi > 30
    lo, hi = reach.level_range("right", (0, 25))
    assert lo < 10 and hi > 30
    r = reach.level_range("left", (12, 36))  # far across: the left arm can't hold anything level there
    assert r is None or r[1] < 20


def test_what_is_seen_inside_a_standing_tube_is_part_of_it_not_a_new_thing():
    tools.reset()
    tools.remember([dict(name="cylinder", found_as="cylinder", center_cm=[6.0, 19.4], size_cm=[9.2, 9.2, 25.7],
                         shape="upright"),
                    dict(name="ball", found_as="ball", center_cm=[-4.5, 32.3], size_cm=[6.7] * 3, shape="round")])
    tools.remember([dict(name="cup", found_as="cup", center_cm=[3.6, 24.8], size_cm=[5.1, 5.1, 18.6], shape="upright"),
                    dict(name="tennis ball", found_as="tennis ball", center_cm=[6.5, 24.4], size_cm=[6.6, 3.5, 2.5],
                         shape="flat"),
                    dict(name="ball", found_as="ball", center_cm=[13.3, 19.4], size_cm=[6.7] * 3, shape="round")])
    names = sorted(o["name"] for o in tools._memory["objects"])
    assert len(names) == 3 and "cylinder" in names  # the cup and the flat ball were the tube; the ball beside it isn't


def test_the_sliver_of_a_half_covered_ball_is_that_ball():
    tools.reset()
    tools.remember([dict(name="ball", found_as="ball", center_cm=[-7.5, 32.1], size_cm=[7.2, 6.6, 7.1], shape="round")])
    tools.remember([dict(name="tennis ball", found_as="tennis ball", center_cm=[-10.3, 32.0], size_cm=[1.8, 1.8, 7.2],
                         shape="upright")])
    assert [o["name"] for o in tools._memory["objects"]] == ["ball"]


def test_a_corrected_size_and_spot_hold_over_later_looks():
    tools.reset()
    cube = lambda c, s: dict(name="blue cube", center_cm=c, size_cm=s, z_cm=s[2] / 2, shape="upright", yaw=0.0)
    tools.remember([cube([1.6, 24.5], [2.8, 2.8, 5.5]), cube([18.9, 36.4], [0.9, 0.9, 8.3])])
    names = sorted(o["name"] for o in tools._memory["objects"])
    tools.correct({names[1]: {"size_cm": [2.7, 2.7, 2.7], "center_cm": [21.0, 39.6]}})
    tools.remember([cube([1.6, 24.5], [2.8, 2.8, 5.5]), cube([19.5, 37.0], [0.9, 0.9, 8.3])])  # the sliver again
    o = next(o for o in tools._memory["objects"] if o["name"] == names[1])
    assert o["size_cm"] == [2.7, 2.7, 2.7] and o["z_cm"] == 1.35 and o["shape"].startswith("flat")
    assert o["center_cm"] == [21.0, 39.6]
    assert "no such object" in tools.correct({"nothing": {"size_cm": [1, 1, 1]}})[0]


def test_the_prompt_is_about_the_rig_and_never_a_task():
    text = plan_system(scene()).lower()
    assert "{lx" not in text and f"{scene().stroke_mm:.0f} mm" in text
    for word in ("tube", "tennis", "cube", "towel", "brush", "mug", "container"):
        assert word not in text, word


def test_an_object_an_arm_covers_keeps_where_it_was_seen_whole():
    tools.reset()
    ball = dict(name="yellow object", center_cm=[-17.0, 39.0], size_cm=[7.0, 7.0, 7.0], z_cm=3.5)
    tools.remember([ball])
    tools.remember([dict(ball, center_cm=[-15.0, 37.5], size_cm=[4.0, 3.0, 7.0])])  # half under the arm
    tools.remember([])  # all of it under the arm
    (o,) = tools._memory["objects"]
    assert o["center_cm"] == [-17.0, 39.0] and o["seen"] == 1
    tools.remember([dict(ball, center_cm=[-5.0, 30.0])])  # seen whole somewhere else: it rolled
    assert [o["center_cm"] for o in tools._memory["objects"]] == [[-5.0, 30.0], [-17.0, 39.0]]
    for _ in range(tools.FORGET_LOOKS):
        tools.remember([dict(ball, center_cm=[-5.0, 30.0])])
    assert [o["center_cm"] for o in tools._memory["objects"]] == [[-5.0, 30.0]]


def test_an_object_seen_round_once_keeps_its_height():
    tools.reset()
    ball = dict(name="yellow object", center_cm=[-11.0, 38.0], size_cm=[7.0, 7.0, 7.0], z_cm=3.5)
    tools.remember([ball])
    tools.remember([dict(ball, size_cm=[7.2, 6.6, 2.5], z_cm=1.2)])  # this frame's blob wasn't round enough
    (o,) = tools._memory["objects"]
    assert o["size_cm"] == [7.2, 6.6, 7.0] and o["z_cm"] == 3.5 and o["seen"] == 2
