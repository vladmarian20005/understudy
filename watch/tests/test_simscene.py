"""A simulator scene from a video: Muse's account cleaned to what the simulator builds, laid out where the arms reach
and the camera sees, and loaded by the HD simulator."""
import json

import numpy as np
import pytest

from watch import simscene

BASES = {"left": np.array([-18.0, -1.0]), "right": np.array([18.0, -1.0])}
MUSE = [{"name": "Tennis ball can", "kind": "container", "size_cm": [7.8, 26], "color": "#1c1c1c", "count": 1,
         "hand": "right"},
        {"name": "tennis ball", "kind": "ball", "size_cm": [6.7], "color": "#d4e04a", "count": 4, "hand": "left"},
        {"name": "brick", "kind": "rock", "size_cm": [40], "color": "red", "count": 99, "hand": "?"}]


def test_muses_account_becomes_what_the_simulator_can_build():
    can, ball, brick = simscene.clean(MUSE)
    assert can["kind"] == "tube" and can["name"] == "tennis ball can" and can["size_cm"] == [7.8, 26.0]
    assert ball["kind"] == "ball" and ball["count"] == 4 and ball["hand"] == "left"
    # an unknown kind is a block, a size of the wrong shape the default, the count capped, the colour grey
    assert brick["kind"] == "box" and brick["size_cm"] == [6.0, 6.0, 6.0] and brick["count"] == simscene.MAX_EACH
    assert brick["color"] == "#b4b4b4" and brick["hand"] == "middle"


def test_the_layout_puts_things_on_their_hands_side_in_reach_and_apart():
    placed, notes = simscene.layout(simscene.clean(MUSE[:2]), BASES)
    assert notes == [] and [p["name"] for p in placed] == ["tennis ball can"] + [f"tennis ball {k}" for k in range(1, 5)]
    can, balls = placed[0], placed[1:]
    for side, b in BASES.items():  # the container is in the middle, in both arms' reach
        assert simscene.REACH_CM[0] <= np.linalg.norm(np.array(can["xy_cm"]) - b) <= simscene.REACH_CM[1]
    assert can["xy_cm"][1] <= simscene.TALL_Y_MAX  # tall: kept near enough for the camera to see its top
    for b in balls:  # the left hand's balls: on the left, in the left arm's reach, clear of everything
        assert b["xy_cm"][0] < 0
        assert simscene.REACH_CM[0] <= np.linalg.norm(np.array(b["xy_cm"]) - BASES["left"]) <= simscene.REACH_CM[1]
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            gap = np.linalg.norm(np.subtract(a["xy_cm"], b["xy_cm"])) - a["size_cm"][0] / 2 - b["size_cm"][0] / 2
            assert gap >= simscene.GAP_CM - 1e-6


def test_too_many_things_are_left_out_with_a_note():
    many = [dict(MUSE[1], name=f"ball {k}", count=6) for k in range(3)]
    placed, notes = simscene.layout(simscene.clean(many), BASES)
    assert len(placed) == simscene.MAX_ALL and any("at most" in n for n in notes)


def test_a_scene_file_is_what_the_hd_simulator_builds(tmp_path):
    mjsim = pytest.importorskip("harness.mjsim")
    placed, _ = simscene.layout(simscene.clean(MUSE[:2]), BASES)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(dict(objects=placed)))
    objs = mjsim.scene_objects(str(path))
    assert [o[0] for o in objs] == [p["name"] for p in placed]
    name, kind, size, bgr, x, y, yaw = objs[0]
    assert kind == "tube" and size == pytest.approx((0.078, 0.26)) and bgr == (0x1c, 0x1c, 0x1c)
    assert (x, y) == pytest.approx((placed[0]["xy_cm"][0] / 100, placed[0]["xy_cm"][1] / 100))
    assert objs[1][2] == pytest.approx(0.067)  # a ball is its diameter
    with pytest.raises(ValueError, match="no scene"):
        mjsim.scene_objects("nowhere")
