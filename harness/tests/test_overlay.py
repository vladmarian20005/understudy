"""The overhead image an agent sees: the table grid, the objects and both grippers drawn where they are."""
import numpy as np

from harness import overlay
from harness.scene import Scene

ARMS = {"left": dict(tip_cm=[-10.0, 25.0, 8.0], yaw=0.0, width_mm=40.0, holding=False),
        "right": dict(tip_cm=[12.0, 22.0, 3.0], yaw=90.0, width_mm=30.0, holding=True)}
OBJS = [dict(id="#1", name="brush", px=[400, 220])]


def test_overlay_draws_grid_objects_and_grippers():
    scene = Scene()
    cam = scene.camera
    img = np.zeros((cam.height, cam.width, 3), np.uint8)
    out = overlay.draw(img, scene, OBJS, ARMS)
    assert out.shape == img.shape and out.any()
    assert not img.any()  # drawn on a copy
    # the grid's x = 0 line passes through the projection of (0, 20 cm) on the table
    u, v = cam.project([[0.0, 0.20, 0.0]])[0].round().astype(int)
    assert out[v - 1:v + 2, u - 1:u + 2].max() > 60
    # each gripper's colour is at its fingertips
    for side, arm in ARMS.items():
        u, v = cam.project([np.array(arm["tip_cm"]) / 100])[0].round().astype(int)
        patch = out[v - 2:v + 3, u - 2:u + 3].reshape(-1, 3)
        assert any(np.abs(px.astype(int) - overlay.COLORS[side]).max() < 40 for px in patch), side
