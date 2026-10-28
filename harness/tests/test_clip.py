"""Recording a clip from a camera stream, here the simulator's."""
import json

import cv2

from harness import cameras, clip
from harness.robotd import Core


def test_a_clip_of_the_overhead_camera(tmp_path):
    core = Core(sim_mode=True, speed=0, quiet=True, dataset=None)
    pub = cameras.Publisher(lambda names: core.cameras.grab(names), fps=30, base_port=5910)
    try:
        info = clip.record(str(tmp_path / "pass.mp4"), 1.0, base_port=5910)
    finally:
        pub.close()
        core.close()
    assert info["frames"] >= 10 and info["camera_model"]["width"] == 640
    video = cv2.VideoCapture(str(tmp_path / "pass.mp4"))
    ok, frame = video.read()
    assert ok and frame.shape == (480, 640, 3)
    assert json.load(open(tmp_path / "pass.json"))["frames"] == info["frames"]
