import base64
import json
import threading
import time

import cv2
import numpy as np
import pytest
import zmq

from harness import cameras
from harness.cameras import CameraDown, ZmqCameras


def publish(port, name, stop):
    s = zmq.Context.instance().socket(zmq.PUB)
    s.setsockopt(zmq.LINGER, 0)
    s.bind(f"tcp://127.0.0.1:{port}")
    img = np.full((48, 64, 3), 120, np.uint8)
    b64 = base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()
    while not stop.is_set():
        s.send_string(json.dumps({"timestamps": {name: time.time()}, "images": {name: b64}}))
        time.sleep(0.02)
    s.close()


def test_latest_frames_and_stale_ones(monkeypatch):
    stop = threading.Event()
    threading.Thread(target=publish, args=(5895, "overhead", stop), daemon=True).start()
    cams = ZmqCameras(names=("overhead",), base_port=5895)
    try:
        cams.wait(timeout=5)
        assert cams.grab(["overhead"])["overhead"].shape == (48, 64, 3)
        stop.set()
        monkeypatch.setattr(cameras, "MAX_AGE", 0.2)
        time.sleep(0.4)
        with pytest.raises(CameraDown):
            cams.grab(["overhead"])
    finally:
        stop.set()
        cams.close()


def test_missing_camera_is_reported():
    cams = ZmqCameras(names=("right_wrist",), base_port=5896)
    try:
        with pytest.raises(CameraDown, match="right_wrist"):
            cams.wait(timeout=0.5)
    finally:
        cams.close()
